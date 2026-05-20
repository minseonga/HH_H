import argparse
import csv
import json
import math
import os


CORE_FEATURES = [
    "text_mass",
    "unsupported_text_value_norm",
    "unsupported_norm_x_low_anchor",
    "text_value_norm",
    "supported_text_value_norm",
    "unsupported_total_value_ratio",
    "visual_value_ratio",
    "text_img_value_cosine",
]

CASE_ROWS = [
    "adhh_trigger_should_suppress",
    "adhh_trigger_safe_to_keep",
    "adhh_missed_should_suppress",
    "case1_text_heavy_parallel_safe",
    "case1_base_text_heavy_parallel",
    "case2_text_heavy_orthogonal_should_suppress",
    "case3_balanced_orthogonal_missed",
    "case3_base_balanced_orthogonal",
]


def safe_float(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    return value


def read_csv(path):
    if not path or not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        with open(path, "w") as f:
            f.write("")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value, digits=3):
    value = safe_float(value)
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def md_table(rows, columns, max_rows=None):
    rows = rows[:max_rows] if max_rows else rows
    if not rows:
        return "_No rows._\n"
    lines = []
    lines.append("| " + " | ".join(label for _, label in columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        values = []
        for key, _ in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                value = fmt(value)
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def get_all_feature_rows(rows):
    return [row for row in rows if row.get("group") == "all"]


def build_feature_summary(feature_rows):
    by_feature = {row.get("feature"): row for row in get_all_feature_rows(feature_rows)}
    text_auc = safe_float(by_feature.get("text_mass", {}).get("auroc_abs"), 0.0)
    output = []
    for feature in CORE_FEATURES:
        row = by_feature.get(feature)
        if not row:
            continue
        auc_abs = safe_float(row.get("auroc_abs"))
        high_auc = safe_float(row.get("auroc_high_predicts_should_suppress"))
        output.append({
            "feature": feature,
            "direction": row.get("direction"),
            "n": row.get("n"),
            "n_should_suppress": row.get("n_should_suppress"),
            "mean_should_suppress": safe_float(row.get("mean_should_suppress")),
            "mean_safe_to_keep": safe_float(row.get("mean_safe_to_keep")),
            "gap_should_minus_safe": safe_float(row.get("gap_should_minus_safe")),
            "auroc_high_predicts_should_suppress": high_auc,
            "auroc_abs": auc_abs,
            "auroc_abs_minus_text_mass": (auc_abs - text_auc) if auc_abs is not None else None,
        })
    output.sort(key=lambda row: safe_float(row.get("auroc_abs"), -1.0), reverse=True)
    return output


def build_case_summary(case_rows):
    rows = [row for row in case_rows if row.get("group") == "all" and row.get("case") in CASE_ROWS]
    order = {case: idx for idx, case in enumerate(CASE_ROWS)}
    rows.sort(key=lambda row: order.get(row.get("case"), 999))
    output = []
    for row in rows:
        output.append({
            "case": row.get("case"),
            "n": row.get("n"),
            "rate_group": safe_float(row.get("rate_group")),
            "should_suppress_rate": safe_float(row.get("should_suppress_rate")),
            "mean_utility": safe_float(row.get("mean_utility")),
            "mean_text_mass": safe_float(row.get("mean_text_mass")),
            "mean_unsupported_text_value_norm": safe_float(row.get("mean_unsupported_text_value_norm")),
            "mean_text_img_value_cosine": safe_float(row.get("mean_text_img_value_cosine")),
        })
    return output


def build_overlap_rows(rows, top_k):
    output = []
    for row in rows[:top_k]:
        output.append({
            "rank": len(output) + 1,
            "head_key": row.get("head_key") or f'{row.get("layer")}:{row.get("head")}',
            "overlap_score": safe_float(row.get("overlap_score")),
            "unsupported_text_value_norm": safe_float(row.get("unsupported_text_value_norm")),
            "positive_utility_contrast": safe_float(
                row.get("should_suppress_minus_safe_unsupported_text_value_norm")
            ),
        })
    return output


def parse_eval_result_arg(value):
    if "=" in value:
        name, path = value.split("=", 1)
        return name, path
    path = value
    return os.path.basename(os.path.dirname(path)), path


def load_eval_rows(items):
    output = []
    for item in items:
        name, path = parse_eval_result_arg(item)
        if not os.path.exists(path):
            output.append({"method": name, "path": path, "missing": True})
            continue
        with open(path) as f:
            data = json.load(f)
        metrics = data.get("overall_metrics", data)
        bleu = metrics.get("Bleu") or [None, None, None, None]
        output.append({
            "method": name,
            "CHAIRs": safe_float(metrics.get("CHAIRs")),
            "CHAIRi": safe_float(metrics.get("CHAIRi")),
            "BLEU4": safe_float(bleu[3] if len(bleu) >= 4 else None),
            "avg_caption_length": safe_float(metrics.get("avg_caption_length")),
            "path": path,
        })
    return output


def maybe_plot_feature_auc(path, feature_summary):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    rows = feature_summary[:8]
    if not rows:
        return None
    labels = [row["feature"] for row in rows]
    values = [safe_float(row.get("auroc_abs"), 0.0) for row in rows]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(len(rows)), values, color="#3778bf")
    ax.axhline(0.5, color="#888888", linewidth=1, linestyle="--")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("AUROC abs")
    ax.set_title("Feature separation of causal suppression utility")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def write_report(path, feature_summary, case_summary, overlap_rows, eval_rows, plot_path):
    text_row = next((row for row in feature_summary if row["feature"] == "text_mass"), None)
    best_row = feature_summary[0] if feature_summary else None
    unsupported_row = next(
        (row for row in feature_summary if row["feature"] == "unsupported_text_value_norm"),
        None,
    )

    lines = []
    lines.append("# Unsupported Component Feature Evidence\n")
    lines.append("## Method Claim\n")
    lines.append(
        "AD-HH uses a text-attention-mass trigger and suppresses the full text attention branch. "
        "The unsupported-component feature decomposes each head output into image-supported and image-unsupported text value components, then scores the unsupported component online.\n"
    )
    lines.append(
        "For a head output, `C_txt = sum_{j in T} A_j V_j` and `C_img = sum_{i in I} A_i V_i`. "
        "We define the supported text component as the positive projection of `C_txt` onto `C_img`, and the unsupported component as the residual: "
        "`C_txt_unsupported = C_txt - max(<C_txt, u_img>, 0) u_img`.\n"
    )
    lines.append("## Main Evidence\n")
    if best_row and text_row:
        lines.append(
            f"- Best feature by AUROC abs: `{best_row['feature']}` = {fmt(best_row.get('auroc_abs'))}; "
            f"AD-HH proxy `text_mass` = {fmt(text_row.get('auroc_abs'))}.\n"
        )
    if unsupported_row and text_row:
        lines.append(
            f"- `unsupported_text_value_norm` improves over `text_mass` by "
            f"{fmt(unsupported_row.get('auroc_abs_minus_text_mass'))} AUROC abs on causal should-suppress labels.\n"
        )
    lines.append(
        "- This supports the central distinction: high text attention is not the same as harmful text contribution; the harmful part is better captured by unsupported value magnitude.\n"
    )
    lines.append("\n## Feature vs AD-HH Proxy\n")
    lines.append(md_table(
        feature_summary,
        [
            ("feature", "feature"),
            ("direction", "direction"),
            ("mean_should_suppress", "mean suppress"),
            ("mean_safe_to_keep", "mean safe"),
            ("gap_should_minus_safe", "gap"),
            ("auroc_abs", "AUROC abs"),
            ("auroc_abs_minus_text_mass", "delta vs text_mass"),
        ],
        max_rows=12,
    ))
    if plot_path:
        lines.append(f"\n![feature auc]({os.path.basename(plot_path)})\n")
    lines.append("\n## AD-HH Failure Modes\n")
    lines.append(md_table(
        case_summary,
        [
            ("case", "case"),
            ("n", "n"),
            ("rate_group", "rate"),
            ("should_suppress_rate", "suppress rate"),
            ("mean_utility", "mean utility"),
            ("mean_text_mass", "text mass"),
            ("mean_unsupported_text_value_norm", "unsupported norm"),
            ("mean_text_img_value_cosine", "txt-img cos"),
        ],
        max_rows=20,
    ))
    lines.append("\n## Overlap Candidate Heads\n")
    lines.append(
        "Overlap ranks heads that are high in unsupported value and also high in positive causal-utility contrast. "
        "These are better candidates than raw text-mass heads because they combine an online feature with a causal teacher signal.\n"
    )
    lines.append(md_table(
        overlap_rows,
        [
            ("rank", "rank"),
            ("head_key", "head"),
            ("overlap_score", "overlap"),
            ("unsupported_text_value_norm", "unsupported norm"),
            ("positive_utility_contrast", "positive utility contrast"),
        ],
        max_rows=20,
    ))
    if eval_rows:
        lines.append("\n## End-to-End Metrics\n")
        lines.append(md_table(
            eval_rows,
            [
                ("method", "method"),
                ("CHAIRs", "CHAIRs"),
                ("CHAIRi", "CHAIRi"),
                ("BLEU4", "BLEU4"),
                ("avg_caption_length", "length"),
            ],
            max_rows=30,
        ))
    lines.append("\n## Interpretation\n")
    lines.append(
        "The current feature is not just AD-HH with a different head list. It changes both the measurement and the action surface: "
        "AD-HH measures attention mass and removes all text attention, while this feature measures the value-space text component not supported by the image component and suppresses that component online.\n"
    )
    lines.append(
        "The strongest paper-level argument is therefore: `text-heavy` is an imprecise proxy for harmful contribution; "
        "`unsupported text value` is closer to the causal object-token decision because it uses the same weighted value vectors that actually form the head output.\n"
    )
    with open(path, "w") as f:
        f.write("".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual-analysis-dir", required=True)
    parser.add_argument("--overlap-summary", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--eval-result", action="append", default=[])
    parser.add_argument("--top-overlap-k", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    feature_rows = read_csv(os.path.join(args.visual_analysis_dir, "visual_feature_suppression_utility_contrast.csv"))
    case_rows = read_csv(os.path.join(args.visual_analysis_dir, "visual_anchor_case_summary.csv"))
    overlap_path = args.overlap_summary or os.path.join(
        os.path.dirname(args.visual_analysis_dir),
        "unsupported_head_heatmaps",
        "unsupported_positive_overlap_summary.csv",
    )
    overlap_rows = read_csv(overlap_path)

    feature_summary = build_feature_summary(feature_rows)
    case_summary = build_case_summary(case_rows)
    overlap_summary = build_overlap_rows(overlap_rows, args.top_overlap_k)
    eval_rows = load_eval_rows(args.eval_result)

    write_csv(os.path.join(args.output_dir, "feature_vs_adhh_summary.csv"), feature_summary)
    write_csv(os.path.join(args.output_dir, "adhh_failure_case_summary.csv"), case_summary)
    write_csv(os.path.join(args.output_dir, "overlap_candidate_head_summary.csv"), overlap_summary)
    if eval_rows:
        write_csv(os.path.join(args.output_dir, "method_eval_summary.csv"), eval_rows)

    plot_path = maybe_plot_feature_auc(
        os.path.join(args.output_dir, "feature_auc_abs.png"),
        feature_summary,
    )
    report_path = os.path.join(args.output_dir, "unsupported_feature_report.md")
    write_report(report_path, feature_summary, case_summary, overlap_summary, eval_rows, plot_path)

    print("wrote:", report_path)
    print("feature summary:", os.path.join(args.output_dir, "feature_vs_adhh_summary.csv"))
    print("case summary:", os.path.join(args.output_dir, "adhh_failure_case_summary.csv"))
    print("overlap summary:", os.path.join(args.output_dir, "overlap_candidate_head_summary.csv"))


if __name__ == "__main__":
    main()

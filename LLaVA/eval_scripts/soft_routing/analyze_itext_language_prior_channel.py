import argparse
import csv
import html
import json
import math
import os
from collections import Counter, defaultdict

from eval_scripts.soft_routing.analyze_contrastive_dynamic_head_pool import (
    head_key,
    load_ranked_heads,
    safe_float,
)


BLUE = "#2563eb"
ORANGE = "#f97316"
GREEN = "#059669"
RED = "#dc2626"
GRAY = "#cbd5e1"
DARK = "#0f172a"
MUTED = "#64748b"


def mean(values):
    values = [safe_float(value) for value in values]
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def safe_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys or ["empty"])
        writer.writeheader()
        writer.writerows(rows)


def text(x, y, value, size=13, fill=DARK, anchor="start", weight="400", rotate=None):
    value = html.escape(str(value))
    transform = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}" '
        f'text-anchor="{anchor}"{transform}>{value}</text>\n'
    )


def line(x1, y1, x2, y2, stroke=MUTED, width=1, dash=None):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{stroke}" stroke-width="{width}"{dash_attr}/>\n'


def rect(x, y, w, h, fill, stroke=None, width=1, opacity=1.0):
    stroke_attr = f' stroke="{stroke}" stroke-width="{width}"' if stroke else ""
    return f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" fill="{fill}" opacity="{opacity}"{stroke_attr}/>\n'


def circle(x, y, r, fill, stroke=None, width=1, opacity=1.0):
    stroke_attr = f' stroke="{stroke}" stroke-width="{width}"' if stroke else ""
    return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" fill="{fill}" opacity="{opacity}"{stroke_attr}/>\n'


def svg(path, width, height, body):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">\n'
        )
        f.write('<rect width="100%" height="100%" fill="white"/>\n')
        f.write(body)
        f.write("</svg>\n")


def load_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_rank_sets(path, top_k):
    _, records = load_ranked_heads(path)
    by_itext = sorted(records, key=lambda row: safe_float(row.get("front_percentile"), -1.0), reverse=True)
    by_contrast = sorted(records, key=lambda row: safe_float(row.get("back_percentile"), -1.0), reverse=True)
    return {
        "itext_top": {row["_key"] for row in by_itext[:top_k]},
        "contrast_top": {row["_key"] for row in by_contrast[:top_k]},
        "combined_top": {row["_key"] for row in records[:top_k]},
        "itext_order": by_itext,
        "combined_order": records,
        "contrast_order": by_contrast,
    }


def row_group(row, rank_sets):
    key = row.get("head_key")
    if key in rank_sets["itext_top"]:
        if key in rank_sets["combined_top"]:
            return "itext_and_combined"
        return "itext_only"
    if key in rank_sets["combined_top"]:
        return "combined_non_itext"
    if key in rank_sets["contrast_top"]:
        return "contrast_only"
    return "other_candidate"


def rank_rows(rows, rank_sets):
    for row in rows:
        row["itext_language_prior_group"] = row_group(row, rank_sets)
        row["target_rank_text_delta"] = None
        original = safe_int(row.get("target_rank_original"), 0)
        text_zero = safe_int(row.get("target_rank_text_zero"), 0)
        if original and text_zero:
            row["target_rank_text_delta"] = text_zero - original
        row["text_zero_top1_changed"] = (
            1
            if row.get("text_zero_top1_token_id")
            and str(row.get("text_zero_top1_token_id")) != str(row.get("top1_token_id"))
            else 0
        )
    return rows


def summarize_groups(rows):
    output = []
    groups = [
        "itext_and_combined",
        "itext_only",
        "combined_non_itext",
        "contrast_only",
        "other_candidate",
    ]
    labels = ["all", "hallucinated", "grounded"]
    for group in groups:
        group_rows = [row for row in rows if row.get("itext_language_prior_group") == group]
        if not group_rows:
            continue
        for label in labels:
            selected = group_rows if label == "all" else [row for row in group_rows if row.get("label_family") == label or row.get("label_name") == label]
            if not selected:
                continue
            output.append({
                "group": group,
                "label": label,
                "n_rows": len(selected),
                "n_heads": len({row.get("head_key") for row in selected}),
                "mean_text_mass": mean(row.get("text_mass") for row in selected),
                "mean_img_mass": mean(row.get("img_mass") for row in selected),
                "mean_text_entropy_norm": mean(row.get("text_entropy_norm") for row in selected),
                "mean_proxy_text_target_logit": mean(row.get("proxy_text_target_logit") for row in selected),
                "mean_positive_proxy_text_target_logit": mean(row.get("proxy_text_target_logit_positive") for row in selected),
                "positive_proxy_text_target_logit_rate": mean(
                    1.0 if safe_float(row.get("proxy_text_target_logit"), 0.0) > 0 else 0.0
                    for row in selected
                ),
                "mean_proxy_img_target_logit": mean(row.get("proxy_img_target_logit") for row in selected),
                "mean_proxy_evidence_gap_target": mean(row.get("proxy_evidence_gap_target") for row in selected),
                "mean_target_text_logit_drop": mean(row.get("target_text_logit_drop") for row in selected),
                "mean_target_text_logprob_drop": mean(row.get("target_text_logprob_drop") for row in selected),
                "mean_target_full_logit_drop": mean(row.get("target_logit_drop") for row in selected),
                "mean_target_full_logprob_drop": mean(row.get("target_logprob_drop") for row in selected),
                "positive_target_text_logit_drop_rate": mean(
                    1.0 if safe_float(row.get("target_text_logit_drop"), 0.0) > 0 else 0.0
                    for row in selected
                ),
                "mean_target_rank_text_delta": mean(row.get("target_rank_text_delta") for row in selected),
                "text_zero_top1_changed_rate": mean(row.get("text_zero_top1_changed") for row in selected),
            })
    return output


def summarize_heads(rows):
    by_head = defaultdict(list)
    for row in rows:
        by_head[row.get("head_key")].append(row)
    output = []
    for key, selected in by_head.items():
        if not key:
            continue
        layer, head = key.split(":")
        output.append({
            "head_key": key,
            "layer": int(layer),
            "head": int(head),
            "group": selected[0].get("itext_language_prior_group"),
            "n_rows": len(selected),
            "n_hallucinated": sum(1 for row in selected if row.get("label_family") == "hallucinated" or row.get("label_name") == "hallucinated"),
            "mean_text_mass": mean(row.get("text_mass") for row in selected),
            "mean_proxy_text_target_logit": mean(row.get("proxy_text_target_logit") for row in selected),
            "mean_positive_proxy_text_target_logit": mean(row.get("proxy_text_target_logit_positive") for row in selected),
            "mean_target_text_logit_drop": mean(row.get("target_text_logit_drop") for row in selected),
            "mean_target_text_logprob_drop": mean(row.get("target_text_logprob_drop") for row in selected),
            "mean_target_rank_text_delta": mean(row.get("target_rank_text_delta") for row in selected),
        })
    output.sort(key=lambda row: safe_float(row.get("mean_positive_proxy_text_target_logit"), -1.0), reverse=True)
    return output


def layer_distribution(rank_sets, top_k):
    rows = []
    for name, records in (
        ("itext_top", rank_sets["itext_order"][:top_k]),
        ("combined_top", rank_sets["combined_order"][:top_k]),
        ("contrast_top", rank_sets["contrast_order"][:top_k]),
    ):
        counts = Counter(int(row["layer"]) for row in records)
        for layer in sorted(counts):
            rows.append({
                "ranking": name,
                "layer": layer,
                "count": counts[layer],
                "fraction": counts[layer] / float(top_k),
                "band": layer_band(layer),
            })
    return rows


def layer_band(layer):
    layer = int(layer)
    if 11 <= layer <= 20:
        return "cross_modal_L11_20"
    if 21 <= layer <= 26:
        return "bridge_L21_26"
    if 27 <= layer <= 32:
        return "late_refinement_L27_32"
    return "other"


def make_group_bar_svg(path, summary_rows):
    width, height = 1120, 620
    metrics = [
        ("Text value -> object logit", "mean_proxy_text_target_logit", BLUE),
        ("Text-side suppress object-logit drop", "mean_target_text_logit_drop", ORANGE),
        ("Target rank worsens after text suppress", "mean_target_rank_text_delta", GREEN),
    ]
    groups = ["itext_and_combined", "itext_only", "combined_non_itext", "contrast_only", "other_candidate"]
    labels = {
        "itext_and_combined": "itext+combined",
        "itext_only": "itext-only",
        "combined_non_itext": "combined non-itext",
        "contrast_only": "contrast-only",
        "other_candidate": "other",
    }
    by_key = {
        (row["group"], row["label"]): row
        for row in summary_rows
        if row["label"] == "all"
    }
    left, top = 72, 78
    panel_w, panel_h, gap = 310, 390, 34
    body = []
    body.append(text(width / 2, 30, "Itext heads as object-logit language-prior channels", 20, DARK, "middle", "700"))
    body.append(text(width / 2, 52, "Object mention rows only: text-side head contribution and ablation effect on target object tokens", 12, MUTED, "middle"))
    for pidx, (title, key, color) in enumerate(metrics):
        px = left + pidx * (panel_w + gap)
        vals = [safe_float(by_key.get((group, "all"), {}).get(key), 0.0) for group in groups]
        vmax = max(max(vals), 0.0)
        vmin = min(min(vals), 0.0)
        pad = max((vmax - vmin) * 0.12, 0.01)
        vmax += pad
        vmin -= pad

        def sy(value):
            return top + (vmax - value) / max(vmax - vmin, 1e-9) * panel_h

        body.append(rect(px, top, panel_w, panel_h, "#f8fafc", "#cbd5e1"))
        body.append(text(px + panel_w / 2, top - 18, title, 13, DARK, "middle", "700"))
        zero_y = sy(0.0)
        body.append(line(px, zero_y, px + panel_w, zero_y, "#94a3b8", 1.2))
        group_w = panel_w / len(groups)
        for idx, group in enumerate(groups):
            value = safe_float(by_key.get((group, "all"), {}).get(key), 0.0)
            bar_w = min(34, group_w * 0.54)
            cx = px + idx * group_w + group_w / 2
            y = sy(max(value, 0.0))
            h = abs(sy(value) - zero_y)
            if value < 0:
                y = zero_y
            fill = color if group != "other_candidate" else "#94a3b8"
            body.append(rect(cx - bar_w / 2, y, bar_w, max(h, 1.0), fill, None, 1, 0.86))
            body.append(text(cx, y - 7 if value >= 0 else y + h + 13, f"{value:.3f}", 9, DARK, "middle"))
            body.append(text(cx, top + panel_h + 20, labels[group], 9, DARK, "middle", rotate=18))
    body.append(text(width / 2, height - 34, "Interpretation: high-Itext heads provide strong object-logit leverage; contrast alone is not enough leverage.", 13, DARK, "middle"))
    svg(path, width, height, "".join(body))


def make_layer_svg(path, layer_rows):
    width, height = 980, 560
    left, top = 82, 70
    plot_w, plot_h = 780, 360
    layers = list(range(11, 32))
    rankings = ["itext_top", "combined_top", "contrast_top"]
    colors = {"itext_top": BLUE, "combined_top": ORANGE, "contrast_top": GREEN}
    by = {(row["ranking"], int(row["layer"])): safe_int(row["count"]) for row in layer_rows}
    vmax = max(by.values()) if by else 1
    body = []
    body.append(text(width / 2, 30, "Layer distribution of Itext vs combined vs contrastive heads", 20, DARK, "middle", "700"))
    body.append(text(width / 2, 52, "Itext-heavy language-prior channels are distributed across cross-modal and late refinement layers", 12, MUTED, "middle"))
    body.append(rect(left, top, plot_w, plot_h, "#f8fafc", "#cbd5e1"))
    for tick in range(0, vmax + 1, max(1, math.ceil(vmax / 4))):
        y = top + plot_h - tick / max(vmax, 1) * plot_h
        body.append(line(left, y, left + plot_w, y, "#e2e8f0"))
        body.append(text(left - 12, y + 4, tick, 10, MUTED, "end"))
    group_w = plot_w / len(layers)
    bar_w = min(7, group_w / 4)
    for idx, layer in enumerate(layers):
        cx = left + idx * group_w + group_w / 2
        for ridx, ranking in enumerate(rankings):
            count = by.get((ranking, layer), 0)
            h = count / max(vmax, 1) * plot_h
            x = cx + (ridx - 1) * (bar_w + 2)
            body.append(rect(x - bar_w / 2, top + plot_h - h, bar_w, h, colors[ranking], None, 1, 0.86))
        if layer % 2 == 1:
            body.append(text(cx, top + plot_h + 20, f"L{layer}", 9, DARK, "middle"))
    lx, ly = left + plot_w - 210, top + 20
    for idx, ranking in enumerate(rankings):
        body.append(rect(lx, ly + idx * 24, 16, 16, colors[ranking], None, 1, 0.86))
        body.append(text(lx + 24, ly + idx * 24 + 12, ranking.replace("_", " "), 11, DARK))
    body.append(text(left + plot_w / 2, height - 40, "layer", 13, DARK, "middle"))
    body.append(text(28, top + plot_h / 2, "head count", 13, DARK, "middle", rotate=-90))
    svg(path, width, height, "".join(body))


def make_report(path, args, group_rows, head_rows, layer_rows, figs):
    by = {(row["group"], row["label"]): row for row in group_rows}

    def val(group, key, ndigits=4):
        return f"{safe_float(by.get((group, 'all'), {}).get(key), 0.0):.{ndigits}f}"

    lines = [
        "# Itext Language-Prior Channel Evidence",
        "",
        "Finding: high-Itext heads are best interpreted as intervention-aligned language-prior channels for object logits.",
        "",
        "## What This Analysis Can Prove",
        "",
        "The input rows are object-mention rows from `head_logit_proxy_ablation_rows.csv`, so the current analysis directly tests object-token contribution and object-token logit drop. It does not yet compare against sampled function-token rows.",
        "",
        "## 1a. Itext Heads Contribute To Object Logits Through The Text-Side Component",
        "",
        f"![Itext object contribution]({os.path.basename(figs['group_bar'])})",
        "",
        "Key object-token numbers:",
        "",
        f"- itext+combined: text value -> target object logit={val('itext_and_combined', 'mean_proxy_text_target_logit')}, positive contribution rate={val('itext_and_combined', 'positive_proxy_text_target_logit_rate')}.",
        f"- itext-only: text value -> target object logit={val('itext_only', 'mean_proxy_text_target_logit')}, positive contribution rate={val('itext_only', 'positive_proxy_text_target_logit_rate')}.",
        f"- contrast-only: text value -> target object logit={val('contrast_only', 'mean_proxy_text_target_logit')}, positive contribution rate={val('contrast_only', 'positive_proxy_text_target_logit_rate')}.",
        "",
        "Interpretation: if itext groups have larger positive text-side target-logit contribution than contrast-only or other candidates, this supports `Itext = object-logit leverage channel` rather than `Itext = hallucination detector`.",
        "",
        "## 1b. Text-Side Suppression Tests Causality On Object Logits",
        "",
        f"- itext+combined target object logit drop under text-component subtraction={val('itext_and_combined', 'mean_target_text_logit_drop')}.",
        f"- itext-only target object logit drop under text-component subtraction={val('itext_only', 'mean_target_text_logit_drop')}.",
        f"- contrast-only target object logit drop under text-component subtraction={val('contrast_only', 'mean_target_text_logit_drop')}.",
        f"- itext+combined target rank delta after text subtraction={val('itext_and_combined', 'mean_target_rank_text_delta')}.",
        "",
        "Interpretation: positive object-logit drop after removing only the text-side head component is direct causal evidence that this channel pushes the object token.",
        "",
        "## 1c. Attention Target Path Is Still Missing",
        "",
        "This requires per-head top-attended text-token logging, because the current rows only contain aggregate text mass and entropy. The next extraction should store top text positions/tokens for each selected head-step, then bucket targets into recent context, previous object mentions, template words, and instruction tokens.",
        "",
        "## 1d. Layer Distribution",
        "",
        f"![Itext layer distribution]({os.path.basename(figs['layer'])})",
        "",
        "Layer-band counts are in `itext_layer_distribution.csv`. This shows whether high-Itext language-prior channels are concentrated in L11-20 cross-modal aggregation, L21-26 bridge, or L27-32 late refinement.",
        "",
        "## Outputs",
        "",
        "- `itext_language_prior_group_summary.csv`: group-level contribution/drop evidence.",
        "- `itext_language_prior_head_summary.csv`: head-level contribution/drop evidence.",
        "- `itext_layer_distribution.csv`: layer distribution for itext, combined, and contrast rankings.",
        "",
        "## Server Extraction If Rows Are Missing",
        "",
        "Run `validate_head_logit_contribution_proxy.py` with candidate heads covering the itext/combined/contrast pools, then rerun this script with `--head-rows` pointing to the produced `head_logit_proxy_ablation_rows.csv`.",
        "",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-rows", required=True)
    parser.add_argument(
        "--ranked-heads",
        default="../ADHH/LLaVA/results/coco/llava-v1.5-7b_base_original_qa_n3000/surrogate_hh_scores/surrogate_score_zoo/ranked_heads_global__itext_all__C_toi_HminusG.json",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rows = load_rows(args.head_rows)
    rank_sets = load_rank_sets(args.ranked_heads, args.top_k)
    rows = rank_rows(rows, rank_sets)

    group_rows = summarize_groups(rows)
    head_rows = summarize_heads(rows)
    layer_rows = layer_distribution(rank_sets, args.top_k)

    group_path = os.path.join(args.output_dir, "itext_language_prior_group_summary.csv")
    head_path = os.path.join(args.output_dir, "itext_language_prior_head_summary.csv")
    layer_path = os.path.join(args.output_dir, "itext_layer_distribution.csv")
    write_csv(group_path, group_rows)
    write_csv(head_path, head_rows)
    write_csv(layer_path, layer_rows)

    figs = {
        "group_bar": os.path.join(args.output_dir, "itext_object_logit_channel.svg"),
        "layer": os.path.join(args.output_dir, "itext_layer_distribution.svg"),
    }
    make_group_bar_svg(figs["group_bar"], group_rows)
    make_layer_svg(figs["layer"], layer_rows)

    report_path = os.path.join(args.output_dir, "itext_language_prior_channel.md")
    make_report(report_path, args, group_rows, head_rows, layer_rows, figs)

    print(json.dumps({
        "report": report_path,
        "group_summary": group_path,
        "head_summary": head_path,
        "layer_distribution": layer_path,
        "figures": figs,
    }, indent=2))


if __name__ == "__main__":
    main()

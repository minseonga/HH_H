#!/usr/bin/env python3
"""Build locked Section III-D actuator figures and significance summaries.

Inputs:
  1. head-level score CSV with selected/non-selected L9-L16 heads.
  2. exact top-100 L9-L16 static-object logprob probe rows.

The script intentionally separates:
  - D1: non-selection image-routing check.
  - D2: causal fragility under the exact selected-pool hard/static probe.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "dark": "#111827",
    "muted": "#64748b",
    "grid": "#e5e7eb",
    "axis": "#cbd5e1",
    "selected": "#7c3aed",
    "nonselected": "#94a3b8",
    "grounded": "#059669",
    "hall": "#dc2626",
}


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_head_score_rows(path, top_k):
    path = Path(path)
    if path.suffix.lower() == ".json":
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        records = data.get("heads", data if isinstance(data, list) else [])
        rows = []
        for idx, item in enumerate(records):
            layer = int(item["layer"])
            head = int(item["head"])
            score_key = data.get("score_name", "score") if isinstance(data, dict) else "score"
            score = item.get(score_key, item.get("score", item.get("fused_score", 0.0)))
            rows.append(
                {
                    "rank": idx + 1,
                    "layer": layer,
                    "head": head,
                    "head_key": item.get("head_id", f"{layer}:{head}").replace("L", "").replace("H", ":")
                    if "head_id" in item
                    else f"{layer}:{head}",
                    "selected": 1 if idx < top_k else 0,
                    "selection_allowed": 1,
                    "score": score,
                    "image_mass_hallucinated": item.get("Img_hallucinated", item.get("image_mass_hallucinated", 0.0)),
                    "image_mass_grounded": item.get(
                        "Img_non_hallucinated",
                        item.get("Img_grounded", item.get("image_mass_grounded", 0.0)),
                    ),
                }
            )
        return rows
    return read_csv(path)


def write_csv(path, rows):
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def as_float(row, key, default=0.0):
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def as_int(row, key, default=0):
    try:
        return int(float(row.get(key, default)))
    except (TypeError, ValueError):
        return default


def mean(values):
    arr = np.asarray(values, dtype=np.float64)
    return float(np.mean(arr)) if arr.size else float("nan")


def quantile(values, q):
    arr = np.asarray(values, dtype=np.float64)
    return float(np.quantile(arr, q)) if arr.size else float("nan")


def rankdata_average(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    tie_counts = []
    pos = 0
    while pos < len(order):
        end = pos + 1
        while end < len(order) and values[order[end]] == values[order[pos]]:
            end += 1
        avg_rank = (pos + 1 + end) / 2.0
        for idx in range(pos, end):
            ranks[order[idx]] = avg_rank
        tie_counts.append(end - pos)
        pos = end
    return ranks, tie_counts


def mann_whitney_u(x, y):
    """Two-sided Mann-Whitney U with normal tie correction.

    This avoids a scipy dependency for server portability. For these sample
    sizes, the normal approximation is adequate for reporting a diagnostic
    p-value.
    """
    x = [float(v) for v in x]
    y = [float(v) for v in y]
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return {"u": None, "p_two_sided": None, "auc_probability": None}
    values = x + y
    ranks, tie_counts = rankdata_average(values)
    r1 = sum(ranks[:n1])
    u1 = r1 - n1 * (n1 + 1) / 2.0
    u2 = n1 * n2 - u1
    u = min(u1, u2)
    mean_u = n1 * n2 / 2.0
    n = n1 + n2
    tie_term = sum(t ** 3 - t for t in tie_counts)
    var_u = n1 * n2 / 12.0 * ((n + 1) - tie_term / max(n * (n - 1), 1))
    if var_u <= 0:
        p = 1.0
    else:
        z = (abs(u1 - mean_u) - 0.5) / math.sqrt(var_u)
        p = math.erfc(abs(z) / math.sqrt(2.0))
    return {
        "u": float(u1),
        "u_min": float(u),
        "p_two_sided": float(p),
        "auc_probability": float(u1 / (n1 * n2)),
    }


def bootstrap_ci(values, n_boot=5000, alpha=0.05, seed=13):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def setup_style():
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 320,
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": COLORS["axis"],
            "axes.linewidth": 0.9,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig, output_dir, name, formats):
    paths = {}
    for fmt in formats:
        path = Path(output_dir) / f"{name}.{fmt}"
        fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
        paths[fmt] = str(path)
    plt.close(fig)
    return paths


def filter_l9_l16_heads(rows):
    out = []
    for row in rows:
        layer = as_int(row, "layer")
        if 9 <= layer <= 16 and as_int(row, "selection_allowed", 1) == 1:
            out.append(row)
    return out


def d1_image_drop_rows(head_rows):
    rows = []
    for row in filter_l9_l16_heads(head_rows):
        selected = as_int(row, "selected")
        img_g = as_float(row, "image_mass_grounded")
        img_h = as_float(row, "image_mass_hallucinated")
        rows.append(
            {
                "group": "selected" if selected else "non-selected",
                "layer": as_int(row, "layer"),
                "head": as_int(row, "head"),
                "head_key": row.get("head_key", f"{row.get('layer')}:{row.get('head')}"),
                "image_mass_grounded": img_g,
                "image_mass_hallucinated": img_h,
                "image_drop_GminusH": img_g - img_h,
                "score": as_float(row, "score"),
            }
        )
    return rows


def summarize_group(values):
    ci_low, ci_high = bootstrap_ci(values)
    return {
        "n": len(values),
        "mean": mean(values),
        "median": quantile(values, 0.5),
        "q25": quantile(values, 0.25),
        "q75": quantile(values, 0.75),
        "q90": quantile(values, 0.9),
        "mean_ci95_low": ci_low,
        "mean_ci95_high": ci_high,
    }


def plot_d1(output_dir, formats, rows, stats):
    setup_style()
    groups = ["non-selected", "selected"]
    colors = [COLORS["nonselected"], COLORS["selected"]]
    data = [[row["image_drop_GminusH"] for row in rows if row["group"] == group] for group in groups]
    means = [mean(vals) for vals in data]
    ci = [bootstrap_ci(vals) for vals in data]

    fig, ax = plt.subplots(figsize=(4.1, 3.1), constrained_layout=True)
    x = np.arange(len(groups))
    yerr = np.array([[means[i] - ci[i][0] for i in range(len(groups))], [ci[i][1] - means[i] for i in range(len(groups))]])
    ax.bar(x, means, yerr=yerr, width=0.58, color=colors, alpha=0.9, capsize=3, error_kw={"elinewidth": 1.1})
    rng = np.random.default_rng(7)
    for i, vals in enumerate(data):
        jitter = rng.normal(0, 0.055, size=len(vals))
        ax.scatter(np.full(len(vals), x[i]) + jitter, vals, s=10, color=colors[i], alpha=0.36, linewidths=0)
        ax.text(x[i], means[i] + 0.006, f"{means[i]:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ratio = means[1] / means[0] if means[0] else float("nan")
    ax.text(
        0.5,
        max(max(vals) for vals in data) + 0.015,
        f"{ratio:.1f}x image-drop gap",
        ha="center",
        va="center",
        fontsize=9.3,
        color=COLORS["selected"],
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#f5f3ff", edgecolor="none"),
    )
    p = stats["mann_whitney_p_two_sided"]
    ax.text(0.5, -0.09, f"Mann-Whitney p={p:.2e}", ha="center", va="top", fontsize=8.2, color=COLORS["muted"])
    ax.axhline(0, color="#94a3b8", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(["non-selected", "selected"], fontsize=9)
    ax.set_ylabel("image mass drop G-H", fontsize=9.5)
    ax.set_title("Selected heads weaken visual routing at hallucinated steps", fontsize=10.7, fontweight="bold")
    ax.grid(axis="y")
    return save(fig, output_dir, "section_d_image_mass_drop_locked", formats)


def d2_fragility_rows(rows):
    out = []
    for row in rows:
        label = row.get("label")
        if label not in {"grounded_object", "hallucinated_object"}:
            continue
        base_next = row.get("base_next_token_id")
        static_next = row.get("static_next_token_id")
        out.append(
            {
                "label": label,
                "short_label": "grounded" if label == "grounded_object" else "hallucinated",
                "delta_logprob": as_float(row, "target_logprob_drop_static"),
                "top1_changed": 1.0 if str(base_next) != str(static_next) else 0.0,
                "object_word": row.get("object_word", ""),
                "image_id": row.get("image_id", ""),
            }
        )
    return out


def plot_d2(output_dir, formats, rows, stats):
    setup_style()
    groups = ["grounded", "hallucinated"]
    labels = ["grounded\nobject", "hallucinated\nobject"]
    colors = [COLORS["grounded"], COLORS["hall"]]
    data = [[row["delta_logprob"] for row in rows if row["short_label"] == group] for group in groups]
    flips = [[row["top1_changed"] for row in rows if row["short_label"] == group] for group in groups]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), constrained_layout=True, gridspec_kw={"width_ratios": [1.35, 0.9]})

    ax = axes[0]
    parts = ax.violinplot(data, positions=[0, 1], widths=0.66, showmeans=False, showextrema=False, showmedians=False)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.24)
    bp = ax.boxplot(
        data,
        positions=[0, 1],
        widths=0.26,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": COLORS["dark"], "linewidth": 1.3},
        boxprops={"linewidth": 1.0, "edgecolor": COLORS["dark"]},
        whiskerprops={"linewidth": 1.0, "color": COLORS["dark"]},
        capprops={"linewidth": 1.0, "color": COLORS["dark"]},
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    rng = np.random.default_rng(11)
    for i, vals in enumerate(data):
        sample = np.asarray(vals)
        if sample.size > 120:
            sample = rng.choice(sample, size=120, replace=False)
        jitter = rng.normal(0, 0.045, size=sample.size)
        ax.scatter(np.full(sample.size, i) + jitter, sample, s=8, color=colors[i], alpha=0.22, linewidths=0)
        ax.text(i, max(vals) + 0.035, f"mean {mean(vals):.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=colors[i])
    ax.axhline(0, color="#94a3b8", linewidth=0.9)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(r"$\Delta \log p(y_t)$", fontsize=9.5)
    ax.set_title("Object-token fragility under selected-pool suppression", fontsize=10.2, fontweight="bold")
    ax.grid(axis="y")
    p = stats["delta_logprob_mann_whitney_p_two_sided"]
    ax.text(0.5, -0.23, f"Mann-Whitney p={p:.2e}", transform=ax.transAxes, ha="center", fontsize=8.2, color=COLORS["muted"])

    ax = axes[1]
    rates = [mean(vals) for vals in flips]
    ci = [bootstrap_ci(vals) for vals in flips]
    x = np.arange(2)
    yerr = np.array([[rates[i] - ci[i][0] for i in range(2)], [ci[i][1] - rates[i] for i in range(2)]])
    ax.bar(x, rates, width=0.58, color=colors, alpha=0.88, yerr=yerr, capsize=3, error_kw={"elinewidth": 1.1})
    for i, rate in enumerate(rates):
        ax.text(i, rate + 0.012, f"{rate * 100:.1f}%", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, max(rates) * 1.55 if max(rates) > 0 else 1)
    ax.set_ylabel("top-1 token changed", fontsize=9.5)
    ax.set_title("Discrete top-1 effect", fontsize=10.2, fontweight="bold")
    ax.grid(axis="y")

    return save(fig, output_dir, "section_d_causal_fragility_locked", formats)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--head-scores-path",
        default="LLaVA/results/coco/layer_band_dynamic_ablation_head_files/ranked_heads_global__itext_all__C_toi_HminusG_l9_l16.json",
    )
    parser.add_argument("--fragility-rows-csv", required=True)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--output-dir", default="LLaVA/results/coco/section_d_locked_figures")
    parser.add_argument("--formats", default="svg,png,pdf")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    formats = [item.strip() for item in args.formats.split(",") if item.strip()]

    head_rows = load_head_score_rows(args.head_scores_path, args.top_k)
    d1_rows = d1_image_drop_rows(head_rows)
    write_csv(output_dir / "section_d_image_mass_drop_rows.csv", d1_rows)
    selected_drop = [row["image_drop_GminusH"] for row in d1_rows if row["group"] == "selected"]
    non_drop = [row["image_drop_GminusH"] for row in d1_rows if row["group"] == "non-selected"]
    d1_test = mann_whitney_u(selected_drop, non_drop)
    d1_summary = {
        "selected": summarize_group(selected_drop),
        "non-selected": summarize_group(non_drop),
        "selected_over_nonselected_mean_ratio": mean(selected_drop) / mean(non_drop),
        "mann_whitney_u": d1_test["u"],
        "mann_whitney_p_two_sided": d1_test["p_two_sided"],
        "auc_probability_selected_gt_nonselected": d1_test["auc_probability"],
    }
    d1_paths = plot_d1(output_dir, formats, d1_rows, d1_summary)

    fragility_source_rows = read_csv(args.fragility_rows_csv)
    d2_rows = d2_fragility_rows(fragility_source_rows)
    write_csv(output_dir / "section_d_causal_fragility_rows.csv", d2_rows)
    grounded = [row["delta_logprob"] for row in d2_rows if row["short_label"] == "grounded"]
    hall = [row["delta_logprob"] for row in d2_rows if row["short_label"] == "hallucinated"]
    grounded_flip = [row["top1_changed"] for row in d2_rows if row["short_label"] == "grounded"]
    hall_flip = [row["top1_changed"] for row in d2_rows if row["short_label"] == "hallucinated"]
    d2_test = mann_whitney_u(hall, grounded)
    d2_summary = {
        "grounded": summarize_group(grounded),
        "hallucinated": summarize_group(hall),
        "h_minus_g_mean_delta_logprob": mean(hall) - mean(grounded),
        "grounded_top1_changed_rate": mean(grounded_flip),
        "hallucinated_top1_changed_rate": mean(hall_flip),
        "top1_changed_rate_gap_H_minus_G": mean(hall_flip) - mean(grounded_flip),
        "delta_logprob_mann_whitney_u": d2_test["u"],
        "delta_logprob_mann_whitney_p_two_sided": d2_test["p_two_sided"],
        "auc_probability_hall_gt_grounded": d2_test["auc_probability"],
    }
    d2_paths = plot_d2(output_dir, formats, d2_rows, d2_summary)

    summary = {
        "head_scores_path": args.head_scores_path,
        "top_k": args.top_k,
        "fragility_rows_csv": args.fragility_rows_csv,
        "d1_image_mass_drop": d1_summary,
        "d2_causal_fragility": d2_summary,
        "figures": {
            "image_mass_drop": d1_paths,
            "causal_fragility": d2_paths,
        },
    }
    with (output_dir / "section_d_locked_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_csv(
        output_dir / "section_d_locked_summary_flat.csv",
        [
            {
                "metric": "D1 image drop selected",
                **d1_summary["selected"],
                "p_two_sided": d1_summary["mann_whitney_p_two_sided"],
            },
            {
                "metric": "D1 image drop non-selected",
                **d1_summary["non-selected"],
                "p_two_sided": d1_summary["mann_whitney_p_two_sided"],
            },
            {
                "metric": "D2 delta logp grounded",
                **d2_summary["grounded"],
                "p_two_sided": d2_summary["delta_logprob_mann_whitney_p_two_sided"],
                "top1_changed_rate": d2_summary["grounded_top1_changed_rate"],
            },
            {
                "metric": "D2 delta logp hallucinated",
                **d2_summary["hallucinated"],
                "p_two_sided": d2_summary["delta_logprob_mann_whitney_p_two_sided"],
                "top1_changed_rate": d2_summary["hallucinated_top1_changed_rate"],
            },
        ],
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

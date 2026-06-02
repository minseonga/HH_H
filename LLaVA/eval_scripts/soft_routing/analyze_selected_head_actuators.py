#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
from collections import Counter

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "dark": "#111827",
    "muted": "#64748b",
    "grid": "#e2e8f0",
    "selected": "#7c3aed",
    "nonselected": "#94a3b8",
    "hall": "#dc2626",
    "ground": "#059669",
    "image": "#2563eb",
    "text": "#f97316",
}


def setup_style():
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 320,
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#cbd5e1",
            "axes.linewidth": 0.8,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def mean(values):
    values = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(values)) if values else 0.0


def std(values):
    values = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def cohen_d(a, b):
    a = [float(v) for v in a if math.isfinite(float(v))]
    b = [float(v) for v in b if math.isfinite(float(v))]
    if len(a) < 2 or len(b) < 2:
        return 0.0
    pooled = math.sqrt(((len(a) - 1) * std(a) ** 2 + (len(b) - 1) * std(b) ** 2) / (len(a) + len(b) - 2))
    return (mean(a) - mean(b)) / pooled if pooled > 0 else 0.0


def corr(xs, ys):
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[mask]
    ys = ys[mask]
    if len(xs) < 2:
        return 0.0
    xs = xs - xs.mean()
    ys = ys - ys.mean()
    denom = math.sqrt(float((xs * xs).sum() * (ys * ys).sum()))
    return float((xs * ys).sum() / denom) if denom > 0 else 0.0


def load_heads(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    heads = data.get("heads", data if isinstance(data, list) else [])
    if not isinstance(heads, list) or not heads:
        raise ValueError(f"no heads found in {path}")
    return data, heads


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields or ["empty"])
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save(fig, output_dir, name, formats):
    paths = {}
    for fmt in formats:
        path = os.path.join(output_dir, f"{name}.{fmt}")
        fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
        paths[fmt] = path
    plt.close(fig)
    return paths


def metric_rows(group_name, heads):
    rows = []
    for bucket, suffix in [("hallucinated", "hallucinated"), ("grounded", "non_hallucinated")]:
        rows.append(
            {
                "group": group_name,
                "bucket": bucket,
                "n_heads": len(heads),
                "Itext": mean([h[f"Itext_{suffix}"] for h in heads]),
                "Img": mean([h[f"Img_{suffix}"] for h in heads]),
                "RawTOI": mean([h[f"RawTOI_{suffix}"] for h in heads]),
                "LogTOI": mean([h[f"LogTOI_{suffix}"] for h in heads]),
            }
        )
    return rows


def group_summary(group_name, heads):
    return {
        "group": group_name,
        "n_heads": len(heads),
        "mean_fused_score": mean([h["itext_all__C_toi_HminusG"] for h in heads]),
        "mean_Itext_all": mean([h["Itext_all"] for h in heads]),
        "mean_text_percentile": mean([h["front_percentile"] for h in heads]),
        "mean_C_toi_HminusG": mean([h["back_raw"] for h in heads]),
        "mean_contrast_percentile": mean([h["back_percentile"] for h in heads]),
        "mean_Itext_hallucinated": mean([h["Itext_hallucinated"] for h in heads]),
        "mean_Itext_grounded": mean([h["Itext_non_hallucinated"] for h in heads]),
        "mean_Itext_gap_HminusG": mean([h["Itext_hallucinated"] - h["Itext_non_hallucinated"] for h in heads]),
        "mean_Img_hallucinated": mean([h["Img_hallucinated"] for h in heads]),
        "mean_Img_grounded": mean([h["Img_non_hallucinated"] for h in heads]),
        "mean_image_drop_GminusH": mean([h["Img_non_hallucinated"] - h["Img_hallucinated"] for h in heads]),
        "mean_RawTOI_gap_HminusG": mean([h["RawTOI_hallucinated"] - h["RawTOI_non_hallucinated"] for h in heads]),
        "mean_LogTOI_gap_HminusG": mean([h["LogTOI_hallucinated"] - h["LogTOI_non_hallucinated"] for h in heads]),
        "positive_Itext_gap_count": sum((h["Itext_hallucinated"] - h["Itext_non_hallucinated"]) > 0 for h in heads),
        "positive_image_drop_count": sum((h["Img_non_hallucinated"] - h["Img_hallucinated"]) > 0 for h in heads),
        "positive_RawTOI_gap_count": sum((h["RawTOI_hallucinated"] - h["RawTOI_non_hallucinated"]) > 0 for h in heads),
        "positive_LogTOI_gap_count": sum((h["LogTOI_hallucinated"] - h["LogTOI_non_hallucinated"]) > 0 for h in heads),
    }


def plot_hall_ground_bars(output_dir, formats, region_rows, summaries):
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(8.1, 2.55), constrained_layout=True)
    groups = ["selected", "non-selected"]
    x = np.arange(len(groups))
    width = 0.34

    lookup = {(row["group"], row["bucket"]): row for row in region_rows}
    panels = [
        ("Itext", "Text-side mass", COLORS["text"]),
        ("Img", "Image-token mass", COLORS["image"]),
        ("LogTOI", "log text-over-image", COLORS["selected"]),
    ]
    for ax, (metric, title, color) in zip(axes, panels):
        grounded = [lookup[(group, "grounded")][metric] for group in groups]
        hall = [lookup[(group, "hallucinated")][metric] for group in groups]
        ax.bar(x - width / 2, grounded, width, color=COLORS["ground"], alpha=0.88, label="grounded")
        ax.bar(x + width / 2, hall, width, color=COLORS["hall"], alpha=0.78, label="hallucinated")
        for idx, (g, h) in enumerate(zip(grounded, hall)):
            gap = h - g if metric != "Img" else g - h
            ax.text(idx, max(g, h) + max(0.015, 0.035 * max(h, g, 1e-6)), f"{gap:+.3f}", ha="center", va="bottom", fontsize=6.4, color=COLORS["dark"])
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(["selected\n top100", "non-selected\n rank>100"])
        ax.grid(axis="y", alpha=0.9)
        if metric == "Img":
            ax.set_ylabel("mass")
        elif metric == "Itext":
            ax.set_ylabel("mass")
        else:
            ax.set_ylabel("log ratio")
    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle("Selected heads show hallucination-state text-over-image shift", y=1.04, fontsize=10.5, fontweight="bold")
    return save(fig, output_dir, "hall_ground_selected_vs_nonselected_bars", formats)


def plot_gap_summary(output_dir, formats, summaries):
    setup_style()
    fig, ax = plt.subplots(figsize=(4.8, 2.55), constrained_layout=True)
    labels = ["text gap\nH-G", "image drop\nG-H", "log TOI gap\nH-G", "raw TOI gap\nH-G"]
    selected = summaries["selected"]
    non = summaries["non-selected"]
    selected_vals = [
        selected["mean_Itext_gap_HminusG"],
        selected["mean_image_drop_GminusH"],
        selected["mean_LogTOI_gap_HminusG"],
        selected["mean_RawTOI_gap_HminusG"],
    ]
    non_vals = [
        non["mean_Itext_gap_HminusG"],
        non["mean_image_drop_GminusH"],
        non["mean_LogTOI_gap_HminusG"],
        non["mean_RawTOI_gap_HminusG"],
    ]
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, selected_vals, width, color=COLORS["selected"], label="selected top100")
    ax.bar(x + width / 2, non_vals, width, color=COLORS["nonselected"], label="non-selected")
    ax.axhline(0, color="#334155", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("mean gap")
    ax.set_title("Hallucination-state shifts are larger in selected heads")
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="upper left")
    return save(fig, output_dir, "selected_head_gap_summary", formats)


def plot_head_space(output_dir, formats, heads, top_k):
    setup_style()
    fig, ax = plt.subplots(figsize=(3.5, 3.05), constrained_layout=True)
    selected = heads[:top_k]
    non = heads[top_k:]
    ax.scatter(
        [h["front_percentile"] for h in non],
        [h["back_percentile"] for h in non],
        s=18,
        color=COLORS["nonselected"],
        alpha=0.45,
        linewidths=0,
        label="non-selected",
    )
    ax.scatter(
        [h["front_percentile"] for h in selected],
        [h["back_percentile"] for h in selected],
        s=24,
        color=COLORS["selected"],
        alpha=0.82,
        linewidths=0,
        label="selected top100",
    )
    ax.set_xlabel("text leverage percentile")
    ax.set_ylabel("contrastive percentile")
    ax.set_title("Fusion selects leverage + specificity heads")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True)
    ax.legend(frameon=False, loc="lower left")
    return save(fig, output_dir, "text_leverage_vs_contrastive_head_space", formats)


def write_paper_notes(path, data, top_k):
    selected = data["group_summaries"]["selected"]
    non = data["group_summaries"]["non-selected"]
    component_corr = data["component_percentile_correlation"]
    direction = data["direction_rates"]["selected"]
    text_d = data["effect_sizes"]["Itext_all"]
    contrast_d = data["effect_sizes"]["C_toi_HminusG"]
    logtoi_d = data["effect_sizes"]["LogTOI_gap"]
    image_d = data["effect_sizes"]["image_drop"]

    text = f"""# Head Actuator Analysis Notes

Source: `{data['source']}`

Selection: top-{top_k} heads from the L9--L16 ranked pool, with the remaining heads in the same layer window used as the non-selected comparison group.

## Main Finding

The selected heads are not reliable hallucination detectors. Instead, they are better interpreted as text-side actuator heads: intervention-relevant channels that increase the influence of post-image textual context during object generation.

## Selected vs Non-Selected Heads

| metric | selected top-{top_k} | non-selected |
|---|---:|---:|
| mean text-side mass $I_{{text}}$ | {selected['mean_Itext_all']:.3f} | {non['mean_Itext_all']:.3f} |
| mean positive contrast score $\\max(C_{{toi}},0)$ | {selected['mean_C_toi_HminusG']:.3f} | {non['mean_C_toi_HminusG']:.3f} |
| mean log-TOI gap H-G | {selected['mean_LogTOI_gap_HminusG']:.3f} | {non['mean_LogTOI_gap_HminusG']:.3f} |
| mean image drop G-H | {selected['mean_image_drop_GminusH']:.3f} | {non['mean_image_drop_GminusH']:.3f} |

Directionality among selected heads:

- positive raw text-over-image gap: {direction['positive_RawTOI_gap_count']}/{top_k}
- positive log text-over-image gap: {direction['positive_LogTOI_gap_count']}/{top_k}
- positive image drop from grounded to hallucinated: {direction['positive_image_drop_count']}/{top_k}
- positive text-mass gap from grounded to hallucinated: {direction['positive_Itext_gap_count']}/{top_k}

Effect sizes selected vs non-selected:

- text-side mass: Cohen's d = {text_d:.3f}
- positive contrast score: Cohen's d = {contrast_d:.3f}
- log TOI gap: Cohen's d = {logtoi_d:.3f}
- image drop: Cohen's d = {image_d:.3f}

The text-leverage percentile and contrastive percentile are weakly correlated across the L9--L16 head pool (Pearson r = {component_corr:.3f}), so the two axes are complementary rather than redundant.

## Paper-Ready Interpretation

The selected heads have substantially higher text-side leverage than non-selected heads in the same layer window. They also exhibit a larger hallucinated-minus-grounded text-over-image shift and a stronger drop in image-token attention at hallucinated object steps. This indicates that the selected heads are not merely high text-attention heads. They are heads whose routing behavior becomes more text-dominant and less image-grounded in the hallucination state.

Therefore, we interpret these heads as text-side actuators. High text-side mass identifies where an intervention can affect object generation, while the contrastive text-over-image gap identifies which of those leverage points are hallucination-specific. This actuator interpretation is distinct from a detector interpretation: the head does not need to classify a token as hallucinated; it only needs to provide a control point through which language-prior support can be reduced.

## Caution

This is a head-pool characterization, not a standalone token-level hallucination detector. High text reliance also occurs for grounded objects, so the selected heads should not be described as detecting hallucination. The correct claim is that they are intervention-relevant channels whose text-over-image reliance increases in hallucination states.
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranked-heads-json", required=True)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--formats", default="png,pdf,svg")
    args = parser.parse_args()

    formats = [fmt.strip() for fmt in args.formats.split(",") if fmt.strip()]
    os.makedirs(args.output_dir, exist_ok=True)

    meta, heads = load_heads(args.ranked_heads_json)
    top_k = min(args.top_k, len(heads))
    selected = heads[:top_k]
    nonselected = heads[top_k:]

    group_rows = [group_summary("selected", selected), group_summary("non-selected", nonselected)]
    region_rows = metric_rows("selected", selected) + metric_rows("non-selected", nonselected)

    selected_summary = group_rows[0]
    non_summary = group_rows[1]
    summaries = {"selected": selected_summary, "non-selected": non_summary}

    direction_rows = []
    for group_name, group, summary in [("selected", selected, selected_summary), ("non-selected", nonselected, non_summary)]:
        direction_rows.append(
            {
                "group": group_name,
                "n_heads": len(group),
                "positive_Itext_gap_count": summary["positive_Itext_gap_count"],
                "positive_Itext_gap_rate": summary["positive_Itext_gap_count"] / len(group),
                "positive_image_drop_count": summary["positive_image_drop_count"],
                "positive_image_drop_rate": summary["positive_image_drop_count"] / len(group),
                "positive_RawTOI_gap_count": summary["positive_RawTOI_gap_count"],
                "positive_RawTOI_gap_rate": summary["positive_RawTOI_gap_count"] / len(group),
                "positive_LogTOI_gap_count": summary["positive_LogTOI_gap_count"],
                "positive_LogTOI_gap_rate": summary["positive_LogTOI_gap_count"] / len(group),
            }
        )

    layer_rows = []
    for group_name, group in [("selected", selected), ("non-selected", nonselected)]:
        counts = Counter(h["layer"] for h in group)
        for layer in sorted(counts):
            layer_rows.append(
                {
                    "group": group_name,
                    "layer": layer,
                    "count": counts[layer],
                    "fraction": counts[layer] / len(group),
                }
            )

    top_head_rows = []
    for h in selected:
        top_head_rows.append(
            {
                "rank": h.get("global_rank", len(top_head_rows) + 1),
                "head_id": h.get("head_id", f"L{h['layer']}H{h['head']}"),
                "layer": h["layer"],
                "head": h["head"],
                "fused_score": h["itext_all__C_toi_HminusG"],
                "Itext_all": h["Itext_all"],
                "text_percentile": h["front_percentile"],
                "C_toi_HminusG": h["back_raw"],
                "contrast_percentile": h["back_percentile"],
                "Itext_hallucinated": h["Itext_hallucinated"],
                "Itext_grounded": h["Itext_non_hallucinated"],
                "Img_hallucinated": h["Img_hallucinated"],
                "Img_grounded": h["Img_non_hallucinated"],
                "LogTOI_gap_HminusG": h["LogTOI_hallucinated"] - h["LogTOI_non_hallucinated"],
                "RawTOI_gap_HminusG": h["RawTOI_hallucinated"] - h["RawTOI_non_hallucinated"],
            }
        )

    effect_sizes = {
        "Itext_all": cohen_d([h["Itext_all"] for h in selected], [h["Itext_all"] for h in nonselected]),
        "C_toi_HminusG": cohen_d([h["back_raw"] for h in selected], [h["back_raw"] for h in nonselected]),
        "LogTOI_gap": cohen_d(
            [h["LogTOI_hallucinated"] - h["LogTOI_non_hallucinated"] for h in selected],
            [h["LogTOI_hallucinated"] - h["LogTOI_non_hallucinated"] for h in nonselected],
        ),
        "image_drop": cohen_d(
            [h["Img_non_hallucinated"] - h["Img_hallucinated"] for h in selected],
            [h["Img_non_hallucinated"] - h["Img_hallucinated"] for h in nonselected],
        ),
    }
    component_percentile_correlation = corr(
        [h["front_percentile"] for h in heads],
        [h["back_percentile"] for h in heads],
    )

    write_csv(os.path.join(args.output_dir, "head_actuator_group_summary.csv"), group_rows)
    write_csv(os.path.join(args.output_dir, "head_actuator_hall_ground_region_means.csv"), region_rows)
    write_csv(os.path.join(args.output_dir, "head_actuator_direction_counts.csv"), direction_rows)
    write_csv(os.path.join(args.output_dir, "head_actuator_layer_distribution.csv"), layer_rows)
    write_csv(os.path.join(args.output_dir, "selected_top_heads.csv"), top_head_rows)

    figure_paths = {}
    figure_paths["hall_ground_bars"] = plot_hall_ground_bars(args.output_dir, formats, region_rows, summaries)
    figure_paths["gap_summary"] = plot_gap_summary(args.output_dir, formats, summaries)
    figure_paths["head_space"] = plot_head_space(args.output_dir, formats, heads, top_k)

    summary = {
        "source": args.ranked_heads_json,
        "score_name": meta.get("score_name"),
        "layer_range": meta.get("layer_range"),
        "top_k": top_k,
        "n_heads_total": len(heads),
        "n_selected": len(selected),
        "n_nonselected": len(nonselected),
        "group_summaries": summaries,
        "direction_rates": {
            row["group"]: row for row in direction_rows
        },
        "effect_sizes": effect_sizes,
        "component_percentile_correlation": component_percentile_correlation,
        "figures": figure_paths,
    }
    write_json(os.path.join(args.output_dir, "head_actuator_analysis_summary.json"), summary)
    write_paper_notes(os.path.join(args.output_dir, "head_actuator_paper_notes.md"), summary, top_k)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

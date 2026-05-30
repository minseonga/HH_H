import argparse
import json
import math
import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from eval_scripts.soft_routing.build_method_phase_figures import (
    add_component_ranks,
    gate_delta,
    load_ranked_heads,
    mean,
    online_ratio,
    quantile,
    safe_float,
    write_csv,
    write_json,
)


COLORS = {
    "text": "#f97316",
    "hall": "#dc2626",
    "ground": "#2563eb",
    "image": "#2563eb",
    "score": "#7c3aed",
    "tail": "#94a3b8",
    "dark": "#0f172a",
    "muted": "#64748b",
    "grid": "#e2e8f0",
    "window": "#fef3c7",
}


def setup_style():
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "font.family": "DejaVu Sans",
        "font.size": 7.2,
        "axes.titlesize": 8.2,
        "axes.labelsize": 7.2,
        "xtick.labelsize": 6.4,
        "ytick.labelsize": 6.4,
        "legend.fontsize": 6.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#cbd5e1",
        "axes.linewidth": 0.8,
        "grid.color": COLORS["grid"],
        "grid.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    root, _ = os.path.splitext(path)
    fig.savefig(root + ".pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(root + ".png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def selected_and_tail(records, top_k, tail_start):
    return records[:top_k], records[tail_start:]


def summarize(records, top_k, tail_start, strength, beta, tau):
    selected, tail = selected_and_tail(records, top_k, tail_start)
    hall = [redistribute(row, "hallucinated", strength, beta, tau) for row in selected]
    ground = [redistribute(row, "non_hallucinated", strength, beta, tau) for row in selected]
    return [{
        "top_k": top_k,
        "tail_start": tail_start,
        "selected_mean_score": mean(row["_score"] for row in selected),
        "tail_mean_score": mean(row["_score"] for row in tail),
        "selected_mean_itext_all": mean(row.get("Itext_all") for row in selected),
        "tail_mean_itext_all": mean(row.get("Itext_all") for row in tail),
        "selected_mean_log_toi_gap": mean(
            safe_float(row.get("LogTOI_hallucinated")) - safe_float(row.get("LogTOI_non_hallucinated"))
            for row in selected
        ),
        "tail_mean_log_toi_gap": mean(
            safe_float(row.get("LogTOI_hallucinated")) - safe_float(row.get("LogTOI_non_hallucinated"))
            for row in tail
        ),
        "selected_positive_raw_toi_gap_fraction": mean(
            1.0
            if safe_float(row.get("RawTOI_hallucinated")) > safe_float(row.get("RawTOI_non_hallucinated"))
            else 0.0
            for row in selected
        ),
        "tail_positive_raw_toi_gap_fraction": mean(
            1.0
            if safe_float(row.get("RawTOI_hallucinated")) > safe_float(row.get("RawTOI_non_hallucinated"))
            else 0.0
            for row in tail
        ),
        "hall_mean_ratio": mean(item["ratio"] for item in hall),
        "ground_mean_ratio": mean(item["ratio"] for item in ground),
        "hall_mean_delta": mean(item["delta"] for item in hall),
        "ground_mean_delta": mean(item["delta"] for item in ground),
        "hall_text_before": mean(item["text_before"] for item in hall),
        "hall_text_after": mean(item["text_after"] for item in hall),
        "ground_text_before": mean(item["text_before"] for item in ground),
        "ground_text_after": mean(item["text_after"] for item in ground),
    }]


def redistribute(row, label, strength, beta, tau):
    t = safe_float(row.get(f"Itext_{label}"))
    i = safe_float(row.get(f"Img_{label}"))
    ratio = t / max(t + i, 1e-12)
    delta = gate_delta(ratio, safe_float(row.get("_score")), strength, beta, tau)
    text_after_raw = (1.0 - delta) * t
    denom = max(text_after_raw + i, 1e-12)
    return {
        "ratio": ratio,
        "delta": delta,
        "text_before": t / max(t + i, 1e-12),
        "image_before": i / max(t + i, 1e-12),
        "text_after": text_after_raw / denom,
        "image_after": i / denom,
    }


def fig_text_mass(path, records, top_k, tail_start):
    selected, tail = selected_and_tail(records, top_k, tail_start)
    labels = [f"top-{top_k}", f"rank>{tail_start}"]
    values = [mean(row.get("Itext_all") for row in selected), mean(row.get("Itext_all") for row in tail)]

    fig, ax = plt.subplots(figsize=(2.15, 1.65))
    ax.bar(labels, values, color=[COLORS["text"], COLORS["tail"]], width=0.58)
    for idx, value in enumerate(values):
        ax.text(idx, value + 0.018, f"{value:.2f}", ha="center", va="bottom", color=COLORS["dark"])
    ax.set_ylim(0, max(values) * 1.32)
    ax.set_ylabel("mean text-side mass")
    ax.set_title("Text leverage")
    ax.grid(axis="y")
    save(fig, path)


def fig_contrast_bias(path, records, top_k):
    selected = records[:top_k]
    hall = np.array([safe_float(row.get("LogTOI_hallucinated")) for row in selected])
    ground = np.array([safe_float(row.get("LogTOI_non_hallucinated")) for row in selected])
    bins = np.linspace(min(hall.min(), ground.min()), max(hall.max(), ground.max()), 24)

    fig, ax = plt.subplots(figsize=(2.65, 1.75))
    ax.hist(ground, bins=bins, color=COLORS["ground"], alpha=0.45, density=True, label="grounded")
    ax.hist(hall, bins=bins, color=COLORS["hall"], alpha=0.45, density=True, label="hallucinated")
    ax.axvline(float(ground.mean()), color=COLORS["ground"], lw=1.5)
    ax.axvline(float(hall.mean()), color=COLORS["hall"], lw=1.5)
    ax.set_title("Contrastive bias")
    ax.set_xlabel("log text-over-image ratio")
    ax.set_ylabel("density")
    ax.legend(frameon=False, loc="upper left", handlelength=1.3)
    ax.grid(axis="y")
    gap = float(hall.mean() - ground.mean())
    ax.text(
        0.98,
        0.92,
        f"mean gap = {gap:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=COLORS["dark"],
        fontsize=6.8,
    )
    save(fig, path)


def fig_layer_score_profile(path, records, top_k, layer_min, layer_max, window_start, window_end):
    selected = records[:top_k]
    layers = list(range(layer_min, layer_max + 1))
    sums = []
    for layer in layers:
        sums.append(sum(row["_score"] for row in selected if int(row["layer"]) == layer))

    fig, ax = plt.subplots(figsize=(3.65, 1.75))
    ax.axvspan(window_start - 0.5, window_end + 0.5, color=COLORS["window"], zorder=0)
    ax.plot(layers, sums, color=COLORS["score"], marker="o", ms=3.2, lw=1.8)
    ax.set_xlim(layer_min - 0.5, layer_max + 0.5)
    ax.set_title("Layer score mass")
    ax.set_xlabel("layer")
    ax.set_ylabel("sum fused score")
    ax.grid(axis="y")
    ax.text(
        (window_start + window_end) / 2,
        max(sums) * 0.94 if sums else 1.0,
        "L9-L16",
        ha="center",
        va="top",
        fontsize=6.7,
        color="#92400e",
    )
    ax.set_xticks(layers)
    ax.tick_params(axis="x", labelrotation=45)
    save(fig, path)


def fig_head_heatmap(path, records, top_k, layer_min, layer_max, head_count, window_start, window_end):
    selected = records[:top_k]
    matrix = np.full((head_count, layer_max - layer_min + 1), np.nan)
    for row in selected:
        layer = int(row["layer"])
        head = int(row["head"])
        if layer_min <= layer <= layer_max and 0 <= head < head_count:
            matrix[head, layer - layer_min] = row["_score"]

    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#f1f5f9")
    fig, ax = plt.subplots(figsize=(3.75, 2.35))
    ax.imshow(matrix, aspect="auto", origin="lower", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.axvspan(window_start - layer_min - 0.5, window_end - layer_min + 0.5, color=COLORS["window"], alpha=0.26)
    ax.add_patch(Rectangle(
        (window_start - layer_min - 0.5, -0.5),
        window_end - window_start + 1,
        head_count,
        fill=False,
        edgecolor="#f59e0b",
        linewidth=1.0,
    ))
    ax.set_title("Head score heatmap")
    ax.set_xlabel("layer")
    ax.set_ylabel("head")
    ax.set_xticks(range(layer_max - layer_min + 1))
    ax.set_xticklabels([str(layer) for layer in range(layer_min, layer_max + 1)], rotation=45)
    ax.set_yticks([0, 8, 16, 24, 31])
    cbar = fig.colorbar(ax.images[0], ax=ax, fraction=0.032, pad=0.02)
    cbar.set_label("fused score", rotation=90)
    save(fig, path)


def fig_rank_fusion(path, records, top_k):
    selected = records[:top_k]
    med_text = quantile([row.get("front_percentile") for row in selected], 0.5)
    med_contrast = quantile([row.get("back_percentile") for row in selected], 0.5)
    med_score = quantile([row.get("_score") for row in selected], 0.5)

    fig, ax = plt.subplots(figsize=(4.45, 1.28))
    ax.axis("off")
    boxes = [
        (0.055, "text\nleverage", med_text, COLORS["text"]),
        (0.390, "hall-ground\ncontrast", med_contrast, COLORS["ground"]),
        (0.725, "fused\nscore", med_score, COLORS["score"]),
    ]
    for x, label, value, color in boxes:
        ax.add_patch(Rectangle((x, 0.31), 0.22, 0.40, facecolor="white", edgecolor=color, linewidth=1.25))
        ax.text(x + 0.11, 0.58, label, ha="center", va="center", fontsize=7.3, color=COLORS["dark"])
        ax.text(x + 0.11, 0.405, f"p50={value:.2f}", ha="center", va="center", fontsize=6.5, color=color)
    ax.annotate("", xy=(0.365, 0.51), xytext=(0.285, 0.51), arrowprops=dict(arrowstyle="->", lw=1.1, color=COLORS["muted"]))
    ax.text(0.325, 0.62, "+", ha="center", va="center", fontsize=9, color=COLORS["muted"])
    ax.annotate("", xy=(0.700, 0.51), xytext=(0.620, 0.51), arrowprops=dict(arrowstyle="->", lw=1.1, color=COLORS["muted"]))
    ax.text(0.660, 0.62, "avg", ha="center", va="center", fontsize=6.4, color=COLORS["muted"])
    ax.text(0.5, 0.94, "Rank fusion", ha="center", va="top", fontsize=8.2, weight="bold")
    ax.text(0.5, 0.08, f"median over top-{top_k} heads", ha="center", va="bottom", fontsize=6.4, color=COLORS["muted"])
    save(fig, path)


def fig_gate_curve(path, records, top_k, strength, beta, tau):
    selected = records[:top_k]
    hall = np.array([online_ratio(row, "hallucinated") for row in selected])
    ground = np.array([online_ratio(row, "non_hallucinated") for row in selected])
    xs = np.linspace(0.55, 1.0, 350)
    ys = np.clip(strength * np.exp(beta * (xs - tau)), 0, 1)

    fig, ax = plt.subplots(figsize=(2.75, 1.9))
    ax.plot(xs, ys, color=COLORS["score"], lw=2.0)
    ax.axvline(tau, color=COLORS["muted"], ls="--", lw=1.1)
    ax.text(tau + 0.005, 0.05, r"$\tau=0.9$", color=COLORS["muted"], fontsize=6.8)
    for y, values, color, label in [
        (-0.048, hall, COLORS["hall"], "hall"),
        (-0.087, ground, COLORS["ground"], "ground"),
    ]:
        q25, q50, q75 = np.quantile(values, [0.25, 0.5, 0.75])
        ax.hlines(y, q25, q75, color=color, lw=2.1, clip_on=False)
        ax.plot(q50, y, marker="o", ms=3.0, color=color, clip_on=False, label=label)
    ax.set_ylim(-0.13, 1.05)
    ax.set_xlim(0.55, 1.0)
    ax.set_title("Continuous gate")
    ax.set_xlabel(r"text ratio $r=T/(T+I)$")
    ax.set_ylabel(r"$\exp(q(r-\tau))$")
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="upper left", ncol=2, handlelength=1.0)
    save(fig, path)


def fig_delta_distribution(path, records, top_k, strength, beta, tau):
    selected = records[:top_k]
    hall = np.array([redistribute(row, "hallucinated", strength, beta, tau)["delta"] for row in selected])
    ground = np.array([redistribute(row, "non_hallucinated", strength, beta, tau)["delta"] for row in selected])

    fig, ax = plt.subplots(figsize=(2.05, 1.65))
    means = [hall.mean(), ground.mean()]
    lows = [means[0] - np.quantile(hall, 0.25), means[1] - np.quantile(ground, 0.25)]
    highs = [np.quantile(hall, 0.75) - means[0], np.quantile(ground, 0.75) - means[1]]
    ax.bar([0, 1], means, color=[COLORS["hall"], COLORS["ground"]], alpha=0.78, width=0.58)
    ax.errorbar([0, 1], means, yerr=[lows, highs], fmt="none", ecolor=COLORS["dark"], elinewidth=0.9, capsize=2.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["hall", "ground"])
    ax.set_ylabel(r"suppression $\delta$")
    ax.set_title("Mean suppression")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y")
    ax.text(0, hall.mean() + 0.05, f"{hall.mean():.2f}", ha="center", color=COLORS["hall"], fontsize=7.0)
    ax.text(1, ground.mean() + 0.05, f"{ground.mean():.2f}", ha="center", color=COLORS["ground"], fontsize=7.0)
    save(fig, path)


def fig_redistribution(path, records, top_k, strength, beta, tau):
    selected = records[:top_k]
    hall = [redistribute(row, "hallucinated", strength, beta, tau) for row in selected]
    ground = [redistribute(row, "non_hallucinated", strength, beta, tau) for row in selected]
    rows = [
        ("hall\nbefore", mean(x["text_before"] for x in hall), mean(x["image_before"] for x in hall)),
        ("hall\nafter", mean(x["text_after"] for x in hall), mean(x["image_after"] for x in hall)),
        ("ground\nbefore", mean(x["text_before"] for x in ground), mean(x["image_before"] for x in ground)),
        ("ground\nafter", mean(x["text_after"] for x in ground), mean(x["image_after"] for x in ground)),
    ]

    fig, ax = plt.subplots(figsize=(2.65, 1.9))
    x = np.arange(len(rows))
    image_vals = np.array([row[2] for row in rows])
    text_vals = np.array([row[1] for row in rows])
    ax.bar(x, image_vals, color=COLORS["image"], alpha=0.78, width=0.62, label="image")
    ax.bar(x, text_vals, bottom=image_vals, color=COLORS["text"], alpha=0.80, width=0.62, label="text-side")
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([row[0] for row in rows])
    ax.set_ylabel("share within T/I slice")
    ax.set_title("Suppress text, renormalize")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y")
    save(fig, path)


def fig_delta_flow(path, records, top_k, strength, beta, tau):
    row = records[0]
    score = safe_float(row.get("_score"))
    ratio = online_ratio(row, "hallucinated")
    gate = math.exp(beta * (ratio - tau))
    delta = gate_delta(ratio, score, strength, beta, tau)

    fig, ax = plt.subplots(figsize=(3.65, 1.2))
    ax.axis("off")
    blocks = [
        (0.04, f"S(l,h)\n{score:.2f}", COLORS["score"]),
        (0.29, f"r=T/(T+I)\n{ratio:.2f}", COLORS["text"]),
        (0.54, f"exp(q(r-tau))\n{gate:.2f}", COLORS["ground"]),
        (0.79, f"delta\n{delta:.2f}", COLORS["hall"]),
    ]
    for x, label, color in blocks:
        ax.add_patch(Rectangle((x, 0.33), 0.17, 0.42, facecolor="white", edgecolor=color, linewidth=1.35))
        ax.text(x + 0.085, 0.54, label, ha="center", va="center", fontsize=8, color=COLORS["dark"])
    for x0, x1 in [(0.22, 0.29), (0.47, 0.54), (0.72, 0.79)]:
        ax.annotate("", xy=(x1 - 0.01, 0.54), xytext=(x0, 0.54), arrowprops=dict(arrowstyle="->", lw=1.2, color=COLORS["muted"]))
    ax.text(0.5, 0.94, "Score-weighted delta", ha="center", va="top", fontsize=8.2, weight="bold")
    ax.text(0.5, 0.09, f"example: L{row['layer']}H{row['head']}; action: T *= (1-delta), renorm", ha="center", va="bottom", fontsize=6.2, color=COLORS["muted"])
    save(fig, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ranked-heads",
        default="../ADHH/LLaVA/results/coco/llava-v1.5-7b_base_original_qa_n3000/surrogate_hh_scores/surrogate_score_zoo/ranked_heads_global__itext_all__C_toi_HminusG.json",
    )
    parser.add_argument("--output-dir", default="./results/coco/method_paper_figures")
    parser.add_argument("--top-k", type=int, default=150)
    parser.add_argument("--tail-start", type=int, default=200)
    parser.add_argument("--layer-profile-min", type=int, default=13)
    parser.add_argument("--layer-profile-max", type=int, default=31)
    parser.add_argument("--heatmap-layer-min", type=int, default=9)
    parser.add_argument("--heatmap-layer-max", type=int, default=20)
    parser.add_argument("--head-count", type=int, default=32)
    parser.add_argument("--highlight-layer-start", type=int, default=9)
    parser.add_argument("--highlight-layer-end", type=int, default=16)
    parser.add_argument("--gate-strength", type=float, default=0.7)
    parser.add_argument("--gate-beta", type=float, default=10.0)
    parser.add_argument("--gate-tau", type=float, default=0.9)
    args = parser.parse_args()

    setup_style()
    os.makedirs(args.output_dir, exist_ok=True)
    data, records, score_key = load_ranked_heads(args.ranked_heads)
    add_component_ranks(records)

    outputs = {
        "phase1_text_mass": os.path.join(args.output_dir, "paper_phase1_text_mass.svg"),
        "phase1_contrast_bias": os.path.join(args.output_dir, "paper_phase1_contrast_bias.svg"),
        "phase1_layer_score_profile": os.path.join(args.output_dir, "paper_phase1_layer_score_profile.svg"),
        "phase1_head_heatmap": os.path.join(args.output_dir, "paper_phase1_head_heatmap.svg"),
        "phase1_rank_fusion": os.path.join(args.output_dir, "paper_phase1_rank_fusion.svg"),
        "phase2_gate_curve": os.path.join(args.output_dir, "paper_phase2_gate_curve.svg"),
        "phase2_delta_distribution": os.path.join(args.output_dir, "paper_phase2_delta_distribution.svg"),
        "phase2_redistribution": os.path.join(args.output_dir, "paper_phase2_redistribution.svg"),
        "phase2_delta_flow": os.path.join(args.output_dir, "paper_phase2_delta_flow.svg"),
    }

    fig_text_mass(outputs["phase1_text_mass"], records, args.top_k, args.tail_start)
    fig_contrast_bias(outputs["phase1_contrast_bias"], records, args.top_k)
    fig_layer_score_profile(
        outputs["phase1_layer_score_profile"],
        records,
        args.top_k,
        args.layer_profile_min,
        args.layer_profile_max,
        args.highlight_layer_start,
        args.highlight_layer_end,
    )
    fig_head_heatmap(
        outputs["phase1_head_heatmap"],
        records,
        args.top_k,
        args.heatmap_layer_min,
        args.heatmap_layer_max,
        args.head_count,
        args.highlight_layer_start,
        args.highlight_layer_end,
    )
    fig_rank_fusion(outputs["phase1_rank_fusion"], records, args.top_k)
    fig_gate_curve(outputs["phase2_gate_curve"], records, args.top_k, args.gate_strength, args.gate_beta, args.gate_tau)
    fig_delta_distribution(outputs["phase2_delta_distribution"], records, args.top_k, args.gate_strength, args.gate_beta, args.gate_tau)
    fig_redistribution(outputs["phase2_redistribution"], records, args.top_k, args.gate_strength, args.gate_beta, args.gate_tau)
    fig_delta_flow(outputs["phase2_delta_flow"], records, args.top_k, args.gate_strength, args.gate_beta, args.gate_tau)

    summary_rows = summarize(records, args.top_k, args.tail_start, args.gate_strength, args.gate_beta, args.gate_tau)
    summary_csv = os.path.join(args.output_dir, "paper_figure_summary.csv")
    write_csv(summary_csv, summary_rows)
    summary = {
        "ranked_heads": args.ranked_heads,
        "score_key": score_key,
        "source_layer_range": data.get("layer_range"),
        "top_k": args.top_k,
        "tail_start": args.tail_start,
        "highlight_layer_window": [args.highlight_layer_start, args.highlight_layer_end],
        "note": (
            "Paper heatmap intentionally shows L9-L12 as blank when the ranked-head source only covers L13-L31. "
            "Regenerate ranked heads over L9-L31 for a complete L9-L16 attribution heatmap."
        ),
        "figures": outputs,
        "summary_csv": summary_csv,
    }
    write_json(os.path.join(args.output_dir, "paper_figures_summary.json"), summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

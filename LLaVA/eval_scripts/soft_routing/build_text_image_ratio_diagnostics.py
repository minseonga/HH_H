#!/usr/bin/env python3
import argparse
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from eval_scripts.soft_routing.build_dual_ratio_detector_diagnostics import (
        COLORS,
        auc_score,
        compute_delta,
        group_token_rows,
        load_head_set,
        read_rows,
        roc_curve,
        setup_style,
        summarize_by_label,
        threshold_metrics,
        trigger_summary,
        write_csv,
    )
except ImportError:
    from build_dual_ratio_detector_diagnostics import (
        COLORS,
        auc_score,
        compute_delta,
        group_token_rows,
        load_head_set,
        read_rows,
        roc_curve,
        setup_style,
        summarize_by_label,
        threshold_metrics,
        trigger_summary,
        write_csv,
    )


def save(fig, output_dir, name, formats):
    paths = {}
    for fmt in formats:
        path = os.path.join(output_dir, f"{name}.{fmt}")
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        paths[fmt] = path
    plt.close(fig)
    return paths


def plot_detector(token_rows, output_dir, formats):
    setup_style()
    labels = [1 if row["label"] == "hallucinated" else 0 for row in token_rows]
    scores = [row["r_img"] for row in token_rows]
    auc = auc_score(scores, labels)
    curve = roc_curve(scores, labels)

    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.55), constrained_layout=True)

    bins = np.linspace(0.45, 1.0, 32)
    for label in ["grounded", "hallucinated"]:
        vals = [row["r_img"] for row in token_rows if row["label"] == label]
        color = COLORS[label]
        axes[0].hist(vals, bins=bins, density=True, histtype="stepfilled", alpha=0.22, color=color)
        axes[0].hist(vals, bins=bins, density=True, histtype="step", linewidth=1.45, color=color, label=label)
        if vals:
            axes[0].axvline(float(np.median(vals)), color=color, linestyle="--", linewidth=1.0)
    axes[0].axvline(0.9, color="#111827", linestyle=":", linewidth=1.0)
    axes[0].set_title(r"Object-token $T/(T+I)$", fontsize=9.6, fontweight="bold")
    axes[0].set_xlabel("token-level mean ratio")
    axes[0].set_ylabel("density")
    axes[0].grid(axis="y")
    axes[0].legend(frameon=False, fontsize=7.7)

    axes[1].plot([p[0] for p in curve], [p[1] for p in curve], color=COLORS["r_img"], linewidth=1.9)
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="#94a3b8", linewidth=1.0)
    axes[1].text(0.58, 0.12, f"AUC = {auc:.3f}", fontsize=10, color=COLORS["dark"], fontweight="bold")
    axes[1].set_title("Detector test", fontsize=9.6, fontweight="bold")
    axes[1].set_xlabel("grounded false-positive rate")
    axes[1].set_ylabel("hallucinated recall")
    axes[1].grid(True)

    fig.suptitle(r"$r = T/(T+I)$ is useful as a control signal, not a token detector", fontsize=10.6, fontweight="bold")
    return save(fig, output_dir, "text_image_ratio_detector_overlap", formats), auc


def plot_trigger_delta(rows, token_rows, tau, output_dir, formats):
    setup_style()
    head_trigger = {row["label"]: row for row in trigger_summary(rows, "r_img", tau)}
    token_trigger = {row["label"]: row for row in trigger_summary(token_rows, "r_img", tau)}

    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.55), constrained_layout=True)

    labels = ["grounded", "hallucinated"]
    x = np.arange(len(labels))
    head_vals = [head_trigger[label]["trigger_rate"] * 100 for label in labels]
    token_vals = [token_trigger[label]["trigger_rate"] * 100 for label in labels]
    width = 0.34
    axes[0].bar(x - width / 2, head_vals, width, color="#93c5fd", label="head-step")
    axes[0].bar(x + width / 2, token_vals, width, color="#bfdbfe", label="token mean")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["grounded", "hallucinated"])
    axes[0].set_ylabel(r"trigger rate at $\tau=0.9$ (%)")
    axes[0].set_title("Static threshold exposure", fontsize=9.6, fontweight="bold")
    axes[0].grid(axis="y")
    axes[0].legend(frameon=False, fontsize=7.7)
    for xpos, val in zip(x - width / 2, head_vals):
        axes[0].text(xpos, val + 1.0, f"{val:.1f}", ha="center", va="bottom", fontsize=7.2)
    for xpos, val in zip(x + width / 2, token_vals):
        axes[0].text(xpos, val + 1.0, f"{val:.1f}", ha="center", va="bottom", fontsize=7.2)

    bins = np.linspace(0, 1, 32)
    for label in labels:
        vals = [row["delta_img"] for row in token_rows if row["label"] == label]
        color = COLORS[label]
        axes[1].hist(vals, bins=bins, density=True, histtype="stepfilled", alpha=0.22, color=color)
        axes[1].hist(vals, bins=bins, density=True, histtype="step", linewidth=1.45, color=color, label=label)
        if vals:
            axes[1].axvline(float(np.median(vals)), color=color, linestyle="--", linewidth=1.0)
    axes[1].set_title(r"Dynamic $\delta$ from $T/(T+I)$", fontsize=9.6, fontweight="bold")
    axes[1].set_xlabel("token-level mean suppression")
    axes[1].set_ylabel("density")
    axes[1].grid(axis="y")
    axes[1].legend(frameon=False, fontsize=7.7)

    fig.suptitle("The same ratio also exposes grounded object steps", fontsize=10.6, fontweight="bold")
    return save(fig, output_dir, "text_image_ratio_trigger_delta", formats)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratio-csv", default="experiments_in_server/method_figure_source_trace_n100_k150_l9_16/selected_head_object_ratio_distribution.csv")
    parser.add_argument("--head-file", default="ADHH/LLaVA/results_summary/coco/ranked_heads_global__itext_all__C_toi_HminusG.json")
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--tau", type=float, default=0.9)
    parser.add_argument("--q", type=float, default=8.0)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--output-dir", default="LLaVA/results/coco/text_image_ratio_diagnostics_top100_l9_l16")
    parser.add_argument("--formats", default="svg,png,pdf")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    formats = [fmt.strip() for fmt in args.formats.split(",") if fmt.strip()]

    head_set, ordered_heads = load_head_set(args.head_file, args.topk)
    rows, seen_heads = read_rows(args.ratio_csv, head_set)
    compute_delta(rows, args.tau, args.q, args.strength)
    token_rows = group_token_rows(rows)
    labels = [1 if row["label"] == "hallucinated" else 0 for row in token_rows]

    detector_paths, auc = plot_detector(token_rows, args.output_dir, formats)
    trigger_delta_paths = plot_trigger_delta(rows, token_rows, args.tau, args.output_dir, formats)

    summary = {
        "ratio_definition": "r_img = text_mass / (text_mass + image_mass + eps)",
        "ratio_csv": args.ratio_csv,
        "head_file": args.head_file,
        "topk": args.topk,
        "tau": args.tau,
        "q": args.q,
        "strength": args.strength,
        "requested_heads": len(ordered_heads),
        "seen_requested_heads": len(seen_heads),
        "missing_requested_heads": [f"{l}:{h}" for l, h in ordered_heads if (l, h) not in seen_heads],
        "n_head_step_rows": len(rows),
        "n_token_rows": len(token_rows),
        "n_grounded_tokens": sum(1 for row in token_rows if row["label"] == "grounded"),
        "n_hallucinated_tokens": sum(1 for row in token_rows if row["label"] == "hallucinated"),
        "token_auc": auc,
        "token_summary": summarize_by_label(token_rows, "r_img"),
        "head_step_summary": summarize_by_label(rows, "r_img"),
        "token_threshold_metrics": threshold_metrics(token_rows, "r_img", args.tau),
        "head_step_trigger_summary": trigger_summary(rows, "r_img", args.tau),
        "token_trigger_summary": trigger_summary(token_rows, "r_img", args.tau),
        "token_delta_summary": summarize_by_label(token_rows, "delta_img"),
        "head_step_delta_summary": summarize_by_label(rows, "delta_img"),
        "figures": {
            "detector_overlap": detector_paths,
            "trigger_delta": trigger_delta_paths,
        },
        "caveat": (
            "This is computed from the available trace. If missing_requested_heads is non-empty, regenerate the exact top-k trace before using as final paper numbers."
        ),
    }

    write_csv(os.path.join(args.output_dir, "text_image_ratio_token_rows.csv"), token_rows)
    write_csv(
        os.path.join(args.output_dir, "text_image_ratio_summary_flat.csv"),
        [
            {"level": "token", "field": "r_img", **row}
            for row in summary["token_summary"]
        ]
        + [
            {"level": "head_step", "field": "r_img", **row}
            for row in summary["head_step_summary"]
        ]
        + [
            {"level": "token", "field": "delta_img", **row}
            for row in summary["token_delta_summary"]
        ]
        + [
            {"level": "head_step", "field": "delta_img", **row}
            for row in summary["head_step_delta_summary"]
        ],
    )
    with open(os.path.join(args.output_dir, "text_image_ratio_detector_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

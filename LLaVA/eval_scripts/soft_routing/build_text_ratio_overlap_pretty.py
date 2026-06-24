#!/usr/bin/env python3
"""Build a clean Section III-C text-reliance overlap figure without guide lines."""

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "grounded": "#159A5B",
    "hallucinated": "#D12F2F",
    "grounded_fill": "#159A5B",
    "hallucinated_fill": "#D12F2F",
    "dark": "#111827",
    "muted": "#667085",
    "grid": "#E7EAF0",
    "panel": "#F8FAFC",
}


def read_values(path):
    values = {"grounded": [], "hallucinated": []}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = row.get("label")
            if label not in values:
                continue
            raw = row.get("r_img", row.get("bounded_ratio"))
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if 0.0 <= value <= 1.0:
                values[label].append(value)
    return {key: np.asarray(vals, dtype=np.float64) for key, vals in values.items()}


def auc_score(scores, labels):
    pairs = sorted(zip(scores, labels), key=lambda item: item[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    rank_sum = 0.0
    pos = 0
    while pos < len(pairs):
        end = pos + 1
        while end < len(pairs) and pairs[end][0] == pairs[pos][0]:
            end += 1
        avg_rank = (pos + 1 + end) / 2.0
        if pairs[pos][1] == 1:
            rank_sum += avg_rank * sum(label for _, label in pairs[pos:end])
        else:
            rank_sum += avg_rank * sum(label for _, label in pairs[pos:end])
        pos = end
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def kde(values, grid, bandwidth=None):
    if values.size == 0:
        return np.zeros_like(grid)
    std = float(np.std(values))
    if bandwidth is None:
        bandwidth = max(0.018, 1.06 * std * values.size ** (-1 / 5))
    z = (grid[:, None] - values[None, :]) / bandwidth
    density = np.exp(-0.5 * z * z).sum(axis=1) / (values.size * bandwidth * np.sqrt(2 * np.pi))
    return density


def setup_style():
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 360,
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#CBD5E1",
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
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        paths[fmt] = str(path)
    plt.close(fig)
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--token-csv",
        default="LLaVA/results/coco/text_image_ratio_diagnostics_top100_l9_l16/text_image_ratio_token_rows.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="LLaVA/results/coco/text_image_ratio_diagnostics_top100_l9_l16/pretty",
    )
    parser.add_argument("--formats", default="svg,png,pdf")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    formats = [fmt.strip() for fmt in args.formats.split(",") if fmt.strip()]

    values = read_values(args.token_csv)
    grounded = values["grounded"]
    hall = values["hallucinated"]
    scores = np.concatenate([grounded, hall]).tolist()
    labels = ([0] * grounded.size) + ([1] * hall.size)
    auc = auc_score(scores, labels)

    grid = np.linspace(0.48, 1.0, 420)
    dg = kde(grounded, grid)
    dh = kde(hall, grid)
    ymax = max(float(dg.max()), float(dh.max())) * 1.18

    setup_style()
    fig, ax = plt.subplots(figsize=(4.45, 2.85), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.fill_between(grid, dg, color=COLORS["grounded_fill"], alpha=0.16, linewidth=0)
    ax.fill_between(grid, dh, color=COLORS["hallucinated_fill"], alpha=0.16, linewidth=0)
    ax.plot(grid, dg, color=COLORS["grounded"], linewidth=2.4, label=f"grounded  n={grounded.size}")
    ax.plot(grid, dh, color=COLORS["hallucinated"], linewidth=2.4, label=f"hallucinated  n={hall.size}")

    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0, ymax)
    ax.set_xlabel(r"text reliance $r=T/(T+I)$", fontsize=9.2)
    ax.set_ylabel("density", fontsize=9.2)
    ax.set_title("Object-token text reliance", fontsize=10.8, fontweight="bold", pad=7)
    ax.grid(axis="y")
    ax.tick_params(labelsize=8.4, length=3.2, color="#94A3B8")
    ax.legend(frameon=False, loc="upper left", fontsize=7.9, handlelength=2.0, borderaxespad=0.35)

    ax.text(
        0.972,
        ymax * 0.90,
        f"AUC = {auc:.3f}",
        ha="right",
        va="center",
        fontsize=8.7,
        fontweight="bold",
        color=COLORS["dark"],
        bbox=dict(boxstyle="round,pad=0.24", facecolor=COLORS["panel"], edgecolor="#D8DEE9", linewidth=0.8),
    )
    figures = save(fig, output_dir, "text_image_ratio_detector_overlap_pretty", formats)
    summary = {
        "token_csv": args.token_csv,
        "auc": auc,
        "grounded": {
            "n": int(grounded.size),
            "mean": float(np.mean(grounded)),
            "median": float(np.median(grounded)),
            "q75": float(np.quantile(grounded, 0.75)),
            "q90": float(np.quantile(grounded, 0.9)),
        },
        "hallucinated": {
            "n": int(hall.size),
            "mean": float(np.mean(hall)),
            "median": float(np.median(hall)),
            "q75": float(np.quantile(hall, 0.75)),
            "q90": float(np.quantile(hall, 0.9)),
        },
        "figures": figures,
    }
    with (output_dir / "text_image_ratio_detector_overlap_pretty_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

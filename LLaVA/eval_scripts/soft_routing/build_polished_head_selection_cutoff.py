#!/usr/bin/env python3
"""Build a polished single-panel visualization of fused head selection."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib import colors as mcolors


DARK = "#0f172a"
MUTED = "#64748b"
GRID = "#e2e8f0"
TEXT = "#38bdf8"
CONTRAST = "#f59e0b"
LINE = "#0f766e"
CUT = "#ef4444"
SELECT_BG = "#ecfeff"


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_bool(value: str | None) -> bool:
    return value in {"1", "true", "True", "yes", "Yes"}


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def bin_values(values: np.ndarray, n_bins: int) -> np.ndarray:
    chunks = np.array_split(values, n_bins)
    return np.array([chunk.mean() if len(chunk) else 0.0 for chunk in chunks], dtype=float)


def mix(c1: str, c2: str, t: float) -> str:
    a = np.array(mcolors.to_rgb(c1), dtype=float)
    b = np.array(mcolors.to_rgb(c2), dtype=float)
    return mcolors.to_hex((1.0 - t) * a + t * b)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--score-csv",
        default="experiments_in_server/method_figure_source_trace_n100_k150_l9_16/head_scores_all.csv",
    )
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--bins", type=int, default=24)
    parser.add_argument("--output-dir", default="LLaVA/results/coco/teaser_figure/head_selection")
    args = parser.parse_args()

    rows = load_rows(Path(args.score_csv))
    candidates = [r for r in rows if as_bool(r.get("selection_allowed"))]
    ranked = sorted(candidates, key=lambda r: (f(r, "score"), f(r, "text_percentile")), reverse=True)
    if args.top_k < 1 or args.top_k > len(ranked):
        raise SystemExit(f"--top-k must be in [1, {len(ranked)}]")

    contrast = np.array([f(r, "contrast_percentile") for r in ranked], dtype=float)
    text = np.array([f(r, "text_percentile") for r in ranked], dtype=float)
    fused = contrast + text

    n_bins = min(max(args.bins, 8), len(ranked))
    text_b = bin_values(text, n_bins)
    contrast_b = bin_values(contrast, n_bins)
    fused_b = text_b + contrast_b
    ranks = np.linspace(1, len(ranked), n_bins)
    cutoff_bin = int(np.searchsorted(ranks, args.top_k, side="left"))

    fig = plt.figure(figsize=(4.25, 3.05), dpi=280)
    fig.patch.set_facecolor("#f8fafc")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#ffffff")

    # Rounded white plotting card.
    card = FancyBboxPatch(
        (0, 0),
        1,
        1,
        transform=ax.transAxes,
        boxstyle="round,pad=0.012,rounding_size=0.04",
        linewidth=1.1,
        edgecolor="#dbe3ef",
        facecolor="#ffffff",
        zorder=-10,
        clip_on=False,
    )
    ax.add_patch(card)

    ax.set_xlim(-0.7, n_bins + 0.7)
    ax.set_ylim(-0.42, 2.64)
    ax.axis("off")

    # Top discrete fused-score blocks.
    x0 = np.arange(n_bins)
    max_fused = max(float(fused_b.max()), 1e-6)
    heights = 0.42 + 0.86 * (fused_b / max_fused)
    selected = x0 <= cutoff_bin
    for idx, (xv, height, is_sel, score) in enumerate(zip(x0, heights, selected, fused_b)):
        intensity = float(score / max_fused)
        color = mix("#ede9fe", "#6d28d9", intensity)
        alpha = 0.98 if is_sel else 0.46
        ax.add_patch(
            FancyBboxPatch(
                (xv - 0.39, 1.01),
                0.78,
                height,
                boxstyle="round,pad=0.003,rounding_size=0.035",
                facecolor=color,
                edgecolor="#ffffff",
                linewidth=0.55,
                alpha=alpha,
            )
        )

    # Two discrete component strips under the fused rank blocks.
    strip_y = [0.46, 0.14]
    strips = [(contrast_b, CONTRAST), (text_b, TEXT)]
    for y, (vals, cmap_color) in zip(strip_y, strips):
        for idx, value in enumerate(vals):
            alpha = 0.24 + 0.72 * float(value)
            ax.add_patch(
                Rectangle(
                    (idx - 0.39, y),
                    0.78,
                    0.22,
                    facecolor=cmap_color,
                    edgecolor="#ffffff",
                    linewidth=0.45,
                    alpha=alpha,
                )
            )

    # Selected region and cutoff marker.
    ax.add_patch(
        FancyBboxPatch(
            (-0.58, 0.04),
            cutoff_bin + 1.08,
            1.98,
            boxstyle="round,pad=0.018,rounding_size=0.10",
            facecolor="#ecfeff",
            edgecolor="#67e8f9",
            linewidth=1.15,
            alpha=0.36,
            zorder=-2,
        )
    )
    ax.axvline(cutoff_bin + 0.5, ymin=0.03, ymax=0.91, color=CUT, lw=2.0, alpha=0.92)
    ax.scatter([cutoff_bin + 0.5], [1.97], s=38, color=CUT, edgecolor="#ffffff", linewidth=1.0, zorder=5)
    ax.text(cutoff_bin + 0.85, 1.98, f"top-{args.top_k}", color="#991b1b", fontsize=8.1, weight="bold", ha="left", va="center")

    ax.text(-0.28, 2.48, "Fused head ranking", fontsize=10.8, weight="bold", color=DARK, ha="left", va="center")
    ax.text(-0.46, 0.91, "fused", fontsize=7.4, weight="bold", color="#6d28d9", ha="left", va="center")
    ax.text(-0.46, 0.57, "TOI", fontsize=7.0, weight="bold", color="#92400e", ha="left", va="center")
    ax.text(-0.46, 0.25, "text", fontsize=7.0, weight="bold", color="#0369a1", ha="left", va="center")
    ax.text(0.0, -0.18, "high", fontsize=6.8, color=MUTED, weight="bold", ha="left")
    ax.text(n_bins, -0.18, "low", fontsize=6.8, color=MUTED, weight="bold", ha="right")

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.subplots_adjust(left=0.02, right=0.99, bottom=0.03, top=0.97)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"head_selection_cutoff_discrete_top{args.top_k}"
    for ext in ["png", "svg", "pdf"]:
        path = stem.with_suffix(f".{ext}")
        fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()

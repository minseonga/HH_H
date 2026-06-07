#!/usr/bin/env python3
"""Build a compact rank-percentile fusion schematic from real head scores."""

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


PALETTE = ["#2563EB", "#10B981", "#EF4444", "#F59E0B", "#8B5CF6", "#06B6D4", "#F97316", "#64748B"]
DARK = "#0F172A"
MUTED = "#64748B"
GRID = "#E2E8F0"


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_bool(value: str | None) -> bool:
    return value in {"1", "true", "True", "yes", "Yes"}


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def draw_rank_panel(ax: plt.Axes, rows: list[dict[str, str]], value_key: str, title: str, color_map: dict[str, str]) -> None:
    vals = np.array([f(r, value_key) for r in rows], dtype=float)
    vals = vals / max(float(vals.max()), 1e-8)
    x = np.arange(len(rows))
    colors = [color_map[r["head_key"]] for r in rows]

    ax.bar(x, vals, color=colors, width=0.66, edgecolor="#FFFFFF", linewidth=0.75)
    ax.set_ylim(0, 1.08)
    ax.set_xlim(-0.55, len(rows) - 0.45)
    ax.set_title(title, fontsize=11.4, weight="bold", color=DARK, pad=5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("none")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.1)
        spine.set_edgecolor("#CBD5E1")
    ax.grid(axis="y", color=GRID, lw=0.8, alpha=0.55)
    ax.text(0.02, -0.13, "high", transform=ax.transAxes, ha="left", va="top", fontsize=7.4, color=MUTED, weight="bold")
    ax.text(0.98, -0.13, "low", transform=ax.transAxes, ha="right", va="top", fontsize=7.4, color=MUTED, weight="bold")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--score-csv",
        default="experiments_in_server/method_figure_source_trace_n100_k150_l9_16/head_scores_all.csv",
    )
    parser.add_argument("--top-n", type=int, default=6)
    parser.add_argument("--transparent", action="store_true")
    parser.add_argument("--output-dir", default="LLaVA/results/coco/teaser_figure/head_selection")
    parser.add_argument("--formats", default="png,svg,pdf")
    args = parser.parse_args()

    rows = [r for r in load_rows(Path(args.score_csv)) if as_bool(r.get("selection_allowed"))]
    text_rows = sorted(rows, key=lambda r: f(r, "text_percentile"), reverse=True)[: args.top_n]
    contrast_rows = sorted(rows, key=lambda r: f(r, "contrast_percentile"), reverse=True)[: args.top_n]

    # Use consistent colors for heads appearing in both lists; assign remaining colors by first appearance.
    ordered_keys: list[str] = []
    for r in text_rows + contrast_rows:
        if r["head_key"] not in ordered_keys:
            ordered_keys.append(r["head_key"])
    color_map = {key: PALETTE[idx % len(PALETTE)] for idx, key in enumerate(ordered_keys)}

    bg = "none" if args.transparent else "white"
    fig = plt.figure(figsize=(5.4, 2.05), dpi=260)
    fig.patch.set_facecolor(bg)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.26, 1.0], wspace=0.12)
    ax_text = fig.add_subplot(gs[0, 0])
    ax_plus = fig.add_subplot(gs[0, 1])
    ax_contrast = fig.add_subplot(gs[0, 2])

    draw_rank_panel(ax_text, text_rows, "text_percentile", r"$I_{\mathrm{text}}$", color_map)
    draw_rank_panel(ax_contrast, contrast_rows, "contrast_percentile", r"$C_{\mathrm{toi}}$", color_map)

    ax_plus.axis("off")
    ax_plus.text(0.5, 0.54, "+", ha="center", va="center", fontsize=30, color=DARK, weight="bold")

    fig.suptitle("Rank-Percentile Fusion", fontsize=14.2, weight="bold", color=DARK, y=1.08)
    fig.text(0.075, 0.50, "heads", rotation=90, ha="center", va="center", fontsize=11.0, weight="bold", color=DARK)
    fig.text(0.29, -0.02, r"$I_{\mathrm{text}}$ rank", ha="center", va="center", fontsize=9.5, color=DARK, weight="bold")
    fig.text(0.74, -0.02, r"$C_{\mathrm{toi}}$ rank", ha="center", va="center", fontsize=9.5, color=DARK, weight="bold")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"rank_percentile_fusion_schematic_top{args.top_n}"
    for fmt in [s.strip() for s in args.formats.split(",") if s.strip()]:
        path = stem.with_suffix(f".{fmt}")
        fig.savefig(path, bbox_inches="tight", facecolor=bg, transparent=args.transparent)
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()

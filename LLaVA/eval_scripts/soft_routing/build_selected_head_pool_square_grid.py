#!/usr/bin/env python3
"""Build a compact square grid of selected heads over fused score intensity."""

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
from matplotlib.colors import LinearSegmentedColormap


DARK = "#0F172A"
MUTED = "#64748B"
SELECTED = "#F59E0B"


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_bool(value: str | None) -> bool:
    return value in {"1", "true", "True", "yes", "Yes"}


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--score-csv",
        default="experiments_in_server/method_figure_source_trace_n100_k150_l9_16/head_scores_all.csv",
    )
    parser.add_argument("--layers", default="9-16")
    parser.add_argument("--selected-top-k", type=int, default=100)
    parser.add_argument("--flat-square", action="store_true")
    parser.add_argument("--transparent", action="store_true")
    parser.add_argument("--hide-labels", action="store_true")
    parser.add_argument("--output-dir", default="LLaVA/results/coco/teaser_figure/head_selection")
    parser.add_argument("--formats", default="png,svg,pdf")
    args = parser.parse_args()

    start, end = [int(x) for x in args.layers.split("-", 1)]
    layers = list(range(start, end + 1))
    n_layers = len(layers)
    n_heads = 32

    rows = load_rows(Path(args.score_csv))
    rows_by_key = {(int(r["layer"]), int(r["head"])): r for r in rows}
    allowed = [r for r in rows if as_bool(r.get("selection_allowed"))]
    ranked = sorted(allowed, key=lambda r: (f(r, "score"), f(r, "text_percentile")), reverse=True)
    selected_keys = {(int(r["layer"]), int(r["head"])) for r in ranked[: args.selected_top_k]}

    score = np.full((n_layers, n_heads), np.nan, dtype=float)
    selected = np.zeros((n_layers, n_heads), dtype=bool)
    for li, layer in enumerate(layers):
        for head in range(n_heads):
            row = rows_by_key.get((layer, head))
            if row is None:
                continue
            score[li, head] = f(row, "score")
            selected[li, head] = (layer, head) in selected_keys

    valid = np.isfinite(score)
    score_norm = np.zeros_like(score)
    if valid.any():
        mn = float(np.nanmin(score))
        mx = float(np.nanmax(score))
        score_norm[valid] = (score[valid] - mn) / max(mx - mn, 1e-8)

    cmap = LinearSegmentedColormap.from_list("score_blue", ["#EAF2FB", "#8BBCE6", "#315AA6", "#1E2F67"])
    rgba = cmap(score_norm)
    rgba[..., 3] = np.where(valid, 1.0, 0.0)
    rgba[selected] = matplotlib.colors.to_rgba(SELECTED)

    if args.flat_square:
        side = int(np.ceil(np.sqrt(n_layers * n_heads)))
        flat_rgba = np.zeros((side * side, 4), dtype=float)
        flat_rgba[:, 3] = 0.0
        flat_rgba[: n_layers * n_heads] = rgba.reshape(-1, 4)
        rgba = flat_rgba.reshape(side, side, 4)
        grid_rows, grid_cols = side, side
    else:
        grid_rows, grid_cols = n_layers, n_heads

    bg = "none" if args.transparent else "white"
    fig, ax = plt.subplots(figsize=(3.05, 3.05), dpi=300)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.imshow(rgba, aspect="equal", interpolation="nearest")

    # Cell borders.
    ax.set_xticks(np.arange(-0.5, grid_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_rows, 1), minor=True)
    ax.grid(which="minor", color="#172033", linewidth=0.42, alpha=0.78)
    ax.tick_params(which="minor", bottom=False, left=False)

    if args.hide_labels:
        ax.set_xticks([])
        ax.set_yticks([])
    elif args.flat_square:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.set_xticks([0, 7, 15, 23, 31])
        ax.set_xticklabels(["0", "7", "15", "23", "31"], fontsize=6.5, color=MUTED)
        ax.set_yticks(np.arange(n_layers))
        ax.set_yticklabels([f"L{l}" for l in layers], fontsize=6.5, color=MUTED)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_edgecolor("#172033")

    ax.set_title("Selected Head Pool", fontsize=10.2, weight="bold", color=DARK, pad=5)
    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.04, top=0.90)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "flat16" if args.flat_square else "lattice"
    stem = out_dir / f"selected_head_pool_square_grid_{suffix}_l{start}_l{end}_k{args.selected_top_k}"
    for fmt in [s.strip() for s in args.formats.split(",") if s.strip()]:
        path = stem.with_suffix(f".{fmt}")
        fig.savefig(path, bbox_inches="tight", facecolor=bg, transparent=args.transparent)
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()

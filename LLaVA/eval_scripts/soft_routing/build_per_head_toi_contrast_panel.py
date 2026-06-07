#!/usr/bin/env python3
"""Plot per-head hall/ground text-over-image contrast."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "grounded": "#159A5B",
    "hallucinated": "#D12F2F",
    "connector": "#94A3B8",
    "selected": "#7C3AED",
    "dark": "#0F172A",
    "muted": "#64748B",
    "grid": "#E7EAF0",
}


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_bool(value: str | None) -> bool:
    return value in {"1", "true", "True", "yes", "Yes"}


def get_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def setup_style() -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--score-csv",
        default="experiments_in_server/method_figure_source_trace_n100_k150_l9_16/head_scores_all.csv",
    )
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument("--style", choices=["points", "paired_gap", "gap_bar", "overlap_bar", "histogram"], default="points")
    parser.add_argument("--bins", type=int, default=16)
    parser.add_argument("--transparent", action="store_true")
    parser.add_argument("--output-dir", default="LLaVA/results/coco/teaser_figure/contrastive_toi")
    parser.add_argument("--formats", default="png,svg,pdf")
    args = parser.parse_args()

    rows = load_rows(Path(args.score_csv))
    rows = [r for r in rows if as_bool(r.get("selection_allowed"))]
    if args.selection_only:
        rows = [r for r in rows if as_bool(r.get("selected"))]

    if args.style == "gap_bar":
        ranked = sorted(
            rows,
            key=lambda r: np.log1p(get_float(r, "raw_toi_hallucinated")) - np.log1p(get_float(r, "raw_toi_grounded")),
            reverse=True,
        )
    else:
        ranked = sorted(rows, key=lambda r: get_float(r, "raw_toi_gap_hall_minus_grounded"), reverse=True)
    selected = ranked[: args.top_n]
    if not selected:
        raise SystemExit("no rows selected")

    labels = [r["head_key"] for r in selected]
    raw_ground = np.array([get_float(r, "raw_toi_grounded") for r in selected], dtype=float)
    raw_hall = np.array([get_float(r, "raw_toi_hallucinated") for r in selected], dtype=float)
    y_ground = np.log1p(raw_ground)
    y_hall = np.log1p(raw_hall)
    gaps = y_hall - y_ground
    x = np.arange(len(selected))

    setup_style()
    figsize = (4.25, 3.2) if args.style == "histogram" else (6.2, 3.15)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    bg = "none" if args.transparent else "white"
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    if args.style == "histogram":
        lo = float(min(y_ground.min(), y_hall.min()))
        hi = float(max(y_ground.max(), y_hall.max()))
        bins = np.linspace(lo - 0.08, hi + 0.08, args.bins)
        ax.hist(
            y_hall,
            bins=bins,
            density=True,
            color=COLORS["hallucinated"],
            alpha=0.54,
            edgecolor="#991B1B",
            linewidth=0.8,
            label="hallucinated",
            zorder=3,
        )
        ax.hist(
            y_ground,
            bins=bins,
            density=True,
            color=COLORS["grounded"],
            alpha=0.46,
            edgecolor="#065F46",
            linewidth=0.8,
            label="grounded",
            zorder=2,
        )
    elif args.style == "overlap_bar":
        ax.bar(
            x,
            y_hall,
            width=0.72,
            color=COLORS["hallucinated"],
            alpha=0.78,
            edgecolor="white",
            linewidth=0.7,
            label="hallucinated object",
            zorder=2,
        )
        ax.bar(
            x,
            y_ground,
            width=0.44,
            color=COLORS["grounded"],
            alpha=0.86,
            edgecolor="white",
            linewidth=0.7,
            label="grounded object",
            zorder=3,
        )
    elif args.style == "paired_gap":
        for idx in range(len(selected)):
            ax.plot([idx, idx], [y_ground[idx], y_hall[idx]], color="#CBD5E1", lw=3.0, alpha=0.85, zorder=1)
            ax.plot([idx, idx], [y_ground[idx], y_hall[idx]], color="#EF4444", lw=1.45, alpha=0.58, zorder=2)

        ax.scatter(x, y_ground, s=42, color=COLORS["grounded"], edgecolor="white", linewidth=0.9, label="grounded", zorder=4)
        ax.scatter(x, y_hall, s=46, color=COLORS["hallucinated"], edgecolor="white", linewidth=0.9, label="hallucinated", zorder=5)
    elif args.style == "gap_bar":
        ax.axhline(0, color="#94A3B8", lw=1.0, alpha=0.85, zorder=1)
        ax.bar(
            x,
            gaps,
            width=0.62,
            color="#7C3AED",
            alpha=0.9,
            edgecolor="#4C1D95",
            linewidth=0.7,
            zorder=3,
        )
    else:
        for idx in range(len(selected)):
            ax.plot([idx, idx], [y_ground[idx], y_hall[idx]], color=COLORS["connector"], lw=1.2, alpha=0.65, zorder=1)

        ax.scatter(x, y_ground, s=34, color=COLORS["grounded"], edgecolor="white", linewidth=0.8, label="grounded object", zorder=3)
        ax.scatter(x, y_hall, s=38, color=COLORS["hallucinated"], edgecolor="white", linewidth=0.8, label="hallucinated object", zorder=4)

        # Subtle gap fill makes the hall-minus-ground shift visible without adding text.
        ax.fill_between(x, y_ground, y_hall, where=y_hall >= y_ground, color="#FEE2E2", alpha=0.28, step=None, zorder=0)

    if args.style == "histogram":
        title = "Contrastive Text-over-Image Bias"
    elif args.style == "overlap_bar":
        title = "Hall vs. Ground TOI per Head"
    elif args.style == "gap_bar":
        title = "Head-wise Hall-Ground TOI Gap"
    else:
        title = "Head-wise Hall-Ground TOI Gap"
    ax.set_title(title, fontsize=12.6, weight="bold", color=COLORS["dark"], pad=8)
    ylabel = r"$\Delta \log(1 + T/I)$" if args.style == "gap_bar" else r"$\log(1 + T/I)$"
    ax.set_ylabel(ylabel, fontsize=10.4, color=COLORS["dark"])
    if args.style == "histogram":
        ax.set_xlabel(r"per-head TOI  $\log(1+T/I)$", fontsize=10.4, color=COLORS["dark"])
        ax.set_ylabel("density", fontsize=10.4, color=COLORS["dark"])
    else:
        xlabel = "heads sorted by contrastive TOI score" if args.style == "overlap_bar" else "heads sorted by hall-ground TOI gap"
        ax.set_xlabel(xlabel, fontsize=10.4, color=COLORS["dark"])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7.7, color=COLORS["dark"])
    ax.tick_params(axis="y", labelsize=8.8, colors=COLORS["dark"])
    ax.grid(axis="y")
    if args.style != "gap_bar":
        ax.legend(frameon=False, loc="upper left", ncols=2, fontsize=8.6, handletextpad=0.4, columnspacing=1.0)

    if args.style != "overlap_bar":
        gap_text = f"mean gap = {float(np.mean(gaps)):.2f}"
        ax.text(
            0.99,
            0.92,
            gap_text,
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=8.7,
            weight="bold",
            color=COLORS["dark"],
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#F8FAFC", edgecolor="#D8DEE9", linewidth=0.8),
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = {"overlap_bar": "bars", "histogram": "hist", "paired_gap": "paired_gap", "gap_bar": "gap_bar", "points": "points"}[args.style]
    stem = out_dir / f"per_head_toi_hall_ground_{suffix}_top{args.top_n}"
    outputs: dict[str, str] = {}
    for fmt in [s.strip() for s in args.formats.split(",") if s.strip()]:
        path = stem.with_suffix(f".{fmt}")
        fig.savefig(path, bbox_inches="tight", facecolor=bg, transparent=args.transparent)
        outputs[fmt] = str(path)
        print(path)
    plt.close(fig)

    summary = {
        "score_csv": args.score_csv,
        "top_n": args.top_n,
        "selection_only": args.selection_only,
        "style": args.style,
        "bins": args.bins,
        "transparent": args.transparent,
        "mean_log_gap": float(np.mean(gaps)),
        "heads": [
            {
                "head_key": r["head_key"],
                "raw_toi_grounded": get_float(r, "raw_toi_grounded"),
                "raw_toi_hallucinated": get_float(r, "raw_toi_hallucinated"),
                "raw_toi_gap_hall_minus_grounded": get_float(r, "raw_toi_gap_hall_minus_grounded"),
                "log_gap": float(gap),
                "selected": as_bool(r.get("selected")),
            }
            for r, gap in zip(selected, gaps)
        ],
        "outputs": outputs,
    }
    with (out_dir / f"per_head_toi_hall_ground_{suffix}_top{args.top_n}_summary.json").open("w", encoding="utf-8") as out_file:
        json.dump(summary, out_file, indent=2)


if __name__ == "__main__":
    main()

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
    parser.add_argument(
        "--style",
        choices=["points", "paired_gap", "gap_bar", "overlap_bar", "contrastive_schematic", "histogram"],
        default="points",
    )
    parser.add_argument("--bins", type=int, default=16)
    parser.add_argument("--transparent", action="store_true")
    parser.add_argument("--hide-legend", action="store_true")
    parser.add_argument("--hide-head-labels", action="store_true")
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
    if args.style == "histogram":
        figsize = (4.25, 3.2)
    elif args.style == "contrastive_schematic":
        figsize = (4.65, 3.15)
    else:
        figsize = (6.2, 3.15)
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
    elif args.style == "contrastive_schematic":
        hall_width = 0.74
        ground_width = 0.48
        ax.bar(
            x,
            y_hall,
            width=hall_width,
            color="#FCA5A5",
            alpha=0.88,
            edgecolor="#111827",
            linewidth=1.1,
            label="Hallucinated",
            zorder=2,
        )
        ax.bar(
            x,
            y_ground,
            width=ground_width,
            color="#8FD14F",
            alpha=0.96,
            edgecolor="#111827",
            linewidth=1.1,
            label="Grounded",
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
    elif args.style == "contrastive_schematic":
        title = r"Contrastive TOI Bias $(C_{\mathrm{toi}})$"
    elif args.style == "overlap_bar":
        title = "Hall vs. Ground TOI per Head"
    elif args.style == "gap_bar":
        title = "Head-wise Hall-Ground TOI Gap"
    else:
        title = "Head-wise Hall-Ground TOI Gap"
    title_size = 13.8 if args.style == "contrastive_schematic" else 12.6
    ax.set_title(title, fontsize=title_size, weight="bold", color=COLORS["dark"], pad=8)
    ylabel = r"$\Delta \log(1 + T/I)$" if args.style == "gap_bar" else r"$\log(1 + T/I)$"
    ax.set_ylabel(ylabel, fontsize=10.4, color=COLORS["dark"])
    if args.style == "histogram":
        ax.set_xlabel(r"per-head TOI  $\log(1+T/I)$", fontsize=10.4, color=COLORS["dark"])
        ax.set_ylabel("density", fontsize=10.4, color=COLORS["dark"])
    elif args.style == "contrastive_schematic":
        ax.set_xlabel("Heads", fontsize=13.6, color=COLORS["dark"], labelpad=7, weight="bold")
        ax.set_ylabel("TOI", fontsize=13.6, color=COLORS["dark"], labelpad=7, weight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ["left", "bottom"]:
            ax.spines[side].set_visible(True)
            ax.spines[side].set_color("#111827")
            ax.spines[side].set_linewidth(3.0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)
    else:
        xlabel = "heads sorted by contrastive TOI score" if args.style == "overlap_bar" else "heads sorted by hall-ground TOI gap"
        ax.set_xlabel(xlabel, fontsize=10.4, color=COLORS["dark"])
        ax.set_xticks(x)
        if args.hide_head_labels:
            ax.set_xticklabels([str(i + 1) for i in x], fontsize=8.2, color=COLORS["dark"])
        else:
            ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7.7, color=COLORS["dark"])
    if args.style != "contrastive_schematic":
        ax.tick_params(axis="y", labelsize=8.8, colors=COLORS["dark"])
        ax.grid(axis="y")
    if args.style != "gap_bar" and not args.hide_legend:
        if args.style == "contrastive_schematic":
            handles, labels_ = ax.get_legend_handles_labels()
            order = [labels_.index("Grounded"), labels_.index("Hallucinated")]
            ax.legend(
                [handles[i] for i in order],
                [labels_[i] for i in order],
                frameon=False,
                loc="upper center",
                bbox_to_anchor=(0.72, 1.02),
                fontsize=8.8,
                handlelength=1.25,
                handletextpad=0.35,
                labelspacing=0.25,
                borderpad=0.0,
            )
        else:
            ax.legend(frameon=False, loc="upper left", ncols=2, fontsize=8.6, handletextpad=0.4, columnspacing=1.0)

    if args.style not in {"overlap_bar", "contrastive_schematic"}:
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
    suffix = {
        "overlap_bar": "bars",
        "contrastive_schematic": "schematic",
        "histogram": "hist",
        "paired_gap": "paired_gap",
        "gap_bar": "gap_bar",
        "points": "points",
    }[args.style]
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
        "hide_legend": args.hide_legend,
        "hide_head_labels": args.hide_head_labels,
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

#!/usr/bin/env python3
import argparse
import csv
import json
import os
from statistics import median

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "grounded": "#16a34a",
    "hallucinated": "#dc2626",
    "object": "#475569",
    "all": "#94a3b8",
    "grid": "#e5e7eb",
    "dark": "#111827",
    "muted": "#64748b",
}


TABLE_STATS = {
    "All tokens": {"mean": 0.818, "q25": 0.701, "q50": 0.856, "q75": 0.938, "q90": 0.955},
    "Object tokens": {"mean": 0.832, "q25": 0.707, "q50": 0.882, "q75": 0.958, "q90": 0.973},
    "Grounded objects": {"mean": 0.822, "q25": 0.697, "q50": 0.878, "q75": 0.956, "q90": 0.971},
    "Hallucinated objects": {"mean": 0.883, "q25": 0.820, "q50": 0.916, "q75": 0.966, "q90": 0.978},
}


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_ratio_rows(path):
    values = {"grounded": [], "hallucinated": []}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row.get("label")
            if label not in values:
                continue
            value = safe_float(row.get("bounded_ratio"))
            if value is None:
                continue
            if 0.0 <= value <= 1.0:
                values[label].append(value)
    return values


def quantile(vals, q):
    if not vals:
        return None
    arr = np.asarray(vals, dtype=float)
    return float(np.quantile(arr, q))


def summarize_values(values):
    rows = []
    for label, vals in values.items():
        rows.append(
            {
                "label": label,
                "n": len(vals),
                "mean": float(np.mean(vals)) if vals else None,
                "q25": quantile(vals, 0.25),
                "q50": float(median(vals)) if vals else None,
                "q75": quantile(vals, 0.75),
                "q90": quantile(vals, 0.90),
            }
        )
    return rows


def setup_style():
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 320,
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#cbd5e1",
            "axes.linewidth": 0.9,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_distribution(ax, values):
    bins = np.linspace(0.45, 1.0, 38)
    label_y = {"grounded": 5.75, "hallucinated": 5.35}
    label_dx = {"grounded": -0.025, "hallucinated": 0.025}
    for label in ["grounded", "hallucinated"]:
        vals = values[label]
        color = COLORS[label]
        ax.hist(vals, bins=bins, density=True, histtype="stepfilled", alpha=0.22, color=color)
        ax.hist(vals, bins=bins, density=True, histtype="step", linewidth=1.5, color=color, label=label)
        if vals:
            med = median(vals)
            ax.axvline(med, color=color, linewidth=1.2, linestyle="--", alpha=0.9)
            ax.text(
                med + label_dx[label],
                label_y[label],
                f"{med:.3f}",
                color=color,
                ha="center",
                va="top",
                fontsize=7.7,
                fontweight="bold",
            )
    ax.set_xlim(0.45, 1.0)
    ax.set_xlabel(r"text-reliance ratio $r=T/(T+I)$")
    ax.set_ylabel("density")
    ax.set_title("Object-step ratio distributions overlap", fontsize=9.7, fontweight="bold")
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="upper left", fontsize=8)


def plot_quantiles(ax):
    labels = ["All tokens", "Object tokens", "Grounded objects", "Hallucinated objects"]
    y = np.arange(len(labels))
    color_by = {
        "All tokens": COLORS["all"],
        "Object tokens": COLORS["object"],
        "Grounded objects": COLORS["grounded"],
        "Hallucinated objects": COLORS["hallucinated"],
    }
    for idx, label in enumerate(labels):
        row = TABLE_STATS[label]
        color = color_by[label]
        ax.plot([row["q25"], row["q75"]], [idx, idx], color=color, linewidth=7, solid_capstyle="round", alpha=0.38)
        ax.plot([row["q50"]], [idx], marker="o", markersize=5.5, color=color)
        ax.plot([row["q90"]], [idx], marker="|", markersize=12, color=color, markeredgewidth=2)
        offset = -0.015 if label == "Hallucinated objects" else 0.014
        ax.text(
            row["q50"] + offset,
            idx + 0.20,
            f"{row['q50']:.3f}",
            color=color,
            ha="center",
            va="bottom",
            fontsize=7.5,
            fontweight="bold" if label in {"Grounded objects", "Hallucinated objects"} else "400",
        )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0.65, 1.0)
    ax.set_xlabel(r"$r=T/(T+I)$")
    ax.set_title("Hallucinated shifts right, but ranges overlap", fontsize=9.7, fontweight="bold")
    ax.grid(axis="x")
    ax.text(
        0.5,
        -0.25,
        "line: Q25-Q75, dot: median, tick: Q90",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.2,
        color=COLORS["muted"],
    )


def save(fig, output_dir, name, formats):
    paths = {}
    for fmt in formats:
        path = os.path.join(output_dir, f"{name}.{fmt}")
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        paths[fmt] = path
    plt.close(fig)
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ratio-csv",
        default="experiments_in_server/method_figure_source_trace_n100_k150_l9_16/selected_head_object_ratio_distribution.csv",
    )
    parser.add_argument("--output-dir", default="LLaVA/results/coco/section_iii_c_text_reliance_detector_overlap")
    parser.add_argument("--formats", default="svg,png,pdf")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    formats = [item.strip() for item in args.formats.split(",") if item.strip()]
    values = read_ratio_rows(args.ratio_csv)
    summary = {
        "ratio_csv": args.ratio_csv,
        "row_level_summary": summarize_values(values),
        "table_1_stats": TABLE_STATS,
        "interpretation": (
            "Hallucinated object steps shift toward higher text reliance, but the grounded and "
            "hallucinated distributions overlap substantially. Text reliance alone is therefore "
            "not a reliable token-level hallucination detector."
        ),
    }

    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), constrained_layout=True)
    plot_distribution(axes[0], values)
    plot_quantiles(axes[1])
    fig.suptitle("Text-side reliance alone is not a hallucination detector", fontsize=10.8, fontweight="bold")
    figures = save(fig, args.output_dir, "section_iii_c_text_reliance_overlap", formats)

    with open(os.path.join(args.output_dir, "section_iii_c_text_reliance_overlap_summary.json"), "w", encoding="utf-8") as f:
        json.dump({**summary, "figures": figures}, f, indent=2)
    print(json.dumps({**summary, "figures": figures}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "greedy": "#64748b",
    "text": "#f97316",
    "contrast": "#2563eb",
    "fused": "#7c3aed",
    "grid": "#e2e8f0",
    "dark": "#0f172a",
    "muted": "#64748b",
}


METHOD_SPECS = [
    {
        "key": "greedy",
        "label": "greedy",
        "family": "greedy",
        "match": lambda name: name == "greedy",
    },
    {
        "key": "itext_hard_k150",
        "label": "text-only\nhard",
        "family": "text",
        "match": lambda name: name == "itext_hard_k150",
    },
    {
        "key": "contrast_continuous_k150",
        "label": "contrast-only\ncontinuous",
        "family": "contrast",
        "match": lambda name: name.startswith("contrast_continuous_k150"),
    },
    {
        "key": "combined_hard_k150",
        "label": "fused\nhard",
        "family": "fused",
        "match": lambda name: name == "combined_hard_k150",
    },
    {
        "key": "combined_continuous_k150",
        "label": "fused\ncontinuous",
        "family": "fused",
        "match": lambda name: name.startswith("combined_continuous_k150"),
    },
]


def setup_style():
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 320,
            "font.family": "DejaVu Sans",
            "font.size": 7.2,
            "axes.titlesize": 8.6,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 6.2,
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


def safe_float(value, default=0.0):
    try:
        value = float(value)
    except Exception:
        return default
    return value if math.isfinite(value) else default


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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
        json.dump(data, f, indent=2, ensure_ascii=False)


def save(fig, output_dir, name, formats):
    paths = {}
    os.makedirs(output_dir, exist_ok=True)
    for fmt in formats:
        path = os.path.join(output_dir, f"{name}.{fmt}")
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        paths[fmt] = path
    plt.close(fig)
    return paths


def select_methods(rows):
    by_name = {row.get("method", ""): row for row in rows}
    selected = []
    for spec in METHOD_SPECS:
        matches = [row for name, row in by_name.items() if spec["match"](name)]
        if not matches:
            continue
        row = sorted(matches, key=lambda item: item.get("method", ""))[0]
        item = dict(row)
        item["plot_key"] = spec["key"]
        item["plot_label"] = spec["label"]
        item["plot_family"] = spec["family"]
        selected.append(item)
    return selected


def metric(row, key):
    return safe_float(row.get(key))


def plot_outcome_panel(rows, output_dir, formats):
    selected = select_methods(rows)
    if len(selected) < 2:
        raise ValueError("Need at least two recognized methods in summary CSV")

    labels = [row["plot_label"] for row in selected]
    colors = [COLORS[row["plot_family"]] for row in selected]
    chairs = np.array([metric(row, "CHAIRs") for row in selected], dtype=float)
    bleus = np.array([metric(row, "Bleu1") for row in selected], dtype=float)
    chairi = np.array([metric(row, "CHAIRi") for row in selected], dtype=float)
    lengths = np.array([metric(row, "avg_caption_length") for row in selected], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(9.3, 2.75), gridspec_kw={"width_ratios": [1.25, 1.0, 1.0]})

    ax = axes[0]
    for row, x, y, length, color in zip(selected, bleus, chairs, lengths, colors):
        size = 44 + max(length - np.nanmin(lengths), 0.0) * 3.2 if len(lengths) else 58
        ax.scatter([x], [y], s=size, color=color, alpha=0.86, edgecolors="white", linewidths=0.7)
        ax.annotate(row["plot_label"].replace("\n", " "), (x, y), xytext=(4, 4), textcoords="offset points", fontsize=6.0, color=COLORS["dark"])
    ax.set_xlabel("BLEU-1 ↑")
    ax.set_ylabel("CHAIRs ↓")
    ax.set_title("A. behavioral trade-off")
    ax.grid(True)
    ax.text(0.03, 0.97, "not a feature-value plot:\nactual decoding outcome", transform=ax.transAxes, ha="left", va="top", fontsize=6.2, color=COLORS["muted"])

    x = np.arange(len(selected))
    ax = axes[1]
    ax.bar(x, chairs, color=colors, alpha=0.86)
    ax.set_title("B. hallucination")
    ax.set_ylabel("CHAIRs ↓")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.grid(axis="y")
    for idx, value in enumerate(chairs):
        ax.text(idx, value + max(chairs) * 0.025, f"{value:.3f}", ha="center", va="bottom", fontsize=5.8)

    ax = axes[2]
    ax.bar(x, bleus, color=colors, alpha=0.86)
    ax.set_title("C. caption quality")
    ax.set_ylabel("BLEU-1 ↑")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.grid(axis="y")
    for idx, value in enumerate(bleus):
        ax.text(idx, value + max(bleus) * 0.02, f"{value:.3f}", ha="center", va="bottom", fontsize=5.8)

    fig.suptitle("Feature axes are validated by held-out decoding behavior", y=1.02, fontsize=10.2, weight="bold")
    fig.subplots_adjust(top=0.78, bottom=0.34, left=0.065, right=0.995, wspace=0.34)
    paths = save(fig, output_dir, "method_axis_outcome_panel", formats)

    selected_rows = []
    for row in selected:
        selected_rows.append(
            {
                "method": row.get("method"),
                "plot_label": row["plot_label"].replace("\n", " "),
                "plot_family": row["plot_family"],
                "CHAIRs": row.get("CHAIRs"),
                "CHAIRi": row.get("CHAIRi"),
                "Bleu1": row.get("Bleu1"),
                "avg_caption_length": row.get("avg_caption_length"),
            }
        )
    selected_csv = os.path.join(output_dir, "method_axis_outcome_selected.csv")
    write_csv(selected_csv, selected_rows)
    return paths, selected_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--formats", default="png,pdf,svg")
    args = parser.parse_args()

    setup_style()
    formats = [fmt.strip().lstrip(".") for fmt in args.formats.split(",") if fmt.strip()]
    rows = read_csv(args.summary_csv)
    paths, selected_csv = plot_outcome_panel(rows, args.output_dir, formats)
    manifest = {
        "summary_csv": os.path.abspath(args.summary_csv),
        "output_dir": os.path.abspath(args.output_dir),
        "selected_csv": selected_csv,
        "figures": {"method_axis_outcome_panel": paths},
    }
    manifest_path = os.path.join(args.output_dir, "method_axis_outcome_manifest.json")
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

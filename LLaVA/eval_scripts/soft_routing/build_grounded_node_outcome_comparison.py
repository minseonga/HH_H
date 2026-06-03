#!/usr/bin/env python3
import argparse
import csv
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {
    "dark": "#111827",
    "muted": "#64748b",
    "grid": "#e5e7eb",
    "preserve": "#cbd5e1",
    "partial": "#f59e0b",
    "disappear": "#ef4444",
}


def load_summary(path, label):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    n = int(data["base_ground_nodes"])
    partial = int(data["partial_loss_ground_nodes"])
    disappeared = int(data["disappeared_ground_nodes"])
    reduced = int(data["reduced_ground_nodes"])
    preserved = max(0, n - reduced)
    return {
        "label": label,
        "path": path,
        "base_ground_nodes": n,
        "preserved_nodes": preserved,
        "partial_loss_ground_nodes": partial,
        "disappeared_ground_nodes": disappeared,
        "reduced_ground_nodes": reduced,
        "preserved_rate": preserved / n if n else 0.0,
        "partial_rate": partial / n if n else 0.0,
        "disappeared_rate": disappeared / n if n else 0.0,
        "reduced_rate": reduced / n if n else 0.0,
        "target_chairs": (data.get("target_metrics") or {}).get("CHAIRs"),
        "target_chairi": (data.get("target_metrics") or {}).get("CHAIRi"),
        "hallucinated_removal_rate": data.get("hallucinated_removal_rate"),
    }


def write_csv(path, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


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
            "grid.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig, output_dir, name, formats):
    paths = {}
    for fmt in formats:
        path = os.path.join(output_dir, f"{name}.{fmt}")
        fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
        paths[fmt] = path
    plt.close(fig)
    return paths


def plot(rows, output_dir, formats):
    setup_style()
    fig, ax = plt.subplots(figsize=(4.4, 2.7), constrained_layout=True)
    y_positions = list(range(len(rows)))
    for y, row in zip(y_positions, rows):
        left = 0.0
        segments = [
            ("preserved_rate", COLORS["preserve"], "preserved"),
            ("partial_rate", COLORS["partial"], "partial"),
            ("disappeared_rate", COLORS["disappear"], "disappeared"),
        ]
        for key, color, _ in segments:
            width = row[key] * 100.0
            ax.barh(y, width, left=left, height=0.44, color=color, alpha=0.9)
            if key != "preserved_rate":
                ax.text(left + width / 2, y, f"{width:.1f}%", ha="center", va="center", fontsize=7.2, fontweight="bold")
            left += width
        ax.text(
            row["preserved_rate"] * 50.0,
            y,
            f"{row['preserved_rate'] * 100:.1f}% preserved",
            ha="center",
            va="center",
            fontsize=7.6,
            color=COLORS["dark"],
            fontweight="bold",
        )
        ax.text(
            103,
            y,
            f"reduced {row['reduced_rate'] * 100:.1f}%",
            ha="left",
            va="center",
            fontsize=7.6,
            color=COLORS["muted"],
        )
    ax.set_yticks(y_positions)
    ax.set_yticklabels([row["label"] for row in rows], fontsize=8.3)
    ax.set_xlim(0, 122)
    ax.set_xlabel("grounded object nodes (%)", fontsize=8.5)
    ax.set_title("Grounded object-node outcome", fontsize=10.2, fontweight="bold")
    ax.grid(axis="x")
    ax.text(
        0.5,
        -0.30,
        "same greedy baseline; partial + disappeared = reduced",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.4,
        color=COLORS["muted"],
    )
    return save(fig, output_dir, "grounded_node_outcome_comparison", formats)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="append", required=True, help="label:path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--formats", default="svg,png,pdf")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    formats = [fmt.strip() for fmt in args.formats.split(",") if fmt.strip()]
    rows = []
    for item in args.summary:
        if ":" not in item:
            raise ValueError(f"--summary must be label:path, got {item}")
        label, path = item.split(":", 1)
        rows.append(load_summary(path, label))
    write_csv(os.path.join(args.output_dir, "grounded_node_outcome_comparison.csv"), rows)
    out = {
        "rows": rows,
        "figures": plot(rows, args.output_dir, formats),
    }
    with open(os.path.join(args.output_dir, "grounded_node_outcome_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

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


COLORS = {
    "dark": "#111827",
    "muted": "#64748b",
    "grid": "#e5e7eb",
    "hall": "#dc2626",
    "ground": "#059669",
    "partial": "#f59e0b",
    "disappear": "#ef4444",
    "preserve": "#cbd5e1",
}


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


def safe_float(value, default=0.0):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def load_top_heads(path, top_k):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    heads = data.get("heads", data if isinstance(data, list) else [])
    keys = []
    for row in heads[:top_k]:
        if "head_key" in row:
            keys.append(str(row["head_key"]))
        else:
            keys.append(f"{int(row['layer'])}:{int(row['head'])}")
    return set(keys)


def load_grounded_outcomes(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    n = int(data["base_ground_nodes"])
    partial = int(data["partial_loss_ground_nodes"])
    disappeared = int(data["disappeared_ground_nodes"])
    reduced = int(data["reduced_ground_nodes"])
    preserved = max(0, n - reduced)
    return {
        "grounded_nodes": n,
        "preserved": preserved,
        "partial": partial,
        "disappeared": disappeared,
        "reduced": reduced,
        "preserved_rate": preserved / n if n else 0.0,
        "partial_rate": partial / n if n else 0.0,
        "disappeared_rate": disappeared / n if n else 0.0,
        "reduced_rate": reduced / n if n else 0.0,
        "source": path,
    }


def summarize_triggers(rows, head_keys, text_tau, ratio_tau):
    kept = [row for row in rows if row.get("head_key") in head_keys and row.get("label") in {"grounded", "hallucinated"}]
    out = []
    for label in ["hallucinated", "grounded"]:
        items = [row for row in kept if row.get("label") == label]
        n = len(items)
        text_triggered = sum(safe_float(row.get("text_before")) >= text_tau for row in items)
        ratio_triggered = sum(safe_float(row.get("bounded_ratio")) >= ratio_tau for row in items)
        out.append(
            {
                "label": label,
                "n_head_steps": n,
                "text_tau": text_tau,
                "text_triggered_head_steps": text_triggered,
                "text_trigger_rate": text_triggered / n if n else 0.0,
                "ratio_tau": ratio_tau,
                "ratio_triggered_head_steps": ratio_triggered,
                "ratio_trigger_rate": ratio_triggered / n if n else 0.0,
            }
        )
    return out


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


def plot(output_dir, formats, trigger_rows, outcome):
    setup_style()
    fig = plt.figure(figsize=(7.3, 2.8), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.2])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    trigger = {row["label"]: row for row in trigger_rows}
    labels = ["hallucinated", "grounded"]
    values = [trigger[label]["text_trigger_rate"] * 100 for label in labels]
    colors = [COLORS["hall"], COLORS["ground"]]
    x = range(len(labels))
    ax0.bar(x, values, color=colors, width=0.56, alpha=0.88)
    for idx, value in enumerate(values):
        ax0.text(idx, value + 2.0, f"{value:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold", color=COLORS["dark"])
    ax0.set_xticks(list(x))
    ax0.set_xticklabels(["hallucinated\nobject", "grounded\nobject"], fontsize=8.5)
    ax0.set_ylim(0, max(55, max(values) + 9))
    ax0.set_ylabel("triggered head-steps (%)", fontsize=8.5)
    ax0.set_title("Static trigger exposure", fontsize=10.5, fontweight="bold")
    ax0.grid(axis="y")
    ax0.text(
        0.5,
        -0.33,
        "top-100 L9-L16 heads, hard rule: text mass >= 0.4",
        transform=ax0.transAxes,
        ha="center",
        va="top",
        fontsize=7.4,
        color=COLORS["muted"],
    )

    rates = [
        outcome["preserved_rate"],
        outcome["partial_rate"],
        outcome["disappeared_rate"],
    ]
    names = ["preserved", "partially\nreduced", "disappeared"]
    cols = [COLORS["preserve"], COLORS["partial"], COLORS["disappear"]]
    left = 0.0
    segment_centers = []
    for rate, name, color in zip(rates, names, cols):
        ax1.barh([0], [rate * 100], left=[left * 100], color=color, height=0.38, alpha=0.9)
        segment_centers.append(((left + rate / 2) * 100, rate, name, color))
        left += rate
    preserve_x, preserve_rate, _, _ = segment_centers[0]
    ax1.text(
        preserve_x,
        0,
        f"{preserve_rate * 100:.1f}%\npreserved",
        ha="center",
        va="center",
        fontsize=8.8,
        color=COLORS["dark"],
        fontweight="bold",
    )
    for x_center, rate, name, color in segment_centers[1:]:
        ax1.text(
            x_center,
            0,
            f"{rate * 100:.1f}%",
            ha="center",
            va="center",
            fontsize=8.2,
            color=COLORS["dark"],
            fontweight="bold",
        )
    ax1.set_xlim(0, 100)
    ax1.set_ylim(-0.36, 0.78)
    ax1.set_yticks([])
    ax1.set_xlabel("grounded object nodes (%)", fontsize=8.5)
    ax1.set_title("Grounded collateral outcome", fontsize=10.5, fontweight="bold")
    ax1.grid(axis="x")
    ax1.text(
        0.5,
        -0.33,
        f"{outcome['reduced_rate'] * 100:.1f}% reduced = {outcome['partial_rate'] * 100:.1f}% partial + {outcome['disappeared_rate'] * 100:.1f}% disappeared",
        transform=ax1.transAxes,
        ha="center",
        va="top",
        fontsize=7.4,
        color=COLORS["muted"],
    )
    fig.suptitle("Static hard suppression is broad at object steps", fontsize=11.6, fontweight="bold", y=1.03)
    return save(fig, output_dir, "static_overbroad_top100", formats)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--object-ratio-csv", required=True)
    parser.add_argument("--ranked-heads-json", required=True)
    parser.add_argument("--hard-outcome-json", required=True)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--text-tau", type=float, default=0.4)
    parser.add_argument("--ratio-tau", type=float, default=0.9)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--formats", default="svg,png,pdf")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    formats = [fmt.strip() for fmt in args.formats.split(",") if fmt.strip()]
    head_keys = load_top_heads(args.ranked_heads_json, args.top_k)
    rows = read_csv(args.object_ratio_csv)
    trigger_rows = summarize_triggers(rows, head_keys, args.text_tau, args.ratio_tau)
    outcome = load_grounded_outcomes(args.hard_outcome_json)

    write_csv(os.path.join(args.output_dir, "static_overbroad_top100_trigger_summary.csv"), trigger_rows)
    with open(os.path.join(args.output_dir, "static_overbroad_top100_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "object_ratio_csv": args.object_ratio_csv,
                "ranked_heads_json": args.ranked_heads_json,
                "hard_outcome_json": args.hard_outcome_json,
                "top_k": args.top_k,
                "trigger_summary": trigger_rows,
                "grounded_outcome": outcome,
                "figures": plot(args.output_dir, formats, trigger_rows, outcome),
            },
            f,
            indent=2,
        )
    print(json.dumps({"trigger_summary": trigger_rows, "grounded_outcome": outcome}, indent=2))


if __name__ == "__main__":
    main()

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
    "nonselected": "#94a3b8",
    "selected": "#7c3aed",
    "ground": "#059669",
    "hall": "#dc2626",
}


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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


def load_group_summary(path):
    rows = read_csv(path)
    by_group = {row.get("group"): row for row in rows}
    selected = by_group.get("selected")
    nonselected = by_group.get("non-selected")
    if not selected or not nonselected:
        raise ValueError(f"expected selected and non-selected rows in {path}")
    return selected, nonselected


def load_causal_summary(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    perturb = data.get("logprob_perturbation", data)
    ground = perturb.get("grounded_object")
    hall = perturb.get("hallucinated_object")
    if not ground or not hall:
        raise ValueError(f"expected grounded_object and hallucinated_object in {path}")
    return data, ground, hall


def plot_image_drop(output_dir, formats, selected, nonselected):
    setup_style()
    sel = safe_float(selected.get("mean_image_drop_GminusH"))
    non = safe_float(nonselected.get("mean_image_drop_GminusH"))
    ratio = sel / non if non > 0 else 0.0

    fig, ax = plt.subplots(figsize=(2.85, 2.55), constrained_layout=True)
    ax.bar([0], [non], color=COLORS["nonselected"], width=0.56, alpha=0.78)
    ax.bar([1], [sel], color=COLORS["selected"], width=0.56, alpha=0.92)
    ymax = max(sel, non) * 1.46 if max(sel, non) > 0 else 1.0
    ax.plot([0, 1], [non, sel], color="#6b7280", linewidth=1.35, alpha=0.82)
    ax.text(0, non + ymax * 0.04, f"{non:.3f}", ha="center", fontsize=8.1, color=COLORS["dark"])
    ax.text(1, sel + ymax * 0.04, f"{sel:.3f}", ha="center", fontsize=8.1, color=COLORS["dark"])
    ax.text(
        0.5,
        ymax * 0.92,
        f"{ratio:.1f}x",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=COLORS["selected"],
        bbox=dict(boxstyle="round,pad=0.18", facecolor="#f5f3ff", edgecolor="none"),
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["non", "sel"], fontsize=8.5)
    ax.set_ylabel("image mass drop", fontsize=8.5)
    ax.set_ylim(0, ymax)
    ax.set_title("Image-token routing weakens", fontsize=10.2, fontweight="bold")
    ax.grid(axis="y")
    ax.text(
        0.5,
        -0.31,
        "G-H image mass drop\n(non-selection check)",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.4,
        color=COLORS["muted"],
    )
    paths = save(fig, output_dir, "section_d_image_mass_drop", formats)
    return paths, {"selected_image_drop": sel, "nonselected_image_drop": non, "ratio": ratio}


def plot_causal_fragility(output_dir, formats, ground, hall):
    setup_style()
    ground_drop = safe_float(ground.get("mean_delta_logprob"))
    hall_drop = safe_float(hall.get("mean_delta_logprob"))
    ground_top1 = safe_float(ground.get("top1_loss_fraction"))
    hall_top1 = safe_float(hall.get("top1_loss_fraction"))
    ratio = hall_drop / ground_drop if ground_drop > 0 else 0.0

    fig, ax = plt.subplots(figsize=(3.25, 2.55), constrained_layout=True)
    values = [ground_drop, hall_drop]
    colors = [COLORS["ground"], COLORS["hall"]]
    ax.bar([0, 1], values, width=0.58, color=colors, alpha=0.88)
    ymax = max(values) * 1.42 if max(values) > 0 else 1.0
    for idx, value in enumerate(values):
        ax.text(idx, value + ymax * 0.04, f"{value:.3f}", ha="center", fontsize=8.2, fontweight="bold")
    ax.text(
        0.5,
        ymax * 0.92,
        f"{ratio:.1f}x larger drop",
        ha="center",
        va="center",
        fontsize=9.3,
        fontweight="bold",
        color=COLORS["hall"],
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#fef2f2", edgecolor="none"),
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["grounded\nobject", "hallucinated\nobject"], fontsize=8.2)
    ax.set_ylabel(r"$\Delta \log p(y_t)$", fontsize=8.5)
    ax.set_ylim(0, ymax)
    ax.set_title("Causal fragility under suppression", fontsize=10.2, fontweight="bold")
    ax.grid(axis="y")
    ax.text(
        0.5,
        -0.36,
        f"top-1 token changes: {ground_top1 * 100:.1f}% grounded, {hall_top1 * 100:.1f}% hallucinated",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.2,
        color=COLORS["muted"],
    )
    paths = save(fig, output_dir, "section_d_causal_fragility", formats)
    return paths, {
        "grounded_mean_delta_logprob": ground_drop,
        "hallucinated_mean_delta_logprob": hall_drop,
        "hall_over_ground_delta_ratio": ratio,
        "grounded_top1_loss_fraction": ground_top1,
        "hallucinated_top1_loss_fraction": hall_top1,
    }


def write_notes(path, image_stats, causal_stats, args):
    text = f"""# Section III-D Split Figure Notes

The D figure is split into two non-overlapping claims.

## Panel A: image mass drop

Source: `{args.group_summary_csv}`

This panel uses only the image-token mass drop:

```text
E_G[M_img] - E_H[M_img]
```

This metric is not one of the two primary selection axes (`I_text` and `C_toi`), so it is the least circular observational statistic for D.

- selected: {image_stats['selected_image_drop']:.6f}
- non-selected: {image_stats['nonselected_image_drop']:.6f}
- ratio: {image_stats['ratio']:.3f}x

## Panel B: causal fragility

Source: `{args.causal_summary_json}`

This panel reports:

```text
Delta log p(y_t) = log p_base(y_t) - log p_suppressed(y_t)
```

- grounded mean Delta logp: {causal_stats['grounded_mean_delta_logprob']:.6f}
- hallucinated mean Delta logp: {causal_stats['hallucinated_mean_delta_logprob']:.6f}
- hallucinated / grounded ratio: {causal_stats['hall_over_ground_delta_ratio']:.3f}x
- grounded top-1 token change fraction: {causal_stats['grounded_top1_loss_fraction']:.3f}
- hallucinated top-1 token change fraction: {causal_stats['hallucinated_top1_loss_fraction']:.3f}

Use this as a causal diagnostic panel. If a stricter current-pool generic suppression run is available, replace `--causal-summary-json` with that source and regenerate the same panel.
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group-summary-csv", required=True)
    parser.add_argument("--causal-summary-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--formats", default="svg,png,pdf")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    formats = [item.strip() for item in args.formats.split(",") if item.strip()]
    selected, nonselected = load_group_summary(args.group_summary_csv)
    causal_source, ground, hall = load_causal_summary(args.causal_summary_json)
    image_paths, image_stats = plot_image_drop(args.output_dir, formats, selected, nonselected)
    causal_paths, causal_stats = plot_causal_fragility(args.output_dir, formats, ground, hall)
    out = {
        "group_summary_csv": args.group_summary_csv,
        "causal_summary_json": args.causal_summary_json,
        "image_drop": image_stats,
        "causal_fragility": causal_stats,
        "source_meta": causal_source.get("sources", {}),
        "figures": {
            "image_mass_drop": image_paths,
            "causal_fragility": causal_paths,
        },
    }
    with open(os.path.join(args.output_dir, "section_d_split_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    write_notes(os.path.join(args.output_dir, "section_d_split_notes.md"), image_stats, causal_stats, args)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

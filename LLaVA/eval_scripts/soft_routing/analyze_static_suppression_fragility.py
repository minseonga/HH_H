#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
from collections import defaultdict

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "dark": "#111827",
    "muted": "#64748b",
    "grid": "#e2e8f0",
    "hall": "#dc2626",
    "ground": "#059669",
    "nonobject": "#94a3b8",
    "selected": "#7c3aed",
}


def setup_style():
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 320,
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
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
    if value is None or value == "":
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def safe_int(value, default=0):
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def mean(values):
    values = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(values)) if values else 0.0


def quantile(values, q):
    values = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.quantile(values, q)) if values else 0.0


def frac(values, predicate):
    values = [float(v) for v in values if math.isfinite(float(v))]
    if not values:
        return 0.0
    return sum(1 for v in values if predicate(v)) / len(values)


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
        json.dump(data, f, indent=2)


def save(fig, output_dir, name, formats):
    paths = {}
    for fmt in formats:
        path = os.path.join(output_dir, f"{name}.{fmt}")
        fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
        paths[fmt] = path
    plt.close(fig)
    return paths


def summarize_samples(samples, n_selected_heads):
    rows = []
    totals = {
        "n_samples": len(samples),
        "total_steps": 0,
        "hallucinated_object_steps": 0,
        "grounded_object_steps": 0,
        "object_steps": 0,
        "non_object_steps": 0,
    }
    for row in samples:
        n_steps = safe_int(row.get("n_steps"))
        n_hall = safe_int(row.get("n_hallucinated_steps"))
        n_ground = safe_int(row.get("n_grounded_steps"))
        n_object = n_hall + n_ground
        n_non_object = max(0, n_steps - n_object)
        totals["total_steps"] += n_steps
        totals["hallucinated_object_steps"] += n_hall
        totals["grounded_object_steps"] += n_ground
        totals["object_steps"] += n_object
        totals["non_object_steps"] += n_non_object
        rows.append(
            {
                "sample_index": row.get("sample_index"),
                "question_id": row.get("question_id"),
                "n_steps": n_steps,
                "hallucinated_object_steps": n_hall,
                "grounded_object_steps": n_ground,
                "non_object_steps": n_non_object,
                "static_hall_head_step_exposures": n_hall * n_selected_heads,
                "static_grounded_head_step_exposures": n_ground * n_selected_heads,
                "static_non_object_head_step_exposures": n_non_object * n_selected_heads,
            }
        )

    total_steps = max(1, totals["total_steps"])
    head_total = total_steps * n_selected_heads
    exposure_rows = [
        {
            "bucket": "hallucinated_object",
            "interpretation": "desired intervention target",
            "token_steps": totals["hallucinated_object_steps"],
            "token_step_fraction": totals["hallucinated_object_steps"] / total_steps,
            "static_head_step_exposures": totals["hallucinated_object_steps"] * n_selected_heads,
            "static_head_step_fraction": (totals["hallucinated_object_steps"] * n_selected_heads) / head_total,
        },
        {
            "bucket": "grounded_object",
            "interpretation": "possible collateral object target",
            "token_steps": totals["grounded_object_steps"],
            "token_step_fraction": totals["grounded_object_steps"] / total_steps,
            "static_head_step_exposures": totals["grounded_object_steps"] * n_selected_heads,
            "static_head_step_fraction": (totals["grounded_object_steps"] * n_selected_heads) / head_total,
        },
        {
            "bucket": "non_object",
            "interpretation": "unnecessary non-object intervention",
            "token_steps": totals["non_object_steps"],
            "token_step_fraction": totals["non_object_steps"] / total_steps,
            "static_head_step_exposures": totals["non_object_steps"] * n_selected_heads,
            "static_head_step_fraction": (totals["non_object_steps"] * n_selected_heads) / head_total,
        },
    ]
    totals["n_selected_heads"] = n_selected_heads
    totals["static_head_step_exposures"] = head_total
    return totals, rows, exposure_rows


def summarize_distribution(values):
    return {
        "n": len(values),
        "mean": mean(values),
        "q10": quantile(values, 0.10),
        "q25": quantile(values, 0.25),
        "q50": quantile(values, 0.50),
        "q75": quantile(values, 0.75),
        "q90": quantile(values, 0.90),
    }


def summarize_object_trace(rows):
    head_keys = sorted({row.get("head_key") for row in rows if row.get("head_key")})
    by_label_rows = defaultdict(list)
    by_step = defaultdict(list)
    for row in rows:
        label = row.get("label", "")
        if label:
            by_label_rows[label].append(row)
        key = (
            row.get("question_id"),
            row.get("image"),
            row.get("step_idx"),
            row.get("token_id"),
            row.get("token_text"),
            label,
        )
        by_step[key].append(row)

    step_rows = []
    for key, items in sorted(by_step.items()):
        qid, image, step_idx, token_id, token_text, label = key
        step_rows.append(
            {
                "question_id": qid,
                "image": image,
                "step_idx": step_idx,
                "token_id": token_id,
                "token_text": token_text,
                "label": label,
                "n_selected_head_rows": len(items),
                "mean_text_before": mean([safe_float(row.get("text_before")) for row in items]),
                "mean_image_before": mean([safe_float(row.get("image_before")) for row in items]),
                "mean_bounded_ratio": mean([safe_float(row.get("bounded_ratio")) for row in items]),
                "mean_delta": mean([safe_float(row.get("delta")) for row in items]),
                "frac_ratio_ge_0p9": frac([safe_float(row.get("bounded_ratio")) for row in items], lambda v: v >= 0.9),
                "frac_text_ge_0p4": frac([safe_float(row.get("text_before")) for row in items], lambda v: v >= 0.4),
            }
        )

    summary_rows = []
    for level, grouped in [
        ("head_step_row", by_label_rows),
        ("object_step_mean", defaultdict(list)),
    ]:
        if level == "object_step_mean":
            for row in step_rows:
                grouped[row["label"]].append(row)
        for label in sorted(grouped):
            items = grouped[label]
            prefix = "" if level == "head_step_row" else "mean_"
            ratio_values = [safe_float(row.get(f"{prefix}bounded_ratio")) for row in items]
            text_values = [safe_float(row.get(f"{prefix}text_before")) for row in items]
            image_values = [safe_float(row.get(f"{prefix}image_before")) for row in items]
            delta_values = [safe_float(row.get(f"{prefix}delta")) for row in items if f"{prefix}delta" in row]
            ratio_stats = summarize_distribution(ratio_values)
            text_stats = summarize_distribution(text_values)
            image_stats = summarize_distribution(image_values)
            delta_stats = summarize_distribution(delta_values)
            summary_rows.append(
                {
                    "level": level,
                    "label": label,
                    "n": len(items),
                    "text_mean": text_stats["mean"],
                    "text_q50": text_stats["q50"],
                    "text_q75": text_stats["q75"],
                    "image_mean": image_stats["mean"],
                    "image_q50": image_stats["q50"],
                    "image_q75": image_stats["q75"],
                    "bounded_ratio_mean": ratio_stats["mean"],
                    "bounded_ratio_q25": ratio_stats["q25"],
                    "bounded_ratio_q50": ratio_stats["q50"],
                    "bounded_ratio_q75": ratio_stats["q75"],
                    "bounded_ratio_q90": ratio_stats["q90"],
                    "frac_bounded_ratio_ge_0p9": frac(ratio_values, lambda v: v >= 0.9),
                    "frac_text_ge_0p4": frac(text_values, lambda v: v >= 0.4),
                    "delta_mean": delta_stats["mean"],
                    "delta_q50": delta_stats["q50"],
                    "delta_q75": delta_stats["q75"],
                    "frac_delta_ge_0p8": frac(delta_values, lambda v: v >= 0.8),
                }
            )
    return head_keys, step_rows, summary_rows


def summarize_teacher_jsonl(path):
    if not path or not os.path.exists(path):
        return [], {}

    grouped = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            grouped[row.get("label_family", row.get("label", ""))].append(row)

    rows = []
    for label in sorted(grouped):
        items = grouped[label]
        effects = [safe_float(row.get("causal_effect")) for row in items]
        positive = [v for v in effects if v > 0]
        rows.append(
            {
                "label_family": label,
                "n_rows": len(items),
                "mean_causal_effect": mean(effects),
                "q50_causal_effect": quantile(effects, 0.50),
                "q90_causal_effect": quantile(effects, 0.90),
                "positive_effect_fraction": len(positive) / len(effects) if effects else 0.0,
                "positive_effect_mean": mean(positive),
                "positive_effect_sum_per_row": sum(positive) / len(effects) if effects else 0.0,
            }
        )
    meta = {
        "path": path,
        "n_rows": sum(len(items) for items in grouped.values()),
        "labels": sorted(grouped),
        "caveat": "single-head zero-ablation diagnostic, not full selected-pool static suppression",
    }
    return rows, meta


def plot_static_exposure(output_dir, formats, exposure_rows):
    setup_style()
    fig, ax = plt.subplots(figsize=(5.9, 1.9), constrained_layout=True)
    left = 0.0
    labels = {
        "hallucinated_object": "hallucinated\nobject",
        "grounded_object": "grounded\nobject",
        "non_object": "non-object\nsteps",
    }
    colors = {
        "hallucinated_object": COLORS["hall"],
        "grounded_object": COLORS["ground"],
        "non_object": COLORS["nonobject"],
    }
    callout_xytext = {
        "hallucinated_object": (0.055, 0.78),
        "grounded_object": (0.135, 0.50),
    }
    for row in exposure_rows:
        width = row["token_step_fraction"]
        bucket = row["bucket"]
        center = left + width / 2
        ax.barh([0], [width], left=[left], height=0.38, color=colors[bucket], alpha=0.9)
        if width > 0.18:
            ax.text(center, 0, f"{labels[bucket]}\n{width * 100:.1f}%", ha="center", va="center", color=COLORS["dark"], fontsize=7.2, fontweight="bold")
        elif bucket in callout_xytext:
            ax.annotate(
                f"{labels[bucket]}\n{width * 100:.1f}%",
                xy=(center, 0.18),
                xytext=callout_xytext[bucket],
                textcoords="data",
                ha="center",
                va="bottom",
                color=colors[bucket],
                fontsize=7.0,
                fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=colors[bucket], linewidth=0.9),
            )
        elif width > 0.035:
            ax.annotate(
                f"{labels[bucket]}\n{width * 100:.1f}%",
                xy=(center, 0.18),
                xytext=(center + 0.045, 0.57),
                textcoords="data",
                ha="center",
                va="bottom",
                color=colors[bucket],
                fontsize=7.0,
                fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=colors[bucket], linewidth=0.9),
            )
        else:
            ax.annotate(
                f"{labels[bucket]}\n{width * 100:.1f}%",
                xy=(center, 0.18),
                xytext=(center + 0.035, 0.78),
                textcoords="data",
                ha="center",
                va="bottom",
                color=colors[bucket],
                fontsize=7.0,
                fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=colors[bucket], linewidth=0.9),
            )
        left += width
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.35, 0.95)
    ax.set_yticks([])
    ax.set_xlabel("fraction of decoding steps receiving static suppression")
    ax.set_title("Static head suppression over-triggers by construction", fontweight="bold")
    ax.grid(axis="x")
    return save(fig, output_dir, "static_suppression_exposure_stacked", formats)


def plot_ratio_overlap(output_dir, formats, object_rows, step_rows):
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.55), constrained_layout=True)
    bins = np.linspace(0, 1, 32)
    for ax, rows, title, field in [
        (axes[0], object_rows, "selected head-step rows", "bounded_ratio"),
        (axes[1], step_rows, "object steps averaged over selected heads", "mean_bounded_ratio"),
    ]:
        for label, color in [("grounded", COLORS["ground"]), ("hallucinated", COLORS["hall"])]:
            values = [safe_float(row.get(field)) for row in rows if row.get("label") == label]
            ax.hist(values, bins=bins, density=True, histtype="stepfilled", alpha=0.28, color=color, label=label)
            ax.hist(values, bins=bins, density=True, histtype="step", linewidth=1.2, color=color)
            if values:
                ax.axvline(np.median(values), color=color, linewidth=1.2, linestyle="--")
        ax.set_xlabel("text reliance T/(T+I)")
        ax.set_ylabel("density")
        ax.set_title(title)
        ax.grid(axis="y")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Selected heads are also active in grounded object generation", y=1.06, fontsize=10.5, fontweight="bold")
    return save(fig, output_dir, "selected_head_grounded_hall_ratio_overlap", formats)


def plot_teacher_effects(output_dir, formats, teacher_rows):
    if not teacher_rows:
        return {}
    setup_style()
    labels = [row["label_family"] for row in teacher_rows]
    positive_means = [row["positive_effect_mean"] for row in teacher_rows]
    positive_rates = [row["positive_effect_fraction"] for row in teacher_rows]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(4.9, 2.55), constrained_layout=True)
    colors = [COLORS["hall"] if "hall" in label else COLORS["ground"] for label in labels]
    ax.bar(x, positive_means, width=0.58, color=colors, alpha=0.82)
    for idx, (value, rate) in enumerate(zip(positive_means, positive_rates)):
        ax.text(idx, value + max(positive_means) * 0.04, f"{rate * 100:.1f}% +", ha="center", va="bottom", fontsize=7.0, color=COLORS["dark"])
    ax.set_xticks(x)
    ax.set_xticklabels([label.replace("_", "\n") for label in labels])
    ax.set_ylabel("mean positive Δ log p")
    ax.set_title("Grounded object likelihood is not untouched by head ablation", fontweight="bold")
    ax.grid(axis="y")
    return save(fig, output_dir, "single_head_teacher_effect_by_label", formats)


def write_notes(path, summary):
    exposure = {row["bucket"]: row for row in summary["static_exposure"]}
    ratio = {(row["level"], row["label"]): row for row in summary["object_ratio_summary"]}
    teacher = {row["label_family"]: row for row in summary.get("teacher_effect_summary", [])}

    hall_frac = exposure["hallucinated_object"]["token_step_fraction"]
    ground_frac = exposure["grounded_object"]["token_step_fraction"]
    non_frac = exposure["non_object"]["token_step_fraction"]
    grounded_row = ratio.get(("head_step_row", "grounded"), {})
    hall_row = ratio.get(("head_step_row", "hallucinated"), {})
    grounded_step = ratio.get(("object_step_mean", "grounded"), {})
    hall_step = ratio.get(("object_step_mean", "hallucinated"), {})
    kept = teacher.get("kept_grounded", {})
    lost = teacher.get("lost_grounded", {})
    hall = teacher.get("hallucinated", {})

    text = f"""# Static Suppression Fragility Notes

## What This Diagnostic Supports

This analysis supports Section III-F: static head suppression is a coarse intervention because it treats the selected head pool as always dangerous.

## 1. Static Suppression Over-Triggers

In the traced captions, hallucinated object positions are rare relative to all decoding positions:

| token bucket | fraction of decoding steps |
|---|---:|
| hallucinated object | {hall_frac * 100:.2f}% |
| grounded object | {ground_frac * 100:.2f}% |
| non-object | {non_frac * 100:.2f}% |

A static method that suppresses the selected heads at every step therefore applies most of its interventions outside the desired hallucinated-object target. This does not mean static suppression cannot reduce CHAIR. It means the intervention unit is too coarse: head identity alone does not encode whether the current token state is hallucination-prone.

## 2. The Same Selected Heads Are Active for Grounded Objects

For selected head-step rows, the text-reliance ratio distributions overlap strongly:

| label | median T/(T+I) | q75 T/(T+I) | frac ratio >= 0.9 |
|---|---:|---:|---:|
| grounded | {grounded_row.get('bounded_ratio_q50', 0.0):.3f} | {grounded_row.get('bounded_ratio_q75', 0.0):.3f} | {grounded_row.get('frac_bounded_ratio_ge_0p9', 0.0) * 100:.1f}% |
| hallucinated | {hall_row.get('bounded_ratio_q50', 0.0):.3f} | {hall_row.get('bounded_ratio_q75', 0.0):.3f} | {hall_row.get('frac_bounded_ratio_ge_0p9', 0.0) * 100:.1f}% |

At the object-step level, after averaging over selected heads, the same overlap remains:

| label | median mean T/(T+I) | q75 mean T/(T+I) |
|---|---:|---:|
| grounded | {grounded_step.get('bounded_ratio_q50', 0.0):.3f} | {grounded_step.get('bounded_ratio_q75', 0.0):.3f} |
| hallucinated | {hall_step.get('bounded_ratio_q50', 0.0):.3f} | {hall_step.get('bounded_ratio_q75', 0.0):.3f} |

This is the key evidence against a detector interpretation. The selected heads are useful intervention channels, but their activation is not hallucination-only.

## 3. Grounded Object Log Probability Can Also Depend on Text-Heavy Heads

The single-head teacher diagnostic is not full static-pool suppression, but it shows that ablating text-heavy heads can reduce the original target object's log probability even for grounded examples:

| label family | positive-effect fraction | mean positive Δ log p | q90 Δ log p |
|---|---:|---:|---:|
| kept grounded | {kept.get('positive_effect_fraction', 0.0) * 100:.1f}% | {kept.get('positive_effect_mean', 0.0):.4f} | {kept.get('q90_causal_effect', 0.0):.4f} |
| lost grounded | {lost.get('positive_effect_fraction', 0.0) * 100:.1f}% | {lost.get('positive_effect_mean', 0.0):.4f} | {lost.get('q90_causal_effect', 0.0):.4f} |
| hallucinated | {hall.get('positive_effect_fraction', 0.0) * 100:.1f}% | {hall.get('positive_effect_mean', 0.0):.4f} | {hall.get('q90_causal_effect', 0.0):.4f} |

The hallucinated and lost-grounded groups are more fragile, but kept-grounded rows are not zero. This supports the claim that static suppression can reduce hallucinated likelihood while also perturbing ordinary grounded object realization.

## Paper Paragraph Draft

Static suppression exposes the limitation of the detector-centric view. If the selected heads were hallucination-only detectors, suppressing them uniformly would mainly affect hallucinated object positions. The trace shows otherwise. In 100 sampled captions, hallucinated object positions account for only {hall_frac * 100:.2f}% of decoding steps, whereas grounded object positions account for {ground_frac * 100:.2f}% and non-object positions for {non_frac * 100:.2f}%. Moreover, the same selected heads are strongly text-reliant for grounded object generation: grounded selected head-step rows have median text-reliance ratio {grounded_row.get('bounded_ratio_q50', 0.0):.3f}, close to hallucinated rows at {hall_row.get('bounded_ratio_q50', 0.0):.3f}. A single-head counterfactual diagnostic further shows that text-heavy head ablation produces positive target-log-probability drops even for kept-grounded objects in {kept.get('positive_effect_fraction', 0.0) * 100:.1f}% of rows. Static suppression can therefore lower hallucinated object likelihood, but it does so with a coarse intervention unit: it suppresses intervention-relevant heads regardless of whether the current token state is hallucinated, grounded, or non-object.

## Caveat

The log-probability diagnostic uses single-head zero ablation from the existing teacher rows. It is a mechanistic collateral-damage diagnostic, not a direct measurement of full selected-pool static suppression.
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-csv", required=True)
    parser.add_argument("--object-ratio-csv", required=True)
    parser.add_argument("--teacher-jsonl", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--formats", default="png,pdf,svg")
    args = parser.parse_args()

    formats = [fmt.strip() for fmt in args.formats.split(",") if fmt.strip()]
    os.makedirs(args.output_dir, exist_ok=True)

    samples = read_csv(args.samples_csv)
    object_rows = read_csv(args.object_ratio_csv)
    head_keys, step_rows, object_ratio_summary = summarize_object_trace(object_rows)
    n_selected_heads = len(head_keys)
    if n_selected_heads <= 0:
        raise ValueError(f"could not infer selected heads from {args.object_ratio_csv}")

    sample_totals, sample_rows, exposure_rows = summarize_samples(samples, n_selected_heads)
    teacher_rows, teacher_meta = summarize_teacher_jsonl(args.teacher_jsonl)

    write_csv(os.path.join(args.output_dir, "static_suppression_exposure_by_sample.csv"), sample_rows)
    write_csv(os.path.join(args.output_dir, "static_suppression_exposure_summary.csv"), exposure_rows)
    write_csv(os.path.join(args.output_dir, "selected_head_object_step_means.csv"), step_rows)
    write_csv(os.path.join(args.output_dir, "selected_head_object_ratio_overlap_summary.csv"), object_ratio_summary)
    if teacher_rows:
        write_csv(os.path.join(args.output_dir, "single_head_teacher_effect_summary.csv"), teacher_rows)

    figure_paths = {
        "static_exposure": plot_static_exposure(args.output_dir, formats, exposure_rows),
        "ratio_overlap": plot_ratio_overlap(args.output_dir, formats, object_rows, step_rows),
    }
    if teacher_rows:
        figure_paths["teacher_effects"] = plot_teacher_effects(args.output_dir, formats, teacher_rows)

    summary = {
        "sources": {
            "samples_csv": args.samples_csv,
            "object_ratio_csv": args.object_ratio_csv,
            "teacher_jsonl": args.teacher_jsonl,
        },
        "n_selected_heads": n_selected_heads,
        "sample_totals": sample_totals,
        "static_exposure": exposure_rows,
        "object_ratio_summary": object_ratio_summary,
        "teacher_effect_summary": teacher_rows,
        "teacher_meta": teacher_meta,
        "figures": figure_paths,
    }
    write_json(os.path.join(args.output_dir, "static_suppression_diagnostics_summary.json"), summary)
    write_notes(os.path.join(args.output_dir, "section_iii_f_static_suppression_notes.md"), summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

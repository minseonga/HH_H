#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import textwrap
from collections import defaultdict
from xml.sax.saxutils import escape as xml_escape

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


COLORS = {
    "dark": "#0f172a",
    "muted": "#64748b",
    "grid": "#e2e8f0",
    "tail": "#cbd5e1",
    "text": "#f97316",
    "image": "#2563eb",
    "system": "#64748b",
    "hall": "#dc2626",
    "ground": "#059669",
    "score": "#7c3aed",
    "window": "#fef3c7",
}


def setup_style():
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 320,
            "font.family": "DejaVu Sans",
            "font.size": 7.2,
            "axes.titlesize": 8.4,
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


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)


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


def as_float(row, key, default=0.0):
    try:
        value = float(row.get(key, default))
        return value if math.isfinite(value) else default
    except Exception:
        return default


def as_int(row, key, default=0):
    try:
        return int(float(row.get(key, default)))
    except Exception:
        return default


def mean(values):
    values = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(values)) if values else 0.0


def parse_layers(value):
    if value is None or value == "" or value == "all":
        return []
    if isinstance(value, list):
        return sorted(int(item) for item in value)
    layers = []
    for piece in str(value).split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            start_text, end_text = piece.split("-", 1)
            start, end = int(start_text), int(end_text)
            step = 1 if end >= start else -1
            layers.extend(range(start, end + step, step))
        else:
            layers.append(int(piece))
    return sorted(set(layers))


def save_figure(fig, output_dir, name, formats):
    os.makedirs(output_dir, exist_ok=True)
    paths = {}
    for fmt in formats:
        path = os.path.join(output_dir, f"{name}.{fmt}")
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        paths[fmt] = path
    plt.close(fig)
    return paths


def require(path, description):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def load_source(source_dir, ratio_source):
    paths = {
        "summary": os.path.join(source_dir, "method_figure_source_summary.json"),
        "head_scores": require(os.path.join(source_dir, "head_scores_all.csv"), "head score CSV"),
        "selected_ratio": require(
            os.path.join(source_dir, "selected_head_object_ratio_distribution.csv"),
            "selected ratio distribution CSV",
        ),
        "gate_curve": require(os.path.join(source_dir, "gate_curve.csv"), "gate curve CSV"),
        "gate_markers": os.path.join(source_dir, "gate_markers.csv"),
        "redistribution": require(
            os.path.join(source_dir, "attention_redistribution_summary.csv"),
            "attention redistribution summary CSV",
        ),
        "rank_fusion": os.path.join(source_dir, "rank_fusion_summary.csv"),
        "all_head_npz": os.path.join(source_dir, "all_head_object_attention.npz"),
        "samples": os.path.join(source_dir, "samples.csv"),
    }
    summary = read_json(paths["summary"], default={}) or {}
    head_rows = read_csv(paths["head_scores"])
    ratio_rows = read_csv(paths["selected_ratio"])
    if ratio_source == "all":
        ratio_rows = load_all_head_ratio_rows(paths["all_head_npz"])
    return {
        "paths": paths,
        "summary": summary,
        "head_rows": head_rows,
        "ratio_rows": ratio_rows,
        "gate_curve": read_csv(paths["gate_curve"]),
        "gate_markers": read_csv(paths["gate_markers"]) if os.path.exists(paths["gate_markers"]) else [],
        "redistribution": read_csv(paths["redistribution"]),
        "rank_fusion": read_csv(paths["rank_fusion"]) if os.path.exists(paths["rank_fusion"]) else [],
        "samples": read_csv(paths["samples"]) if os.path.exists(paths["samples"]) else [],
    }


def sample_caption_text(source, override=""):
    if override:
        return override
    for row in source.get("samples", []):
        caption = str(row.get("caption", "")).strip()
        if caption:
            return caption.replace("\\n", "\n")
    return ""


def load_all_head_ratio_rows(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing all-head NPZ: {path}")
    data = np.load(path, allow_pickle=True)
    labels = np.asarray(data["labels"]).astype(str)
    bounded = np.asarray(data["bounded_ratio"], dtype=np.float32)
    counterfactual_delta = np.asarray(data["counterfactual_delta"], dtype=np.float32)
    rows = []
    for label in ("grounded", "hallucinated"):
        mask = labels == label
        if not np.any(mask):
            continue
        ratios = bounded[mask].reshape(-1)
        deltas = counterfactual_delta[mask].reshape(-1)
        sample_cap = min(len(ratios), 250000)
        if len(ratios) > sample_cap:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(ratios), size=sample_cap, replace=False)
            ratios = ratios[idx]
            deltas = deltas[idx]
        rows.extend({"label": label, "bounded_ratio": float(r), "delta": float(d)} for r, d in zip(ratios, deltas))
    return rows


def selected_rows(head_rows):
    rows = [row for row in head_rows if as_int(row, "selected") == 1]
    return sorted(rows, key=lambda row: as_int(row, "rank"))


def in_layer_window(row, layers):
    return bool(layers) and as_int(row, "layer") in set(layers)


def source_top_k(summary, head_rows, fallback):
    if fallback:
        return int(fallback)
    if "top_k" in summary:
        return int(summary["top_k"])
    return len(selected_rows(head_rows))


def source_layers(summary, override):
    if override:
        return parse_layers(override)
    return parse_layers(summary.get("selection_layers", []))


def figure_text_mass_sorted(head_rows, output_dir, formats):
    ranked = sorted(head_rows, key=lambda row: as_float(row, "text_mass_all"), reverse=True)
    values = np.array([as_float(row, "text_mass_all") for row in ranked], dtype=np.float64)
    selected = np.array([as_int(row, "selected") == 1 for row in ranked], dtype=bool)
    x = np.arange(1, len(values) + 1)

    fig, ax = plt.subplots(figsize=(3.2, 1.75))
    ax.bar(x, values, width=1.0, color=COLORS["tail"], linewidth=0)
    if np.any(selected):
        ax.bar(x[selected], values[selected], width=1.0, color=COLORS["text"], linewidth=0)
    ax.set_xlim(0, len(values) + 1)
    ax.set_ylim(0, min(1.0, max(0.05, float(values.max()) * 1.08)))
    ax.set_title("Text-side leverage")
    ax.set_xlabel("all heads sorted by text-side mass")
    ax.set_ylabel(r"$I_{\mathrm{text}}$")
    ax.grid(axis="y")
    ax.text(
        0.98,
        0.92,
        f"selected: {int(selected.sum())} heads",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=COLORS["muted"],
    )
    return save_figure(fig, output_dir, "phase1_text_mass_sorted", formats)


def positive_contrast(row):
    return max(as_float(row, "raw_toi_gap_hall_minus_grounded"), 0.0)


def figure_contrastive_specificity_sorted(head_rows, output_dir, formats):
    ranked = sorted(head_rows, key=positive_contrast, reverse=True)
    raw_values = np.array([positive_contrast(row) for row in ranked], dtype=np.float64)
    values = np.log1p(raw_values)
    selected = np.array([as_int(row, "selected") == 1 for row in ranked], dtype=bool)
    x = np.arange(1, len(values) + 1)

    fig, ax = plt.subplots(figsize=(3.2, 1.75))
    ax.bar(x, values, width=1.0, color=COLORS["tail"], linewidth=0)
    if np.any(selected):
        ax.bar(x[selected], values[selected], width=1.0, color=COLORS["image"], linewidth=0)
    ax.set_xlim(0, len(values) + 1)
    ax.set_ylim(0, max(0.2, float(values.max()) * 1.08))
    ax.set_title("Contrastive specificity")
    ax.set_xlabel(r"all heads sorted by $C_{\mathrm{toi}}$")
    ax.set_ylabel(r"$\log(1+C_{\mathrm{toi}})$")
    ax.grid(axis="y")
    positive_rate = float(np.mean(raw_values > 0.0)) if raw_values.size else 0.0
    ax.text(
        0.98,
        0.92,
        f"positive heads: {positive_rate:.0%}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=COLORS["muted"],
    )
    return save_figure(fig, output_dir, "phase1_contrastive_specificity_sorted", formats)


def figure_ratio_distribution(ratio_rows, output_dir, formats):
    grouped = ratio_values(ratio_rows, "bounded_ratio")
    fig, ax = plt.subplots(figsize=(3.05, 1.85))
    bins = np.linspace(0.45, 1.0, 34)
    stats = {}
    for label, color in [("grounded", COLORS["ground"]), ("hallucinated", COLORS["hall"])]:
        values = grouped.get(label, np.array([], dtype=np.float64))
        values = values[np.isfinite(values)]
        values = values[(values >= 0.0) & (values <= 1.0)]
        if values.size == 0:
            continue
        ax.hist(
            values,
            bins=bins,
            density=True,
            histtype="stepfilled",
            alpha=0.36,
            color=color,
            edgecolor=color,
            linewidth=0.7,
            label=f"{label} (area=1)",
        )
        median = float(np.median(values))
        ax.axvline(median, color=color, linewidth=1.4)
        stats[label] = {"median": median, "mean": float(np.mean(values)), "n": int(values.size)}
    ax.set_xlim(0.45, 1.0)
    ax.set_title("Online text-ratio overlap")
    ax.set_xlabel(r"$r=T/(T+I)$")
    ax.set_ylabel("density (area=1)")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y")
    if "grounded" in stats and "hallucinated" in stats:
        gap = stats["hallucinated"]["median"] - stats["grounded"]["median"]
        ax.text(
            0.98,
            0.92,
            f"median gap={gap:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            color=COLORS["dark"],
        )
    paths = save_figure(fig, output_dir, "phase1_ratio_distribution", formats)
    return paths, stats


def figure_head_score_heatmap(head_rows, layers, output_dir, formats):
    max_layer = max(as_int(row, "layer") for row in head_rows)
    max_head = max(as_int(row, "head") for row in head_rows)
    matrix = np.zeros((max_layer + 1, max_head + 1), dtype=np.float64)
    selected = np.zeros_like(matrix, dtype=bool)
    for row in head_rows:
        layer, head = as_int(row, "layer"), as_int(row, "head")
        matrix[layer, head] = as_float(row, "score")
        selected[layer, head] = as_int(row, "selected") == 1

    fig, ax = plt.subplots(figsize=(3.75, 2.55))
    im = ax.imshow(matrix, aspect="auto", origin="lower", cmap="magma", vmin=0.0, vmax=1.0)
    if layers:
        start, end = min(layers), max(layers)
        ax.axhspan(start - 0.5, end + 0.5, color=COLORS["window"], alpha=0.18, zorder=1)
        ax.add_patch(
            Rectangle(
                (-0.5, start - 0.5),
                max_head + 1,
                end - start + 1,
                fill=False,
                edgecolor="#f59e0b",
                linewidth=1.0,
                zorder=3,
            )
        )
    ys, xs = np.where(selected)
    ax.scatter(xs, ys, s=7, facecolors="none", edgecolors="white", linewidths=0.45, zorder=4)
    ax.set_title(r"Fused head score $S(l,h)$")
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    ax.set_xticks([0, 8, 16, 24, max_head])
    ax.set_yticks([0, 4, 8, 12, 16, 20, 24, 28, max_layer])
    cbar = fig.colorbar(im, ax=ax, fraction=0.034, pad=0.02)
    cbar.set_label("score")
    return save_figure(fig, output_dir, "phase1_head_score_heatmap", formats)


def figure_final_head_selection_map(head_rows, layers, output_dir, formats):
    max_layer = max(as_int(row, "layer") for row in head_rows)
    max_head = max(as_int(row, "head") for row in head_rows)
    matrix = np.zeros((max_layer + 1, max_head + 1), dtype=np.float64)
    selected = np.zeros_like(matrix, dtype=bool)
    for row in head_rows:
        layer, head = as_int(row, "layer"), as_int(row, "head")
        selected[layer, head] = as_int(row, "selected") == 1
        if selected[layer, head]:
            matrix[layer, head] = as_float(row, "score")

    fig, ax = plt.subplots(figsize=(3.45, 2.25))
    im = ax.imshow(matrix, aspect="auto", origin="lower", cmap="Purples", vmin=0.0, vmax=1.0)
    if layers:
        start, end = min(layers), max(layers)
        ax.add_patch(
            Rectangle(
                (-0.5, start - 0.5),
                max_head + 1,
                end - start + 1,
                fill=False,
                edgecolor="#f59e0b",
                linewidth=1.0,
                zorder=3,
            )
        )
    ys, xs = np.where(selected)
    ax.scatter(xs, ys, s=9, facecolors="none", edgecolors=COLORS["dark"], linewidths=0.35)
    ax.set_title("Final selected head pool")
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    ax.set_xticks([0, 8, 16, 24, max_head])
    ax.set_yticks([0, 4, 8, 12, 16, 20, 24, 28, max_layer])
    cbar = fig.colorbar(im, ax=ax, fraction=0.034, pad=0.02)
    cbar.set_label("selected score")
    return save_figure(fig, output_dir, "phase1_final_head_selection_map", formats)


def figure_layer_score_profiles(head_rows, layers, output_dir, formats):
    max_layer = max(as_int(row, "layer") for row in head_rows)
    all_layers = list(range(max_layer + 1))
    selected = selected_rows(head_rows)
    grouped = defaultdict(lambda: {"score": 0.0, "text": 0.0, "contrast": 0.0, "count": 0})
    for row in selected:
        layer = as_int(row, "layer")
        grouped[layer]["score"] += as_float(row, "score")
        grouped[layer]["text"] += as_float(row, "text_percentile")
        grouped[layer]["contrast"] += as_float(row, "contrast_percentile")
        grouped[layer]["count"] += 1

    fig, ax = plt.subplots(figsize=(4.0, 1.95))
    if layers:
        ax.axvspan(min(layers) - 0.5, max(layers) + 0.5, color=COLORS["window"], alpha=0.65, zorder=0)
    ax.plot(all_layers, [grouped[layer]["score"] for layer in all_layers], color=COLORS["score"], marker="o", ms=2.5, lw=1.6, label="fused")
    ax.plot(all_layers, [0.5 * grouped[layer]["text"] for layer in all_layers], color=COLORS["text"], marker="o", ms=2.1, lw=1.15, label="0.5 text")
    ax.plot(all_layers, [0.5 * grouped[layer]["contrast"] for layer in all_layers], color=COLORS["image"], marker="o", ms=2.1, lw=1.15, label="0.5 contrast")
    ax.set_title("Per-layer selected score mass")
    ax.set_xlabel("layer")
    ax.set_ylabel("sum score")
    ax.set_xlim(-0.5, max_layer + 0.5)
    ax.grid(axis="y")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    paths_sum = save_figure(fig, output_dir, "phase1_layer_component_score_sum", formats)

    text_sum = np.array([grouped[layer]["text"] for layer in all_layers], dtype=np.float64)
    contrast_sum = np.array([grouped[layer]["contrast"] for layer in all_layers], dtype=np.float64)
    text_share = text_sum / max(float(text_sum.sum()), 1e-12)
    contrast_share = contrast_sum / max(float(contrast_sum.sum()), 1e-12)

    fig, ax = plt.subplots(figsize=(4.0, 1.9))
    if layers:
        ax.axvspan(min(layers) - 0.5, max(layers) + 0.5, color=COLORS["window"], alpha=0.65, zorder=0)
    ax.plot(all_layers, text_share, color=COLORS["text"], marker="o", ms=2.3, lw=1.45, label="text score share")
    ax.plot(all_layers, contrast_share, color=COLORS["image"], marker="o", ms=2.3, lw=1.45, label="contrast score share")
    ax.set_title("Per-layer component share")
    ax.set_xlabel("layer")
    ax.set_ylabel("share")
    ax.set_xlim(-0.5, max_layer + 0.5)
    ax.grid(axis="y")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    paths_share = save_figure(fig, output_dir, "phase1_layer_component_score_share", formats)
    return {"sum": paths_sum, "share": paths_share}


def layer_window_stats(head_rows, layers):
    layer_set = set(layers)
    window = [row for row in head_rows if as_int(row, "layer") in layer_set]
    outside = [row for row in head_rows if as_int(row, "layer") not in layer_set]
    metrics = {
        "score": "score",
        "text_percentile": "text_percentile",
        "contrast_percentile": "contrast_percentile",
        "text_mass_all": "text_mass_all",
    }

    def summarize(rows, prefix):
        out = {
            f"{prefix}_n_heads": len(rows),
            f"{prefix}_selected_fraction": mean(as_int(row, "selected") for row in rows),
        }
        for name, key in metrics.items():
            values = np.array([as_float(row, key) for row in rows], dtype=np.float64)
            out[f"{prefix}_mean_{name}"] = float(np.mean(values)) if values.size else 0.0
            out[f"{prefix}_q90_{name}"] = float(np.quantile(values, 0.9)) if values.size else 0.0
        return out

    return {**summarize(window, "window"), **summarize(outside, "outside")}


def figure_layer_window_comparison(head_rows, layers, output_dir, formats):
    if not layers:
        return {}, {}
    layer_set = set(layers)
    groups = {
        f"L{min(layers)}-L{max(layers)}": [row for row in head_rows if as_int(row, "layer") in layer_set],
        "other layers": [row for row in head_rows if as_int(row, "layer") not in layer_set],
    }
    metrics = [
        ("score", "fused\nscore", COLORS["score"]),
        ("text_percentile", "text\nrank", COLORS["text"]),
        ("contrast_percentile", "contrast\nrank", COLORS["image"]),
        ("text_mass_all", r"$I_{text}$", COLORS["muted"]),
    ]
    labels = list(groups)
    x = np.arange(len(metrics))
    width = 0.34

    fig, ax = plt.subplots(figsize=(3.7, 1.9))
    for group_idx, group in enumerate(labels):
        rows = groups[group]
        vals = [mean(as_float(row, key) for row in rows) for key, _, _ in metrics]
        color = COLORS["dark"] if group_idx == 0 else COLORS["tail"]
        ax.bar(x + (group_idx - 0.5) * width, vals, width, color=color, alpha=0.88, label=group)
    for tick, (_, label, color) in zip(x, metrics):
        ax.text(tick, -0.14, label, transform=ax.get_xaxis_transform(), ha="center", va="top", color=color)
    ax.set_xticks([])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("mean value")
    ax.set_title("Layer window vs. other layers")
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="upper left")
    paths = save_figure(fig, output_dir, "phase1_layer_window_comparison", formats)
    return paths, layer_window_stats(head_rows, layers)


def figure_rank_fusion(head_rows, rank_fusion_rows, output_dir, formats):
    if not rank_fusion_rows:
        selected = selected_rows(head_rows)
        tail = [row for row in head_rows if as_int(row, "selected") != 1]
        rank_fusion_rows = [
            {
                "bucket": "selected",
                "mean_text_percentile": mean(as_float(row, "text_percentile") for row in selected),
                "mean_contrast_percentile": mean(as_float(row, "contrast_percentile") for row in selected),
                "mean_score": mean(as_float(row, "score") for row in selected),
            },
            {
                "bucket": "tail",
                "mean_text_percentile": mean(as_float(row, "text_percentile") for row in tail),
                "mean_contrast_percentile": mean(as_float(row, "contrast_percentile") for row in tail),
                "mean_score": mean(as_float(row, "score") for row in tail),
            },
        ]
    by_bucket = {row["bucket"]: row for row in rank_fusion_rows}
    buckets = [bucket for bucket in ("selected", "tail") if bucket in by_bucket]
    metrics = [
        ("mean_text_percentile", "text", COLORS["text"]),
        ("mean_contrast_percentile", "contrast", COLORS["image"]),
        ("mean_score", "fused", COLORS["score"]),
    ]
    x = np.arange(len(metrics))
    width = 0.32

    fig, ax = plt.subplots(figsize=(3.25, 1.85))
    for idx, bucket in enumerate(buckets):
        vals = [as_float(by_bucket[bucket], key) for key, _, _ in metrics]
        color = COLORS["dark"] if bucket == "selected" else COLORS["tail"]
        alpha = 0.88 if bucket == "selected" else 0.72
        ax.bar(x + (idx - 0.5) * width, vals, width, color=color, alpha=alpha, label=bucket)
    for tick, (_, label, color) in zip(x, metrics):
        ax.text(tick, -0.12, label, transform=ax.get_xaxis_transform(), ha="center", va="top", color=color)
    ax.set_xticks([])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("rank percentile")
    ax.set_title("Rank fusion keeps both axes high")
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="upper left")
    return save_figure(fig, output_dir, "phase1_rank_fusion_bars", formats)


def figure_rank_fusion_scatter(head_rows, output_dir, formats, layers):
    layer_set = set(layers)
    selected = [row for row in head_rows if as_int(row, "selected") == 1]
    window = [row for row in head_rows if as_int(row, "layer") in layer_set and as_int(row, "selected") != 1]
    outside = [row for row in head_rows if as_int(row, "layer") not in layer_set]

    fig, ax = plt.subplots(figsize=(3.05, 2.35))
    for rows, color, alpha, size, label, zorder in [
        (outside, COLORS["tail"], 0.36, 10, "other layers", 1),
        (window, "#fbbf24", 0.45, 12, "L-window not selected", 2),
        (selected, COLORS["score"], 0.88, 18, "selected", 3),
    ]:
        if not rows:
            continue
        ax.scatter(
            [as_float(row, "text_percentile") for row in rows],
            [as_float(row, "contrast_percentile") for row in rows],
            s=size,
            color=color,
            alpha=alpha,
            linewidths=0,
            label=label,
            zorder=zorder,
        )
    ax.axvline(0.75, color=COLORS["grid"], linestyle="--", linewidth=0.9)
    ax.axhline(0.75, color=COLORS["grid"], linestyle="--", linewidth=0.9)
    ax.text(0.93, 0.18, "text-heavy\nonly", ha="right", va="center", color=COLORS["text"], fontsize=6.6)
    ax.text(0.93, 0.92, "high leverage\n+ specificity", ha="right", va="top", color=COLORS["score"], fontsize=6.6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("text leverage rank")
    ax.set_ylabel("contrastive rank")
    ax.set_title("Text leverage vs. contrastive specificity")
    ax.grid(True)
    ax.legend(frameon=False, loc="lower left", handletextpad=0.1, borderpad=0.1)
    return save_figure(fig, output_dir, "phase1_rank_fusion_scatter", formats)


def ratio_values(rows, key):
    grouped = defaultdict(list)
    for row in rows:
        label = row.get("label", "")
        if label:
            grouped[label].append(as_float(row, key))
    return {label: np.array(values, dtype=np.float64) for label, values in grouped.items()}


def figure_gate_curve(gate_rows, marker_rows, output_dir, formats):
    xs = np.array([as_float(row, "r_online") for row in gate_rows], dtype=np.float64)
    ys = np.array([as_float(row, "gate") for row in gate_rows], dtype=np.float64)
    if xs.size == 0:
        return {}
    beta = as_float(gate_rows[0], "beta", 10.0)
    tau = as_float(gate_rows[0], "tau", 0.9)
    markers = {row["label"]: row for row in marker_rows}

    fig, ax = plt.subplots(figsize=(3.0, 1.9))
    ax.plot(xs, ys, color=COLORS["score"], linewidth=2.0)
    ax.axvline(tau, color=COLORS["hall"], linestyle="--", linewidth=1.0)
    ax.text(tau + 0.004, max(ys) * 0.08, rf"$\tau={tau:.2f}$", color=COLORS["hall"], fontsize=6.8)
    for label, color in [("grounded", COLORS["ground"]), ("hallucinated", COLORS["hall"])]:
        row = markers.get(label)
        if not row:
            continue
        r = as_float(row, "bounded_ratio_median")
        g = as_float(row, "gate_at_median")
        ax.scatter([r], [g], s=24, color=color, edgecolors="white", linewidths=0.5, zorder=4, label=f"{label} median")
    ax.set_title("Exponential online gate")
    ax.set_xlabel(r"$r_{\mathrm{online}}=T/(T+I)$")
    ax.set_ylabel(r"$g=\exp(q(r-\tau))$")
    ax.set_xlim(max(0.45, float(xs.min())), min(1.0, float(xs.max())))
    ax.set_ylim(0, max(1.0, float(ys.max()) * 1.08))
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="upper left")
    if "grounded" in markers and "hallucinated" in markers:
        gap_r = as_float(markers["hallucinated"], "bounded_ratio_median") - as_float(markers["grounded"], "bounded_ratio_median")
        gap_g = as_float(markers["hallucinated"], "gate_at_median") - as_float(markers["grounded"], "gate_at_median")
        ax.text(
            0.98,
            0.88,
            rf"$\Delta r={gap_r:.3f}$" + "\n" + rf"$\Delta g={gap_g:.2f}$",
            transform=ax.transAxes,
            ha="right",
            va="top",
            color=COLORS["dark"],
        )
    return save_figure(fig, output_dir, "phase2_gate_curve", formats)


def figure_attention_redistribution(redistribution_rows, output_dir, formats):
    by_label = {row["label"]: row for row in redistribution_rows}
    labels = [label for label in ("grounded", "hallucinated") if label in by_label]
    x = np.arange(len(labels) * 2)
    names = []
    bars = []
    for label in labels:
        row = by_label[label]
        names.append(f"{label}\nbefore")
        bars.append((label, as_float(row, "system_before"), as_float(row, "image_before"), as_float(row, "text_before")))
        names.append(f"{label}\nafter")
        bars.append((label, as_float(row, "system_after"), as_float(row, "image_after"), as_float(row, "text_after")))
    values = np.array([bar[1:] for bar in bars], dtype=np.float64) if bars else np.zeros((0, 3))
    edgecolors = [COLORS["hall"] if bar[0] == "hallucinated" else COLORS["ground"] for bar in bars]

    fig, ax = plt.subplots(figsize=(3.35, 1.95))
    bottom = np.zeros(len(values))
    for idx, (key, color, label) in enumerate(
        [
            ("system", COLORS["system"], "system"),
            ("image", COLORS["image"], "visual"),
            ("text", COLORS["text"], "text-side"),
        ]
    ):
        vals = values[:, idx] if len(values) else []
        ax.bar(x, vals, bottom=bottom, color=color, width=0.62, alpha=0.86, label=label)
        bottom = bottom + vals
    for xpos, edgecolor in zip(x, edgecolors):
        ax.add_patch(Rectangle((xpos - 0.31, 0), 0.62, 1.0, fill=False, edgecolor=edgecolor, linewidth=1.0))
    for idx in range(1, len(x), 2):
        ax.axvline(idx - 0.5, color=COLORS["grid"], linewidth=0.8)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("attention share")
    ax.set_title("Suppress text-side, renormalize")
    ax.grid(axis="y")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.23))
    return save_figure(fig, output_dir, "phase2_attention_redistribution", formats)


def figure_delta_distribution(ratio_rows, output_dir, formats):
    grouped = ratio_values(ratio_rows, "delta")
    if not grouped:
        grouped = ratio_values(ratio_rows, "applied_delta")
    fig, ax = plt.subplots(figsize=(2.75, 1.8))
    bins = np.linspace(0.0, 1.0, 31)
    stats = {}
    for label, color in [("grounded", COLORS["ground"]), ("hallucinated", COLORS["hall"])]:
        values = grouped.get(label, np.array([], dtype=np.float64))
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        ax.hist(values, bins=bins, density=True, alpha=0.38, color=color, label=label)
        mean_value = float(np.mean(values))
        ax.axvline(mean_value, color=color, linewidth=1.35)
        stats[label] = {"mean": mean_value, "median": float(np.median(values)), "n": int(values.size)}
    ax.set_xlim(0, 1)
    ax.set_title(r"Applied suppression $\delta$")
    ax.set_xlabel(r"$\delta$")
    ax.set_ylabel("density")
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="upper right")
    return save_figure(fig, output_dir, "phase2_delta_distribution", formats), stats


def figure_delta_flow(head_rows, gate_markers, output_dir, formats, beta, tau):
    selected = selected_rows(head_rows)
    mean_score = mean(as_float(row, "score") for row in selected)
    markers = {row["label"]: row for row in gate_markers}
    hall_r = as_float(markers.get("hallucinated", {}), "bounded_ratio_median", tau)
    gate = math.exp(beta * (hall_r - tau))
    delta = min(max(mean_score * gate, 0.0), 1.0)

    fig, ax = plt.subplots(figsize=(4.2, 1.18))
    ax.axis("off")
    blocks = [
        (0.03, "offline score", rf"$S={mean_score:.2f}$", COLORS["score"]),
        (0.29, "online ratio", rf"$r={hall_r:.3f}$", COLORS["text"]),
        (0.55, "gate", rf"$g={gate:.2f}$", COLORS["image"]),
        (0.80, "action", rf"$\delta={delta:.2f}$", COLORS["hall"]),
    ]
    for x0, title, value, color in blocks:
        ax.add_patch(Rectangle((x0, 0.28), 0.17, 0.44, facecolor="white", edgecolor=color, linewidth=1.15))
        ax.text(x0 + 0.085, 0.58, title, ha="center", va="center", color=COLORS["muted"], fontsize=6.4)
        ax.text(x0 + 0.085, 0.42, value, ha="center", va="center", color=COLORS["dark"], fontsize=8.0)
    for x0, x1 in [(0.20, 0.29), (0.46, 0.55), (0.72, 0.80)]:
        ax.annotate("", xy=(x1 - 0.012, 0.50), xytext=(x0 + 0.012, 0.50), arrowprops=dict(arrowstyle="->", lw=1.0, color=COLORS["muted"]))
    ax.text(0.5, 0.93, r"$\delta=\mathrm{clip}(S(l,h) \cdot \exp(q(r-\tau)),0,1)$", ha="center", va="top", color=COLORS["dark"], fontsize=7.6)
    ax.text(0.5, 0.10, "text-side attention is multiplied by (1-delta), then the row is renormalized", ha="center", va="bottom", color=COLORS["muted"], fontsize=6.3)
    return save_figure(fig, output_dir, "phase2_delta_flow", formats)


def figure_overview_panel(source, output_dir, formats, layers):
    head_rows = source["head_rows"]
    ratio_rows = source["ratio_rows"]
    gate_rows = source["gate_curve"]
    marker_rows = source["gate_markers"]
    redistribution_rows = source["redistribution"]

    fig = plt.figure(figsize=(9.4, 4.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.92], wspace=0.46, hspace=0.72)
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[0, 3]),
        fig.add_subplot(gs[1, 0:2]),
        fig.add_subplot(gs[1, 2:4]),
    ]
    plot_text_mass_on_axis(axes[0], head_rows)
    plot_ratio_on_axis(axes[1], ratio_rows)
    plot_heatmap_on_axis(axes[2], head_rows, layers)
    plot_rank_fusion_scatter_on_axis(axes[3], head_rows, layers)
    plot_gate_on_axis(axes[4], gate_rows, marker_rows)
    plot_redistribution_on_axis(axes[5], redistribution_rows)
    fig.text(0.5, 0.985, "Phase 1: head attribution", ha="center", va="top", color=COLORS["dark"], fontsize=9.2, weight="bold")
    fig.text(0.5, 0.465, "Phase 2: dynamic suppression", ha="center", va="top", color=COLORS["dark"], fontsize=9.2, weight="bold")
    fig.add_artist(plt.Line2D([0.03, 0.97], [0.955, 0.955], transform=fig.transFigure, color=COLORS["grid"], linewidth=1.0))
    fig.add_artist(plt.Line2D([0.03, 0.97], [0.435, 0.435], transform=fig.transFigure, color=COLORS["grid"], linewidth=1.0))
    return save_figure(fig, output_dir, "method_phase_overview_panel", formats)


def figure_method_claim_panel(source, output_dir, formats, layers, caption_text):
    head_rows = source["head_rows"]
    gate_rows = source["gate_curve"]
    marker_rows = source["gate_markers"]

    fig = plt.figure(figsize=(9.8, 4.45))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.86], wspace=0.50, hspace=0.78)
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[0, 3]),
        fig.add_subplot(gs[1, 0:2]),
        fig.add_subplot(gs[1, 2:4]),
    ]
    plot_offline_selection_formula_on_axis(axes[0], head_rows, layers)
    plot_high_text_head_on_axis(axes[1], head_rows)
    plot_contrastive_head_on_axis(axes[2], head_rows)
    plot_head_selection_score_on_axis(axes[3], head_rows, layers)
    plot_gate_on_axis(axes[4], gate_rows, marker_rows, show_formula=True)
    plot_caption_or_formula_on_axis(axes[5], caption_text)
    fig.text(0.5, 0.985, "Phase 1: offline head-pool selection", ha="center", va="top", color=COLORS["dark"], fontsize=9.2, weight="bold")
    fig.text(0.5, 0.455, "Phase 2: dynamic suppression", ha="center", va="top", color=COLORS["dark"], fontsize=9.2, weight="bold")
    fig.add_artist(plt.Line2D([0.03, 0.97], [0.955, 0.955], transform=fig.transFigure, color=COLORS["grid"], linewidth=1.0))
    fig.add_artist(plt.Line2D([0.03, 0.97], [0.427, 0.427], transform=fig.transFigure, color=COLORS["grid"], linewidth=1.0))
    return save_figure(fig, output_dir, "method_claim_compact_panel", formats)


SVG = {
    "bg": "#141412",
    "panel": "#1d1d1a",
    "panel2": "#24231f",
    "stroke": "#4b4a43",
    "muted": "#a5a197",
    "text": "#f7f2e8",
    "blue": "#78aee8",
    "orange": "#f09a72",
    "purple": "#a98bff",
    "green": "#55c7a5",
    "red": "#ea6b6b",
    "yellow": "#f6c95d",
    "gray": "#7f8794",
}


def svg_attr(value):
    return xml_escape(str(value), {'"': "&quot;"})


def svg_text(x, y, value, size=9, fill=None, weight="", anchor="start", family="Inter,Arial,sans-serif"):
    fill = fill or SVG["text"]
    weight_attr = f' font-weight="{svg_attr(weight)}"' if weight else ""
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" fill="{fill}" font-family="{family}" '
        f'font-size="{size}" text-anchor="{anchor}"{weight_attr}>{xml_escape(str(value))}</text>'
    )


def svg_multiline(x, y, lines, size=8, fill=None, line_height=11, weight="", anchor="start"):
    out = []
    for idx, line in enumerate(lines):
        out.append(svg_text(x, y + idx * line_height, line, size=size, fill=fill, weight=weight, anchor=anchor))
    return "\n".join(out)


def svg_rect(x, y, w, h, fill, stroke=None, rx=8, width=1.0, dash=""):
    stroke_attr = f' stroke="{stroke}" stroke-width="{width}"' if stroke else ""
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" rx="{rx}" fill="{fill}"{stroke_attr}{dash_attr}/>'


def svg_line(x1, y1, x2, y2, stroke, width=1.0, dash="", marker=False):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    marker_attr = ' marker-end="url(#arrow)"' if marker else ""
    return (
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
        f'stroke="{stroke}" stroke-width="{width}"{dash_attr}{marker_attr}/>'
    )


def svg_polyline(points, stroke, width=1.5, fill="none"):
    payload = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polyline points="{payload}" fill="{fill}" stroke="{stroke}" stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"/>'


def teaser_card(x, y, w, h, title, lines, color, title_size=8.2):
    out = [
        svg_rect(x, y, w, h, SVG["panel2"], color, rx=7, width=1.2),
        svg_text(x + 9, y + 14, title, size=title_size, fill=color, weight="700"),
    ]
    out.append(svg_multiline(x + 9, y + 29, lines, size=7.2, fill=SVG["text"], line_height=10))
    return "\n".join(out)


def teaser_stacked_bar(x, y, w, h, parts):
    out = []
    cursor = y + h
    for value, color, label in parts:
        seg_h = max(0.0, h * value)
        cursor -= seg_h
        out.append(svg_rect(x, cursor, w, seg_h, color, stroke=None, rx=1))
        if seg_h > 10 and label:
            out.append(svg_text(x + w / 2, cursor + seg_h / 2 + 3, label, size=6.5, fill="#141412", weight="700", anchor="middle"))
    out.append(svg_rect(x, y, w, h, "none", SVG["stroke"], rx=2, width=0.7))
    return "\n".join(out)


def teaser_layer_strip(head_rows, layers, x, y, w, h):
    by_layer = defaultdict(float)
    selected_counts = defaultdict(int)
    selected_scores = defaultdict(float)
    for row in head_rows:
        layer = as_int(row, "layer")
        by_layer[layer] += as_float(row, "score")
        if as_int(row, "selected") == 1:
            selected_counts[layer] += 1
            selected_scores[layer] += as_float(row, "score")
    max_layer = max(by_layer) if by_layer else 31
    values = [selected_scores.get(layer, 0.0) for layer in range(max_layer + 1)]
    max_value = max(values) if values else 1.0
    max_value = max(max_value, 1e-6)
    out = [svg_rect(x, y, w, h, "#181815", SVG["stroke"], rx=7, width=0.9)]
    if layers:
        start, end = min(layers), max(layers)
        cell = w / (max_layer + 1)
        out.append(svg_rect(x + start * cell, y + 2, (end - start + 1) * cell, h - 4, "#32291c", SVG["yellow"], rx=4, width=0.8))
    for layer, value in enumerate(values):
        cell = w / (max_layer + 1)
        bar_h = (h - 18) * value / max_value
        bx = x + layer * cell + 1.0
        by = y + h - 8 - bar_h
        color = SVG["purple"] if selected_counts.get(layer, 0) else "#34342f"
        out.append(svg_rect(bx, by, max(1.5, cell - 2.0), bar_h, color, stroke=None, rx=1))
    out.append(svg_text(x + 8, y + 14, "Selected score mass by layer", size=7.4, fill=SVG["muted"], weight="700"))
    if layers:
        out.append(svg_text(x + w - 8, y + 14, f"actuation window L{min(layers)}-L{max(layers)}", size=7.4, fill=SVG["yellow"], weight="700", anchor="end"))
    return "\n".join(out)


def teaser_gate_curve(gate_rows, marker_rows, x, y, w, h):
    rows = [
        row
        for row in gate_rows
        if math.isfinite(as_float(row, "r_online")) and math.isfinite(as_float(row, "gate"))
    ]
    if not rows:
        return ""
    x_min, x_max = 0.5, 1.0
    y_max = max(1.0, max(as_float(row, "gate") for row in rows) * 1.04)

    def sx(value):
        return x + (float(value) - x_min) / (x_max - x_min) * w

    def sy(value):
        return y + h - float(value) / y_max * h

    points = [(sx(as_float(row, "r_online")), sy(as_float(row, "gate"))) for row in rows]
    tau = as_float(rows[0], "tau", 0.9)
    out = [
        svg_rect(x - 8, y - 20, w + 16, h + 36, "#181815", SVG["stroke"], rx=7, width=0.9),
        svg_text(x, y - 7, "Online gate", size=8, fill=SVG["text"], weight="700"),
        svg_line(x, y + h, x + w, y + h, SVG["stroke"], width=0.8),
        svg_line(x, y, x, y + h, SVG["stroke"], width=0.8),
        svg_polyline(points, SVG["purple"], width=2.0),
        svg_line(sx(tau), y, sx(tau), y + h, SVG["red"], width=1.0, dash="3 3"),
        svg_text(sx(tau) + 4, y + 10, "tau=0.9", size=6.5, fill=SVG["red"]),
    ]
    for row in marker_rows:
        label = row.get("label", "")
        color = SVG["red"] if label == "hallucinated" else SVG["green"]
        px = sx(as_float(row, "bounded_ratio_median"))
        py = sy(as_float(row, "gate_at_median"))
        out.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="3.3" fill="{color}" stroke="#ffffff" stroke-width="0.8"/>')
        out.append(svg_text(px + 5, py - 4, "hall" if label == "hallucinated" else "ground", size=6.2, fill=color, weight="700"))
    out.append(svg_text(x + w / 2, y + h + 13, "r = T / (T + I)", size=6.6, fill=SVG["muted"], anchor="middle"))
    return "\n".join(out)


def teaser_redistribution(redistribution_rows, x, y, w, h):
    rows = {row.get("label", ""): row for row in redistribution_rows}
    out = [svg_rect(x, y, w, h, "#181815", SVG["stroke"], rx=7, width=0.9)]
    out.append(svg_text(x + 10, y + 15, "Suppression effect", size=8, fill=SVG["text"], weight="700"))
    pairs = [("hallucinated", SVG["red"], x + 14), ("grounded", SVG["green"], x + w / 2 + 9)]
    for label, color, bx in pairs:
        row = rows.get(label)
        if not row:
            continue
        title = "hallucinated step" if label == "hallucinated" else "grounded step"
        out.append(svg_text(bx, y + 32, title, size=7.2, fill=color, weight="700"))
        before = [
            (as_float(row, "system_before"), SVG["gray"], ""),
            (as_float(row, "image_before"), SVG["blue"], "I"),
            (as_float(row, "text_before"), SVG["orange"], "T"),
        ]
        after = [
            (as_float(row, "system_after"), SVG["gray"], ""),
            (as_float(row, "image_after"), SVG["blue"], "I"),
            (as_float(row, "text_after"), SVG["orange"], "T"),
        ]
        out.append(teaser_stacked_bar(bx, y + 43, 26, 44, before))
        out.append(teaser_stacked_bar(bx + 48, y + 43, 26, 44, after))
        out.append(svg_line(bx + 30, y + 65, bx + 44, y + 65, SVG["muted"], width=1.0, marker=True))
        out.append(svg_text(bx + 13, y + 97, "before", size=6.2, fill=SVG["muted"], anchor="middle"))
        out.append(svg_text(bx + 61, y + 97, "after", size=6.2, fill=SVG["muted"], anchor="middle"))
        delta = as_float(row, "delta_median")
        out.append(svg_text(bx + 4, y + h - 7, f"median delta={delta:.2f}", size=6.4, fill=color))
    return "\n".join(out)


def figure_method_teaser_dark_svg(source, output_dir, layers, caption_text=""):
    os.makedirs(output_dir, exist_ok=True)
    head_rows = source["head_rows"]
    selected = selected_rows(head_rows)
    top_k = len(selected)
    layer_text = f"L{min(layers)}-L{max(layers)}" if layers else "all layers"
    text_head = text_head_example(head_rows)
    contrast_head = contrastive_head_example(head_rows)
    high_t_ground = as_float(text_head, "text_mass_grounded")
    high_t_hall = as_float(text_head, "text_mass_hallucinated")
    contrast_ground = as_float(contrast_head, "bounded_ratio_grounded")
    contrast_hall = as_float(contrast_head, "bounded_ratio_hallucinated")

    path = os.path.join(output_dir, "method_teaser_dark.svg")
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1444" height="884" viewBox="0 0 722 442">',
        "<defs>",
        '<marker id="arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">',
        f'<path d="M0,0 L7,3.5 L0,7 Z" fill="{SVG["muted"]}"/>',
        "</marker>",
        "</defs>",
        svg_rect(0, 0, 722, 442, SVG["bg"], stroke=None, rx=0),
        svg_rect(8, 8, 706, 196, "#171714", SVG["stroke"], rx=10, width=1.0, dash="4 4"),
        svg_text(18, 23, "Phase 1: offline head-pool selection", size=10, fill=SVG["text"], weight="800"),
        svg_text(430, 23, "Choose actuators: leverage + hallucination specificity", size=7.4, fill=SVG["muted"]),
        teaser_card(
            18,
            36,
            138,
            55,
            "Text-side mass",
            ["I_text = E sum_T A", f"example {text_head.get('head_key', '')}: T={max(high_t_ground, high_t_hall):.2f}", "leverage signal"],
            SVG["blue"],
        ),
        teaser_card(
            171,
            36,
            148,
            55,
            "Contrastive bias",
            ["C_toi = E_H r - E_G r", f"example {contrast_head.get('head_key', '')}: d r={contrast_hall - contrast_ground:.2f}", "specificity signal"],
            SVG["orange"],
        ),
        teaser_card(
            80,
            109,
            158,
            45,
            "Rank fusion",
            ["S(l,h)=0.5 P(I_text)", "+ 0.5 P(C_toi)"],
            SVG["purple"],
            title_size=7.8,
        ),
        teaser_card(
            258,
            109,
            118,
            45,
            "Head pool",
            [f"Top-{top_k}", f"restricted to {layer_text}"],
            SVG["yellow"],
            title_size=7.8,
        ),
        svg_line(94, 91, 130, 109, SVG["muted"], width=1.0, marker=True),
        svg_line(242, 91, 190, 109, SVG["muted"], width=1.0, marker=True),
        svg_line(238, 132, 258, 132, SVG["muted"], width=1.0, marker=True),
        svg_rect(402, 34, 300, 126, "#1a1a17", SVG["stroke"], rx=9, width=0.9),
        svg_text(414, 50, "Two concrete head signals", size=8.6, fill=SVG["text"], weight="800"),
        svg_text(416, 69, f"High-text head {text_head.get('head_key', '')}", size=7.0, fill=SVG["blue"], weight="700"),
        svg_text(545, 69, f"Contrastive head {contrast_head.get('head_key', '')}", size=7.0, fill=SVG["orange"], weight="700"),
    ]
    # Mini high-text stacked bars.
    text_parts_ground = [
        (max(0.0, 1.0 - high_t_ground - as_float(text_head, "image_mass_grounded")), SVG["gray"], ""),
        (as_float(text_head, "image_mass_grounded"), SVG["blue"], "I"),
        (high_t_ground, SVG["orange"], "T"),
    ]
    text_parts_hall = [
        (max(0.0, 1.0 - high_t_hall - as_float(text_head, "image_mass_hallucinated")), SVG["gray"], ""),
        (as_float(text_head, "image_mass_hallucinated"), SVG["blue"], "I"),
        (high_t_hall, SVG["orange"], "T"),
    ]
    parts.extend(
        [
            teaser_stacked_bar(420, 80, 24, 58, text_parts_ground),
            teaser_stacked_bar(456, 80, 24, 58, text_parts_hall),
            svg_text(432, 150, "G", size=6.4, fill=SVG["green"], anchor="middle", weight="700"),
            svg_text(468, 150, "H", size=6.4, fill=SVG["red"], anchor="middle", weight="700"),
        ]
    )
    # Mini contrastive bars.
    bar_max = 1.0
    for idx, (label, value, color) in enumerate([("G", contrast_ground, SVG["green"]), ("H", contrast_hall, SVG["red"])]):
        bx = 560 + idx * 46
        bh = 58 * value / bar_max
        parts.append(svg_rect(bx, 138 - bh, 28, bh, color, stroke=None, rx=3))
        parts.append(svg_rect(bx, 80, 28, 58, "none", SVG["stroke"], rx=3, width=0.7))
        parts.append(svg_text(bx + 14, 150, label, size=6.4, fill=color, anchor="middle", weight="700"))
        parts.append(svg_text(bx + 14, 75, f"{value:.2f}", size=6.2, fill=color, anchor="middle", weight="700"))
    parts.append(teaser_layer_strip(head_rows, layers, 18, 166, 684, 28))

    parts.extend(
        [
            svg_rect(8, 218, 706, 214, "#171714", SVG["stroke"], rx=10, width=1.0, dash="4 4"),
            svg_text(18, 234, "Phase 2: dynamic suppression", size=10, fill=SVG["text"], weight="800"),
            svg_text(444, 234, "Offline score gates online text reliance continuously", size=7.4, fill=SVG["muted"]),
        ]
    )
    flow_y = 248
    cards = [
        (20, "Attention head", ["A_t,l,h"]),
        (154, "Text ratio", ["r = T/(T+I)"]),
        (288, "Exp gate", ["g = exp(q(r-tau))"]),
        (422, "Strength", ["delta = clip(S*g,0,1)"]),
        (556, "Action", ["A_T *= 1-delta", "renormalize"]),
    ]
    for x, title, lines in cards:
        parts.append(teaser_card(x, flow_y, 118, 56, title, lines, SVG["purple"] if title in {"Exp gate", "Strength"} else SVG["blue"], title_size=7.6))
    for x1, x2 in [(138, 154), (272, 288), (406, 422), (540, 556)]:
        parts.append(svg_line(x1, flow_y + 28, x2, flow_y + 28, SVG["muted"], width=1.0, marker=True))
    parts.append(teaser_gate_curve(source["gate_curve"], source["gate_markers"], 28, 340, 220, 62))
    parts.append(teaser_redistribution(source["redistribution"], 286, 318, 404, 104))
    parts.append(svg_text(32, 428, "Method: suppress only text-side attention in selected heads, then proportionally renormalize.", size=7.2, fill=SVG["muted"]))
    if caption_text:
        clipped = textwrap.shorten(" ".join(caption_text.split()), width=88, placeholder="...")
        parts.append(svg_text(386, 428, f"example caption: {clipped}", size=6.8, fill=SVG["muted"]))
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    return {"svg": path}


def plot_text_mass_on_axis(ax, head_rows):
    ranked = sorted(head_rows, key=lambda row: as_float(row, "text_mass_all"), reverse=True)
    values = np.array([as_float(row, "text_mass_all") for row in ranked], dtype=np.float64)
    selected = np.array([as_int(row, "selected") == 1 for row in ranked], dtype=bool)
    x = np.arange(len(values))
    ax.bar(x, values, width=1.0, color=COLORS["tail"], linewidth=0)
    if np.any(selected):
        ax.bar(x[selected], values[selected], width=1.0, color=COLORS["text"], linewidth=0)
    ax.set_title("A. Text leverage")
    ax.set_xlabel("heads sorted")
    ax.set_ylabel(r"$I_{\mathrm{text}}$")
    ax.set_xlim(0, len(values))
    ax.grid(axis="y")


def plot_contrastive_specificity_on_axis(ax, head_rows):
    ranked = sorted(head_rows, key=positive_contrast, reverse=True)
    raw_values = np.array([positive_contrast(row) for row in ranked], dtype=np.float64)
    values = np.log1p(raw_values)
    selected = np.array([as_int(row, "selected") == 1 for row in ranked], dtype=bool)
    x = np.arange(len(values))
    ax.bar(x, values, width=1.0, color=COLORS["tail"], linewidth=0)
    if np.any(selected):
        ax.bar(x[selected], values[selected], width=1.0, color=COLORS["image"], linewidth=0)
    ax.set_title("B. Contrastive specificity")
    ax.set_xlabel("heads sorted")
    ax.set_ylabel(r"$\log(1+C_{\mathrm{toi}})$")
    ax.set_xlim(0, len(values))
    ax.grid(axis="y")


def plot_offline_selection_formula_on_axis(ax, head_rows, layers):
    selected = selected_rows(head_rows)
    top_k = len(selected)
    layer_text = f"L{min(layers)}-L{max(layers)}" if layers else "all layers"
    ax.axis("off")
    ax.set_title("A. Offline score")
    boxes = [
        (
            0.03,
            0.68,
            r"$I_{\mathrm{text}}(l,h)=\mathbb{E}_t\sum_{j\in T}A_{t,l,h,j}$",
            "text leverage",
            COLORS["text"],
        ),
        (
            0.03,
            0.43,
            r"$C_{\mathrm{toi}}(l,h)=\mathbb{E}_{H}r-\mathbb{E}_{G}r,\quad r=\frac{T}{I+\epsilon}$",
            "hallucination contrast",
            COLORS["image"],
        ),
        (
            0.03,
            0.18,
            r"$S(l,h)=\frac{1}{2}P(I_{\mathrm{text}})+\frac{1}{2}P(C_{\mathrm{toi}})$",
            f"Top-{top_k} in {layer_text}",
            COLORS["score"],
        ),
    ]
    for x, y, formula, label, color in boxes:
        ax.add_patch(Rectangle((x, y), 0.94, 0.19, transform=ax.transAxes, facecolor="white", edgecolor=color, linewidth=1.0))
        ax.text(x + 0.03, y + 0.125, formula, ha="left", va="center", color=COLORS["dark"], fontsize=6.4, transform=ax.transAxes)
        ax.text(x + 0.03, y + 0.045, label, ha="left", va="center", color=color, fontsize=5.9, transform=ax.transAxes)
    for y0, y1 in [(0.68, 0.62), (0.43, 0.37)]:
        ax.annotate("", xy=(0.50, y1), xytext=(0.50, y0), xycoords=ax.transAxes, arrowprops=dict(arrowstyle="->", lw=0.9, color=COLORS["muted"]))


def text_head_example(head_rows):
    candidates = selected_rows(head_rows) or head_rows
    return max(candidates, key=lambda row: as_float(row, "text_mass_all"))


def contrastive_head_example(head_rows):
    candidates = selected_rows(head_rows) or head_rows
    return max(
        candidates,
        key=lambda row: as_float(row, "bounded_ratio_hallucinated") - as_float(row, "bounded_ratio_grounded"),
    )


def plot_high_text_head_on_axis(ax, head_rows):
    row = text_head_example(head_rows)
    labels = ["grounded", "hallucinated"]
    text = np.array([as_float(row, "text_mass_grounded"), as_float(row, "text_mass_hallucinated")], dtype=np.float64)
    image = np.array([as_float(row, "image_mass_grounded"), as_float(row, "image_mass_hallucinated")], dtype=np.float64)
    system = np.maximum(1.0 - text - image, 0.0)
    x = np.arange(len(labels))
    ax.bar(x, system, color=COLORS["system"], alpha=0.72, width=0.55, label="system/other")
    ax.bar(x, image, bottom=system, color=COLORS["image"], alpha=0.82, width=0.55, label="image")
    ax.bar(x, text, bottom=system + image, color=COLORS["text"], alpha=0.86, width=0.55, label="text")
    ax.set_title(f"B. High text head {row.get('head_key', '')}")
    ax.set_xticks(x)
    ax.set_xticklabels(["grounded", "hall."])
    ax.set_ylabel("attention mass")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.01), handlelength=1.0, handletextpad=0.25, borderpad=0.1)
    for idx, value in enumerate(text):
        ax.text(idx, min(0.97, system[idx] + image[idx] + value - 0.06), f"T={value:.2f}", ha="center", va="top", color="white", fontsize=6.0, weight="bold")


def plot_contrastive_head_on_axis(ax, head_rows):
    row = contrastive_head_example(head_rows)
    labels = ["grounded", "hall."]
    ratios = np.array(
        [as_float(row, "bounded_ratio_grounded"), as_float(row, "bounded_ratio_hallucinated")],
        dtype=np.float64,
    )
    colors = [COLORS["ground"], COLORS["hall"]]
    x = np.arange(len(labels))
    ax.bar(x, ratios, color=colors, alpha=0.72, width=0.55)
    ax.set_title(f"C. Contrastive head {row.get('head_key', '')}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"$r=T/(T+I)$")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y")
    for idx, value in enumerate(ratios):
        ax.text(idx, value + 0.035, f"{value:.2f}", ha="center", va="bottom", color=colors[idx], fontsize=7.0, weight="bold")
    gap = ratios[1] - ratios[0]
    ax.annotate(
        "",
        xy=(1, ratios[1] + 0.11),
        xytext=(0, ratios[0] + 0.11),
        arrowprops=dict(arrowstyle="->", lw=1.0, color=COLORS["dark"]),
    )
    ax.text(0.5, min(0.96, max(ratios) + 0.16), rf"$\Delta r={gap:.2f}$", ha="center", va="bottom", color=COLORS["dark"], fontsize=7.0)


def plot_head_selection_score_on_axis(ax, head_rows, layers):
    layer_set = set(layers)
    if layer_set:
        allowed = [row for row in head_rows if as_int(row, "layer") in layer_set]
    else:
        allowed = list(head_rows)
    selected = [row for row in allowed if as_int(row, "selected") == 1]
    rejected = [row for row in allowed if as_int(row, "selected") != 1]
    selected_scores = np.array([as_float(row, "score") for row in selected], dtype=np.float64)
    rejected_scores = np.array([as_float(row, "score") for row in rejected], dtype=np.float64)
    bins = np.linspace(0, 1.0, 28)
    if len(rejected_scores):
        ax.hist(rejected_scores, bins=bins, color=COLORS["tail"], alpha=0.82, linewidth=0, label="not selected")
    if len(selected_scores):
        ax.hist(selected_scores, bins=bins, histtype="stepfilled", color=COLORS["score"], alpha=0.52, linewidth=0, label="selected")
        cutoff = float(np.min(selected_scores))
        ax.axvline(cutoff, color=COLORS["score"], linestyle="--", linewidth=1.0)
        ax.text(cutoff + 0.015, 0.92, f"cutoff {cutoff:.2f}", ha="left", va="top", color=COLORS["score"], fontsize=6.2, transform=ax.get_xaxis_transform())
    layer_text = f"L{min(layers)}-L{max(layers)}" if layers else "all layers"
    ax.set_title(f"D. Select Top-{len(selected)}")
    ax.set_xlabel(r"fused score $S(l,h)$")
    ax.set_ylabel("heads")
    ax.set_xlim(0, 1.0)
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="upper left", handlelength=1.0, borderpad=0.1, handletextpad=0.25)
    ax.text(0.98, 0.82, layer_text, ha="right", va="top", color="#b45309", fontsize=7.0, weight="bold", transform=ax.transAxes)


def plot_hall_ground_ratio_on_axis(ax, ratio_rows):
    grouped = ratio_values(ratio_rows, "bounded_ratio")
    bins = np.linspace(0.45, 1.0, 32)
    medians = {}
    for label, color, alpha in [("grounded", COLORS["ground"], 0.32), ("hallucinated", COLORS["hall"], 0.32)]:
        values = grouped.get(label, np.array([], dtype=np.float64))
        if not values.size:
            continue
        ax.hist(values, bins=bins, density=True, histtype="stepfilled", alpha=alpha, color=color, edgecolor=color, linewidth=0.7, label=label)
        median = float(np.median(values))
        medians[label] = median
        ax.axvline(median, color=color, linewidth=1.3)
    ax.set_title("C. Hall vs grounded contrast")
    ax.set_xlabel(r"$r=T/(T+I)$")
    ax.set_ylabel("density")
    ax.set_xlim(0.45, 1.0)
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="upper left", handlelength=1.2, borderpad=0.1, handletextpad=0.3)
    if "grounded" in medians and "hallucinated" in medians:
        ax.text(
            0.98,
            0.92,
            f"median: {medians['grounded']:.3f} -> {medians['hallucinated']:.3f}",
            ha="right",
            va="top",
            color=COLORS["dark"],
            fontsize=6.1,
            transform=ax.transAxes,
        )


def plot_distribution_overlay(ax, all_values, selected_values, bins, colors, labels):
    ax.hist(all_values, bins=bins, density=True, color=COLORS["tail"], alpha=0.78, linewidth=0, label=labels[0])
    if len(selected_values):
        ax.hist(selected_values, bins=bins, density=True, histtype="step", color=colors[1], linewidth=1.6, label=labels[1])
        ax.axvline(float(np.median(selected_values)), color=colors[1], linewidth=1.2)
    if len(all_values):
        ax.axvline(float(np.median(all_values)), color=COLORS["muted"], linewidth=0.9, linestyle=":")
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="upper right", handlelength=1.2, borderpad=0.1, handletextpad=0.3)


def plot_text_mass_distribution_on_axis(ax, head_rows):
    all_values = np.array([as_float(row, "text_mass_all") for row in head_rows], dtype=np.float64)
    selected_values = np.array([as_float(row, "text_mass_all") for row in selected_rows(head_rows)], dtype=np.float64)
    plot_distribution_overlay(
        ax,
        all_values,
        selected_values,
        np.linspace(0.0, 1.0, 28),
        (COLORS["tail"], COLORS["text"]),
        ("all heads", "selected"),
    )
    ax.set_title("A. Text leverage")
    ax.set_xlabel(r"$I_{\mathrm{text}}$")
    ax.set_ylabel("density")
    ax.set_xlim(0.0, 1.0)
    if len(selected_values):
        ax.text(
            0.98,
            0.72,
            f"selected median\n{np.median(selected_values):.2f}",
            ha="right",
            va="top",
            color=COLORS["text"],
            fontsize=6.0,
            transform=ax.transAxes,
        )


def plot_contrastive_distribution_on_axis(ax, head_rows):
    all_values = np.array([math.log1p(positive_contrast(row)) for row in head_rows], dtype=np.float64)
    selected_values = np.array([math.log1p(positive_contrast(row)) for row in selected_rows(head_rows)], dtype=np.float64)
    upper = max(0.2, float(np.percentile(all_values, 99.5)))
    plot_distribution_overlay(
        ax,
        np.clip(all_values, 0.0, upper),
        np.clip(selected_values, 0.0, upper),
        np.linspace(0.0, upper, 28),
        (COLORS["tail"], COLORS["image"]),
        ("all heads", "selected"),
    )
    ax.set_title("B. Contrastive specificity")
    ax.set_xlabel(r"$\log(1+C_{\mathrm{toi}}^+)$")
    ax.set_ylabel("density")
    ax.set_xlim(0.0, upper)
    if len(selected_values):
        ax.text(
            0.98,
            0.72,
            f"selected median\n{np.median(selected_values):.2f}",
            ha="right",
            va="top",
            color=COLORS["image"],
            fontsize=6.0,
            transform=ax.transAxes,
        )


def plot_layer_window_profile_on_axis(ax, head_rows, layers):
    by_layer = defaultdict(list)
    for row in head_rows:
        by_layer[as_int(row, "layer")].append(row)
    layer_ids = sorted(by_layer)
    mean_score = np.array([mean(as_float(row, "score") for row in by_layer[layer]) for layer in layer_ids], dtype=np.float64)
    mean_text = np.array([mean(as_float(row, "text_percentile") for row in by_layer[layer]) for layer in layer_ids], dtype=np.float64)
    ax.axvspan(-0.5, 1.5, color="#f8fafc", zorder=0)
    if layers:
        ax.axvspan(min(layers) - 0.5, max(layers) + 0.5, color=COLORS["window"], zorder=0)
    ax.plot(layer_ids, mean_score, color=COLORS["score"], linewidth=1.6, marker="o", markersize=2.4, label="fused score")
    ax.plot(layer_ids, mean_text, color=COLORS["text"], linewidth=1.25, marker="o", markersize=2.0, label="text leverage")
    ax.set_title("D. Layer window")
    ax.set_xlabel("layer")
    ax.set_ylabel("mean percentile")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlim(min(layer_ids) - 0.5, max(layer_ids) + 0.5)
    ax.set_xticks([0, 8, 16, 24, 31])
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="lower right", handlelength=1.4)
    ax.text(0.7, 0.96, "early\nsink", ha="center", va="top", color=COLORS["muted"], fontsize=5.8)
    if layers:
        ax.text(
            (min(layers) + max(layers)) / 2,
            0.97,
            f"used L{min(layers)}-L{max(layers)}",
            ha="center",
            va="top",
            color="#b45309",
            fontsize=6.2,
            weight="bold",
        )


def plot_rank_fusion_readable_on_axis(ax, head_rows, layers):
    layer_set = set(layers)
    selected = [row for row in head_rows if as_int(row, "selected") == 1]
    window = [row for row in head_rows if as_int(row, "layer") in layer_set and as_int(row, "selected") != 1]
    outside = [row for row in head_rows if as_int(row, "layer") not in layer_set]
    ax.add_patch(Rectangle((0.65, 0.65), 0.35, 0.35, facecolor="#ede9fe", edgecolor="none", zorder=0))
    for rows, color, alpha, size, label, zorder in [
        (outside, COLORS["tail"], 0.24, 7, "outside", 1),
        (window, "#fbbf24", 0.55, 9, "L-window", 2),
        (selected, COLORS["score"], 0.88, 15, "selected", 3),
    ]:
        if not rows:
            continue
        ax.scatter(
            [as_float(row, "text_percentile") for row in rows],
            [as_float(row, "contrast_percentile") for row in rows],
            s=size,
            color=color,
            alpha=alpha,
            linewidths=0,
            label=label,
            zorder=zorder,
        )
    ax.axvline(0.65, color=COLORS["grid"], linestyle="--", linewidth=0.8)
    ax.axhline(0.65, color=COLORS["grid"], linestyle="--", linewidth=0.8)
    ax.text(0.98, 0.97, "high-high\nselected", ha="right", va="top", color=COLORS["score"], fontsize=6.0)
    ax.text(0.95, 0.18, "text-only", ha="right", va="center", color=COLORS["text"], fontsize=5.8)
    ax.text(0.18, 0.92, "specific-only", ha="center", va="top", color=COLORS["image"], fontsize=5.8)
    ax.set_title("D. Fusion-selected pool")
    ax.set_xlabel("text leverage percentile")
    ax.set_ylabel("contrast percentile")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.legend(frameon=False, loc="lower left", handletextpad=0.1, borderpad=0.1)


def plot_ratio_on_axis(ax, ratio_rows):
    bins = np.linspace(0.45, 1.0, 28)
    grouped = ratio_values(ratio_rows, "bounded_ratio")
    for label, color in [("grounded", COLORS["ground"]), ("hallucinated", COLORS["hall"])]:
        values = grouped.get(label, np.array([], dtype=np.float64))
        if values.size:
            ax.hist(values, bins=bins, density=True, histtype="stepfilled", alpha=0.34, color=color, edgecolor=color, linewidth=0.55, label=label)
            ax.axvline(float(np.median(values)), color=color, linewidth=1.2)
    ax.set_title("B. Ratio overlap")
    ax.set_xlabel(r"$T/(T+I)$")
    ax.set_ylabel("density\n(area=1)")
    ax.set_xlim(0.45, 1.0)
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="upper left")


def plot_selected_head_map_on_axis(ax, head_rows, layers):
    max_layer = max(as_int(row, "layer") for row in head_rows)
    max_head = max(as_int(row, "head") for row in head_rows)
    matrix = np.zeros((max_layer + 1, max_head + 1), dtype=np.float64)
    selected = np.zeros_like(matrix, dtype=bool)
    for row in head_rows:
        layer, head = as_int(row, "layer"), as_int(row, "head")
        selected[layer, head] = as_int(row, "selected") == 1
        if selected[layer, head]:
            matrix[layer, head] = as_float(row, "score")
    ax.imshow(matrix, aspect="auto", origin="lower", cmap="Purples", vmin=0.0, vmax=1.0)
    if layers:
        ax.add_patch(Rectangle((-0.5, min(layers) - 0.5), max_head + 1, max(layers) - min(layers) + 1, fill=False, edgecolor="#f59e0b", linewidth=0.9))
    ys, xs = np.where(selected)
    ax.scatter(xs, ys, s=5, facecolors="none", edgecolors=COLORS["dark"], linewidths=0.25)
    ax.set_title("D. Final head pool")
    ax.set_xlabel("head")
    ax.set_ylabel("layer")


def plot_heatmap_on_axis(ax, head_rows, layers):
    max_layer = max(as_int(row, "layer") for row in head_rows)
    max_head = max(as_int(row, "head") for row in head_rows)
    matrix = np.zeros((max_layer + 1, max_head + 1), dtype=np.float64)
    selected = np.zeros_like(matrix, dtype=bool)
    for row in head_rows:
        layer, head = as_int(row, "layer"), as_int(row, "head")
        matrix[layer, head] = as_float(row, "score")
        selected[layer, head] = as_int(row, "selected") == 1
    ax.imshow(matrix, aspect="auto", origin="lower", cmap="magma", vmin=0.0, vmax=1.0)
    if layers:
        ax.add_patch(Rectangle((-0.5, min(layers) - 0.5), max_head + 1, max(layers) - min(layers) + 1, fill=False, edgecolor="#f59e0b", linewidth=0.9))
    ys, xs = np.where(selected)
    ax.scatter(xs, ys, s=5, facecolors="none", edgecolors="white", linewidths=0.35)
    ax.set_title("C. Fused head score")
    ax.set_xlabel("head")
    ax.set_ylabel("layer")


def plot_rank_fusion_on_axis(ax, head_rows, rank_fusion_rows):
    if not rank_fusion_rows:
        return
    by_bucket = {row["bucket"]: row for row in rank_fusion_rows}
    metrics = [("mean_text_percentile", "text"), ("mean_contrast_percentile", "contrast"), ("mean_score", "fused")]
    x = np.arange(len(metrics))
    width = 0.32
    for idx, bucket in enumerate([bucket for bucket in ("selected", "tail") if bucket in by_bucket]):
        vals = [as_float(by_bucket[bucket], key) for key, _ in metrics]
        color = COLORS["dark"] if bucket == "selected" else COLORS["tail"]
        ax.bar(x + (idx - 0.5) * width, vals, width, color=color, alpha=0.82, label=bucket)
    ax.text(0.95, 0.18, "text-only", ha="right", va="center", color=COLORS["text"], fontsize=5.8)
    ax.text(0.95, 0.92, "both high", ha="right", va="top", color=COLORS["score"], fontsize=5.8)
    ax.set_title("D. Rank fusion")
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics])
    ax.set_ylabel("percentile")
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="upper left")


def plot_rank_fusion_scatter_on_axis(ax, head_rows, layers):
    layer_set = set(layers)
    selected = [row for row in head_rows if as_int(row, "selected") == 1]
    window = [row for row in head_rows if as_int(row, "layer") in layer_set and as_int(row, "selected") != 1]
    outside = [row for row in head_rows if as_int(row, "layer") not in layer_set]
    for rows, color, alpha, size, label, zorder in [
        (outside, COLORS["tail"], 0.32, 8, "other", 1),
        (window, "#fbbf24", 0.40, 9, "window", 2),
        (selected, COLORS["score"], 0.86, 12, "selected", 3),
    ]:
        if not rows:
            continue
        ax.scatter(
            [as_float(row, "text_percentile") for row in rows],
            [as_float(row, "contrast_percentile") for row in rows],
            s=size,
            color=color,
            alpha=alpha,
            linewidths=0,
            label=label,
            zorder=zorder,
        )
    ax.axvline(0.75, color=COLORS["grid"], linestyle="--", linewidth=0.8)
    ax.axhline(0.75, color=COLORS["grid"], linestyle="--", linewidth=0.8)
    ax.set_title("D. Rank fusion")
    ax.set_xlabel("text rank")
    ax.set_ylabel("contrast rank")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.legend(frameon=False, loc="lower left", handletextpad=0.1, borderpad=0.1)


def plot_gate_on_axis(ax, gate_rows, marker_rows, show_formula=False):
    xs = np.array([as_float(row, "r_online") for row in gate_rows], dtype=np.float64)
    ys = np.array([as_float(row, "gate") for row in gate_rows], dtype=np.float64)
    if xs.size == 0:
        return
    tau = as_float(gate_rows[0], "tau", 0.9)
    ax.plot(xs, ys, color=COLORS["score"], linewidth=1.9)
    ax.axvline(tau, color=COLORS["hall"], linestyle="--", linewidth=0.9)
    for row in marker_rows:
        label = row.get("label")
        color = COLORS["hall"] if label == "hallucinated" else COLORS["ground"]
        ax.scatter([as_float(row, "bounded_ratio_median")], [as_float(row, "gate_at_median")], color=color, s=20, edgecolors="white", linewidths=0.45)
    ax.set_title("E. Exponential gate")
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$g$")
    ax.set_xlim(0.5, 1.0)
    ax.grid(axis="y")
    if show_formula:
        ax.text(
            0.03,
            0.94,
            r"$\delta=\mathrm{clip}(S(l,h)\exp(q(r-\tau)),0,1)$",
            ha="left",
            va="top",
            color=COLORS["dark"],
            fontsize=7.0,
            transform=ax.transAxes,
        )


def plot_caption_or_formula_on_axis(ax, caption_text):
    ax.axis("off")
    ax.set_title("Caption output" if caption_text else "Suppression action")
    if caption_text:
        wrapped = "\n".join(textwrap.wrap(caption_text, width=62))
        ax.text(
            0.02,
            0.82,
            wrapped,
            ha="left",
            va="top",
            color=COLORS["dark"],
            fontsize=7.3,
            linespacing=1.25,
            transform=ax.transAxes,
        )
    else:
        lines = [
            r"$\delta_{t,l,h}=\mathrm{clip}(S(l,h)\exp(q(r_{t,l,h}-\tau)),0,1)$",
            r"$A'_{t,l,h,j}=(1-\delta_{t,l,h})A_{t,l,h,j},\quad j\in T$",
            "then renormalize the attention row",
        ]
        ax.text(
            0.02,
            0.78,
            "\n".join(lines),
            ha="left",
            va="top",
            color=COLORS["dark"],
            fontsize=8.0,
            linespacing=1.55,
            transform=ax.transAxes,
        )


def plot_redistribution_on_axis(ax, redistribution_rows):
    by_label = {row["label"]: row for row in redistribution_rows}
    labels = [label for label in ("grounded", "hallucinated") if label in by_label]
    names = []
    bars = []
    for label in labels:
        row = by_label[label]
        short = "ground" if label == "grounded" else "hall"
        names.append(f"{short}\nbefore")
        bars.append((label, as_float(row, "system_before"), as_float(row, "image_before"), as_float(row, "text_before")))
        names.append(f"{short}\nafter")
        bars.append((label, as_float(row, "system_after"), as_float(row, "image_after"), as_float(row, "text_after")))
    values = np.array([bar[1:] for bar in bars], dtype=np.float64) if bars else np.zeros((0, 3))
    edgecolors = [COLORS["hall"] if bar[0] == "hallucinated" else COLORS["ground"] for bar in bars]
    x = np.arange(len(values))
    bottom = np.zeros(len(values))
    for idx, (color, label) in enumerate([(COLORS["system"], "system"), (COLORS["image"], "visual"), (COLORS["text"], "text")]):
        vals = values[:, idx] if len(values) else []
        ax.bar(x, vals, bottom=bottom, color=color, width=0.62, alpha=0.86, label=label)
        bottom = bottom + vals
    for xpos, edgecolor in zip(x, edgecolors):
        ax.add_patch(Rectangle((xpos - 0.31, 0), 0.62, 1.0, fill=False, edgecolor=edgecolor, linewidth=0.9))
    ax.set_title("F. Redistribution")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("share")
    ax.grid(axis="y")


def build_numeric_summary(source, ratio_stats, delta_stats, layers):
    selected = selected_rows(source["head_rows"])
    all_rows = source["head_rows"]
    layer_counts = defaultdict(int)
    layer_score = defaultdict(float)
    for row in selected:
        layer = as_int(row, "layer")
        layer_counts[layer] += 1
        layer_score[layer] += as_float(row, "score")
    return {
        "source_dir": source["source_dir"],
        "top_k": len(selected),
        "selection_layers": layers if layers else "all",
        "n_all_heads": len(all_rows),
        "n_selected_heads": len(selected),
        "selected_mean_score": mean(as_float(row, "score") for row in selected),
        "selected_mean_text_mass_all": mean(as_float(row, "text_mass_all") for row in selected),
        "selected_mean_text_percentile": mean(as_float(row, "text_percentile") for row in selected),
        "selected_mean_contrast_percentile": mean(as_float(row, "contrast_percentile") for row in selected),
        "selected_layer_counts": {str(layer): int(layer_counts[layer]) for layer in sorted(layer_counts)},
        "selected_layer_score_sum": {str(layer): float(layer_score[layer]) for layer in sorted(layer_score)},
        "layer_window_stats": layer_window_stats(source["head_rows"], layers) if layers else {},
        "ratio_stats": ratio_stats,
        "delta_stats": delta_stats,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default="./results/coco/method_figure_source_trace_n100_k150_l9_16")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--selection-layers", default="")
    parser.add_argument("--ratio-source", choices=["selected", "all"], default="selected")
    parser.add_argument("--formats", default="png,pdf,svg")
    parser.add_argument("--caption-text", default="")
    parser.add_argument("--no-overview-panel", action="store_true")
    args = parser.parse_args()

    setup_style()
    source_dir = os.path.abspath(args.source_dir)
    output_dir = os.path.abspath(args.output_dir or os.path.join(source_dir, "figures"))
    formats = [fmt.strip().lstrip(".") for fmt in args.formats.split(",") if fmt.strip()]
    source = load_source(source_dir, args.ratio_source)
    source["source_dir"] = source_dir
    layers = source_layers(source["summary"], args.selection_layers)
    top_k = source_top_k(source["summary"], source["head_rows"], args.top_k)
    caption_text = sample_caption_text(source, args.caption_text)

    figures = {}
    figures["phase1_text_mass_sorted"] = figure_text_mass_sorted(source["head_rows"], output_dir, formats)
    figures["phase1_contrastive_specificity_sorted"] = figure_contrastive_specificity_sorted(source["head_rows"], output_dir, formats)
    ratio_paths, ratio_stats = figure_ratio_distribution(source["ratio_rows"], output_dir, formats)
    figures["phase1_ratio_distribution"] = ratio_paths
    figures["phase1_head_score_heatmap"] = figure_head_score_heatmap(source["head_rows"], layers, output_dir, formats)
    figures["phase1_final_head_selection_map"] = figure_final_head_selection_map(source["head_rows"], layers, output_dir, formats)
    figures["phase1_layer_component_scores"] = figure_layer_score_profiles(source["head_rows"], layers, output_dir, formats)
    window_paths, window_stats = figure_layer_window_comparison(source["head_rows"], layers, output_dir, formats)
    figures["phase1_layer_window_comparison"] = window_paths
    figures["phase1_rank_fusion_bars"] = figure_rank_fusion(source["head_rows"], source["rank_fusion"], output_dir, formats)
    figures["phase1_rank_fusion_scatter"] = figure_rank_fusion_scatter(source["head_rows"], output_dir, formats, layers)
    figures["phase2_gate_curve"] = figure_gate_curve(source["gate_curve"], source["gate_markers"], output_dir, formats)
    figures["phase2_attention_redistribution"] = figure_attention_redistribution(source["redistribution"], output_dir, formats)
    delta_paths, delta_stats = figure_delta_distribution(source["ratio_rows"], output_dir, formats)
    figures["phase2_delta_distribution"] = delta_paths
    gate = source["summary"].get("gate", {}) if isinstance(source["summary"].get("gate", {}), dict) else {}
    figures["phase2_delta_flow"] = figure_delta_flow(
        source["head_rows"],
        source["gate_markers"],
        output_dir,
        formats,
        float(gate.get("beta", 10.0)),
        float(gate.get("tau", 0.9)),
    )
    if not args.no_overview_panel:
        figures["method_phase_overview_panel"] = figure_overview_panel(source, output_dir, formats, layers)
        figures["method_claim_compact_panel"] = figure_method_claim_panel(source, output_dir, formats, layers, caption_text)
        figures["method_teaser_dark"] = figure_method_teaser_dark_svg(source, output_dir, layers, caption_text)

    numeric_summary = build_numeric_summary(source, ratio_stats, delta_stats, layers)
    if window_stats:
        numeric_summary["layer_window_stats"] = window_stats
    summary_csv = os.path.join(output_dir, "method_figure_visualization_numeric_summary.csv")
    write_csv(
        summary_csv,
        [
            {
                "top_k": top_k,
                "selection_layers": ",".join(map(str, layers)) if layers else "all",
                "n_selected_heads": numeric_summary["n_selected_heads"],
                "selected_mean_score": numeric_summary["selected_mean_score"],
                "selected_mean_text_mass_all": numeric_summary["selected_mean_text_mass_all"],
                "selected_mean_text_percentile": numeric_summary["selected_mean_text_percentile"],
                "selected_mean_contrast_percentile": numeric_summary["selected_mean_contrast_percentile"],
                "window_mean_score": numeric_summary.get("layer_window_stats", {}).get("window_mean_score", ""),
                "outside_mean_score": numeric_summary.get("layer_window_stats", {}).get("outside_mean_score", ""),
                "window_mean_text_percentile": numeric_summary.get("layer_window_stats", {}).get("window_mean_text_percentile", ""),
                "outside_mean_text_percentile": numeric_summary.get("layer_window_stats", {}).get("outside_mean_text_percentile", ""),
                "window_mean_contrast_percentile": numeric_summary.get("layer_window_stats", {}).get("window_mean_contrast_percentile", ""),
                "outside_mean_contrast_percentile": numeric_summary.get("layer_window_stats", {}).get("outside_mean_contrast_percentile", ""),
                "grounded_ratio_median": ratio_stats.get("grounded", {}).get("median", ""),
                "hallucinated_ratio_median": ratio_stats.get("hallucinated", {}).get("median", ""),
                "grounded_delta_mean": delta_stats.get("grounded", {}).get("mean", ""),
                "hallucinated_delta_mean": delta_stats.get("hallucinated", {}).get("mean", ""),
            }
        ],
    )
    manifest = {
        "source_dir": source_dir,
        "output_dir": output_dir,
        "ratio_source": args.ratio_source,
        "formats": formats,
        "top_k": top_k,
        "selection_layers": layers if layers else "all",
        "caption_text": caption_text,
        "figures": figures,
        "numeric_summary": numeric_summary,
        "numeric_summary_csv": summary_csv,
    }
    manifest_path = os.path.join(output_dir, "method_figure_visualization_manifest.json")
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

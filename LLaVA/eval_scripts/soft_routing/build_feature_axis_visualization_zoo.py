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
from matplotlib.colors import LogNorm


COLORS = {
    "dark": "#0f172a",
    "muted": "#64748b",
    "grid": "#e2e8f0",
    "tail": "#cbd5e1",
    "window": "#fbbf24",
    "selected": "#7c3aed",
    "text": "#f97316",
    "contrast": "#2563eb",
    "ground": "#059669",
    "hall": "#dc2626",
    "image": "#2563eb",
}


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


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


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


def load_source(source_dir, selection_layers):
    head_path = os.path.join(source_dir, "head_scores_all.csv")
    if not os.path.exists(head_path):
        raise FileNotFoundError(f"missing head_scores_all.csv: {head_path}")
    summary = read_json(os.path.join(source_dir, "method_figure_source_summary.json"), default={}) or {}
    layers = selection_layers or parse_layers(summary.get("selection_layers", []))
    rows = read_csv(head_path)
    return rows, summary, layers


def save(fig, output_dir, name, formats):
    os.makedirs(output_dir, exist_ok=True)
    paths = {}
    for fmt in formats:
        path = os.path.join(output_dir, f"{name}.{fmt}")
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        paths[fmt] = path
    plt.close(fig)
    return paths


def arr(rows, key):
    return np.array([as_float(row, key) for row in rows], dtype=np.float64)


def selected_rows(rows):
    return [row for row in rows if as_int(row, "selected") == 1]


def window_rows(rows, layers):
    layer_set = set(layers)
    return [row for row in rows if as_int(row, "layer") in layer_set]


def positive_raw_gap(rows):
    return np.maximum(arr(rows, "raw_toi_gap_hall_minus_grounded"), 0.0)


def log_raw_gap(rows):
    return np.log1p(positive_raw_gap(rows))


def log_raw_toi(rows, key):
    return np.log1p(arr(rows, key))


def layer_head_matrix(rows, values):
    max_layer = max(as_int(row, "layer") for row in rows)
    max_head = max(as_int(row, "head") for row in rows)
    mat = np.zeros((max_layer + 1, max_head + 1), dtype=np.float64)
    sel = np.zeros_like(mat, dtype=bool)
    for row, value in zip(rows, values):
        layer, head = as_int(row, "layer"), as_int(row, "head")
        mat[layer, head] = value
        sel[layer, head] = as_int(row, "selected") == 1
    return mat, sel


def annotate_window(ax, layers, axis="x"):
    if not layers:
        return
    start, end = min(layers), max(layers)
    if axis == "x":
        ax.axvspan(start - 0.5, end + 0.5, color="#fef3c7", alpha=0.7, zorder=0)
    else:
        ax.axhspan(start - 0.5, end + 0.5, color="#fef3c7", alpha=0.7, zorder=0)


def plot_sorted_axis(rows, output_dir, formats):
    configs = [
        ("text_mass_all", "text_mass_sorted_all_selected", "Text mass sorted over all heads", r"$I_{text}$", COLORS["text"], lambda x: x),
        ("raw_toi_gap_hall_minus_grounded", "contrastive_gap_sorted_all_selected", "Contrastive gap sorted over all heads", r"$\log(1+C_{toi}^+)$", COLORS["contrast"], np.log1p),
        ("score", "fused_score_sorted_all_selected", "Fused score sorted over all heads", r"$S(l,h)$", COLORS["selected"], lambda x: x),
    ]
    out = {}
    for key, name, title, ylabel, color, transform in configs:
        ranked = sorted(rows, key=lambda row: transform(max(as_float(row, key), 0.0)), reverse=True)
        values = transform(np.maximum(np.array([as_float(row, key) for row in ranked], dtype=np.float64), 0.0))
        selected = np.array([as_int(row, "selected") == 1 for row in ranked], dtype=bool)
        x = np.arange(1, len(values) + 1)
        fig, ax = plt.subplots(figsize=(4.4, 2.1))
        ax.bar(x, values, width=1.0, color=COLORS["tail"], linewidth=0, label="all heads")
        if np.any(selected):
            ax.bar(x[selected], values[selected], width=1.0, color=color, linewidth=0, label="selected")
        ax.set_title(title)
        ax.set_xlabel("heads sorted by feature")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, len(values) + 1)
        ax.grid(axis="y")
        ax.legend(frameon=False, loc="upper right")
        out[name] = save(fig, output_dir, name, formats)
    return out


def plot_distribution_overlays(rows, output_dir, formats, layers):
    groups = {
        "all": rows,
        "window": window_rows(rows, layers) if layers else [],
        "selected": selected_rows(rows),
    }
    configs = [
        ("text_mass_all", "text_mass_distribution_all_window_selected", r"$I_{text}$", np.linspace(0, 1, 36), COLORS["text"], lambda row: as_float(row, "text_mass_all")),
        ("contrast", "contrastive_distribution_all_window_selected", r"$\log(1+C_{toi}^+)$", None, COLORS["contrast"], lambda row: math.log1p(max(as_float(row, "raw_toi_gap_hall_minus_grounded"), 0.0))),
        ("score", "fused_score_distribution_all_window_selected", r"$S(l,h)$", np.linspace(0, 1, 36), COLORS["selected"], lambda row: as_float(row, "score")),
    ]
    out = {}
    for _, name, xlabel, bins, color, getter in configs:
        all_values = np.array([getter(row) for row in rows], dtype=np.float64)
        if bins is None:
            upper = max(0.1, float(np.percentile(all_values, 99.5)))
            bins = np.linspace(0, upper, 36)
        fig, ax = plt.subplots(figsize=(4.2, 2.25))
        for label, group, group_color, alpha, lw in [
            ("all heads", groups["all"], COLORS["tail"], 0.55, 0.8),
            ("L-window", groups["window"], COLORS["window"], 0.45, 1.0),
            ("selected", groups["selected"], color, 0.0, 1.75),
        ]:
            if not group:
                continue
            values = np.array([getter(row) for row in group], dtype=np.float64)
            if label == "selected":
                ax.hist(values, bins=bins, density=True, histtype="step", color=group_color, linewidth=lw, label=label)
            else:
                ax.hist(values, bins=bins, density=True, color=group_color, alpha=alpha, linewidth=0, label=label)
            ax.axvline(float(np.median(values)), color=group_color, linewidth=1.0, linestyle="--")
        ax.set_title(xlabel + " distribution")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("density")
        ax.grid(axis="y")
        ax.legend(frameon=False, loc="upper right")
        out[name] = save(fig, output_dir, name, formats)
    return out


def plot_scatter_views(rows, output_dir, formats, layers):
    selected = selected_rows(rows)
    layer_set = set(layers)
    window = [row for row in rows if as_int(row, "layer") in layer_set and as_int(row, "selected") != 1]
    outside = [row for row in rows if as_int(row, "layer") not in layer_set and as_int(row, "selected") != 1]
    out = {}

    fig, ax = plt.subplots(figsize=(3.45, 3.05))
    for group, color, alpha, size, label in [
        (outside, COLORS["tail"], 0.28, 10, "other layers"),
        (window, COLORS["window"], 0.52, 13, "L-window"),
        (selected, COLORS["selected"], 0.9, 18, "selected"),
    ]:
        if not group:
            continue
        ax.scatter(arr(group, "text_percentile"), arr(group, "contrast_percentile"), s=size, color=color, alpha=alpha, linewidths=0, label=label)
    ax.axvline(0.65, color=COLORS["grid"], linestyle="--", linewidth=0.9)
    ax.axhline(0.65, color=COLORS["grid"], linestyle="--", linewidth=0.9)
    ax.set_title("Text leverage rank vs. contrastive rank")
    ax.set_xlabel("text-mass percentile")
    ax.set_ylabel("contrastive percentile")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.legend(frameon=False, loc="lower left")
    out["scatter_text_rank_vs_contrast_rank"] = save(fig, output_dir, "scatter_text_rank_vs_contrast_rank", formats)

    fig, ax = plt.subplots(figsize=(3.55, 3.05))
    ax.scatter(arr(rows, "text_mass_all"), log_raw_gap(rows), s=9, color=COLORS["tail"], alpha=0.35, linewidths=0, label="all heads")
    ax.scatter(arr(selected, "text_mass_all"), log_raw_gap(selected), s=18, color=COLORS["selected"], alpha=0.88, linewidths=0, label="selected")
    ax.set_title("Raw feature space")
    ax.set_xlabel(r"$I_{text}$")
    ax.set_ylabel(r"$\log(1+C_{toi}^+)$")
    ax.grid(True)
    ax.legend(frameon=False, loc="upper left")
    out["scatter_text_mass_vs_contrastive_gap"] = save(fig, output_dir, "scatter_text_mass_vs_contrastive_gap", formats)

    fig, ax = plt.subplots(figsize=(3.25, 3.1))
    hb = ax.hexbin(arr(rows, "text_percentile"), arr(rows, "contrast_percentile"), gridsize=18, cmap="Blues", mincnt=1)
    ax.scatter(arr(selected, "text_percentile"), arr(selected, "contrast_percentile"), s=12, facecolors="none", edgecolors=COLORS["selected"], linewidths=0.8, label="selected")
    ax.set_title("All-head density with selected overlay")
    ax.set_xlabel("text-mass percentile")
    ax.set_ylabel("contrastive percentile")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.colorbar(hb, ax=ax, fraction=0.045, pad=0.03).set_label("head count")
    ax.legend(frameon=False, loc="lower left")
    out["hexbin_text_rank_vs_contrast_rank"] = save(fig, output_dir, "hexbin_text_rank_vs_contrast_rank", formats)
    return out


def plot_hall_ground_comparisons(rows, output_dir, formats):
    selected = selected_rows(rows)
    out = {}
    comparisons = [
        (
            "text_mass_hall_vs_ground_all_selected",
            "Text mass: hallucinated vs. grounded object steps",
            "text_mass_grounded",
            "text_mass_hallucinated",
            r"grounded $I_{text}$",
            r"hallucinated $I_{text}$",
            (0, 1),
            (0, 1),
        ),
        (
            "bounded_ratio_hall_vs_ground_all_selected",
            "Bounded text ratio: hallucinated vs. grounded",
            "bounded_ratio_grounded",
            "bounded_ratio_hallucinated",
            r"grounded $T/(T+I)$",
            r"hallucinated $T/(T+I)$",
            (0, 1),
            (0, 1),
        ),
        (
            "raw_toi_hall_vs_ground_log_all_selected",
            "Raw text-over-image ratio: hallucinated vs. grounded",
            "raw_toi_grounded",
            "raw_toi_hallucinated",
            r"grounded $\log(1+T/I)$",
            r"hallucinated $\log(1+T/I)$",
            None,
            None,
        ),
    ]
    for name, title, x_key, y_key, xlabel, ylabel, xlim, ylim in comparisons:
        fig, ax = plt.subplots(figsize=(3.35, 3.1))
        x_all = arr(rows, x_key)
        y_all = arr(rows, y_key)
        x_sel = arr(selected, x_key)
        y_sel = arr(selected, y_key)
        if "raw_toi" in name:
            x_all = np.log1p(x_all)
            y_all = np.log1p(y_all)
            x_sel = np.log1p(x_sel)
            y_sel = np.log1p(y_sel)
        ax.scatter(x_all, y_all, s=9, color=COLORS["tail"], alpha=0.35, linewidths=0, label="all heads")
        ax.scatter(x_sel, y_sel, s=18, color=COLORS["selected"], alpha=0.88, linewidths=0, label="selected")
        lo = min(float(np.min(x_all)), float(np.min(y_all))) if len(x_all) else 0.0
        hi = max(float(np.max(x_all)), float(np.max(y_all))) if len(x_all) else 1.0
        if xlim is not None:
            lo, hi = xlim
        ax.plot([lo, hi], [lo, hi], color=COLORS["muted"], linestyle="--", linewidth=0.9)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if xlim:
            ax.set_xlim(*xlim)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(True)
        ax.legend(frameon=False, loc="upper left")
        out[name] = save(fig, output_dir, name, formats)
    return out


def percentile_bins():
    return [(0.0, 0.50), (0.50, 0.75), (0.75, 0.90), (0.90, 0.95), (0.95, 0.99), (0.99, 1.0)]


def bin_label(lo, hi):
    return f"{int(lo * 100)}-{int(hi * 100)}"


def bin_rows(rows, key, bins):
    groups = []
    for lo, hi in bins:
        group = [row for row in rows if as_float(row, key) >= lo and (as_float(row, key) < hi if hi < 1 else as_float(row, key) <= hi)]
        groups.append(group)
    return groups


def plot_binned_lines(rows, output_dir, formats):
    bins = percentile_bins()
    labels = [bin_label(lo, hi) for lo, hi in bins]
    x = np.arange(len(labels))
    out = {}

    groups = bin_rows(rows, "text_percentile", bins)
    fig, ax = plt.subplots(figsize=(4.4, 2.55))
    ax.plot(x, [mean(as_float(row, "text_mass_grounded") for row in group) for group in groups], marker="o", color=COLORS["ground"], linewidth=1.7, label="grounded text")
    ax.plot(x, [mean(as_float(row, "text_mass_hallucinated") for row in group) for group in groups], marker="o", color=COLORS["hall"], linewidth=1.7, label="hallucinated text")
    ax.plot(x, [mean((as_float(row, "image_mass_grounded") + as_float(row, "image_mass_hallucinated")) / 2 for row in group) for group in groups], marker="o", color=COLORS["image"], linestyle="--", linewidth=1.2, label="image")
    ax.set_title("Object-step mass by text-mass percentile")
    ax.set_xlabel("text-mass percentile bin")
    ax.set_ylabel("mean attention mass")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0, 1)
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="upper left")
    out["line_object_mass_by_text_percentile"] = save(fig, output_dir, "line_object_mass_by_text_percentile", formats)

    groups = bin_rows(rows, "contrast_percentile", bins)
    fig, ax = plt.subplots(figsize=(4.4, 2.55))
    ax.plot(x, [mean(math.log1p(as_float(row, "raw_toi_grounded")) for row in group) for group in groups], marker="o", color=COLORS["ground"], linewidth=1.7, label="grounded")
    ax.plot(x, [mean(math.log1p(as_float(row, "raw_toi_hallucinated")) for row in group) for group in groups], marker="o", color=COLORS["hall"], linewidth=1.7, label="hallucinated")
    ax.set_title("Text-over-image bias by contrastive percentile")
    ax.set_xlabel("contrastive percentile bin")
    ax.set_ylabel(r"mean $\log(1+T/I)$")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="upper left")
    out["line_toi_by_contrast_percentile"] = save(fig, output_dir, "line_toi_by_contrast_percentile", formats)

    fig, ax = plt.subplots(figsize=(4.4, 2.55))
    ax.plot(x, [mean(as_float(row, "bounded_ratio_grounded") for row in group) for group in groups], marker="o", color=COLORS["ground"], linewidth=1.7, label="grounded")
    ax.plot(x, [mean(as_float(row, "bounded_ratio_hallucinated") for row in group) for group in groups], marker="o", color=COLORS["hall"], linewidth=1.7, label="hallucinated")
    ax.set_title("Bounded ratio by contrastive percentile")
    ax.set_xlabel("contrastive percentile bin")
    ax.set_ylabel(r"mean $T/(T+I)$")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0, 1)
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="upper left")
    out["line_bounded_ratio_by_contrast_percentile"] = save(fig, output_dir, "line_bounded_ratio_by_contrast_percentile", formats)
    return out


def plot_layer_profiles(rows, output_dir, formats, layers):
    max_layer = max(as_int(row, "layer") for row in rows)
    layer_ids = np.arange(max_layer + 1)
    by_layer = defaultdict(list)
    for row in rows:
        by_layer[as_int(row, "layer")].append(row)
    selected_by_layer = defaultdict(list)
    for row in selected_rows(rows):
        selected_by_layer[as_int(row, "layer")].append(row)
    out = {}

    fig, ax = plt.subplots(figsize=(5.0, 2.55))
    annotate_window(ax, layers, axis="x")
    ax.plot(layer_ids, [mean(as_float(row, "text_percentile") for row in by_layer[layer]) for layer in layer_ids], color=COLORS["text"], marker="o", ms=2.3, linewidth=1.4, label="mean text percentile")
    ax.plot(layer_ids, [mean(as_float(row, "contrast_percentile") for row in by_layer[layer]) for layer in layer_ids], color=COLORS["contrast"], marker="o", ms=2.3, linewidth=1.4, label="mean contrast percentile")
    ax.plot(layer_ids, [mean(as_float(row, "score") for row in by_layer[layer]) for layer in layer_ids], color=COLORS["selected"], marker="o", ms=2.3, linewidth=1.4, label="mean fused score")
    ax.set_title("Per-layer mean feature percentiles")
    ax.set_xlabel("layer")
    ax.set_ylabel("mean percentile / score")
    ax.set_xlim(-0.5, max_layer + 0.5)
    ax.set_ylim(0, 1)
    ax.grid(axis="y")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    out["layer_mean_feature_profiles"] = save(fig, output_dir, "layer_mean_feature_profiles", formats)

    fig, ax = plt.subplots(figsize=(5.0, 2.55))
    annotate_window(ax, layers, axis="x")
    ax.plot(layer_ids, [sum(as_float(row, "score") for row in selected_by_layer[layer]) for layer in layer_ids], color=COLORS["selected"], marker="o", ms=2.4, linewidth=1.6, label="sum fused")
    ax.plot(layer_ids, [0.5 * sum(as_float(row, "text_percentile") for row in selected_by_layer[layer]) for layer in layer_ids], color=COLORS["text"], marker="o", ms=2.2, linewidth=1.2, label="0.5 sum text")
    ax.plot(layer_ids, [0.5 * sum(as_float(row, "contrast_percentile") for row in selected_by_layer[layer]) for layer in layer_ids], color=COLORS["contrast"], marker="o", ms=2.2, linewidth=1.2, label="0.5 sum contrast")
    ax.set_title("Selected-head score mass by layer")
    ax.set_xlabel("layer")
    ax.set_ylabel("sum over selected heads")
    ax.set_xlim(-0.5, max_layer + 0.5)
    ax.grid(axis="y")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    out["layer_selected_score_mass_components"] = save(fig, output_dir, "layer_selected_score_mass_components", formats)

    fig, ax = plt.subplots(figsize=(5.0, 2.3))
    annotate_window(ax, layers, axis="x")
    ax.bar(layer_ids, [len(selected_by_layer[layer]) for layer in layer_ids], color=COLORS["selected"], alpha=0.82)
    ax.set_title("Selected-head count by layer")
    ax.set_xlabel("layer")
    ax.set_ylabel("selected heads")
    ax.set_xlim(-0.5, max_layer + 0.5)
    ax.grid(axis="y")
    out["layer_selected_head_counts"] = save(fig, output_dir, "layer_selected_head_counts", formats)
    return out


def plot_heatmaps(rows, output_dir, formats, layers):
    configs = [
        ("text_percentile", arr(rows, "text_percentile"), "heatmap_text_percentile_layer_head", "Text-mass percentile", "viridis"),
        ("contrast_percentile", arr(rows, "contrast_percentile"), "heatmap_contrast_percentile_layer_head", "Contrastive percentile", "magma"),
        ("score", arr(rows, "score"), "heatmap_fused_score_layer_head", "Fused score", "Purples"),
        ("log_gap", log_raw_gap(rows), "heatmap_log_contrastive_gap_layer_head", r"$\log(1+C_{toi}^+)$", "Blues"),
    ]
    out = {}
    for _, values, name, title, cmap in configs:
        mat, sel = layer_head_matrix(rows, values)
        fig, ax = plt.subplots(figsize=(4.25, 3.0))
        im = ax.imshow(mat, aspect="auto", origin="lower", cmap=cmap)
        annotate_window(ax, layers, axis="y")
        ys, xs = np.where(sel)
        ax.scatter(xs, ys, s=7, facecolors="none", edgecolors="white", linewidths=0.45)
        ax.set_title(title + " by layer/head")
        ax.set_xlabel("head")
        ax.set_ylabel("layer")
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025).set_label(title)
        out[name] = save(fig, output_dir, name, formats)
    return out


def plot_bin_matrices(rows, output_dir, formats):
    bins = np.linspace(0, 1, 6)
    labels = [f"{int(bins[i] * 100)}-{int(bins[i + 1] * 100)}" for i in range(len(bins) - 1)]
    matrices = {
        "selection_rate": np.zeros((5, 5), dtype=np.float64),
        "mean_score": np.zeros((5, 5), dtype=np.float64),
        "head_count": np.zeros((5, 5), dtype=np.float64),
    }

    def in_bin(value, idx):
        return value >= bins[idx] and (value <= bins[idx + 1] if idx == 4 else value < bins[idx + 1])

    for yi in range(5):
        for xi in range(5):
            group = [
                row
                for row in rows
                if in_bin(as_float(row, "text_percentile"), xi)
                and in_bin(as_float(row, "contrast_percentile"), yi)
            ]
            matrices["head_count"][yi, xi] = len(group)
            matrices["selection_rate"][yi, xi] = mean(as_int(row, "selected") for row in group)
            matrices["mean_score"][yi, xi] = mean(as_float(row, "score") for row in group)
    out = {}
    for key, title, cmap, fmt in [
        ("selection_rate", "Selected fraction by 2D feature bin", "Purples", "{:.0%}"),
        ("mean_score", "Mean fused score by 2D feature bin", "Purples", "{:.2f}"),
        ("head_count", "All-head count by 2D feature bin", "Blues", "{:.0f}"),
    ]:
        fig, ax = plt.subplots(figsize=(3.55, 3.2))
        im = ax.imshow(matrices[key], origin="lower", cmap=cmap)
        for yi in range(5):
            for xi in range(5):
                value = matrices[key][yi, xi]
                ax.text(xi, yi, fmt.format(value), ha="center", va="center", fontsize=6.1, color=COLORS["dark"])
        ax.set_title(title)
        ax.set_xlabel("text-mass percentile bin")
        ax.set_ylabel("contrastive percentile bin")
        ax.set_xticks(range(5))
        ax.set_xticklabels(labels, rotation=28, ha="right")
        ax.set_yticks(range(5))
        ax.set_yticklabels(labels)
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.035)
        out[f"matrix_{key}_text_contrast_bins"] = save(fig, output_dir, f"matrix_{key}_text_contrast_bins", formats)
    return out


def plot_top_rank_curves(rows, output_dir, formats):
    ranked = sorted(rows, key=lambda row: as_float(row, "score"), reverse=True)
    k = np.arange(1, len(ranked) + 1)
    selected = np.array([as_int(row, "selected") for row in ranked], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(5.0, 2.55))
    ax.plot(k, arr(ranked, "score"), color=COLORS["selected"], linewidth=1.6, label="fused score")
    ax.plot(k, arr(ranked, "text_percentile"), color=COLORS["text"], linewidth=1.1, label="text percentile")
    ax.plot(k, arr(ranked, "contrast_percentile"), color=COLORS["contrast"], linewidth=1.1, label="contrast percentile")
    if selected.any():
        cutoff = int(selected.sum())
        ax.axvline(cutoff, color=COLORS["selected"], linestyle="--", linewidth=1.0)
        ax.text(cutoff + 8, 0.98, f"top-{cutoff}", va="top", color=COLORS["selected"], fontsize=6.4)
    ax.set_title("Feature values along fused-score ranking")
    ax.set_xlabel("rank by fused score")
    ax.set_ylabel("value")
    ax.set_xlim(1, len(ranked))
    ax.set_ylim(0, 1)
    ax.grid(axis="y")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    return {"curve_fused_rank_feature_values": save(fig, output_dir, "curve_fused_rank_feature_values", formats)}


def build_summary(rows, layers):
    selected = selected_rows(rows)
    window = window_rows(rows, layers) if layers else []
    groups = {"all": rows, "window": window, "selected": selected}
    output = []
    for name, group in groups.items():
        if not group:
            continue
        output.append(
            {
                "group": name,
                "n_heads": len(group),
                "mean_score": mean(as_float(row, "score") for row in group),
                "mean_text_mass_all": mean(as_float(row, "text_mass_all") for row in group),
                "median_text_mass_all": float(np.median(arr(group, "text_mass_all"))),
                "mean_text_percentile": mean(as_float(row, "text_percentile") for row in group),
                "mean_contrast_percentile": mean(as_float(row, "contrast_percentile") for row in group),
                "mean_log_positive_raw_gap": mean(math.log1p(max(as_float(row, "raw_toi_gap_hall_minus_grounded"), 0.0)) for row in group),
                "mean_bounded_gap_hall_minus_grounded": mean(as_float(row, "bounded_ratio_hallucinated") - as_float(row, "bounded_ratio_grounded") for row in group),
            }
        )
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default="./results/coco/method_figure_source_trace_n100_k150_l9_16")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--selection-layers", default="")
    parser.add_argument("--formats", default="png,pdf,svg")
    args = parser.parse_args()

    setup_style()
    source_dir = os.path.abspath(args.source_dir)
    output_dir = os.path.abspath(args.output_dir or os.path.join(source_dir, "feature_axis_visualization_zoo"))
    formats = [fmt.strip().lstrip(".") for fmt in args.formats.split(",") if fmt.strip()]
    rows, source_summary, layers = load_source(source_dir, parse_layers(args.selection_layers))

    figures = {}
    for chunk in [
        plot_sorted_axis(rows, output_dir, formats),
        plot_distribution_overlays(rows, output_dir, formats, layers),
        plot_scatter_views(rows, output_dir, formats, layers),
        plot_hall_ground_comparisons(rows, output_dir, formats),
        plot_binned_lines(rows, output_dir, formats),
        plot_layer_profiles(rows, output_dir, formats, layers),
        plot_heatmaps(rows, output_dir, formats, layers),
        plot_bin_matrices(rows, output_dir, formats),
        plot_top_rank_curves(rows, output_dir, formats),
    ]:
        figures.update(chunk)

    summary_rows = build_summary(rows, layers)
    summary_csv = os.path.join(output_dir, "feature_axis_visualization_zoo_summary.csv")
    write_csv(summary_csv, summary_rows)
    manifest = {
        "source_dir": source_dir,
        "output_dir": output_dir,
        "formats": formats,
        "selection_layers": layers if layers else "all",
        "source_summary": {
            "top_k": source_summary.get("top_k"),
            "num_samples": source_summary.get("num_samples"),
            "n_object_records": source_summary.get("n_object_records"),
        },
        "summary_csv": summary_csv,
        "figures": figures,
    }
    manifest_path = os.path.join(output_dir, "feature_axis_visualization_zoo_manifest.json")
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print("[summary] feature-axis groups")
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()

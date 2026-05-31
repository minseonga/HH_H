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
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import Rectangle


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
        fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
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


def finite_values(values):
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def kde_curve(values, xmin=None, xmax=None, n_points=320):
    values = finite_values(values)
    if len(values) == 0:
        grid = np.linspace(0.0, 1.0, n_points)
        return grid, np.zeros_like(grid)
    if xmin is None:
        xmin = float(np.min(values))
    if xmax is None:
        xmax = float(np.max(values))
    if xmax <= xmin:
        xmax = xmin + 1.0
    values = np.clip(values, xmin, xmax)
    grid = np.linspace(xmin, xmax, n_points)
    std = float(np.std(values))
    span = xmax - xmin
    bw = 1.06 * std * (len(values) ** -0.2) if std > 0 else span / 25.0
    bw = max(bw, span / 80.0, 1e-4)
    z = (grid[:, None] - values[None, :]) / bw
    density = np.exp(-0.5 * z * z).mean(axis=1) / (bw * math.sqrt(2.0 * math.pi))
    peak = float(np.max(density))
    if peak > 0:
        density = density / peak
    return grid, density


def head_key(row):
    return row.get("head_key") or f"{as_int(row, 'layer')}:{as_int(row, 'head')}"


def top_by(rows, key_fn, top_k):
    return sorted(rows, key=key_fn, reverse=True)[:top_k]


def independent_feature_sets(rows, top_k):
    return {
        "text_mass": top_by(rows, lambda row: as_float(row, "text_mass_all"), top_k),
        "contrastive": top_by(rows, lambda row: max(as_float(row, "raw_toi_gap_hall_minus_grounded"), 0.0), top_k),
        "fused": top_by(rows, lambda row: as_float(row, "score"), top_k),
    }


def row_lookup(rows):
    return {head_key(row): row for row in rows}


def feature_relation_stats(rows, top_k):
    sets = independent_feature_sets(rows, top_k)
    text_keys = {head_key(row) for row in sets["text_mass"]}
    contrast_keys = {head_key(row) for row in sets["contrastive"]}
    fused_keys = {head_key(row) for row in sets["fused"]}
    overlap = text_keys & contrast_keys
    union = text_keys | contrast_keys
    corr = float(np.corrcoef(arr(rows, "text_percentile"), arr(rows, "contrast_percentile"))[0, 1]) if len(rows) > 1 else 0.0
    return {
        "sets": sets,
        "text_keys": text_keys,
        "contrast_keys": contrast_keys,
        "fused_keys": fused_keys,
        "overlap": overlap,
        "jaccard": len(overlap) / max(len(union), 1),
        "corr": corr,
        "fused_text_overlap": len(fused_keys & text_keys),
        "fused_contrast_overlap": len(fused_keys & contrast_keys),
    }


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


def dark_axis(ax):
    ax.set_facecolor("#20211f")
    ax.tick_params(colors="#a8a29e", length=0)
    for spine in ax.spines.values():
        spine.set_color("#3f3f3a")
    ax.grid(False)


def plot_dark_feature_story_panel(rows, output_dir, formats, top_k, layers):
    selected = selected_rows(rows)
    feature_sets = independent_feature_sets(rows, top_k)
    text_feature_rows = feature_sets["text_mass"]
    contrast_feature_rows = feature_sets["contrastive"]
    use_layers = layers or sorted(set(as_int(row, "layer") for row in selected))
    layer_label = f"layers {min(use_layers)}-{max(use_layers)}" if use_layers else "selected layers"
    dark_bg = "#1f201d"
    muted = "#9a9690"
    offwhite = "#f4f1e9"
    green = "#58a88c"
    hall = "#b46a50"
    purple = "#6f5bd6"
    pale = "#e8e6fb"
    cmap = LinearSegmentedColormap.from_list("story_purple", ["#262521", pale, "#7a6be8", "#342b83"])

    fig = plt.figure(figsize=(10.6, 4.05), facecolor=dark_bg)
    ax1 = fig.add_axes([0.055, 0.25, 0.28, 0.54])
    ax2 = fig.add_axes([0.395, 0.25, 0.28, 0.54])
    ax3 = fig.add_axes([0.735, 0.28, 0.235, 0.50])
    axes = [ax1, ax2, ax3]
    for ax in axes:
        dark_axis(ax)

    fig.text(0.055, 0.92, "Text-side mass", color=offwhite, fontsize=12.5, weight="bold")
    fig.text(0.055, 0.865, "leverage, but overlaps", color=muted, fontsize=10.2, weight="bold")
    fig.text(0.395, 0.92, "Text-over-image ratio", color=offwhite, fontsize=12.5, weight="bold")
    fig.text(0.395, 0.865, "contrastive: hall vs non-hall shift", color=muted, fontsize=10.2, weight="bold")
    fig.text(0.735, 0.92, "Fused selection", color=offwhite, fontsize=12.5, weight="bold")
    fig.text(0.735, 0.865, "S = 1/2 P(I) + 1/2 P(C)", color=muted, fontsize=10.2, weight="bold")

    x_g = arr(text_feature_rows, "text_mass_grounded")
    x_h = arr(text_feature_rows, "text_mass_hallucinated")
    grid_g, den_g = kde_curve(x_g, 0.0, 1.0)
    grid_h, den_h = kde_curve(x_h, 0.0, 1.0)
    ax1.fill_between(grid_g, den_g, color=green, alpha=0.58)
    ax1.fill_between(grid_h, den_h, color=hall, alpha=0.58)
    ax1.plot(grid_g, den_g, color=green, linewidth=1.6)
    ax1.plot(grid_h, den_h, color=hall, linewidth=1.6)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.18)
    ax1.set_xlabel("text-side mass I", color=muted, fontsize=10.0, labelpad=6)
    ax1.set_ylabel("density", color=muted, fontsize=10.0, labelpad=8)
    ax1.set_xticks([])
    ax1.set_yticks([])
    med_g = float(np.median(x_g))
    med_h = float(np.median(x_h))
    y_bar = 0.91
    ax1.plot([med_g, med_h], [y_bar, y_bar], color=muted, linewidth=0.9)
    ax1.text((med_g + med_h) / 2, y_bar + 0.035, "heavy overlap", color=muted, fontsize=7.4, ha="center")
    ax1.scatter([], [], s=70, marker="s", color=green, label="non-hall")
    ax1.scatter([], [], s=70, marker="s", color=hall, label="hallucinated")
    leg = ax1.legend(frameon=False, loc="lower left", bbox_to_anchor=(-0.03, -0.16), ncol=2, handlelength=0.8, handletextpad=0.4, columnspacing=1.0)
    for text in leg.get_texts():
        text.set_color(offwhite)
        text.set_fontweight("bold")
    ax1.text(0.5, -0.25, "cannot separate alone", transform=ax1.transAxes, color=muted, ha="center", va="top", fontsize=8.1, style="italic", weight="bold")

    raw_g = np.log1p(arr(contrast_feature_rows, "raw_toi_grounded"))
    raw_h = np.log1p(arr(contrast_feature_rows, "raw_toi_hallucinated"))
    combined = np.concatenate([finite_values(raw_g), finite_values(raw_h)])
    xmin = float(np.percentile(combined, 1.0)) if len(combined) else 0.0
    xmax = float(np.percentile(combined, 99.2)) if len(combined) else 1.0
    grid_g, den_g = kde_curve(raw_g, xmin, xmax)
    grid_h, den_h = kde_curve(raw_h, xmin, xmax)
    ax2.fill_between(grid_g, den_g, color=green, alpha=0.62)
    ax2.fill_between(grid_h, den_h, color=hall, alpha=0.62)
    ax2.plot(grid_g, den_g, color=green, linewidth=1.6)
    ax2.plot(grid_h, den_h, color=hall, linewidth=1.6)
    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(0, 1.18)
    ax2.set_xlabel("log(1 + T / I), per head", color=muted, fontsize=10.0, labelpad=6)
    ax2.set_ylabel("density", color=muted, fontsize=10.0, labelpad=8)
    ax2.set_xticks([])
    ax2.set_yticks([])
    mean_g = float(np.mean(finite_values(raw_g)))
    mean_h = float(np.mean(finite_values(raw_h)))
    ax2.axvline(mean_g, color=green, linestyle="--", linewidth=0.9, alpha=0.85, ymin=0.32, ymax=0.83)
    ax2.axvline(mean_h, color=hall, linestyle="--", linewidth=0.9, alpha=0.85, ymin=0.32, ymax=0.83)
    y_arrow = 0.96
    ax2.plot([mean_g, mean_h], [y_arrow, y_arrow], color=muted, linewidth=0.8)
    ax2.plot([mean_g, mean_g], [y_arrow - 0.02, y_arrow + 0.02], color=muted, linewidth=0.8)
    ax2.plot([mean_h, mean_h], [y_arrow - 0.02, y_arrow + 0.02], color=muted, linewidth=0.8)
    ax2.text((mean_g + mean_h) / 2, y_arrow + 0.055, "C = E[r|H] - E[r|G]", color=muted, fontsize=7.4, ha="center", weight="bold")
    ax2.text(0.5, -0.25, "hallucination-specific separation", transform=ax2.transAxes, color=muted, ha="center", va="top", fontsize=8.1, style="italic", weight="bold")

    mat, _ = layer_head_matrix(rows, arr(rows, "score"))
    im = ax3.imshow(mat, aspect="auto", origin="lower", cmap=cmap, vmin=0.0, vmax=1.0, interpolation="nearest")
    ax3.set_xlabel("")
    ax3.set_ylabel("")
    ax3.set_xticks([])
    yticks = [0]
    if use_layers:
        yticks += [min(use_layers), max(use_layers)]
    yticks += [mat.shape[0] - 1]
    yticks = sorted(set(tick for tick in yticks if 0 <= tick < mat.shape[0]))
    ax3.set_yticks(yticks)
    ax3.set_yticklabels([f"L{tick}" for tick in yticks], color=muted)
    if use_layers:
        start, end = min(use_layers), max(use_layers)
        ax3.add_patch(Rectangle((-0.5, start - 0.5), mat.shape[1], end - start + 1, fill=False, edgecolor="#ff5a4f", linewidth=1.1, linestyle="--"))
    selected = selected_rows(rows)
    if selected:
        ax3.scatter([as_int(row, "head") for row in selected], [as_int(row, "layer") for row in selected], s=7, facecolors="none", edgecolors="#181a18", linewidths=0.35)
    ax3.text(0.5, -0.12, f"selected pool ({layer_label})", transform=ax3.transAxes, ha="center", va="top", color=offwhite, fontsize=7.6, weight="bold")
    cax = fig.add_axes([0.735, 0.17, 0.235, 0.022])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.set_ticks([])
    cax.set_facecolor(dark_bg)
    fig.text(0.735, 0.13, "low", color=muted, fontsize=8.0, ha="left", weight="bold")
    fig.text(0.970, 0.13, "high S", color=muted, fontsize=8.0, ha="right", weight="bold")
    fig.text(0.735, 0.06, "leverage / specificity", color=muted, fontsize=8.4, ha="left", style="italic", weight="bold")

    fig.text(0.352, 0.505, "->", color="#6d6a64", fontsize=21.0, ha="center", va="center")
    fig.text(0.694, 0.505, "->", color="#6d6a64", fontsize=21.0, ha="center", va="center")
    return {"dark_feature_story_panel": save(fig, output_dir, "dark_feature_story_panel", formats)}


def draw_head_grid(ax, group, color, title, subtitle, max_layer, max_head, marker="o"):
    ax.set_facecolor("#f8fafc")
    ax.scatter(
        [as_int(row, "head") for row in group],
        [as_int(row, "layer") for row in group],
        s=18,
        color=color,
        alpha=0.86,
        linewidths=0,
        marker=marker,
    )
    ax.set_title(title, fontsize=8.2, pad=4)
    ax.set_xlim(-0.5, max_head + 0.5)
    ax.set_ylim(-0.5, max_layer + 0.5)
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    ax.grid(True, color="#e2e8f0", linewidth=0.45)
    ax.text(0.02, 0.96, subtitle, transform=ax.transAxes, ha="left", va="top", fontsize=6.2, color=COLORS["muted"])


def plot_non_circular_feature_rationale_panel(rows, output_dir, formats, top_k):
    stats = feature_relation_stats(rows, top_k)
    sets = stats["sets"]
    max_layer = max(as_int(row, "layer") for row in rows)
    max_head = max(as_int(row, "head") for row in rows)
    text_rank = arr(rows, "text_percentile")
    contrast_rank = arr(rows, "contrast_percentile")

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.35))
    draw_head_grid(
        axes[0, 0],
        sets["text_mass"],
        COLORS["text"],
        "A. text_mass top-k",
        "feature applied alone",
        max_layer,
        max_head,
    )
    draw_head_grid(
        axes[0, 1],
        sets["contrastive"],
        COLORS["contrast"],
        "B. contrastive top-k",
        "feature applied alone",
        max_layer,
        max_head,
        marker="s",
    )

    ax = axes[1, 0]
    text_keys = stats["text_keys"]
    contrast_keys = stats["contrast_keys"]
    overlap = stats["overlap"]
    text_only_x, text_only_y = [], []
    contrast_only_x, contrast_only_y = [], []
    overlap_x, overlap_y = [], []
    other_x, other_y = [], []
    for row in rows:
        key = head_key(row)
        point = (as_float(row, "text_percentile"), as_float(row, "contrast_percentile"))
        if key in overlap:
            overlap_x.append(point[0])
            overlap_y.append(point[1])
        elif key in text_keys:
            text_only_x.append(point[0])
            text_only_y.append(point[1])
        elif key in contrast_keys:
            contrast_only_x.append(point[0])
            contrast_only_y.append(point[1])
        else:
            other_x.append(point[0])
            other_y.append(point[1])
    ax.scatter(other_x, other_y, s=8, color=COLORS["tail"], alpha=0.30, linewidths=0, label="other")
    ax.scatter(text_only_x, text_only_y, s=16, color=COLORS["text"], alpha=0.78, linewidths=0, label="text only")
    ax.scatter(contrast_only_x, contrast_only_y, s=16, color=COLORS["contrast"], alpha=0.78, linewidths=0, marker="s", label="contrast only")
    ax.scatter(overlap_x, overlap_y, s=34, color=COLORS["selected"], alpha=0.9, linewidths=0, marker="*", label="overlap")
    ax.set_title("C. independent feature relation", fontsize=8.2, pad=4)
    ax.set_xlabel("text_mass percentile")
    ax.set_ylabel("contrastive percentile")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.text(
        0.03,
        0.97,
        f"r={stats['corr']:.2f}\nJaccard={stats['jaccard']:.2f}\noverlap={len(overlap)}/{top_k}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=COLORS["dark"],
        fontsize=6.6,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.2},
    )
    ax.legend(frameon=False, loc="lower right", handletextpad=0.2, borderpad=0.1)

    ax = axes[1, 1]
    labels = ["text only", "overlap", "contrast only", "fused & text", "fused & contrast"]
    values = [
        len(text_keys - contrast_keys),
        len(overlap),
        len(contrast_keys - text_keys),
        stats["fused_text_overlap"],
        stats["fused_contrast_overlap"],
    ]
    colors = [COLORS["text"], COLORS["selected"], COLORS["contrast"], COLORS["text"], COLORS["contrast"]]
    x = np.arange(len(labels))
    ax.bar(x, values, color=colors, alpha=0.82)
    ax.axhline(top_k, color=COLORS["muted"], linestyle="--", linewidth=0.8)
    ax.text(len(labels) - 0.45, top_k + max(values) * 0.015, f"top-{top_k}", ha="right", va="bottom", color=COLORS["muted"], fontsize=6.2)
    ax.set_title("D. fused ranking overlaps both axes", fontsize=8.2, pad=4)
    ax.set_ylabel("head count")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=24, ha="right")
    ax.grid(axis="y")
    ax.text(
        0.03,
        0.86,
        "Feature-first: score all heads,\nthen inspect independent overlap.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=COLORS["muted"],
        fontsize=5.9,
    )
    fig.suptitle("Non-circular feature rationale", y=0.965, fontsize=10.0, weight="bold")
    fig.subplots_adjust(top=0.88, bottom=0.12, left=0.08, right=0.985, hspace=0.52, wspace=0.28)
    return {"non_circular_feature_rationale_panel": save(fig, output_dir, "non_circular_feature_rationale_panel", formats)}


def plot_independent_topk_layer_head_maps(rows, output_dir, formats, top_k):
    stats = feature_relation_stats(rows, top_k)
    sets = stats["sets"]
    max_layer = max(as_int(row, "layer") for row in rows)
    max_head = max(as_int(row, "head") for row in rows)
    text_keys = stats["text_keys"]
    contrast_keys = stats["contrast_keys"]
    overlap_keys = stats["overlap"]

    fig, axes = plt.subplots(1, 3, figsize=(9.4, 2.85), sharex=True, sharey=True)
    panels = [
        ("A. text_mass top-k", text_keys, COLORS["text"], "text-heavy heads"),
        ("B. contrastive top-k", contrast_keys, COLORS["contrast"], "hall-specific heads"),
        ("C. overlap / disagreement", text_keys | contrast_keys, COLORS["selected"], "two independent feature sets"),
    ]
    for ax, (title, keys, color, subtitle) in zip(axes, panels):
        ax.set_facecolor("#f8fafc")
        ax.set_title(title)
        ax.set_xlim(-0.5, max_head + 0.5)
        ax.set_ylim(-0.5, max_layer + 0.5)
        ax.set_xlabel("head")
        ax.grid(True, color="#e2e8f0", linewidth=0.45)
        if ax is axes[0]:
            ax.set_ylabel("layer")
        ax.text(0.02, 0.96, subtitle, transform=ax.transAxes, ha="left", va="top", fontsize=6.2, color=COLORS["muted"])
    axes[0].scatter(
        [as_int(row, "head") for row in sets["text_mass"]],
        [as_int(row, "layer") for row in sets["text_mass"]],
        s=18,
        color=COLORS["text"],
        alpha=0.86,
        linewidths=0,
    )
    axes[1].scatter(
        [as_int(row, "head") for row in sets["contrastive"]],
        [as_int(row, "layer") for row in sets["contrastive"]],
        s=18,
        color=COLORS["contrast"],
        alpha=0.86,
        linewidths=0,
    )
    lookup = row_lookup(rows)
    for key in sorted(text_keys | contrast_keys):
        row = lookup[key]
        if key in overlap_keys:
            color, marker, size, label = COLORS["selected"], "*", 42, "overlap"
        elif key in text_keys:
            color, marker, size, label = COLORS["text"], "o", 20, "text only"
        else:
            color, marker, size, label = COLORS["contrast"], "s", 16, "contrast only"
        axes[2].scatter([as_int(row, "head")], [as_int(row, "layer")], s=size, color=color, alpha=0.86, marker=marker, linewidths=0)
    axes[2].scatter([], [], s=20, color=COLORS["text"], marker="o", label="text only")
    axes[2].scatter([], [], s=16, color=COLORS["contrast"], marker="s", label="contrast only")
    axes[2].scatter([], [], s=42, color=COLORS["selected"], marker="*", label="overlap")
    axes[2].legend(frameon=False, loc="lower right", handletextpad=0.2, borderpad=0.1)
    fig.suptitle(f"Independent top-{top_k} heads selected by each feature", y=1.02, fontsize=10.0, weight="bold")
    return {"independent_topk_layer_head_maps": save(fig, output_dir, "independent_topk_layer_head_maps", formats)}


def plot_independent_topk_layer_distribution(rows, output_dir, formats, top_k):
    sets = independent_feature_sets(rows, top_k)
    max_layer = max(as_int(row, "layer") for row in rows)
    layers = np.arange(max_layer + 1)

    def counts(group):
        out = np.zeros(max_layer + 1, dtype=np.float64)
        for row in group:
            out[as_int(row, "layer")] += 1
        return out

    def score_sum(group, score_key):
        out = np.zeros(max_layer + 1, dtype=np.float64)
        for row in group:
            if score_key == "log_gap":
                value = math.log1p(max(as_float(row, "raw_toi_gap_hall_minus_grounded"), 0.0))
            else:
                value = as_float(row, score_key)
            out[as_int(row, "layer")] += value
        total = float(out.sum())
        return out / total if total > 0 else out

    fig, ax = plt.subplots(figsize=(5.2, 2.65))
    ax.plot(layers, counts(sets["text_mass"]), marker="o", ms=2.6, color=COLORS["text"], linewidth=1.5, label="text_mass top-k")
    ax.plot(layers, counts(sets["contrastive"]), marker="o", ms=2.6, color=COLORS["contrast"], linewidth=1.5, label="contrastive top-k")
    ax.plot(layers, counts(sets["fused"]), marker="o", ms=2.6, color=COLORS["selected"], linewidth=1.5, label="fused top-k")
    ax.set_title(f"Layer distribution of independent top-{top_k} sets")
    ax.set_xlabel("layer")
    ax.set_ylabel("head count")
    ax.set_xlim(-0.5, max_layer + 0.5)
    ax.grid(axis="y")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    paths = {"independent_topk_layer_counts": save(fig, output_dir, "independent_topk_layer_counts", formats)}

    fig, ax = plt.subplots(figsize=(5.2, 2.65))
    ax.plot(layers, score_sum(sets["text_mass"], "text_mass_all"), marker="o", ms=2.5, color=COLORS["text"], linewidth=1.45, label="text_mass top-k: text mass share")
    ax.plot(layers, score_sum(sets["contrastive"], "log_gap"), marker="o", ms=2.5, color=COLORS["contrast"], linewidth=1.45, label="contrastive top-k: contrast share")
    ax.plot(layers, score_sum(sets["fused"], "score"), marker="o", ms=2.5, color=COLORS["selected"], linewidth=1.45, label="fused top-k: score share")
    ax.set_title(f"Layer-wise score share within independent top-{top_k} sets")
    ax.set_xlabel("layer")
    ax.set_ylabel("share within set")
    ax.set_xlim(-0.5, max_layer + 0.5)
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="upper left")
    paths["independent_topk_layer_score_share"] = save(fig, output_dir, "independent_topk_layer_score_share", formats)
    return paths


def plot_independent_rank_relationship(rows, output_dir, formats, top_k):
    stats = feature_relation_stats(rows, top_k)
    text_keys = stats["text_keys"]
    contrast_keys = stats["contrast_keys"]
    overlap = stats["overlap"]
    text_rank = arr(rows, "text_percentile")
    contrast_rank = arr(rows, "contrast_percentile")
    corr = stats["corr"]

    colors = []
    sizes = []
    markers = []
    for row in rows:
        key = head_key(row)
        if key in overlap:
            colors.append(COLORS["selected"])
            sizes.append(34)
            markers.append("*")
        elif key in text_keys:
            colors.append(COLORS["text"])
            sizes.append(18)
            markers.append("o")
        elif key in contrast_keys:
            colors.append(COLORS["contrast"])
            sizes.append(16)
            markers.append("s")
        else:
            colors.append(COLORS["tail"])
            sizes.append(9)
            markers.append("o")

    fig, ax = plt.subplots(figsize=(3.75, 3.35))
    for marker in sorted(set(markers)):
        idx = [i for i, value in enumerate(markers) if value == marker]
        ax.scatter(text_rank[idx], contrast_rank[idx], s=np.array(sizes)[idx], c=np.array(colors)[idx], alpha=0.74 if marker != "o" else 0.42, linewidths=0, marker=marker)
    ax.scatter([], [], s=18, color=COLORS["text"], marker="o", label="text_mass top-k")
    ax.scatter([], [], s=16, color=COLORS["contrast"], marker="s", label="contrastive top-k")
    ax.scatter([], [], s=34, color=COLORS["selected"], marker="*", label="overlap")
    ax.set_title("Independent feature rankings")
    ax.set_xlabel("text_mass percentile")
    ax.set_ylabel("contrastive percentile")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.text(0.03, 0.97, f"Pearson r={corr:.2f}\noverlap={len(overlap)}/{top_k}", transform=ax.transAxes, ha="left", va="top", color=COLORS["dark"], fontsize=6.8, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.4})
    ax.legend(frameon=False, loc="lower right")
    paths = {"independent_rank_scatter_text_vs_contrast": save(fig, output_dir, "independent_rank_scatter_text_vs_contrast", formats)}

    fig, ax = plt.subplots(figsize=(4.3, 2.45))
    order = np.argsort(text_rank)
    ax.plot(text_rank[order], contrast_rank[order], color=COLORS["muted"], linewidth=1.0, alpha=0.55, label="heads ordered by text_mass")
    ax.scatter(text_rank[order], contrast_rank[order], s=8, color=COLORS["tail"], alpha=0.45)
    ax.set_title("Contrastive rank is not determined by text_mass rank")
    ax.set_xlabel("text_mass percentile")
    ax.set_ylabel("contrastive percentile")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.text(0.03, 0.94, f"rank correlation proxy r={corr:.2f}", transform=ax.transAxes, ha="left", va="top", color=COLORS["dark"], fontsize=6.8)
    paths["independent_rank_curve_contrast_given_text_order"] = save(fig, output_dir, "independent_rank_curve_contrast_given_text_order", formats)
    return paths


def plot_independent_feature_profile_bars(rows, output_dir, formats, top_k):
    sets = independent_feature_sets(rows, top_k)
    groups = [
        ("all heads", rows, COLORS["tail"]),
        ("text_mass top-k", sets["text_mass"], COLORS["text"]),
        ("contrastive top-k", sets["contrastive"], COLORS["contrast"]),
        ("fused top-k", sets["fused"], COLORS["selected"]),
    ]
    metrics = [
        ("mean_text_mass", r"$I_{text}$", lambda group: mean(as_float(row, "text_mass_all") for row in group)),
        ("mean_log_gap", r"$\log(1+C_{toi}^+)$", lambda group: mean(math.log1p(max(as_float(row, "raw_toi_gap_hall_minus_grounded"), 0.0)) for row in group)),
        ("mean_bounded_gap", r"$\Delta T/(T+I)$", lambda group: mean(as_float(row, "bounded_ratio_hallucinated") - as_float(row, "bounded_ratio_grounded") for row in group)),
        ("mean_score", r"$S(l,h)$", lambda group: mean(as_float(row, "score") for row in group)),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(9.4, 2.4))
    for ax, (_, title, getter) in zip(axes, metrics):
        values = [getter(group) for _, group, _ in groups]
        colors = [color for _, _, color in groups]
        x = np.arange(len(groups))
        ax.bar(x, values, color=colors, alpha=0.82)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([label.replace(" ", "\n") for label, _, _ in groups], rotation=0)
        ax.grid(axis="y")
    fig.suptitle(f"Feature profiles of independent top-{top_k} sets", y=1.04, fontsize=9.4, weight="bold")
    return {"independent_topk_feature_profile_bars": save(fig, output_dir, "independent_topk_feature_profile_bars", formats)}


def plot_independent_coarse_attention_patterns(rows, output_dir, formats, top_k):
    sets = independent_feature_sets(rows, top_k)
    groups = [
        ("text_mass top-k", sets["text_mass"], COLORS["text"]),
        ("contrastive top-k", sets["contrastive"], COLORS["contrast"]),
        ("fused top-k", sets["fused"], COLORS["selected"]),
    ]
    labels = []
    bars = []
    edgecolors = []
    for group_name, group, color in groups:
        for label in ("grounded", "hallucinated"):
            text = mean(as_float(row, f"text_mass_{label}") for row in group)
            image = mean(as_float(row, f"image_mass_{label}") for row in group)
            other = max(0.0, 1.0 - text - image)
            labels.append(group_name.replace(" top-k", "") + "\n" + ("G" if label == "grounded" else "H"))
            bars.append((other, image, text))
            edgecolors.append(COLORS["ground"] if label == "grounded" else COLORS["hall"])
    values = np.array(bars, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6.0, 2.65))
    x = np.arange(len(values))
    bottom = np.zeros(len(values), dtype=np.float64)
    for idx, (name, color) in enumerate([("other", COLORS["muted"]), ("image", COLORS["image"]), ("text", COLORS["text"])]):
        ax.bar(x, values[:, idx], bottom=bottom, color=color, alpha=0.82, label=name)
        bottom += values[:, idx]
    for xpos, edgecolor in zip(x, edgecolors):
        ax.add_patch(plt.Rectangle((xpos - 0.4, 0), 0.8, 1.0, fill=False, edgecolor=edgecolor, linewidth=1.0))
    ax.set_title(f"Coarse attention pattern of independent top-{top_k} sets")
    ax.set_ylabel("mean attention mass")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.text(0.99, 0.98, "border: green=grounded, red=hallucinated", transform=ax.transAxes, ha="right", va="top", color=COLORS["muted"], fontsize=6.2)
    return {"independent_topk_coarse_attention_patterns": save(fig, output_dir, "independent_topk_coarse_attention_patterns", formats)}


def independent_topk_summary(rows, top_k):
    sets = independent_feature_sets(rows, top_k)
    text_keys = {head_key(row) for row in sets["text_mass"]}
    contrast_keys = {head_key(row) for row in sets["contrastive"]}
    fused_keys = {head_key(row) for row in sets["fused"]}
    output = []
    for name, group in sets.items():
        keys = {head_key(row) for row in group}
        output.append(
            {
                "feature_set": name,
                "top_k": top_k,
                "n_heads": len(group),
                "mean_layer": mean(as_int(row, "layer") for row in group),
                "mean_text_mass_all": mean(as_float(row, "text_mass_all") for row in group),
                "mean_text_percentile": mean(as_float(row, "text_percentile") for row in group),
                "mean_log_positive_raw_gap": mean(math.log1p(max(as_float(row, "raw_toi_gap_hall_minus_grounded"), 0.0)) for row in group),
                "mean_contrast_percentile": mean(as_float(row, "contrast_percentile") for row in group),
                "mean_fused_score": mean(as_float(row, "score") for row in group),
                "overlap_with_text_topk": len(keys & text_keys),
                "overlap_with_contrastive_topk": len(keys & contrast_keys),
                "overlap_with_fused_topk": len(keys & fused_keys),
                "heads": " ".join(head_key(row) for row in group),
            }
        )
    output.append(
        {
            "feature_set": "text_vs_contrastive_overlap",
            "top_k": top_k,
            "n_heads": len(text_keys & contrast_keys),
            "jaccard_text_contrastive": len(text_keys & contrast_keys) / max(len(text_keys | contrast_keys), 1),
            "pearson_text_contrast_percentile": float(np.corrcoef(arr(rows, "text_percentile"), arr(rows, "contrast_percentile"))[0, 1]),
            "heads": " ".join(sorted(text_keys & contrast_keys)),
        }
    )
    return output


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
    parser.add_argument("--feature-top-k", type=int, default=0)
    parser.add_argument("--formats", default="png,pdf,svg")
    args = parser.parse_args()

    setup_style()
    source_dir = os.path.abspath(args.source_dir)
    output_dir = os.path.abspath(args.output_dir or os.path.join(source_dir, "feature_axis_visualization_zoo"))
    formats = [fmt.strip().lstrip(".") for fmt in args.formats.split(",") if fmt.strip()]
    rows, source_summary, layers = load_source(source_dir, parse_layers(args.selection_layers))
    feature_top_k = int(args.feature_top_k or source_summary.get("top_k") or len(selected_rows(rows)) or 150)

    figures = {}
    for chunk in [
        plot_dark_feature_story_panel(rows, output_dir, formats, feature_top_k, layers),
        plot_non_circular_feature_rationale_panel(rows, output_dir, formats, feature_top_k),
        plot_independent_topk_layer_head_maps(rows, output_dir, formats, feature_top_k),
        plot_independent_topk_layer_distribution(rows, output_dir, formats, feature_top_k),
        plot_independent_rank_relationship(rows, output_dir, formats, feature_top_k),
        plot_independent_feature_profile_bars(rows, output_dir, formats, feature_top_k),
        plot_independent_coarse_attention_patterns(rows, output_dir, formats, feature_top_k),
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
    independent_rows = independent_topk_summary(rows, feature_top_k)
    independent_summary_csv = os.path.join(output_dir, "feature_axis_independent_topk_summary.csv")
    write_csv(independent_summary_csv, independent_rows)
    manifest = {
        "source_dir": source_dir,
        "output_dir": output_dir,
        "formats": formats,
        "selection_layers": layers if layers else "all",
        "feature_top_k": feature_top_k,
        "source_summary": {
            "top_k": source_summary.get("top_k"),
            "num_samples": source_summary.get("num_samples"),
            "n_object_records": source_summary.get("n_object_records"),
        },
        "summary_csv": summary_csv,
        "independent_topk_summary_csv": independent_summary_csv,
        "figures": figures,
    }
    manifest_path = os.path.join(output_dir, "feature_axis_visualization_zoo_manifest.json")
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print("[summary] feature-axis groups")
    for row in summary_rows:
        print(row)
    print("[summary] independent top-k feature sets")
    for row in independent_rows:
        preview = dict(row)
        if "heads" in preview and len(str(preview["heads"])) > 180:
            preview["heads"] = str(preview["heads"])[:180] + "..."
        print(preview)


if __name__ == "__main__":
    main()

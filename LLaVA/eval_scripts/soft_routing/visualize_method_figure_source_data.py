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
    }


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


def plot_gate_on_axis(ax, gate_rows, marker_rows):
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

    figures = {}
    figures["phase1_text_mass_sorted"] = figure_text_mass_sorted(source["head_rows"], output_dir, formats)
    ratio_paths, ratio_stats = figure_ratio_distribution(source["ratio_rows"], output_dir, formats)
    figures["phase1_ratio_distribution"] = ratio_paths
    figures["phase1_head_score_heatmap"] = figure_head_score_heatmap(source["head_rows"], layers, output_dir, formats)
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
        "figures": figures,
        "numeric_summary": numeric_summary,
        "numeric_summary_csv": summary_csv,
    }
    manifest_path = os.path.join(output_dir, "method_figure_visualization_manifest.json")
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

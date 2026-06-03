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
    "grounded": "#16a34a",
    "hallucinated": "#dc2626",
    "r_img": "#2563eb",
    "r_full": "#7c3aed",
    "grid": "#e5e7eb",
    "dark": "#111827",
    "muted": "#64748b",
}


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_head_set(path, topk):
    if not path:
        return None, []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "heads" in data:
        records = data["heads"][:topk]
        heads = [(int(row["layer"]), int(row["head"])) for row in records]
    elif isinstance(data, dict) and "selected_heads" in data:
        heads = [(int(l), int(h)) for l, h in data["selected_heads"][:topk]]
    elif isinstance(data, list) and data and isinstance(data[0], dict):
        heads = [(int(row["layer"]), int(row["head"])) for row in data[:topk]]
    elif isinstance(data, list) and data and isinstance(data[0], list):
        heads = [(int(l), int(h)) for l, h in data[:topk]]
    else:
        raise ValueError(f"unsupported head file format: {path}")
    return set(heads), heads


def read_rows(path, head_set):
    rows = []
    seen_heads = set()
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            label = raw.get("label")
            if label not in {"grounded", "hallucinated"}:
                continue
            layer = int(raw["layer"])
            head = int(raw["head"])
            if head_set is not None and (layer, head) not in head_set:
                continue
            system = safe_float(raw.get("system_before"))
            image = safe_float(raw.get("image_before"))
            text = safe_float(raw.get("text_before"))
            denom_img = text + image
            denom_full = text + image + system
            r_img = text / denom_img if denom_img > 0 else 0.0
            r_full = text / denom_full if denom_full > 0 else 0.0
            score = safe_float(raw.get("score"), 1.0)
            item = {
                "question_id": str(raw.get("question_id", "")),
                "image": raw.get("image", ""),
                "step_idx": str(raw.get("step_idx", "")),
                "token_id": str(raw.get("token_id", "")),
                "token_text": raw.get("token_text", ""),
                "label": label,
                "layer": layer,
                "head": head,
                "head_key": f"{layer}:{head}",
                "score": score,
                "system": system,
                "image_mass": image,
                "text_mass": text,
                "r_img": r_img,
                "r_full": r_full,
            }
            rows.append(item)
            seen_heads.add((layer, head))
    return rows, seen_heads


def group_token_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (row["question_id"], row["step_idx"], row["token_id"], row["token_text"], row["label"])
        groups[key].append(row)
    out = []
    for key, items in groups.items():
        qid, step_idx, token_id, token_text, label = key
        out.append(
            {
                "question_id": qid,
                "step_idx": step_idx,
                "token_id": token_id,
                "token_text": token_text,
                "label": label,
                "n_heads": len(items),
                "r_img": float(np.mean([row["r_img"] for row in items])),
                "r_full": float(np.mean([row["r_full"] for row in items])),
                "delta_img": float(np.mean([row["delta_img"] for row in items])) if "delta_img" in items[0] else 0.0,
                "delta_full": float(np.mean([row["delta_full"] for row in items])) if "delta_full" in items[0] else 0.0,
            }
        )
    return out


def auc_score(scores, labels):
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    rank_sum = 0.0
    i = 0
    rank = 1
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (rank + rank + (j - i) - 1) / 2.0
        for k in range(i, j):
            if pairs[k][1] == 1:
                rank_sum += avg_rank
        rank += j - i
        i = j
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def roc_curve(scores, labels):
    thresholds = sorted(set(scores), reverse=True)
    thresholds = [float("inf")] + thresholds + [-float("inf")]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    points = []
    for thr in thresholds:
        tp = fp = 0
        for score, label in zip(scores, labels):
            pred = score >= thr
            if pred and label:
                tp += 1
            elif pred and not label:
                fp += 1
        tpr = tp / n_pos if n_pos else 0.0
        fpr = fp / n_neg if n_neg else 0.0
        points.append((fpr, tpr, thr))
    points.sort()
    return points


def quantile(values, q):
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=float), q))


def summarize_by_label(rows, field):
    out = []
    for label in ["grounded", "hallucinated"]:
        vals = [row[field] for row in rows if row["label"] == label]
        out.append(
            {
                "label": label,
                "n": len(vals),
                "mean": float(np.mean(vals)) if vals else None,
                "q25": quantile(vals, 0.25),
                "q50": quantile(vals, 0.50),
                "q75": quantile(vals, 0.75),
                "q90": quantile(vals, 0.90),
            }
        )
    return out


def compute_delta(rows, tau, q, strength):
    for row in rows:
        gate_img = math.exp(q * (row["r_img"] - tau))
        gate_full = math.exp(q * (row["r_full"] - tau))
        row["delta_img"] = min(max(strength * row["score"] * gate_img, 0.0), 1.0)
        row["delta_full"] = min(max(strength * row["score"] * gate_full, 0.0), 1.0)


def trigger_summary(rows, field, tau):
    out = []
    for label in ["grounded", "hallucinated"]:
        items = [row for row in rows if row["label"] == label]
        n = len(items)
        triggered = sum(1 for row in items if row[field] >= tau)
        out.append({"label": label, "n": n, "triggered": triggered, "trigger_rate": triggered / n if n else None})
    return out


def threshold_metrics(rows, field, tau):
    labels = [1 if row["label"] == "hallucinated" else 0 for row in rows]
    preds = [1 if row[field] >= tau else 0 for row in rows]
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    tn = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 0)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)
    return {
        "field": field,
        "tau": tau,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "hall_recall": tp / (tp + fn) if tp + fn else None,
        "ground_fpr": fp / (fp + tn) if fp + tn else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "flagged_rate": (tp + fp) / len(rows) if rows else None,
        "base_hall_rate": sum(labels) / len(labels) if labels else None,
    }


def write_csv(path, rows):
    if not rows:
        return
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
            "grid.color": "#e5e7eb",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig, output_dir, name, formats):
    paths = {}
    for fmt in formats:
        path = os.path.join(output_dir, f"{name}.{fmt}")
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        paths[fmt] = path
    plt.close(fig)
    return paths


def plot_roc(token_rows, output_dir, formats):
    setup_style()
    fig, ax = plt.subplots(figsize=(3.55, 3.05), constrained_layout=True)
    labels = [1 if row["label"] == "hallucinated" else 0 for row in token_rows]
    aucs = {}
    for field, color, label in [("r_img", COLORS["r_img"], r"$T/(T+I)$"), ("r_full", COLORS["r_full"], r"$T/(S+I+T)$")]:
        scores = [row[field] for row in token_rows]
        auc = auc_score(scores, labels)
        aucs[field] = auc
        curve = roc_curve(scores, labels)
        ax.plot([p[0] for p in curve], [p[1] for p in curve], color=color, linewidth=1.8, label=f"{label}, AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#94a3b8", linewidth=1.0)
    ax.set_xlabel("grounded false-positive rate")
    ax.set_ylabel("hallucinated recall")
    ax.set_title("Token-level detector test", fontsize=10, fontweight="bold")
    ax.grid(True)
    ax.legend(frameon=False, fontsize=7.6, loc="lower right")
    return save(fig, output_dir, "dual_ratio_token_level_roc", formats), aucs


def plot_distribution(token_rows, output_dir, formats):
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.55), constrained_layout=True)
    bins = np.linspace(0.35, 1.0, 34)
    for ax, field, title in [
        (axes[0], "r_img", r"$r_{img}=T/(T+I)$"),
        (axes[1], "r_full", r"$r_{full}=T/(S+I+T)$"),
    ]:
        for label in ["grounded", "hallucinated"]:
            vals = [row[field] for row in token_rows if row["label"] == label]
            color = COLORS[label]
            ax.hist(vals, bins=bins, density=True, histtype="stepfilled", alpha=0.22, color=color)
            ax.hist(vals, bins=bins, density=True, histtype="step", linewidth=1.35, color=color, label=label)
            if vals:
                ax.axvline(float(np.median(vals)), color=color, linestyle="--", linewidth=1.0)
        ax.set_title(title, fontsize=9.5, fontweight="bold")
        ax.set_xlabel("token-level mean ratio")
        ax.set_ylabel("density")
        ax.grid(axis="y")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Both ratio definitions show substantial H/G overlap", fontsize=10.5, fontweight="bold")
    return save(fig, output_dir, "dual_ratio_token_level_distributions", formats)


def plot_delta(token_rows, output_dir, formats):
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.55), constrained_layout=True)
    bins = np.linspace(0, 1, 32)
    for ax, field, title in [
        (axes[0], "delta_img", r"$\delta$ from $T/(T+I)$"),
        (axes[1], "delta_full", r"$\delta$ from $T/(S+I+T)$"),
    ]:
        for label in ["grounded", "hallucinated"]:
            vals = [row[field] for row in token_rows if row["label"] == label]
            color = COLORS[label]
            ax.hist(vals, bins=bins, density=True, histtype="stepfilled", alpha=0.22, color=color)
            ax.hist(vals, bins=bins, density=True, histtype="step", linewidth=1.35, color=color, label=label)
            if vals:
                ax.axvline(float(np.median(vals)), color=color, linestyle="--", linewidth=1.0)
        ax.set_title(title, fontsize=9.5, fontweight="bold")
        ax.set_xlabel("mean suppression strength")
        ax.set_ylabel("density")
        ax.grid(axis="y")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Dynamic strength separates only if high-r tail separates", fontsize=10.5, fontweight="bold")
    return save(fig, output_dir, "dual_ratio_token_level_delta_distributions", formats)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratio-csv", default="experiments_in_server/method_figure_source_trace_n100_k150_l9_16/selected_head_object_ratio_distribution.csv")
    parser.add_argument("--head-file", default="ADHH/LLaVA/results_summary/coco/ranked_heads_global__itext_all__C_toi_HminusG.json")
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--tau", type=float, default=0.9)
    parser.add_argument("--q", type=float, default=8.0)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--output-dir", default="LLaVA/results/coco/dual_ratio_detector_diagnostics_top100_l9_l16")
    parser.add_argument("--formats", default="svg,png,pdf")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    formats = [fmt.strip() for fmt in args.formats.split(",") if fmt.strip()]
    head_set, ordered_heads = load_head_set(args.head_file, args.topk)
    rows, seen_heads = read_rows(args.ratio_csv, head_set)
    compute_delta(rows, args.tau, args.q, args.strength)
    token_rows = group_token_rows(rows)

    labels = [1 if row["label"] == "hallucinated" else 0 for row in token_rows]
    summary = {
        "ratio_csv": args.ratio_csv,
        "head_file": args.head_file,
        "topk": args.topk,
        "tau": args.tau,
        "q": args.q,
        "strength": args.strength,
        "requested_heads": len(ordered_heads),
        "seen_requested_heads": len(seen_heads),
        "missing_requested_heads": [f"{l}:{h}" for l, h in ordered_heads if (l, h) not in seen_heads],
        "n_head_step_rows": len(rows),
        "n_token_rows": len(token_rows),
        "n_grounded_tokens": sum(1 for row in token_rows if row["label"] == "grounded"),
        "n_hallucinated_tokens": sum(1 for row in token_rows if row["label"] == "hallucinated"),
        "head_step_summary": {
            "r_img": summarize_by_label(rows, "r_img"),
            "r_full": summarize_by_label(rows, "r_full"),
            "delta_img": summarize_by_label(rows, "delta_img"),
            "delta_full": summarize_by_label(rows, "delta_full"),
        },
        "token_summary": {
            "r_img": summarize_by_label(token_rows, "r_img"),
            "r_full": summarize_by_label(token_rows, "r_full"),
            "delta_img": summarize_by_label(token_rows, "delta_img"),
            "delta_full": summarize_by_label(token_rows, "delta_full"),
        },
        "token_auc": {
            "r_img": auc_score([row["r_img"] for row in token_rows], labels),
            "r_full": auc_score([row["r_full"] for row in token_rows], labels),
        },
        "token_threshold_metrics": {
            "r_img": threshold_metrics(token_rows, "r_img", args.tau),
            "r_full": threshold_metrics(token_rows, "r_full", args.tau),
        },
        "head_step_trigger_summary": {
            "r_img": trigger_summary(rows, "r_img", args.tau),
            "r_full": trigger_summary(rows, "r_full", args.tau),
        },
        "token_trigger_summary": {
            "r_img": trigger_summary(token_rows, "r_img", args.tau),
            "r_full": trigger_summary(token_rows, "r_full", args.tau),
        },
        "caveat": (
            "If missing_requested_heads is non-empty, the ratio CSV does not fully cover the requested exact head pool. "
            "Use this as a diagnostic only; regenerate the trace for final paper numbers."
        ),
    }

    write_csv(os.path.join(args.output_dir, "dual_ratio_token_rows.csv"), token_rows)
    write_csv(
        os.path.join(args.output_dir, "dual_ratio_summary_flat.csv"),
        [
            {"level": "token", "field": field, **row}
            for field, items in summary["token_summary"].items()
            for row in items
        ]
        + [
            {"level": "head_step", "field": field, **row}
            for field, items in summary["head_step_summary"].items()
            for row in items
        ],
    )
    roc_paths, _ = plot_roc(token_rows, args.output_dir, formats)
    dist_paths = plot_distribution(token_rows, args.output_dir, formats)
    delta_paths = plot_delta(token_rows, args.output_dir, formats)
    summary["figures"] = {"roc": roc_paths, "distribution": dist_paths, "delta": delta_paths}

    with open(os.path.join(args.output_dir, "dual_ratio_detector_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

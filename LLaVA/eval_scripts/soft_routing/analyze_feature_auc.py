import argparse
import csv
import json
import math
import os

import numpy as np
from sklearn.metrics import roc_auc_score


DEFAULT_FEATURES = [
    "diag_mean_text_mass",
    "diag_max_text_mass",
    "diag_mean_img_mass",
    "diag_min_img_mass",
    "diag_mean_margin",
    "diag_max_margin",
    "diag_mean_text_img_log_ratio",
    "diag_max_text_img_log_ratio",
    "diag_triggered_head_count",
    "diag_triggered_head_frac",
    "diag_mean_soft_alpha",
    "diag_max_soft_alpha",
    "original_entropy",
    "original_top1_top2_margin",
    "hard_entropy",
    "soft_entropy",
    "kl_original_to_hard",
    "kl_original_to_soft",
    "kl_hard_to_soft",
]


def rankdata(values):
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def spearman(x, y):
    if len(x) < 2:
        return None
    rx = rankdata(np.asarray(x, dtype=float))
    ry = rankdata(np.asarray(y, dtype=float))
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def load_records(path, strong_labels, weak_labels):
    records = []
    with open(path, "r") as f:
        for line in f:
            item = json.loads(line)
            if not item.get("has_divergence", False):
                continue
            label_name = item.get("winner") or item.get("case_file")
            if label_name in strong_labels:
                item["_label"] = 1
            elif label_name in weak_labels:
                item["_label"] = 0
            else:
                continue
            records.append(item)
    return records


def numeric(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def analyze(records, features):
    rows = []
    labels_all = [item["_label"] for item in records]
    for feature in features:
        x = []
        y = []
        for item in records:
            value = numeric(item.get(feature))
            if value is None:
                continue
            x.append(value)
            y.append(item["_label"])
        if len(set(y)) < 2 or len(x) < 2:
            continue
        auc = float(roc_auc_score(y, x))
        rows.append({
            "feature": feature,
            "n": len(x),
            "auc_strong_when_high": auc,
            "auc_abs": max(auc, 1.0 - auc),
            "direction": "strong_when_high" if auc >= 0.5 else "weak_when_high",
            "spearman_with_strong_label": spearman(x, y),
            "mean_strong": float(np.mean([v for v, label in zip(x, y) if label == 1])),
            "mean_weak": float(np.mean([v for v, label in zip(x, y) if label == 0])),
        })
    rows.sort(key=lambda row: row["auc_abs"], reverse=True)
    return {
        "num_records": len(records),
        "num_strong": int(sum(labels_all)),
        "num_weak": int(len(labels_all) - sum(labels_all)),
        "features": rows,
    }


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--strong-labels", nargs="+", default=["hard_win"])
    parser.add_argument("--weak-labels", nargs="+", default=["soft_win", "hard_tradeoff"])
    parser.add_argument("--features", nargs="+", default=DEFAULT_FEATURES)
    args = parser.parse_args()

    records = load_records(args.diagnostics_jsonl, set(args.strong_labels), set(args.weak_labels))
    summary = analyze(records, args.features)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "feature_auc_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    write_csv(os.path.join(args.output_dir, "feature_auc.csv"), summary["features"])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

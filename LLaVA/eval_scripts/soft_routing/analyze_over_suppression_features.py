import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from eval_scripts.soft_routing.head_prior_utils import head_key, load_head_priors


DEFAULT_FEATURES = [
    "trigger_count",
    "trigger_frac",
    "weighted_trigger_count",
    "mean_excess",
    "max_excess",
    "sum_excess",
    "weighted_excess",
    "max_weighted_excess",
    "mean_text_mass",
    "max_text_mass",
    "mean_prior_text_mass",
    "max_prior_text_mass",
    "entropy_delta_hard_minus_original",
    "entropy_delta_soft_minus_original",
    "hard_kl_from_original",
    "soft_kl_from_original",
    "hard_changed_token",
    "soft_kept_original",
    "hard_changed_soft_kept",
]


QUAL_FIELDS = [
    "image_id",
    "winner",
    "reason",
    "divergence_step",
    "original_next_token",
    "hard_next_token",
    "soft_next_token",
    "original_next_token_type",
    "hard_next_token_type",
    "soft_next_token_type",
    "trigger_count",
    "weighted_trigger_count",
    "mean_excess",
    "weighted_excess",
    "max_excess",
    "entropy_delta_hard_minus_original",
    "hard_kl_from_original",
    "prefix_text",
    "hard_caption",
    "soft_caption",
]


def load_records(path):
    records = []
    with open(path, "r") as f:
        for line in f:
            item = json.loads(line)
            if item.get("has_divergence", False):
                records.append(item)
    return records


def load_priors(path, top_k):
    if not path:
        return {}
    _, priors, source = load_head_priors(path, top_k=top_k, prior_mode="score")
    return priors, source


def numeric(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(value) or math.isinf(value):
        return default
    return value


def row_label(item):
    return item.get("winner") or item.get("case_file") or "unknown"


def compute_features(item, priors, threshold):
    diagnostics = item.get("diagnostics") or []
    trigger_count = 0
    weighted_trigger_count = 0.0
    excess_values = []
    weighted_excess_values = []
    text_masses = []
    prior_text_masses = []

    for record in diagnostics:
        key = record.get("head_key") or head_key(record.get("layer"), record.get("head"))
        prior = float(priors.get(key, 1.0))
        text_mass = numeric(record.get("text_mass"))
        excess = max(0.0, text_mass - threshold)
        triggered = 1.0 if text_mass >= threshold else 0.0
        trigger_count += int(triggered)
        weighted_trigger_count += prior * triggered
        excess_values.append(excess)
        weighted_excess_values.append(prior * excess)
        text_masses.append(text_mass)
        prior_text_masses.append(prior * text_mass)

    n = max(len(diagnostics), 1)
    hard_changed = int(item.get("hard_next_token_id") != item.get("original_next_token_id"))
    soft_kept = int(item.get("soft_next_token_id") == item.get("original_next_token_id"))
    features = {
        "winner": row_label(item),
        "trigger_count": trigger_count,
        "trigger_frac": trigger_count / n,
        "weighted_trigger_count": weighted_trigger_count,
        "mean_excess": float(np.mean(excess_values)) if excess_values else 0.0,
        "max_excess": float(np.max(excess_values)) if excess_values else 0.0,
        "sum_excess": float(np.sum(excess_values)) if excess_values else 0.0,
        "weighted_excess": float(np.sum(weighted_excess_values)) if weighted_excess_values else 0.0,
        "max_weighted_excess": float(np.max(weighted_excess_values)) if weighted_excess_values else 0.0,
        "mean_text_mass": float(np.mean(text_masses)) if text_masses else 0.0,
        "max_text_mass": float(np.max(text_masses)) if text_masses else 0.0,
        "mean_prior_text_mass": float(np.mean(prior_text_masses)) if prior_text_masses else 0.0,
        "max_prior_text_mass": float(np.max(prior_text_masses)) if prior_text_masses else 0.0,
        "entropy_delta_hard_minus_original": numeric(item.get("hard_entropy")) - numeric(item.get("original_entropy")),
        "entropy_delta_soft_minus_original": numeric(item.get("soft_entropy")) - numeric(item.get("original_entropy")),
        "hard_kl_from_original": numeric(item.get("kl_original_to_hard")),
        "soft_kl_from_original": numeric(item.get("kl_original_to_soft")),
        "hard_changed_token": hard_changed,
        "soft_kept_original": soft_kept,
        "hard_changed_soft_kept": int(hard_changed and soft_kept),
    }
    return features


def mean_or_none(values):
    return float(np.mean(values)) if values else None


def group_means(rows, features):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["winner"]].append(row)

    output = []
    for group, items in sorted(grouped.items()):
        result = {"group": group, "n": len(items)}
        for feature in features:
            result[feature] = mean_or_none([numeric(item.get(feature)) for item in items])
        output.append(result)
    return output


def auc_rows(rows, features, positive_label):
    output = []
    labels = [1 if row["winner"] == positive_label else 0 for row in rows]
    for feature in features:
        values = [numeric(row.get(feature)) for row in rows]
        if len(set(labels)) < 2 or len(set(values)) < 2:
            continue
        auc = float(roc_auc_score(labels, values))
        auprc = float(average_precision_score(labels, values))
        pos_values = [v for v, y in zip(values, labels) if y == 1]
        neg_values = [v for v, y in zip(values, labels) if y == 0]
        output.append({
            "feature": feature,
            "positive_label": positive_label,
            "n": len(values),
            "auroc_high_predicts_positive": auc,
            "auroc_abs": max(auc, 1.0 - auc),
            "direction": "high_predicts_positive" if auc >= 0.5 else "low_predicts_positive",
            "auprc_high_predicts_positive": auprc,
            "mean_positive": mean_or_none(pos_values),
            "mean_negative": mean_or_none(neg_values),
        })
    output.sort(key=lambda row: row["auroc_abs"], reverse=True)
    return output


def bin_rows(rows, features, positive_label, num_bins):
    output = []
    for feature in features:
        values = [numeric(row.get(feature)) for row in rows]
        values_sorted = sorted(values)
        if not values_sorted:
            continue
        edges = []
        for idx in range(1, num_bins):
            pos = round((len(values_sorted) - 1) * idx / num_bins)
            edges.append(values_sorted[pos])
        counts = defaultdict(Counter)
        for row in rows:
            value = numeric(row.get(feature))
            lo = "-inf"
            bin_name = None
            for edge in edges:
                if value <= edge:
                    bin_name = f"({lo},{edge:.4g}]"
                    break
                lo = f"{edge:.4g}"
            if bin_name is None:
                bin_name = f"({lo},inf)"
            counts[bin_name][row["winner"]] += 1
        for bin_name, counter in counts.items():
            total = sum(counter.values())
            positive = counter.get(positive_label, 0)
            output.append({
                "feature": feature,
                "bin": bin_name,
                "total": total,
                f"{positive_label}_count": positive,
                f"{positive_label}_rate": positive / total if total else 0.0,
                **dict(counter),
            })
    return output


def qualitative_rows(records, feature_rows, label, top_k):
    by_image = {str(record.get("image_id")): record for record in records}
    selected = [row for row in feature_rows if row["winner"] == label]
    selected.sort(key=lambda row: (row["hard_changed_soft_kept"], row["weighted_excess"]), reverse=True)
    output = []
    for row in selected[:top_k]:
        item = by_image.get(str(row.get("image_id"))) or {}
        merged = {**item, **row}
        output.append({field: merged.get(field) for field in QUAL_FIELDS})
    return output


def write_csv(path, rows):
    if not rows:
        with open(path, "w") as f:
            f.write("")
        return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prior-path", default="")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--adhh-threshold", type=float, default=0.4)
    parser.add_argument("--positive-label", default="hard_tradeoff")
    parser.add_argument("--qual-label", default="hard_tradeoff")
    parser.add_argument("--qual-top-k", type=int, default=20)
    parser.add_argument("--num-bins", type=int, default=5)
    args = parser.parse_args()

    records = load_records(args.diagnostics_jsonl)
    priors, prior_source = load_priors(args.prior_path, args.top_k) if args.prior_path else ({}, "uniform_default")
    feature_rows = []
    for item in records:
        row = {
            "image_id": str(item.get("image_id")),
            "reason": item.get("reason", ""),
            "case_file": item.get("case_file", ""),
        }
        row.update(compute_features(item, priors, args.adhh_threshold))
        feature_rows.append(row)

    summary = {
        "num_records": len(feature_rows),
        "winner_counts": dict(Counter(row["winner"] for row in feature_rows)),
        "prior_source": prior_source,
        "positive_label": args.positive_label,
        "features": DEFAULT_FEATURES,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    write_csv(os.path.join(args.output_dir, "record_features.csv"), feature_rows)
    write_csv(os.path.join(args.output_dir, "group_feature_means.csv"), group_means(feature_rows, DEFAULT_FEATURES))
    write_csv(os.path.join(args.output_dir, "hard_tradeoff_auc.csv"), auc_rows(feature_rows, DEFAULT_FEATURES, args.positive_label))
    write_csv(os.path.join(args.output_dir, "feature_bins.csv"), bin_rows(feature_rows, DEFAULT_FEATURES, args.positive_label, args.num_bins))
    write_csv(os.path.join(args.output_dir, f"qualitative_{args.qual_label}.csv"), qualitative_rows(records, feature_rows, args.qual_label, args.qual_top_k))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

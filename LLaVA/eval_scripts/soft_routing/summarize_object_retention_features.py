import argparse
import csv
import os
from collections import defaultdict

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from eval_scripts.soft_routing.analyze_object_retention_steps import FEATURES, write_csv


def read_csv(path):
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def mean(values):
    return float(np.mean(values)) if values else None


def to_float(row, key):
    value = row.get(key, "")
    if value in {"", None}:
        return 0.0
    return float(value)


def group_means(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["label"]].append(row)
    output = []
    for label, items in sorted(grouped.items()):
        record = {"label": label, "n": len(items)}
        for feature in FEATURES:
            record[feature] = mean([to_float(item, feature) for item in items])
        output.append(record)
    return output


def pairwise_auc(rows, positive_name, negative_name, positive_fn, negative_fn):
    filtered = [row for row in rows if positive_fn(row) or negative_fn(row)]
    labels = [1 if positive_fn(row) else 0 for row in filtered]
    output = []
    for feature in FEATURES:
        values = [to_float(row, feature) for row in filtered]
        if len(set(labels)) < 2 or len(set(values)) < 2:
            continue
        auc = float(roc_auc_score(labels, values))
        output.append({
            "feature": feature,
            "positive_label": positive_name,
            "negative_label": negative_name,
            "n": len(values),
            "auroc_high_predicts_positive": auc,
            "auroc_abs": max(auc, 1.0 - auc),
            "direction": "high_predicts_positive" if auc >= 0.5 else "low_predicts_positive",
            "auprc_high_predicts_positive": float(average_precision_score(labels, values)),
            "mean_positive": mean([v for v, y in zip(values, labels) if y == 1]),
            "mean_negative": mean([v for v, y in zip(values, labels) if y == 0]),
        })
    output.sort(key=lambda item: item["auroc_abs"], reverse=True)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-csv", required=True)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    rows = read_csv(args.features_csv)
    output_dir = args.output_dir or os.path.dirname(args.features_csv)
    os.makedirs(output_dir, exist_ok=True)

    write_csv(os.path.join(output_dir, "group_feature_means.csv"), group_means(rows))
    write_csv(
        os.path.join(output_dir, "lost_vs_hallucinated_auc.csv"),
        pairwise_auc(
            rows,
            "lost_grounded",
            "hallucinated_object*",
            lambda row: row["label"] == "lost_grounded",
            lambda row: row["label"].startswith("hallucinated_object"),
        ),
    )
    write_csv(
        os.path.join(output_dir, "lost_vs_hallucinated_hard_auc.csv"),
        pairwise_auc(
            rows,
            "lost_grounded",
            "hallucinated_object_hard",
            lambda row: row["label"] == "lost_grounded",
            lambda row: row["label"] == "hallucinated_object_hard",
        ),
    )
    write_csv(
        os.path.join(output_dir, "lost_vs_hallucinated_soft_auc.csv"),
        pairwise_auc(
            rows,
            "lost_grounded",
            "hallucinated_object_soft",
            lambda row: row["label"] == "lost_grounded",
            lambda row: row["label"] == "hallucinated_object_soft",
        ),
    )
    write_csv(
        os.path.join(output_dir, "hallucinated_vs_kept_auc.csv"),
        pairwise_auc(
            rows,
            "hallucinated_object*",
            "kept_grounded",
            lambda row: row["label"].startswith("hallucinated_object"),
            lambda row: row["label"] == "kept_grounded",
        ),
    )
    print(f"wrote summaries to {output_dir}")


if __name__ == "__main__":
    main()

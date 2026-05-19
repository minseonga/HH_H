import argparse
import csv
import json
import math
import os
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


RAW_FEATURES = [
    "text_mass",
    "img_mass",
    "text_ratio",
    "img_entropy_norm",
    "text_ratio_img_entropy",
    "text_img_log_ratio",
    "text_value_norm",
    "img_value_norm",
    "text_img_value_dot",
    "text_img_value_cosine",
    "text_img_value_abs_cosine",
    "text_img_value_orthogonality",
    "supported_text_value_norm",
    "unsupported_text_value_norm",
    "unsupported_text_value_ratio",
    "unsupported_total_value_ratio",
    "visual_mass_ratio",
    "visual_value_ratio",
    "question_attention",
    "output_attention",
    "recent_output_attention",
    "recent_output_ratio",
    "question_value_norm",
    "output_value_norm",
    "recent_output_value_norm",
]


BASE_SELECTOR_FEATURES = [
    "text_mass",
    "text_ratio",
    "text_value_norm",
    "text_ratio_img_entropy",
    "text_img_log_ratio",
    "visual_value_ratio",
    "text_img_value_cosine",
    "text_img_value_abs_cosine",
    "text_img_value_orthogonality",
    "unsupported_text_value_norm",
    "unsupported_text_value_ratio",
    "unsupported_total_value_ratio",
    "recent_output_ratio",
    "img_value_norm",
    "sample_norm_text_mass",
    "sample_norm_text_value_norm",
    "sample_norm_visual_value_ratio",
    "sample_norm_text_img_value_cosine",
    "sample_norm_text_img_value_abs_cosine",
    "sample_norm_text_img_value_orthogonality",
    "sample_norm_unsupported_text_value_norm",
    "sample_norm_unsupported_text_value_ratio",
    "sample_norm_unsupported_total_value_ratio",
    "sample_norm_text_x_norm",
    "sample_norm_text_ratio_x_value",
    "sample_norm_text_ratio_x_entropy",
    "sample_norm_value_x_low_visual",
    "sample_norm_text_x_norm_x_low_visual",
    "layer_norm_text_mass",
    "layer_norm_text_value_norm",
    "layer_norm_visual_value_ratio",
    "layer_norm_text_img_value_cosine",
    "layer_norm_text_img_value_abs_cosine",
    "layer_norm_text_img_value_orthogonality",
    "layer_norm_unsupported_text_value_norm",
    "layer_norm_unsupported_text_value_ratio",
    "layer_norm_unsupported_total_value_ratio",
    "layer_norm_text_x_norm",
    "layer_norm_text_ratio_x_value",
    "layer_norm_text_ratio_x_entropy",
    "layer_norm_value_x_low_visual",
    "layer_norm_text_x_norm_x_low_visual",
]


MODEL_FEATURES = [
    "sample_norm_text_mass",
    "sample_norm_text_ratio",
    "sample_norm_text_value_norm",
    "sample_norm_img_value_norm",
    "sample_norm_visual_value_ratio",
    "sample_norm_text_img_value_cosine",
    "sample_norm_text_img_value_abs_cosine",
    "sample_norm_text_img_value_orthogonality",
    "sample_norm_unsupported_text_value_norm",
    "sample_norm_unsupported_text_value_ratio",
    "sample_norm_unsupported_total_value_ratio",
    "sample_norm_recent_output_ratio",
    "sample_norm_text_x_norm",
    "sample_norm_text_ratio_x_value",
    "sample_norm_value_x_low_visual",
    "sample_norm_text_x_norm_x_low_visual",
    "layer_norm_text_mass",
    "layer_norm_text_ratio",
    "layer_norm_text_value_norm",
    "layer_norm_img_value_norm",
    "layer_norm_visual_value_ratio",
    "layer_norm_text_img_value_cosine",
    "layer_norm_text_img_value_abs_cosine",
    "layer_norm_text_img_value_orthogonality",
    "layer_norm_unsupported_text_value_norm",
    "layer_norm_unsupported_text_value_ratio",
    "layer_norm_unsupported_total_value_ratio",
    "layer_norm_recent_output_ratio",
    "layer_norm_text_x_norm",
    "layer_norm_text_ratio_x_value",
    "layer_norm_value_x_low_visual",
    "layer_norm_text_x_norm_x_low_visual",
]


def safe_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(value):
        return 0.0
    return value


def utility_for_row(row):
    effect = safe_float(row.get("causal_effect", 0.0))
    family = row.get("label_family") or row.get("label", "")
    if str(family).startswith("hallucinated"):
        return effect
    if family in {"lost_grounded", "kept_grounded"}:
        return -effect
    return effect


def load_rows(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            row["causal_effect"] = safe_float(row.get("causal_effect", 0.0))
            row["suppression_utility"] = utility_for_row(row)
            family = row.get("label_family") or row.get("label", "")
            if family in {"lost_grounded", "kept_grounded"}:
                row["label_family"] = family
            elif str(family).startswith("hallucinated") or str(row.get("label", "")).startswith("hallucinated"):
                row["label_family"] = "hallucinated"
            else:
                row["label_family"] = str(family)
            for feature in RAW_FEATURES:
                row[feature] = safe_float(row.get(feature, 0.0))
            rows.append(row)
    add_step_normalized_features(rows)
    return rows


def add_minmax(records, feature, output_key):
    values = np.array([safe_float(row.get(feature, 0.0)) for row in records], dtype=float)
    if len(values) == 0:
        return
    low = float(values.min())
    high = float(values.max())
    denom = high - low
    for row in records:
        row[output_key] = 0.0 if denom <= 1e-12 else float((safe_float(row.get(feature, 0.0)) - low) / denom)


def add_step_normalized_features(rows):
    by_step = defaultdict(list)
    by_step_layer = defaultdict(list)
    for row in rows:
        by_step[row["step_id"]].append(row)
        by_step_layer[(row["step_id"], int(row.get("layer", -1)))].append(row)

    for records in by_step.values():
        for feature in RAW_FEATURES:
            add_minmax(records, feature, f"sample_norm_{feature}")
    for records in by_step_layer.values():
        for feature in RAW_FEATURES:
            add_minmax(records, feature, f"layer_norm_{feature}")

    for row in rows:
        row["sample_norm_text_x_norm"] = row["sample_norm_text_mass"] * row["sample_norm_text_value_norm"]
        row["layer_norm_text_x_norm"] = row["layer_norm_text_mass"] * row["layer_norm_text_value_norm"]
        row["sample_norm_text_ratio_x_value"] = row["sample_norm_text_ratio"] * row["sample_norm_text_value_norm"]
        row["layer_norm_text_ratio_x_value"] = row["layer_norm_text_ratio"] * row["layer_norm_text_value_norm"]
        row["sample_norm_text_ratio_x_entropy"] = row["sample_norm_text_ratio"] * row["sample_norm_img_entropy_norm"]
        row["layer_norm_text_ratio_x_entropy"] = row["layer_norm_text_ratio"] * row["layer_norm_img_entropy_norm"]
        row["sample_norm_value_x_low_visual"] = row["sample_norm_text_value_norm"] * (1.0 - row["sample_norm_visual_value_ratio"])
        row["layer_norm_value_x_low_visual"] = row["layer_norm_text_value_norm"] * (1.0 - row["layer_norm_visual_value_ratio"])
        row["sample_norm_text_x_norm_x_low_visual"] = row["sample_norm_text_x_norm"] * (1.0 - row["sample_norm_visual_value_ratio"])
        row["layer_norm_text_x_norm_x_low_visual"] = row["layer_norm_text_x_norm"] * (1.0 - row["layer_norm_visual_value_ratio"])


def average_ranks(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    idx = 0
    while idx < len(values):
        end = idx + 1
        while end < len(values) and values[order[end]] == values[order[idx]]:
            end += 1
        rank = (idx + 1 + end) / 2.0
        ranks[order[idx:end]] = rank
        idx = end
    return ranks


def auroc(labels, scores):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(set(labels.tolist())) < 2 or len(set(scores.tolist())) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    if len(x) < 2:
        return None
    return pearson(average_ranks(x), average_ranks(y))


def percentile(values, q):
    if not values:
        return None
    return float(np.percentile(np.array(values, dtype=float), q))


def write_csv(path, rows):
    if not rows:
        with open(path, "w") as f:
            f.write("")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def utility_auc_rows(rows, threshold, features):
    output = []
    groups = ["all", "grounded"] + sorted(set(row["label_family"] for row in rows))
    for group in groups:
        if group == "all":
            group_rows = rows
        elif group == "grounded":
            group_rows = [row for row in rows if row["label_family"] in {"lost_grounded", "kept_grounded"}]
        else:
            group_rows = [row for row in rows if row["label_family"] == group]
        if not group_rows:
            continue
        labels = [1 if row["suppression_utility"] > threshold else 0 for row in group_rows]
        utility = [row["suppression_utility"] for row in group_rows]
        for feature in features:
            values = [safe_float(row.get(feature, 0.0)) for row in group_rows]
            auc = auroc(labels, values)
            if auc is None:
                continue
            output.append({
                "group": group,
                "feature": feature,
                "n": len(group_rows),
                "positive_threshold": threshold,
                "n_positive": int(sum(labels)),
                "auroc_high_predicts_should_suppress": auc,
                "auroc_abs": max(auc, 1.0 - auc),
                "direction": "high_predicts_should_suppress" if auc >= 0.5 else "low_predicts_should_suppress",
                "mean_positive": float(np.mean([v for v, y in zip(values, labels) if y])) if sum(labels) else None,
                "mean_negative": float(np.mean([v for v, y in zip(values, labels) if not y])) if sum(labels) < len(labels) else None,
                "spearman_utility": spearman(values, utility),
            })
    group_order = {"all": 0, "grounded": 1, "hallucinated": 2, "lost_grounded": 3, "kept_grounded": 4}
    output.sort(key=lambda row: (group_order.get(row["group"], 99), -row["auroc_abs"]))
    return output


def selector_recovery_rows(rows, threshold, selector_top_k, features, directions=("high", "low")):
    by_step = defaultdict(list)
    for row in rows:
        by_step[row["step_id"]].append(row)
    output = []
    for group in ["all", "hallucinated", "grounded", "lost_grounded", "kept_grounded"]:
        if group == "all":
            step_items = list(by_step.values())
        elif group == "grounded":
            step_items = [
                items for items in by_step.values()
                if items and items[0]["label_family"] in {"lost_grounded", "kept_grounded"}
            ]
        else:
            step_items = [
                items for items in by_step.values()
                if items and items[0]["label_family"] == group
            ]
        if not step_items:
            continue
        for feature in features:
            for direction in directions:
                metrics = []
                for items in step_items:
                    positives = {
                        row["head_key"]
                        for row in items
                        if row["suppression_utility"] > threshold
                    }
                    ranked = sorted(
                        items,
                        key=lambda row: safe_float(row.get(feature, 0.0)),
                        reverse=(direction == "high"),
                    )
                    selected = {row["head_key"] for row in ranked[:selector_top_k]}
                    inter = positives & selected
                    union = positives | selected
                    metrics.append({
                        "positive_count": len(positives),
                        "selected_count": len(selected),
                        "overlap": len(inter),
                        "precision": len(inter) / max(len(selected), 1),
                        "recall": len(inter) / max(len(positives), 1),
                        "jaccard": len(inter) / max(len(union), 1),
                    })
                for metric in ("positive_count", "overlap", "precision", "recall", "jaccard"):
                    values = [row[metric] for row in metrics]
                    output.append({
                        "group": group,
                        "feature": feature,
                        "direction": direction,
                        "selector_top_k": selector_top_k,
                        "metric": metric,
                        "mean": float(np.mean(values)),
                        "p10": percentile(values, 10),
                        "p50": percentile(values, 50),
                        "p90": percentile(values, 90),
                    })
    group_order = {"all": 0, "hallucinated": 1, "grounded": 2, "lost_grounded": 3, "kept_grounded": 4}
    metric_order = {"overlap": 0, "precision": 1, "recall": 2, "jaccard": 3, "positive_count": 4}
    output.sort(key=lambda row: (
        group_order.get(row["group"], 99),
        metric_order.get(row["metric"], 99),
        -row["mean"],
        row["feature"],
        row["direction"],
    ))
    return output


def split_steps(rows, seed, train_frac):
    steps = sorted(set(row["step_id"] for row in rows))
    rng = np.random.default_rng(seed)
    rng.shuffle(steps)
    split = int(round(len(steps) * train_frac))
    train_steps = set(steps[:split])
    train = [row for row in rows if row["step_id"] in train_steps]
    test = [row for row in rows if row["step_id"] not in train_steps]
    return train, test


def matrix(rows, features):
    return np.array([[safe_float(row.get(feature, 0.0)) for feature in features] for row in rows], dtype=float)


def learned_model(rows, threshold, selector_top_k, seed, train_frac):
    train, test = split_steps(rows, seed, train_frac)
    y_train = np.array([1 if row["suppression_utility"] > threshold else 0 for row in train], dtype=int)
    y_test = np.array([1 if row["suppression_utility"] > threshold else 0 for row in test], dtype=int)
    if len(set(y_train.tolist())) < 2 or len(set(y_test.tolist())) < 2:
        return [], []
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
    )
    model.fit(matrix(train, MODEL_FEATURES), y_train)
    scores = model.predict_proba(matrix(test, MODEL_FEATURES))[:, 1]
    metrics = [{
        "model": "logistic_regression",
        "seed": seed,
        "train_rows": len(train),
        "test_rows": len(test),
        "train_positive": int(y_train.sum()),
        "test_positive": int(y_test.sum()),
        "auroc": float(roc_auc_score(y_test, scores)),
        "auprc": float(average_precision_score(y_test, scores)),
    }]
    scored_rows = []
    for row, score in zip(test, scores):
        scored = dict(row)
        scored["model_score"] = float(score)
        scored_rows.append(scored)
    recovery = selector_recovery_rows(scored_rows, threshold, selector_top_k, ["model_score"], directions=("high",))
    for row in recovery:
        row["model"] = "logistic_regression"
        row["seed"] = seed
    return metrics, recovery


def add_utility_columns(rows, threshold):
    output = []
    for row in rows:
        item = dict(row)
        item["should_suppress"] = bool(row["suppression_utility"] > threshold)
        item["harm_if_suppressed"] = bool(row["causal_effect"] > threshold and row["label_family"] in {"lost_grounded", "kept_grounded"})
        output.append(item)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--positive-utility-threshold", type=float, default=0.02)
    parser.add_argument("--selector-top-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.7)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rows = load_rows(args.teacher_jsonl)
    features = list(dict.fromkeys(BASE_SELECTOR_FEATURES + MODEL_FEATURES))

    auc_rows = utility_auc_rows(rows, args.positive_utility_threshold, features)
    recovery_rows = selector_recovery_rows(rows, args.positive_utility_threshold, args.selector_top_k, features)
    model_metrics, model_recovery = learned_model(
        rows,
        args.positive_utility_threshold,
        args.selector_top_k,
        args.seed,
        args.train_frac,
    )

    write_csv(os.path.join(args.output_dir, "suppression_utility_rows.csv"), add_utility_columns(rows, args.positive_utility_threshold))
    write_csv(os.path.join(args.output_dir, "feature_suppression_utility_auc.csv"), auc_rows)
    write_csv(os.path.join(args.output_dir, "selector_suppression_utility_recovery.csv"), recovery_rows)
    write_csv(os.path.join(args.output_dir, "learned_suppression_utility_metrics.csv"), model_metrics)
    write_csv(os.path.join(args.output_dir, "learned_selector_recovery.csv"), model_recovery)

    summary = {
        "teacher_jsonl": args.teacher_jsonl,
        "num_rows": len(rows),
        "positive_utility_threshold": args.positive_utility_threshold,
        "selector_top_k": args.selector_top_k,
        "label_counts": {},
        "should_suppress_count": int(sum(row["suppression_utility"] > args.positive_utility_threshold for row in rows)),
    }
    for row in rows:
        summary["label_counts"][row["label_family"]] = summary["label_counts"].get(row["label_family"], 0) + 1
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print("top all-group utility AUC features:")
    for row in [r for r in auc_rows if r["group"] == "all"][:12]:
        print(row["feature"], row["auroc_high_predicts_should_suppress"], row["direction"], row["n_positive"])
    if model_metrics:
        print("learned model:", model_metrics[0])


if __name__ == "__main__":
    main()

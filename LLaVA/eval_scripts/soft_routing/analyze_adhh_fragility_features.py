import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict, deque

from eval_scripts.soft_routing.analyze_pairwise_object_inventory import (
    load_eval,
    node_pairs,
    normalize_image_id,
    row_key,
    safe_rate,
    write_csv,
)


NON_FEATURE_COLUMNS = {
    "image_id",
    "question_id",
    "image",
    "caption",
    "object_word",
    "word",
    "node_word",
    "label",
    "label_name",
    "token_pos",
    "token_ids",
    "prior_source",
    "outcome",
    "fragility_family",
    "mention_key",
    "base_object_word",
    "base_node_word",
    "base_label_name",
    "target_status",
    "feature_available",
}


def safe_float(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    return value


def mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def percentile(values, q):
    values = sorted(value for value in values if value is not None)
    if not values:
        return None
    idx = int(round((len(values) - 1) * q / 100.0))
    return values[idx]


def auc_score(labels, scores):
    pairs = [(float(score), int(label)) for label, score in zip(labels, scores) if score is not None]
    if not pairs:
        return None
    positives = sum(label for _, label in pairs)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return None
    pairs.sort(key=lambda item: item[0])
    rank_sum = 0.0
    idx = 0
    while idx < len(pairs):
        end = idx + 1
        while end < len(pairs) and pairs[end][0] == pairs[idx][0]:
            end += 1
        avg_rank = (idx + 1 + end) / 2.0
        rank_sum += avg_rank * sum(label for _, label in pairs[idx:end])
        idx = end
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def load_feature_rows(path):
    if not path:
        return []
    if path.endswith(".jsonl"):
        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def feature_key(image_id, label_name, node_word, occurrence_idx):
    return (
        normalize_image_id(image_id),
        str(label_name),
        str(node_word),
        int(occurrence_idx),
    )


def label_name_from_row(row):
    label_name = str(row.get("label_name", "")).strip().lower()
    if label_name in {"grounded", "hallucinated"}:
        return label_name
    label = row.get("label")
    try:
        return "hallucinated" if int(float(label)) == 1 else "grounded"
    except (TypeError, ValueError):
        return label_name


def build_feature_index(feature_rows):
    counters = defaultdict(int)
    index = {}
    numeric_features = set()
    for row in feature_rows:
        image_id = row.get("image_id") or row.get("question_id") or row.get("image")
        label_name = label_name_from_row(row)
        node_word = row.get("node_word", "")
        occurrence_idx = counters[(normalize_image_id(image_id), label_name, node_word)]
        counters[(normalize_image_id(image_id), label_name, node_word)] += 1
        key = feature_key(image_id, label_name, node_word, occurrence_idx)
        index[key] = row
        for column, value in row.items():
            if column in NON_FEATURE_COLUMNS:
                continue
            if safe_float(value) is not None:
                numeric_features.add(column)
    return index, sorted(numeric_features)


def add_derived_features(rows, numeric_features):
    numeric_features = set(numeric_features)
    specs = [
        ("neg_top1_top2_margin", lambda row: -safe_float(row.get("top1_top2_margin"), 0.0)),
        ("entropy_x_mean_i_text", lambda row: safe_float(row.get("entropy")) * safe_float(row.get("mean_i_text"))),
        ("entropy_x_mean_excess", lambda row: safe_float(row.get("entropy")) * safe_float(row.get("mean_excess"))),
        ("entropy_x_sum_excess", lambda row: safe_float(row.get("entropy")) * safe_float(row.get("sum_excess"))),
        (
            "entropy_x_mean_triggered_text_mass",
            lambda row: safe_float(row.get("entropy")) * safe_float(row.get("mean_triggered_text_mass")),
        ),
        (
            "entropy_x_weighted_percentile_active_count",
            lambda row: safe_float(row.get("entropy")) * safe_float(row.get("weighted_percentile_active_count")),
        ),
        (
            "mean_excess_x_neg_margin",
            lambda row: safe_float(row.get("mean_excess")) * -safe_float(row.get("top1_top2_margin")),
        ),
    ]
    for row in rows:
        if not row["feature_available"]:
            continue
        for name, fn in specs:
            try:
                value = fn(row)
            except (TypeError, ValueError):
                value = None
            value = safe_float(value)
            if value is not None:
                row[name] = value
                numeric_features.add(name)
    return sorted(numeric_features)


def mention_list(sentence):
    mentions = []
    for label_name, key in (
        ("grounded", "mscoco_non_hallucinated_words"),
        ("hallucinated", "mscoco_hallucinated_words"),
    ):
        for word, node_word in node_pairs(sentence, key):
            mentions.append({
                "object_word": word,
                "node_word": node_word,
                "label_name": label_name,
            })
    counters = defaultdict(int)
    for mention in mentions:
        occurrence_idx = counters[(mention["label_name"], mention["node_word"])]
        counters[(mention["label_name"], mention["node_word"])] += 1
        mention["occurrence_idx"] = occurrence_idx
    return mentions


def target_queues(target):
    queues = {
        "grounded": defaultdict(deque),
        "hallucinated": defaultdict(deque),
    }
    for label_name, key in (
        ("grounded", "mscoco_non_hallucinated_words"),
        ("hallucinated", "mscoco_hallucinated_words"),
    ):
        for word, node_word in node_pairs(target, key):
            queues[label_name][node_word].append(word)
    return queues


def assign_outcome(mention, queues):
    node_word = mention["node_word"]
    label_name = mention["label_name"]
    if label_name == "grounded":
        if queues["grounded"][node_word]:
            queues["grounded"][node_word].popleft()
            return "grounded_retained", "retained"
        if queues["hallucinated"][node_word]:
            queues["hallucinated"][node_word].popleft()
            return "grounded_to_hallucinated", "damaged"
        return "grounded_lost", "damaged"

    if queues["hallucinated"][node_word]:
        queues["hallucinated"][node_word].popleft()
        return "hallucinated_retained", "retained"
    if queues["grounded"][node_word]:
        queues["grounded"][node_word].popleft()
        return "hallucinated_to_grounded", "fixed"
    return "hallucinated_removed", "fixed"


def build_rows(base_by_id, target_by_id, feature_index, numeric_features):
    rows = []
    for key in sorted(set(base_by_id) & set(target_by_id)):
        base = base_by_id[key]
        target = target_by_id[key]
        queues = target_queues(target)
        image_id = row_key(base)
        for mention in mention_list(base):
            outcome, target_status = assign_outcome(mention, queues)
            fkey = feature_key(
                image_id,
                mention["label_name"],
                mention["node_word"],
                mention["occurrence_idx"],
            )
            feature_row = feature_index.get(fkey, {})
            row = {
                "image_id": image_id,
                "image": base.get("image") or target.get("image"),
                "base_object_word": mention["object_word"],
                "base_node_word": mention["node_word"],
                "base_label_name": mention["label_name"],
                "base_occurrence_idx": mention["occurrence_idx"],
                "outcome": outcome,
                "target_status": target_status,
                "feature_available": int(bool(feature_row)),
            }
            for feature in numeric_features:
                value = safe_float(feature_row.get(feature))
                if value is not None:
                    row[feature] = value
            rows.append(row)
    return rows


def summarize_counts(rows, base_overall, target_overall, base_name, target_name):
    counts = Counter(row["outcome"] for row in rows)
    grounded_total = sum(counts[key] for key in ("grounded_retained", "grounded_lost", "grounded_to_hallucinated"))
    hallucinated_total = sum(counts[key] for key in ("hallucinated_retained", "hallucinated_removed", "hallucinated_to_grounded"))
    grounded_damaged = counts["grounded_lost"] + counts["grounded_to_hallucinated"]
    hallucinated_fixed = counts["hallucinated_removed"] + counts["hallucinated_to_grounded"]
    grounded_damage_rate = safe_rate(grounded_damaged, grounded_total)
    hallucinated_fix_rate = safe_rate(hallucinated_fixed, hallucinated_total)
    return [{
        "base_name": base_name,
        "target_name": target_name,
        "base_CHAIRs": base_overall.get("CHAIRs"),
        "target_CHAIRs": target_overall.get("CHAIRs"),
        "base_CHAIRi": base_overall.get("CHAIRi"),
        "target_CHAIRi": target_overall.get("CHAIRi"),
        "n_mentions": len(rows),
        "feature_available_rate": mean(row["feature_available"] for row in rows),
        "grounded_total": grounded_total,
        "grounded_retained": counts["grounded_retained"],
        "grounded_lost": counts["grounded_lost"],
        "grounded_to_hallucinated": counts["grounded_to_hallucinated"],
        "grounded_damage_rate": grounded_damage_rate,
        "hallucinated_total": hallucinated_total,
        "hallucinated_retained": counts["hallucinated_retained"],
        "hallucinated_removed": counts["hallucinated_removed"],
        "hallucinated_to_grounded": counts["hallucinated_to_grounded"],
        "hallucinated_fix_rate": hallucinated_fix_rate,
        "fragility_ratio": (
            hallucinated_fix_rate / grounded_damage_rate
            if hallucinated_fix_rate is not None and grounded_damage_rate not in (None, 0)
            else None
        ),
    }]


def coverage_summary(rows):
    groups = [("all", rows)]
    for label_name in sorted(set(row["base_label_name"] for row in rows)):
        groups.append((f"label:{label_name}", [row for row in rows if row["base_label_name"] == label_name]))
    for outcome in sorted(set(row["outcome"] for row in rows)):
        groups.append((f"outcome:{outcome}", [row for row in rows if row["outcome"] == outcome]))
    for target_status in sorted(set(row["target_status"] for row in rows)):
        groups.append((f"target_status:{target_status}", [row for row in rows if row["target_status"] == target_status]))

    output = []
    for group, group_rows in groups:
        n_total = len(group_rows)
        n_feature = sum(int(row["feature_available"]) for row in group_rows)
        missing = Counter(row["base_node_word"] for row in group_rows if not row["feature_available"])
        output.append({
            "group": group,
            "n_total": n_total,
            "n_feature_available": n_feature,
            "feature_available_rate": safe_rate(n_feature, n_total),
            "n_missing": n_total - n_feature,
            "top_missing_node_words": ";".join(
                f"{word}:{count}" for word, count in missing.most_common(20)
            ),
        })
    return output


def group_feature_summary(rows, numeric_features):
    output = []
    for group in sorted(set(row["outcome"] for row in rows)):
        group_rows = [row for row in rows if row["outcome"] == group and row["feature_available"]]
        for feature in numeric_features:
            values = [safe_float(row.get(feature)) for row in group_rows]
            values = [value for value in values if value is not None]
            if not values:
                continue
            output.append({
                "group": group,
                "feature": feature,
                "n": len(values),
                "mean": mean(values),
                "p25": percentile(values, 25),
                "p50": percentile(values, 50),
                "p75": percentile(values, 75),
                "p90": percentile(values, 90),
            })
    return output


def contrast_auc(rows, numeric_features):
    contrasts = [
        ("hall_removed_vs_retained", {"hallucinated_removed", "hallucinated_to_grounded"}, {"hallucinated_retained"}),
        ("grounded_lost_vs_retained", {"grounded_lost", "grounded_to_hallucinated"}, {"grounded_retained"}),
        ("desired_fixed_hall_vs_damaged_grounded", {"hallucinated_removed", "hallucinated_to_grounded"}, {"grounded_lost", "grounded_to_hallucinated"}),
        ("side_effect_grounded_lost_vs_hall_removed", {"grounded_lost", "grounded_to_hallucinated"}, {"hallucinated_removed", "hallucinated_to_grounded"}),
    ]
    output = []
    for name, positive_outcomes, negative_outcomes in contrasts:
        subset = [
            row for row in rows
            if row["feature_available"] and row["outcome"] in (positive_outcomes | negative_outcomes)
        ]
        labels = [1 if row["outcome"] in positive_outcomes else 0 for row in subset]
        for feature in numeric_features:
            values = [safe_float(row.get(feature)) for row in subset]
            pairs = [(label, value) for label, value in zip(labels, values) if value is not None]
            if not pairs:
                continue
            kept_labels = [label for label, _ in pairs]
            kept_values = [value for _, value in pairs]
            pos_values = [value for label, value in pairs if label == 1]
            neg_values = [value for label, value in pairs if label == 0]
            if not pos_values or not neg_values:
                continue
            output.append({
                "contrast": name,
                "feature": feature,
                "n_pos": len(pos_values),
                "n_neg": len(neg_values),
                "pos_mean": mean(pos_values),
                "neg_mean": mean(neg_values),
                "pos_minus_neg": mean(pos_values) - mean(neg_values),
                "auroc_high_predicts_pos": auc_score(kept_labels, kept_values),
            })
    output.sort(key=lambda row: (
        row["contrast"],
        -abs((row["auroc_high_predicts_pos"] or 0.5) - 0.5),
        row["feature"],
    ))
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-eval", required=True)
    parser.add_argument("--target-eval", required=True)
    parser.add_argument("--object-step-features", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-name", default="greedy")
    parser.add_argument("--target-name", default="adhh_hard")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    base_by_id, base_overall = load_eval(args.base_eval)
    target_by_id, target_overall = load_eval(args.target_eval)
    feature_rows = load_feature_rows(args.object_step_features)
    feature_index, numeric_features = build_feature_index(feature_rows)
    rows = build_rows(base_by_id, target_by_id, feature_index, numeric_features)
    numeric_features = add_derived_features(rows, numeric_features)

    summary = summarize_counts(rows, base_overall, target_overall, args.base_name, args.target_name)
    coverage_rows = coverage_summary(rows)
    group_summary = group_feature_summary(rows, numeric_features)
    auc_rows = contrast_auc(rows, numeric_features)

    write_csv(os.path.join(args.output_dir, "adhh_fragility_mention_rows.csv"), rows)
    write_csv(os.path.join(args.output_dir, "adhh_fragility_summary.csv"), summary)
    write_csv(os.path.join(args.output_dir, "adhh_fragility_feature_coverage.csv"), coverage_rows)
    write_csv(os.path.join(args.output_dir, "adhh_fragility_feature_group_summary.csv"), group_summary)
    write_csv(os.path.join(args.output_dir, "adhh_fragility_feature_auc.csv"), auc_rows)
    with open(os.path.join(args.output_dir, "adhh_fragility_summary.json"), "w") as f:
        json.dump({
            "base_eval": args.base_eval,
            "target_eval": args.target_eval,
            "object_step_features": args.object_step_features,
            "summary": summary[0] if summary else {},
            "numeric_features": numeric_features,
        }, f, indent=2)
    print("[summary] AD-HH fragility")
    print(summary[0] if summary else {})


if __name__ == "__main__":
    main()

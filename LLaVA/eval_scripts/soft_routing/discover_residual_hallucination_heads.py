import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict

from eval_scripts.soft_routing.analyze_adhh_fragility_features import (
    build_rows,
    feature_key,
)
from eval_scripts.soft_routing.analyze_pairwise_object_inventory import load_eval
from eval_scripts.soft_routing.head_prior_utils import default_heads_for_model, head_key, parse_head_key


DEFAULT_PRIMARY_POSITIVES = "hallucinated_retained"
DEFAULT_HALLUCINATED_OUTCOMES = "hallucinated_retained,hallucinated_removed,hallucinated_to_grounded"
DEFAULT_GROUNDED_OUTCOMES = "grounded_retained,grounded_lost,grounded_to_hallucinated"


def write_csv(path, rows):
    if not rows:
        with open(path, "w") as f:
            f.write("")
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    return value


def safe_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def mean(values):
    values = [value for value in values if value is not None and math.isfinite(value)]
    return float(sum(values) / len(values)) if values else None


def positive_mean(values):
    values = [max(0.0, value) for value in values if value is not None and math.isfinite(value)]
    return float(sum(values) / len(values)) if values else None


def average_ranks(values):
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    idx = 0
    while idx < len(values):
        end = idx + 1
        while end < len(values) and values[order[end]] == values[order[idx]]:
            end += 1
        rank = (idx + 1 + end) / 2.0
        for order_idx in order[idx:end]:
            ranks[order_idx] = rank
        idx = end
    return ranks


def auroc(labels, scores):
    pairs = [
        (float(score), int(label))
        for label, score in zip(labels, scores)
        if score is not None and math.isfinite(float(score))
    ]
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


def split_csv_set(text):
    return {item.strip() for item in str(text).replace(" ", ",").split(",") if item.strip()}


def load_head_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def build_outcome_index(base_eval, target_eval):
    base_by_id, _ = load_eval(base_eval)
    target_by_id, _ = load_eval(target_eval)
    outcome_rows = build_rows(base_by_id, target_by_id, {}, [])
    output = {}
    for row in outcome_rows:
        key = feature_key(
            row["image_id"],
            row["base_label_name"],
            row["base_node_word"],
            row["base_occurrence_idx"],
        )
        output[key] = row
    return output, outcome_rows


def attach_outcomes(head_rows, outcome_index):
    output = []
    missing = Counter()
    for row in head_rows:
        label_name = str(row.get("label_name", "")).strip()
        if label_name not in {"grounded", "hallucinated"}:
            label_name = "hallucinated" if safe_int(row.get("label"), 0) == 1 else "grounded"
        key = feature_key(
            row.get("image_id") or row.get("question_id") or row.get("image"),
            label_name,
            row.get("node_word", ""),
            safe_int(row.get("occurrence_idx"), 0),
        )
        outcome = outcome_index.get(key)
        if not outcome:
            missing[(label_name, row.get("node_word", ""))] += 1
            continue
        merged = dict(row)
        merged["outcome"] = outcome["outcome"]
        merged["target_status"] = outcome["target_status"]
        merged["base_label_name"] = outcome["base_label_name"]
        merged["base_node_word"] = outcome["base_node_word"]
        merged["base_occurrence_idx"] = outcome["base_occurrence_idx"]
        output.append(merged)
    return output, missing


def maybe_filter_rows(rows, column, min_value, max_value):
    if not column:
        return rows
    output = []
    for row in rows:
        value = safe_float(row.get(column))
        if value is None:
            continue
        if min_value is not None and value < min_value:
            continue
        if max_value is not None and value > max_value:
            continue
        output.append(row)
    return output


def summarize_feature(rows, feature):
    values = [safe_float(row.get(feature)) for row in rows]
    values = [value for value in values if value is not None]
    return {
        f"mean_{feature}": mean(values),
        f"positive_mean_{feature}": positive_mean(values),
        f"sum_positive_{feature}": sum(max(0.0, value) for value in values) if values else None,
    }


def score_heads(rows, args):
    primary_outcomes = split_csv_set(args.primary_positive_outcomes)
    hallucinated_outcomes = split_csv_set(args.hallucinated_outcomes)
    grounded_outcomes = split_csv_set(args.grounded_outcomes)
    teacher = args.teacher_feature
    secondary_teacher = args.secondary_teacher_feature
    proxy_features = [
        "text_mass",
        "proxy_text_target_logit",
        "proxy_evidence_gap_target",
        "proxy_img_target_logit",
        "full_entropy_norm",
        "text_entropy_norm",
        "img_mass",
    ]

    adhh_heads = default_heads_for_model(args.model_path)
    adhh_rank = {head_key(layer, head): idx + 1 for idx, (layer, head) in enumerate(adhh_heads)}

    grouped = defaultdict(list)
    for row in rows:
        key = row.get("head_key") or head_key(row.get("layer"), row.get("head"))
        grouped[key].append(row)

    output = []
    for key, items in sorted(grouped.items()):
        layer, head = parse_head_key(key)
        primary = [row for row in items if row["outcome"] in primary_outcomes]
        hallucinated = [row for row in items if row["outcome"] in hallucinated_outcomes]
        grounded = [row for row in items if row["outcome"] in grounded_outcomes]
        grounded_lost = [row for row in items if row["outcome"] in {"grounded_lost", "grounded_to_hallucinated"}]
        removed = [row for row in items if row["outcome"] in {"hallucinated_removed", "hallucinated_to_grounded"}]

        primary_values = [safe_float(row.get(teacher)) for row in primary]
        hallucinated_values = [safe_float(row.get(teacher)) for row in hallucinated]
        grounded_values = [safe_float(row.get(teacher)) for row in grounded]
        grounded_lost_values = [safe_float(row.get(teacher)) for row in grounded_lost]
        removed_values = [safe_float(row.get(teacher)) for row in removed]

        primary_pos = positive_mean(primary_values)
        hallucinated_pos = positive_mean(hallucinated_values)
        grounded_pos = positive_mean(grounded_values)
        grounded_lost_pos = positive_mean(grounded_lost_values)
        removed_pos = positive_mean(removed_values)

        score = (
            (primary_pos or 0.0)
            - float(args.grounded_penalty_alpha) * (grounded_pos or 0.0)
            - float(args.grounded_lost_penalty_alpha) * (grounded_lost_pos or 0.0)
        )
        hallucination_score = (
            (hallucinated_pos or 0.0)
            - float(args.grounded_penalty_alpha) * (grounded_pos or 0.0)
        )
        residual_gain = (primary_pos or 0.0) - (removed_pos or 0.0)

        contrast_rows = primary + grounded
        labels = [1 if row["outcome"] in primary_outcomes else 0 for row in contrast_rows]
        contrast_values = [safe_float(row.get(teacher)) for row in contrast_rows]

        row = {
            "layer": layer,
            "head": head,
            "head_key": key,
            "score": score,
            "hallucination_score": hallucination_score,
            "residual_gain_vs_removed": residual_gain,
            "teacher_feature": teacher,
            "secondary_teacher_feature": secondary_teacher,
            "n_rows": len(items),
            "n_primary": len(primary),
            "n_hallucinated": len(hallucinated),
            "n_removed": len(removed),
            "n_grounded": len(grounded),
            "n_grounded_lost": len(grounded_lost),
            "primary_positive_mean_teacher": primary_pos,
            "hallucinated_positive_mean_teacher": hallucinated_pos,
            "removed_positive_mean_teacher": removed_pos,
            "grounded_positive_mean_teacher": grounded_pos,
            "grounded_lost_positive_mean_teacher": grounded_lost_pos,
            "primary_mean_teacher": mean([value for value in primary_values if value is not None]),
            "hallucinated_mean_teacher": mean([value for value in hallucinated_values if value is not None]),
            "removed_mean_teacher": mean([value for value in removed_values if value is not None]),
            "grounded_mean_teacher": mean([value for value in grounded_values if value is not None]),
            "grounded_lost_mean_teacher": mean([value for value in grounded_lost_values if value is not None]),
            "primary_vs_grounded_auroc": auroc(labels, contrast_values),
            "in_adhh_default": int(key in adhh_rank),
            "adhh_rank": adhh_rank.get(key, ""),
        }
        if secondary_teacher:
            row.update({
                f"primary_positive_mean_{secondary_teacher}": positive_mean(
                    [safe_float(item.get(secondary_teacher)) for item in primary]
                ),
                f"hallucinated_positive_mean_{secondary_teacher}": positive_mean(
                    [safe_float(item.get(secondary_teacher)) for item in hallucinated]
                ),
                f"grounded_positive_mean_{secondary_teacher}": positive_mean(
                    [safe_float(item.get(secondary_teacher)) for item in grounded]
                ),
            })
        for feature in proxy_features:
            for prefix, group_rows in (
                ("primary", primary),
                ("hallucinated", hallucinated),
                ("grounded", grounded),
            ):
                summary = summarize_feature(group_rows, feature)
                for name, value in summary.items():
                    row[f"{prefix}_{name}"] = value
        output.append(row)

    output.sort(key=lambda row: (
        -safe_float(row.get("score"), -1e30),
        -safe_float(row.get("hallucination_score"), -1e30),
        row["head_key"],
    ))
    for idx, row in enumerate(output, start=1):
        row["rank"] = idx
    return output


def write_prior(path, rows, top_k, score_field, metadata):
    selected = rows[: int(top_k)] if top_k else rows
    contrastive_scores = []
    for row in rows:
        contrastive_scores.append({
            "layer": int(row["layer"]),
            "head": int(row["head"]),
            "head_key": row["head_key"],
            "score": safe_float(row.get(score_field), 0.0),
            "rank": int(row["rank"]),
        })
    payload = {
        "hal_heads": [[int(row["layer"]), int(row["head"])] for row in selected],
        "contrastive_scores": contrastive_scores,
        "metadata": metadata,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return payload


def selection_summary(rows, top_ks):
    output = []
    for top_k in top_ks:
        selected = rows[: int(top_k)]
        output.append({
            "top_k": int(top_k),
            "n_selected": len(selected),
            "adhh_overlap": sum(int(row.get("in_adhh_default", 0)) for row in selected),
            "mean_score": mean([safe_float(row.get("score")) for row in selected]),
            "mean_primary_positive_teacher": mean([
                safe_float(row.get("primary_positive_mean_teacher")) for row in selected
            ]),
            "mean_grounded_positive_teacher": mean([
                safe_float(row.get("grounded_positive_mean_teacher")) for row in selected
            ]),
            "heads": ",".join(row["head_key"] for row in selected),
        })
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-eval", required=True)
    parser.add_argument("--target-eval", required=True)
    parser.add_argument("--head-rows", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--teacher-feature", default="target_text_logprob_drop")
    parser.add_argument("--secondary-teacher-feature", default="target_logprob_drop")
    parser.add_argument("--primary-positive-outcomes", default=DEFAULT_PRIMARY_POSITIVES)
    parser.add_argument("--hallucinated-outcomes", default=DEFAULT_HALLUCINATED_OUTCOMES)
    parser.add_argument("--grounded-outcomes", default=DEFAULT_GROUNDED_OUTCOMES)
    parser.add_argument("--grounded-penalty-alpha", type=float, default=0.15)
    parser.add_argument("--grounded-lost-penalty-alpha", type=float, default=0.05)
    parser.add_argument("--filter-column", default="")
    parser.add_argument("--filter-min", type=float, default=None)
    parser.add_argument("--filter-max", type=float, default=None)
    parser.add_argument("--prior-top-k", type=int, default=20)
    parser.add_argument("--summary-top-ks", default="5,10,20")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    outcome_index, outcome_rows = build_outcome_index(args.base_eval, args.target_eval)
    raw_head_rows = load_head_rows(args.head_rows)
    rows, missing = attach_outcomes(raw_head_rows, outcome_index)
    rows = maybe_filter_rows(rows, args.filter_column, args.filter_min, args.filter_max)
    scored = score_heads(rows, args)
    top_ks = [safe_int(item, 0) for item in str(args.summary_top_ks).replace(" ", ",").split(",") if item.strip()]
    top_ks = [item for item in top_ks if item > 0]
    summary_rows = selection_summary(scored, top_ks)

    write_csv(os.path.join(args.output_dir, "residual_hallucination_head_scores.csv"), scored)
    write_csv(os.path.join(args.output_dir, "residual_hallucination_head_selection_summary.csv"), summary_rows)
    write_csv(os.path.join(args.output_dir, "head_rows_with_outcomes.csv"), rows)

    metadata = {
        "base_eval": args.base_eval,
        "target_eval": args.target_eval,
        "head_rows": args.head_rows,
        "teacher_feature": args.teacher_feature,
        "secondary_teacher_feature": args.secondary_teacher_feature,
        "primary_positive_outcomes": sorted(split_csv_set(args.primary_positive_outcomes)),
        "hallucinated_outcomes": sorted(split_csv_set(args.hallucinated_outcomes)),
        "grounded_outcomes": sorted(split_csv_set(args.grounded_outcomes)),
        "grounded_penalty_alpha": args.grounded_penalty_alpha,
        "grounded_lost_penalty_alpha": args.grounded_lost_penalty_alpha,
        "filter_column": args.filter_column,
        "filter_min": args.filter_min,
        "filter_max": args.filter_max,
        "n_outcome_mentions": len(outcome_rows),
        "n_head_rows_raw": len(raw_head_rows),
        "n_head_rows_scored": len(rows),
        "n_heads_scored": len(scored),
        "missing_outcome_rows": sum(missing.values()),
        "top_missing_outcomes": [
            {"label_name": key[0], "node_word": key[1], "count": count}
            for key, count in missing.most_common(20)
        ],
    }
    prior_path = os.path.join(args.output_dir, "residual_hallucination_head_prior.json")
    prior = write_prior(prior_path, scored, args.prior_top_k, "score", metadata)
    heads_txt_path = os.path.join(args.output_dir, "residual_hallucination_heads.txt")
    with open(heads_txt_path, "w") as f:
        f.write(",".join(head_key(layer, head) for layer, head in prior["hal_heads"]))
        f.write("\n")
    with open(os.path.join(args.output_dir, "residual_hallucination_head_discovery_summary.json"), "w") as f:
        json.dump({
            **metadata,
            "outputs": {
                "scores": os.path.join(args.output_dir, "residual_hallucination_head_scores.csv"),
                "selection_summary": os.path.join(args.output_dir, "residual_hallucination_head_selection_summary.csv"),
                "rows_with_outcomes": os.path.join(args.output_dir, "head_rows_with_outcomes.csv"),
                "head_prior": prior_path,
                "heads_txt": heads_txt_path,
            },
            "selection_summary": summary_rows,
            "top_heads": scored[: min(20, len(scored))],
        }, f, indent=2)

    print("[summary] residual hallucination head discovery")
    print(json.dumps({
        "n_head_rows_raw": len(raw_head_rows),
        "n_head_rows_scored": len(rows),
        "n_heads_scored": len(scored),
        "missing_outcome_rows": sum(missing.values()),
        "top_heads": [
            {
                "rank": row["rank"],
                "head_key": row["head_key"],
                "score": row["score"],
                "n_primary": row["n_primary"],
                "primary_positive_mean_teacher": row["primary_positive_mean_teacher"],
                "grounded_positive_mean_teacher": row["grounded_positive_mean_teacher"],
                "in_adhh_default": row["in_adhh_default"],
            }
            for row in scored[:10]
        ],
        "selection_summary": summary_rows,
    }, indent=2))


if __name__ == "__main__":
    main()

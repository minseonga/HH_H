import argparse
import csv
import json
import math
import os
from collections import defaultdict

import numpy as np


def safe_float(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    return value


def load_rows(path):
    rows = []
    if path.endswith(".csv"):
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows.extend(dict(row) for row in reader)
    else:
        with open(path, "r") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    for row in rows:
        family = row.get("label_family") or row.get("label", "")
        if str(family).startswith("hallucinated"):
            family = "hallucinated"
        row["label_family"] = family
        if "suppression_utility" not in row:
            effect = safe_float(row.get("causal_effect", 0.0))
            row["suppression_utility"] = effect if family == "hallucinated" else -effect
        for key in [
            "text_mass",
            "img_mass",
            "text_ratio",
            "text_value_norm",
            "img_value_norm",
            "visual_value_ratio",
            "text_img_value_cosine",
            "text_img_value_abs_cosine",
            "text_img_value_orthogonality",
            "supported_text_value_norm",
            "unsupported_text_value_norm",
            "unsupported_text_value_ratio",
            "unsupported_total_value_ratio",
            "suppression_utility",
            "causal_effect",
        ]:
            row[key] = safe_float(row.get(key, 0.0))
    return rows


def parse_float_list(value):
    return [float(item) for item in str(value).split(",") if item.strip()]


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


def mean_value(rows, key):
    if not rows:
        return None
    return float(np.mean([safe_float(row.get(key, 0.0)) for row in rows]))


def percentile_value(rows, key, percentile):
    if not rows:
        return None
    values = np.asarray([safe_float(row.get(key, 0.0)) for row in rows], dtype=np.float64)
    return float(np.percentile(values, percentile))


def std_value(rows, key):
    if not rows:
        return None
    values = np.asarray([safe_float(row.get(key, 0.0)) for row in rows], dtype=np.float64)
    return float(np.std(values))


def high_score_auc(labels, scores):
    labels = [int(label) for label in labels]
    scores = [safe_float(score, 0.0) for score in scores]
    n_positive = sum(labels)
    n_negative = len(labels) - n_positive
    if n_positive == 0 or n_negative == 0:
        return None

    order = sorted(range(len(scores)), key=lambda idx: scores[idx])
    ranks = [0.0] * len(scores)
    idx = 0
    while idx < len(order):
        end = idx + 1
        while end < len(order) and scores[order[end]] == scores[order[idx]]:
            end += 1
        average_rank = (idx + 1 + end) / 2.0
        for rank_idx in range(idx, end):
            ranks[order[rank_idx]] = average_rank
        idx = end

    rank_sum_positive = sum(rank for rank, label in zip(ranks, labels) if label)
    auc = (rank_sum_positive - n_positive * (n_positive + 1) / 2.0) / (n_positive * n_negative)
    return float(auc)


def get_group_rows(rows):
    groups = {"all": rows}
    for family in sorted(set(row["label_family"] for row in rows)):
        groups[family] = [row for row in rows if row["label_family"] == family]
    groups["grounded"] = [row for row in rows if row["label_family"] in {"kept_grounded", "lost_grounded"}]
    return groups


def summarize_case(name, rows, mask, group, base_masks):
    selected = [row for row in rows if mask(row)]
    n_group = len(rows)
    base_counts = {
        key: sum(1 for row in rows if base_mask(row))
        for key, base_mask in base_masks.items()
    }
    output = {
        "group": group,
        "case": name,
        "n": len(selected),
        "n_group": n_group,
        "rate_group": len(selected) / max(n_group, 1),
        "should_suppress_rate": mean_value(selected, "should_suppress"),
        "mean_utility": mean_value(selected, "suppression_utility"),
        "mean_causal_effect": mean_value(selected, "causal_effect"),
        "mean_text_mass": mean_value(selected, "text_mass"),
        "mean_img_mass": mean_value(selected, "img_mass"),
        "mean_text_value_norm": mean_value(selected, "text_value_norm"),
        "mean_img_value_norm": mean_value(selected, "img_value_norm"),
        "mean_visual_value_ratio": mean_value(selected, "visual_value_ratio"),
        "mean_text_img_value_cosine": mean_value(selected, "text_img_value_cosine"),
        "mean_text_img_value_abs_cosine": mean_value(selected, "text_img_value_abs_cosine"),
        "mean_supported_text_value_norm": mean_value(selected, "supported_text_value_norm"),
        "mean_unsupported_text_value_norm": mean_value(selected, "unsupported_text_value_norm"),
        "mean_unsupported_text_value_ratio": mean_value(selected, "unsupported_text_value_ratio"),
        "mean_unsupported_total_value_ratio": mean_value(selected, "unsupported_total_value_ratio"),
    }
    for key, count in base_counts.items():
        output[f"n_{key}"] = count
        output[f"rate_within_{key}"] = len(selected) / count if count else None
    return output


def build_case_summary(rows, adhh_threshold, utility_threshold, parallel_cos, orth_abs_cos):
    for row in rows:
        row["is_text_heavy"] = row["text_mass"] >= adhh_threshold
        row["is_balanced_or_nontrigger"] = row["text_mass"] < adhh_threshold
        row["is_parallel"] = row["text_img_value_cosine"] >= parallel_cos
        row["is_orthogonal"] = row["text_img_value_abs_cosine"] <= orth_abs_cos
        row["should_suppress"] = row["suppression_utility"] > utility_threshold
        row["safe_to_keep"] = not row["should_suppress"]

    case_masks = {
        "case1_text_heavy_parallel_safe": (
            lambda row: row["is_text_heavy"] and row["is_parallel"] and row["safe_to_keep"]
        ),
        "case1_base_text_heavy_parallel": (
            lambda row: row["is_text_heavy"] and row["is_parallel"]
        ),
        "case2_text_heavy_orthogonal_should_suppress": (
            lambda row: row["is_text_heavy"] and row["is_orthogonal"] and row["should_suppress"]
        ),
        "case3_balanced_orthogonal_missed": (
            lambda row: row["is_balanced_or_nontrigger"] and row["is_orthogonal"] and row["should_suppress"]
        ),
        "case3_base_balanced_orthogonal": (
            lambda row: row["is_balanced_or_nontrigger"] and row["is_orthogonal"]
        ),
        "adhh_trigger_should_suppress": (
            lambda row: row["is_text_heavy"] and row["should_suppress"]
        ),
        "adhh_trigger_safe_to_keep": (
            lambda row: row["is_text_heavy"] and row["safe_to_keep"]
        ),
        "adhh_missed_should_suppress": (
            lambda row: row["is_balanced_or_nontrigger"] and row["should_suppress"]
        ),
    }
    base_masks = {
        "text_heavy": lambda row: row["is_text_heavy"],
        "balanced_or_nontrigger": lambda row: row["is_balanced_or_nontrigger"],
        "should_suppress": lambda row: row["should_suppress"],
        "safe_to_keep": lambda row: row["safe_to_keep"],
    }
    groups = get_group_rows(rows)

    output = []
    for group, group_rows in groups.items():
        if not group_rows:
            continue
        for name, mask in case_masks.items():
            output.append(summarize_case(name, group_rows, mask, group, base_masks))
    return output


def build_threshold_sweep(rows, adhh_threshold, utility_threshold, parallel_values, orth_values):
    sweep_rows = []
    for parallel_cos in parallel_values:
        for orth_abs_cos in orth_values:
            summary = build_case_summary(
                [dict(row) for row in rows],
                adhh_threshold,
                utility_threshold,
                parallel_cos,
                orth_abs_cos,
            )
            for row in summary:
                if row["case"] not in {
                    "case1_text_heavy_parallel_safe",
                    "case1_base_text_heavy_parallel",
                    "case3_balanced_orthogonal_missed",
                    "case3_base_balanced_orthogonal",
                }:
                    continue
                output = {
                    "parallel_cos_threshold": parallel_cos,
                    "orthogonal_abs_cos_threshold": orth_abs_cos,
                    **row,
                }
                sweep_rows.append(output)
    return sweep_rows


ALIGNMENT_FEATURES = [
    "text_img_value_cosine",
    "text_img_value_abs_cosine",
    "supported_text_value_norm",
    "unsupported_text_value_norm",
    "unsupported_text_value_ratio",
    "unsupported_total_value_ratio",
    "text_value_norm",
    "img_value_norm",
    "visual_value_ratio",
    "text_mass",
    "img_mass",
    "suppression_utility",
]


def build_alignment_populations(rows):
    return {
        "text_heavy": [row for row in rows if row["is_text_heavy"]],
        "case1_base_text_heavy_parallel": [
            row for row in rows if row["is_text_heavy"] and row["is_parallel"]
        ],
        "case1_safe_text_heavy_parallel": [
            row for row in rows if row["is_text_heavy"] and row["is_parallel"] and row["safe_to_keep"]
        ],
    }


def summarize_distribution(population_name, group, rows, feature):
    return {
        "population": population_name,
        "group": group,
        "feature": feature,
        "n": len(rows),
        "mean": mean_value(rows, feature),
        "std": std_value(rows, feature),
        "p10": percentile_value(rows, feature, 10),
        "p25": percentile_value(rows, feature, 25),
        "p50": percentile_value(rows, feature, 50),
        "p75": percentile_value(rows, feature, 75),
        "p90": percentile_value(rows, feature, 90),
        "min": percentile_value(rows, feature, 0),
        "max": percentile_value(rows, feature, 100),
        "safe_to_keep_rate": mean_value(rows, "safe_to_keep"),
        "should_suppress_rate": mean_value(rows, "should_suppress"),
        "mean_utility": mean_value(rows, "suppression_utility"),
    }


def build_case1_alignment_distribution(rows):
    output = []
    for population_name, population_rows in build_alignment_populations(rows).items():
        groups = get_group_rows(population_rows)
        for group, group_rows in groups.items():
            if not group_rows:
                continue
            for feature in ALIGNMENT_FEATURES:
                output.append(summarize_distribution(population_name, group, group_rows, feature))
    return output


def build_case1_grounded_hallucinated_contrast(rows):
    output = []
    for population_name, population_rows in build_alignment_populations(rows).items():
        grounded_rows = [
            row for row in population_rows
            if row["label_family"] in {"kept_grounded", "lost_grounded"}
        ]
        hallucinated_rows = [
            row for row in population_rows
            if row["label_family"] == "hallucinated"
        ]
        comparison_rows = grounded_rows + hallucinated_rows
        labels = [1] * len(grounded_rows) + [0] * len(hallucinated_rows)
        for feature in ALIGNMENT_FEATURES:
            scores = [safe_float(row.get(feature, 0.0)) for row in comparison_rows]
            auc = high_score_auc(labels, scores)
            output.append({
                "population": population_name,
                "feature": feature,
                "n_grounded": len(grounded_rows),
                "n_hallucinated": len(hallucinated_rows),
                "grounded_mean": mean_value(grounded_rows, feature),
                "hallucinated_mean": mean_value(hallucinated_rows, feature),
                "grounded_minus_hallucinated_mean": (
                    mean_value(grounded_rows, feature) - mean_value(hallucinated_rows, feature)
                    if grounded_rows and hallucinated_rows else None
                ),
                "grounded_p25": percentile_value(grounded_rows, feature, 25),
                "grounded_p50": percentile_value(grounded_rows, feature, 50),
                "grounded_p75": percentile_value(grounded_rows, feature, 75),
                "hallucinated_p25": percentile_value(hallucinated_rows, feature, 25),
                "hallucinated_p50": percentile_value(hallucinated_rows, feature, 50),
                "hallucinated_p75": percentile_value(hallucinated_rows, feature, 75),
                "auroc_high_predicts_grounded": auc,
                "auroc_abs": max(auc, 1.0 - auc) if auc is not None else None,
            })
    return output


def build_case1_rho_threshold_sweep(rows, rho_values):
    output = []
    text_heavy_rows = [row for row in rows if row["is_text_heavy"]]
    groups = get_group_rows(text_heavy_rows)
    for rho in rho_values:
        for group, group_rows in groups.items():
            if not group_rows:
                continue
            protected = [
                row for row in group_rows
                if row["text_img_value_cosine"] >= rho
            ]
            safe_rows = [row for row in group_rows if row["safe_to_keep"]]
            should_rows = [row for row in group_rows if row["should_suppress"]]
            protected_safe = [row for row in protected if row["safe_to_keep"]]
            protected_should = [row for row in protected if row["should_suppress"]]
            output.append({
                "rho": rho,
                "group": group,
                "n_text_heavy": len(group_rows),
                "n_protected": len(protected),
                "protected_rate": len(protected) / max(len(group_rows), 1),
                "n_safe_to_keep": len(safe_rows),
                "n_should_suppress": len(should_rows),
                "n_protected_safe_to_keep": len(protected_safe),
                "n_protected_should_suppress": len(protected_should),
                "protected_safe_precision": len(protected_safe) / max(len(protected), 1) if protected else None,
                "protected_should_suppress_rate": len(protected_should) / max(len(protected), 1) if protected else None,
                "safe_recall": len(protected_safe) / max(len(safe_rows), 1) if safe_rows else None,
                "should_suppress_leak_rate": (
                    len(protected_should) / max(len(should_rows), 1) if should_rows else None
                ),
                "mean_protected_utility": mean_value(protected, "suppression_utility"),
                "mean_protected_cosine": mean_value(protected, "text_img_value_cosine"),
                "mean_unprotected_utility": mean_value(
                    [row for row in group_rows if row["text_img_value_cosine"] < rho],
                    "suppression_utility",
                ),
                "mean_unprotected_cosine": mean_value(
                    [row for row in group_rows if row["text_img_value_cosine"] < rho],
                    "text_img_value_cosine",
                ),
            })
    return output


def build_examples(rows, adhh_threshold, utility_threshold, parallel_cos, orth_abs_cos, max_examples):
    summary_rows = build_case_summary(rows, adhh_threshold, utility_threshold, parallel_cos, orth_abs_cos)
    case_names = {
        "case1_text_heavy_parallel_safe",
        "case3_balanced_orthogonal_missed",
    }
    case_by_name = {row["case"]: row for row in summary_rows if row["group"] == "all" and row["case"] in case_names}
    examples = []
    for row in rows:
        case = None
        if row["is_text_heavy"] and row["is_parallel"] and row["safe_to_keep"]:
            case = "case1_text_heavy_parallel_safe"
        elif row["is_balanced_or_nontrigger"] and row["is_orthogonal"] and row["should_suppress"]:
            case = "case3_balanced_orthogonal_missed"
        if case is None:
            continue
        examples.append({
            "case": case,
            "step_id": row.get("step_id"),
            "image_id": row.get("image_id"),
            "label_family": row.get("label_family"),
            "object_node": row.get("object_node"),
            "target_token": row.get("target_token"),
            "layer": row.get("layer"),
            "head": row.get("head"),
            "head_key": row.get("head_key"),
            "text_mass": row.get("text_mass"),
            "img_mass": row.get("img_mass"),
            "text_value_norm": row.get("text_value_norm"),
            "img_value_norm": row.get("img_value_norm"),
            "visual_value_ratio": row.get("visual_value_ratio"),
            "text_img_value_cosine": row.get("text_img_value_cosine"),
            "text_img_value_abs_cosine": row.get("text_img_value_abs_cosine"),
            "supported_text_value_norm": row.get("supported_text_value_norm"),
            "unsupported_text_value_norm": row.get("unsupported_text_value_norm"),
            "unsupported_text_value_ratio": row.get("unsupported_text_value_ratio"),
            "unsupported_total_value_ratio": row.get("unsupported_total_value_ratio"),
            "suppression_utility": row.get("suppression_utility"),
            "causal_effect": row.get("causal_effect"),
            "target_logprob_original": row.get("target_logprob_original"),
            "target_logprob_single_head_zero": row.get("target_logprob_single_head_zero"),
            "case_n_all": case_by_name.get(case, {}).get("n"),
        })
    examples.sort(key=lambda row: abs(safe_float(row.get("suppression_utility"))), reverse=True)
    by_case = defaultdict(list)
    for row in examples:
        by_case[row["case"]].append(row)
    limited = []
    for case in sorted(by_case):
        limited.extend(by_case[case][:max_examples])
    return limited


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--adhh-threshold", type=float, default=0.4)
    parser.add_argument("--positive-utility-threshold", type=float, default=0.02)
    parser.add_argument("--parallel-cos-threshold", type=float, default=0.3)
    parser.add_argument("--orthogonal-abs-cos-threshold", type=float, default=0.1)
    parser.add_argument("--parallel-sweep", default="0.2,0.3,0.4,0.5")
    parser.add_argument("--orthogonal-sweep", default="0.05,0.1,0.15,0.2")
    parser.add_argument("--case1-rho-sweep", default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7")
    parser.add_argument("--max-examples", type=int, default=25)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rows = load_rows(args.teacher_jsonl)
    if not rows:
        raise ValueError("No rows found")
    if "text_img_value_cosine" not in rows[0] or all(row["text_img_value_cosine"] == 0.0 for row in rows):
        raise ValueError(
            "Missing nonzero text_img_value_cosine. Re-run online_causal_head_teacher with the updated diagnostics."
        )

    summary = build_case_summary(
        rows,
        args.adhh_threshold,
        args.positive_utility_threshold,
        args.parallel_cos_threshold,
        args.orthogonal_abs_cos_threshold,
    )
    sweep = build_threshold_sweep(
        rows,
        args.adhh_threshold,
        args.positive_utility_threshold,
        parse_float_list(args.parallel_sweep),
        parse_float_list(args.orthogonal_sweep),
    )
    examples = build_examples(
        rows,
        args.adhh_threshold,
        args.positive_utility_threshold,
        args.parallel_cos_threshold,
        args.orthogonal_abs_cos_threshold,
        args.max_examples,
    )
    alignment_distribution = build_case1_alignment_distribution(rows)
    alignment_contrast = build_case1_grounded_hallucinated_contrast(rows)
    alignment_rho_sweep = build_case1_rho_threshold_sweep(
        rows,
        parse_float_list(args.case1_rho_sweep),
    )

    write_csv(os.path.join(args.output_dir, "visual_anchor_case_summary.csv"), summary)
    write_csv(os.path.join(args.output_dir, "visual_anchor_case_threshold_sweep.csv"), sweep)
    write_csv(os.path.join(args.output_dir, "visual_anchor_case_examples.csv"), examples)
    write_csv(
        os.path.join(args.output_dir, "case1_alignment_distribution.csv"),
        alignment_distribution,
    )
    write_csv(
        os.path.join(args.output_dir, "case1_alignment_grounded_vs_hallucinated.csv"),
        alignment_contrast,
    )
    write_csv(
        os.path.join(args.output_dir, "case1_alignment_threshold_sweep.csv"),
        alignment_rho_sweep,
    )

    config = {
        "teacher_jsonl": args.teacher_jsonl,
        "n_rows": len(rows),
        "adhh_threshold": args.adhh_threshold,
        "positive_utility_threshold": args.positive_utility_threshold,
        "parallel_cos_threshold": args.parallel_cos_threshold,
        "orthogonal_abs_cos_threshold": args.orthogonal_abs_cos_threshold,
        "case1_rho_sweep": args.case1_rho_sweep,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(json.dumps(config, indent=2))
    print("case summary:")
    for row in summary:
        if row["group"] == "all" and row["case"] in {
            "case1_text_heavy_parallel_safe",
            "case1_base_text_heavy_parallel",
            "case3_balanced_orthogonal_missed",
            "case3_base_balanced_orthogonal",
        }:
            print(row["case"], row["n"], row["rate_group"], row["mean_utility"])
    print("case1 grounded-vs-hallucinated alignment:")
    for row in alignment_contrast:
        if row["population"] in {"text_heavy", "case1_base_text_heavy_parallel"} and row["feature"] in {
            "text_img_value_cosine",
            "supported_text_value_norm",
            "unsupported_text_value_norm",
            "unsupported_total_value_ratio",
        }:
            print(row)


if __name__ == "__main__":
    main()

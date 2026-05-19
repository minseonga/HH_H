import argparse
import csv
import json
import os
from collections import Counter, defaultdict

import numpy as np
import torch
from tqdm import tqdm

from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from eval_scripts.soft_routing.analyze_object_retention_steps import (
    build_prompt_inputs,
    configure_model,
    load_sentences,
    one_step,
    select_rows,
)
from eval_scripts.soft_routing.head_prior_utils import default_heads_for_model, head_key


FEATURES = [
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
    "soft_alpha",
    "soft_strength",
]


SELECTOR_FEATURES = [
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
    "layer_norm_text_mass",
    "layer_norm_text_value_norm",
    "layer_norm_text_x_norm",
    "layer_norm_text_ratio_x_value",
    "layer_norm_text_ratio_x_entropy",
    "sample_norm_text_mass",
    "sample_norm_text_value_norm",
    "sample_norm_text_x_norm",
    "sample_norm_text_ratio_x_value",
    "sample_norm_text_ratio_x_entropy",
]


def safe_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(value):
        return 0.0
    return value


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


def auroc(labels, scores):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = average_ranks(scores)
    rank_sum_pos = float(ranks[pos].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


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


def set_all_head_diagnostics(model, enabled):
    setattr(model.config, "record_all_head_diagnostics", bool(enabled))


def one_step_all_head_diagnostics(model, tokenizer, prompt_ids, prefix_ids, image_tensor, image_size, target_token_id):
    old_record_all = bool(getattr(model.config, "record_all_head_diagnostics", False))
    set_all_head_diagnostics(model, True)
    output = one_step(
        model,
        tokenizer,
        prompt_ids,
        prefix_ids,
        image_tensor,
        image_size,
        target_token_id,
        "none",
        record=True,
    )
    set_all_head_diagnostics(model, old_record_all)
    return output


def single_head_zero(
    model,
    tokenizer,
    prompt_ids,
    prefix_ids,
    image_tensor,
    image_size,
    target_token_id,
    layer,
    head,
    ablation_threshold,
):
    old_heads = getattr(model.config, "hal_attention_heads", None)
    old_threshold = float(getattr(model.config, "adhh_threshold", 0.0))
    model.config.hal_attention_heads = [[int(layer), int(head)]]
    model.config.adhh_threshold = float(ablation_threshold)
    try:
        output = one_step(
            model,
            tokenizer,
            prompt_ids,
            prefix_ids,
            image_tensor,
            image_size,
            target_token_id,
            "fixed:1.0",
            record=False,
        )
    finally:
        model.config.hal_attention_heads = old_heads
        model.config.adhh_threshold = old_threshold
    return output


def normalize_records(records):
    by_layer = defaultdict(list)
    for record in records:
        record["layer"] = int(record["layer"])
        record["head"] = int(record["head"])
        record["head_key"] = record.get("head_key") or head_key(record["layer"], record["head"])
        for feature in FEATURES:
            record[feature] = safe_float(record.get(feature, 0.0))
        by_layer[record["layer"]].append(record)

    for feature in FEATURES:
        add_minmax(records, feature, f"sample_norm_{feature}")
    for layer_records in by_layer.values():
        for feature in FEATURES:
            add_minmax(layer_records, feature, f"layer_norm_{feature}")
    for record in records:
        record["sample_norm_text_x_norm"] = (
            record["sample_norm_text_mass"] * record["sample_norm_text_value_norm"]
        )
        record["layer_norm_text_x_norm"] = (
            record["layer_norm_text_mass"] * record["layer_norm_text_value_norm"]
        )
        record["sample_norm_text_ratio_x_value"] = (
            record["sample_norm_text_ratio"] * record["sample_norm_text_value_norm"]
        )
        record["layer_norm_text_ratio_x_value"] = (
            record["layer_norm_text_ratio"] * record["layer_norm_text_value_norm"]
        )
        record["sample_norm_text_ratio_x_entropy"] = (
            record["sample_norm_text_ratio"] * record["sample_norm_img_entropy_norm"]
        )
        record["layer_norm_text_ratio_x_entropy"] = (
            record["layer_norm_text_ratio"] * record["layer_norm_img_entropy_norm"]
        )
    return records


def add_minmax(records, feature, output_key):
    values = np.array([safe_float(record.get(feature, 0.0)) for record in records], dtype=float)
    if len(values) == 0:
        return
    low = float(np.min(values))
    high = float(np.max(values))
    denom = high - low
    for record in records:
        record[output_key] = 0.0 if denom <= 1e-12 else float((record[feature] - low) / denom)


def select_candidate_records(records, policy, max_heads, text_tau):
    def add_unique(output, candidates):
        seen = {record["head_key"] for record in output}
        for record in candidates:
            if max_heads > 0 and len(output) >= max_heads:
                break
            if record["head_key"] in seen:
                continue
            output.append(record)
            seen.add(record["head_key"])
        return output

    if policy == "all":
        selected = sorted(records, key=lambda record: record["text_mass"], reverse=True)
        return selected if max_heads <= 0 else selected[:max_heads]
    if policy == "text_triggered":
        selected = [record for record in records if record["text_mass"] >= text_tau]
        return sorted(selected, key=lambda record: record["text_mass"], reverse=True)[:max_heads]
    if policy == "text_topk":
        return sorted(records, key=lambda record: record["text_mass"], reverse=True)[:max_heads]
    if policy == "norm_topk":
        return sorted(records, key=lambda record: record["text_value_norm"], reverse=True)[:max_heads]
    if policy == "text_norm_union":
        selected = {}
        for record in sorted(records, key=lambda item: item["text_mass"], reverse=True)[:max_heads]:
            selected[record["head_key"]] = record
        for record in sorted(records, key=lambda item: item["text_value_norm"], reverse=True)[:max_heads]:
            selected[record["head_key"]] = record
        return sorted(selected.values(), key=lambda record: record["text_mass"] * record["text_value_norm"], reverse=True)
    if policy == "balanced_orthogonal":
        balanced = [record for record in records if record["text_mass"] < text_tau]
        selected = sorted(
            balanced,
            key=lambda record: (
                record.get("unsupported_text_value_norm", 0.0),
                -record.get("text_img_value_abs_cosine", 1.0),
                record.get("text_value_norm", 0.0),
            ),
            reverse=True,
        )
        return selected[:max_heads]
    if policy == "anchor_mixed":
        if max_heads <= 0:
            max_heads = len(records)
        text_quota = max(1, max_heads // 3)
        balanced_quota = max(1, max_heads // 3)
        selected = []
        add_unique(
            selected,
            sorted(records, key=lambda record: record["text_mass"], reverse=True)[:text_quota],
        )
        balanced = [record for record in records if record["text_mass"] < text_tau]
        add_unique(
            selected,
            sorted(
                balanced,
                key=lambda record: (
                    record.get("unsupported_text_value_norm", 0.0),
                    -record.get("text_img_value_abs_cosine", 1.0),
                    record.get("text_value_norm", 0.0),
                ),
                reverse=True,
            )[:balanced_quota],
        )
        add_unique(
            selected,
            sorted(records, key=lambda record: record.get("unsupported_text_value_norm", 0.0), reverse=True),
        )
        add_unique(
            selected,
            sorted(records, key=lambda record: record["text_value_norm"], reverse=True),
        )
        return selected[:max_heads]
    raise ValueError(f"Unknown candidate_policy={policy}")


def label_family(label):
    if str(label).startswith("hallucinated_object"):
        return "hallucinated"
    if str(label) == "lost_grounded":
        return "lost_grounded"
    if str(label) == "kept_grounded":
        return "kept_grounded"
    return str(label)


def analyze_teacher_rows(rows, positive_effect_threshold, selector_top_k):
    correlation_rows = []
    auc_rows = []
    selector_rows = []
    groups = sorted(set(["all"] + [row["label_family"] for row in rows]))

    for group in groups:
        group_rows = rows if group == "all" else [row for row in rows if row["label_family"] == group]
        if not group_rows:
            continue
        labels = [1 if row["causal_effect"] > positive_effect_threshold else 0 for row in group_rows]
        effects = [row["causal_effect"] for row in group_rows]
        for feature in SELECTOR_FEATURES:
            values = [safe_float(row.get(feature, 0.0)) for row in group_rows]
            auc = auroc(labels, values)
            correlation_rows.append({
                "group": group,
                "feature": feature,
                "n": len(group_rows),
                "pearson_effect": pearson(values, effects),
                "spearman_effect": spearman(values, effects),
                "mean_feature": float(np.mean(values)),
                "mean_effect": float(np.mean(effects)),
            })
            if auc is not None:
                auc_rows.append({
                    "group": group,
                    "feature": feature,
                    "n": len(group_rows),
                    "positive_threshold": positive_effect_threshold,
                    "n_positive": int(sum(labels)),
                    "auroc_high_predicts_positive_effect": auc,
                    "auroc_abs": max(auc, 1.0 - auc),
                    "direction": "high_predicts_positive_effect" if auc >= 0.5 else "low_predicts_positive_effect",
                })

    step_groups = defaultdict(list)
    for row in rows:
        step_groups[row["step_id"]].append(row)
    for step_id, items in step_groups.items():
        teacher_top = {
            row["head_key"]
            for row in sorted(items, key=lambda item: item["causal_effect"], reverse=True)[:selector_top_k]
        }
        for feature in SELECTOR_FEATURES:
            selected = {
                row["head_key"]
                for row in sorted(items, key=lambda item: safe_float(item.get(feature, 0.0)), reverse=True)[:selector_top_k]
            }
            inter = teacher_top & selected
            union = teacher_top | selected
            selector_rows.append({
                "step_id": step_id,
                "label": items[0]["label"],
                "label_family": items[0]["label_family"],
                "feature": feature,
                "selector_top_k": selector_top_k,
                "overlap": len(inter),
                "precision": len(inter) / max(len(selected), 1),
                "recall": len(inter) / max(len(teacher_top), 1),
                "jaccard": len(inter) / max(len(union), 1),
                "selected_count": len(selected),
                "teacher_count": len(teacher_top),
            })

    selector_summary = []
    by_feature_group = defaultdict(list)
    for row in selector_rows:
        by_feature_group[(row["label_family"], row["feature"])].append(row)
        by_feature_group[("all", row["feature"])].append(row)
    for (group, feature), items in sorted(by_feature_group.items()):
        selector_summary.append({
            "group": group,
            "feature": feature,
            "selector_top_k": selector_top_k,
            "n_steps": len(items),
            "mean_overlap": float(np.mean([row["overlap"] for row in items])),
            "mean_precision": float(np.mean([row["precision"] for row in items])),
            "mean_jaccard": float(np.mean([row["jaccard"] for row in items])),
            "p50_overlap": float(np.percentile([row["overlap"] for row in items], 50)),
            "p90_overlap": float(np.percentile([row["overlap"] for row in items], 90)),
        })

    correlation_rows.sort(key=lambda row: abs(row["spearman_effect"] or 0.0), reverse=True)
    auc_rows.sort(key=lambda row: row["auroc_abs"], reverse=True)
    selector_summary.sort(key=lambda row: (row["group"], -row["mean_overlap"]))
    return correlation_rows, auc_rows, selector_summary, selector_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hard-results", required=True)
    parser.add_argument("--soft-results", required=True)
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prior-path", default="")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--model-path", default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--conv-mode", default="vicuna_v1")
    parser.add_argument("--max-per-label", type=int, default=20)
    parser.add_argument("--hallucinated-source", type=str, default="both", choices=["soft", "hard", "both"])
    parser.add_argument(
        "--candidate-policy",
        default="text_topk",
        choices=[
            "text_triggered",
            "text_topk",
            "norm_topk",
            "text_norm_union",
            "all",
            "balanced_orthogonal",
            "anchor_mixed",
        ],
    )
    parser.add_argument("--candidate-max-heads", type=int, default=32)
    parser.add_argument("--candidate-text-tau", type=float, default=0.4)
    parser.add_argument("--ablation-threshold", type=float, default=0.0)
    parser.add_argument("--positive-effect-threshold", type=float, default=0.02)
    parser.add_argument("--selector-top-k", type=int, default=8)
    parser.add_argument("--adhh-threshold", type=float, default=0.4)
    parser.add_argument("--soft-gamma", type=float, default=0.75)
    parser.add_argument("--soft-temperature", type=float, default=0.05)
    parser.add_argument("--resume", action="store_true", default=False)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    jsonl_path = os.path.join(args.output_dir, "online_causal_head_teacher.jsonl")
    done = set()
    if args.resume and os.path.exists(jsonl_path):
        with open(jsonl_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                done.add((row["step_id"], row["head_key"]))

    disable_torch_init()
    model_name = get_model_name_from_path(os.path.expanduser(args.model_path))
    tokenizer, model, image_processor, _ = load_pretrained_model(args.model_path, args.model_base, model_name)
    heads, priors, prior_source = configure_model(
        model,
        args.model_path,
        args.prior_path,
        args.top_k,
        args.adhh_threshold,
        args.soft_gamma,
        args.soft_temperature,
    )

    hard_by_id = load_sentences(args.hard_results)
    soft_by_id = load_sentences(args.soft_results)
    selected_steps = select_rows(hard_by_id, soft_by_id, tokenizer, args.max_per_label, args.hallucinated_source)

    mode = "a" if args.resume else "w"
    step_summaries = []
    with open(jsonl_path, mode) as out:
        for step_idx, step in enumerate(tqdm(selected_steps)):
            prompt_ids, image_tensor, image_size = build_prompt_inputs(
                step, args.image_folder, tokenizer, image_processor, model.config, args.conv_mode
            )
            prompt_ids = prompt_ids.to(device="cuda", non_blocking=True)
            image_tensor = image_tensor.to(dtype=torch.float16, device="cuda", non_blocking=True)
            prefix_ids = step["probe_caption_ids"][:step["target_token_pos"]]
            target_token_id = int(step["target_token_id"])
            step_id = f'{step["image_id"]}:{step["caption_source"]}:{step["label"]}:{step["object_node"]}:{step["target_token_pos"]}'

            original = one_step_all_head_diagnostics(
                model,
                tokenizer,
                prompt_ids,
                prefix_ids,
                image_tensor,
                image_size,
                target_token_id,
            )
            diagnostics = normalize_records(list(original["diagnostics"]))
            candidates = select_candidate_records(
                diagnostics,
                args.candidate_policy,
                args.candidate_max_heads,
                args.candidate_text_tau,
            )

            step_summaries.append({
                "step_id": step_id,
                "image_id": step["image_id"],
                "image": step["image"],
                "label": step["label"],
                "label_family": label_family(step["label"]),
                "caption_source": step["caption_source"],
                "object_node": step["object_node"],
                "object_word": step["object_word"],
                "target_token": tokenizer.decode([target_token_id]),
                "target_token_id": target_token_id,
                "target_logprob_original": original["target_logprob"],
                "original_target_rank": original["target_rank"],
                "candidate_count": len(candidates),
            })

            for record in candidates:
                key = record["head_key"]
                if (step_id, key) in done:
                    continue
                ablated = single_head_zero(
                    model,
                    tokenizer,
                    prompt_ids,
                    prefix_ids,
                    image_tensor,
                    image_size,
                    target_token_id,
                    record["layer"],
                    record["head"],
                    args.ablation_threshold,
                )
                causal_effect = original["target_logprob"] - ablated["target_logprob"]
                output = {
                    "step_id": step_id,
                    "image_id": step["image_id"],
                    "image": step["image"],
                    "label": step["label"],
                    "label_family": label_family(step["label"]),
                    "caption_source": step["caption_source"],
                    "object_node": step["object_node"],
                    "object_word": step["object_word"],
                    "target_token": tokenizer.decode([target_token_id]),
                    "target_token_id": target_token_id,
                    "target_token_pos": int(step["target_token_pos"]),
                    "target_logprob_original": original["target_logprob"],
                    "target_logprob_single_head_zero": ablated["target_logprob"],
                    "causal_effect": causal_effect,
                    "target_rank_original": original["target_rank"],
                    "target_rank_single_head_zero": ablated["target_rank"],
                    "rank_delta": ablated["target_rank"] - original["target_rank"],
                    "candidate_policy": args.candidate_policy,
                    "candidate_max_heads": args.candidate_max_heads,
                    "ablation_threshold": args.ablation_threshold,
                    "prior_source": prior_source,
                    **{feature: record.get(feature, 0.0) for feature in FEATURES},
                    **{feature: record.get(feature, 0.0) for feature in SELECTOR_FEATURES if feature not in FEATURES},
                    "layer": record["layer"],
                    "head": record["head"],
                    "head_key": key,
                }
                out.write(json.dumps(output) + "\n")
                out.flush()

    rows = []
    with open(jsonl_path, "r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    correlation_rows, auc_rows, selector_summary, selector_rows = analyze_teacher_rows(
        rows,
        args.positive_effect_threshold,
        args.selector_top_k,
    )

    write_csv(os.path.join(args.output_dir, "step_summary.csv"), step_summaries)
    write_csv(os.path.join(args.output_dir, "feature_effect_correlations.csv"), correlation_rows)
    write_csv(os.path.join(args.output_dir, "feature_positive_effect_auc.csv"), auc_rows)
    write_csv(os.path.join(args.output_dir, "selector_recovery_summary.csv"), selector_summary)
    write_csv(os.path.join(args.output_dir, "selector_recovery_by_step.csv"), selector_rows)
    summary = {
        "num_steps": len(selected_steps),
        "num_teacher_rows": len(rows),
        "label_counts": dict(Counter(row["label"] for row in rows)),
        "candidate_policy": args.candidate_policy,
        "candidate_max_heads": args.candidate_max_heads,
        "candidate_text_tau": args.candidate_text_tau,
        "ablation_threshold": args.ablation_threshold,
        "positive_effect_threshold": args.positive_effect_threshold,
        "selector_top_k": args.selector_top_k,
        "prior_source": prior_source,
        "heads": heads,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print("top hallucinated positive-effect AUC features:")
    for row in [r for r in auc_rows if r["group"] == "hallucinated"][:10]:
        print(row["feature"], row["auroc_high_predicts_positive_effect"], row["direction"], row["n_positive"])
    print("top hallucinated selector recovery:")
    for row in [r for r in selector_summary if r["group"] == "hallucinated"][:10]:
        print(row["feature"], row["mean_overlap"], row["mean_precision"], row["mean_jaccard"])


if __name__ == "__main__":
    main()

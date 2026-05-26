import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from eval_scripts.soft_routing.analyze_query_direction_chair_alignment import (
    build_prompt_inputs,
    filter_sentences,
    image_id_key,
    image_name_from_sentence,
    load_ids_from_text_or_file,
    prepare_mentions,
    select_directions,
)
from eval_scripts.soft_routing.head_prior_utils import default_heads_for_model, head_key


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
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return None
    return pearson(average_ranks(x), average_ranks(y))


def mean(values):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(sum(values) / len(values)) if values else None


def safe_float(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    return value


def parse_head_key(text):
    layer, head = str(text).strip().split(":", 1)
    return int(layer), int(head)


def parse_head_list(text):
    heads = []
    if not text:
        return heads
    for item in str(text).replace(" ", ",").split(","):
        item = item.strip()
        if not item:
            continue
        heads.append(parse_head_key(item))
    return heads


def dedupe_heads(heads):
    seen = set()
    output = []
    for layer, head in heads:
        key = (int(layer), int(head))
        if key in seen:
            continue
        output.append(key)
        seen.add(key)
    return output


def load_eval_sentences(path):
    with open(path) as f:
        data = json.load(f)
    return data.get("sentences", [])


def select_candidate_heads(args, model_path):
    heads = []
    if args.query_calibration:
        for item in select_directions(
            args.query_calibration,
            top_k=args.query_top_k,
            min_auroc=args.query_min_auroc,
            select_by=args.query_select_by,
        ):
            heads.append((int(item["layer"]), int(item["head"])))
    if args.include_adhh_top_k:
        heads.extend(default_heads_for_model(model_path)[: int(args.include_adhh_top_k)])
    heads.extend(parse_head_list(args.candidate_heads))
    return dedupe_heads(heads)


def get_layers(model):
    base = getattr(model, "model", model)
    layers = getattr(base, "layers", None)
    if layers is not None:
        return layers
    nested = getattr(base, "model", None)
    layers = getattr(nested, "layers", None)
    if layers is not None:
        return layers
    raise AttributeError("Could not locate transformer layers on model")


def clear_interventions(model):
    for name in [
        "adaptive_deactivate",
        "soft_deactivate",
        "dynamic_deactivate",
        "attribution_soft_deactivate",
        "retention_aware_deactivate",
        "fixed_strength_deactivate",
        "query_direction_project",
        "query_logit_correction",
        "query_visual_attention_boost",
        "record_intervention_diagnostics",
    ]:
        if hasattr(model.config, name):
            setattr(model.config, name, False)
    model.config.intervention_diagnostics = None


def set_head_output_recording(model, heads_by_layer, record_components=False):
    model.config.record_head_output_diagnostics = True
    model.config.head_output_diagnostics = []
    model.config.head_output_record_all_heads = False
    model.config.head_output_record_batch_index = 0
    model.config.head_output_record_components = bool(record_components)
    model.config.head_output_record_min_layer = min(heads_by_layer) if heads_by_layer else None
    model.config.head_output_record_max_layer = max(heads_by_layer) if heads_by_layer else None
    model.config.head_output_record_heads_by_layer = {
        int(layer): [int(head) for head in heads]
        for layer, heads in heads_by_layer.items()
    }


def clear_head_output_recording(model):
    model.config.record_head_output_diagnostics = False
    model.config.head_output_diagnostics = None
    model.config.head_output_record_all_heads = False
    model.config.head_output_record_heads = None
    model.config.head_output_record_heads_by_layer = None
    model.config.head_output_record_components = False
    model.config.head_output_record_min_layer = None
    model.config.head_output_record_max_layer = None


def one_step_scores(
    model,
    prompt_ids,
    prefix_ids,
    image_tensor,
    image_size,
    zero_head=None,
    subtract_head_component=None,
    record_head_outputs=False,
    record_components=False,
    heads_by_layer=None,
):
    clear_interventions(model)
    if prefix_ids:
        prefix_tensor = torch.tensor(prefix_ids, device=prompt_ids.device, dtype=prompt_ids.dtype).unsqueeze(0)
        step_input = torch.cat([prompt_ids, prefix_tensor], dim=1)
    else:
        step_input = prompt_ids

    handle = None
    if zero_head is not None or subtract_head_component is not None:
        if zero_head is not None:
            layer, head = zero_head
            subtract_value = None
        else:
            layer, head, subtract_value = subtract_head_component
        layers = get_layers(model)
        module = layers[int(layer)].self_attn.o_proj
        head_dim = int(model.config.hidden_size // model.config.num_attention_heads)
        start = int(head) * head_dim
        end = start + head_dim

        def patch_current_position_head(_module, inputs):
            hidden = inputs[0]
            patched = hidden.clone()
            if subtract_value is None:
                patched[:, -1:, start:end] = 0
            else:
                value = subtract_value.to(device=patched.device, dtype=patched.dtype).view(1, 1, -1)
                patched[:, -1:, start:end] = patched[:, -1:, start:end] - value
            return (patched,)

        handle = module.register_forward_pre_hook(patch_current_position_head)

    if record_head_outputs:
        set_head_output_recording(model, heads_by_layer or {}, record_components=record_components)
    else:
        clear_head_output_recording(model)

    try:
        with torch.inference_mode():
            output = model.generate(
                step_input,
                images=image_tensor,
                image_sizes=[image_size],
                do_sample=False,
                temperature=0,
                top_p=None,
                num_beams=1,
                max_new_tokens=1,
                use_cache=True,
                output_scores=True,
                return_dict_in_generate=True,
            )
    finally:
        if handle is not None:
            handle.remove()

    score = output["scores"][0][0].detach().float()
    log_probs = F.log_softmax(score, dim=-1)
    sorted_ids = torch.argsort(score, descending=True)
    diagnostics = list(getattr(model.config, "head_output_diagnostics", []) or [])
    if record_head_outputs:
        clear_head_output_recording(model)
    return {
        "score": score,
        "log_probs": log_probs,
        "sorted_ids": sorted_ids,
        "top1_id": int(sorted_ids[0].item()),
        "top1_logit": float(score[int(sorted_ids[0])].item()),
        "top1_logprob": float(log_probs[int(sorted_ids[0])].item()),
        "diagnostics": diagnostics,
    }


def head_output_proxy(model, layer, head, head_output, token_id):
    layers = get_layers(model)
    o_proj = layers[int(layer)].self_attn.o_proj
    lm_head = model.get_output_embeddings()
    head_dim = int(model.config.hidden_size // model.config.num_attention_heads)
    start = int(head) * head_dim
    end = start + head_dim
    head_output = head_output.detach().float().cpu()
    o_weight = o_proj.weight.detach().float().cpu()[:, start:end]
    unembed = lm_head.weight.detach().float().cpu()[int(token_id)]
    residual_delta = torch.matmul(head_output, o_weight.transpose(0, 1))
    return float(torch.dot(residual_delta, unembed).item()), float(torch.linalg.vector_norm(residual_delta).item())


def head_component_proxy(model, layer, head, component_value, token_id):
    return head_output_proxy(model, layer, head, component_value, token_id)[0]


def target_rank(sorted_ids, token_id):
    matches = (sorted_ids == int(token_id)).nonzero(as_tuple=False)
    if matches.numel() == 0:
        return None
    return int(matches[0].item() + 1)


def row_label_family(row):
    return "hallucinated" if int(row.get("label", 0)) == 1 else "grounded"


def build_mentions(args, tokenizer):
    sentences = load_eval_sentences(args.eval_results)
    exclude_ids = set()
    for item in args.exclude_image_ids:
        exclude_ids |= load_ids_from_text_or_file(item)
    sentences = filter_sentences(
        sentences,
        match_eval_results=args.match_eval_results,
        exclude_image_ids=exclude_ids,
        max_sentences=args.max_sentences,
    )
    mentions = []
    per_label = Counter()
    for sentence in sentences:
        image_file = image_name_from_sentence(sentence, image_split=args.image_split)
        occurrence_counters = defaultdict(int)
        for mention in prepare_mentions(tokenizer, sentence, score_span="first"):
            family = row_label_family(mention)
            if args.label_filter != "all" and family != args.label_filter:
                continue
            if args.max_per_label and per_label[family] >= args.max_per_label:
                continue
            occurrence_idx = occurrence_counters[(family, mention["node_word"])]
            occurrence_counters[(family, mention["node_word"])] += 1
            positions = mention.get("score_positions") or [mention["token_pos"]]
            token_pos = int(positions[0])
            mentions.append({
                **mention,
                "image_id": image_id_key(sentence),
                "image": image_file,
                "occurrence_idx": occurrence_idx,
                "target_token_pos": token_pos,
                "target_token_id": int(mention["caption_ids"][token_pos]),
            })
            per_label[family] += 1
            if args.max_mentions and len(mentions) >= args.max_mentions:
                return mentions
    return mentions


def summarize_correlations(rows):
    summary = []
    groups = ["all", "hallucinated", "grounded"]
    proxies = [
        "proxy_target_logit",
        "proxy_top1_logit",
        "proxy_target_logit_positive",
        "proxy_top1_logit_positive",
        "proxy_text_target_logit",
        "proxy_img_target_logit",
        "proxy_evidence_gap_target",
        "proxy_text_target_logit_positive",
        "proxy_img_target_logit_positive",
        "proxy_evidence_gap_target_positive",
        "proxy_text_top1_logit",
        "proxy_img_top1_logit",
        "proxy_evidence_gap_top1",
        "proxy_text_top1_logit_positive",
        "proxy_img_top1_logit_positive",
        "proxy_evidence_gap_top1_positive",
        "head_residual_norm",
    ]
    effects = [
        "target_logit_drop",
        "target_logprob_drop",
        "top1_logit_drop",
        "top1_logprob_drop",
        "target_text_logit_drop",
        "target_text_logprob_drop",
        "top1_text_logit_drop",
        "top1_text_logprob_drop",
    ]
    for group in groups:
        group_rows = rows if group == "all" else [row for row in rows if row["label_family"] == group]
        if not group_rows:
            continue
        for proxy in proxies:
            for effect in effects:
                pairs = [
                    (safe_float(row.get(proxy)), safe_float(row.get(effect)))
                    for row in group_rows
                ]
                pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
                if not pairs:
                    continue
                x = [item[0] for item in pairs]
                y = [item[1] for item in pairs]
                summary.append({
                    "group": group,
                    "proxy": proxy,
                    "effect": effect,
                    "n": len(pairs),
                    "pearson": pearson(x, y),
                    "spearman": spearman(x, y),
                    "mean_proxy": mean(x),
                    "mean_effect": mean(y),
                })
    summary.sort(key=lambda row: abs(row["spearman"] or 0.0), reverse=True)
    return summary


def summarize_topk(rows, top_ks):
    by_step = defaultdict(list)
    for row in rows:
        by_step[row["step_id"]].append(row)
    output = []
    for effect in [
        "target_logit_drop",
        "target_logprob_drop",
        "top1_logit_drop",
        "target_text_logit_drop",
        "target_text_logprob_drop",
        "top1_text_logit_drop",
    ]:
        for proxy in [
            "proxy_target_logit",
            "proxy_top1_logit",
            "proxy_target_logit_positive",
            "proxy_text_target_logit",
            "proxy_evidence_gap_target",
            "proxy_text_target_logit_positive",
            "proxy_evidence_gap_target_positive",
        ]:
            for k in top_ks:
                step_rows = []
                for step_id, items in by_step.items():
                    kk = min(int(k), len(items))
                    if kk <= 0:
                        continue
                    effect_items = [row for row in items if safe_float(row.get(effect)) is not None]
                    proxy_items = [row for row in items if safe_float(row.get(proxy)) is not None]
                    if len(effect_items) < kk or len(proxy_items) < kk:
                        continue
                    teacher = {
                        row["head_key"]
                        for row in sorted(effect_items, key=lambda item: safe_float(item.get(effect), -1e30), reverse=True)[:kk]
                    }
                    selected = {
                        row["head_key"]
                        for row in sorted(proxy_items, key=lambda item: safe_float(item.get(proxy), -1e30), reverse=True)[:kk]
                    }
                    inter = teacher & selected
                    union = teacher | selected
                    step_rows.append({
                        "label_family": items[0]["label_family"],
                        "overlap": len(inter),
                        "precision": len(inter) / max(len(selected), 1),
                        "recall": len(inter) / max(len(teacher), 1),
                        "jaccard": len(inter) / max(len(union), 1),
                    })
                for group in ["all", "hallucinated", "grounded"]:
                    group_rows = step_rows if group == "all" else [
                        row for row in step_rows if row["label_family"] == group
                    ]
                    if not group_rows:
                        continue
                    output.append({
                        "group": group,
                        "proxy": proxy,
                        "effect": effect,
                        "top_k": int(k),
                        "n_steps": len(group_rows),
                        "mean_overlap": mean([row["overlap"] for row in group_rows]),
                        "mean_precision": mean([row["precision"] for row in group_rows]),
                        "mean_recall": mean([row["recall"] for row in group_rows]),
                        "mean_jaccard": mean([row["jaccard"] for row in group_rows]),
                    })
    output.sort(key=lambda row: (row["group"], row["top_k"], -(row["mean_jaccard"] or 0.0)))
    return output


def aggregate_mention_features(rows):
    by_step = defaultdict(list)
    for row in rows:
        by_step[row["step_id"]].append(row)

    features = [
        "proxy_target_logit",
        "proxy_top1_logit",
        "proxy_text_target_logit",
        "proxy_img_target_logit",
        "proxy_evidence_gap_target",
        "proxy_text_target_logit_positive",
        "proxy_img_target_logit_positive",
        "proxy_evidence_gap_target_positive",
        "proxy_text_top1_logit",
        "proxy_img_top1_logit",
        "proxy_evidence_gap_top1",
        "proxy_text_top1_logit_positive",
        "proxy_img_top1_logit_positive",
        "proxy_evidence_gap_top1_positive",
        "target_logit_drop",
        "target_logprob_drop",
        "target_text_logit_drop",
        "target_text_logprob_drop",
        "text_mass",
        "img_mass",
        "full_entropy_norm",
        "text_entropy_norm",
    ]
    output = []
    for _, items in sorted(
        by_step.items(),
        key=lambda item: min(int(row.get("mention_index_global", 0)) for row in item[1]),
    ):
        first = items[0]
        row = {
            "image_id": first.get("image_id"),
            "image": first.get("image"),
            "word": first.get("word"),
            "node_word": first.get("node_word"),
            "occurrence_idx": first.get("occurrence_idx"),
            "label": first.get("label"),
            "label_name": first.get("label_name"),
            "token_pos": first.get("token_pos"),
            "target_token_id": first.get("target_token_id"),
            "target_token": first.get("target_token"),
            "top1_token_id": first.get("top1_token_id"),
            "top1_token": first.get("top1_token"),
            "target_rank_original": first.get("target_rank_original"),
            "n_heads": len(items),
        }
        for feature in features:
            values = [safe_float(item.get(feature)) for item in items]
            pairs = [
                (safe_float(item.get(feature)), item.get("head_key"))
                for item in items
                if safe_float(item.get(feature)) is not None
            ]
            values = [value for value in values if value is not None]
            if not values:
                continue
            row[f"mean_{feature}"] = mean(values)
            row[f"sum_{feature}"] = float(sum(values))
            row[f"max_{feature}"] = float(max(values))
            row[f"min_{feature}"] = float(min(values))
            positive = [max(0.0, value) for value in values]
            row[f"sum_positive_{feature}"] = float(sum(positive))
            row[f"mean_positive_{feature}"] = mean(positive)
            best_value, best_head = max(pairs, key=lambda item: item[0])
            row[f"max_{feature}_head"] = best_head
            row[f"max_{feature}_value"] = best_value
        text_sum = safe_float(row.get("sum_positive_proxy_text_target_logit"), 0.0)
        img_sum = safe_float(row.get("sum_positive_proxy_img_target_logit"), 0.0)
        row["sum_positive_text_minus_img_target"] = text_sum - img_sum
        row["sum_positive_text_over_img_target"] = text_sum / max(img_sum, 1e-6)
        output.append(row)
    return output


def summarize_mention_teacher_proxy(mention_rows):
    if not mention_rows:
        return []

    groups = ["all", "hallucinated", "grounded"]
    teachers = [
        "max_target_text_logprob_drop",
        "mean_target_text_logprob_drop",
        "sum_target_text_logprob_drop",
        "sum_positive_target_text_logprob_drop",
        "max_target_text_logit_drop",
        "mean_target_text_logit_drop",
        "sum_positive_target_text_logit_drop",
        "max_target_logprob_drop",
        "mean_target_logprob_drop",
        "sum_positive_target_logprob_drop",
    ]
    proxies = [
        "min_proxy_text_target_logit",
        "mean_proxy_text_target_logit",
        "sum_proxy_text_target_logit",
        "max_proxy_text_target_logit",
        "mean_positive_proxy_text_target_logit",
        "sum_positive_proxy_text_target_logit",
        "min_proxy_evidence_gap_target",
        "mean_proxy_evidence_gap_target",
        "sum_proxy_evidence_gap_target",
        "max_proxy_evidence_gap_target",
        "mean_positive_proxy_evidence_gap_target",
        "sum_positive_proxy_evidence_gap_target",
        "min_proxy_img_target_logit",
        "mean_proxy_img_target_logit",
        "sum_proxy_img_target_logit",
        "max_proxy_img_target_logit",
        "mean_positive_proxy_img_target_logit",
        "sum_positive_proxy_img_target_logit",
        "mean_text_mass",
        "max_text_mass",
        "sum_text_mass",
        "mean_img_mass",
        "max_img_mass",
        "sum_img_mass",
        "mean_full_entropy_norm",
        "max_full_entropy_norm",
        "sum_full_entropy_norm",
        "mean_text_entropy_norm",
        "max_text_entropy_norm",
        "sum_text_entropy_norm",
        "sum_positive_text_minus_img_target",
        "sum_positive_text_over_img_target",
        "occurrence_idx",
        "target_rank_original",
    ]

    output = []
    for group in groups:
        group_rows = mention_rows if group == "all" else [
            row for row in mention_rows if row.get("label_name") == group
        ]
        if not group_rows:
            continue
        for proxy in proxies:
            for teacher in teachers:
                pairs = [
                    (safe_float(row.get(proxy)), safe_float(row.get(teacher)))
                    for row in group_rows
                ]
                pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
                if not pairs:
                    continue
                x = [item[0] for item in pairs]
                y = [item[1] for item in pairs]
                output.append({
                    "group": group,
                    "proxy": proxy,
                    "teacher": teacher,
                    "n_mentions": len(pairs),
                    "pearson": pearson(x, y),
                    "spearman": spearman(x, y),
                    "mean_proxy": mean(x),
                    "mean_teacher": mean(y),
                })
    output.sort(key=lambda row: abs(row["spearman"] or 0.0), reverse=True)
    return output


def write_summaries(args, rows_path, all_rows, candidate_heads=None, mentions=None):
    correlation_rows = summarize_correlations(all_rows)
    top_ks = [int(item) for item in str(args.top_k_summary).replace(" ", ",").split(",") if item.strip()]
    topk_rows = summarize_topk(all_rows, top_ks)
    mention_feature_rows = aggregate_mention_features(all_rows)
    mention_teacher_proxy_rows = summarize_mention_teacher_proxy(mention_feature_rows)
    correlations_path = os.path.join(args.output_dir, "head_logit_proxy_ablation_correlations.csv")
    topk_path = os.path.join(args.output_dir, "head_logit_proxy_ablation_topk_overlap.csv")
    mention_features_path = os.path.join(args.output_dir, "contribution_gap_mention_features.csv")
    mention_teacher_proxy_path = os.path.join(args.output_dir, "mention_proxy_teacher_correlations.csv")
    write_csv(correlations_path, correlation_rows)
    write_csv(topk_path, topk_rows)
    write_csv(mention_features_path, mention_feature_rows)
    write_csv(mention_teacher_proxy_path, mention_teacher_proxy_rows)

    if candidate_heads is None:
        head_keys = sorted({row.get("head_key") for row in all_rows if row.get("head_key")})
        candidate_heads = [parse_head_key(key) for key in head_keys]
    label_counts = Counter(row.get("label_family") for row in all_rows)
    if not label_counts:
        label_counts = Counter(row.get("label_name") for row in all_rows)
    summary = {
        "eval_results": args.eval_results,
        "n_mentions": len(mentions) if mentions is not None else len({row.get("step_id") for row in all_rows}),
        "n_rows": len(all_rows),
        "candidate_heads": [
            {"layer": layer, "head": head, "head_key": head_key(layer, head)}
            for layer, head in candidate_heads
        ],
        "label_counts": dict(label_counts),
        "outputs": {
            "rows": rows_path,
            "correlations": correlations_path,
            "topk_overlap": topk_path,
            "mention_features": mention_features_path,
            "mention_proxy_teacher_correlations": mention_teacher_proxy_path,
        },
        "top_correlations": correlation_rows[:10],
        "topk_overlap": topk_rows[:10],
        "top_mention_proxy_teacher_correlations": mention_teacher_proxy_rows[:10],
    }
    with open(os.path.join(args.output_dir, "head_logit_proxy_ablation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("[summary] head logit contribution proxy validation")
    print(json.dumps({
        "n_mentions": summary["n_mentions"],
        "n_rows": summary["n_rows"],
        "label_counts": summary["label_counts"],
        "top_correlations": summary["top_correlations"][:5],
        "topk_overlap": summary["topk_overlap"][:5],
        "top_mention_proxy_teacher_correlations": summary["top_mention_proxy_teacher_correlations"][:5],
    }, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-results", default="")
    parser.add_argument("--image-folder", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--conv-mode", default="vicuna_v1")
    parser.add_argument("--image-split", default="val2014")
    parser.add_argument("--match-eval-results", default="")
    parser.add_argument("--exclude-image-ids", action="append", default=[])
    parser.add_argument("--max-sentences", type=int, default=0)
    parser.add_argument("--max-mentions", type=int, default=20)
    parser.add_argument("--max-per-label", type=int, default=0)
    parser.add_argument("--label-filter", choices=["all", "hallucinated", "grounded"], default="all")
    parser.add_argument("--candidate-heads", default="")
    parser.add_argument("--query-calibration", default="")
    parser.add_argument("--query-top-k", type=int, default=20)
    parser.add_argument("--query-min-auroc", type=float, default=0.0)
    parser.add_argument("--query-select-by", choices=["high", "abs"], default="high")
    parser.add_argument("--include-adhh-top-k", type=int, default=20)
    parser.add_argument("--top-k-summary", default="1,3,5")
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--aggregate-only", action="store_true", default=False)
    parser.add_argument("--skip-full-head-ablation", action="store_true", default=False)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rows_path = os.path.join(args.output_dir, "head_logit_proxy_ablation_rows.csv")
    if args.aggregate_only:
        if not os.path.exists(rows_path):
            raise FileNotFoundError(rows_path)
        with open(rows_path, newline="") as f:
            all_rows = list(csv.DictReader(f))
        write_summaries(args, rows_path, all_rows)
        return
    if not args.eval_results:
        raise ValueError("--eval-results is required unless --aggregate-only is set")
    if not args.image_folder:
        raise ValueError("--image-folder is required unless --aggregate-only is set")
    done = set()
    if args.resume and os.path.exists(rows_path):
        with open(rows_path, newline="") as f:
            for row in csv.DictReader(f):
                done.add((row["step_id"], row["head_key"]))

    disable_torch_init()
    model_name = get_model_name_from_path(os.path.expanduser(args.model_path))
    tokenizer, model, image_processor, _ = load_pretrained_model(args.model_path, args.model_base, model_name)
    device = next(model.parameters()).device
    if model.config.model_type != "llava":
        # Keep the script useful for local variants, but use the same image span defaults as LLaVA-1.5.
        pass
    if args.model_path == "liuhaotian/llava-v1.6-34b":
        model.config.img_start_pos = 33
        model.config.img_length = 1948
    else:
        model.config.img_start_pos = 35
        model.config.img_length = 576

    candidate_heads = select_candidate_heads(args, args.model_path)
    if not candidate_heads:
        raise ValueError("No candidate heads selected")
    heads_by_layer = defaultdict(list)
    for layer, head in candidate_heads:
        heads_by_layer[int(layer)].append(int(head))
    heads_by_layer = {layer: sorted(set(heads)) for layer, heads in heads_by_layer.items()}

    mentions = build_mentions(args, tokenizer)
    mode = "a" if args.resume and os.path.exists(rows_path) else "w"
    field_written = mode == "a"
    all_rows = []

    with open(rows_path, mode, newline="") as f:
        writer = None
        for mention_idx, mention in enumerate(tqdm(mentions, desc="mentions")):
            prompt_ids, image_tensor, image_size = build_prompt_inputs(
                mention["image"],
                args.image_folder,
                tokenizer,
                image_processor,
                model.config,
                args.conv_mode,
                image_split=args.image_split,
            )
            prompt_ids = prompt_ids.to(device=device, non_blocking=True)
            image_tensor = image_tensor.to(device=device, dtype=torch.float16, non_blocking=True)
            prefix_ids = mention["caption_ids"][: int(mention["target_token_pos"])]
            target_token_id = int(mention["target_token_id"])
            step_id = (
                f'{mention["image_id"]}:{mention["mention_index"]}:'
                f'{mention["label_name"]}:{mention["target_token_pos"]}'
            )

            baseline = one_step_scores(
                model,
                prompt_ids,
                prefix_ids,
                image_tensor,
                image_size,
                record_head_outputs=True,
                record_components=True,
                heads_by_layer=heads_by_layer,
            )
            diagnostics = {
                (int(record["layer"]), int(record["head"])): record
                for record in baseline["diagnostics"]
            }
            target_rank_original = target_rank(baseline["sorted_ids"], target_token_id)
            top1_id = int(baseline["top1_id"])
            target_logit_original = float(baseline["score"][target_token_id].item())
            target_logprob_original = float(baseline["log_probs"][target_token_id].item())
            top1_logit_original = float(baseline["score"][top1_id].item())
            top1_logprob_original = float(baseline["log_probs"][top1_id].item())

            for layer, head in candidate_heads:
                key = head_key(layer, head)
                if (step_id, key) in done:
                    continue
                record = diagnostics.get((int(layer), int(head)))
                if record is None or "head_output" not in record:
                    continue
                text_value = record.get("text_value")
                img_value = record.get("img_value")
                proxy_target, residual_norm = head_output_proxy(
                    model,
                    layer,
                    head,
                    record["head_output"],
                    target_token_id,
                )
                proxy_top1, _ = head_output_proxy(
                    model,
                    layer,
                    head,
                    record["head_output"],
                    top1_id,
                )
                proxy_text_target = (
                    head_component_proxy(model, layer, head, text_value, target_token_id)
                    if text_value is not None else None
                )
                proxy_img_target = (
                    head_component_proxy(model, layer, head, img_value, target_token_id)
                    if img_value is not None else None
                )
                proxy_text_top1 = (
                    head_component_proxy(model, layer, head, text_value, top1_id)
                    if text_value is not None else None
                )
                proxy_img_top1 = (
                    head_component_proxy(model, layer, head, img_value, top1_id)
                    if img_value is not None else None
                )
                evidence_gap_target = (
                    proxy_text_target - proxy_img_target
                    if proxy_text_target is not None and proxy_img_target is not None else None
                )
                evidence_gap_top1 = (
                    proxy_text_top1 - proxy_img_top1
                    if proxy_text_top1 is not None and proxy_img_top1 is not None else None
                )
                ablated = None
                if not args.skip_full_head_ablation:
                    ablated = one_step_scores(
                        model,
                        prompt_ids,
                        prefix_ids,
                        image_tensor,
                        image_size,
                        zero_head=(layer, head),
                        record_head_outputs=False,
                    )
                text_ablated = None
                if text_value is not None:
                    text_ablated = one_step_scores(
                        model,
                        prompt_ids,
                        prefix_ids,
                        image_tensor,
                        image_size,
                        subtract_head_component=(layer, head, text_value),
                        record_head_outputs=False,
                    )
                if ablated is not None:
                    target_logit_zero = float(ablated["score"][target_token_id].item())
                    target_logprob_zero = float(ablated["log_probs"][target_token_id].item())
                    top1_logit_zero = float(ablated["score"][top1_id].item())
                    top1_logprob_zero = float(ablated["log_probs"][top1_id].item())
                    zero_top1_id = int(ablated["top1_id"])
                    zero_target_rank = target_rank(ablated["sorted_ids"], target_token_id)
                else:
                    target_logit_zero = None
                    target_logprob_zero = None
                    top1_logit_zero = None
                    top1_logprob_zero = None
                    zero_top1_id = None
                    zero_target_rank = None
                if text_ablated is not None:
                    target_logit_text_zero = float(text_ablated["score"][target_token_id].item())
                    target_logprob_text_zero = float(text_ablated["log_probs"][target_token_id].item())
                    top1_logit_text_zero = float(text_ablated["score"][top1_id].item())
                    top1_logprob_text_zero = float(text_ablated["log_probs"][top1_id].item())
                    text_zero_top1_id = int(text_ablated["top1_id"])
                    text_zero_target_rank = target_rank(text_ablated["sorted_ids"], target_token_id)
                else:
                    target_logit_text_zero = None
                    target_logprob_text_zero = None
                    top1_logit_text_zero = None
                    top1_logprob_text_zero = None
                    text_zero_top1_id = None
                    text_zero_target_rank = None
                out_row = {
                    "step_id": step_id,
                    "mention_index_global": mention_idx,
                    "image_id": mention["image_id"],
                    "image": mention["image"],
                    "word": mention["word"],
                    "node_word": mention["node_word"],
                    "occurrence_idx": int(mention.get("occurrence_idx", 0)),
                    "label": int(mention["label"]),
                    "label_name": mention["label_name"],
                    "label_family": row_label_family(mention),
                    "token_pos": int(mention["target_token_pos"]),
                    "target_token_id": target_token_id,
                    "target_token": tokenizer.decode([target_token_id]),
                    "top1_token_id": top1_id,
                    "top1_token": tokenizer.decode([top1_id]),
                    "target_rank_original": target_rank_original,
                    "target_logit_original": target_logit_original,
                    "target_logprob_original": target_logprob_original,
                    "top1_logit_original": top1_logit_original,
                    "top1_logprob_original": top1_logprob_original,
                    "layer": int(layer),
                    "head": int(head),
                    "head_key": key,
                    "proxy_target_logit": proxy_target,
                    "proxy_top1_logit": proxy_top1,
                    "proxy_target_logit_positive": max(0.0, proxy_target),
                    "proxy_top1_logit_positive": max(0.0, proxy_top1),
                    "proxy_text_target_logit": proxy_text_target,
                    "proxy_img_target_logit": proxy_img_target,
                    "proxy_evidence_gap_target": evidence_gap_target,
                    "proxy_text_target_logit_positive": max(0.0, proxy_text_target) if proxy_text_target is not None else None,
                    "proxy_img_target_logit_positive": max(0.0, proxy_img_target) if proxy_img_target is not None else None,
                    "proxy_evidence_gap_target_positive": max(0.0, evidence_gap_target) if evidence_gap_target is not None else None,
                    "proxy_text_top1_logit": proxy_text_top1,
                    "proxy_img_top1_logit": proxy_img_top1,
                    "proxy_evidence_gap_top1": evidence_gap_top1,
                    "proxy_text_top1_logit_positive": max(0.0, proxy_text_top1) if proxy_text_top1 is not None else None,
                    "proxy_img_top1_logit_positive": max(0.0, proxy_img_top1) if proxy_img_top1 is not None else None,
                    "proxy_evidence_gap_top1_positive": max(0.0, evidence_gap_top1) if evidence_gap_top1 is not None else None,
                    "text_mass": record.get("text_mass"),
                    "img_mass": record.get("img_mass"),
                    "full_entropy_norm": record.get("full_entropy_norm"),
                    "text_entropy_norm": record.get("text_entropy_norm"),
                    "head_residual_norm": residual_norm,
                    "target_logit_zero": target_logit_zero,
                    "target_logprob_zero": target_logprob_zero,
                    "top1_logit_zero": top1_logit_zero,
                    "top1_logprob_zero": top1_logprob_zero,
                    "target_logit_drop": (
                        target_logit_original - target_logit_zero
                        if target_logit_zero is not None else None
                    ),
                    "target_logprob_drop": (
                        target_logprob_original - target_logprob_zero
                        if target_logprob_zero is not None else None
                    ),
                    "top1_logit_drop": (
                        top1_logit_original - top1_logit_zero
                        if top1_logit_zero is not None else None
                    ),
                    "top1_logprob_drop": (
                        top1_logprob_original - top1_logprob_zero
                        if top1_logprob_zero is not None else None
                    ),
                    "zero_top1_token_id": zero_top1_id,
                    "zero_top1_token": tokenizer.decode([zero_top1_id]) if zero_top1_id is not None else None,
                    "target_rank_zero": zero_target_rank,
                    "target_logit_text_zero": target_logit_text_zero,
                    "target_logprob_text_zero": target_logprob_text_zero,
                    "top1_logit_text_zero": top1_logit_text_zero,
                    "top1_logprob_text_zero": top1_logprob_text_zero,
                    "target_text_logit_drop": (
                        target_logit_original - target_logit_text_zero
                        if target_logit_text_zero is not None else None
                    ),
                    "target_text_logprob_drop": (
                        target_logprob_original - target_logprob_text_zero
                        if target_logprob_text_zero is not None else None
                    ),
                    "top1_text_logit_drop": (
                        top1_logit_original - top1_logit_text_zero
                        if top1_logit_text_zero is not None else None
                    ),
                    "top1_text_logprob_drop": (
                        top1_logprob_original - top1_logprob_text_zero
                        if top1_logprob_text_zero is not None else None
                    ),
                    "text_zero_top1_token_id": text_zero_top1_id,
                    "text_zero_top1_token": tokenizer.decode([text_zero_top1_id]) if text_zero_top1_id is not None else None,
                    "target_rank_text_zero": text_zero_target_rank,
                }
                all_rows.append(out_row)
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(out_row.keys()))
                    if not field_written:
                        writer.writeheader()
                        field_written = True
                writer.writerow(out_row)
                f.flush()

    if args.resume and os.path.exists(rows_path):
        with open(rows_path, newline="") as f:
            all_rows = list(csv.DictReader(f))

    write_summaries(args, rows_path, all_rows, candidate_heads=candidate_heads, mentions=mentions)


if __name__ == "__main__":
    main()

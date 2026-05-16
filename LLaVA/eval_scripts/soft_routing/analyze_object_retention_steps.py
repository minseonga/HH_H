import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from eval_scripts.soft_routing.head_prior_utils import default_heads_for_model, head_key, load_head_priors


FEATURES = [
    "trigger_count",
    "weighted_trigger_count",
    "mean_excess",
    "max_excess",
    "weighted_excess",
    "max_weighted_excess",
    "mean_prior_text_mass",
    "max_prior_text_mass",
    "mean_question_attention",
    "mean_output_attention",
    "mean_recent_output_attention",
    "mean_image_attention",
    "weighted_question_attention",
    "weighted_recent_output_attention",
    "weighted_image_attention",
    "mean_removed_text_value_norm",
    "max_removed_text_value_norm",
    "weighted_removed_text_value_norm",
    "weighted_recent_output_value_norm",
    "target_logprob_drop_hard",
    "target_logprob_drop_soft",
    "target_drop_gap_hard_minus_soft",
    "kl_original_to_hard",
    "kl_original_to_soft",
    "entropy_hard_minus_original",
    "entropy_soft_minus_original",
    "hard_target_rank",
    "soft_target_rank",
]


def load_sentences(path):
    with open(path, "r") as f:
        data = json.load(f)
    return {str(item["image_id"]): item for item in data["sentences"]}


def node_pairs(sentence, key):
    pairs = []
    for item in sentence.get(key, []):
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            pairs.append((str(item[0]), str(item[1])))
        else:
            value = str(item)
            pairs.append((value, value))
    return pairs


def generated_nodes(sentence):
    generated = set()
    for item in sentence.get("mscoco_generated_words", []):
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            generated.add(str(item[1]))
        else:
            generated.add(str(item))
    return generated


def grounded_nodes(sentence):
    return {node for _, node in node_pairs(sentence, "mscoco_non_hallucinated_words")}


def hallucinated_nodes(sentence):
    return {node for _, node in node_pairs(sentence, "mscoco_hallucinated_words")}


def first_words_by_node(sentence, key="mscoco_non_hallucinated_words"):
    mapping = {}
    for word, node in node_pairs(sentence, key):
        mapping.setdefault(node, word)
    return mapping


def token_candidates(tokenizer, text):
    candidates = []
    seen = set()
    for value in (text, " " + text, text.lower(), " " + text.lower()):
        ids = tokenizer(value, add_special_tokens=False)["input_ids"]
        key = tuple(ids)
        if ids and key not in seen:
            candidates.append(ids)
            seen.add(key)
    return candidates


def find_subsequence(sequence, subsequence):
    if not subsequence:
        return None
    for idx in range(0, len(sequence) - len(subsequence) + 1):
        if sequence[idx:idx + len(subsequence)] == subsequence:
            return idx
    return None


def find_object_first_token(tokenizer, caption_ids, object_word):
    for candidate in token_candidates(tokenizer, object_word):
        pos = find_subsequence(caption_ids, candidate)
        if pos is not None:
            return pos, caption_ids[pos], candidate
        if len(candidate) > 1:
            pos = find_subsequence(caption_ids, candidate[-1:])
            if pos is not None:
                return pos, caption_ids[pos], candidate[-1:]
    return None, None, None


def add_row(rows, counts, tokenizer, sentence, hard, label, node, word, max_per_label, caption_source):
    if counts[label] >= max_per_label:
        return
    caption_ids = tokenizer(sentence["caption"], add_special_tokens=False)["input_ids"]
    token_pos, target_id, matched_ids = find_object_first_token(tokenizer, caption_ids, word)
    if token_pos is None:
        return
    rows.append({
        "image_id": str(sentence["image_id"]),
        "image": sentence["image"],
        "label": label,
        "caption_source": caption_source,
        "object_node": node,
        "object_word": word,
        "target_token_pos": token_pos,
        "target_token_id": target_id,
        "matched_token_ids": matched_ids,
        "probe_caption_ids": caption_ids,
        "probe_caption": sentence["caption"],
        "soft_caption": sentence["caption"] if caption_source == "soft" else "",
        "hard_caption": hard["caption"] if hard is not None else "",
        "hard_generated_nodes": sorted(generated_nodes(hard)) if hard is not None else [],
        "probe_generated_nodes": sorted(generated_nodes(sentence)),
    })
    counts[label] += 1


def select_rows(hard_by_id, soft_by_id, tokenizer, max_per_label, hallucinated_source="soft"):
    rows = []
    counts = Counter()
    for image_id, soft in soft_by_id.items():
        hard = hard_by_id.get(image_id)
        if hard is None:
            continue
        soft_grounded = grounded_nodes(soft)
        hard_grounded = grounded_nodes(hard)
        soft_words = first_words_by_node(soft, "mscoco_non_hallucinated_words")

        groups = [
            ("lost_grounded", sorted(soft_grounded - hard_grounded)),
            ("kept_grounded", sorted(soft_grounded & hard_grounded)),
        ]
        for label, nodes in groups:
            for node in nodes:
                word = soft_words.get(node, node)
                add_row(rows, counts, tokenizer, soft, hard, label, node, word, max_per_label, "soft")

        hall_sources = []
        if hallucinated_source in {"soft", "both"}:
            hall_sources.append(("soft", soft))
        if hallucinated_source in {"hard", "both"}:
            hall_sources.append(("hard", hard))
        for source_name, sentence in hall_sources:
            hall_words = first_words_by_node(sentence, "mscoco_hallucinated_words")
            for node in sorted(hallucinated_nodes(sentence)):
                word = hall_words.get(node, node)
                label = f"hallucinated_object_{source_name}" if hallucinated_source == "both" else "hallucinated_object"
                add_row(rows, counts, tokenizer, sentence, hard, label, node, word, max_per_label, source_name)
    return rows


def configure_model(model, model_path, prior_path, top_k, threshold, soft_gamma, soft_temperature):
    heads, priors, prior_source = load_head_priors(
        prior_path,
        top_k=top_k,
        prior_mode="score" if prior_path else "rank",
        default_heads=default_heads_for_model(model_path),
    )
    model.config.hal_attention_heads = heads
    model.config.head_attribution_priors = priors
    model.config.head_attribution_prior_source = prior_source
    model.config.adhh_threshold = threshold
    model.config.soft_gamma = soft_gamma
    model.config.soft_temperature = soft_temperature
    if model_path == "liuhaotian/llava-v1.6-34b":
        model.config.img_start_pos = 33
        model.config.img_length = 1948
    else:
        model.config.img_start_pos = 35
        model.config.img_length = 576
    return heads, priors, prior_source


def clear_modes(model):
    for name in [
        "adaptive_deactivate",
        "soft_deactivate",
        "dynamic_deactivate",
        "attribution_soft_deactivate",
        "retention_aware_deactivate",
        "record_intervention_diagnostics",
    ]:
        if hasattr(model.config, name):
            setattr(model.config, name, False)
    model.config.intervention_diagnostics = None


def set_mode(model, mode, record=False):
    clear_modes(model)
    if mode == "hard":
        model.config.adaptive_deactivate = True
    elif mode == "soft":
        model.config.soft_deactivate = True
    elif mode == "none":
        pass
    else:
        raise ValueError(f"Unknown mode={mode}")
    model.config.record_intervention_diagnostics = bool(record)
    model.config.intervention_diagnostics = [] if record else None


def build_prompt_inputs(row, image_folder, tokenizer, image_processor, model_config, conv_mode):
    qs = "Please describe this image in detail."
    if model_config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    image = Image.open(os.path.join(image_folder, row["image"])).convert("RGB")
    image_tensor = process_images([image], image_processor, model_config)[0]
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    return input_ids.unsqueeze(0), image_tensor.unsqueeze(0), image.size


def one_step(model, tokenizer, prompt_ids, prefix_ids, image_tensor, image_size, target_token_id, mode, record=False):
    prefix_tensor = torch.tensor(prefix_ids, device=prompt_ids.device, dtype=prompt_ids.dtype).unsqueeze(0)
    step_input = torch.cat([prompt_ids, prefix_tensor], dim=1) if prefix_ids else prompt_ids
    if record:
        model.config.diagnostic_output_start_pos = int(prompt_ids.shape[1] - 1 + getattr(model.config, "img_length", 576))
        model.config.diagnostic_recent_window = 16
    set_mode(model, mode, record=record)
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
    score = output["scores"][0][0].detach().float()
    log_probs = F.log_softmax(score, dim=-1)
    probs = log_probs.exp()
    entropy = float(-(probs * log_probs).sum().item())
    target_logprob = float(log_probs[int(target_token_id)].item())
    sorted_ids = torch.argsort(score, descending=True)
    target_rank = int((sorted_ids == int(target_token_id)).nonzero(as_tuple=False)[0].item() + 1)
    next_token_id = int(torch.argmax(score).item())
    return {
        "score": score,
        "entropy": entropy,
        "target_logprob": target_logprob,
        "target_rank": target_rank,
        "next_token_id": next_token_id,
        "next_token": tokenizer.decode([next_token_id]),
        "diagnostics": list(getattr(model.config, "intervention_diagnostics", []) or []),
    }


def kl_divergence(p_score, q_score):
    p_log = F.log_softmax(p_score.float(), dim=-1)
    q_log = F.log_softmax(q_score.float(), dim=-1)
    p = p_log.exp()
    return float((p * (p_log - q_log)).sum().item())


def aggregate_diagnostics(records, priors, threshold, soft_gamma, soft_temperature):
    trigger_count = 0
    weighted_trigger_count = 0.0
    excesses = []
    weighted_excesses = []
    prior_texts = []
    text_masses = []
    question_attentions = []
    output_attentions = []
    recent_output_attentions = []
    image_attentions = []
    weighted_question_attentions = []
    weighted_recent_output_attentions = []
    weighted_image_attentions = []
    removed_text_value_norms = []
    weighted_removed_text_value_norms = []
    weighted_recent_output_value_norms = []
    for record in records:
        key = record.get("head_key") or head_key(record.get("layer"), record.get("head"))
        prior = float(priors.get(key, 1.0))
        text_mass = float(record.get("text_mass", 0.0))
        question_attention = float(record.get("question_attention", 0.0))
        output_attention = float(record.get("output_attention", 0.0))
        recent_output_attention = float(record.get("recent_output_attention", 0.0))
        image_attention = float(record.get("img_mass", 0.0))
        removed_text_value_norm = float(record.get("removed_text_value_norm", 0.0))
        recent_output_value_norm = float(record.get("recent_output_value_norm", 0.0))
        excess = max(0.0, text_mass - threshold)
        trigger = 1.0 if text_mass >= threshold else 0.0
        trigger_count += int(trigger)
        weighted_trigger_count += prior * trigger
        excesses.append(excess)
        weighted_excesses.append(prior * excess)
        prior_texts.append(prior * text_mass)
        text_masses.append(text_mass)
        question_attentions.append(question_attention)
        output_attentions.append(output_attention)
        recent_output_attentions.append(recent_output_attention)
        image_attentions.append(image_attention)
        weighted_question_attentions.append(prior * question_attention)
        weighted_recent_output_attentions.append(prior * recent_output_attention)
        weighted_image_attentions.append(prior * image_attention)
        removed_text_value_norms.append(removed_text_value_norm)
        weighted_removed_text_value_norms.append(prior * removed_text_value_norm)
        weighted_recent_output_value_norms.append(prior * recent_output_value_norm)
    return {
        "trigger_count": trigger_count,
        "weighted_trigger_count": weighted_trigger_count,
        "mean_excess": float(np.mean(excesses)) if excesses else 0.0,
        "max_excess": float(np.max(excesses)) if excesses else 0.0,
        "weighted_excess": float(np.sum(weighted_excesses)) if weighted_excesses else 0.0,
        "max_weighted_excess": float(np.max(weighted_excesses)) if weighted_excesses else 0.0,
        "mean_prior_text_mass": float(np.mean(prior_texts)) if prior_texts else 0.0,
        "max_prior_text_mass": float(np.max(prior_texts)) if prior_texts else 0.0,
        "mean_question_attention": float(np.mean(question_attentions)) if question_attentions else 0.0,
        "mean_output_attention": float(np.mean(output_attentions)) if output_attentions else 0.0,
        "mean_recent_output_attention": float(np.mean(recent_output_attentions)) if recent_output_attentions else 0.0,
        "mean_image_attention": float(np.mean(image_attentions)) if image_attentions else 0.0,
        "weighted_question_attention": float(np.sum(weighted_question_attentions)) if weighted_question_attentions else 0.0,
        "weighted_recent_output_attention": float(np.sum(weighted_recent_output_attentions)) if weighted_recent_output_attentions else 0.0,
        "weighted_image_attention": float(np.sum(weighted_image_attentions)) if weighted_image_attentions else 0.0,
        "mean_removed_text_value_norm": float(np.mean(removed_text_value_norms)) if removed_text_value_norms else 0.0,
        "max_removed_text_value_norm": float(np.max(removed_text_value_norms)) if removed_text_value_norms else 0.0,
        "weighted_removed_text_value_norm": float(np.sum(weighted_removed_text_value_norms)) if weighted_removed_text_value_norms else 0.0,
        "weighted_recent_output_value_norm": float(np.sum(weighted_recent_output_value_norms)) if weighted_recent_output_value_norms else 0.0,
    }


def mean(values):
    return float(np.mean(values)) if values else None


def group_means(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["label"]].append(row)
    output = []
    for label, items in sorted(grouped.items()):
        record = {"label": label, "n": len(items)}
        for feature in FEATURES:
            record[feature] = mean([float(item.get(feature, 0.0)) for item in items])
        output.append(record)
    return output


def auc_rows(rows):
    labels = [1 if row["label"] == "lost_grounded" else 0 for row in rows]
    output = []
    for feature in FEATURES:
        values = [float(row.get(feature, 0.0)) for row in rows]
        if len(set(labels)) < 2 or len(set(values)) < 2:
            continue
        auc = float(roc_auc_score(labels, values))
        output.append({
            "feature": feature,
            "positive_label": "lost_grounded",
            "n": len(values),
            "auroc_high_predicts_lost": auc,
            "auroc_abs": max(auc, 1.0 - auc),
            "direction": "high_predicts_lost" if auc >= 0.5 else "low_predicts_lost",
            "auprc_high_predicts_lost": float(average_precision_score(labels, values)),
            "mean_lost": mean([v for v, y in zip(values, labels) if y == 1]),
            "mean_kept": mean([v for v, y in zip(values, labels) if y == 0]),
        })
    output.sort(key=lambda item: item["auroc_abs"], reverse=True)
    return output


def pairwise_auc_rows(rows, positive_label, negative_label):
    filtered = [row for row in rows if row["label"] in {positive_label, negative_label}]
    return pairwise_auc_for_rows(filtered, positive_label, negative_label, lambda row: row["label"] == positive_label)


def pairwise_prefix_auc_rows(rows, positive_label, negative_prefix):
    filtered = [
        row
        for row in rows
        if row["label"] == positive_label or row["label"].startswith(negative_prefix)
    ]
    return pairwise_auc_for_rows(
        filtered,
        positive_label,
        f"{negative_prefix}*",
        lambda row: row["label"] == positive_label,
    )


def pairwise_auc_for_rows(filtered, positive_label, negative_label, is_positive):
    labels = [1 if is_positive(row) else 0 for row in filtered]
    output = []
    for feature in FEATURES:
        values = [float(row.get(feature, 0.0)) for row in filtered]
        if len(set(labels)) < 2 or len(set(values)) < 2:
            continue
        auc = float(roc_auc_score(labels, values))
        output.append({
            "feature": feature,
            "positive_label": positive_label,
            "negative_label": negative_label,
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


def pairwise_positive_prefix_auc_rows(rows, positive_prefix, negative_label):
    filtered = [
        row
        for row in rows
        if row["label"].startswith(positive_prefix) or row["label"] == negative_label
    ]
    return pairwise_auc_for_rows(
        filtered,
        f"{positive_prefix}*",
        negative_label,
        lambda row: row["label"].startswith(positive_prefix),
    )


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
    parser.add_argument("--hard-results", required=True)
    parser.add_argument("--soft-results", required=True)
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prior-path", default="")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--model-path", default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--conv-mode", default="vicuna_v1")
    parser.add_argument("--max-per-label", type=int, default=100)
    parser.add_argument("--hallucinated-source", type=str, default="soft", choices=["soft", "hard", "both"])
    parser.add_argument("--adhh-threshold", type=float, default=0.4)
    parser.add_argument("--soft-gamma", type=float, default=0.75)
    parser.add_argument("--soft-temperature", type=float, default=0.05)
    args = parser.parse_args()

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
    selected = select_rows(hard_by_id, soft_by_id, tokenizer, args.max_per_label, args.hallucinated_source)

    os.makedirs(args.output_dir, exist_ok=True)
    output_rows = []
    with open(os.path.join(args.output_dir, "object_retention_steps.jsonl"), "w") as out:
        for row in tqdm(selected):
            prompt_ids, image_tensor, image_size = build_prompt_inputs(
                row, args.image_folder, tokenizer, image_processor, model.config, args.conv_mode
            )
            prompt_ids = prompt_ids.to(device="cuda", non_blocking=True)
            image_tensor = image_tensor.to(dtype=torch.float16, device="cuda", non_blocking=True)
            prefix_ids = row["probe_caption_ids"][:row["target_token_pos"]]
            target_token_id = int(row["target_token_id"])

            original = one_step(model, tokenizer, prompt_ids, prefix_ids, image_tensor, image_size, target_token_id, "none", record=True)
            hard = one_step(model, tokenizer, prompt_ids, prefix_ids, image_tensor, image_size, target_token_id, "hard")
            soft = one_step(model, tokenizer, prompt_ids, prefix_ids, image_tensor, image_size, target_token_id, "soft")
            features = aggregate_diagnostics(
                original["diagnostics"],
                priors,
                args.adhh_threshold,
                args.soft_gamma,
                args.soft_temperature,
            )

            output = {
                "image_id": row["image_id"],
                "image": row["image"],
                "label": row["label"],
                "caption_source": row["caption_source"],
                "object_node": row["object_node"],
                "object_word": row["object_word"],
                "target_token": tokenizer.decode([target_token_id]),
                "target_token_id": target_token_id,
                "target_token_pos": int(row["target_token_pos"]),
                "prefix_text": tokenizer.decode(prefix_ids, skip_special_tokens=True).strip(),
                "target_logprob_original": original["target_logprob"],
                "target_logprob_hard": hard["target_logprob"],
                "target_logprob_soft": soft["target_logprob"],
                "target_logprob_drop_hard": original["target_logprob"] - hard["target_logprob"],
                "target_logprob_drop_soft": original["target_logprob"] - soft["target_logprob"],
                "target_drop_gap_hard_minus_soft": (original["target_logprob"] - hard["target_logprob"]) - (original["target_logprob"] - soft["target_logprob"]),
                "hard_target_rank": hard["target_rank"],
                "soft_target_rank": soft["target_rank"],
                "original_target_rank": original["target_rank"],
                "original_next_token": original["next_token"],
                "hard_next_token": hard["next_token"],
                "soft_next_token": soft["next_token"],
                "entropy_original": original["entropy"],
                "entropy_hard": hard["entropy"],
                "entropy_soft": soft["entropy"],
                "entropy_hard_minus_original": hard["entropy"] - original["entropy"],
                "entropy_soft_minus_original": soft["entropy"] - original["entropy"],
                "kl_original_to_hard": kl_divergence(original["score"], hard["score"]),
                "kl_original_to_soft": kl_divergence(original["score"], soft["score"]),
                "probe_caption": row["probe_caption"],
                "soft_caption": row["soft_caption"],
                "hard_caption": row["hard_caption"],
                "prior_source": prior_source,
                **features,
            }
            output_rows.append(output)
            out.write(json.dumps({**output, "diagnostics": original["diagnostics"]}) + "\n")
            out.flush()

    summary = {
        "num_records": len(output_rows),
        "label_counts": dict(Counter(row["label"] for row in output_rows)),
        "prior_source": prior_source,
        "heads": heads,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    write_csv(os.path.join(args.output_dir, "object_retention_features.csv"), output_rows)
    write_csv(os.path.join(args.output_dir, "group_feature_means.csv"), group_means(output_rows))
    write_csv(os.path.join(args.output_dir, "lost_grounded_auc.csv"), auc_rows(output_rows))
    write_csv(
        os.path.join(args.output_dir, "lost_vs_hallucinated_auc.csv"),
        pairwise_prefix_auc_rows(output_rows, "lost_grounded", "hallucinated_object"),
    )
    write_csv(
        os.path.join(args.output_dir, "lost_vs_hallucinated_hard_auc.csv"),
        pairwise_auc_rows(output_rows, "lost_grounded", "hallucinated_object_hard"),
    )
    write_csv(
        os.path.join(args.output_dir, "lost_vs_hallucinated_soft_auc.csv"),
        pairwise_auc_rows(output_rows, "lost_grounded", "hallucinated_object_soft"),
    )
    write_csv(
        os.path.join(args.output_dir, "hallucinated_vs_kept_auc.csv"),
        pairwise_positive_prefix_auc_rows(output_rows, "hallucinated_object", "kept_grounded"),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

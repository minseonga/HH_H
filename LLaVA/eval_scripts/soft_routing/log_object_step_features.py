import argparse
import json
import os
from collections import defaultdict

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
from eval_scripts.soft_routing.head_prior_utils import (
    default_heads_for_model,
    headwise_percentile_thresholds,
    load_head_priors,
)


def load_eval_sentences(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data["sentences"]


def build_prompt_inputs(image_file, image_folder, tokenizer, image_processor, model_config, conv_mode):
    qs = "Please describe this image in detail."
    if model_config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    image = Image.open(os.path.join(image_folder, image_file)).convert("RGB")
    image_tensor = process_images([image], image_processor, model_config)[0]
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    return input_ids.unsqueeze(0), image_tensor.unsqueeze(0), image.size


def object_mentions(sentence):
    mentions = []
    for word, node_word in sentence.get("mscoco_non_hallucinated_words", []):
        mentions.append({"word": word, "node_word": node_word, "label": 0})
    for word, node_word in sentence.get("mscoco_hallucinated_words", []):
        mentions.append({"word": word, "node_word": node_word, "label": 1})
    return mentions


def token_id_candidates_for_word(tokenizer, word):
    candidates = []
    seen = set()
    for text in (word, " " + word, word.lower(), " " + word.lower()):
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        key = tuple(ids)
        if ids and key not in seen:
            candidates.append(ids)
            seen.add(key)
    return candidates


def find_next_subsequence(sequence, subsequence, start):
    if not subsequence:
        return None
    for idx in range(start, len(sequence) - len(subsequence) + 1):
        if sequence[idx:idx + len(subsequence)] == subsequence:
            return idx
    return None


def align_mentions(tokenizer, caption, mentions):
    caption_ids = tokenizer(caption, add_special_tokens=False)["input_ids"]
    cursor = 0
    aligned = []
    for mention in mentions:
        word_ids = []
        pos = None
        for candidate_ids in token_id_candidates_for_word(tokenizer, mention["word"]):
            pos = find_next_subsequence(caption_ids, candidate_ids, cursor)
            if pos is None and candidate_ids:
                pos = find_next_subsequence(caption_ids, candidate_ids[-1:], cursor)
            if pos is not None:
                word_ids = candidate_ids
                break
        if pos is None:
            continue
        aligned.append({**mention, "token_pos": pos, "token_ids": word_ids})
        cursor = max(pos + 1, cursor)
    return caption_ids, aligned


def configure_model(model, model_path, attribution_path, top_k, head_prior_mode, adhh_threshold, soft_gamma, soft_temperature):
    heads, priors, prior_source = load_head_priors(
        attribution_path,
        top_k=top_k,
        prior_mode=head_prior_mode,
        default_heads=default_heads_for_model(model_path),
    )
    model.config.hal_attention_heads = heads
    model.config.head_attribution_priors = priors
    model.config.head_attribution_prior_source = prior_source
    model.config.adhh_threshold = adhh_threshold
    model.config.soft_gamma = soft_gamma
    model.config.soft_temperature = soft_temperature
    if model_path == "liuhaotian/llava-v1.6-34b":
        model.config.img_start_pos = 33
        model.config.img_length = 1948
    else:
        model.config.img_start_pos = 35
        model.config.img_length = 576
    return heads, priors, prior_source


def one_step_diagnostics(model, prompt_ids, prefix_ids, image_tensor, image_size):
    if prefix_ids:
        prefix_tensor = torch.tensor(prefix_ids, device=prompt_ids.device, dtype=prompt_ids.dtype).unsqueeze(0)
        step_input = torch.cat([prompt_ids, prefix_tensor], dim=1)
    else:
        step_input = prompt_ids
    model.config.record_intervention_diagnostics = True
    model.config.intervention_diagnostics = []
    model.config.adaptive_deactivate = False
    model.config.soft_deactivate = False
    model.config.dynamic_deactivate = False
    model.config.attribution_soft_deactivate = False
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
    scores = output["scores"][0][0].detach().float()
    probs = F.softmax(scores, dim=-1)
    log_probs = F.log_softmax(scores, dim=-1)
    entropy = float(-(probs * log_probs).sum().item())
    top_vals, _ = torch.topk(scores, 2)
    return list(model.config.intervention_diagnostics), {
        "entropy": entropy,
        "top1_top2_margin": float((top_vals[0] - top_vals[1]).item()),
    }


def aggregate_features(records, hard_threshold=0.4, head_thresholds=None):
    if not records:
        return {}
    text = np.array([r["text_mass"] for r in records], dtype=float)
    img = np.array([r["img_mass"] for r in records], dtype=float)
    prior = np.array([r["attribution_prior"] for r in records], dtype=float)
    trigger = np.array([1.0 if r["hard_trigger"] else 0.0 for r in records], dtype=float)
    weighted_text = prior * text
    excess = np.maximum(text - hard_threshold, 0.0)
    weighted_excess = prior * excess
    features = {
        "max_i_text": float(text.max()),
        "mean_i_text": float(text.mean()),
        "max_text_ratio": float((text / (text + img + 1e-6)).max()),
        "mean_text_ratio": float((text / (text + img + 1e-6)).mean()),
        "max_prior_i_text": float(weighted_text.max()),
        "mean_prior_i_text": float(weighted_text.mean()),
        "trigger_count": int(trigger.sum()),
        "weighted_trigger_count": float((prior * trigger).sum()),
        "sum_prior_excess": float(weighted_excess.sum()),
        "max_prior_excess": float(weighted_excess.max()),
    }
    if head_thresholds:
        percentile_excess = []
        percentile_active = []
        for record in records:
            threshold = head_thresholds.get(record["head_key"], {})
            low = float(threshold.get("low", hard_threshold))
            high = float(threshold.get("high", max(hard_threshold + 1e-4, 0.9)))
            denom = max(high - low, 1e-6)
            value = min(1.0, max(0.0, (float(record["text_mass"]) - low) / denom))
            percentile_excess.append(value)
            percentile_active.append(1.0 if value > 0 else 0.0)
        percentile_excess = np.array(percentile_excess, dtype=float)
        percentile_active = np.array(percentile_active, dtype=float)
        weighted_percentile_excess = prior * percentile_excess
        features.update({
            "sum_prior_percentile_excess": float(weighted_percentile_excess.sum()),
            "max_prior_percentile_excess": float(weighted_percentile_excess.max()),
            "weighted_percentile_active_count": float((prior * percentile_active).sum()),
        })
    return features


def feature_summary(rows, feature_names):
    labels = np.array([row["label"] for row in rows], dtype=int)
    summary = []
    for feature in feature_names:
        values = np.array([row.get(feature, 0.0) for row in rows], dtype=float)
        if len(set(labels.tolist())) < 2:
            auc = None
            ap = None
        else:
            auc = float(roc_auc_score(labels, values))
            ap = float(average_precision_score(labels, values))
        hall = values[labels == 1]
        non = values[labels == 0]
        summary.append({
            "feature": feature,
            "auroc": auc,
            "auprc": ap,
            "hall_mean": float(hall.mean()) if len(hall) else None,
            "nonhall_mean": float(non.mean()) if len(non) else None,
            "hall_minus_nonhall": float(hall.mean() - non.mean()) if len(hall) and len(non) else None,
        })
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-results", required=True)
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--model-path", default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--conv-mode", default="vicuna_v1")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--attention-head-path", default="results/coco/llava_3000/identify_attention_head/attribution_result.json")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--head-prior-mode", default="auto", choices=["auto", "score", "rank", "uniform"])
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--adhh-threshold", type=float, default=0.4)
    parser.add_argument("--soft-gamma", type=float, default=0.75)
    parser.add_argument("--soft-temperature", type=float, default=0.05)
    parser.add_argument("--q-low", type=float, default=60)
    parser.add_argument("--q-high", type=float, default=90)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    sentences = load_eval_sentences(args.eval_results)[:args.max_samples]

    disable_torch_init()
    model_name = get_model_name_from_path(os.path.expanduser(args.model_path))
    tokenizer, model, image_processor, _ = load_pretrained_model(args.model_path, args.model_base, model_name)
    _, priors, prior_source = configure_model(
        model, args.model_path, args.attention_head_path, args.top_k, args.head_prior_mode,
        args.adhh_threshold, args.soft_gamma, args.soft_temperature,
    )

    rows = []
    head_records = defaultdict(list)
    for sentence in tqdm(sentences):
        mentions = object_mentions(sentence)
        if not mentions:
            continue
        caption_ids, aligned = align_mentions(tokenizer, sentence["caption"], mentions)
        if not aligned:
            continue
        prompt_ids, image_tensor, image_size = build_prompt_inputs(
            sentence["image"], args.image_folder, tokenizer, image_processor, model.config, args.conv_mode
        )
        prompt_ids = prompt_ids.to(device="cuda", non_blocking=True)
        image_tensor = image_tensor.to(dtype=torch.float16, device="cuda", non_blocking=True)
        for mention in aligned:
            prefix_ids = caption_ids[:mention["token_pos"]]
            diagnostics, logit_stats = one_step_diagnostics(model, prompt_ids, prefix_ids, image_tensor, image_size)
            for record in diagnostics:
                head_records[record["head_key"]].append(record["text_mass"])
            row = {
                "image_id": sentence["image_id"],
                "image": sentence["image"],
                "object_word": mention["word"],
                "node_word": mention["node_word"],
                "label": int(mention["label"]),
                "label_name": "hallucinated" if mention["label"] else "grounded",
                "token_pos": int(mention["token_pos"]),
                "prior_source": prior_source,
                "_head_diagnostics": diagnostics,
                **logit_stats,
            }
            rows.append(row)

    head_thresholds = headwise_percentile_thresholds(head_records, args.q_low, args.q_high)
    output_jsonl = os.path.join(args.output_dir, "object_step_features.jsonl")
    with open(output_jsonl, "w") as out:
        for row in rows:
            diagnostics = row.pop("_head_diagnostics")
            row.update(aggregate_features(diagnostics, hard_threshold=args.adhh_threshold, head_thresholds=head_thresholds))
            out.write(json.dumps({**row, "head_diagnostics": diagnostics}) + "\n")

    feature_names = [
        "max_i_text",
        "mean_i_text",
        "max_text_ratio",
        "mean_text_ratio",
        "max_prior_i_text",
        "mean_prior_i_text",
        "trigger_count",
        "weighted_trigger_count",
        "sum_prior_excess",
        "max_prior_excess",
        "sum_prior_percentile_excess",
        "max_prior_percentile_excess",
        "weighted_percentile_active_count",
        "entropy",
        "top1_top2_margin",
    ]
    summary = {
        "num_object_steps": len(rows),
        "num_hallucinated": sum(row["label"] for row in rows),
        "num_grounded": sum(1 - row["label"] for row in rows),
        "prior_source": prior_source,
        "feature_summary": feature_summary(rows, feature_names),
        "head_text_thresholds": head_thresholds,
        "head_attribution_priors": priors,
    }
    with open(os.path.join(args.output_dir, "feature_validation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.output_dir, "head_thresholds.json"), "w") as f:
        json.dump({"head_text_thresholds": summary["head_text_thresholds"]}, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

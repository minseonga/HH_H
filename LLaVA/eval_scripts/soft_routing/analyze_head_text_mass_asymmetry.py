import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict

import torch
from tqdm import tqdm

from eval_scripts.soft_routing.analyze_query_direction_chair_alignment import (
    build_prompt_inputs,
    collect_exclude_ids,
    filter_sentences,
    image_name_from_sentence,
    load_eval_sentences,
    prepare_mentions,
)
from eval_scripts.soft_routing.analyze_text_heavy_object_alignment import auc_score
from eval_scripts.soft_routing.head_prior_utils import (
    default_heads_for_model,
    head_key,
    load_head_priors,
)


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
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


def mean(values):
    clean = []
    for value in values:
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            clean.append(value)
    return sum(clean) / len(clean) if clean else None


def percentile(values, q):
    clean = []
    for value in values:
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            clean.append(value)
    clean.sort()
    if not clean:
        return None
    idx = int(round((len(clean) - 1) * float(q)))
    idx = max(0, min(idx, len(clean) - 1))
    return clean[idx]


def generation_attentions_to_layers(attentions):
    if attentions is None:
        return None
    if isinstance(attentions, (tuple, list)) and attentions:
        first = attentions[0]
        if isinstance(first, (tuple, list)):
            return first
        return attentions
    return None


def one_step_attention(model, prompt_ids, prefix_ids, image_tensor, image_size):
    if prefix_ids:
        prefix_tensor = torch.tensor(prefix_ids, device=prompt_ids.device, dtype=prompt_ids.dtype).unsqueeze(0)
        step_input = torch.cat([prompt_ids, prefix_tensor], dim=1)
    else:
        step_input = prompt_ids
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
            output_attentions=True,
            output_scores=False,
            return_dict_in_generate=True,
        )
    return output


def score_head_features(attn_layers, prompt_len, model_config, caption_prefix_len):
    img_start = int(getattr(model_config, "img_start_pos", 35))
    img_length = int(getattr(model_config, "img_length", 576))
    img_end = img_start + img_length
    expanded_prompt_len = int(prompt_len) - 1 + img_length
    rows = []
    for layer, layer_attention in enumerate(attn_layers):
        attention = layer_attention[0, :, -1, :].detach().float().cpu()
        kv_len = int(attention.shape[-1])
        text_start = min(img_end, kv_len)
        prompt_text_end = min(expanded_prompt_len, kv_len)
        generated_start = min(expanded_prompt_len, kv_len)
        recent_start = max(generated_start, kv_len - 16)

        text_mass = attention[:, text_start:].sum(dim=-1)
        image_mass = attention[:, img_start:min(img_end, kv_len)].sum(dim=-1) if img_start < kv_len else torch.zeros(attention.shape[0])
        prompt_text_mass = attention[:, text_start:prompt_text_end].sum(dim=-1) if text_start < prompt_text_end else torch.zeros(attention.shape[0])
        generated_mass = attention[:, generated_start:].sum(dim=-1) if generated_start < kv_len else torch.zeros(attention.shape[0])
        recent_generated_mass = attention[:, recent_start:].sum(dim=-1) if recent_start < kv_len else torch.zeros(attention.shape[0])
        prefix_before_image_mass = attention[:, :min(img_start, kv_len)].sum(dim=-1)
        max_attention = attention.max(dim=-1).values
        top1_index = attention.argmax(dim=-1)
        top1_is_image = ((top1_index >= img_start) & (top1_index < img_end)).float()
        top1_is_generated = (top1_index >= expanded_prompt_len).float()
        for head in range(attention.shape[0]):
            rows.append({
                "layer": layer,
                "head": head,
                "head_key": f"{layer}:{head}",
                "text_mass": float(text_mass[head].item()),
                "image_mass": float(image_mass[head].item()),
                "prompt_text_mass": float(prompt_text_mass[head].item()),
                "generated_mass": float(generated_mass[head].item()),
                "recent_generated_mass": float(recent_generated_mass[head].item()),
                "prefix_before_image_mass": float(prefix_before_image_mass[head].item()),
                "max_attention": float(max_attention[head].item()),
                "top1_index": int(top1_index[head].item()),
                "top1_is_image": float(top1_is_image[head].item()),
                "top1_is_generated": float(top1_is_generated[head].item()),
                "caption_prefix_len": int(caption_prefix_len),
                "kv_len": kv_len,
            })
    return rows


def summarize_heads(rows, adhh_ranks, eps=1e-8):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["head_key"]].append(row)
    summary = []
    metrics = [
        "text_mass",
        "prompt_text_mass",
        "generated_mass",
        "recent_generated_mass",
        "image_mass",
        "prefix_before_image_mass",
        "max_attention",
        "top1_is_image",
        "top1_is_generated",
    ]
    for key, items in grouped.items():
        labels = [int(item["label"]) for item in items]
        n_pos = sum(labels)
        n_neg = len(labels) - n_pos
        layer, head = key.split(":")
        row = {
            "head_key": key,
            "layer": int(layer),
            "head": int(head),
            "n": len(items),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "in_adhh_top20": int(key in adhh_ranks),
            "adhh_rank": adhh_ranks.get(key, ""),
        }
        for metric in metrics:
            values = [float(item[metric]) for item in items]
            pos_values = [value for value, label in zip(values, labels) if label == 1]
            neg_values = [value for value, label in zip(values, labels) if label == 0]
            pos_mean = mean(pos_values)
            neg_mean = mean(neg_values)
            auc = auc_score(labels, values)
            row[f"hall_mean_{metric}"] = pos_mean
            row[f"grounded_mean_{metric}"] = neg_mean
            row[f"hall_minus_grounded_{metric}"] = (
                pos_mean - neg_mean if pos_mean is not None and neg_mean is not None else None
            )
            row[f"hall_over_grounded_{metric}"] = (
                (pos_mean + eps) / (neg_mean + eps)
                if pos_mean is not None and neg_mean is not None else None
            )
            row[f"hall_p90_{metric}"] = percentile(pos_values, 0.9)
            row[f"grounded_p90_{metric}"] = percentile(neg_values, 0.9)
            row[f"auroc_high_{metric}"] = auc
            row[f"auroc_low_{metric}"] = (1.0 - auc) if auc is not None else None
        summary.append(row)

    summary.sort(
        key=lambda row: (
            -(row.get("auroc_high_text_mass") or 0.0),
            -(row.get("hall_minus_grounded_text_mass") or 0.0),
            row["layer"],
            row["head"],
        )
    )
    for rank, row in enumerate(summary, start=1):
        row["rank_by_text_mass_auroc"] = rank
    summary.sort(
        key=lambda row: (
            -(row.get("hall_over_grounded_text_mass") or 0.0),
            row["layer"],
            row["head"],
        )
    )
    for rank, row in enumerate(summary, start=1):
        row["rank_by_text_mass_ratio"] = rank
    summary.sort(
        key=lambda row: (
            -(row.get("auroc_high_text_mass") or 0.0),
            -(row.get("hall_over_grounded_text_mass") or 0.0),
        )
    )
    return summary


def summarize_layers(head_rows):
    grouped = defaultdict(list)
    for row in head_rows:
        grouped[int(row["layer"])].append(row)
    rows = []
    for layer, items in sorted(grouped.items()):
        rows.append({
            "layer": layer,
            "n_heads": len(items),
            "mean_auroc_high_text_mass": mean([row.get("auroc_high_text_mass") for row in items]),
            "max_auroc_high_text_mass": max(row.get("auroc_high_text_mass") or 0.0 for row in items),
            "best_head_by_auroc": max(items, key=lambda row: row.get("auroc_high_text_mass") or 0.0)["head_key"],
            "mean_hall_over_grounded_text_mass": mean([row.get("hall_over_grounded_text_mass") for row in items]),
            "max_hall_over_grounded_text_mass": max(row.get("hall_over_grounded_text_mass") or 0.0 for row in items),
            "best_head_by_ratio": max(items, key=lambda row: row.get("hall_over_grounded_text_mass") or 0.0)["head_key"],
            "n_adhh_top20_heads": sum(int(row.get("in_adhh_top20", 0)) for row in items),
        })
    return rows


def compact_mention_row(row):
    return {
        "question_id": row["question_id"],
        "image_id": row["image_id"],
        "image": row["image"],
        "word": row["word"],
        "node_word": row["node_word"],
        "label": row["label"],
        "label_name": row["label_name"],
        "token_pos": row["token_pos"],
        "target_token_ids": row["target_token_ids"],
        "target_tokens": row["target_tokens"],
        "head_key": row["head_key"],
        "layer": row["layer"],
        "head": row["head"],
        "text_mass": row["text_mass"],
        "image_mass": row["image_mass"],
        "prompt_text_mass": row["prompt_text_mass"],
        "generated_mass": row["generated_mass"],
        "recent_generated_mass": row["recent_generated_mass"],
        "prefix_before_image_mass": row["prefix_before_image_mass"],
        "max_attention": row["max_attention"],
        "top1_index": row["top1_index"],
        "top1_is_image": row["top1_is_image"],
        "top1_is_generated": row["top1_is_generated"],
    }


def count_unique_mentions(rows):
    labels_by_mention = {}
    for row in rows:
        key = (
            row.get("question_id", ""),
            row.get("image_id", ""),
            row.get("word", ""),
            row.get("node_word", ""),
            row.get("token_pos", ""),
            row.get("score_token_pos", row.get("token_pos", "")),
            row.get("label_name", ""),
        )
        labels_by_mention.setdefault(key, row.get("label_name", ""))
    return dict(Counter(label for label in labels_by_mention.values() if label))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-results", required=True)
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--image-split", default="val2014")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--conv-mode", default="vicuna_v1")
    parser.add_argument("--prior-path", default="")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--score-span", choices=["first", "all"], default="first")
    parser.add_argument("--match-eval-results", default="")
    parser.add_argument("--exclude-eval-results", nargs="*", default=[])
    parser.add_argument("--exclude-image-ids", nargs="*", default=[])
    parser.add_argument("--exclude-probe-steps", nargs="*", default=[])
    parser.add_argument("--exclude-query-probe-dir", default="")
    parser.add_argument("--max-sentences", type=int, default=0)
    parser.add_argument("--max-mentions", type=int, default=0)
    parser.add_argument("--write-mention-head-rows", action="store_true", default=False)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    from llava.mm_utils import get_model_name_from_path
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init

    disable_torch_init()
    model_name = get_model_name_from_path(os.path.expanduser(args.model_path))
    tokenizer, model, image_processor, _ = load_pretrained_model(args.model_path, args.model_base, model_name)
    model.eval()
    if args.model_path == "liuhaotian/llava-v1.6-34b":
        model.config.img_start_pos = 33
        model.config.img_length = 1948
    else:
        model.config.img_start_pos = 35
        model.config.img_length = 576

    heads, priors, prior_source = load_head_priors(
        args.prior_path,
        top_k=args.top_k,
        default_heads=default_heads_for_model(args.model_path),
    )
    adhh_ranks = {head_key(layer, head): idx + 1 for idx, (layer, head) in enumerate(heads)}

    all_sentences = load_eval_sentences(args.eval_results)
    exclude_ids = collect_exclude_ids(args)
    sentences = filter_sentences(
        all_sentences,
        match_eval_results=args.match_eval_results,
        exclude_image_ids=exclude_ids,
        max_sentences=args.max_sentences,
    )
    print(
        f"[info] sentences: input={len(all_sentences)} "
        f"after_filter={len(sentences)} excluded_ids={len(exclude_ids)}"
    )

    mention_head_rows = []
    compact_rows = []
    skipped = Counter()
    processed_mentions = 0

    for sentence in tqdm(sentences, desc="sentences"):
        mentions = prepare_mentions(tokenizer, sentence, score_span=args.score_span)
        if not mentions:
            skipped["no_aligned_mentions"] += 1
            continue

        prompt_ids, image_tensor, image_size = build_prompt_inputs(
            image_name_from_sentence(sentence, args.image_split),
            args.image_folder,
            tokenizer,
            image_processor,
            model.config,
            args.conv_mode,
            image_split=args.image_split,
        )
        prompt_ids = prompt_ids.to(device="cuda", non_blocking=True)
        image_tensor = image_tensor.to(dtype=torch.float16, device="cuda", non_blocking=True)
        position_cache = {}
        for mention in mentions:
            if args.max_mentions and processed_mentions >= args.max_mentions:
                break
            for token_pos in mention["score_positions"]:
                if token_pos < 0 or token_pos >= len(mention["caption_ids"]):
                    continue
                if token_pos not in position_cache:
                    prefix_ids = mention["caption_ids"][:token_pos]
                    output = one_step_attention(model, prompt_ids, prefix_ids, image_tensor, image_size)
                    attn_layers = generation_attentions_to_layers(getattr(output, "attentions", None))
                    if attn_layers is None:
                        raise RuntimeError("Generation did not return attentions.")
                    position_cache[token_pos] = score_head_features(
                        attn_layers,
                        prompt_len=int(prompt_ids.shape[1]),
                        model_config=model.config,
                        caption_prefix_len=len(prefix_ids),
                    )
                target_token_id = int(mention["caption_ids"][token_pos])
                base = {
                    "question_id": sentence.get("question_id", sentence.get("image_id", "")),
                    "image_id": sentence.get("image_id", ""),
                    "image": image_name_from_sentence(sentence, args.image_split),
                    "word": mention["word"],
                    "node_word": mention["node_word"],
                    "word_idx": mention.get("word_idx", ""),
                    "label": int(mention["label"]),
                    "label_name": mention["label_name"],
                    "token_pos": int(mention["token_pos"]),
                    "score_token_pos": int(token_pos),
                    "token_len": int(mention.get("token_len", 1)),
                    "target_token_ids": str(target_token_id),
                    "target_tokens": tokenizer.decode([target_token_id]).replace("\n", "\\n"),
                }
                for feature_row in position_cache[token_pos]:
                    row = {**base, **feature_row}
                    row["in_adhh_top20"] = int(row["head_key"] in adhh_ranks)
                    row["adhh_rank"] = adhh_ranks.get(row["head_key"], "")
                    if args.write_mention_head_rows:
                        mention_head_rows.append(row)
                    compact_rows.append(compact_mention_row(row))
            processed_mentions += 1
        if args.max_mentions and processed_mentions >= args.max_mentions:
            break

    rows_for_summary = mention_head_rows if args.write_mention_head_rows else compact_rows
    head_rows = summarize_heads(rows_for_summary, adhh_ranks)
    layer_rows = summarize_layers(head_rows)
    adhh_rows = [row for row in head_rows if int(row.get("in_adhh_top20", 0))]
    top_ratio_rows = sorted(
        head_rows,
        key=lambda row: (
            -(row.get("hall_over_grounded_text_mass") or 0.0),
            -(row.get("auroc_high_text_mass") or 0.0),
        )
    )[:100]
    top_auc_rows = sorted(
        head_rows,
        key=lambda row: (
            -(row.get("auroc_high_text_mass") or 0.0),
            -(row.get("hall_over_grounded_text_mass") or 0.0),
        )
    )[:100]

    write_csv(os.path.join(args.output_dir, "head_text_mass_summary.csv"), head_rows)
    write_csv(os.path.join(args.output_dir, "head_text_mass_layer_summary.csv"), layer_rows)
    write_csv(os.path.join(args.output_dir, "adhh_head_text_mass_summary.csv"), sorted(adhh_rows, key=lambda row: row["adhh_rank"]))
    write_csv(os.path.join(args.output_dir, "top_text_mass_ratio_heads.csv"), top_ratio_rows)
    write_csv(os.path.join(args.output_dir, "top_text_mass_auc_heads.csv"), top_auc_rows)
    if args.write_mention_head_rows:
        write_csv(os.path.join(args.output_dir, "head_text_mass_mention_rows.csv"), mention_head_rows)

    summary = {
        "eval_results": args.eval_results,
        "n_input_sentences": len(all_sentences),
        "n_sentences": len(sentences),
        "n_processed_mentions": processed_mentions,
        "n_head_rows": len(head_rows),
        "label_counts": count_unique_mentions(rows_for_summary),
        "prior_source": prior_source,
        "adhh_heads": heads,
        "skipped": dict(skipped),
        "outputs": {
            "head_summary": os.path.join(args.output_dir, "head_text_mass_summary.csv"),
            "layer_summary": os.path.join(args.output_dir, "head_text_mass_layer_summary.csv"),
            "adhh_summary": os.path.join(args.output_dir, "adhh_head_text_mass_summary.csv"),
            "top_ratio": os.path.join(args.output_dir, "top_text_mass_ratio_heads.csv"),
            "top_auc": os.path.join(args.output_dir, "top_text_mass_auc_heads.csv"),
        },
        "top_text_mass_auc_heads": top_auc_rows[:20],
        "top_text_mass_ratio_heads": top_ratio_rows[:20],
    }
    with open(os.path.join(args.output_dir, "head_text_mass_asymmetry_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("[summary] head text-mass asymmetry")
    print(json.dumps({
        "n_processed_mentions": processed_mentions,
        "n_head_rows": len(head_rows),
        "label_counts": summary["label_counts"],
        "top_text_mass_auc_heads": [
            {
                "head_key": row["head_key"],
                "auroc_high_text_mass": row["auroc_high_text_mass"],
                "hall_over_grounded_text_mass": row["hall_over_grounded_text_mass"],
                "in_adhh_top20": row["in_adhh_top20"],
                "adhh_rank": row["adhh_rank"],
            }
            for row in top_auc_rows[:10]
        ],
    }, indent=2))


if __name__ == "__main__":
    main()

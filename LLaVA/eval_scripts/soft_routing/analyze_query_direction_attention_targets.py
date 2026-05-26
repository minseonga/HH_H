import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict

import numpy as np
import torch
from tqdm import tqdm

from eval_scripts.soft_routing.analyze_query_direction_chair_alignment import (
    aggregate_span_scores,
    build_prompt_inputs,
    collect_exclude_ids,
    filter_sentences,
    image_name_from_sentence,
    load_eval_sentences,
    prepare_mentions,
    select_directions,
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


def l2_normalize(vector, eps=1e-12):
    denom = float(np.linalg.norm(vector))
    if denom <= eps:
        return vector
    return vector / denom


def generation_attentions_to_layers(attentions):
    if attentions is None:
        return None
    # HF generation for decoder-only models usually returns:
    #   tuple[num_generated_steps](tuple[num_layers](B,H,Q,K)).
    if isinstance(attentions, (tuple, list)) and attentions:
        first = attentions[0]
        if isinstance(first, (tuple, list)):
            return first
        return attentions
    return None


def token_text(tokenizer, token_id):
    try:
        return tokenizer.decode([int(token_id)]).replace("\n", "\\n")
    except Exception:
        return ""


def categorize_index(index, img_start, img_end, expanded_prompt_len, total_len, prefix_ids):
    index = int(index)
    if index < img_start:
        return "prefix_before_image", "", ""
    if img_start <= index < img_end:
        return "image", index - img_start, ""
    if index < expanded_prompt_len:
        return "prompt_text", "", ""
    caption_offset = index - expanded_prompt_len
    if caption_offset < 0:
        return "unknown", "", ""
    if caption_offset >= len(prefix_ids):
        return "future_or_current", caption_offset, ""
    if index == total_len - 1:
        return "generated_previous_token", caption_offset, ""
    return "generated_prefix", caption_offset, ""


def object_span_index(mentions):
    spans = []
    for mention in mentions:
        label_name = str(mention.get("label_name", ""))
        token_pos = int(mention.get("token_pos", -1))
        token_len = int(mention.get("token_len", 1))
        if token_pos < 0:
            continue
        spans.append((token_pos, token_pos + max(token_len, 1), label_name, mention.get("word", "")))
    return spans


def object_at_caption_offset(offset, spans):
    for start, end, label_name, word in spans:
        if start <= int(offset) < end:
            return label_name, word
    return "", ""


def one_step_attention(model, tokenizer, prompt_ids, prefix_ids, image_tensor, image_size):
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


def score_and_attention_position(
    model,
    tokenizer,
    prompt_ids,
    image_tensor,
    image_size,
    caption_ids,
    token_pos,
    directions,
    query_normalization,
    object_spans,
    top_attention_k,
):
    prefix_ids = caption_ids[:token_pos]
    model.config.query_diagnostics = []
    output = one_step_attention(model, tokenizer, prompt_ids, prefix_ids, image_tensor, image_size)
    query_records = list(getattr(model.config, "query_diagnostics", []) or [])
    query_by_head = {str(record.get("head_key")): record for record in query_records}

    attn_layers = generation_attentions_to_layers(getattr(output, "attentions", None))
    if attn_layers is None:
        raise RuntimeError("Generation did not return attentions; check output_attentions=True support.")

    img_start = int(getattr(model.config, "img_start_pos", 35))
    img_length = int(getattr(model.config, "img_length", 576))
    img_end = img_start + img_length
    expanded_prompt_len = int(prompt_ids.shape[1]) - 1 + img_length

    results = {}
    for item in directions:
        key = item["head_key"]
        record = query_by_head.get(key)
        if record is None:
            continue
        query = record["query"]
        if isinstance(query, torch.Tensor):
            query = query.numpy()
        query = np.asarray(query, dtype=np.float32)
        if query_normalization == "l2":
            query = l2_normalize(query)
        elif query_normalization != "none":
            raise ValueError(f"Unknown query normalization={query_normalization}")
        score = float(np.dot(query, item["direction"]))
        margin = score - float(item["threshold"])

        layer = int(item["layer"])
        head = int(item["head"])
        layer_attention = attn_layers[layer]
        attention = layer_attention[0, head, -1, :].detach().float().cpu()
        total_len = int(attention.shape[-1])
        top_values, top_indices = torch.topk(attention, k=min(int(top_attention_k), total_len))

        img_mass = float(attention[img_start:min(img_end, total_len)].sum().item()) if img_start < total_len else 0.0
        prefix_before_image_mass = float(attention[:min(img_start, total_len)].sum().item())
        prompt_text_mass = (
            float(attention[min(img_end, total_len):min(expanded_prompt_len, total_len)].sum().item())
            if img_end < total_len else 0.0
        )
        generated_mass = float(attention[min(expanded_prompt_len, total_len):].sum().item())
        recent_start = max(expanded_prompt_len, total_len - 16)
        recent_generated_mass = float(attention[recent_start:].sum().item()) if recent_start < total_len else 0.0
        prev_token_attention = float(attention[-1].item()) if total_len > 0 else 0.0

        top_rows = []
        for rank, (value, index) in enumerate(zip(top_values.tolist(), top_indices.tolist()), start=1):
            region, image_offset, caption_offset = categorize_index(
                index,
                img_start,
                img_end,
                expanded_prompt_len,
                total_len,
                prefix_ids,
            )
            target_token = ""
            object_label = ""
            object_word = ""
            if region.startswith("generated") and caption_offset != "":
                offset = int(caption_offset)
                if 0 <= offset < len(prefix_ids):
                    target_token = token_text(tokenizer, prefix_ids[offset])
                object_label, object_word = object_at_caption_offset(offset, object_spans)
            top_rows.append({
                "attn_rank": rank,
                "attn_value": float(value),
                "attn_index": int(index),
                "attn_region": region,
                "attn_img_offset": image_offset,
                "attn_caption_offset": caption_offset,
                "attn_token": target_token,
                "attn_object_label": object_label,
                "attn_object_word": object_word,
            })

        top1 = top_rows[0] if top_rows else {}
        results[key] = {
            "score": score,
            "margin": margin,
            "img_mass": img_mass,
            "prefix_before_image_mass": prefix_before_image_mass,
            "prompt_text_mass": prompt_text_mass,
            "generated_mass": generated_mass,
            "recent_generated_mass": recent_generated_mass,
            "prev_token_attention": prev_token_attention,
            "full_top1_index": top1.get("attn_index", ""),
            "full_top1_attention": top1.get("attn_value", ""),
            "full_top1_region": top1.get("attn_region", ""),
            "full_top1_img_offset": top1.get("attn_img_offset", ""),
            "full_top1_caption_offset": top1.get("attn_caption_offset", ""),
            "full_top1_token": top1.get("attn_token", ""),
            "full_top1_object_label": top1.get("attn_object_label", ""),
            "full_top1_object_word": top1.get("attn_object_word", ""),
            "top_attention_rows": top_rows,
        }
    return results


def summarize(rows, margin_quantile):
    if not rows:
        return [], []
    margin_cutoff = percentile([row["margin"] for row in rows], margin_quantile)
    groups = defaultdict(list)
    for row in rows:
        groups["all"].append(row)
        groups[f"label:{row['label_name']}"].append(row)
        groups[f"margin_positive:{int(float(row['margin']) > 0.0)}"].append(row)
        if margin_cutoff is not None:
            groups[f"margin_top_q{margin_quantile:g}:{int(float(row['margin']) >= margin_cutoff)}"].append(row)
    summary_rows = []
    region_rows = []
    numeric_keys = [
        "score",
        "margin",
        "img_mass",
        "prefix_before_image_mass",
        "prompt_text_mass",
        "generated_mass",
        "recent_generated_mass",
        "prev_token_attention",
        "full_top1_attention",
    ]
    for group, items in sorted(groups.items()):
        row = {
            "group": group,
            "n": len(items),
            "margin_cutoff": margin_cutoff if group.startswith("margin_top") else "",
        }
        for key in numeric_keys:
            row[f"mean_{key}"] = mean([item.get(key) for item in items])
            row[f"p50_{key}"] = percentile([item.get(key) for item in items], 0.5)
            row[f"p90_{key}"] = percentile([item.get(key) for item in items], 0.9)
        summary_rows.append(row)

        counts = Counter(str(item.get("full_top1_region", "")) for item in items)
        for region, count in counts.most_common():
            region_rows.append({
                "group": group,
                "full_top1_region": region,
                "count": count,
                "rate": count / len(items) if items else 0.0,
                "total": len(items),
            })
    return summary_rows, region_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-results", required=True)
    parser.add_argument("--calibration-npz", required=True)
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--image-split", default="val2014")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--conv-mode", default="vicuna_v1")
    parser.add_argument("--direction-top-k", type=int, default=1)
    parser.add_argument("--min-direction-auroc", type=float, default=0.0)
    parser.add_argument("--select-by", choices=["high", "abs"], default="high")
    parser.add_argument("--query-normalization", choices=["l2", "none"], default="l2")
    parser.add_argument("--score-span", choices=["first", "all"], default="first")
    parser.add_argument("--span-aggregation", choices=["max", "mean"], default="max")
    parser.add_argument("--top-attention-k", type=int, default=5)
    parser.add_argument("--high-margin-quantile", type=float, default=0.9)
    parser.add_argument("--match-eval-results", default="")
    parser.add_argument("--exclude-eval-results", nargs="*", default=[])
    parser.add_argument("--exclude-image-ids", nargs="*", default=[])
    parser.add_argument("--exclude-probe-steps", nargs="*", default=[])
    parser.add_argument("--exclude-query-probe-dir", default="")
    parser.add_argument("--max-sentences", type=int, default=0)
    parser.add_argument("--max-mentions", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    directions = select_directions(
        args.calibration_npz,
        args.direction_top_k,
        args.min_direction_auroc,
        select_by=args.select_by,
    )
    if not directions:
        raise ValueError("No query directions selected.")
    write_csv(os.path.join(args.output_dir, "selected_query_directions.csv"), [
        {
            "rank": item["rank"],
            "layer": item["layer"],
            "head": item["head"],
            "head_key": item["head_key"],
            "threshold": item["threshold"],
            "calibration_test_auroc": item["calibration_test_auroc"],
            "calibration_oriented_sign": item["calibration_oriented_sign"],
        }
        for item in directions
    ])

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

    selected_layers = sorted({item["layer"] for item in directions})
    selected_heads = sorted({item["head"] for item in directions})
    model.config.record_query_diagnostics = True
    model.config.query_record_all_heads = False
    model.config.query_record_heads = selected_heads
    model.config.query_record_min_layer = min(selected_layers)
    model.config.query_record_max_layer = max(selected_layers)
    model.config.query_record_batch_index = 0

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

    mention_rows = []
    top_attention_rows = []
    skipped = Counter()
    processed_mentions = 0
    for sentence in tqdm(sentences, desc="sentences"):
        mentions = prepare_mentions(tokenizer, sentence, score_span=args.score_span)
        if not mentions:
            skipped["no_aligned_mentions"] += 1
            continue
        object_spans = object_span_index(mentions)

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
            span_scores = []
            position_results = []
            target_token_ids = []
            for token_pos in mention["score_positions"]:
                if token_pos < 0 or token_pos >= len(mention["caption_ids"]):
                    continue
                if token_pos not in position_cache:
                    position_cache[token_pos] = score_and_attention_position(
                        model,
                        tokenizer,
                        prompt_ids,
                        image_tensor,
                        image_size,
                        mention["caption_ids"],
                        token_pos,
                        directions,
                        args.query_normalization,
                        object_spans,
                        args.top_attention_k,
                    )
                current = position_cache[token_pos]
                score_map = {
                    key: {"score": value["score"], "margin": value["margin"]}
                    for key, value in current.items()
                }
                span_scores.append(score_map)
                position_results.append(current)
                target_token_ids.append(int(mention["caption_ids"][token_pos]))
            if not span_scores:
                skipped["no_scored_positions"] += 1
                continue
            head_scores = aggregate_span_scores(span_scores, directions, args.span_aggregation)
            if not head_scores:
                skipped["no_direction_records"] += 1
                continue

            for item in directions:
                key = item["head_key"]
                score = head_scores.get(key)
                if score is None:
                    continue
                if args.span_aggregation == "max":
                    best_position = max(
                        (result[key] for result in position_results if key in result),
                        key=lambda row: row["margin"],
                    )
                else:
                    candidates = [result[key] for result in position_results if key in result]
                    best_position = candidates[0]
                row = {
                    "question_id": sentence.get("question_id", sentence.get("image_id", "")),
                    "image_id": sentence.get("image_id", ""),
                    "image": image_name_from_sentence(sentence, args.image_split),
                    "word": mention["word"],
                    "node_word": mention["node_word"],
                    "word_idx": mention.get("word_idx", ""),
                    "label": int(mention["label"]),
                    "label_name": mention["label_name"],
                    "token_pos": int(mention["token_pos"]),
                    "token_len": int(mention.get("token_len", 1)),
                    "score_positions": " ".join(str(pos) for pos in mention["score_positions"]),
                    "target_token_ids": " ".join(str(item_id) for item_id in target_token_ids),
                    "target_tokens": " ".join(token_text(tokenizer, item_id) for item_id in target_token_ids),
                    "rank": item["rank"],
                    "head_key": key,
                    "layer": item["layer"],
                    "head": item["head"],
                    "score": score["score"],
                    "margin": score["margin"],
                    "threshold": item["threshold"],
                    "calibration_test_auroc": item["calibration_test_auroc"],
                }
                for metric in [
                    "img_mass",
                    "prefix_before_image_mass",
                    "prompt_text_mass",
                    "generated_mass",
                    "recent_generated_mass",
                    "prev_token_attention",
                    "full_top1_index",
                    "full_top1_attention",
                    "full_top1_region",
                    "full_top1_img_offset",
                    "full_top1_caption_offset",
                    "full_top1_token",
                    "full_top1_object_label",
                    "full_top1_object_word",
                ]:
                    row[metric] = best_position.get(metric, "")
                mention_rows.append(row)

                for top_row in best_position.get("top_attention_rows", []):
                    top_attention_rows.append({
                        **{k: row[k] for k in [
                            "question_id",
                            "image_id",
                            "image",
                            "word",
                            "node_word",
                            "label",
                            "label_name",
                            "token_pos",
                            "rank",
                            "head_key",
                            "layer",
                            "head",
                            "score",
                            "margin",
                        ]},
                        **top_row,
                    })
            processed_mentions += 1
        if args.max_mentions and processed_mentions >= args.max_mentions:
            break

    model.config.record_query_diagnostics = False
    model.config.query_diagnostics = None

    summary_rows, region_rows = summarize(mention_rows, args.high_margin_quantile)
    image_sink_rows = []
    high_cutoff = percentile([row["margin"] for row in mention_rows], args.high_margin_quantile)
    high_top = [
        row for row in top_attention_rows
        if high_cutoff is not None
        and float(row.get("margin", 0.0)) >= high_cutoff
        and row.get("attn_region") == "image"
    ]
    sink_counts = Counter(str(row.get("attn_img_offset", "")) for row in high_top if str(row.get("attn_img_offset", "")) != "")
    total_high_img = sum(sink_counts.values())
    for offset, count in sink_counts.most_common(50):
        image_sink_rows.append({
            "img_offset": offset,
            "count": count,
            "rate_among_high_margin_image_topk": count / total_high_img if total_high_img else 0.0,
            "total_high_margin_image_topk": total_high_img,
        })

    write_csv(os.path.join(args.output_dir, "query_attention_target_mentions.csv"), mention_rows)
    write_csv(os.path.join(args.output_dir, "query_attention_top_targets.csv"), top_attention_rows)
    write_csv(os.path.join(args.output_dir, "query_attention_target_summary.csv"), summary_rows)
    write_csv(os.path.join(args.output_dir, "query_attention_top1_region_summary.csv"), region_rows)
    write_csv(os.path.join(args.output_dir, "query_attention_image_sink_offsets.csv"), image_sink_rows)
    with open(os.path.join(args.output_dir, "query_attention_target_summary.json"), "w") as f:
        json.dump({
            "n_mentions": len(mention_rows),
            "n_top_attention_rows": len(top_attention_rows),
            "skipped": dict(skipped),
            "high_margin_quantile": args.high_margin_quantile,
            "high_margin_cutoff": high_cutoff,
            "outputs": {
                "mentions": os.path.join(args.output_dir, "query_attention_target_mentions.csv"),
                "top_targets": os.path.join(args.output_dir, "query_attention_top_targets.csv"),
                "summary": os.path.join(args.output_dir, "query_attention_target_summary.csv"),
                "top1_region_summary": os.path.join(args.output_dir, "query_attention_top1_region_summary.csv"),
                "image_sink_offsets": os.path.join(args.output_dir, "query_attention_image_sink_offsets.csv"),
            },
        }, f, indent=2)
    print("[summary] query attention target analysis")
    print(json.dumps({
        "n_mentions": len(mention_rows),
        "n_top_attention_rows": len(top_attention_rows),
        "skipped": dict(skipped),
        "high_margin_cutoff": high_cutoff,
    }, indent=2))


if __name__ == "__main__":
    main()

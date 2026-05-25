import argparse
import csv
import json
import math
import os
from collections import Counter

import numpy as np
import torch
from tqdm import tqdm

from eval_scripts.soft_routing.analyze_text_heavy_object_alignment import (
    align_mentions,
    auc_score,
    load_eval_sentences,
    object_mentions,
)


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


def mean(values):
    clean = []
    for value in values:
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            clean.append(value)
    return sum(clean) / len(clean) if clean else None


def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def l2_normalize(vector, eps=1e-12):
    denom = float(np.linalg.norm(vector))
    if denom <= eps:
        return vector
    return vector / denom


def parse_int_list(text):
    values = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    return values


def image_id_key(sentence):
    return str(sentence.get("image_id", sentence.get("question_id", "")))


def filter_sentences(sentences, match_eval_results="", max_sentences=0):
    if match_eval_results:
        match_ids = {image_id_key(item) for item in load_eval_sentences(match_eval_results)}
        sentences = [item for item in sentences if image_id_key(item) in match_ids]
    if max_sentences and max_sentences > 0:
        sentences = sentences[:max_sentences]
    return sentences


def image_name_from_sentence(sentence, image_split="val2014"):
    image = sentence.get("image")
    if image:
        return str(image)
    image_id = sentence.get("image_id", sentence.get("question_id"))
    if image_id is None:
        return ""
    try:
        image_id = int(image_id)
    except (TypeError, ValueError):
        return str(image_id)
    return f"COCO_{image_split}_{image_id:012d}.jpg"


def resolve_image_path(image_file, image_folder, image_split="val2014"):
    candidates = []
    image_file = str(image_file or "")
    if os.path.isabs(image_file):
        candidates.append(image_file)
    if image_file:
        candidates.extend([
            os.path.join(image_folder, image_file),
            os.path.join(image_folder, os.path.basename(image_file)),
            os.path.join(image_folder, image_split, image_file),
            os.path.join(image_folder, image_split, os.path.basename(image_file)),
        ])
    seen = set()
    unique = []
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    for candidate in unique:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        "Could not resolve image path. "
        f"image_file={image_file!r}, image_folder={image_folder!r}, "
        f"image_split={image_split!r}, tried={unique[:8]!r}"
    )


def build_prompt_inputs(image_file, image_folder, tokenizer, image_processor, model_config, conv_mode, image_split="val2014"):
    from PIL import Image

    from llava.constants import (
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
        IMAGE_TOKEN_INDEX,
    )
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token

    qs = "Please describe this image in detail."
    if model_config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    image_path = resolve_image_path(image_file, image_folder, image_split=image_split)
    image = Image.open(image_path).convert("RGB")
    image_tensor = process_images([image], image_processor, model_config)[0]
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    return input_ids.unsqueeze(0), image_tensor.unsqueeze(0), image.size


def clear_generation_interventions(model):
    for name in [
        "adaptive_deactivate",
        "soft_deactivate",
        "dynamic_deactivate",
        "attribution_soft_deactivate",
        "retention_aware_deactivate",
        "fixed_strength_deactivate",
        "record_intervention_diagnostics",
    ]:
        if hasattr(model.config, name):
            setattr(model.config, name, False)
    model.config.intervention_diagnostics = None


def one_step(model, prompt_ids, prefix_ids, image_tensor, image_size):
    if prefix_ids:
        prefix_tensor = torch.tensor(prefix_ids, device=prompt_ids.device, dtype=prompt_ids.dtype).unsqueeze(0)
        step_input = torch.cat([prompt_ids, prefix_tensor], dim=1)
    else:
        step_input = prompt_ids
    clear_generation_interventions(model)
    with torch.inference_mode():
        model.generate(
            step_input,
            images=image_tensor,
            image_sizes=[image_size],
            do_sample=False,
            temperature=0,
            top_p=None,
            num_beams=1,
            max_new_tokens=1,
            use_cache=True,
            output_scores=False,
            return_dict_in_generate=True,
        )


def select_directions(calibration_npz, top_k, min_auroc, select_by="high"):
    data = np.load(calibration_npz)
    layers = data["layers"].astype(int)
    heads = data["heads"].astype(int)
    directions = data["directions"].astype(np.float32)
    if "threshold_midpoint" in data.files:
        thresholds = data["threshold_midpoint"]
    else:
        thresholds = np.zeros((directions.shape[0],), dtype=np.float32)
    thresholds = thresholds.astype(np.float32)
    if "test_auroc" in data.files:
        test_auroc = data["test_auroc"]
    else:
        test_auroc = np.zeros((directions.shape[0],), dtype=np.float32)
    test_auroc = test_auroc.astype(np.float32)

    candidates = []
    for idx, auc in enumerate(test_auroc.tolist()):
        auc = float(auc)
        if select_by == "high":
            if auc < float(min_auroc):
                continue
            rank_score = auc
            sign = 1.0
        elif select_by == "abs":
            auc_abs = max(auc, 1.0 - auc)
            if auc_abs < float(min_auroc):
                continue
            rank_score = auc_abs
            sign = 1.0 if auc >= 0.5 else -1.0
        else:
            raise ValueError(f"Unknown select_by={select_by}")
        candidates.append((rank_score, idx, sign))
    candidates.sort(key=lambda item: item[0], reverse=True)
    if top_k and top_k > 0:
        candidates = candidates[:top_k]

    selected = []
    for rank, (_, idx, sign) in enumerate(candidates, start=1):
        layer = int(layers[idx])
        head = int(heads[idx])
        direction = directions[idx].astype(np.float32).copy()
        direction = l2_normalize(direction).astype(np.float32)
        threshold = float(thresholds[idx])
        if sign < 0:
            direction = -direction
            threshold = -threshold
        selected.append({
            "rank": rank,
            "layer": layer,
            "head": head,
            "head_key": f"{layer}:{head}",
            "direction": direction,
            "threshold": threshold,
            "calibration_test_auroc": float(test_auroc[idx]),
            "calibration_oriented_sign": sign,
        })
    return selected


def prepare_mentions(tokenizer, sentence, score_span="first"):
    caption = sentence.get("caption") or sentence.get("text") or ""
    mentions = object_mentions(sentence)
    caption_ids, aligned = align_mentions(tokenizer, caption, mentions)
    rows = []
    for idx, mention in enumerate(aligned):
        token_pos = int(mention["token_pos"])
        token_len = int(mention.get("token_len", 1))
        if score_span == "all":
            positions = list(range(token_pos, min(token_pos + token_len, len(caption_ids))))
        else:
            positions = [token_pos]
        rows.append({
            **mention,
            "mention_index": idx,
            "caption": caption,
            "caption_ids": caption_ids,
            "score_positions": positions,
        })
    return rows


def score_position(
    model,
    prompt_ids,
    image_tensor,
    image_size,
    caption_ids,
    token_pos,
    directions,
    normalization,
):
    prefix_ids = caption_ids[:token_pos]
    target_token_id = int(caption_ids[token_pos])
    model.config.query_diagnostics = []
    one_step(
        model,
        prompt_ids,
        prefix_ids,
        image_tensor,
        image_size,
    )
    records = list(getattr(model.config, "query_diagnostics", []) or [])
    by_head = {str(record.get("head_key")): record for record in records}
    scores = {}
    for item in directions:
        record = by_head.get(item["head_key"])
        if record is None:
            continue
        query = record["query"]
        if isinstance(query, torch.Tensor):
            query = query.numpy()
        query = np.asarray(query, dtype=np.float32)
        if normalization == "l2":
            query = l2_normalize(query)
        elif normalization != "none":
            raise ValueError(f"Unknown normalization={normalization}")
        score = float(np.dot(query, item["direction"]))
        scores[item["head_key"]] = {
            "score": score,
            "margin": score - float(item["threshold"]),
        }
    return scores, target_token_id


def aggregate_span_scores(span_scores, directions, mode):
    output = {}
    for item in directions:
        key = item["head_key"]
        values = [scores[key] for scores in span_scores if key in scores]
        if not values:
            continue
        if mode == "max":
            best_score = max(values, key=lambda row: row["score"])
            best_margin = max(values, key=lambda row: row["margin"])
            output[key] = {
                "score": float(best_score["score"]),
                "margin": float(best_margin["margin"]),
            }
        elif mode == "mean":
            output[key] = {
                "score": mean([row["score"] for row in values]),
                "margin": mean([row["margin"] for row in values]),
            }
        else:
            raise ValueError(f"Unknown span aggregation={mode}")
    return output


def add_ensemble_features(row, directions, head_scores, ensemble_top_ks, temperature):
    ranked = [item for item in directions if item["head_key"] in head_scores]
    for k in ensemble_top_ks:
        active = ranked[:min(k, len(ranked))]
        if not active:
            continue
        scores = [head_scores[item["head_key"]]["score"] for item in active]
        margins = [head_scores[item["head_key"]]["margin"] for item in active]
        hard = [1.0 if margin > 0 else 0.0 for margin in margins]
        sig = [sigmoid(margin / max(float(temperature), 1e-6)) for margin in margins]
        prefix = f"top{k}"
        row[f"{prefix}_score_mean"] = mean(scores)
        row[f"{prefix}_score_max"] = max(scores)
        row[f"{prefix}_margin_mean"] = mean(margins)
        row[f"{prefix}_margin_max"] = max(margins)
        row[f"{prefix}_hard_rate"] = mean(hard)
        row[f"{prefix}_sigmoid_mean"] = mean(sig)


def auc_rows_from_features(mention_rows, feature_names):
    rows = []
    labels = [int(row["label"]) for row in mention_rows]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    for feature in feature_names:
        values = []
        feature_labels = []
        for row in mention_rows:
            value = row.get(feature)
            if value is None or value == "":
                continue
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value):
                continue
            values.append(value)
            feature_labels.append(int(row["label"]))
        pos_values = [value for value, label in zip(values, feature_labels) if label == 1]
        neg_values = [value for value, label in zip(values, feature_labels) if label == 0]
        auc = auc_score(feature_labels, values)
        rows.append({
            "feature": feature,
            "n": len(values),
            "n_pos": len(pos_values),
            "n_neg": len(neg_values),
            "total_mentions_pos": n_pos,
            "total_mentions_neg": n_neg,
            "pos_mean": mean(pos_values),
            "neg_mean": mean(neg_values),
            "pos_minus_neg": (
                mean(pos_values) - mean(neg_values)
                if mean(pos_values) is not None and mean(neg_values) is not None
                else None
            ),
            "auroc_high_predicts_hallucinated": auc,
            "auroc_low_predicts_hallucinated": (1.0 - auc) if auc is not None else None,
        })
    return rows


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
    parser.add_argument("--direction-top-k", type=int, default=20)
    parser.add_argument("--min-direction-auroc", type=float, default=0.65)
    parser.add_argument("--select-by", choices=["high", "abs"], default="high")
    parser.add_argument("--query-normalization", choices=["l2", "none"], default="l2")
    parser.add_argument("--ensemble-top-ks", default="1,3,5,10,20")
    parser.add_argument("--gate-temperature", type=float, default=0.1)
    parser.add_argument("--score-span", choices=["first", "all"], default="first")
    parser.add_argument("--span-aggregation", choices=["max", "mean"], default="max")
    parser.add_argument("--match-eval-results", default="")
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
        raise ValueError(
            f"No query directions selected from {args.calibration_npz}; "
            f"lower --min-direction-auroc or check --select-by."
        )
    direction_rows = [
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
    ]
    write_csv(os.path.join(args.output_dir, "selected_query_directions.csv"), direction_rows)

    from llava.mm_utils import get_model_name_from_path
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init

    disable_torch_init()
    model_name = get_model_name_from_path(os.path.expanduser(args.model_path))
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
    )
    model.eval()

    selected_layers = sorted({item["layer"] for item in directions})
    selected_heads = sorted({item["head"] for item in directions})
    model.config.record_query_diagnostics = True
    model.config.query_record_all_heads = False
    model.config.query_record_heads = selected_heads
    model.config.query_record_min_layer = min(selected_layers)
    model.config.query_record_max_layer = max(selected_layers)
    model.config.query_record_batch_index = 0

    sentences = filter_sentences(
        load_eval_sentences(args.eval_results),
        match_eval_results=args.match_eval_results,
        max_sentences=args.max_sentences,
    )
    ensemble_top_ks = parse_int_list(args.ensemble_top_ks)

    mention_rows = []
    head_score_rows = []
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
            span_scores = []
            target_token_ids = []
            for token_pos in mention["score_positions"]:
                if token_pos < 0 or token_pos >= len(mention["caption_ids"]):
                    continue
                if token_pos not in position_cache:
                    position_cache[token_pos] = score_position(
                        model,
                        prompt_ids,
                        image_tensor,
                        image_size,
                        mention["caption_ids"],
                        token_pos,
                        directions,
                        args.query_normalization,
                    )
                scores, target_token_id = position_cache[token_pos]
                span_scores.append(scores)
                target_token_ids.append(target_token_id)
            if not span_scores:
                skipped["no_scored_positions"] += 1
                continue
            head_scores = aggregate_span_scores(span_scores, directions, args.span_aggregation)
            if not head_scores:
                skipped["no_direction_records"] += 1
                continue

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
                "target_token_ids": " ".join(str(item) for item in target_token_ids),
                "target_tokens": " ".join(tokenizer.decode([item]).replace("\n", "\\n") for item in target_token_ids),
            }
            add_ensemble_features(row, directions, head_scores, ensemble_top_ks, args.gate_temperature)
            for item in directions:
                score = head_scores.get(item["head_key"])
                if score is None:
                    continue
                row[f"{item['head_key']}_score"] = score["score"]
                row[f"{item['head_key']}_margin"] = score["margin"]
                head_score_rows.append({
                    "question_id": row["question_id"],
                    "image_id": row["image_id"],
                    "image": row["image"],
                    "word": row["word"],
                    "node_word": row["node_word"],
                    "label": row["label"],
                    "label_name": row["label_name"],
                    "token_pos": row["token_pos"],
                    "rank": item["rank"],
                    "head_key": item["head_key"],
                    "layer": item["layer"],
                    "head": item["head"],
                    "score": score["score"],
                    "margin": score["margin"],
                    "threshold": item["threshold"],
                    "calibration_test_auroc": item["calibration_test_auroc"],
                })
            mention_rows.append(row)
            processed_mentions += 1
        if args.max_mentions and processed_mentions >= args.max_mentions:
            break

    model.config.record_query_diagnostics = False
    model.config.query_diagnostics = None

    feature_names = []
    for k in ensemble_top_ks:
        for suffix in ("score_mean", "score_max", "margin_mean", "margin_max", "hard_rate", "sigmoid_mean"):
            feature_names.append(f"top{k}_{suffix}")
    for item in directions:
        feature_names.append(f"{item['head_key']}_score")
        feature_names.append(f"{item['head_key']}_margin")

    mention_auc_rows = auc_rows_from_features(mention_rows, feature_names)
    head_auc_rows = []
    by_head = {}
    for row in head_score_rows:
        by_head.setdefault(row["head_key"], []).append(row)
    for item in directions:
        rows = by_head.get(item["head_key"], [])
        labels = [int(row["label"]) for row in rows]
        scores = [float(row["score"]) for row in rows]
        margins = [float(row["margin"]) for row in rows]
        score_auc = auc_score(labels, scores)
        margin_auc = auc_score(labels, margins)
        pos_scores = [score for score, label in zip(scores, labels) if label == 1]
        neg_scores = [score for score, label in zip(scores, labels) if label == 0]
        head_auc_rows.append({
            "rank": item["rank"],
            "head_key": item["head_key"],
            "layer": item["layer"],
            "head": item["head"],
            "n": len(rows),
            "n_pos": sum(labels),
            "n_neg": len(labels) - sum(labels),
            "pos_mean_score": mean(pos_scores),
            "neg_mean_score": mean(neg_scores),
            "pos_minus_neg_score": (
                mean(pos_scores) - mean(neg_scores)
                if mean(pos_scores) is not None and mean(neg_scores) is not None
                else None
            ),
            "auroc_score_high_predicts_hallucinated": score_auc,
            "auroc_score_low_predicts_hallucinated": (1.0 - score_auc) if score_auc is not None else None,
            "auroc_margin_high_predicts_hallucinated": margin_auc,
            "auroc_margin_low_predicts_hallucinated": (1.0 - margin_auc) if margin_auc is not None else None,
            "threshold": item["threshold"],
            "calibration_test_auroc": item["calibration_test_auroc"],
            "calibration_oriented_sign": item["calibration_oriented_sign"],
        })

    write_csv(os.path.join(args.output_dir, "query_direction_chair_mentions.csv"), mention_rows)
    write_csv(os.path.join(args.output_dir, "query_direction_chair_head_scores.csv"), head_score_rows)
    write_csv(os.path.join(args.output_dir, "query_direction_chair_auc.csv"), mention_auc_rows)
    write_csv(os.path.join(args.output_dir, "query_direction_chair_head_auc.csv"), head_auc_rows)

    label_counts = Counter(row["label_name"] for row in mention_rows)
    summary = {
        "eval_results": args.eval_results,
        "calibration_npz": args.calibration_npz,
        "n_sentences": len(sentences),
        "n_mentions": len(mention_rows),
        "label_counts": dict(label_counts),
        "selected_direction_count": len(directions),
        "selected_directions": direction_rows,
        "ensemble_top_ks": ensemble_top_ks,
        "score_span": args.score_span,
        "span_aggregation": args.span_aggregation,
        "query_normalization": args.query_normalization,
        "skipped": dict(skipped),
        "top_auc_features": sorted(
            mention_auc_rows,
            key=lambda row: max(
                row["auroc_high_predicts_hallucinated"] or 0.0,
                row["auroc_low_predicts_hallucinated"] or 0.0,
            ),
            reverse=True,
        )[:20],
    }
    with open(os.path.join(args.output_dir, "query_direction_chair_alignment_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({
        "n_sentences": summary["n_sentences"],
        "n_mentions": summary["n_mentions"],
        "label_counts": summary["label_counts"],
        "selected_direction_count": summary["selected_direction_count"],
        "top_auc_features": summary["top_auc_features"][:10],
        "outputs": {
            "mentions": os.path.join(args.output_dir, "query_direction_chair_mentions.csv"),
            "auc": os.path.join(args.output_dir, "query_direction_chair_auc.csv"),
            "head_auc": os.path.join(args.output_dir, "query_direction_chair_head_auc.csv"),
        },
    }, indent=2))


if __name__ == "__main__":
    main()

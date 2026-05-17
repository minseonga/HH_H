import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from eval_scripts.soft_routing.head_prior_utils import default_heads_for_model, head_key, load_head_priors


SCORE_COLUMNS = [
    "mean_text_ratio",
    "mean_img_entropy_norm",
    "mean_text_ratio_img_entropy",
    "mean_text_mass",
    "mean_img_mass",
    "mean_text_img_log_ratio",
]


def load_eval_rows(path, max_samples):
    with open(path, "r") as f:
        data = json.load(f)
    rows = data["sentences"] if isinstance(data, dict) and "sentences" in data else data
    if max_samples and max_samples > 0:
        rows = rows[:max_samples]
    return rows


def image_name(row):
    if row.get("image"):
        return row["image"]
    image_id = int(row.get("image_id", row.get("question_id")))
    return f"COCO_val2014_{image_id:012d}.jpg"


def build_prompt_inputs(row, image_folder, tokenizer, image_processor, model_config, conv_mode):
    qs = row.get("text") or "Please describe this image in detail."
    if model_config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    image = Image.open(os.path.join(image_folder, image_name(row))).convert("RGB")
    image_tensor = process_images([image], image_processor, model_config)[0]
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    return input_ids.unsqueeze(0), image_tensor.unsqueeze(0), image.size


def configure_model(model, model_path, attention_head_path, top_k, head_prior_mode, adhh_threshold):
    heads, priors, prior_source = load_head_priors(
        attention_head_path,
        top_k=top_k,
        prior_mode=head_prior_mode,
        default_heads=default_heads_for_model(model_path),
    )
    model.config.hal_attention_heads = heads
    model.config.head_attribution_priors = priors
    model.config.head_attribution_prior_source = prior_source
    model.config.adhh_threshold = adhh_threshold
    model.config.soft_gamma = 0.75
    model.config.soft_temperature = 0.05
    if model_path == "liuhaotian/llava-v1.6-34b":
        model.config.img_start_pos = 33
        model.config.img_length = 1948
    else:
        model.config.img_start_pos = 35
        model.config.img_length = 576
    return heads, prior_source


def clear_intervention_modes(model):
    for name in [
        "adaptive_deactivate",
        "soft_deactivate",
        "dynamic_deactivate",
        "attribution_soft_deactivate",
        "retention_aware_deactivate",
        "visual_gate_deactivate",
        "fixed_strength_deactivate",
    ]:
        if hasattr(model.config, name):
            setattr(model.config, name, False)


def generate_warmup(model, prompt_ids, image_tensor, image_size, warmup_tokens):
    if warmup_tokens <= 0:
        return []
    clear_intervention_modes(model)
    model.config.record_intervention_diagnostics = False
    model.config.record_all_head_diagnostics = False
    with torch.inference_mode():
        output = model.generate(
            prompt_ids,
            images=image_tensor,
            image_sizes=[image_size],
            do_sample=False,
            temperature=0,
            top_p=None,
            num_beams=1,
            max_new_tokens=warmup_tokens,
            use_cache=True,
            return_dict_in_generate=True,
        )
    return output["sequences"][0, prompt_ids.shape[1]:].detach().cpu().tolist()


def record_all_head_diagnostics(model, prompt_ids, prefix_ids, image_tensor, image_size):
    if prefix_ids:
        prefix_tensor = torch.tensor(prefix_ids, device=prompt_ids.device, dtype=prompt_ids.dtype).unsqueeze(0)
        step_input = torch.cat([prompt_ids, prefix_tensor], dim=1)
    else:
        step_input = prompt_ids
    clear_intervention_modes(model)
    model.config.record_intervention_diagnostics = True
    model.config.record_all_head_diagnostics = True
    model.config.intervention_diagnostics = []
    model.config.diagnostic_output_start_pos = int(prompt_ids.shape[1] - 1 + getattr(model.config, "img_length", 576))
    model.config.diagnostic_recent_window = 16
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
            return_dict_in_generate=True,
        )
    records = list(getattr(model.config, "intervention_diagnostics", []) or [])
    model.config.record_intervention_diagnostics = False
    model.config.record_all_head_diagnostics = False
    return records


def aggregate_head_scores(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[record["head_key"]].append(record)

    rows = []
    for key, items in grouped.items():
        layer, head = key.split(":")
        row = {
            "layer": int(layer),
            "head": int(head),
            "head_key": key,
            "n": len(items),
        }
        for column in SCORE_COLUMNS:
            row[column] = float(np.mean([float(item.get(column.replace("mean_", ""), item.get(column, 0.0))) for item in items]))
        row["mean_text_ratio"] = float(np.mean([float(item.get("text_ratio", 0.0)) for item in items]))
        row["mean_img_entropy_norm"] = float(np.mean([float(item.get("img_entropy_norm", 0.0)) for item in items]))
        row["mean_text_ratio_img_entropy"] = float(np.mean([float(item.get("text_ratio_img_entropy", 0.0)) for item in items]))
        row["mean_text_mass"] = float(np.mean([float(item.get("text_mass", 0.0)) for item in items]))
        row["mean_img_mass"] = float(np.mean([float(item.get("img_mass", 0.0)) for item in items]))
        row["mean_text_img_log_ratio"] = float(np.mean([float(item.get("text_img_log_ratio", 0.0)) for item in items]))
        rows.append(row)
    return rows


def update_head_stats(stats, records):
    for record in records:
        key = record["head_key"]
        stats[key]["layer"] = int(record["layer"])
        stats[key]["head"] = int(record["head"])
        stats[key]["n"] += 1
        stats[key]["text_ratio"] += float(record.get("text_ratio", 0.0))
        stats[key]["img_entropy_norm"] += float(record.get("img_entropy_norm", 0.0))
        stats[key]["text_ratio_img_entropy"] += float(record.get("text_ratio_img_entropy", 0.0))
        stats[key]["text_mass"] += float(record.get("text_mass", 0.0))
        stats[key]["img_mass"] += float(record.get("img_mass", 0.0))
        stats[key]["text_img_log_ratio"] += float(record.get("text_img_log_ratio", 0.0))


def aggregate_head_scores_from_stats(stats):
    rows = []
    for key, sums in stats.items():
        n = max(int(sums["n"]), 1)
        rows.append({
            "layer": int(sums["layer"]),
            "head": int(sums["head"]),
            "head_key": key,
            "n": int(sums["n"]),
            "mean_text_ratio": float(sums["text_ratio"] / n),
            "mean_img_entropy_norm": float(sums["img_entropy_norm"] / n),
            "mean_text_ratio_img_entropy": float(sums["text_ratio_img_entropy"] / n),
            "mean_text_mass": float(sums["text_mass"] / n),
            "mean_img_mass": float(sums["img_mass"] / n),
            "mean_text_img_log_ratio": float(sums["text_img_log_ratio"] / n),
        })
    return rows


def add_layer_normalized_scores(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["layer"])].append(row)

    def minmax(items, key, value):
        values = [float(item[key]) for item in items]
        low = min(values)
        high = max(values)
        if high <= low:
            return 0.0
        return (float(value) - low) / (high - low)

    for items in grouped.values():
        for row in items:
            row["layer_norm_text_ratio"] = minmax(items, "mean_text_ratio", row["mean_text_ratio"])
            row["layer_norm_img_entropy"] = minmax(items, "mean_img_entropy_norm", row["mean_img_entropy_norm"])
            row["layer_norm_text_ratio_img_entropy"] = (
                row["layer_norm_text_ratio"] * row["layer_norm_img_entropy"]
            )
    return rows


def top_heads(rows, score_name, top_k, min_layer=None, max_layer=None):
    filtered = []
    for row in rows:
        layer = int(row["layer"])
        if min_layer is not None and layer < min_layer:
            continue
        if max_layer is not None and layer > max_layer:
            continue
        filtered.append(row)
    ranked = sorted(filtered, key=lambda item: item[score_name], reverse=True)
    return [[int(item["layer"]), int(item["head"])] for item in ranked[:top_k]]


def overlap_record(name, selected, reference):
    selected_set = {head_key(*head) for head in selected}
    reference_set = {head_key(*head) for head in reference}
    overlap = sorted(selected_set & reference_set)
    union = selected_set | reference_set
    return {
        "selector": name,
        "selected_k": len(selected_set),
        "reference_k": len(reference_set),
        "overlap": len(overlap),
        "overlap_rate": len(overlap) / max(len(reference_set), 1),
        "jaccard": len(overlap) / max(len(union), 1),
        "selected_heads": " ".join(sorted(selected_set)),
        "overlap_heads": " ".join(overlap),
    }


def rank_reference_heads(rows, reference_heads, selectors, min_layer=None, max_layer=None):
    filtered = []
    for row in rows:
        layer = int(row["layer"])
        if min_layer is not None and layer < min_layer:
            continue
        if max_layer is not None and layer > max_layer:
            continue
        filtered.append(row)

    rank_maps = {}
    for selector_name, score_name in selectors.items():
        ranked = sorted(filtered, key=lambda item: item[score_name], reverse=True)
        rank_maps[selector_name] = {
            row["head_key"]: {
                "rank": idx + 1,
                "score": float(row[score_name]),
            }
            for idx, row in enumerate(ranked)
        }

    wide_rows = []
    long_rows = []
    for idx, head in enumerate(reference_heads):
        key = head_key(*head)
        wide = {
            "adhh_order": idx + 1,
            "layer": int(head[0]),
            "head": int(head[1]),
            "head_key": key,
        }
        for selector_name in selectors:
            item = rank_maps[selector_name].get(key, {})
            rank = item.get("rank")
            score = item.get("score")
            wide[f"{selector_name}_rank"] = rank
            wide[f"{selector_name}_score"] = score
            wide[f"{selector_name}_in_top20"] = bool(rank is not None and rank <= 20)
            long_rows.append({
                "selector": selector_name,
                "adhh_order": idx + 1,
                "layer": int(head[0]),
                "head": int(head[1]),
                "head_key": key,
                "rank": rank,
                "score": score,
                "in_top20": bool(rank is not None and rank <= 20),
            })
        wide_rows.append(wide)
    return wide_rows, long_rows


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


def write_head_json(path, heads, rows, score_name, metadata):
    score_by_key = {row["head_key"]: row[score_name] for row in rows}
    payload = {
        **metadata,
        "selection_score": score_name,
        "hal_heads": heads,
        "hal_head_scores": [
            {
                "layer": int(layer),
                "head": int(head),
                "score": float(score_by_key.get(head_key(layer, head), 0.0)),
            }
            for layer, head in heads
        ],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-results", required=True)
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--conv-mode", default="vicuna_v1")
    parser.add_argument("--attention-head-path", default="")
    parser.add_argument("--head-prior-mode", default="auto", choices=["auto", "score", "rank", "uniform"])
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--warmup-tokens", type=int, default=0)
    parser.add_argument("--min-layer", type=int, default=None)
    parser.add_argument("--max-layer", type=int, default=None)
    parser.add_argument("--adhh-threshold", type=float, default=0.4)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    disable_torch_init()
    model_name = get_model_name_from_path(os.path.expanduser(args.model_path))
    tokenizer, model, image_processor, _ = load_pretrained_model(args.model_path, args.model_base, model_name)
    reference_heads, prior_source = configure_model(
        model,
        args.model_path,
        args.attention_head_path,
        args.top_k,
        args.head_prior_mode,
        args.adhh_threshold,
    )

    rows = load_eval_rows(args.eval_results, args.max_samples)
    head_stats = defaultdict(lambda: defaultdict(float))
    with open(os.path.join(args.output_dir, "all_head_records.jsonl"), "w") as record_file:
        for row in tqdm(rows):
            prompt_ids, image_tensor, image_size = build_prompt_inputs(
                row,
                args.image_folder,
                tokenizer,
                image_processor,
                model.config,
                args.conv_mode,
            )
            prompt_ids = prompt_ids.to(device="cuda", non_blocking=True)
            image_tensor = image_tensor.to(dtype=torch.float16, device="cuda", non_blocking=True)
            prefix_ids = generate_warmup(model, prompt_ids, image_tensor, image_size, args.warmup_tokens)
            records = record_all_head_diagnostics(model, prompt_ids, prefix_ids, image_tensor, image_size)
            sample_id = row.get("image_id", row.get("question_id"))
            for record in records:
                record["image_id"] = sample_id
                record["image"] = image_name(row)
                record["warmup_tokens"] = len(prefix_ids)
                record_file.write(json.dumps(record) + "\n")
            record_file.flush()
            update_head_stats(head_stats, records)

    head_rows = add_layer_normalized_scores(aggregate_head_scores_from_stats(head_stats))
    head_rows = sorted(head_rows, key=lambda item: item["mean_text_ratio_img_entropy"], reverse=True)
    write_csv(os.path.join(args.output_dir, "head_scores.csv"), head_rows)

    selectors = {
        "text_ratio": "mean_text_ratio",
        "image_entropy": "mean_img_entropy_norm",
        "text_ratio_x_image_entropy": "mean_text_ratio_img_entropy",
        "text_img_log_ratio": "mean_text_img_log_ratio",
        "layer_norm_text_ratio": "layer_norm_text_ratio",
        "layer_norm_image_entropy": "layer_norm_img_entropy",
        "layer_norm_text_ratio_x_image_entropy": "layer_norm_text_ratio_img_entropy",
    }
    overlap_rows = []
    metadata = {
        "num_samples": len(rows),
        "warmup_tokens": args.warmup_tokens,
        "reference_head_path": args.attention_head_path,
        "reference_prior_source": prior_source,
        "reference_heads": reference_heads,
        "top_k": args.top_k,
        "min_layer": args.min_layer,
        "max_layer": args.max_layer,
    }
    for selector_name, score_name in selectors.items():
        selected_heads = top_heads(head_rows, score_name, args.top_k, args.min_layer, args.max_layer)
        overlap_rows.append(overlap_record(selector_name, selected_heads, reference_heads))
        write_head_json(
            os.path.join(args.output_dir, f"{selector_name}_top{args.top_k}.json"),
            selected_heads,
            head_rows,
            score_name,
            {**metadata, "selector": selector_name},
        )
    rank_rows, rank_long_rows = rank_reference_heads(
        head_rows,
        reference_heads,
        selectors,
        args.min_layer,
        args.max_layer,
    )

    summary = {
        **metadata,
        "overlap": overlap_rows,
    }
    write_csv(os.path.join(args.output_dir, "overlap_summary.csv"), overlap_rows)
    write_csv(os.path.join(args.output_dir, "reference_head_ranks.csv"), rank_rows)
    write_csv(os.path.join(args.output_dir, "reference_head_ranks_long.csv"), rank_long_rows)
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

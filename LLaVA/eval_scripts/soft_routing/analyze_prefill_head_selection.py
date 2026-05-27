import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict

import numpy as np
import torch
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
    normalize_image_id,
    prepare_mentions,
)
from eval_scripts.soft_routing.head_prior_utils import default_heads_for_model, head_key, parse_head_key
from eval_scripts.soft_routing.validate_head_logit_contribution_proxy import (
    average_ranks,
    dedupe_heads,
    one_step_scores,
    parse_head_list,
    pearson,
    row_label_family,
    safe_float,
)


PREFILL_FEATURES = [
    "text_mass",
    "inverse_text_entropy_norm",
    "concentrated_text_leverage",
    "full_entropy_norm",
    "img_mass",
]


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


def mean(values):
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(sum(values) / len(values)) if values else None


def percentile(values, q):
    values = sorted(float(value) for value in values if value is not None and math.isfinite(float(value)))
    if not values:
        return None
    idx = int(round((len(values) - 1) * q / 100.0))
    return float(values[idx])


def spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return None
    return pearson(average_ranks(x), average_ranks(y))


def load_eval_sentences(path):
    with open(path) as f:
        data = json.load(f)
    return data.get("sentences", [])


def parse_top_ks(text):
    return [
        int(item)
        for item in str(text).replace(" ", ",").split(",")
        if item.strip()
    ]


def label_family(row):
    for key in ("label_name", "label_family", "base_label_name"):
        value = str(row.get(key, "")).strip()
        if value in {"hallucinated", "grounded"}:
            return value
    return row_label_family(row)


def candidate_heads_from_args(args):
    if args.candidate_heads:
        return dedupe_heads(parse_head_list(args.candidate_heads))
    heads = []
    for layer in range(int(args.layer_start), int(args.layer_end) + 1):
        for head in range(int(args.head_start), int(args.head_end) + 1):
            heads.append((layer, head))
    return dedupe_heads(heads)


def configure_image_span(model, model_path):
    if model_path == "liuhaotian/llava-v1.6-34b":
        model.config.img_start_pos = 33
        model.config.img_length = 1948
    else:
        model.config.img_start_pos = 35
        model.config.img_length = 576


def select_prefill_samples(args, tokenizer):
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

    samples = []
    seen_images = set()
    per_label = Counter()
    n_selected_mentions = 0
    for sentence in sentences:
        image_id = image_id_key(sentence)
        if args.unique_images and image_id in seen_images:
            continue

        matched = []
        label_counts = Counter()
        for mention in prepare_mentions(tokenizer, sentence, score_span="first"):
            family = row_label_family(mention)
            if args.label_filter != "all" and family != args.label_filter:
                continue
            if args.max_per_label and per_label[family] >= args.max_per_label:
                continue
            matched.append(mention)
            label_counts[family] += 1
            per_label[family] += 1

        if not matched:
            continue

        samples.append({
            "sample_index": len(samples),
            "image_id": image_id,
            "image": image_name_from_sentence(sentence, image_split=args.image_split),
            "caption": sentence.get("caption") or sentence.get("text") or "",
            "n_selected_mentions": len(matched),
            "n_hallucinated_mentions": label_counts.get("hallucinated", 0),
            "n_grounded_mentions": label_counts.get("grounded", 0),
            "matched_words": ",".join(str(item.get("word") or item.get("node_word") or "") for item in matched),
        })
        seen_images.add(image_id)
        n_selected_mentions += len(matched)

        if args.max_samples and len(samples) >= args.max_samples:
            break
        if args.max_mentions and n_selected_mentions >= args.max_mentions:
            break

    return samples


def group_heads_by_layer(heads):
    heads_by_layer = defaultdict(list)
    for layer, head in heads:
        heads_by_layer[int(layer)].append(int(head))
    return {layer: sorted(set(values)) for layer, values in heads_by_layer.items()}


def collect_prefill_rows(args, tokenizer, model, image_processor, candidate_heads, samples):
    device = next(model.parameters()).device
    heads_by_layer = group_heads_by_layer(candidate_heads)
    candidate_set = {head_key(layer, head) for layer, head in candidate_heads}
    rows = []
    sample_rows = []

    for sample in tqdm(samples, desc="prefill samples"):
        prompt_ids, image_tensor, image_size = build_prompt_inputs(
            sample["image"],
            args.image_folder,
            tokenizer,
            image_processor,
            model.config,
            args.conv_mode,
            image_split=args.image_split,
        )
        prompt_ids = prompt_ids.to(device=device, non_blocking=True)
        image_tensor = image_tensor.to(device=device, dtype=torch.float16, non_blocking=True)
        baseline = one_step_scores(
            model,
            prompt_ids,
            [],
            image_tensor,
            image_size,
            record_head_outputs=True,
            record_components=True,
            heads_by_layer=heads_by_layer,
        )
        records = list(baseline.get("diagnostics") or [])
        sample_rows.append({
            **sample,
            "n_candidate_heads": len(candidate_heads),
            "n_recorded_heads": len(records),
            "top1_token_id_from_prompt": baseline.get("top1_id"),
            "top1_logit_from_prompt": baseline.get("top1_logit"),
            "top1_logprob_from_prompt": baseline.get("top1_logprob"),
        })
        for record in records:
            key = str(record.get("head_key") or head_key(record.get("layer"), record.get("head")))
            if key not in candidate_set:
                continue
            text_mass = safe_float(record.get("text_mass"))
            img_mass = safe_float(record.get("img_mass"))
            text_entropy = safe_float(record.get("text_entropy_norm"))
            full_entropy = safe_float(record.get("full_entropy_norm"))
            inverse_text_entropy = None
            concentrated = None
            if text_entropy is not None:
                inverse_text_entropy = max(0.0, 1.0 - text_entropy)
            if text_mass is not None and inverse_text_entropy is not None:
                concentrated = text_mass * inverse_text_entropy
            layer, head = parse_head_key(key)
            rows.append({
                "sample_index": sample["sample_index"],
                "image_id": sample["image_id"],
                "image": sample["image"],
                "n_selected_mentions": sample["n_selected_mentions"],
                "n_hallucinated_mentions": sample["n_hallucinated_mentions"],
                "n_grounded_mentions": sample["n_grounded_mentions"],
                "matched_words": sample["matched_words"],
                "layer": layer,
                "head": head,
                "head_key": key,
                "text_mass": text_mass,
                "img_mass": img_mass,
                "full_entropy_norm": full_entropy,
                "text_entropy_norm": text_entropy,
                "inverse_text_entropy_norm": inverse_text_entropy,
                "concentrated_text_leverage": concentrated,
            })
    return rows, sample_rows


def sorted_head_keys(items, feature):
    return [
        row["head_key"]
        for row in sorted(
            items,
            key=lambda row: (
                -safe_float(row.get(feature), -1e30),
                row["head_key"],
            ),
        )
        if safe_float(row.get(feature)) is not None
    ]


def summarize_adhh_overlay(prefill_rows, model_path, top_ks):
    by_sample = defaultdict(list)
    for row in prefill_rows:
        by_sample[row["sample_index"]].append(row)

    adhh_heads = [head_key(layer, head) for layer, head in default_heads_for_model(model_path)]
    adhh_rank = {key: idx + 1 for idx, key in enumerate(adhh_heads)}
    summary = []
    frequency_rows = []
    for feature in PREFILL_FEATURES:
        for top_k in top_ks:
            overlaps = []
            precisions = []
            recalls = []
            jaccards = []
            selected_counts = Counter()
            selected_scores = defaultdict(list)
            ref = set(adhh_heads[: int(top_k)])
            for _, items in by_sample.items():
                ranked = sorted_head_keys(items, feature)
                selected = set(ranked[: int(top_k)])
                if not selected:
                    continue
                inter = selected & ref
                union = selected | ref
                overlaps.append(len(inter))
                precisions.append(len(inter) / max(len(selected), 1))
                recalls.append(len(inter) / max(len(ref), 1))
                jaccards.append(len(inter) / max(len(union), 1))
                values = {row["head_key"]: safe_float(row.get(feature)) for row in items}
                for key in selected:
                    selected_counts[key] += 1
                    selected_scores[key].append(values.get(key))
            summary.append({
                "feature": feature,
                "top_k": int(top_k),
                "n_samples": len(overlaps),
                "mean_adhh_overlap": mean(overlaps),
                "mean_selected_fraction_in_adhh": mean(precisions),
                "mean_adhh_recall": mean(recalls),
                "mean_adhh_jaccard": mean(jaccards),
            })
            for rank, (key, count) in enumerate(selected_counts.most_common(), start=1):
                layer, head = parse_head_key(key)
                frequency_rows.append({
                    "feature": feature,
                    "top_k": int(top_k),
                    "rank_by_selection_rate": rank,
                    "layer": layer,
                    "head": head,
                    "head_key": key,
                    "selected_count": count,
                    "selected_rate": count / max(len(by_sample), 1),
                    "mean_selected_score": mean(selected_scores[key]),
                    "in_adhh_default": int(key in adhh_rank),
                    "adhh_rank": adhh_rank.get(key),
                })
    return summary, frequency_rows


def distribution_rows(prefill_rows):
    rows = []
    for feature in PREFILL_FEATURES:
        values = [safe_float(row.get(feature)) for row in prefill_rows]
        values = [value for value in values if value is not None]
        rows.append({
            "feature": feature,
            "n": len(values),
            "mean": mean(values),
            "min": min(values) if values else None,
            "q10": percentile(values, 10),
            "q25": percentile(values, 25),
            "q50": percentile(values, 50),
            "q75": percentile(values, 75),
            "q90": percentile(values, 90),
            "q95": percentile(values, 95),
            "max": max(values) if values else None,
        })
    return rows


def load_teacher_scores(path, teacher_feature, label_filter, score_mode):
    if not path or not os.path.exists(path):
        return {}, {}
    grouped = defaultdict(lambda: defaultdict(list))
    label_counts = Counter()
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            label = label_family(row)
            if label_filter != "all" and label != label_filter:
                continue
            image_id = normalize_image_id(row.get("image_id") or row.get("image") or "")
            key = row.get("head_key") or head_key(row.get("layer"), row.get("head"))
            value = safe_float(row.get(teacher_feature))
            if not image_id or value is None:
                continue
            grouped[image_id][key].append(value)
            label_counts[label] += 1

    scores = {}
    for image_id, by_head in grouped.items():
        scores[image_id] = {}
        for key, values in by_head.items():
            if score_mode == "positive_mean":
                scores[image_id][key] = mean([max(0.0, value) for value in values])
            elif score_mode == "mean":
                scores[image_id][key] = mean(values)
            else:
                raise ValueError(f"Unknown teacher_score_mode={score_mode}")
    meta = {
        "teacher_rows": path,
        "teacher_feature": teacher_feature,
        "teacher_label_filter": label_filter,
        "teacher_score_mode": score_mode,
        "label_counts": dict(label_counts),
        "n_teacher_images": len(scores),
        "n_teacher_image_heads": sum(len(items) for items in scores.values()),
    }
    return scores, meta


def summarize_teacher_overlap(prefill_rows, teacher_scores, top_ks):
    if not teacher_scores:
        return [], []
    by_image = defaultdict(list)
    for row in prefill_rows:
        by_image[normalize_image_id(row["image_id"])].append(row)

    image_rows = []
    summary_rows = []
    for feature in PREFILL_FEATURES:
        for image_id, items in by_image.items():
            teacher = teacher_scores.get(image_id)
            if not teacher:
                continue
            prefill = {
                row["head_key"]: safe_float(row.get(feature))
                for row in items
                if safe_float(row.get(feature)) is not None
            }
            common = sorted(set(prefill) & set(teacher), key=parse_head_key)
            if len(common) < 2:
                continue
            corr = spearman([prefill[key] for key in common], [teacher[key] for key in common])
            for top_k in top_ks:
                kk = min(int(top_k), len(common))
                prefill_top = set(sorted(common, key=lambda key: (-(prefill[key]), key))[:kk])
                teacher_top = set(sorted(common, key=lambda key: (-(teacher[key]), key))[:kk])
                inter = prefill_top & teacher_top
                union = prefill_top | teacher_top
                image_rows.append({
                    "feature": feature,
                    "image_id": image_id,
                    "top_k": int(top_k),
                    "n_common_heads": len(common),
                    "spearman": corr,
                    "overlap": len(inter),
                    "precision": len(inter) / max(len(prefill_top), 1),
                    "recall": len(inter) / max(len(teacher_top), 1),
                    "jaccard": len(inter) / max(len(union), 1),
                })

    grouped = defaultdict(list)
    for row in image_rows:
        grouped[(row["feature"], int(row["top_k"]))].append(row)
    for (feature, top_k), rows in sorted(grouped.items()):
        summary_rows.append({
            "feature": feature,
            "top_k": top_k,
            "n_images": len(rows),
            "mean_spearman": mean([row["spearman"] for row in rows]),
            "mean_overlap": mean([row["overlap"] for row in rows]),
            "mean_precision": mean([row["precision"] for row in rows]),
            "mean_recall": mean([row["recall"] for row in rows]),
            "mean_jaccard": mean([row["jaccard"] for row in rows]),
            "median_common_heads": percentile([row["n_common_heads"] for row in rows], 50),
        })
    return summary_rows, image_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-results", required=True)
    parser.add_argument("--match-eval-results", default="")
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--conv-mode", default="vicuna_v1")
    parser.add_argument("--image-split", default="val2014")
    parser.add_argument("--exclude-image-ids", action="append", default=[])
    parser.add_argument("--max-sentences", type=int, default=0)
    parser.add_argument("--max-mentions", type=int, default=80)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-per-label", type=int, default=0)
    parser.add_argument("--label-filter", choices=["all", "hallucinated", "grounded"], default="hallucinated")
    parser.add_argument("--unique-images", action="store_true", default=True)
    parser.add_argument("--candidate-heads", default="")
    parser.add_argument("--layer-start", type=int, default=13)
    parser.add_argument("--layer-end", type=int, default=31)
    parser.add_argument("--head-start", type=int, default=0)
    parser.add_argument("--head-end", type=int, default=31)
    parser.add_argument("--top-ks", default="20,40")
    parser.add_argument("--teacher-head-rows", default="")
    parser.add_argument("--teacher-feature", default="proxy_text_target_logit")
    parser.add_argument("--teacher-label-filter", choices=["all", "hallucinated", "grounded"], default="hallucinated")
    parser.add_argument("--teacher-score-mode", choices=["positive_mean", "mean"], default="positive_mean")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    disable_torch_init()
    model_name = get_model_name_from_path(os.path.expanduser(args.model_path))
    tokenizer, model, image_processor, _ = load_pretrained_model(args.model_path, args.model_base, model_name)
    configure_image_span(model, args.model_path)

    candidate_heads = candidate_heads_from_args(args)
    top_ks = parse_top_ks(args.top_ks)
    samples = select_prefill_samples(args, tokenizer)
    prefill_rows, sample_rows = collect_prefill_rows(
        args,
        tokenizer,
        model,
        image_processor,
        candidate_heads,
        samples,
    )

    prefill_path = os.path.join(args.output_dir, "prefill_head_scores.csv")
    sample_path = os.path.join(args.output_dir, "prefill_samples.csv")
    distribution_path = os.path.join(args.output_dir, "prefill_head_score_distribution.csv")
    overlay_path = os.path.join(args.output_dir, "prefill_adhh_overlay_summary.csv")
    frequency_path = os.path.join(args.output_dir, "prefill_selected_head_frequency.csv")
    write_csv(prefill_path, prefill_rows)
    write_csv(sample_path, sample_rows)
    write_csv(distribution_path, distribution_rows(prefill_rows))
    overlay_rows, frequency_rows = summarize_adhh_overlay(prefill_rows, args.model_path, top_ks)
    write_csv(overlay_path, overlay_rows)
    write_csv(frequency_path, frequency_rows)

    teacher_scores, teacher_meta = load_teacher_scores(
        args.teacher_head_rows,
        args.teacher_feature,
        args.teacher_label_filter,
        args.teacher_score_mode,
    )
    teacher_summary_rows, teacher_image_rows = summarize_teacher_overlap(prefill_rows, teacher_scores, top_ks)
    teacher_summary_path = os.path.join(args.output_dir, "prefill_teacher_overlap_summary.csv")
    teacher_image_path = os.path.join(args.output_dir, "prefill_teacher_overlap_by_image.csv")
    write_csv(teacher_summary_path, teacher_summary_rows)
    write_csv(teacher_image_path, teacher_image_rows)

    summary = {
        "eval_results": args.eval_results,
        "match_eval_results": args.match_eval_results,
        "model_path": args.model_path,
        "label_filter": args.label_filter,
        "n_samples": len(samples),
        "n_prefill_rows": len(prefill_rows),
        "n_candidate_heads": len(candidate_heads),
        "candidate_layers": [args.layer_start, args.layer_end],
        "candidate_heads": [args.head_start, args.head_end],
        "top_ks": top_ks,
        "teacher_meta": teacher_meta,
        "outputs": {
            "prefill_head_scores": prefill_path,
            "prefill_samples": sample_path,
            "prefill_head_score_distribution": distribution_path,
            "prefill_adhh_overlay_summary": overlay_path,
            "prefill_selected_head_frequency": frequency_path,
            "prefill_teacher_overlap_summary": teacher_summary_path,
            "prefill_teacher_overlap_by_image": teacher_image_path,
        },
        "adhh_overlay": overlay_rows,
        "teacher_overlap": teacher_summary_rows,
    }
    with open(os.path.join(args.output_dir, "prefill_head_selection_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("[summary] prefill head selection")
    print(json.dumps({
        "n_samples": len(samples),
        "n_prefill_rows": len(prefill_rows),
        "n_candidate_heads": len(candidate_heads),
        "teacher_meta": teacher_meta,
        "top_adhh_overlay": overlay_rows[:10],
        "top_teacher_overlap": teacher_summary_rows[:10],
    }, indent=2))


if __name__ == "__main__":
    main()

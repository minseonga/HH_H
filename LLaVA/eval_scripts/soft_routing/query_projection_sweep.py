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
    kl_divergence,
    load_sentences,
    one_step,
    select_rows,
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
    return float(np.mean(values)) if values else None


def parse_float_list(text):
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def strength_tag(strength):
    return f"{strength:.2f}".replace(".", "p")


def label_family(label):
    if str(label).startswith("hallucinated_object"):
        return "hallucinated"
    if label in {"lost_grounded", "kept_grounded"}:
        return label
    return str(label)


def load_direction_rows(calibration_npz, top_k, min_auc):
    data = np.load(calibration_npz)
    layers = data["layers"].astype(int)
    heads = data["heads"].astype(int)
    directions = data["directions"].astype(np.float32)
    thresholds = data["threshold_midpoint"].astype(np.float32)
    test_auroc = data["test_auroc"].astype(np.float32)
    test_auprc = data["test_auprc"].astype(np.float32)

    candidates = []
    for idx, auc in enumerate(test_auroc.tolist()):
        if auc < min_auc:
            continue
        candidates.append((auc, idx))
    candidates.sort(reverse=True)
    if top_k > 0:
        candidates = candidates[:top_k]

    rows = []
    direction_dict = {}
    threshold_dict = {}
    for rank, (_, idx) in enumerate(candidates, start=1):
        layer = int(layers[idx])
        head = int(heads[idx])
        key = f"{layer}:{head}"
        direction_dict[key] = torch.from_numpy(directions[idx].copy())
        threshold_dict[key] = float(thresholds[idx])
        rows.append({
            "rank": rank,
            "layer": layer,
            "head": head,
            "head_key": key,
            "threshold_midpoint": float(thresholds[idx]),
            "test_auroc": float(test_auroc[idx]),
            "test_auprc": float(test_auprc[idx]),
        })
    return rows, direction_dict, threshold_dict


def set_projection_config(model, surface, directions, thresholds, strength, gate_mode, temperature, record_diagnostics=True):
    active_diagnostics = bool(record_diagnostics and strength > 0.0)
    if surface == "head_output":
        model.config.head_output_direction_project = bool(strength > 0.0)
        model.config.head_output_direction_directions = directions
        model.config.head_output_direction_thresholds = thresholds
        model.config.head_output_direction_strength = float(strength)
        model.config.head_output_direction_gate_mode = gate_mode
        model.config.head_output_direction_temperature = float(temperature)
        model.config.head_output_direction_positive_only = True
        model.config.record_head_output_projection_diagnostics = active_diagnostics
        model.config.head_output_projection_diagnostics = [] if active_diagnostics else None
        model.config.query_direction_project = False
        model.config.record_query_projection_diagnostics = False
        model.config.query_projection_diagnostics = None
    else:
        model.config.query_direction_project = bool(strength > 0.0)
        model.config.query_direction_directions = directions
        model.config.query_direction_thresholds = thresholds
        model.config.query_direction_strength = float(strength)
        model.config.query_direction_gate_mode = gate_mode
        model.config.query_direction_temperature = float(temperature)
        model.config.query_direction_positive_only = True
        model.config.record_query_projection_diagnostics = active_diagnostics
        model.config.query_projection_diagnostics = [] if active_diagnostics else None
        model.config.head_output_direction_project = False
        model.config.record_head_output_projection_diagnostics = False
        model.config.head_output_projection_diagnostics = None


def clear_projection_config(model):
    model.config.query_direction_project = False
    model.config.query_direction_directions = {}
    model.config.query_direction_thresholds = {}
    model.config.query_direction_strength = 0.0
    model.config.record_query_projection_diagnostics = False
    model.config.query_projection_diagnostics = None
    model.config.head_output_direction_project = False
    model.config.head_output_direction_directions = {}
    model.config.head_output_direction_thresholds = {}
    model.config.head_output_direction_strength = 0.0
    model.config.record_head_output_projection_diagnostics = False
    model.config.head_output_projection_diagnostics = None


def mean_metric(records, key):
    values = [float(record[key]) for record in records if key in record]
    return mean(values)


def diagnostic_summary(diagnostics):
    query_records = [record for record in diagnostics if record.get("kind") in {"query_projection", "head_output_projection"}]
    attention_records = [record for record in diagnostics if record.get("kind") == "attention_projection"]
    active_query_records = [record for record in query_records if float(record.get("active_projection", 0.0)) > 0.0]
    return {
        "projection_head_count": len(query_records),
        "active_projection_head_count": len(active_query_records),
        "mean_gate": mean_metric(query_records, "gate"),
        "mean_raw_score_before": mean_metric(query_records, "raw_score_before"),
        "mean_raw_score_after": mean_metric(query_records, "raw_score_after"),
        "mean_raw_score_delta": mean_metric(query_records, "raw_score_delta"),
        "mean_normalized_score_before": mean_metric(query_records, "normalized_score_before"),
        "mean_normalized_score_after": mean_metric(query_records, "normalized_score_after"),
        "mean_normalized_score_delta": mean_metric(query_records, "normalized_score_delta"),
        "mean_relative_q_delta": mean_metric(query_records, "relative_q_delta"),
        "max_relative_q_delta": max([float(record.get("relative_q_delta", 0.0)) for record in query_records], default=None),
        "mean_relative_head_output_delta": mean_metric(query_records, "relative_head_output_delta"),
        "max_relative_head_output_delta": max([float(record.get("relative_head_output_delta", 0.0)) for record in query_records], default=None),
        "mean_attention_logit_delta_norm": mean_metric(attention_records, "attention_logit_delta_norm"),
        "mean_relative_attention_logit_delta": mean_metric(attention_records, "relative_attention_logit_delta"),
        "mean_attention_kl": mean_metric(attention_records, "attention_kl"),
        "mean_attention_l1": mean_metric(attention_records, "attention_l1"),
    }


def flatten_diagnostics(base_row, strength, diagnostics):
    by_head = defaultdict(dict)
    for record in diagnostics:
        key = record.get("head_key", "")
        by_head[key][record.get("kind", "unknown")] = record
    rows = []
    for head_key, records in sorted(by_head.items()):
        query = records.get("query_projection", records.get("head_output_projection", {}))
        attention = records.get("attention_projection", {})
        rows.append({
            **base_row,
            "strength": strength,
            "projection_kind": query.get("kind"),
            "head_key": head_key,
            "layer": query.get("layer", attention.get("layer")),
            "head": query.get("head", attention.get("head")),
            "gate_mode": query.get("gate_mode"),
            "gate": query.get("gate"),
            "positive_only": query.get("positive_only"),
            "positive_coeff": query.get("positive_coeff"),
            "effective_coeff": query.get("effective_coeff"),
            "active_projection": query.get("active_projection"),
            "threshold": query.get("threshold"),
            "raw_score_before": query.get("raw_score_before"),
            "raw_score_after": query.get("raw_score_after"),
            "raw_score_delta": query.get("raw_score_delta"),
            "normalized_score_before": query.get("normalized_score_before"),
            "normalized_score_after": query.get("normalized_score_after"),
            "normalized_score_delta": query.get("normalized_score_delta"),
            "q_delta_norm": query.get("q_delta_norm"),
            "q_norm": query.get("q_norm"),
            "relative_q_delta": query.get("relative_q_delta"),
            "head_output_delta_norm": query.get("head_output_delta_norm"),
            "head_output_norm": query.get("head_output_norm"),
            "relative_head_output_delta": query.get("relative_head_output_delta"),
            "attention_logit_delta_norm": attention.get("attention_logit_delta_norm"),
            "relative_attention_logit_delta": attention.get("relative_attention_logit_delta"),
            "attention_kl": attention.get("attention_kl"),
            "attention_l1": attention.get("attention_l1"),
        })
    return rows


def run_projection_sweep(model, tokenizer, prompt_ids, prefix_ids, image_tensor, image_size, target_token_id, strengths, directions, thresholds, gate_mode, temperature, surface):
    outputs = {}
    for strength in strengths:
        set_projection_config(model, surface, directions, thresholds, strength, gate_mode, temperature)
        outputs[strength] = one_step(
            model,
            tokenizer,
            prompt_ids,
            prefix_ids,
            image_tensor,
            image_size,
            target_token_id,
            "none",
            record=False,
        )
        if surface == "head_output":
            diagnostics = getattr(model.config, "head_output_projection_diagnostics", [])
        else:
            diagnostics = getattr(model.config, "query_projection_diagnostics", [])
        outputs[strength]["projection_diagnostics"] = list(diagnostics or [])
    clear_projection_config(model)
    return outputs


def summarize_by_group(rows, strengths):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["label_family"]].append(row)
    output = []
    for group, items in sorted(grouped.items()):
        record = {"group": group, "n": len(items)}
        for strength in strengths:
            if abs(strength) < 1e-12:
                continue
            tag = strength_tag(strength)
            record[f"drop_s{tag}"] = mean([float(item.get(f"drop_s{tag}", 0.0)) for item in items])
            record[f"rank_delta_s{tag}"] = mean([float(item.get(f"rank_delta_s{tag}", 0.0)) for item in items])
            record[f"kl_s{tag}"] = mean([float(item.get(f"kl_original_to_s{tag}", 0.0)) for item in items])
        output.append(record)
    return output


def summarize_hall_vs_grounded(rows, strengths):
    groups = {
        "hallucinated": [row for row in rows if row["label_family"] == "hallucinated"],
        "grounded": [row for row in rows if row["label_family"] in {"kept_grounded", "lost_grounded"}],
        "kept_grounded": [row for row in rows if row["label_family"] == "kept_grounded"],
        "lost_grounded": [row for row in rows if row["label_family"] == "lost_grounded"],
    }
    output = []
    for strength in strengths:
        if abs(strength) < 1e-12:
            continue
        tag = strength_tag(strength)
        record = {"strength": strength}
        for group, items in groups.items():
            record[f"{group}_n"] = len(items)
            record[f"{group}_drop"] = mean([float(item.get(f"drop_s{tag}", 0.0)) for item in items])
            record[f"{group}_rank_delta"] = mean([float(item.get(f"rank_delta_s{tag}", 0.0)) for item in items])
        hall_drop = record.get("hallucinated_drop")
        grounded_drop = record.get("grounded_drop")
        kept_drop = record.get("kept_grounded_drop")
        lost_drop = record.get("lost_grounded_drop")
        record["hallucinated_minus_grounded_drop"] = (
            hall_drop - grounded_drop if hall_drop is not None and grounded_drop is not None else None
        )
        record["hallucinated_minus_kept_drop"] = (
            hall_drop - kept_drop if hall_drop is not None and kept_drop is not None else None
        )
        record["hallucinated_minus_lost_drop"] = (
            hall_drop - lost_drop if hall_drop is not None and lost_drop is not None else None
        )
        output.append(record)
    return output


def summarize_diagnostics_by_group(diagnostic_rows):
    metrics = [
        "gate",
        "positive_coeff",
        "effective_coeff",
        "active_projection",
        "raw_score_before",
        "raw_score_after",
        "raw_score_delta",
        "normalized_score_before",
        "normalized_score_after",
        "normalized_score_delta",
        "relative_q_delta",
        "relative_head_output_delta",
        "attention_logit_delta_norm",
        "relative_attention_logit_delta",
        "attention_kl",
        "attention_l1",
    ]
    grouped = defaultdict(list)
    for row in diagnostic_rows:
        grouped[(row["label_family"], float(row["strength"]))].append(row)
    output = []
    for (group, strength), items in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        record = {
            "label_family": group,
            "strength": strength,
            "n": len(items),
            "gate_active_rate": mean([1.0 if safe_float(item.get("gate")) > 0.0 else 0.0 for item in items]),
            "positive_coeff_rate": mean([safe_float(item.get("positive_coeff")) for item in items]),
            "active_projection_rate": mean([safe_float(item.get("active_projection")) for item in items]),
        }
        for metric in metrics:
            record[f"mean_{metric}"] = mean([safe_float(item.get(metric)) for item in items if item.get(metric) is not None])
        output.append(record)
    return output


def safe_float(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(value):
        return default
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hard-results", required=True)
    parser.add_argument("--soft-results", required=True)
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--calibration-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prior-path", default="")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--model-path", default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--conv-mode", default="vicuna_v1")
    parser.add_argument("--max-per-label", type=int, default=100)
    parser.add_argument("--hallucinated-source", type=str, default="both", choices=["soft", "hard", "both"])
    parser.add_argument("--adhh-threshold", type=float, default=0.4)
    parser.add_argument("--soft-gamma", type=float, default=0.75)
    parser.add_argument("--soft-temperature", type=float, default=0.05)
    parser.add_argument("--surface", choices=["query", "head_output"], default="query")
    parser.add_argument("--projection-strengths", default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--direction-top-k", type=int, default=10)
    parser.add_argument("--min-direction-auroc", type=float, default=0.65)
    parser.add_argument("--gate-mode", choices=["none", "positive", "threshold", "sigmoid"], default="threshold")
    parser.add_argument("--query-direction-temperature", type=float, default=0.05)
    args = parser.parse_args()

    strengths = sorted(set(min(max(value, 0.0), 1.0) for value in parse_float_list(args.projection_strengths)))
    if 0.0 not in strengths:
        strengths = [0.0] + strengths

    os.makedirs(args.output_dir, exist_ok=True)
    selected_direction_rows, directions, thresholds = load_direction_rows(
        args.calibration_npz,
        args.direction_top_k,
        args.min_direction_auroc,
    )
    if not selected_direction_rows:
        raise ValueError("No query directions passed the requested top-k/min-AUROC filter")

    disable_torch_init()
    model_name = get_model_name_from_path(os.path.expanduser(args.model_path))
    tokenizer, model, image_processor, _ = load_pretrained_model(args.model_path, args.model_base, model_name)
    heads, _, prior_source = configure_model(
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

    output_rows = []
    diagnostic_rows = []
    with open(os.path.join(args.output_dir, "query_projection_sweep_rows.jsonl"), "w") as out:
        for row in tqdm(selected, desc="projection sweep"):
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
            prefix_ids = row["probe_caption_ids"][:row["target_token_pos"]]
            target_token_id = int(row["target_token_id"])

            clear_projection_config(model)
            outputs = run_projection_sweep(
                model,
                tokenizer,
                prompt_ids,
                prefix_ids,
                image_tensor,
                image_size,
                target_token_id,
                strengths,
                directions,
                thresholds,
                args.gate_mode,
                args.query_direction_temperature,
                args.surface,
            )
            original = outputs[0.0]
            base_diagnostic_row = {
                "image_id": row["image_id"],
                "image": row["image"],
                "label": row["label"],
                "label_family": label_family(row["label"]),
                "surface": args.surface,
                "object_node": row["object_node"],
                "object_word": row["object_word"],
                "target_token": tokenizer.decode([target_token_id]),
                "target_token_id": target_token_id,
                "target_token_pos": int(row["target_token_pos"]),
            }
            output = {
                "image_id": row["image_id"],
                "image": row["image"],
                "label": row["label"],
                "label_family": label_family(row["label"]),
                "caption_source": row["caption_source"],
                "object_node": row["object_node"],
                "object_word": row["object_word"],
                "target_token": tokenizer.decode([target_token_id]),
                "target_token_id": target_token_id,
                "target_token_pos": int(row["target_token_pos"]),
                "original_target_logprob": original["target_logprob"],
                "original_target_rank": original["target_rank"],
                "original_entropy": original["entropy"],
                "original_next_token": original["next_token"],
                "num_query_directions": len(selected_direction_rows),
                "surface": args.surface,
                "gate_mode": args.gate_mode,
            }
            for strength in strengths:
                tag = strength_tag(strength)
                current = outputs[strength]
                output[f"logprob_s{tag}"] = current["target_logprob"]
                output[f"drop_s{tag}"] = original["target_logprob"] - current["target_logprob"]
                output[f"rank_s{tag}"] = current["target_rank"]
                output[f"rank_delta_s{tag}"] = current["target_rank"] - original["target_rank"]
                output[f"entropy_s{tag}"] = current["entropy"]
                output[f"kl_original_to_s{tag}"] = kl_divergence(original["score"], current["score"])
                output[f"next_token_s{tag}"] = current["next_token"]
                if abs(strength) >= 1e-12:
                    diagnostics = current.get("projection_diagnostics", [])
                    for key, value in diagnostic_summary(diagnostics).items():
                        output[f"{key}_s{tag}"] = value
                    diagnostic_rows.extend(flatten_diagnostics(base_diagnostic_row, strength, diagnostics))
            output_rows.append(output)
            out.write(json.dumps(output) + "\n")
            out.flush()

    write_csv(os.path.join(args.output_dir, "selected_query_directions.csv"), selected_direction_rows)
    write_csv(os.path.join(args.output_dir, "query_projection_sweep_rows.csv"), output_rows)
    write_csv(os.path.join(args.output_dir, "query_projection_diagnostics.csv"), diagnostic_rows)
    write_csv(os.path.join(args.output_dir, "query_projection_diagnostics_by_group.csv"), summarize_diagnostics_by_group(diagnostic_rows))
    write_csv(os.path.join(args.output_dir, "query_projection_sweep_by_group.csv"), summarize_by_group(output_rows, strengths))
    write_csv(os.path.join(args.output_dir, "query_projection_sweep_policy_summary.csv"), summarize_hall_vs_grounded(output_rows, strengths))

    summary = {
        "num_records": len(output_rows),
        "label_counts": dict(Counter(row["label"] for row in output_rows)),
        "label_family_counts": dict(Counter(row["label_family"] for row in output_rows)),
        "projection_strengths": strengths,
        "surface": args.surface,
        "direction_top_k": args.direction_top_k,
        "min_direction_auroc": args.min_direction_auroc,
        "gate_mode": args.gate_mode,
        "query_direction_temperature": args.query_direction_temperature,
        "prior_source": prior_source,
        "heads": heads,
        "selected_query_directions": selected_direction_rows,
    }
    with open(os.path.join(args.output_dir, "query_projection_sweep_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

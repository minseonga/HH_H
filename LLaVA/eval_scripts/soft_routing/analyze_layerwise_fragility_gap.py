#!/usr/bin/env python3
import argparse
import collections
import csv
import json
import os

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
    one_step,
)
from eval_scripts.soft_routing.analyze_static_object_logprob_drop import (
    load_sentences,
    select_base_object_rows,
    write_csv,
)


def mean(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return None
    return float(np.mean(np.array(values, dtype=np.float64)))


def percentile(values, q):
    values = [float(v) for v in values if v is not None]
    if not values:
        return None
    return float(np.percentile(np.array(values, dtype=np.float64), q))


def parse_layer_list(text):
    if not text:
        return list(range(32))
    layers = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            lo, hi = item.split("-", 1)
            layers.extend(range(int(lo), int(hi) + 1))
        else:
            layers.append(int(item))
    return sorted(dict.fromkeys(layers))


def layer_bands():
    return {
        "L0-8": list(range(0, 9)),
        "L9-16": list(range(9, 17)),
        "L17-24": list(range(17, 25)),
        "L25-31": list(range(25, 32)),
    }


def head_key(layer, head):
    return f"{int(layer)}:{int(head)}"


def priors_for_heads(heads, original_priors):
    out = {}
    for layer, head in heads:
        key = head_key(layer, head)
        out[key] = float(original_priors.get(key, 1.0))
    return out


def set_probe_heads(model, heads, original_priors):
    model.config.hal_attention_heads = [[int(layer), int(head)] for layer, head in heads]
    model.config.head_attribution_priors = priors_for_heads(heads, original_priors)


def heads_for_layer(layer, mode, selected_by_layer, num_heads):
    if mode == "all":
        return [[int(layer), int(head)] for head in range(int(num_heads))]
    if mode == "selected":
        return selected_by_layer.get(int(layer), [])
    raise ValueError(f"Unknown layer_head_mode={mode}")


def summarize_layer_rows(rows):
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[(int(row["layer"]), row["label"])].append(row)

    labels = ["grounded_object", "hallucinated_object"]
    by_layer = {}
    summary_rows = []
    for (layer, label), items in sorted(grouped.items()):
        drops = [float(row["target_logprob_drop"]) for row in items]
        rank_deltas = [float(row["target_rank_delta"]) for row in items]
        flips = [float(row["next_token_changed"]) for row in items]
        kls = [float(row["kl_base_to_layer_suppressed"]) for row in items]
        out = {
            "layer": layer,
            "label": label,
            "n": len(items),
            "mean_delta_logprob": mean(drops),
            "median_delta_logprob": percentile(drops, 50),
            "q75_delta_logprob": percentile(drops, 75),
            "q90_delta_logprob": percentile(drops, 90),
            "positive_drop_fraction": mean([1.0 if value > 0 else 0.0 for value in drops]),
            "mean_rank_delta": mean(rank_deltas),
            "q90_rank_delta": percentile(rank_deltas, 90),
            "next_token_flip_rate": mean(flips),
            "mean_kl_base_to_layer_suppressed": mean(kls),
        }
        summary_rows.append(out)
        by_layer.setdefault(layer, {})[label] = out

    gap_rows = []
    for layer in sorted(by_layer):
        ground = by_layer[layer].get("grounded_object")
        hall = by_layer[layer].get("hallucinated_object")
        row = {"layer": layer}
        for label, prefix in [("grounded_object", "grounded"), ("hallucinated_object", "hallucinated")]:
            src = by_layer[layer].get(label, {})
            row[f"{prefix}_n"] = src.get("n", 0)
            row[f"{prefix}_mean_delta_logprob"] = src.get("mean_delta_logprob")
            row[f"{prefix}_next_token_flip_rate"] = src.get("next_token_flip_rate")
            row[f"{prefix}_positive_drop_fraction"] = src.get("positive_drop_fraction")
        if ground and hall:
            row["fragility_gap_delta_logprob_H_minus_G"] = (
                float(hall["mean_delta_logprob"]) - float(ground["mean_delta_logprob"])
            )
            row["fragility_ratio_delta_logprob_H_over_G"] = (
                float(hall["mean_delta_logprob"]) / max(float(ground["mean_delta_logprob"]), 1e-8)
            )
            row["flip_gap_H_minus_G"] = (
                float(hall["next_token_flip_rate"]) - float(ground["next_token_flip_rate"])
            )
        else:
            row["fragility_gap_delta_logprob_H_minus_G"] = None
            row["fragility_ratio_delta_logprob_H_over_G"] = None
            row["flip_gap_H_minus_G"] = None
        gap_rows.append(row)
    return summary_rows, gap_rows


def summarize_bands(gap_rows):
    by_layer = {int(row["layer"]): row for row in gap_rows}
    out = []
    for band, layers in layer_bands().items():
        items = [by_layer[layer] for layer in layers if layer in by_layer]
        out.append(
            {
                "band": band,
                "n_layers": len(items),
                "mean_grounded_delta_logprob": mean([row.get("grounded_mean_delta_logprob") for row in items]),
                "mean_hallucinated_delta_logprob": mean([row.get("hallucinated_mean_delta_logprob") for row in items]),
                "mean_fragility_gap_delta_logprob_H_minus_G": mean(
                    [row.get("fragility_gap_delta_logprob_H_minus_G") for row in items]
                ),
                "mean_fragility_ratio_delta_logprob_H_over_G": mean(
                    [row.get("fragility_ratio_delta_logprob_H_over_G") for row in items]
                ),
                "mean_grounded_flip_rate": mean([row.get("grounded_next_token_flip_rate") for row in items]),
                "mean_hallucinated_flip_rate": mean([row.get("hallucinated_next_token_flip_rate") for row in items]),
                "mean_flip_gap_H_minus_G": mean([row.get("flip_gap_H_minus_G") for row in items]),
            }
        )
    return out


def maybe_plot(output_dir, gap_rows):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    layers = [int(row["layer"]) for row in gap_rows]
    ground = [float(row["grounded_mean_delta_logprob"]) for row in gap_rows]
    hall = [float(row["hallucinated_mean_delta_logprob"]) for row in gap_rows]
    gap = [float(row["fragility_gap_delta_logprob_H_minus_G"]) for row in gap_rows]
    flip_gap = [float(row["flip_gap_H_minus_G"]) for row in gap_rows]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7))
    ax = axes[0]
    ax.plot(layers, ground, marker="o", linewidth=1.6, color="#2e8b57", label="grounded")
    ax.plot(layers, hall, marker="o", linewidth=1.6, color="#d95f02", label="hallucinated")
    ax.plot(layers, gap, marker="o", linewidth=2.1, color="#7c3aed", label="H-G gap")
    ax.axhline(0, color="#9ca3af", linewidth=1)
    ax.axvspan(9, 16, color="#7c3aed", alpha=0.10, label="L9-L16")
    ax.set_xlabel("layer")
    ax.set_ylabel(r"$\Delta \log p(y_t)$")
    ax.set_title("Layer-wise object fragility")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    ax.plot(layers, flip_gap, marker="o", linewidth=2.0, color="#7c3aed")
    ax.axhline(0, color="#9ca3af", linewidth=1)
    ax.axvspan(9, 16, color="#7c3aed", alpha=0.10)
    ax.set_xlabel("layer")
    ax.set_ylabel("top-1 flip gap (H-G)")
    ax.set_title("Layer-wise flip fragility gap")
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Layer-wise text-side suppression fragility gap", y=1.03, fontsize=13, fontweight="bold")
    fig.tight_layout()
    paths = []
    for ext in ("png", "svg", "pdf"):
        path = os.path.join(output_dir, f"layerwise_fragility_gap.{ext}")
        fig.savefig(path, dpi=180)
        paths.append(path)
    plt.close(fig)
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-results", required=True)
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prior-path", default="")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--model-path", default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--conv-mode", default="vicuna_v1")
    parser.add_argument("--max-per-label", type=int, default=80)
    parser.add_argument("--adhh-threshold", type=float, default=0.4)
    parser.add_argument("--soft-gamma", type=float, default=0.75)
    parser.add_argument("--soft-temperature", type=float, default=0.05)
    parser.add_argument("--layers", default="0-31")
    parser.add_argument("--layer-head-mode", choices=["all", "selected"], default="all")
    args = parser.parse_args()

    disable_torch_init()
    model_name = get_model_name_from_path(os.path.expanduser(args.model_path))
    tokenizer, model, image_processor, _ = load_pretrained_model(args.model_path, args.model_base, model_name)
    selected_heads, priors, prior_source = configure_model(
        model,
        args.model_path,
        args.prior_path,
        args.top_k,
        args.adhh_threshold,
        args.soft_gamma,
        args.soft_temperature,
    )
    original_priors = dict(priors or {})
    selected_by_layer = collections.defaultdict(list)
    for layer, head in selected_heads:
        selected_by_layer[int(layer)].append([int(layer), int(head)])

    num_heads = int(getattr(model.config, "num_attention_heads", 32))
    layers = parse_layer_list(args.layers)
    sentences = load_sentences(args.base_results)
    selected_rows, counts, missed = select_base_object_rows(sentences, tokenizer, args.max_per_label)
    os.makedirs(args.output_dir, exist_ok=True)

    rows = []
    with open(os.path.join(args.output_dir, "layerwise_fragility_rows.jsonl"), "w", encoding="utf-8") as out:
        for row in tqdm(selected_rows, desc="object mentions"):
            prompt_ids, image_tensor, image_size = build_prompt_inputs(
                row, args.image_folder, tokenizer, image_processor, model.config, args.conv_mode
            )
            prompt_ids = prompt_ids.to(device="cuda", non_blocking=True)
            image_tensor = image_tensor.to(dtype=torch.float16, device="cuda", non_blocking=True)
            prefix_ids = row["probe_caption_ids"][: int(row["target_token_pos"])]
            target_token_id = int(row["target_token_id"])

            set_probe_heads(model, [], original_priors)
            base = one_step(model, tokenizer, prompt_ids, prefix_ids, image_tensor, image_size, target_token_id, "none")

            for layer in layers:
                layer_heads = heads_for_layer(layer, args.layer_head_mode, selected_by_layer, num_heads)
                if not layer_heads:
                    suppressed = base
                else:
                    set_probe_heads(model, layer_heads, original_priors)
                    suppressed = one_step(
                        model, tokenizer, prompt_ids, prefix_ids, image_tensor, image_size, target_token_id, "hard"
                    )
                output = {
                    "image_id": row["image_id"],
                    "image": row["image"],
                    "label": row["label"],
                    "layer": int(layer),
                    "n_probe_heads": len(layer_heads),
                    "layer_head_mode": args.layer_head_mode,
                    "object_node": row["object_node"],
                    "object_word": row["object_word"],
                    "generated_word": row["generated_word"],
                    "target_token": tokenizer.decode([target_token_id]),
                    "target_token_id": target_token_id,
                    "target_token_pos": int(row["target_token_pos"]),
                    "base_target_logprob": base["target_logprob"],
                    "suppressed_target_logprob": suppressed["target_logprob"],
                    "target_logprob_drop": base["target_logprob"] - suppressed["target_logprob"],
                    "base_target_rank": base["target_rank"],
                    "suppressed_target_rank": suppressed["target_rank"],
                    "target_rank_delta": suppressed["target_rank"] - base["target_rank"],
                    "base_next_token_id": base["next_token_id"],
                    "suppressed_next_token_id": suppressed["next_token_id"],
                    "base_next_token": base["next_token"],
                    "suppressed_next_token": suppressed["next_token"],
                    "next_token_changed": 1.0 if base["next_token_id"] != suppressed["next_token_id"] else 0.0,
                    "base_entropy": base["entropy"],
                    "suppressed_entropy": suppressed["entropy"],
                    "suppressed_entropy_minus_base": suppressed["entropy"] - base["entropy"],
                    "kl_base_to_layer_suppressed": (
                        0.0 if suppressed is base else kl_divergence(base["score"], suppressed["score"])
                    ),
                    "prefix_text": tokenizer.decode(prefix_ids, skip_special_tokens=True).strip(),
                    "probe_caption": row["probe_caption"],
                    "gt_words": ", ".join(row["gt_words"]),
                }
                rows.append(output)
                out.write(json.dumps(output, ensure_ascii=False) + "\n")
            out.flush()

    summary_rows, gap_rows = summarize_layer_rows(rows)
    band_rows = summarize_bands(gap_rows)
    figure_paths = maybe_plot(args.output_dir, gap_rows)

    write_csv(os.path.join(args.output_dir, "layerwise_fragility_rows.csv"), rows)
    write_csv(os.path.join(args.output_dir, "layerwise_fragility_label_summary.csv"), summary_rows)
    write_csv(os.path.join(args.output_dir, "layerwise_fragility_gap_summary.csv"), gap_rows)
    write_csv(os.path.join(args.output_dir, "layer_band_fragility_gap_summary.csv"), band_rows)

    meta = {
        "base_results": args.base_results,
        "prior_path": args.prior_path,
        "top_k": args.top_k,
        "adhh_threshold": args.adhh_threshold,
        "prior_source": prior_source,
        "layer_head_mode": args.layer_head_mode,
        "layers": layers,
        "n_selected_heads": len(selected_heads),
        "n_object_rows": len(selected_rows),
        "selected_counts": dict(counts),
        "missed_counts": dict(missed),
        "figure_paths": figure_paths,
        "band_summary": band_rows,
    }
    with open(os.path.join(args.output_dir, "layerwise_fragility_gap_summary.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

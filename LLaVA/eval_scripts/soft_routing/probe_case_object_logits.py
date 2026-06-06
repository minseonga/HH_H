#!/usr/bin/env python3
"""Probe base vs DEACT next-token logits for selected objects in one caption case."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from eval_scripts.soft_routing.analyze_object_retention_steps import (
    build_prompt_inputs,
    kl_divergence,
    node_pairs,
    token_candidates,
)
from eval_scripts.soft_routing.analyze_static_object_logprob_drop import find_all_subsequences
from eval_scripts.soft_routing.head_prior_utils import default_heads_for_model, load_head_priors


def load_sentences(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("sentences", [])


def normalize(text: str) -> str:
    return " ".join(str(text).lower().strip().split())


def parse_targets(values: list[str]) -> list[tuple[str, str]]:
    out = []
    for value in values:
        if ":" not in value:
            raise SystemExit(f"target must be LABEL:OBJECT, got {value!r}")
        label, obj = value.split(":", 1)
        label = label.strip()
        obj = obj.strip()
        if label not in {"grounded", "hallucinated"}:
            raise SystemExit(f"label must be grounded or hallucinated, got {label!r}")
        out.append((label, obj))
    return out


def find_subsequence(sequence: list[int], subsequence: list[int]) -> tuple[int, int] | None:
    if not subsequence:
        return None
    for start, end in find_all_subsequences(sequence, subsequence):
        return start, end
    return None


def find_object_span(tokenizer, caption_ids: list[int], object_word: str) -> tuple[int, int, list[int]]:
    candidates = []
    for candidate in token_candidates(tokenizer, object_word):
        span = find_subsequence(caption_ids, candidate)
        if span is not None:
            candidates.append((span[0], span[1], candidate))
        if len(candidate) > 1:
            tail = candidate[-1:]
            span = find_subsequence(caption_ids, tail)
            if span is not None:
                candidates.append((span[0], span[1], tail))
    if not candidates:
        raise SystemExit(f"could not locate object token in caption: {object_word}")
    candidates.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    return candidates[0]


def object_pairs(sentence: dict, label: str) -> list[tuple[str, str]]:
    key = "mscoco_non_hallucinated_words" if label == "grounded" else "mscoco_hallucinated_words"
    return node_pairs(sentence, key)


def find_object_word(sentence: dict, label: str, target: str) -> tuple[str, str]:
    target_norm = normalize(target)
    for word, node in object_pairs(sentence, label):
        if normalize(word) == target_norm or normalize(node) == target_norm:
            return str(word), str(node)
    raise SystemExit(f"missing {label} object={target!r} in image_id={sentence.get('image_id')}")


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def clear_modes(model) -> None:
    for name in [
        "adaptive_deactivate",
        "soft_deactivate",
        "dynamic_deactivate",
        "attribution_soft_deactivate",
        "retention_aware_deactivate",
        "fixed_strength_deactivate",
        "contrastive_dynamic_deactivate",
        "record_intervention_diagnostics",
    ]:
        if hasattr(model.config, name):
            setattr(model.config, name, False)
    model.config.intervention_diagnostics = None


def set_base_mode(model) -> None:
    clear_modes(model)


def set_deact_mode(model, args) -> None:
    clear_modes(model)
    model.config.contrastive_dynamic_deactivate = True
    model.config.contrastive_dynamic_strength = float(args.dynamic_strength)
    model.config.contrastive_dynamic_beta = float(args.dynamic_beta)
    model.config.contrastive_dynamic_tau = float(args.dynamic_tau)
    model.config.contrastive_dynamic_concentration_mode = args.concentration_mode
    model.config.contrastive_dynamic_concentration_power = float(args.concentration_power)
    model.config.contrastive_dynamic_renormalize = bool(args.renormalize)
    model.config.contrastive_dynamic_eps = 1e-6


def setup_head_pool(model, args, model_path: str) -> tuple[list[list[int]], dict[str, float], str]:
    heads, priors, prior_source = load_head_priors(
        args.prior_path,
        top_k=args.top_k,
        prior_mode="score" if args.prior_path else "rank",
        default_heads=default_heads_for_model(model_path),
    )
    model.config.hal_attention_heads = heads
    model.config.head_attribution_priors = priors
    model.config.head_attribution_prior_source = prior_source
    if model_path == "liuhaotian/llava-v1.6-34b":
        model.config.img_start_pos = 33
        model.config.img_length = 1948
    else:
        model.config.img_start_pos = args.img_start_pos
        model.config.img_length = args.img_length
    return heads, priors, prior_source


def topk_summary(tokenizer, score: torch.Tensor, k: int) -> dict[str, str]:
    log_probs = F.log_softmax(score, dim=-1)
    probs = log_probs.exp()
    vals, ids = torch.topk(score, k=min(k, score.numel()))
    out = {}
    for rank, (token_id, logit) in enumerate(zip(ids.tolist(), vals.tolist()), start=1):
        out[f"top{rank}_id"] = int(token_id)
        out[f"top{rank}_token"] = tokenizer.decode([int(token_id)]).replace("\n", "\\n")
        out[f"top{rank}_logit"] = float(logit)
        out[f"top{rank}_logprob"] = float(log_probs[int(token_id)].item())
        out[f"top{rank}_prob"] = float(probs[int(token_id)].item())
    return out


def one_step_probe(model, tokenizer, prompt_ids, prefix_ids, image_tensor, image_size, target_token_id: int, mode: str, args) -> dict:
    prefix_tensor = torch.tensor([prefix_ids], dtype=torch.long, device=prompt_ids.device)
    step_input = torch.cat([prompt_ids, prefix_tensor], dim=1) if prefix_ids else prompt_ids
    if mode == "base":
        set_base_mode(model)
    elif mode == "deact":
        set_deact_mode(model, args)
    else:
        raise ValueError(mode)
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
    sorted_ids = torch.argsort(score, descending=True)
    target_rank = int((sorted_ids == int(target_token_id)).nonzero(as_tuple=False)[0].item() + 1)
    next_token_id = int(torch.argmax(score).item())
    return {
        "score": score,
        "target_logit": float(score[int(target_token_id)].item()),
        "target_logprob": float(log_probs[int(target_token_id)].item()),
        "target_rank": target_rank,
        "next_token_id": next_token_id,
        "next_token": tokenizer.decode([next_token_id]).replace("\n", "\\n"),
        **topk_summary(tokenizer, score, args.top_k_tokens),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-results", required=True)
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--image-id", default="140231")
    parser.add_argument("--target", action="append", default=None)
    parser.add_argument("--prior-path", required=True)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--model-path", default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--conv-mode", default="llava_v1")
    parser.add_argument("--img-start-pos", type=int, default=35)
    parser.add_argument("--img-length", type=int, default=576)
    parser.add_argument("--dynamic-strength", type=float, default=1.0)
    parser.add_argument("--dynamic-beta", type=float, default=8.0)
    parser.add_argument("--dynamic-tau", type=float, default=0.9)
    parser.add_argument("--concentration-mode", default="none")
    parser.add_argument("--concentration-power", type=float, default=1.0)
    parser.add_argument("--renormalize", action="store_true")
    parser.add_argument("--top-k-tokens", type=int, default=5)
    parser.add_argument("--output-dir", default="LLaVA/results/coco/qualitative_case_studies/logit_probe_140231")
    args = parser.parse_args()
    if args.target is None:
        args.target = ["hallucinated:person", "grounded:keyboard"]

    disable_torch_init()
    model_name = get_model_name_from_path(os.path.expanduser(args.model_path))
    tokenizer, model, image_processor, _ = load_pretrained_model(args.model_path, args.model_base, model_name)
    heads, priors, prior_source = setup_head_pool(model, args, args.model_path)

    sentences = load_sentences(args.base_results)
    sentence = next((item for item in sentences if str(item.get("image_id")) == str(args.image_id)), None)
    if sentence is None:
        raise SystemExit(f"missing image_id={args.image_id} in {args.base_results}")

    caption_ids = tokenizer(sentence["caption"], add_special_tokens=False)["input_ids"]
    rows = []
    for label, target in parse_targets(args.target):
        object_word, object_node = find_object_word(sentence, label, target)
        start, end, matched_ids = find_object_span(tokenizer, caption_ids, object_word)
        target_token_id = int(caption_ids[start])
        row = {
            "image_id": str(sentence["image_id"]),
            "image": sentence.get("image", ""),
            "label": label,
            "object_word": object_word,
            "object_node": object_node,
            "target_token": tokenizer.decode([target_token_id]).replace("\n", "\\n"),
            "target_token_id": target_token_id,
            "target_token_pos": int(start),
            "matched_token_ids": " ".join(str(x) for x in matched_ids),
            "probe_caption_ids": caption_ids,
            "probe_caption": sentence["caption"],
            "gt_words": sorted(str(item) for item in sentence.get("mscoco_gt_words", [])),
        }
        prompt_ids, image_tensor, image_size = build_prompt_inputs(
            row, args.image_folder, tokenizer, image_processor, model.config, args.conv_mode
        )
        prompt_ids = prompt_ids.to(device="cuda", non_blocking=True)
        image_tensor = image_tensor.to(dtype=torch.float16, device="cuda", non_blocking=True)
        prefix_ids = caption_ids[:start]

        base = one_step_probe(model, tokenizer, prompt_ids, prefix_ids, image_tensor, image_size, target_token_id, "base", args)
        deact = one_step_probe(model, tokenizer, prompt_ids, prefix_ids, image_tensor, image_size, target_token_id, "deact", args)

        output = {
            "image_id": row["image_id"],
            "image": row["image"],
            "label": label,
            "object_word": object_word,
            "object_node": object_node,
            "target_token": row["target_token"],
            "target_token_id": target_token_id,
            "target_token_pos": int(start),
            "prefix_text": tokenizer.decode(prefix_ids, skip_special_tokens=True).strip(),
            "base_target_logit": base["target_logit"],
            "deact_target_logit": deact["target_logit"],
            "target_logit_drop": base["target_logit"] - deact["target_logit"],
            "base_target_logprob": base["target_logprob"],
            "deact_target_logprob": deact["target_logprob"],
            "target_logprob_drop": base["target_logprob"] - deact["target_logprob"],
            "base_target_rank": base["target_rank"],
            "deact_target_rank": deact["target_rank"],
            "target_rank_delta": deact["target_rank"] - base["target_rank"],
            "base_top1_token": base["next_token"],
            "deact_top1_token": deact["next_token"],
            "base_top1_id": base["next_token_id"],
            "deact_top1_id": deact["next_token_id"],
            "top1_changed": int(base["next_token_id"] != deact["next_token_id"]),
            "kl_base_to_deact": kl_divergence(base["score"], deact["score"]),
            "prior_source": prior_source,
            "n_heads": len(heads),
        }
        for prefix, result in [("base", base), ("deact", deact)]:
            for key, value in result.items():
                if key.startswith("top"):
                    output[f"{prefix}_{key}"] = value
        rows.append(output)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "case_object_logit_probe_rows.csv", rows)
    with (out_dir / "case_object_logit_probe_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "base_results": args.base_results,
                "image_id": args.image_id,
                "prior_path": args.prior_path,
                "top_k": args.top_k,
                "dynamic_strength": args.dynamic_strength,
                "dynamic_beta": args.dynamic_beta,
                "dynamic_tau": args.dynamic_tau,
                "renormalize": args.renormalize,
                "rows": rows,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"[done] {out_dir / 'case_object_logit_probe_rows.csv'}")
    for row in rows:
        print(
            f"{row['label']} {row['object_word']}: "
            f"base_top1={row['base_top1_token']!r}, deact_top1={row['deact_top1_token']!r}, "
            f"logprob_drop={row['target_logprob_drop']:.4f}, "
            f"rank {row['base_target_rank']}->{row['deact_target_rank']}"
        )


if __name__ == "__main__":
    main()

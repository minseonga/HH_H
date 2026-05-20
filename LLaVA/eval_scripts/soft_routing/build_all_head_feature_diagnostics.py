import argparse
import json
import os
from collections import Counter

import torch
from tqdm import tqdm

from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from eval_scripts.soft_routing.analyze_object_retention_steps import (
    build_prompt_inputs,
    configure_model,
    load_sentences,
    select_rows,
)
from eval_scripts.soft_routing.build_online_causal_head_teacher import (
    label_family,
    normalize_records,
    one_step_all_head_diagnostics,
)


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
    parser.add_argument("--max-per-label", type=int, default=10)
    parser.add_argument("--hallucinated-source", choices=["soft", "hard", "both"], default="both")
    parser.add_argument("--adhh-threshold", type=float, default=0.4)
    parser.add_argument("--soft-gamma", type=float, default=0.75)
    parser.add_argument("--soft-temperature", type=float, default=0.05)
    parser.add_argument("--resume", action="store_true", default=False)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    jsonl_path = os.path.join(args.output_dir, "all_head_feature_diagnostics.jsonl")
    done = set()
    if args.resume and os.path.exists(jsonl_path):
        with open(jsonl_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                done.add((row.get("step_id"), row.get("head_key")))

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
    selected_steps = select_rows(
        hard_by_id,
        soft_by_id,
        tokenizer,
        args.max_per_label,
        args.hallucinated_source,
    )

    mode = "a" if args.resume else "w"
    step_rows = []
    with open(jsonl_path, mode) as out:
        for step in tqdm(selected_steps):
            prompt_ids, image_tensor, image_size = build_prompt_inputs(
                step,
                args.image_folder,
                tokenizer,
                image_processor,
                model.config,
                args.conv_mode,
            )
            prompt_ids = prompt_ids.to(device="cuda", non_blocking=True)
            image_tensor = image_tensor.to(dtype=torch.float16, device="cuda", non_blocking=True)
            prefix_ids = step["probe_caption_ids"][:step["target_token_pos"]]
            target_token_id = int(step["target_token_id"])
            step_id = (
                f'{step["image_id"]}:{step["caption_source"]}:{step["label"]}:'
                f'{step["object_node"]}:{step["target_token_pos"]}'
            )

            original = one_step_all_head_diagnostics(
                model,
                tokenizer,
                prompt_ids,
                prefix_ids,
                image_tensor,
                image_size,
                target_token_id,
            )
            diagnostics = normalize_records(list(original["diagnostics"]))
            step_rows.append({
                "step_id": step_id,
                "image_id": step["image_id"],
                "image": step["image"],
                "label": step["label"],
                "label_family": label_family(step["label"]),
                "caption_source": step["caption_source"],
                "object_node": step["object_node"],
                "object_word": step["object_word"],
                "target_token": tokenizer.decode([target_token_id]),
                "target_token_id": target_token_id,
                "target_token_pos": int(step["target_token_pos"]),
                "target_logprob_original": original["target_logprob"],
                "target_rank_original": original["target_rank"],
                "diagnostic_count": len(diagnostics),
            })
            for record in diagnostics:
                key = record["head_key"]
                if (step_id, key) in done:
                    continue
                output = {
                    "step_id": step_id,
                    "image_id": step["image_id"],
                    "image": step["image"],
                    "label": step["label"],
                    "label_family": label_family(step["label"]),
                    "caption_source": step["caption_source"],
                    "object_node": step["object_node"],
                    "object_word": step["object_word"],
                    "target_token": tokenizer.decode([target_token_id]),
                    "target_token_id": target_token_id,
                    "target_token_pos": int(step["target_token_pos"]),
                    "target_logprob_original": original["target_logprob"],
                    "target_rank_original": original["target_rank"],
                    "prior_source": prior_source,
                    **record,
                }
                out.write(json.dumps(output) + "\n")
                out.flush()

    summary = {
        "num_steps": len(selected_steps),
        "max_per_label": args.max_per_label,
        "hallucinated_source": args.hallucinated_source,
        "prior_source": prior_source,
        "heads": heads,
        "label_counts": dict(Counter(row["label"] for row in step_rows)),
        "jsonl_path": jsonl_path,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

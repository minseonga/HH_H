import argparse
import json
import math
import os
import re
from collections import OrderedDict

import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


FUNCTION_WORDS = {
    "a", "an", "the", "and", "or", "but", "of", "in", "on", "at", "to", "from", "with",
    "without", "for", "by", "as", "is", "are", "was", "were", "be", "being", "been",
    "there", "this", "that", "these", "those", "it", "its", "his", "her", "their",
    "while", "where", "which", "who", "near", "next", "behind", "front", "over",
    "under", "around", "into", "up", "down", "also", "possibly", "likely",
}


def load_case_rows(paths, max_per_case):
    rows = OrderedDict()
    for path in paths:
        case_name = os.path.splitext(os.path.basename(path))[0]
        count = 0
        with open(path, "r") as f:
            for line in f:
                if max_per_case is not None and count >= max_per_case:
                    break
                row = json.loads(line)
                image_id = str(row["image_id"])
                if image_id not in rows:
                    rows[image_id] = {
                        "image_id": image_id,
                        "case_file": case_name,
                        "winner": row.get("winner", case_name),
                        "reason": row.get("reason", ""),
                    }
                    count += 1
    return list(rows.values())


def load_object_vocab(path):
    vocab = set()
    if not path or not os.path.exists(path):
        return vocab
    with open(path, "r") as f:
        for line in f:
            for item in line.strip().split(","):
                item = item.strip().lower()
                if item:
                    vocab.add(item)
    return vocab


def configure_heads(model, model_path):
    if model_path == "liuhaotian/llava-v1.5-7b":
        model.config.hal_attention_heads = [[16, 29], [26, 9], [13, 31], [15, 10], [20, 12], [30, 9], [19, 18], [17, 0], [18, 9], [26, 28],
                                            [19, 27], [18, 26], [15, 25], [14, 16], [31, 26], [15, 24], [31, 3], [22, 20], [27, 29], [17, 28]]
        model.config.img_start_pos = 35
        model.config.img_length = 576
    elif model_path == "liuhaotian/llava-v1.5-13b":
        model.config.hal_attention_heads = [[0, 8], [29, 27], [23, 18], [20, 11], [36, 26], [19, 37], [22, 16], [22, 34], [21, 31], [20, 34],
                                            [37, 11], [17, 25], [35, 10], [17, 5], [15, 26], [0, 22], [19, 5], [19, 0], [14, 1], [23, 20],
                                            [21, 6], [30, 24], [26, 27], [21, 32], [15, 28], [15, 31], [19, 30], [20, 8], [19, 14], [14, 9],
                                            [39, 26], [25, 1], [18, 32], [17, 27], [39, 32]]
        model.config.img_start_pos = 35
        model.config.img_length = 576
    elif model_path == "liuhaotian/llava-v1.6-34b":
        model.config.hal_attention_heads = [[45, 34], [43, 4], [43, 48], [44, 29], [35, 47], [40, 27], [54, 34], [37, 48], [43, 2], [41, 34]]
        model.config.img_start_pos = 33
        model.config.img_length = 1948
    else:
        raise ValueError(f"No built-in hallucination head set for model_path={model_path}")


def clear_intervention_flags(model):
    for name in [
        "adaptive_deactivate",
        "soft_deactivate",
        "dynamic_deactivate",
        "record_intervention_diagnostics",
    ]:
        if hasattr(model.config, name):
            setattr(model.config, name, False)


def set_mode(model, mode, args, record=False):
    clear_intervention_flags(model)
    model.config.adhh_threshold = args.adhh_threshold
    model.config.soft_gamma = args.soft_gamma
    model.config.soft_temperature = args.soft_temperature
    if mode == "hard":
        model.config.adaptive_deactivate = True
    elif mode == "soft":
        model.config.soft_deactivate = True
    elif mode == "none":
        pass
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    model.config.record_intervention_diagnostics = bool(record)
    model.config.intervention_diagnostics = [] if record else None


def build_inputs(coco, image_id, image_folder, tokenizer, image_processor, model_config, conv_mode):
    image_info = coco.loadImgs(int(image_id))[0]
    image_file = image_info["file_name"]
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
    return input_ids, image_tensor, image.size, image_file


def generate_sequence(model, tokenizer, input_ids, image_tensor, image_sizes, mode, args):
    set_mode(model, mode, args, record=False)
    with torch.inference_mode():
        output = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image_sizes],
            do_sample=False,
            temperature=0,
            top_p=None,
            num_beams=1,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            output_scores=True,
            return_dict_in_generate=True,
        )
    token_ids = output["sequences"][0].detach().cpu().tolist()
    text = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
    return token_ids, text


def first_divergence(a, b):
    for idx, (a_token, b_token) in enumerate(zip(a, b)):
        if a_token != b_token:
            return idx
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def topk(score, tokenizer, k=10):
    vals, ids = torch.topk(score, k)
    return [
        {
            "token_id": int(token_id),
            "token": tokenizer.decode([int(token_id)]),
            "logit": float(val),
        }
        for val, token_id in zip(vals.detach().cpu(), ids.detach().cpu())
    ]


def distribution_stats(score):
    score = score.float()
    probs = F.softmax(score, dim=-1)
    log_probs = F.log_softmax(score, dim=-1)
    entropy = -(probs * log_probs).sum().item()
    vals, _ = torch.topk(score, 2)
    return {
        "entropy": float(entropy),
        "top1_top2_margin": float((vals[0] - vals[1]).item()),
    }


def kl_divergence(p_score, q_score):
    p_log = F.log_softmax(p_score.float(), dim=-1)
    q_log = F.log_softmax(q_score.float(), dim=-1)
    p = p_log.exp()
    return float((p * (p_log - q_log)).sum().item())


def one_step(model, tokenizer, prompt_ids, prefix_tokens, image_tensor, image_sizes, mode, args, record=False):
    if prefix_tokens:
        prefix_tensor = torch.tensor(prefix_tokens, device=prompt_ids.device, dtype=prompt_ids.dtype).unsqueeze(0)
        step_input = torch.cat([prompt_ids, prefix_tensor], dim=1)
    else:
        step_input = prompt_ids

    set_mode(model, mode, args, record=record)
    with torch.inference_mode():
        output = model.generate(
            step_input,
            images=image_tensor,
            image_sizes=[image_sizes],
            do_sample=False,
            temperature=0,
            top_p=None,
            num_beams=1,
            max_new_tokens=1,
            use_cache=True,
            output_scores=True,
            return_dict_in_generate=True,
        )
    score = output["scores"][0][0].detach()
    token_id = int(torch.argmax(score).item())
    return {
        "token_id": token_id,
        "token": tokenizer.decode([token_id]),
        "top10": topk(score, tokenizer, k=10),
        "score": score,
        "stats": distribution_stats(score),
        "diagnostics": list(getattr(model.config, "intervention_diagnostics", []) or []),
    }


def token_type(token, object_vocab):
    cleaned = re.sub(r"[^A-Za-z0-9 ]+", "", token).strip().lower()
    if not cleaned:
        return "punct_or_space"
    if cleaned in FUNCTION_WORDS:
        return "function"
    if cleaned in object_vocab:
        return "object_vocab"
    if re.match(r"^[A-Za-z]+$", cleaned):
        return "content"
    return "other"


def aggregate_diagnostics(records):
    if not records:
        return {}
    text = [r["text_mass"] for r in records]
    img = [r["img_mass"] for r in records]
    margin = [r["margin"] for r in records]
    ratio = [r["text_img_log_ratio"] for r in records]
    alpha = [r["soft_alpha"] for r in records]
    triggers = [1.0 if r["hard_trigger"] else 0.0 for r in records]
    return {
        "diag_num_heads": len(records),
        "diag_mean_text_mass": sum(text) / len(text),
        "diag_max_text_mass": max(text),
        "diag_mean_img_mass": sum(img) / len(img),
        "diag_min_img_mass": min(img),
        "diag_mean_margin": sum(margin) / len(margin),
        "diag_max_margin": max(margin),
        "diag_mean_text_img_log_ratio": sum(ratio) / len(ratio),
        "diag_max_text_img_log_ratio": max(ratio),
        "diag_triggered_head_count": int(sum(triggers)),
        "diag_triggered_head_frac": sum(triggers) / len(triggers),
        "diag_mean_soft_alpha": sum(alpha) / len(alpha),
        "diag_max_soft_alpha": max(alpha),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-files", nargs="+", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--model-path", default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--caption-file-path", required=True)
    parser.add_argument("--conv-mode", default="vicuna_v1")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-per-case", type=int, default=30)
    parser.add_argument("--adhh-threshold", type=float, default=0.4)
    parser.add_argument("--soft-gamma", type=float, default=0.75)
    parser.add_argument("--soft-temperature", type=float, default=0.05)
    parser.add_argument("--object-vocab", default="eval_scripts/eval_utils/data/synonyms.txt")
    args = parser.parse_args()

    rows = load_case_rows(args.case_files, args.max_per_case)
    object_vocab = load_object_vocab(args.object_vocab)

    disable_torch_init()
    model_name = get_model_name_from_path(os.path.expanduser(args.model_path))
    tokenizer, model, image_processor, _ = load_pretrained_model(args.model_path, args.model_base, model_name)
    configure_heads(model, args.model_path)
    coco = COCO(args.caption_file_path)

    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)
    with open(args.output_jsonl, "w") as out:
        for row in tqdm(rows):
            image_id = row["image_id"]
            input_ids, image_tensor, image_size, image_file = build_inputs(
                coco, image_id, args.image_folder, tokenizer, image_processor, model.config, args.conv_mode
            )
            input_ids = input_ids.to(device="cuda", non_blocking=True).unsqueeze(0)
            image_tensor = image_tensor.to(dtype=torch.float16, device="cuda", non_blocking=True).unsqueeze(0)

            hard_tokens, hard_text = generate_sequence(model, tokenizer, input_ids, image_tensor, image_size, "hard", args)
            soft_tokens, soft_text = generate_sequence(model, tokenizer, input_ids, image_tensor, image_size, "soft", args)
            divergence_step = first_divergence(hard_tokens, soft_tokens)

            record = {
                **row,
                "image": image_file,
                "hard_caption": hard_text,
                "soft_caption": soft_text,
                "divergence_step": divergence_step,
                "has_divergence": divergence_step is not None,
            }
            if divergence_step is not None:
                prefix_tokens = hard_tokens[:divergence_step]
                none_step = one_step(model, tokenizer, input_ids, prefix_tokens, image_tensor, image_size, "none", args, record=True)
                hard_step = one_step(model, tokenizer, input_ids, prefix_tokens, image_tensor, image_size, "hard", args, record=False)
                soft_step = one_step(model, tokenizer, input_ids, prefix_tokens, image_tensor, image_size, "soft", args, record=False)

                record.update({
                    "prefix_text": tokenizer.decode(prefix_tokens, skip_special_tokens=True).strip(),
                    "hard_next_token_id": hard_step["token_id"],
                    "hard_next_token": hard_step["token"],
                    "soft_next_token_id": soft_step["token_id"],
                    "soft_next_token": soft_step["token"],
                    "original_next_token_id": none_step["token_id"],
                    "original_next_token": none_step["token"],
                    "hard_next_token_type": token_type(hard_step["token"], object_vocab),
                    "soft_next_token_type": token_type(soft_step["token"], object_vocab),
                    "original_next_token_type": token_type(none_step["token"], object_vocab),
                    "original_top10": none_step["top10"],
                    "hard_top10": hard_step["top10"],
                    "soft_top10": soft_step["top10"],
                    "original_entropy": none_step["stats"]["entropy"],
                    "hard_entropy": hard_step["stats"]["entropy"],
                    "soft_entropy": soft_step["stats"]["entropy"],
                    "original_top1_top2_margin": none_step["stats"]["top1_top2_margin"],
                    "hard_top1_top2_margin": hard_step["stats"]["top1_top2_margin"],
                    "soft_top1_top2_margin": soft_step["stats"]["top1_top2_margin"],
                    "kl_original_to_hard": kl_divergence(none_step["score"], hard_step["score"]),
                    "kl_original_to_soft": kl_divergence(none_step["score"], soft_step["score"]),
                    "kl_hard_to_soft": kl_divergence(hard_step["score"], soft_step["score"]),
                    "diagnostics": none_step["diagnostics"],
                })
                record.update(aggregate_diagnostics(none_step["diagnostics"]))
            out.write(json.dumps(record) + "\n")
            out.flush()


if __name__ == "__main__":
    main()

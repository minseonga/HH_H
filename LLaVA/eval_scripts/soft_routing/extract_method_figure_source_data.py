#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import random
from collections import defaultdict

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm

from eval_scripts.eval_utils.chair import CHAIR
from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


EPS = 1e-12
BUCKETS = ("all", "object", "hallucinated", "non_hallucinated")
METRICS = ("system_mass", "image_mass", "text_mass", "generated_text_mass", "raw_toi", "bounded_ratio")


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields or ["empty"])
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)


def finite(value, default=0.0):
    try:
        value = float(value)
    except Exception:
        return default
    return value if math.isfinite(value) else default


def safe_token_label(tokenizer, token_id):
    if token_id == IMAGE_TOKEN_INDEX:
        return "<image>"
    if int(token_id) < 0:
        return f"<special:{int(token_id)}>"
    try:
        return tokenizer.decode([int(token_id)], skip_special_tokens=False)
    except Exception:
        return f"<id:{int(token_id)}>"


def extract_generated_ids(output_ids, input_ids):
    prompt_len = int(input_ids.shape[1])
    if int(output_ids.shape[1]) >= prompt_len and torch.equal(output_ids[:, :prompt_len], input_ids):
        return output_ids[:, prompt_len:]
    return output_ids


def get_special_token_ids(tokenizer):
    special_ids = set()
    for token_id in getattr(tokenizer, "all_special_ids", []) or []:
        if token_id is not None:
            special_ids.add(int(token_id))
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        token_id = getattr(tokenizer, attr, None)
        if token_id is not None:
            special_ids.add(int(token_id))
    return special_ids


def parse_int_ranges(spec):
    values = []
    if not spec:
        return values
    for item in str(spec).split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start, end = int(start_text), int(end_text)
            step = 1 if end >= start else -1
            values.extend(range(start, end + step, step))
        else:
            values.append(int(item))
    return sorted(set(values))


def infer_step_layout(input_ids, generated_prefix_ids, att_seq_len, special_token_ids):
    prompt_ids = input_ids[0].detach().cpu().tolist()
    generated_prefix_ids = [int(token_id) for token_id in generated_prefix_ids]
    image_positions = [idx for idx, token_id in enumerate(prompt_ids) if token_id == IMAGE_TOKEN_INDEX]
    if len(image_positions) != 1:
        raise ValueError(f"Expected exactly one image token, found {len(image_positions)}")

    image_start = image_positions[0]
    visible_len = len(prompt_ids) + len(generated_prefix_ids)
    image_len = int(att_seq_len) - (visible_len - 1)
    if image_len <= 0:
        raise ValueError(
            f"Invalid inferred image span: image_len={image_len}, visible_len={visible_len}, "
            f"att_seq_len={att_seq_len}"
        )
    image_end = image_start + image_len
    if image_end > att_seq_len:
        raise ValueError(f"Invalid image span: image_end={image_end}, att_seq_len={att_seq_len}")

    prompt_after_image_len = len(prompt_ids) - image_start - 1
    generated_start = image_end + prompt_after_image_len
    available_generated_len = max(0, min(len(generated_prefix_ids), att_seq_len - generated_start))
    generated_positions = []
    generated_special_positions = []
    for offset, token_id in enumerate(generated_prefix_ids[:available_generated_len]):
        pos = generated_start + offset
        if token_id in special_token_ids:
            generated_special_positions.append(pos)
        else:
            generated_positions.append(pos)

    return {
        "prompt_visible_len": int(len(prompt_ids)),
        "att_seq_len": int(att_seq_len),
        "image_start": int(image_start),
        "image_end": int(image_end),
        "image_len": int(image_len),
        "generated_start": int(generated_start),
        "generated_positions": generated_positions,
        "generated_special_positions": generated_special_positions,
    }


def classify_generation_steps(tokenizer, generated_ids_only, chair_evaluator, gt_objects):
    labels = []
    previous_object_count = 0
    token_ids = generated_ids_only[0].detach().cpu().tolist()

    for step_idx, token_id in enumerate(token_ids):
        prefix_text = tokenizer.decode(token_ids[: step_idx + 1], skip_special_tokens=True)
        objects = []
        hallucinated = []
        grounded = []
        if chair_evaluator is not None:
            try:
                words, node_words, _, _ = chair_evaluator.caption_to_words(prefix_text)
                for word, node_word in list(zip(words, node_words))[previous_object_count:]:
                    item = {"word": word, "node_word": node_word}
                    objects.append(item)
                    if node_word in gt_objects:
                        grounded.append(item)
                    else:
                        hallucinated.append(item)
                previous_object_count = len(node_words)
            except Exception:
                pass
        labels.append(
            {
                "step_idx": int(step_idx),
                "token_id": int(token_id),
                "token_text": safe_token_label(tokenizer, token_id),
                "is_object": bool(objects),
                "is_hallucinated": bool(hallucinated),
                "is_non_hallucinated": bool(grounded),
                "objects": objects,
                "hallucinated_objects": hallucinated,
                "non_hallucinated_objects": grounded,
            }
        )
    return labels


class MatrixStats:
    def __init__(self):
        self.count = None
        self.sum = {}
        self.sumsq = {}

    def ensure(self, values):
        if self.count is not None:
            return
        shape = tuple(values.shape)
        self.count = torch.zeros(shape, dtype=torch.float64)
        for metric in METRICS:
            self.sum[metric] = torch.zeros(shape, dtype=torch.float64)
            self.sumsq[metric] = torch.zeros(shape, dtype=torch.float64)

    def update(self, values_by_metric):
        base = values_by_metric["text_mass"].detach().cpu().double()
        self.ensure(base)
        self.count += 1.0
        for metric in METRICS:
            value = values_by_metric[metric].detach().cpu().double()
            self.sum[metric] += value
            self.sumsq[metric] += value * value

    def mean_tensor(self, metric):
        if self.count is None:
            return None
        return self.sum[metric] / torch.clamp(self.count, min=1.0)

    def count_tensor(self):
        return self.count


def rank_percentiles(values, reverse=True):
    indexed = list(enumerate(values))
    indexed.sort(key=lambda item: item[1], reverse=reverse)
    n = len(indexed)
    out = np.zeros(n, dtype=np.float64)
    denom = max(n - 1, 1)
    for rank, (idx, _) in enumerate(indexed):
        out[idx] = 1.0 - rank / denom
    return out


def build_prompt_inputs(tokenizer, image_processor, model_config, image_folder, image_file, prompt_text, conv_mode):
    qs = prompt_text
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
    return input_ids.unsqueeze(0), image_tensor.unsqueeze(0), [image.size], prompt


def load_coco_questions(args):
    coco = COCO(args.caption_file_path)
    img_ids = coco.getImgIds()
    random.seed(args.seed)
    sample_n = len(img_ids) if args.num_samples <= 0 else min(int(args.num_samples), len(img_ids))
    sampled = random.sample(img_ids, sample_n)
    questions = []
    for image_id in sampled:
        image_file = coco.loadImgs(image_id)[0]["file_name"]
        questions.append({"question_id": int(image_id), "image": image_file, "text": args.prompt})
    return questions


def step_region_matrices(step_attentions, input_ids, generated_prefix_ids, special_token_ids):
    layers = []
    layout = None
    for layer_idx, attn in enumerate(step_attentions):
        att_cpu = attn.detach().float().cpu()
        if att_cpu.shape[0] != 1:
            raise ValueError(f"Expected batch size 1, got {att_cpu.shape[0]}")
        q_idx = int(att_cpu.shape[-2] - 1)
        rows = att_cpu[0, :, q_idx, :]
        row_sums = rows.sum(dim=-1, keepdim=True)
        if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3):
            rows = torch.softmax(rows, dim=-1)

        if layout is None:
            layout = infer_step_layout(
                input_ids,
                generated_prefix_ids,
                int(rows.shape[-1]),
                special_token_ids,
            )
        image_start = layout["image_start"]
        image_end = layout["image_end"]
        generated_positions = layout["generated_positions"]

        system = rows[:, :image_start].sum(dim=-1)
        image = rows[:, image_start:image_end].sum(dim=-1)
        text = rows[:, image_end:].sum(dim=-1)
        if generated_positions:
            generated = rows[:, generated_positions].sum(dim=-1)
        else:
            generated = torch.zeros_like(text)
        raw_toi = text / torch.clamp(image, min=EPS)
        bounded = text / torch.clamp(text + image, min=EPS)
        layers.append(
            {
                "system_mass": system,
                "image_mass": image,
                "text_mass": text,
                "generated_text_mass": generated,
                "raw_toi": raw_toi,
                "bounded_ratio": bounded,
            }
        )

    stacked = {
        metric: torch.stack([layer_values[metric] for layer_values in layers], dim=0)
        for metric in METRICS
    }
    return stacked, layout


def save_matplotlib(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    root, _ = os.path.splitext(path)
    fig.savefig(root + ".pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(root + ".svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_sources(output_dir, head_rows, ratio_rows, redistribution_summary, gate_rows, top_k, highlight_layers):
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#e2e8f0",
            "grid.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    ranked = sorted(head_rows, key=lambda row: float(row["text_mass_all"]), reverse=True)
    fig, ax = plt.subplots(figsize=(3.2, 1.85))
    values = [float(row["text_mass_all"]) for row in ranked]
    ax.bar(np.arange(len(values)), values, color="#f97316", width=1.0)
    ax.axvline(top_k - 0.5, color="#0f172a", linestyle="--", linewidth=1.0)
    ax.set_title("Text-side attention mass")
    ax.set_xlabel("heads sorted by text mass")
    ax.set_ylabel("mean text mass")
    ax.set_ylim(0, min(1.0, max(values) * 1.12 if values else 1.0))
    save_matplotlib(fig, os.path.join(output_dir, "phase1_text_mass_sorted.png"))

    fig, ax = plt.subplots(figsize=(3.0, 1.9))
    hall = [float(row["bounded_ratio"]) for row in ratio_rows if row["label"] == "hallucinated"]
    ground = [float(row["bounded_ratio"]) for row in ratio_rows if row["label"] == "grounded"]
    bins = np.linspace(0.0, 1.0, 35)
    if ground:
        ax.hist(ground, bins=bins, density=True, alpha=0.45, color="#2563eb", label="grounded")
        ax.axvline(np.median(ground), color="#2563eb", linewidth=1.3)
    if hall:
        ax.hist(hall, bins=bins, density=True, alpha=0.45, color="#dc2626", label="hallucinated")
        ax.axvline(np.median(hall), color="#dc2626", linewidth=1.3)
    ax.set_title("Online text-ratio overlap")
    ax.set_xlabel("text / (text + image)")
    ax.set_ylabel("density")
    ax.legend(frameon=False)
    save_matplotlib(fig, os.path.join(output_dir, "phase1_ratio_distribution.png"))

    max_layer = max(int(row["layer"]) for row in head_rows) if head_rows else 31
    max_head = max(int(row["head"]) for row in head_rows) if head_rows else 31
    mat = np.zeros((max_layer + 1, max_head + 1), dtype=np.float64)
    selected = np.zeros_like(mat, dtype=bool)
    for row in head_rows:
        layer, head = int(row["layer"]), int(row["head"])
        mat[layer, head] = float(row["score"])
        selected[layer, head] = bool(int(row["selected"]))
    fig, ax = plt.subplots(figsize=(3.8, 2.6))
    im = ax.imshow(mat, aspect="auto", origin="lower", cmap="magma", vmin=0.0, vmax=1.0)
    if highlight_layers:
        ax.axhspan(min(highlight_layers) - 0.5, max(highlight_layers) + 0.5, fill=False, edgecolor="#f59e0b", linewidth=1.2)
    ys, xs = np.where(selected)
    ax.scatter(xs, ys, s=5, facecolors="none", edgecolors="white", linewidths=0.45)
    ax.set_title("Fused head score S(l,h)")
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    fig.colorbar(im, ax=ax, fraction=0.034, pad=0.02)
    save_matplotlib(fig, os.path.join(output_dir, "phase1_head_score_heatmap.png"))

    fig, ax = plt.subplots(figsize=(2.9, 1.9))
    x = [float(row["r_online"]) for row in gate_rows]
    y = [float(row["gate"]) for row in gate_rows]
    ax.plot(x, y, color="#7c3aed", linewidth=2.0)
    ax.axvline(0.9, color="#dc2626", linestyle="--", linewidth=1.0)
    if ground:
        gm = float(np.median(ground))
        ax.scatter([gm], [math.exp(float(gate_rows[0]["beta"]) * (gm - 0.9))], color="#2563eb", s=22, zorder=3)
    if hall:
        hm = float(np.median(hall))
        ax.scatter([hm], [math.exp(float(gate_rows[0]["beta"]) * (hm - 0.9))], color="#dc2626", s=22, zorder=3)
    ax.set_title("Exponential online gate")
    ax.set_xlabel("r online")
    ax.set_ylabel("exp(q(r-tau))")
    save_matplotlib(fig, os.path.join(output_dir, "phase2_gate_curve.png"))

    by_label = {row["label"]: row for row in redistribution_summary}
    labels = [label for label in ("grounded", "hallucinated") if label in by_label]
    fig, ax = plt.subplots(figsize=(3.2, 1.95))
    xloc = np.arange(len(labels))
    width = 0.34
    for offset, suffix, alpha in [(-width / 2, "before", 0.95), (width / 2, "after", 0.55)]:
        bottoms = np.zeros(len(labels))
        for key, color in [("system", "#64748b"), ("image", "#2563eb"), ("text", "#f97316")]:
            vals = [float(by_label[label][f"{key}_{suffix}"]) for label in labels]
            ax.bar(xloc + offset, vals, width, bottom=bottoms, color=color, alpha=alpha)
            bottoms += np.array(vals)
    ax.set_xticks(xloc)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_title("Suppress + renormalize")
    ax.set_ylabel("attention share")
    save_matplotlib(fig, os.path.join(output_dir, "phase2_attention_redistribution.png"))


def main(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    disable_torch_init()

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, args.model_base, model_name)
    special_token_ids = get_special_token_ids(tokenizer)

    questions = load_coco_questions(args)
    questions = questions[: args.num_samples] if args.num_samples > 0 else questions
    os.makedirs(args.output_dir, exist_ok=True)

    annotation_dir = args.annotation_dir or os.path.dirname(args.caption_file_path)
    chair = CHAIR([q["question_id"] for q in questions], annotation_dir)
    chair.get_annotations()

    stats = {bucket: MatrixStats() for bucket in BUCKETS}
    object_records = []
    sample_rows = []

    for sample_idx, question in enumerate(tqdm(questions, desc="samples"), start=1):
        input_ids, image_tensor, image_sizes, _ = build_prompt_inputs(
            tokenizer,
            image_processor,
            model.config,
            args.image_folder,
            question["image"],
            question["text"],
            args.conv_mode,
        )
        input_ids = input_ids.to(device="cuda", non_blocking=True)
        image_tensor = image_tensor.to(dtype=torch.float16, device="cuda", non_blocking=True)

        with torch.inference_mode():
            output = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                output_attentions=True,
                return_dict_in_generate=True,
            )

        output_ids = output["sequences"]
        generated_ids_only = extract_generated_ids(output_ids, input_ids)
        caption = tokenizer.batch_decode(generated_ids_only, skip_special_tokens=True)[0].strip()
        gt_objects = set(chair.imid_to_objects.get(question["question_id"], set()))
        labels = classify_generation_steps(tokenizer, generated_ids_only, chair, gt_objects)
        attentions = output.get("attentions", None)
        if attentions is None:
            continue

        sample_rows.append(
            {
                "sample_index": sample_idx,
                "question_id": question["question_id"],
                "image": question["image"],
                "caption": caption,
                "n_steps": min(len(attentions), len(labels)),
                "n_hallucinated_steps": sum(1 for label in labels if label["is_hallucinated"]),
                "n_grounded_steps": sum(1 for label in labels if label["is_non_hallucinated"]),
            }
        )

        generated_ids_list = generated_ids_only[0].detach().cpu().tolist()
        for step_idx, step_attentions in enumerate(attentions[: len(labels)]):
            label = labels[step_idx]
            values, layout = step_region_matrices(
                step_attentions,
                input_ids,
                generated_ids_list[:step_idx],
                special_token_ids,
            )
            buckets = ["all"]
            if label["is_object"]:
                buckets.append("object")
            if label["is_hallucinated"]:
                buckets.append("hallucinated")
            if label["is_non_hallucinated"]:
                buckets.append("non_hallucinated")
            for bucket in buckets:
                stats[bucket].update(values)

            if label["is_hallucinated"] or label["is_non_hallucinated"]:
                object_records.append(
                    {
                        "question_id": question["question_id"],
                        "image": question["image"],
                        "step_idx": int(step_idx),
                        "token_id": label["token_id"],
                        "token_text": label["token_text"],
                        "label": "hallucinated" if label["is_hallucinated"] else "grounded",
                        "objects": label["objects"],
                        "layout": {k: v for k, v in layout.items() if k not in {"generated_positions", "generated_special_positions"}},
                        "system_mass": values["system_mass"].detach().cpu().float(),
                        "image_mass": values["image_mass"].detach().cpu().float(),
                        "text_mass": values["text_mass"].detach().cpu().float(),
                        "raw_toi": values["raw_toi"].detach().cpu().float(),
                        "bounded_ratio": values["bounded_ratio"].detach().cpu().float(),
                    }
                )

    text_all = stats["all"].mean_tensor("text_mass")
    if text_all is None:
        raise RuntimeError("No attention statistics were collected.")

    num_layers, num_heads = text_all.shape
    layers = np.repeat(np.arange(num_layers), num_heads)
    heads = np.tile(np.arange(num_heads), num_layers)
    flat_text_all = text_all.numpy().reshape(-1)

    raw_h = stats["hallucinated"].mean_tensor("raw_toi")
    raw_g = stats["non_hallucinated"].mean_tensor("raw_toi")
    bounded_h = stats["hallucinated"].mean_tensor("bounded_ratio")
    bounded_g = stats["non_hallucinated"].mean_tensor("bounded_ratio")
    image_h = stats["hallucinated"].mean_tensor("image_mass")
    image_g = stats["non_hallucinated"].mean_tensor("image_mass")
    text_h = stats["hallucinated"].mean_tensor("text_mass")
    text_g = stats["non_hallucinated"].mean_tensor("text_mass")

    flat_raw_gap = (
        torch.clamp((raw_h if raw_h is not None else torch.zeros_like(text_all)) - (raw_g if raw_g is not None else torch.zeros_like(text_all)), min=0.0)
        .numpy()
        .reshape(-1)
    )
    front_pct = rank_percentiles(flat_text_all)
    contrast_pct = rank_percentiles(flat_raw_gap)
    score = 0.5 * front_pct + 0.5 * contrast_pct

    allowed_layers = set(parse_int_ranges(args.selection_layers))
    allowed_mask = np.ones_like(score, dtype=bool)
    if allowed_layers:
        allowed_mask = np.array([int(layer) in allowed_layers for layer in layers], dtype=bool)
    order = sorted(range(len(score)), key=lambda idx: (allowed_mask[idx], score[idx], front_pct[idx]), reverse=True)
    text_mass_order = sorted(range(len(score)), key=lambda idx: flat_text_all[idx], reverse=True)
    text_mass_rank = {idx: rank for rank, idx in enumerate(text_mass_order, start=1)}
    selected_indices = set(order[: min(args.top_k, len(order))])

    head_rows = []
    for idx in range(len(score)):
        layer, head = int(layers[idx]), int(heads[idx])
        head_rows.append(
            {
                "rank": order.index(idx) + 1,
                "text_mass_rank": int(text_mass_rank[idx]),
                "layer": layer,
                "head": head,
                "head_key": f"{layer}:{head}",
                "selected": 1 if idx in selected_indices else 0,
                "selection_allowed": 1 if allowed_mask[idx] else 0,
                "score": float(score[idx]),
                "text_percentile": float(front_pct[idx]),
                "contrast_percentile": float(contrast_pct[idx]),
                "text_mass_all": float(flat_text_all[idx]),
                "text_mass_hallucinated": float(text_h.numpy().reshape(-1)[idx]) if text_h is not None else 0.0,
                "text_mass_grounded": float(text_g.numpy().reshape(-1)[idx]) if text_g is not None else 0.0,
                "image_mass_hallucinated": float(image_h.numpy().reshape(-1)[idx]) if image_h is not None else 0.0,
                "image_mass_grounded": float(image_g.numpy().reshape(-1)[idx]) if image_g is not None else 0.0,
                "raw_toi_hallucinated": float(raw_h.numpy().reshape(-1)[idx]) if raw_h is not None else 0.0,
                "raw_toi_grounded": float(raw_g.numpy().reshape(-1)[idx]) if raw_g is not None else 0.0,
                "raw_toi_gap_hall_minus_grounded": float(flat_raw_gap[idx]),
                "bounded_ratio_hallucinated": float(bounded_h.numpy().reshape(-1)[idx]) if bounded_h is not None else 0.0,
                "bounded_ratio_grounded": float(bounded_g.numpy().reshape(-1)[idx]) if bounded_g is not None else 0.0,
            }
        )
    head_rows.sort(key=lambda row: int(row["rank"]))

    selected_pairs = [(int(row["layer"]), int(row["head"]), float(row["score"])) for row in head_rows if int(row["selected"])]
    selected_pair_set = {(layer, head) for layer, head, _ in selected_pairs}
    selected_score = {(layer, head): score for layer, head, score in selected_pairs}

    ratio_rows = []
    redistribution_rows = []
    for record in object_records:
        for layer, head, prior in selected_pairs:
            system = float(record["system_mass"][layer, head].item())
            image = float(record["image_mass"][layer, head].item())
            text = float(record["text_mass"][layer, head].item())
            raw = float(record["raw_toi"][layer, head].item())
            bounded = float(record["bounded_ratio"][layer, head].item())
            gate = math.exp(max(min(args.gate_beta * (bounded - args.gate_tau), 20.0), -20.0))
            delta = min(max(args.gate_strength * prior * gate, 0.0), 1.0)
            text_after_raw = (1.0 - delta) * text
            if args.renormalize:
                denom = max(system + image + text_after_raw, EPS)
            else:
                denom = 1.0
            row_common = {
                "question_id": record["question_id"],
                "image": record["image"],
                "step_idx": record["step_idx"],
                "token_id": record["token_id"],
                "token_text": record["token_text"],
                "label": record["label"],
                "layer": layer,
                "head": head,
                "head_key": f"{layer}:{head}",
                "score": prior,
                "system_before": system,
                "image_before": image,
                "text_before": text,
                "raw_toi": raw,
                "bounded_ratio": bounded,
                "gate": gate,
                "delta": delta,
            }
            ratio_rows.append(row_common)
            redistribution_rows.append(
                {
                    **row_common,
                    "system_after": system / denom,
                    "image_after": image / denom,
                    "text_after": text_after_raw / denom,
                    "renormalize": int(bool(args.renormalize)),
                }
            )

    def agg(rows, label):
        items = [row for row in rows if row["label"] == label]
        if not items:
            return {"label": label, "n": 0}
        return {
            "label": label,
            "n": len(items),
            "system_before": float(np.mean([row["system_before"] for row in items])),
            "image_before": float(np.mean([row["image_before"] for row in items])),
            "text_before": float(np.mean([row["text_before"] for row in items])),
            "system_after": float(np.mean([row["system_after"] for row in items])),
            "image_after": float(np.mean([row["image_after"] for row in items])),
            "text_after": float(np.mean([row["text_after"] for row in items])),
            "bounded_ratio_median": float(np.median([row["bounded_ratio"] for row in items])),
            "bounded_ratio_mean": float(np.mean([row["bounded_ratio"] for row in items])),
            "delta_mean": float(np.mean([row["delta"] for row in items])),
            "delta_median": float(np.median([row["delta"] for row in items])),
        }

    redistribution_summary = [agg(redistribution_rows, "grounded"), agg(redistribution_rows, "hallucinated")]
    ratio_summary = [agg(redistribution_rows, "grounded"), agg(redistribution_rows, "hallucinated")]
    marker_rows = []
    for row in ratio_summary:
        if row.get("n", 0):
            r_value = float(row["bounded_ratio_median"])
            marker_rows.append(
                {
                    "label": row["label"],
                    "bounded_ratio_median": r_value,
                    "gate_at_median": float(math.exp(args.gate_beta * (r_value - args.gate_tau))),
                    "beta": float(args.gate_beta),
                    "tau": float(args.gate_tau),
                }
            )
    top_rows = [row for row in head_rows if int(row["selected"])]
    tail_rows = [row for row in head_rows if int(row["rank"]) > max(args.top_k, 1)]
    rank_fusion_rows = [
        {
            "bucket": "selected",
            "n_heads": len(top_rows),
            "mean_text_percentile": float(np.mean([row["text_percentile"] for row in top_rows])) if top_rows else 0.0,
            "mean_contrast_percentile": float(np.mean([row["contrast_percentile"] for row in top_rows])) if top_rows else 0.0,
            "mean_score": float(np.mean([row["score"] for row in top_rows])) if top_rows else 0.0,
            "median_text_mass": float(np.median([row["text_mass_all"] for row in top_rows])) if top_rows else 0.0,
        },
        {
            "bucket": "tail",
            "n_heads": len(tail_rows),
            "mean_text_percentile": float(np.mean([row["text_percentile"] for row in tail_rows])) if tail_rows else 0.0,
            "mean_contrast_percentile": float(np.mean([row["contrast_percentile"] for row in tail_rows])) if tail_rows else 0.0,
            "mean_score": float(np.mean([row["score"] for row in tail_rows])) if tail_rows else 0.0,
            "median_text_mass": float(np.median([row["text_mass_all"] for row in tail_rows])) if tail_rows else 0.0,
        },
    ]
    gate_rows = []
    for value in np.linspace(args.gate_x_min, args.gate_x_max, 301):
        gate_rows.append(
            {
                "r_online": float(value),
                "gate": float(math.exp(args.gate_beta * (float(value) - args.gate_tau))),
                "beta": float(args.gate_beta),
                "tau": float(args.gate_tau),
            }
        )

    ranked_heads_json = {
        "score_name": "sample_trace__itext_all__C_toi_HminusG",
        "description": "Sample-100 figure source ranking: 0.5*rank(I_text_all)+0.5*rank(positive raw text-over-image gap hall-grounded).",
        "selection_layers": sorted(allowed_layers) if allowed_layers else "all",
        "top_k": int(args.top_k),
        "heads": [
            {
                "layer": int(row["layer"]),
                "head": int(row["head"]),
                "head_id": f"L{int(row['layer'])}H{int(row['head'])}",
                "sample_trace__itext_all__C_toi_HminusG": float(row["score"]),
                "Itext_all": float(row["text_mass_all"]),
                "RawTOI_hallucinated": float(row["raw_toi_hallucinated"]),
                "RawTOI_non_hallucinated": float(row["raw_toi_grounded"]),
                "global_rank": int(row["rank"]),
            }
            for row in head_rows
        ],
    }

    paths = {
        "samples": os.path.join(args.output_dir, "samples.csv"),
        "head_scores": os.path.join(args.output_dir, "head_scores_all.csv"),
        "ranked_heads": os.path.join(args.output_dir, "ranked_heads_sample_trace.json"),
        "ratio_distribution": os.path.join(args.output_dir, "selected_head_object_ratio_distribution.csv"),
        "gate_curve": os.path.join(args.output_dir, "gate_curve.csv"),
        "gate_markers": os.path.join(args.output_dir, "gate_markers.csv"),
        "attention_redistribution": os.path.join(args.output_dir, "selected_head_attention_redistribution.csv"),
        "attention_redistribution_summary": os.path.join(args.output_dir, "attention_redistribution_summary.csv"),
        "rank_fusion_summary": os.path.join(args.output_dir, "rank_fusion_summary.csv"),
        "summary": os.path.join(args.output_dir, "method_figure_source_summary.json"),
    }
    write_csv(paths["samples"], sample_rows)
    write_csv(paths["head_scores"], head_rows)
    write_json(paths["ranked_heads"], ranked_heads_json)
    write_csv(paths["ratio_distribution"], ratio_rows)
    write_csv(paths["gate_curve"], gate_rows)
    write_csv(paths["gate_markers"], marker_rows)
    write_csv(paths["attention_redistribution"], redistribution_rows)
    write_csv(paths["attention_redistribution_summary"], redistribution_summary)
    write_csv(paths["rank_fusion_summary"], rank_fusion_rows)

    summary = {
        "model_path": args.model_path,
        "model_name": model_name,
        "num_samples": len(sample_rows),
        "num_layers": int(num_layers),
        "num_heads": int(num_heads),
        "top_k": int(args.top_k),
        "selection_layers": sorted(allowed_layers) if allowed_layers else "all",
        "gate": {
            "strength": float(args.gate_strength),
            "beta": float(args.gate_beta),
            "tau": float(args.gate_tau),
            "renormalize": bool(args.renormalize),
        },
        "n_object_records": len(object_records),
        "n_ratio_rows": len(ratio_rows),
        "selected_heads": [f"{layer}:{head}" for layer, head, _ in selected_pairs],
        "ratio_summary": ratio_summary,
        "outputs": paths,
    }
    write_json(paths["summary"], summary)

    if args.make_plots:
        plot_sources(
            args.output_dir,
            head_rows,
            ratio_rows,
            redistribution_summary,
            gate_rows,
            args.top_k,
            sorted(allowed_layers),
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--image-folder", default="/home/kms/data/pope/val2014")
    parser.add_argument("--caption-file-path", default="/home/kms/data/images/mscoco/annotations/captions_val2014.json")
    parser.add_argument("--annotation-dir", default="")
    parser.add_argument("--output-dir", default="./results/coco/method_figure_source_trace_n100")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--conv-mode", default="vicuna_v1")
    parser.add_argument("--prompt", default="Please describe this image in detail.")
    parser.add_argument("--top-k", type=int, default=150)
    parser.add_argument("--selection-layers", default="")
    parser.add_argument("--gate-strength", type=float, default=0.7)
    parser.add_argument("--gate-beta", type=float, default=10.0)
    parser.add_argument("--gate-tau", type=float, default=0.9)
    parser.add_argument("--gate-x-min", type=float, default=0.5)
    parser.add_argument("--gate-x-max", type=float, default=1.0)
    parser.add_argument("--renormalize", action="store_true", default=True)
    parser.add_argument("--no-renormalize", dest="renormalize", action="store_false")
    parser.add_argument("--make-plots", action="store_true", default=True)
    parser.add_argument("--no-plots", dest="make_plots", action="store_false")
    main(parser.parse_args())

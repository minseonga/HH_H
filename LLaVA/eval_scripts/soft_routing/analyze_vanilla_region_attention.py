import argparse
import csv
import html
import json
import math
import os
import random
from collections import defaultdict

import torch
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm

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


BLUE = "#2563eb"
GREEN = "#059669"
ORANGE = "#f97316"
PURPLE = "#7c3aed"
GRAY = "#64748b"
DARK = "#0f172a"
MUTED = "#64748b"
GRID = "#e2e8f0"


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames or ["empty"])
        writer.writeheader()
        writer.writerows(rows)


def svg(path, width, height, body):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">\n'
        )
        f.write('<rect width="100%" height="100%" fill="white"/>\n')
        f.write(body)
        f.write("</svg>\n")


def text(x, y, value, size=12, fill=DARK, anchor="start", weight="400", rotate=None):
    value = html.escape(str(value))
    transform = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}" '
        f'text-anchor="{anchor}"{transform}>{value}</text>\n'
    )


def line(x1, y1, x2, y2, stroke=GRID, width=1, dash=None):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{stroke}" stroke-width="{width}"{dash_attr}/>\n'


def rect(x, y, w, h, fill, stroke=None, width=1, opacity=1.0):
    stroke_attr = f' stroke="{stroke}" stroke-width="{width}"' if stroke else ""
    return f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" fill="{fill}" opacity="{opacity}"{stroke_attr}/>\n'


def polyline(points, stroke, width=2.8):
    pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polyline points="{pts}" fill="none" stroke="{stroke}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"/>\n'


def circle(x, y, r, fill, stroke="white", width=1.0):
    return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>\n'


def polygon(points, fill, opacity=0.72):
    pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polygon points="{pts}" fill="{fill}" opacity="{opacity}"/>\n'


def configure_image_span(model, model_path):
    if model_path == "liuhaotian/llava-v1.6-34b":
        model.config.img_start_pos = 33
        model.config.img_length = 1948
    else:
        model.config.img_start_pos = 35
        model.config.img_length = 576


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
    sample_n = len(img_ids) if args.num_samples <= 0 else min(args.num_samples, len(img_ids))
    sampled = random.sample(img_ids, sample_n)
    questions = []
    for image_id in sampled:
        image_file = coco.loadImgs(image_id)[0]["file_name"]
        questions.append({
            "question_id": image_id,
            "image": image_file,
            "text": args.prompt,
        })
    return questions


def region_masses(attn_row, img_start, img_length, expanded_prompt_len):
    # attn_row: [num_heads, kv_len], one query row after softmax.
    kv_len = int(attn_row.shape[-1])
    img_start = max(0, min(int(img_start), kv_len))
    img_end = max(img_start, min(img_start + int(img_length), kv_len))
    prompt_end = max(0, min(int(expanded_prompt_len), kv_len))

    system = torch.zeros(attn_row.shape[0], device=attn_row.device, dtype=attn_row.dtype)
    if img_start > 0:
        system = system + attn_row[:, :img_start].sum(dim=-1)
    if prompt_end > img_end:
        system = system + attn_row[:, img_end:prompt_end].sum(dim=-1)

    visual = attn_row[:, img_start:img_end].sum(dim=-1) if img_end > img_start else torch.zeros_like(system)
    generated_text = (
        attn_row[:, prompt_end:kv_len].sum(dim=-1)
        if kv_len > prompt_end
        else torch.zeros_like(system)
    )
    values = {
        "system": float(system.float().mean().cpu().item()),
        "visual": float(visual.float().mean().cpu().item()),
        "text": float(generated_text.float().mean().cpu().item()),
    }
    total = max(sum(values.values()), 1e-12)
    return {key: value / total for key, value in values.items()}


def add_acc(acc, key, values):
    bucket = acc[key]
    bucket["n"] += 1
    for name, value in values.items():
        bucket[name] += float(value)


def finalize_rows(acc, key_names):
    rows = []
    for key in sorted(acc):
        item = acc[key]
        n = max(int(item["n"]), 1)
        row = {name: value for name, value in zip(key_names, key)}
        row["n"] = item["n"]
        row["system_share"] = item["system"] / n
        row["visual_share"] = item["visual"] / n
        row["text_share"] = item["text"] / n
        row["system_visual_text_sum"] = row["system_share"] + row["visual_share"] + row["text_share"]
        rows.append(row)
    return rows


def make_layer_line_svg(path, rows, title):
    width, height = 1180, 620
    left, top = 82, 82
    plot_w, plot_h = 930, 390
    colors = {"system_share": GRAY, "visual_share": BLUE, "text_share": ORANGE}
    labels = {
        "system_share": "system/prompt",
        "visual_share": "visual",
        "text_share": "generated text",
    }
    by_layer = {int(row["layer"]): row for row in rows}
    n_layers = max(by_layer) + 1 if by_layer else 32

    def sx(layer):
        return left + layer / max(n_layers - 1, 1) * plot_w

    def sy(value):
        return top + plot_h - value * plot_h

    body = []
    body.append(text(width / 2, 32, title, 21, DARK, "middle", "700"))
    body.append(text(width / 2, 55, "Each layer averages head attention over generated steps; three regions are normalized to sum to 1.", 12, MUTED, "middle"))
    body.append(rect(left, top, plot_w, plot_h, "#f8fafc", "#cbd5e1"))
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = sy(tick)
        body.append(line(left, y, left + plot_w, y, GRID))
        body.append(text(left - 12, y + 4, f"{tick:.2f}", 10, MUTED, "end"))
    for layer in range(n_layers):
        x = sx(layer)
        if layer in [11, 21, 27]:
            body.append(line(x, top, x, top + plot_h, "#334155", 1.15, "5 5"))
        elif layer % 2 == 0:
            body.append(line(x, top, x, top + plot_h, "#edf2f7", 0.8))
        body.append(text(x, top + plot_h + 22, f"L{layer}", 8, DARK, "middle", rotate=35))
    for metric in ["system_share", "visual_share", "text_share"]:
        pts = [(sx(layer), sy(float(by_layer.get(layer, {}).get(metric, 0.0)))) for layer in range(n_layers)]
        body.append(polyline(pts, colors[metric]))
        for layer, (x, y) in enumerate(pts):
            value = float(by_layer.get(layer, {}).get(metric, 0.0))
            if value > 0.01:
                body.append(circle(x, y, 3.0, colors[metric]))
    lx, ly = left + plot_w + 34, top + 24
    for idx, metric in enumerate(["system_share", "visual_share", "text_share"]):
        y = ly + idx * 30
        body.append(line(lx, y, lx + 32, y, colors[metric], 3))
        body.append(circle(lx + 16, y, 3.2, colors[metric]))
        body.append(text(lx + 42, y + 4, labels[metric], 11, DARK))
    body.append(text(left + plot_w / 2, height - 36, "layer", 13, DARK, "middle"))
    body.append(text(28, top + plot_h / 2, "attention share", 13, DARK, "middle", rotate=-90))
    svg(path, width, height, "".join(body))


def make_layer_stacked_svg(path, rows, title):
    width, height = 1180, 620
    left, top = 82, 82
    plot_w, plot_h = 930, 390
    colors = [GRAY, BLUE, ORANGE]
    metrics = ["system_share", "visual_share", "text_share"]
    labels = ["system/prompt", "visual", "generated text"]
    by_layer = {int(row["layer"]): row for row in rows}
    n_layers = max(by_layer) + 1 if by_layer else 32

    def sx(layer):
        return left + layer / max(n_layers - 1, 1) * plot_w

    def sy(value):
        return top + plot_h - value * plot_h

    cumulative = [[0.0 for _ in range(n_layers)]]
    for metric in metrics:
        prev = cumulative[-1]
        cumulative.append([
            prev[layer] + float(by_layer.get(layer, {}).get(metric, 0.0))
            for layer in range(n_layers)
        ])

    body = []
    body.append(text(width / 2, 32, title, 21, DARK, "middle", "700"))
    body.append(text(width / 2, 55, "Stacked regions are normalized per layer: system/prompt + visual + generated text = 1.", 12, MUTED, "middle"))
    body.append(rect(left, top, plot_w, plot_h, "#f8fafc", "#cbd5e1"))
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = sy(tick)
        body.append(line(left, y, left + plot_w, y, GRID))
        body.append(text(left - 12, y + 4, f"{tick:.2f}", 10, MUTED, "end"))
    for layer in range(n_layers):
        x = sx(layer)
        if layer in [11, 21, 27]:
            body.append(line(x, top, x, top + plot_h, "#334155", 1.15, "5 5"))
        elif layer % 2 == 0:
            body.append(line(x, top, x, top + plot_h, "#edf2f7", 0.8))
        body.append(text(x, top + plot_h + 22, f"L{layer}", 8, DARK, "middle", rotate=35))
    for idx, color in enumerate(colors):
        lower = cumulative[idx]
        upper = cumulative[idx + 1]
        top_pts = [(sx(layer), sy(upper[layer])) for layer in range(n_layers)]
        bot_pts = [(sx(layer), sy(lower[layer])) for layer in reversed(range(n_layers))]
        body.append(polygon(top_pts + bot_pts, color, 0.68))
        body.append(polyline(top_pts, color, 1.5))
    lx, ly = left + plot_w + 34, top + 24
    for idx, label in enumerate(labels):
        body.append(rect(lx, ly + idx * 30 - 12, 22, 18, colors[idx], None, opacity=0.75))
        body.append(text(lx + 34, ly + idx * 30 + 3, label, 11, DARK))
    body.append(text(left + plot_w / 2, height - 36, "layer", 13, DARK, "middle"))
    body.append(text(28, top + plot_h / 2, "attention share", 13, DARK, "middle", rotate=-90))
    svg(path, width, height, "".join(body))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--image-folder", default="/home/kms/data/pope/val2014")
    parser.add_argument("--caption-file-path", default="/home/kms/data/images/mscoco/annotations/captions_val2014.json")
    parser.add_argument("--dataset", default="coco", choices=["coco"])
    parser.add_argument("--output-dir", default="./results/coco/vanilla_region_attention")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--conv-mode", default="vicuna_v1")
    parser.add_argument("--prompt", default="Please describe this image in detail.")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, args.model_base, model_name)
    configure_image_span(model, model_path)

    questions = load_coco_questions(args)
    os.makedirs(args.output_dir, exist_ok=True)

    layer_acc = defaultdict(lambda: {"n": 0, "system": 0.0, "visual": 0.0, "text": 0.0})
    step_layer_acc = defaultdict(lambda: {"n": 0, "system": 0.0, "visual": 0.0, "text": 0.0})
    sample_rows = []

    for sample_index, question in enumerate(tqdm(questions, desc="samples")):
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
        expanded_prompt_len = int(input_ids.shape[1]) - 1 + int(model.config.img_length)

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

        generated = tokenizer.batch_decode(output["sequences"], skip_special_tokens=True)[0].strip()
        attentions = output.get("attentions", None)
        if attentions is None:
            continue

        sample_rows.append({
            "sample_index": sample_index,
            "question_id": question["question_id"],
            "image": question["image"],
            "caption": generated,
            "n_attention_steps": len(attentions),
        })

        for token_pos, step_attentions in enumerate(attentions):
            for layer, attn in enumerate(step_attentions):
                row = attn[0, :, -1, :].detach()
                values = region_masses(
                    row,
                    model.config.img_start_pos,
                    model.config.img_length,
                    expanded_prompt_len,
                )
                add_acc(layer_acc, (layer,), values)
                add_acc(step_layer_acc, (token_pos, layer), values)

    layer_rows = finalize_rows(layer_acc, ["layer"])
    step_layer_rows = finalize_rows(step_layer_acc, ["token_pos", "layer"])

    layer_csv = os.path.join(args.output_dir, "vanilla_region_attention_by_layer.csv")
    step_layer_csv = os.path.join(args.output_dir, "vanilla_region_attention_by_token_layer.csv")
    sample_csv = os.path.join(args.output_dir, "vanilla_region_attention_samples.csv")
    write_csv(layer_csv, layer_rows)
    write_csv(step_layer_csv, step_layer_rows)
    write_csv(sample_csv, sample_rows)

    line_svg = os.path.join(args.output_dir, "vanilla_region_attention_layer_lines.svg")
    stacked_svg = os.path.join(args.output_dir, "vanilla_region_attention_layer_stacked.svg")
    make_layer_line_svg(line_svg, layer_rows, "Vanilla attention by source region")
    make_layer_stacked_svg(stacked_svg, layer_rows, "Vanilla attention by source region")

    summary = {
        "model_path": args.model_path,
        "num_samples": len(sample_rows),
        "img_start_pos": int(model.config.img_start_pos),
        "img_length": int(model.config.img_length),
        "region_definition": {
            "system_prompt": "[0, image_start) union [image_end, expanded_prompt_end)",
            "visual": "[image_start, image_end)",
            "generated_text": "[expanded_prompt_end, kv_len)",
        },
        "outputs": {
            "layer_csv": layer_csv,
            "step_layer_csv": step_layer_csv,
            "sample_csv": sample_csv,
            "line_svg": line_svg,
            "stacked_svg": stacked_svg,
        },
    }
    with open(os.path.join(args.output_dir, "vanilla_region_attention_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

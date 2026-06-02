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
    node_pairs,
    one_step,
    token_candidates,
)


def load_sentences(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("sentences", [])


def normalize_word(text):
    return " ".join(str(text).lower().strip().split())


def find_all_subsequences(sequence, subsequence, min_pos=0):
    if not subsequence:
        return []
    out = []
    start = max(0, int(min_pos))
    for idx in range(start, len(sequence) - len(subsequence) + 1):
        if sequence[idx : idx + len(subsequence)] == subsequence:
            out.append((idx, idx + len(subsequence)))
    return out


def overlaps(span, used_spans):
    start, end = span
    return any(start < other_end and end > other_start for other_start, other_end in used_spans)


def find_next_object_span(tokenizer, caption_ids, object_word, min_pos, used_spans):
    candidates = []
    for candidate in token_candidates(tokenizer, object_word):
        for start, end in find_all_subsequences(caption_ids, candidate, min_pos=min_pos):
            if not overlaps((start, end), used_spans):
                candidates.append((start, end, candidate))
        if len(candidate) > 1:
            tail = candidate[-1:]
            for start, end in find_all_subsequences(caption_ids, tail, min_pos=min_pos):
                if not overlaps((start, end), used_spans):
                    candidates.append((start, end, tail))
    if not candidates and min_pos > 0:
        return find_next_object_span(tokenizer, caption_ids, object_word, 0, used_spans)
    if not candidates:
        return None, None, None
    candidates.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    start, end, matched = candidates[0]
    return start, end, matched


def mention_queues(sentence):
    queues = collections.defaultdict(collections.deque)
    for label, key in [
        ("grounded_object", "mscoco_non_hallucinated_words"),
        ("hallucinated_object", "mscoco_hallucinated_words"),
    ]:
        for mention_index, (word, node) in enumerate(node_pairs(sentence, key)):
            queues[normalize_word(word)].append(
                {
                    "label": label,
                    "object_word": word,
                    "object_node": node,
                    "mention_index": mention_index,
                }
            )
    return queues


def select_base_object_rows(sentences, tokenizer, max_per_label):
    selected = []
    counts = collections.Counter()
    missed = collections.Counter()

    for sentence in sentences:
        caption = sentence.get("caption", "")
        caption_ids = tokenizer(caption, add_special_tokens=False)["input_ids"]
        queues = mention_queues(sentence)
        used_spans = []
        cursor = 0

        generated_words = list(sentence.get("mscoco_generated_words", []))
        if not generated_words:
            generated_words = [
                word
                for key in ("mscoco_non_hallucinated_words", "mscoco_hallucinated_words")
                for word, _ in node_pairs(sentence, key)
            ]

        for generated_index, generated_word in enumerate(generated_words):
            key = normalize_word(generated_word)
            if not queues.get(key):
                continue
            mention = queues[key].popleft()
            label = mention["label"]
            if counts[label] >= max_per_label:
                continue
            start, end, matched_ids = find_next_object_span(
                tokenizer,
                caption_ids,
                mention["object_word"],
                cursor,
                used_spans,
            )
            if start is None:
                missed[label] += 1
                continue
            used_spans.append((start, end))
            cursor = max(cursor, end)
            selected.append(
                {
                    "image_id": str(sentence["image_id"]),
                    "image": sentence["image"],
                    "label": label,
                    "object_node": mention["object_node"],
                    "object_word": mention["object_word"],
                    "generated_word": generated_word,
                    "generated_index": generated_index,
                    "mention_index": mention["mention_index"],
                    "target_token_pos": start,
                    "target_token_id": int(caption_ids[start]),
                    "matched_token_ids": matched_ids,
                    "probe_caption_ids": caption_ids,
                    "probe_caption": caption,
                    "gt_words": sorted(str(item) for item in sentence.get("mscoco_gt_words", [])),
                }
            )
            counts[label] += 1
    return selected, counts, missed


def write_csv(path, rows):
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def percentile(values, q):
    if not values:
        return None
    return float(np.percentile(np.array(values, dtype=np.float64), q))


def mean(values):
    if not values:
        return None
    return float(np.mean(np.array(values, dtype=np.float64)))


def summarize_rows(rows):
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[row["label"]].append(row)
    summary = []
    for label in sorted(grouped):
        items = grouped[label]
        drops = [float(row["target_logprob_drop_static"]) for row in items]
        rank_deltas = [float(row["target_rank_delta_static"]) for row in items]
        kls = [float(row["kl_base_to_static"]) for row in items]
        summary.append(
            {
                "label": label,
                "n": len(items),
                "mean_delta_logprob": mean(drops),
                "median_delta_logprob": percentile(drops, 50),
                "q75_delta_logprob": percentile(drops, 75),
                "q90_delta_logprob": percentile(drops, 90),
                "positive_drop_fraction": mean([1.0 if value > 0 else 0.0 for value in drops]),
                "mean_positive_delta_logprob": mean([value for value in drops if value > 0]),
                "mean_rank_delta": mean(rank_deltas),
                "q90_rank_delta": percentile(rank_deltas, 90),
                "mean_kl_base_to_static": mean(kls),
                "static_changes_next_token_fraction": mean(
                    [1.0 if row["base_next_token_id"] != row["static_next_token_id"] else 0.0 for row in items]
                ),
                "base_target_top1_fraction": mean(
                    [1.0 if int(row["base_target_rank"]) == 1 else 0.0 for row in items]
                ),
                "static_target_top1_fraction": mean(
                    [1.0 if int(row["static_target_rank"]) == 1 else 0.0 for row in items]
                ),
            }
        )
    return summary


def write_notes(path, summary_rows, meta):
    by_label = {row["label"]: row for row in summary_rows}
    grounded = by_label.get("grounded_object", {})
    hall = by_label.get("hallucinated_object", {})
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Full Static Object Log-Probability Drop\n\n")
        f.write("Diagnostic definition:\n\n")
        f.write("`Delta log p(y_t) = log p_base(y_t) - log p_static(y_t)`\n\n")
        f.write("Rows are object mentions from the base/greedy caption. The same image prompt and generated prefix are replayed twice: once without suppression and once with hard/static text-side suppression on the selected head pool.\n\n")
        f.write("## Summary\n\n")
        f.write("| label | n | mean drop | median drop | q90 drop | positive drop frac | static target top1 frac |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for row in summary_rows:
            f.write(
                f"| {row['label']} | {row['n']} | {row['mean_delta_logprob']:.4f} | "
                f"{row['median_delta_logprob']:.4f} | {row['q90_delta_logprob']:.4f} | "
                f"{row['positive_drop_fraction']:.3f} | {row['static_target_top1_fraction']:.3f} |\n"
            )
        f.write("\n## Interpretation\n\n")
        if grounded and hall:
            f.write(
                "Static suppression produces positive target-log-probability drops for both hallucinated and grounded object tokens. "
                f"For grounded objects, the mean drop is {grounded.get('mean_delta_logprob', 0.0):.4f} "
                f"with a positive-drop fraction of {grounded.get('positive_drop_fraction', 0.0):.3f}. "
                f"For hallucinated objects, the mean drop is {hall.get('mean_delta_logprob', 0.0):.4f} "
                f"with a positive-drop fraction of {hall.get('positive_drop_fraction', 0.0):.3f}. "
                "This supports the coarse-intervention argument: hard/static suppression can weaken hallucinated object likelihood, "
                "but it can also perturb ordinary grounded object realization because the selected heads are not hallucination-only detectors.\n"
            )
        f.write("\n## Metadata\n\n")
        for key, value in meta.items():
            f.write(f"- `{key}`: `{value}`\n")


def maybe_plot(output_dir, rows):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []
    labels = ["grounded_object", "hallucinated_object"]
    data = [
        [float(row["target_logprob_drop_static"]) for row in rows if row["label"] == label]
        for label in labels
    ]
    if not any(data):
        return []
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    parts = ax.violinplot(data, showmeans=True, showextrema=False)
    colors = ["#2e8b57", "#d95f02"]
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.42)
    if "cmeans" in parts:
        parts["cmeans"].set_color("#1f2933")
        parts["cmeans"].set_linewidth(1.4)
    ax.axhline(0, color="#6b7280", linewidth=1, linestyle="--")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["grounded", "hallucinated"])
    ax.set_ylabel(r"$\Delta \log p(y_t)$")
    ax.set_title("Full static suppression affects grounded object likelihood too")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    paths = []
    for ext in ("png", "svg", "pdf"):
        path = os.path.join(output_dir, f"static_object_logprob_drop_by_label.{ext}")
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
    parser.add_argument("--top-k", type=int, default=150)
    parser.add_argument("--model-path", default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--conv-mode", default="vicuna_v1")
    parser.add_argument("--max-per-label", type=int, default=200)
    parser.add_argument("--adhh-threshold", type=float, default=0.4)
    parser.add_argument("--soft-gamma", type=float, default=0.75)
    parser.add_argument("--soft-temperature", type=float, default=0.05)
    args = parser.parse_args()

    disable_torch_init()
    model_name = get_model_name_from_path(os.path.expanduser(args.model_path))
    tokenizer, model, image_processor, _ = load_pretrained_model(args.model_path, args.model_base, model_name)
    heads, priors, prior_source = configure_model(
        model,
        args.model_path,
        args.prior_path,
        args.top_k,
        args.adhh_threshold,
        args.soft_gamma,
        args.soft_temperature,
    )

    sentences = load_sentences(args.base_results)
    selected, counts, missed = select_base_object_rows(sentences, tokenizer, args.max_per_label)
    os.makedirs(args.output_dir, exist_ok=True)

    output_rows = []
    with open(os.path.join(args.output_dir, "static_object_logprob_drop_rows.jsonl"), "w", encoding="utf-8") as out:
        for row in tqdm(selected, desc="object mentions"):
            prompt_ids, image_tensor, image_size = build_prompt_inputs(
                row, args.image_folder, tokenizer, image_processor, model.config, args.conv_mode
            )
            prompt_ids = prompt_ids.to(device="cuda", non_blocking=True)
            image_tensor = image_tensor.to(dtype=torch.float16, device="cuda", non_blocking=True)
            prefix_ids = row["probe_caption_ids"][: int(row["target_token_pos"])]
            target_token_id = int(row["target_token_id"])

            base = one_step(model, tokenizer, prompt_ids, prefix_ids, image_tensor, image_size, target_token_id, "none")
            static = one_step(model, tokenizer, prompt_ids, prefix_ids, image_tensor, image_size, target_token_id, "hard")
            output = {
                "image_id": row["image_id"],
                "image": row["image"],
                "label": row["label"],
                "object_node": row["object_node"],
                "object_word": row["object_word"],
                "generated_word": row["generated_word"],
                "generated_index": row["generated_index"],
                "target_token": tokenizer.decode([target_token_id]),
                "target_token_id": target_token_id,
                "target_token_pos": int(row["target_token_pos"]),
                "prefix_text": tokenizer.decode(prefix_ids, skip_special_tokens=True).strip(),
                "base_target_logprob": base["target_logprob"],
                "static_target_logprob": static["target_logprob"],
                "target_logprob_drop_static": base["target_logprob"] - static["target_logprob"],
                "base_target_rank": base["target_rank"],
                "static_target_rank": static["target_rank"],
                "target_rank_delta_static": static["target_rank"] - base["target_rank"],
                "base_next_token_id": base["next_token_id"],
                "static_next_token_id": static["next_token_id"],
                "base_next_token": base["next_token"],
                "static_next_token": static["next_token"],
                "base_entropy": base["entropy"],
                "static_entropy": static["entropy"],
                "static_entropy_minus_base": static["entropy"] - base["entropy"],
                "kl_base_to_static": kl_divergence(base["score"], static["score"]),
                "probe_caption": row["probe_caption"],
                "gt_words": ", ".join(row["gt_words"]),
                "prior_source": prior_source,
            }
            output_rows.append(output)
            out.write(json.dumps(output, ensure_ascii=False) + "\n")
            out.flush()

    summary_rows = summarize_rows(output_rows)
    meta = {
        "base_results": args.base_results,
        "prior_path": args.prior_path,
        "top_k": args.top_k,
        "adhh_threshold": args.adhh_threshold,
        "prior_source": prior_source,
        "n_heads": len(heads),
        "selected_counts": dict(counts),
        "missed_counts": dict(missed),
    }
    write_csv(os.path.join(args.output_dir, "static_object_logprob_drop_rows.csv"), output_rows)
    write_csv(os.path.join(args.output_dir, "static_object_logprob_drop_summary.csv"), summary_rows)
    figure_paths = maybe_plot(args.output_dir, output_rows)
    summary = {
        **meta,
        "n_rows": len(output_rows),
        "summary": summary_rows,
        "figure_paths": figure_paths,
    }
    with open(os.path.join(args.output_dir, "static_object_logprob_drop_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    write_notes(os.path.join(args.output_dir, "static_object_logprob_drop_notes.md"), summary_rows, meta)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

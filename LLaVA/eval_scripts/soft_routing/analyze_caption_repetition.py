import argparse
import csv
import glob
import json
import os
import re
from collections import Counter


def tokens(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def ngrams(items, n):
    if len(items) < n:
        return []
    return [tuple(items[i:i + n]) for i in range(len(items) - n + 1)]


def flatten_node_words(pairs):
    out = []
    for item in pairs or []:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            out.append(str(item[1]))
    return out


def repetition_stats(items):
    if not items:
        return {
            "count": 0,
            "unique_count": 0,
            "repeat_count": 0,
            "repeat_rate": 0.0,
            "max_repeat": 0,
        }
    counts = Counter(items)
    repeat_count = sum(max(0, count - 1) for count in counts.values())
    return {
        "count": len(items),
        "unique_count": len(counts),
        "repeat_count": repeat_count,
        "repeat_rate": repeat_count / max(len(items), 1),
        "max_repeat": max(counts.values()),
    }


def distinct_ratio(items):
    if not items:
        return 0.0
    return len(set(items)) / len(items)


def caption_metrics(sentence):
    caption = sentence.get("caption") or sentence.get("text") or ""
    words = tokens(caption)
    generated_words = sentence.get("mscoco_generated_words") or []
    if generated_words:
        object_mentions = [str(item) for item in generated_words]
    else:
        object_mentions = (
            flatten_node_words(sentence.get("mscoco_non_hallucinated_words"))
            + flatten_node_words(sentence.get("mscoco_hallucinated_words"))
        )

    object_stats = repetition_stats(object_mentions)
    word_stats = repetition_stats(words)
    bigrams = ngrams(words, 2)
    trigrams = ngrams(words, 3)
    fourgrams = ngrams(words, 4)
    bigram_stats = repetition_stats(bigrams)
    trigram_stats = repetition_stats(trigrams)
    fourgram_stats = repetition_stats(fourgrams)
    gt = set(sentence.get("mscoco_gt_words") or [])
    grounded = set(flatten_node_words(sentence.get("mscoco_non_hallucinated_words")))
    hallucinated = set(flatten_node_words(sentence.get("mscoco_hallucinated_words")))

    return {
        "image_id": sentence.get("image_id"),
        "caption_length": len(words),
        "object_mentions": object_stats["count"],
        "unique_object_mentions": object_stats["unique_count"],
        "object_repeat_count": object_stats["repeat_count"],
        "object_repeat_rate": object_stats["repeat_rate"],
        "object_max_repeat": object_stats["max_repeat"],
        "word_repeat_rate": word_stats["repeat_rate"],
        "distinct_1": distinct_ratio(words),
        "distinct_2": distinct_ratio(bigrams),
        "distinct_3": distinct_ratio(trigrams),
        "repeated_bigram_rate": bigram_stats["repeat_rate"],
        "repeated_trigram_rate": trigram_stats["repeat_rate"],
        "repeated_4gram_rate": fourgram_stats["repeat_rate"],
        "max_repeated_4gram": fourgram_stats["max_repeat"],
        "gt_object_count": len(gt),
        "grounded_unique_objects": len(grounded),
        "hallucinated_unique_objects": len(hallucinated),
        "object_recall_proxy": len(grounded) / max(len(gt), 1),
    }


def mean(values):
    values = [float(value) for value in values]
    return sum(values) / len(values) if values else 0.0


def summarize(rows):
    keys = [
        "caption_length",
        "object_mentions",
        "unique_object_mentions",
        "object_repeat_count",
        "object_repeat_rate",
        "object_max_repeat",
        "distinct_1",
        "distinct_2",
        "distinct_3",
        "repeated_bigram_rate",
        "repeated_trigram_rate",
        "repeated_4gram_rate",
        "max_repeated_4gram",
        "grounded_unique_objects",
        "hallucinated_unique_objects",
        "object_recall_proxy",
    ]
    return {key: mean([row[key] for row in rows]) for key in keys}


def load_eval(path):
    with open(path, "r") as f:
        return json.load(f)


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


def method_name(path, base_dir):
    rel = os.path.relpath(path, base_dir)
    parts = rel.split(os.sep)
    if len(parts) >= 3 and parts[-3].startswith("online_unsupported"):
        return f"{parts[-3]}__{parts[-2]}"
    if len(parts) >= 2:
        return parts[-2]
    return os.path.basename(os.path.dirname(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="results/coco/soft_routing_smoke_n500_seed42_tau0.4_T0.05")
    parser.add_argument("--eval-path", action="append", default=[])
    parser.add_argument("--pattern", action="append", default=[])
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.base_dir, "caption_repetition_analysis")
    os.makedirs(output_dir, exist_ok=True)

    paths = list(args.eval_path)
    for pattern in args.pattern:
        paths.extend(glob.glob(pattern, recursive=True))
    if not paths:
        default_patterns = [
            os.path.join(args.base_dir, "greedy", "captions_eval_results.json"),
            os.path.join(args.base_dir, "hard_tau0.4", "captions_eval_results.json"),
            os.path.join(args.base_dir, "wide_pool_gate_top100_normq75", "*", "captions_eval_results.json"),
            os.path.join(args.base_dir, "online_value_selector_top100_normq75", "*", "captions_eval_results.json"),
            os.path.join(args.base_dir, "online_unsupported*", "*", "captions_eval_results.json"),
        ]
        for pattern in default_patterns:
            paths.extend(glob.glob(pattern, recursive=True))
    paths = sorted(set(path for path in paths if os.path.exists(path)))
    if not paths:
        raise FileNotFoundError("No captions_eval_results.json files found")

    summary_rows = []
    per_caption_rows = []
    for path in paths:
        data = load_eval(path)
        method = method_name(path, args.base_dir)
        sentence_rows = [caption_metrics(sentence) for sentence in data.get("sentences", [])]
        for row in sentence_rows:
            row["method"] = method
            row["eval_path"] = path
            per_caption_rows.append(row)
        summary = summarize(sentence_rows)
        overall = data.get("overall_metrics", {})
        bleu = overall.get("Bleu") or [None, None, None, None]
        summary_rows.append({
            "method": method,
            "eval_path": path,
            "n": len(sentence_rows),
            "CHAIRs": overall.get("CHAIRs"),
            "CHAIRi": overall.get("CHAIRi"),
            "BLEU4": bleu[3],
            "avg_caption_length_chair": overall.get("avg_caption_length"),
            **summary,
        })

    summary_rows.sort(key=lambda row: str(row["method"]))
    write_csv(os.path.join(output_dir, "caption_repetition_summary.csv"), summary_rows)
    write_csv(os.path.join(output_dir, "caption_repetition_per_caption.csv"), per_caption_rows)

    print(os.path.join(output_dir, "caption_repetition_summary.csv"))
    for row in summary_rows:
        print(
            row["method"],
            "CHAIRs", row["CHAIRs"],
            "obj_repeat", f"{row['object_repeat_rate']:.3f}",
            "rep4", f"{row['repeated_4gram_rate']:.3f}",
            "obj_recall", f"{row['object_recall_proxy']:.3f}",
        )


if __name__ == "__main__":
    main()

import argparse
import csv
import json
import os
from collections import Counter


def load_eval_results(path):
    with open(path, "r") as f:
        data = json.load(f)
    return {str(item["image_id"]): item for item in data["sentences"]}


def node_words(pairs):
    return [pair[1] for pair in pairs]


def safe_div(num, den):
    return float(num) / float(den) if den else 0.0


def object_metrics(item):
    hallucinated_nodes = node_words(item.get("mscoco_hallucinated_words", []))
    correct_nodes = node_words(item.get("mscoco_non_hallucinated_words", []))
    gt_objects = set(item.get("mscoco_gt_words", []))

    hallucinated_set = set(hallucinated_nodes)
    correct_set = set(correct_nodes)
    generated_set = hallucinated_set | correct_set

    precision = safe_div(len(correct_set), len(generated_set))
    recall = safe_div(len(correct_set), len(gt_objects))
    f1 = safe_div(2 * precision * recall, precision + recall)

    return {
        "chair_s": int(item["metrics"]["CHAIRs"]),
        "chair_i": float(item["metrics"]["CHAIRi"]),
        "object_precision": precision,
        "object_recall": recall,
        "object_f1": f1,
        "unique_object_count": len(generated_set),
        "gt_object_count": len(gt_objects),
        "gt_object_count_mentioned": len(correct_set),
        "hallucinated_object_count": len(hallucinated_set),
        "correct_object_mention_count": len(correct_nodes),
        "hallucinated_object_mention_count": len(hallucinated_nodes),
        "caption_length": len(item.get("caption", "").split()),
    }


def add_bleu4(records, annotation_file):
    if not annotation_file:
        for item in records.values():
            item["bleu4"] = None
        return

    from pycocotools.coco import COCO
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.bleu.bleu import Bleu

    coco = COCO(annotation_file)
    refs = {}
    cands = {}
    ids = []
    for image_id, item in records.items():
        int_image_id = int(image_id)
        ann_ids = coco.getAnnIds(imgIds=int_image_id)
        anns = coco.loadAnns(ann_ids)
        if not anns:
            item["bleu4"] = None
            continue
        refs[image_id] = [{"caption": ann["caption"]} for ann in anns]
        cands[image_id] = [{"caption": item.get("caption", "")}]
        ids.append(image_id)

    if not ids:
        return

    tokenizer = PTBTokenizer()
    refs = tokenizer.tokenize(refs)
    cands = tokenizer.tokenize(cands)
    _, scores = Bleu().compute_score(refs, cands)
    bleu4_scores = scores[3]

    for image_id, bleu4 in zip(ids, bleu4_scores):
        records[image_id]["bleu4"] = float(bleu4)


def flatten(prefix, metrics):
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def close(a, b, eps):
    return abs(float(a) - float(b)) <= eps


def classify(hard, soft, greedy=None, eps=1e-9):
    hard_chair = hard["chair_i"]
    soft_chair = soft["chair_i"]
    hard_recall = hard["object_recall"]
    soft_recall = soft["object_recall"]
    hard_bleu = hard.get("bleu4")
    soft_bleu = soft.get("bleu4")

    if soft_chair < hard_chair - eps:
        return "soft_win", "lower_chair_i"
    if close(soft_chair, hard_chair, eps) and soft_recall > hard_recall + eps:
        return "soft_win", "same_chair_i_higher_recall"
    if (
        close(soft_chair, hard_chair, eps)
        and close(soft_recall, hard_recall, eps)
        and hard_bleu is not None
        and soft_bleu is not None
        and soft_bleu > hard_bleu + eps
    ):
        return "soft_win", "same_chair_i_recall_higher_bleu4"

    if hard_chair < soft_chair - eps and hard_recall >= soft_recall - eps:
        return "hard_win", "lower_chair_i_recall_not_worse"
    if hard_chair < soft_chair - eps and hard_recall < soft_recall - eps:
        return "hard_tradeoff", "lower_chair_i_lower_recall"

    if greedy is not None:
        hard_failed = hard_chair >= greedy["chair_i"] - eps and hard_chair > eps
        soft_failed = soft_chair >= greedy["chair_i"] - eps and soft_chair > eps
        if hard_failed and soft_failed:
            return "both_fail", "no_chair_i_improvement_vs_greedy"

    if hard_chair > eps and soft_chair > eps:
        return "both_hallucinate_tie", "both_have_hallucination"
    return "tie", "near_equal"


def build_rows(greedy_results, hard_results, soft_results, eps):
    image_ids = sorted(set(hard_results) & set(soft_results), key=lambda x: int(x))
    if greedy_results is not None:
        image_ids = [image_id for image_id in image_ids if image_id in greedy_results]

    rows = []
    for image_id in image_ids:
        hard_item = hard_results[image_id]
        soft_item = soft_results[image_id]
        greedy_item = greedy_results[image_id] if greedy_results is not None else None

        hard_metrics = object_metrics(hard_item)
        soft_metrics = object_metrics(soft_item)
        greedy_metrics = object_metrics(greedy_item) if greedy_item is not None else None

        hard_metrics["bleu4"] = hard_item.get("bleu4")
        soft_metrics["bleu4"] = soft_item.get("bleu4")
        if greedy_metrics is not None:
            greedy_metrics["bleu4"] = greedy_item.get("bleu4")

        winner, reason = classify(hard_metrics, soft_metrics, greedy_metrics, eps=eps)

        row = {
            "image_id": image_id,
            "image": hard_item.get("image", soft_item.get("image", "")),
            "winner": winner,
            "reason": reason,
            "hard_caption": hard_item.get("caption", ""),
            "soft_caption": soft_item.get("caption", ""),
        }
        if greedy_item is not None:
            row["greedy_caption"] = greedy_item.get("caption", "")
            row.update(flatten("greedy", greedy_metrics))

        row.update(flatten("hard", hard_metrics))
        row.update(flatten("soft", soft_metrics))
        row["delta_chair_i_soft_minus_hard"] = soft_metrics["chair_i"] - hard_metrics["chair_i"]
        row["delta_recall_soft_minus_hard"] = soft_metrics["object_recall"] - hard_metrics["object_recall"]
        if hard_metrics.get("bleu4") is not None and soft_metrics.get("bleu4") is not None:
            row["delta_bleu4_soft_minus_hard"] = soft_metrics["bleu4"] - hard_metrics["bleu4"]
        else:
            row["delta_bleu4_soft_minus_hard"] = None
        rows.append(row)
    return rows


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def write_csv(path, rows):
    if not rows:
        with open(path, "w") as f:
            f.write("")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    preferred = [
        "image_id",
        "image",
        "winner",
        "reason",
        "hard_chair_i",
        "soft_chair_i",
        "delta_chair_i_soft_minus_hard",
        "hard_object_recall",
        "soft_object_recall",
        "delta_recall_soft_minus_hard",
        "hard_bleu4",
        "soft_bleu4",
        "delta_bleu4_soft_minus_hard",
        "hard_caption",
        "soft_caption",
        "greedy_caption",
    ]
    ordered = [field for field in preferred if field in fieldnames]
    ordered += [field for field in fieldnames if field not in set(ordered)]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def average(rows, key):
    vals = [row[key] for row in rows if row.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def summarize(rows, top_k):
    counts = Counter(row["winner"] for row in rows)
    summary = {
        "num_samples": len(rows),
        "winner_counts": dict(counts),
        "mean_delta_chair_i_soft_minus_hard": average(rows, "delta_chair_i_soft_minus_hard"),
        "mean_delta_recall_soft_minus_hard": average(rows, "delta_recall_soft_minus_hard"),
        "mean_delta_bleu4_soft_minus_hard": average(rows, "delta_bleu4_soft_minus_hard"),
    }

    soft_wins = [row for row in rows if row["winner"] == "soft_win"]
    soft_wins = sorted(
        soft_wins,
        key=lambda row: (
            row["delta_chair_i_soft_minus_hard"],
            -row["delta_recall_soft_minus_hard"],
            -(row["delta_bleu4_soft_minus_hard"] or 0.0),
        ),
    )
    summary["top_soft_win_image_ids"] = [row["image_id"] for row in soft_wins[:top_k]]
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hard-results", required=True)
    parser.add_argument("--soft-results", required=True)
    parser.add_argument("--greedy-results", default=None)
    parser.add_argument("--annotation-file", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--eps", type=float, default=1e-9)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    hard_results = load_eval_results(args.hard_results)
    soft_results = load_eval_results(args.soft_results)
    greedy_results = load_eval_results(args.greedy_results) if args.greedy_results else None

    add_bleu4(hard_results, args.annotation_file)
    add_bleu4(soft_results, args.annotation_file)
    if greedy_results is not None:
        add_bleu4(greedy_results, args.annotation_file)

    rows = build_rows(greedy_results, hard_results, soft_results, eps=args.eps)
    os.makedirs(args.output_dir, exist_ok=True)

    write_jsonl(os.path.join(args.output_dir, "sample_comparison.jsonl"), rows)
    write_csv(os.path.join(args.output_dir, "sample_comparison.csv"), rows)

    for label in sorted({row["winner"] for row in rows}):
        write_jsonl(
            os.path.join(args.output_dir, f"{label}.jsonl"),
            [row for row in rows if row["winner"] == label],
        )

    summary = summarize(rows, args.top_k)
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

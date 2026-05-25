import argparse
import csv
import json
import os


def safe_int(value, default=None):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def load_sentences(path):
    with open(path) as f:
        data = json.load(f)
    return data.get("sentences", [])


def caption_length(sentence):
    caption = sentence.get("caption", "")
    length = len(caption.split())
    idxs = []
    for key in ("hallucination_idxs", "non_hallucination_idxs"):
        for value in sentence.get(key, []):
            idx = safe_int(value)
            if idx is not None:
                idxs.append(idx)
    if idxs:
        length = max(length, max(idxs) + 1)
    return max(length, 1)


def position_bin(idx, length, split):
    if idx is None:
        return "unknown"
    return "front" if idx < split * length else "back"


def node_pairs(sentence, key):
    out = []
    for item in sentence.get(key, []):
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            out.append((str(item[0]), str(item[1])))
    return out


def summarize_one(path, name, split):
    sentences = load_sentences(path)
    counts = {
        "front_hallucinated": 0,
        "front_grounded": 0,
        "back_hallucinated": 0,
        "back_grounded": 0,
        "unknown_hallucinated": 0,
        "unknown_grounded": 0,
    }
    rows = []
    for sentence in sentences:
        length = caption_length(sentence)
        qid = sentence.get("image_id")
        for idx, (word, node_word) in zip(
            sentence.get("hallucination_idxs", []),
            node_pairs(sentence, "mscoco_hallucinated_words"),
        ):
            idx = safe_int(idx)
            bucket = position_bin(idx, length, split)
            counts[f"{bucket}_hallucinated"] += 1
            rows.append({
                "run": name,
                "question_id": qid,
                "caption_length": length,
                "word_idx": idx,
                "position_bin": bucket,
                "label": "hallucinated",
                "word": word,
                "node_word": node_word,
            })
        for idx, (word, node_word) in zip(
            sentence.get("non_hallucination_idxs", []),
            node_pairs(sentence, "mscoco_non_hallucinated_words"),
        ):
            idx = safe_int(idx)
            bucket = position_bin(idx, length, split)
            counts[f"{bucket}_grounded"] += 1
            rows.append({
                "run": name,
                "question_id": qid,
                "caption_length": length,
                "word_idx": idx,
                "position_bin": bucket,
                "label": "grounded",
                "word": word,
                "node_word": node_word,
            })
    summary = {
        "run": name,
        "eval_results": path,
        "n_sentences": len(sentences),
    }
    for bucket in ("front", "back", "unknown"):
        hall = counts[f"{bucket}_hallucinated"]
        grounded = counts[f"{bucket}_grounded"]
        total = hall + grounded
        summary[f"{bucket}_hallucinated"] = hall
        summary[f"{bucket}_grounded"] = grounded
        summary[f"{bucket}_total_objects"] = total
        summary[f"{bucket}_hallucinated_rate"] = hall / total if total else None
    hall = sum(counts[f"{bucket}_hallucinated"] for bucket in ("front", "back", "unknown"))
    grounded = sum(counts[f"{bucket}_grounded"] for bucket in ("front", "back", "unknown"))
    total = hall + grounded
    summary["hallucinated"] = hall
    summary["grounded"] = grounded
    summary["total_objects"] = total
    summary["hallucinated_rate"] = hall / total if total else None
    return summary, rows


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-results", nargs="+", required=True)
    parser.add_argument("--names", nargs="*", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", type=float, default=0.5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    names = list(args.names)
    if len(names) < len(args.eval_results):
        names.extend(os.path.basename(os.path.dirname(path)) for path in args.eval_results[len(names):])

    summaries = []
    all_rows = []
    for path, name in zip(args.eval_results, names):
        summary, rows = summarize_one(path, name, args.split)
        summaries.append(summary)
        all_rows.extend(rows)

    write_csv(os.path.join(args.output_dir, "hallucination_position_summary.csv"), summaries)
    write_csv(os.path.join(args.output_dir, "hallucination_position_mentions.csv"), all_rows)
    print(json.dumps({
        "output_dir": args.output_dir,
        "summary": summaries,
    }, indent=2))


if __name__ == "__main__":
    main()

import argparse
import csv
import difflib
import json
import os
import re


def tokens(text):
    return re.findall(r"[a-z0-9']+", str(text).lower())


def normalize_text(text):
    return " ".join(tokens(text))


def load_rows(path):
    if path.endswith(".jsonl"):
        rows = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    rows.append({
                        "image_id": item.get("question_id") or item.get("image_id"),
                        "image": item.get("image"),
                        "caption": item.get("text") or item.get("caption") or "",
                        "metrics": {},
                    })
        return rows, {}
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "sentences" in data:
        return data.get("sentences", []), data.get("overall_metrics", {})
    if isinstance(data, list):
        return data, {}
    raise ValueError(f"Unsupported caption file format: {path}")


def row_key(row):
    for key in ("image_id", "question_id", "image"):
        if row.get(key) is not None:
            return str(row.get(key))
    return None


def node_words(pairs):
    output = []
    for item in pairs or []:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            output.append(str(item[1]))
        else:
            output.append(str(item))
    return output


def generated_objects(row):
    words = row.get("mscoco_generated_words")
    if words is not None:
        return [str(word) for word in words]
    return node_words(row.get("mscoco_non_hallucinated_words")) + node_words(row.get("mscoco_hallucinated_words"))


def object_sets(row):
    generated = generated_objects(row)
    grounded = node_words(row.get("mscoco_non_hallucinated_words"))
    hallucinated = node_words(row.get("mscoco_hallucinated_words"))
    gt = [str(item) for item in row.get("mscoco_gt_words") or []]
    return set(generated), set(grounded), set(hallucinated), set(gt)


def chair_s(row):
    metrics = row.get("metrics") or {}
    return int(float(metrics.get("CHAIRs", 0)))


def safe_mean(values):
    values = [float(value) for value in values]
    return sum(values) / len(values) if values else 0.0


def safe_rate(count, total):
    return count / max(total, 1)


def list_string(items):
    return ";".join(sorted(str(item) for item in items))


def compare_rows(base_rows, target_rows):
    base_by_key = {row_key(row): row for row in base_rows if row_key(row) is not None}
    target_by_key = {row_key(row): row for row in target_rows if row_key(row) is not None}
    common = sorted(set(base_by_key) & set(target_by_key))
    output = []
    for key in common:
        base = base_by_key[key]
        target = target_by_key[key]
        base_caption = base.get("caption") or base.get("text") or ""
        target_caption = target.get("caption") or target.get("text") or ""
        base_norm = normalize_text(base_caption)
        target_norm = normalize_text(target_caption)
        base_tokens = tokens(base_caption)
        target_tokens = tokens(target_caption)
        base_token_set = set(base_tokens)
        target_token_set = set(target_tokens)
        seq_ratio = difflib.SequenceMatcher(None, base_norm, target_norm).ratio()
        token_union = base_token_set | target_token_set
        token_jaccard = len(base_token_set & target_token_set) / max(len(token_union), 1)

        base_gen, base_grounded, base_hall, base_gt = object_sets(base)
        target_gen, target_grounded, target_hall, target_gt = object_sets(target)
        base_chair = chair_s(base)
        target_chair = chair_s(target)
        if base_chair > target_chair:
            chair_change = "improved"
        elif base_chair < target_chair:
            chair_change = "worsened"
        else:
            chair_change = "same"

        output.append({
            "key": key,
            "image": base.get("image") or target.get("image"),
            "exact_match": base_caption == target_caption,
            "normalized_exact_match": base_norm == target_norm,
            "sequence_ratio": seq_ratio,
            "token_jaccard": token_jaccard,
            "base_length": len(base_tokens),
            "target_length": len(target_tokens),
            "length_delta": len(target_tokens) - len(base_tokens),
            "base_chair_s": base_chair,
            "target_chair_s": target_chair,
            "chair_change": chair_change,
            "base_generated_n": len(base_gen),
            "target_generated_n": len(target_gen),
            "generated_n_delta": len(target_gen) - len(base_gen),
            "generated_set_changed": base_gen != target_gen,
            "added_generated_objects": list_string(target_gen - base_gen),
            "removed_generated_objects": list_string(base_gen - target_gen),
            "base_grounded_n": len(base_grounded),
            "target_grounded_n": len(target_grounded),
            "grounded_n_delta": len(target_grounded) - len(base_grounded),
            "added_grounded_objects": list_string(target_grounded - base_grounded),
            "removed_grounded_objects": list_string(base_grounded - target_grounded),
            "base_hallucinated_n": len(base_hall),
            "target_hallucinated_n": len(target_hall),
            "hallucinated_n_delta": len(target_hall) - len(base_hall),
            "added_hallucinated_objects": list_string(target_hall - base_hall),
            "removed_hallucinated_objects": list_string(base_hall - target_hall),
            "base_caption": base_caption,
            "target_caption": target_caption,
        })
    return output


def summarize(rows, base_overall, target_overall, base_name, target_name, base_path, target_path):
    n = len(rows)
    improved = [row for row in rows if row["chair_change"] == "improved"]
    worsened = [row for row in rows if row["chair_change"] == "worsened"]
    changed = [row for row in rows if not row["normalized_exact_match"]]
    object_changed = [row for row in rows if row["generated_set_changed"]]
    return [{
        "base_name": base_name,
        "target_name": target_name,
        "base_path": base_path,
        "target_path": target_path,
        "n_common": n,
        "exact_match_rate": safe_rate(sum(1 for row in rows if row["exact_match"]), n),
        "normalized_exact_match_rate": safe_rate(sum(1 for row in rows if row["normalized_exact_match"]), n),
        "caption_changed_rate": safe_rate(len(changed), n),
        "generated_object_set_changed_rate": safe_rate(len(object_changed), n),
        "mean_sequence_ratio": safe_mean(row["sequence_ratio"] for row in rows),
        "mean_token_jaccard": safe_mean(row["token_jaccard"] for row in rows),
        "mean_length_delta": safe_mean(row["length_delta"] for row in rows),
        "base_CHAIRs": base_overall.get("CHAIRs"),
        "target_CHAIRs": target_overall.get("CHAIRs"),
        "delta_CHAIRs": (
            float(target_overall["CHAIRs"]) - float(base_overall["CHAIRs"])
            if "CHAIRs" in base_overall and "CHAIRs" in target_overall else None
        ),
        "base_CHAIRi": base_overall.get("CHAIRi"),
        "target_CHAIRi": target_overall.get("CHAIRi"),
        "delta_CHAIRi": (
            float(target_overall["CHAIRi"]) - float(base_overall["CHAIRi"])
            if "CHAIRi" in base_overall and "CHAIRi" in target_overall else None
        ),
        "chair_sentence_improved_n": len(improved),
        "chair_sentence_improved_rate": safe_rate(len(improved), n),
        "chair_sentence_worsened_n": len(worsened),
        "chair_sentence_worsened_rate": safe_rate(len(worsened), n),
        "mean_generated_n_delta": safe_mean(row["generated_n_delta"] for row in rows),
        "mean_grounded_n_delta": safe_mean(row["grounded_n_delta"] for row in rows),
        "mean_hallucinated_n_delta": safe_mean(row["hallucinated_n_delta"] for row in rows),
    }]


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
    parser.add_argument("--base", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--base-name", default="base")
    parser.add_argument("--target-name", default="target")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-examples", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    base_rows, base_overall = load_rows(args.base)
    target_rows, target_overall = load_rows(args.target)
    pair_rows = compare_rows(base_rows, target_rows)
    summary_rows = summarize(
        pair_rows,
        base_overall,
        target_overall,
        args.base_name,
        args.target_name,
        args.base,
        args.target,
    )

    write_csv(os.path.join(args.output_dir, "caption_pairwise_summary.csv"), summary_rows)
    write_csv(os.path.join(args.output_dir, "caption_pairwise_rows.csv"), pair_rows)

    interesting = sorted(
        pair_rows,
        key=lambda row: (
            row["chair_change"] == "same",
            row["normalized_exact_match"],
            -abs(float(row["hallucinated_n_delta"])),
            -abs(float(row["length_delta"])),
        ),
    )
    write_csv(os.path.join(args.output_dir, "caption_pairwise_examples.csv"), interesting[:args.max_examples])

    print("[summary] caption pairwise")
    for row in summary_rows:
        print(row)
    print("[examples]", os.path.join(args.output_dir, "caption_pairwise_examples.csv"))


if __name__ == "__main__":
    main()

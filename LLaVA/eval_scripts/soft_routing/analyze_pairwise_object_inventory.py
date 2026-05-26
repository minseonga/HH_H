import argparse
import csv
import difflib
import json
import os
import re
from collections import Counter


def normalize_image_id(value):
    if value is None:
        return ""
    text = str(value).strip()
    digits = "".join(ch for ch in text if ch.isdigit())
    if digits:
        return str(int(digits[-12:]))
    return text


def row_key(row):
    for key in ("image_id", "question_id", "image"):
        image_id = normalize_image_id(row.get(key))
        if image_id:
            return image_id
    return ""


def tokens(text):
    return re.findall(r"[a-z0-9']+", str(text).lower())


def normalize_caption(text):
    return " ".join(tokens(text))


def node_pairs(sentence, key):
    pairs = []
    for item in sentence.get(key) or []:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            pairs.append((str(item[0]), str(item[1])))
        else:
            value = str(item)
            pairs.append((value, value))
    return pairs


def node_counter(sentence, key):
    return Counter(node for _, node in node_pairs(sentence, key))


def counter_subtract(left, right):
    out = Counter(left)
    out.subtract(right)
    return Counter({key: value for key, value in out.items() if value > 0})


def counter_intersection(left, right):
    return Counter({key: min(left[key], right[key]) for key in left.keys() & right.keys() if min(left[key], right[key]) > 0})


def counter_total(counter):
    return int(sum(counter.values()))


def counter_to_string(counter):
    items = []
    for key in sorted(counter):
        count = counter[key]
        items.append(f"{key}:{count}" if count > 1 else str(key))
    return ";".join(items)


def set_to_string(items):
    return ";".join(sorted(str(item) for item in items))


def safe_rate(count, total):
    return float(count) / float(total) if total else None


def mean(values):
    values = [float(value) for value in values if value is not None]
    return sum(values) / len(values) if values else None


def load_eval(path):
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, dict) or "sentences" not in data:
        raise ValueError(f"Expected CHAIR eval json with sentences: {path}")
    rows = data.get("sentences") or []
    by_key = {}
    for row in rows:
        key = row_key(row)
        if key:
            by_key[key] = row
    return by_key, data.get("overall_metrics", {})


def caption_text(row):
    return row.get("caption") or row.get("text") or ""


def chair_s(row):
    metrics = row.get("metrics") or {}
    return int(float(metrics.get("CHAIRs", 0)))


def build_pair_row(key, base, target):
    base_caption = caption_text(base)
    target_caption = caption_text(target)
    base_norm = normalize_caption(base_caption)
    target_norm = normalize_caption(target_caption)
    base_tokens = tokens(base_caption)
    target_tokens = tokens(target_caption)
    base_token_set = set(base_tokens)
    target_token_set = set(target_tokens)
    token_union = base_token_set | target_token_set

    base_grounded = node_counter(base, "mscoco_non_hallucinated_words")
    target_grounded = node_counter(target, "mscoco_non_hallucinated_words")
    base_hall = node_counter(base, "mscoco_hallucinated_words")
    target_hall = node_counter(target, "mscoco_hallucinated_words")
    base_generated = base_grounded + base_hall
    target_generated = target_grounded + target_hall

    lost_grounded = counter_subtract(base_grounded, target_grounded)
    gained_grounded = counter_subtract(target_grounded, base_grounded)
    removed_hall = counter_subtract(base_hall, target_hall)
    added_hall = counter_subtract(target_hall, base_hall)
    grounded_to_hall = counter_intersection(base_grounded, target_hall)
    hallucinated_to_grounded = counter_intersection(base_hall, target_grounded)
    removed_generated = counter_subtract(base_generated, target_generated)
    added_generated = counter_subtract(target_generated, base_generated)

    base_grounded_set = set(base_grounded)
    target_grounded_set = set(target_grounded)
    base_hall_set = set(base_hall)
    target_hall_set = set(target_hall)
    base_generated_set = base_grounded_set | base_hall_set
    target_generated_set = target_grounded_set | target_hall_set

    lost_grounded_set = base_grounded_set - target_grounded_set
    gained_grounded_set = target_grounded_set - base_grounded_set
    removed_hall_set = base_hall_set - target_hall_set
    added_hall_set = target_hall_set - base_hall_set
    grounded_to_hall_set = base_grounded_set & target_hall_set
    hallucinated_to_grounded_set = base_hall_set & target_grounded_set

    base_chair = chair_s(base)
    target_chair = chair_s(target)
    if target_chair < base_chair:
        chair_change = "improved"
    elif target_chair > base_chair:
        chair_change = "worsened"
    else:
        chair_change = "same"

    return {
        "image_id": key,
        "image": base.get("image") or target.get("image"),
        "exact_match": base_caption == target_caption,
        "normalized_exact_match": base_norm == target_norm,
        "sequence_ratio": difflib.SequenceMatcher(None, base_norm, target_norm).ratio(),
        "token_jaccard": len(base_token_set & target_token_set) / max(len(token_union), 1),
        "base_length": len(base_tokens),
        "target_length": len(target_tokens),
        "length_delta": len(target_tokens) - len(base_tokens),
        "base_chair_s": base_chair,
        "target_chair_s": target_chair,
        "chair_change": chair_change,
        "base_grounded_mentions": counter_total(base_grounded),
        "target_grounded_mentions": counter_total(target_grounded),
        "lost_grounded_mentions": counter_total(lost_grounded),
        "gained_grounded_mentions": counter_total(gained_grounded),
        "grounded_to_hallucinated_mentions": counter_total(grounded_to_hall),
        "base_hallucinated_mentions": counter_total(base_hall),
        "target_hallucinated_mentions": counter_total(target_hall),
        "removed_hallucinated_mentions": counter_total(removed_hall),
        "added_hallucinated_mentions": counter_total(added_hall),
        "hallucinated_to_grounded_mentions": counter_total(hallucinated_to_grounded),
        "base_generated_mentions": counter_total(base_generated),
        "target_generated_mentions": counter_total(target_generated),
        "removed_generated_mentions": counter_total(removed_generated),
        "added_generated_mentions": counter_total(added_generated),
        "base_grounded_unique": len(base_grounded_set),
        "target_grounded_unique": len(target_grounded_set),
        "lost_grounded_unique": len(lost_grounded_set),
        "gained_grounded_unique": len(gained_grounded_set),
        "grounded_to_hallucinated_unique": len(grounded_to_hall_set),
        "base_hallucinated_unique": len(base_hall_set),
        "target_hallucinated_unique": len(target_hall_set),
        "removed_hallucinated_unique": len(removed_hall_set),
        "added_hallucinated_unique": len(added_hall_set),
        "hallucinated_to_grounded_unique": len(hallucinated_to_grounded_set),
        "base_generated_unique": len(base_generated_set),
        "target_generated_unique": len(target_generated_set),
        "removed_generated_unique": len(base_generated_set - target_generated_set),
        "added_generated_unique": len(target_generated_set - base_generated_set),
        "lost_grounded_objects": counter_to_string(lost_grounded),
        "gained_grounded_objects": counter_to_string(gained_grounded),
        "removed_hallucinated_objects": counter_to_string(removed_hall),
        "added_hallucinated_objects": counter_to_string(added_hall),
        "grounded_to_hallucinated_objects": counter_to_string(grounded_to_hall),
        "hallucinated_to_grounded_objects": counter_to_string(hallucinated_to_grounded),
        "lost_grounded_unique_objects": set_to_string(lost_grounded_set),
        "gained_grounded_unique_objects": set_to_string(gained_grounded_set),
        "removed_hallucinated_unique_objects": set_to_string(removed_hall_set),
        "added_hallucinated_unique_objects": set_to_string(added_hall_set),
        "grounded_to_hallucinated_unique_objects": set_to_string(grounded_to_hall_set),
        "hallucinated_to_grounded_unique_objects": set_to_string(hallucinated_to_grounded_set),
        "base_caption": base_caption,
        "target_caption": target_caption,
    }


def sum_field(rows, field):
    return sum(int(row.get(field) or 0) for row in rows)


def summarize(rows, base_overall, target_overall, base_name, target_name, base_path, target_path):
    n = len(rows)
    base_grounded_mentions = sum_field(rows, "base_grounded_mentions")
    target_grounded_mentions = sum_field(rows, "target_grounded_mentions")
    base_hall_mentions = sum_field(rows, "base_hallucinated_mentions")
    target_hall_mentions = sum_field(rows, "target_hallucinated_mentions")
    base_grounded_unique = sum_field(rows, "base_grounded_unique")
    target_grounded_unique = sum_field(rows, "target_grounded_unique")
    base_hall_unique = sum_field(rows, "base_hallucinated_unique")
    target_hall_unique = sum_field(rows, "target_hallucinated_unique")

    return [{
        "base_name": base_name,
        "target_name": target_name,
        "base_path": base_path,
        "target_path": target_path,
        "n_common": n,
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
        "caption_changed_rate": safe_rate(sum(not row["normalized_exact_match"] for row in rows), n),
        "generated_unique_changed_rate": safe_rate(
            sum(row["removed_generated_unique"] > 0 or row["added_generated_unique"] > 0 for row in rows),
            n,
        ),
        "mean_sequence_ratio": mean(row["sequence_ratio"] for row in rows),
        "mean_token_jaccard": mean(row["token_jaccard"] for row in rows),
        "mean_length_delta": mean(row["length_delta"] for row in rows),
        "chair_sentence_improved_rate": safe_rate(sum(row["chair_change"] == "improved" for row in rows), n),
        "chair_sentence_worsened_rate": safe_rate(sum(row["chair_change"] == "worsened" for row in rows), n),
        "base_grounded_mentions": base_grounded_mentions,
        "target_grounded_mentions": target_grounded_mentions,
        "lost_grounded_mentions": sum_field(rows, "lost_grounded_mentions"),
        "lost_grounded_mention_rate": safe_rate(sum_field(rows, "lost_grounded_mentions"), base_grounded_mentions),
        "gained_grounded_mentions": sum_field(rows, "gained_grounded_mentions"),
        "gained_grounded_mention_rate_vs_target": safe_rate(sum_field(rows, "gained_grounded_mentions"), target_grounded_mentions),
        "grounded_to_hallucinated_mentions": sum_field(rows, "grounded_to_hallucinated_mentions"),
        "grounded_to_hallucinated_mention_rate": safe_rate(sum_field(rows, "grounded_to_hallucinated_mentions"), base_grounded_mentions),
        "base_hallucinated_mentions": base_hall_mentions,
        "target_hallucinated_mentions": target_hall_mentions,
        "removed_hallucinated_mentions": sum_field(rows, "removed_hallucinated_mentions"),
        "hallucination_removal_mention_rate": safe_rate(sum_field(rows, "removed_hallucinated_mentions"), base_hall_mentions),
        "added_hallucinated_mentions": sum_field(rows, "added_hallucinated_mentions"),
        "added_hallucinated_mention_rate_vs_target": safe_rate(sum_field(rows, "added_hallucinated_mentions"), target_hall_mentions),
        "hallucinated_to_grounded_mentions": sum_field(rows, "hallucinated_to_grounded_mentions"),
        "hallucinated_to_grounded_mention_rate": safe_rate(sum_field(rows, "hallucinated_to_grounded_mentions"), base_hall_mentions),
        "base_grounded_unique_sum": base_grounded_unique,
        "target_grounded_unique_sum": target_grounded_unique,
        "lost_grounded_unique_sum": sum_field(rows, "lost_grounded_unique"),
        "lost_grounded_unique_rate": safe_rate(sum_field(rows, "lost_grounded_unique"), base_grounded_unique),
        "gained_grounded_unique_sum": sum_field(rows, "gained_grounded_unique"),
        "gained_grounded_unique_rate_vs_target": safe_rate(sum_field(rows, "gained_grounded_unique"), target_grounded_unique),
        "base_hallucinated_unique_sum": base_hall_unique,
        "target_hallucinated_unique_sum": target_hall_unique,
        "removed_hallucinated_unique_sum": sum_field(rows, "removed_hallucinated_unique"),
        "hallucination_removal_unique_rate": safe_rate(sum_field(rows, "removed_hallucinated_unique"), base_hall_unique),
        "added_hallucinated_unique_sum": sum_field(rows, "added_hallucinated_unique"),
        "added_hallucinated_unique_rate_vs_target": safe_rate(sum_field(rows, "added_hallucinated_unique"), target_hall_unique),
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
    parser.add_argument("--max-examples", type=int, default=80)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    base_by_key, base_overall = load_eval(args.base)
    target_by_key, target_overall = load_eval(args.target)
    common = sorted(set(base_by_key) & set(target_by_key))
    if not common:
        raise ValueError("No common image ids between base and target eval files")

    rows = [build_pair_row(key, base_by_key[key], target_by_key[key]) for key in common]
    summary = summarize(rows, base_overall, target_overall, args.base_name, args.target_name, args.base, args.target)

    write_csv(os.path.join(args.output_dir, "pairwise_object_inventory_summary.csv"), summary)
    write_csv(os.path.join(args.output_dir, "pairwise_object_inventory_rows.csv"), rows)

    examples = sorted(
        rows,
        key=lambda row: (
            row["lost_grounded_mentions"] == 0,
            row["added_hallucinated_mentions"] == 0,
            -row["removed_hallucinated_mentions"],
            -abs(row["length_delta"]),
        ),
    )
    write_csv(os.path.join(args.output_dir, "pairwise_object_inventory_examples.csv"), examples[:args.max_examples])

    print("[summary] pairwise object inventory")
    for row in summary:
        print(row)
    print("[rows]", os.path.join(args.output_dir, "pairwise_object_inventory_rows.csv"))
    print("[examples]", os.path.join(args.output_dir, "pairwise_object_inventory_examples.csv"))


if __name__ == "__main__":
    main()

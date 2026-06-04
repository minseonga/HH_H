#!/usr/bin/env python3
import argparse
import csv
import json
import os


def load_sentences(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "sentences" not in data:
        raise ValueError(f"unsupported eval results file: {path}")
    return data["sentences"], data.get("overall_metrics", {})


def row_key(row):
    for key in ("image_id", "question_id", "image"):
        if row.get(key) is not None:
            return str(row[key])
    return ""


def node_set(items):
    out = set()
    for item in items or []:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            out.add(str(item[1]))
        else:
            out.add(str(item))
    return out


def caption(row):
    return row.get("caption") or row.get("text") or ""


def rate(num, den):
    return float(num) / float(den) if den else 0.0


def summarize(base_rows, target_rows):
    base_by_key = {row_key(row): row for row in base_rows if row_key(row)}
    target_by_key = {row_key(row): row for row in target_rows if row_key(row)}
    rows = []
    totals = {
        "n_common": 0,
        "base_ground_unique": 0,
        "target_ground_unique": 0,
        "preserved_ground_unique": 0,
        "disappeared_ground_unique": 0,
        "added_ground_unique": 0,
        "base_hall_unique": 0,
        "target_hall_unique": 0,
        "removed_hall_unique": 0,
        "retained_hall_unique": 0,
        "added_hall_unique": 0,
        "images_with_ground_unique_added": 0,
        "images_with_ground_unique_disappeared": 0,
        "images_with_hall_unique_removed": 0,
        "images_with_hall_unique_added": 0,
    }

    for key in sorted(set(base_by_key) & set(target_by_key)):
        base = base_by_key[key]
        target = target_by_key[key]
        base_ground = node_set(base.get("mscoco_non_hallucinated_words"))
        target_ground = node_set(target.get("mscoco_non_hallucinated_words"))
        base_hall = node_set(base.get("mscoco_hallucinated_words"))
        target_hall = node_set(target.get("mscoco_hallucinated_words"))

        preserved_ground = base_ground & target_ground
        disappeared_ground = base_ground - target_ground
        added_ground = target_ground - base_ground
        removed_hall = base_hall - target_hall
        retained_hall = base_hall & target_hall
        added_hall = target_hall - base_hall

        row = {
            "key": key,
            "image": base.get("image") or target.get("image"),
            "base_ground_unique": len(base_ground),
            "target_ground_unique": len(target_ground),
            "preserved_ground_unique": len(preserved_ground),
            "disappeared_ground_unique": len(disappeared_ground),
            "added_ground_unique": len(added_ground),
            "base_hall_unique": len(base_hall),
            "target_hall_unique": len(target_hall),
            "removed_hall_unique": len(removed_hall),
            "retained_hall_unique": len(retained_hall),
            "added_hall_unique": len(added_hall),
            "ground_disappeared_words": "; ".join(sorted(disappeared_ground)),
            "ground_added_words": "; ".join(sorted(added_ground)),
            "hall_removed_words": "; ".join(sorted(removed_hall)),
            "hall_added_words": "; ".join(sorted(added_hall)),
            "base_caption": caption(base),
            "target_caption": caption(target),
        }
        rows.append(row)
        totals["n_common"] += 1
        for field in (
            "base_ground_unique",
            "target_ground_unique",
            "preserved_ground_unique",
            "disappeared_ground_unique",
            "added_ground_unique",
            "base_hall_unique",
            "target_hall_unique",
            "removed_hall_unique",
            "retained_hall_unique",
            "added_hall_unique",
        ):
            totals[field] += row[field]
        totals["images_with_ground_unique_added"] += int(row["added_ground_unique"] > 0)
        totals["images_with_ground_unique_disappeared"] += int(row["disappeared_ground_unique"] > 0)
        totals["images_with_hall_unique_removed"] += int(row["removed_hall_unique"] > 0)
        totals["images_with_hall_unique_added"] += int(row["added_hall_unique"] > 0)

    n = totals["n_common"]
    base_target_total = totals["target_ground_unique"] + totals["target_hall_unique"]
    base_total = totals["base_ground_unique"] + totals["base_hall_unique"]
    summary = dict(totals)
    summary.update(
        {
            "mean_base_ground_unique_per_image": rate(totals["base_ground_unique"], n),
            "mean_target_ground_unique_per_image": rate(totals["target_ground_unique"], n),
            "mean_added_ground_unique_per_image": rate(totals["added_ground_unique"], n),
            "mean_disappeared_ground_unique_per_image": rate(totals["disappeared_ground_unique"], n),
            "net_ground_unique_change": totals["target_ground_unique"] - totals["base_ground_unique"],
            "net_ground_unique_change_per_image": rate(totals["target_ground_unique"] - totals["base_ground_unique"], n),
            "ground_unique_recall": rate(totals["preserved_ground_unique"], totals["base_ground_unique"]),
            "ground_unique_disappear_rate": rate(totals["disappeared_ground_unique"], totals["base_ground_unique"]),
            "target_unique_object_precision": rate(totals["target_ground_unique"], base_target_total),
            "base_unique_object_precision": rate(totals["base_ground_unique"], base_total),
            "hall_unique_removal_rate": rate(totals["removed_hall_unique"], totals["base_hall_unique"]),
            "hall_unique_added_per_image": rate(totals["added_hall_unique"], n),
            "images_with_ground_unique_added_rate": rate(totals["images_with_ground_unique_added"], n),
            "images_with_ground_unique_disappeared_rate": rate(totals["images_with_ground_unique_disappeared"], n),
            "images_with_hall_unique_removed_rate": rate(totals["images_with_hall_unique_removed"], n),
            "images_with_hall_unique_added_rate": rate(totals["images_with_hall_unique_added"], n),
        }
    )
    return rows, summary


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--base-name", default="greedy")
    parser.add_argument("--target-name", default="target")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    base_rows, base_metrics = load_sentences(args.base)
    target_rows, target_metrics = load_sentences(args.target)
    rows, summary = summarize(base_rows, target_rows)
    summary.update(
        {
            "base": args.base,
            "target": args.target,
            "base_name": args.base_name,
            "target_name": args.target_name,
            "base_metrics": base_metrics,
            "target_metrics": target_metrics,
        }
    )
    write_csv(os.path.join(args.output_dir, "unique_object_node_transition_rows.csv"), rows)
    write_csv(os.path.join(args.output_dir, "unique_object_node_transition_summary.csv"), [summary])
    with open(os.path.join(args.output_dir, "unique_object_node_transition_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

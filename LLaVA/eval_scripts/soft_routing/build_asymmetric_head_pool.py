import argparse
import csv
import json
import math
import os


def safe_float(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def parse_head_key(key):
    layer, head = str(key).split(":")
    return int(layer), int(head)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-summary", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--sort-key", default="auroc_high_text_mass")
    parser.add_argument("--secondary-key", default="hall_over_grounded_text_mass")
    parser.add_argument("--min-layer", type=int, default=0)
    parser.add_argument("--max-layer", type=int, default=10_000)
    parser.add_argument("--min-auroc", type=float, default=0.0)
    parser.add_argument("--min-ratio", type=float, default=0.0)
    parser.add_argument("--exclude-adhh", action="store_true", default=False)
    args = parser.parse_args()

    rows = []
    with open(args.head_summary) as f:
        for row in csv.DictReader(f):
            layer = int(row["layer"])
            if layer < args.min_layer or layer > args.max_layer:
                continue
            if args.exclude_adhh and str(row.get("in_adhh_top20", "0")) == "1":
                continue
            auroc = safe_float(row.get("auroc_high_text_mass"), 0.0)
            ratio = safe_float(row.get("hall_over_grounded_text_mass"), 0.0)
            if auroc < args.min_auroc or ratio < args.min_ratio:
                continue
            sort_value = safe_float(row.get(args.sort_key), 0.0)
            secondary_value = safe_float(row.get(args.secondary_key), 0.0)
            row["_sort_value"] = sort_value
            row["_secondary_value"] = secondary_value
            rows.append(row)

    rows.sort(
        key=lambda row: (
            -(row["_sort_value"] or 0.0),
            -(row["_secondary_value"] or 0.0),
            int(row["layer"]),
            int(row["head"]),
        )
    )
    selected = rows[:args.top_k]
    heads = [[int(row["layer"]), int(row["head"])] for row in selected]
    score_items = []
    for rank, row in enumerate(selected, start=1):
        score_items.append({
            "layer": int(row["layer"]),
            "head": int(row["head"]),
            "head_key": row["head_key"],
            "rank": rank,
            "score": safe_float(row.get(args.sort_key), 0.0),
            "auroc_high_text_mass": safe_float(row.get("auroc_high_text_mass"), None),
            "hall_over_grounded_text_mass": safe_float(row.get("hall_over_grounded_text_mass"), None),
            "hall_minus_grounded_text_mass": safe_float(row.get("hall_minus_grounded_text_mass"), None),
            "hall_mean_text_mass": safe_float(row.get("hall_mean_text_mass"), None),
            "grounded_mean_text_mass": safe_float(row.get("grounded_mean_text_mass"), None),
            "in_adhh_top20": int(row.get("in_adhh_top20") or 0),
            "adhh_rank": row.get("adhh_rank", ""),
        })

    payload = {
        "hal_heads": heads,
        "contrastive_scores": score_items,
        "metadata": {
            "source": "head_text_mass_asymmetry",
            "head_summary": args.head_summary,
            "top_k": args.top_k,
            "sort_key": args.sort_key,
            "secondary_key": args.secondary_key,
            "min_layer": args.min_layer,
            "max_layer": args.max_layer,
            "min_auroc": args.min_auroc,
            "min_ratio": args.min_ratio,
            "exclude_adhh": args.exclude_adhh,
            "n_candidates": len(rows),
            "n_selected": len(selected),
        },
    }
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(payload, f, indent=2)
    csv_path = os.path.splitext(args.output_path)[0] + ".csv"
    if score_items:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(score_items[0].keys()))
            writer.writeheader()
            writer.writerows(score_items)
    print(json.dumps(payload["metadata"], indent=2))
    print(args.output_path)


if __name__ == "__main__":
    main()

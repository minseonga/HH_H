import argparse
import csv
import json
import os

from eval_scripts.soft_routing.head_prior_utils import head_key, parse_head_key


def parse_float(value, default=0.0):
    if value is None or value == "":
        return default
    return float(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--correlations-csv", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--score-field", default="pearson")
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument("--positive-only", action="store_true", default=True)
    parser.add_argument("--allow-negative", action="store_false", dest="positive_only")
    args = parser.parse_args()

    rows = []
    with open(args.correlations_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("target") and row["target"] != "head":
                continue
            if not row.get("target_head_key"):
                continue
            if args.score_field not in row:
                raise ValueError(f"Missing score field {args.score_field} in {args.correlations_csv}")
            score = parse_float(row.get(args.score_field))
            if args.positive_only and score <= 0:
                continue
            if args.min_score is not None and score < args.min_score:
                continue
            layer, head = parse_head_key(row["target_head_key"])
            rows.append({
                "layer": int(layer),
                "head": int(head),
                "head_key": head_key(layer, head),
                "score": float(score),
                "pearson": parse_float(row.get("pearson")),
                "pearson_abs": parse_float(row.get("pearson_abs")),
                "n": int(float(row.get("n", 0) or 0)),
                "mean_source": parse_float(row.get("mean_source")),
                "mean_target": parse_float(row.get("mean_target")),
                "feature": row.get("feature", ""),
                "source_aggregate": row.get("source_aggregate", ""),
            })

    dedup = {}
    for row in rows:
        key = row["head_key"]
        if key not in dedup or row["score"] > dedup[key]["score"]:
            dedup[key] = row

    sorted_items = sorted(dedup.values(), key=lambda item: item["score"], reverse=True)
    selected = sorted_items[:args.top_k]
    output = {
        "hal_heads": [[item["layer"], item["head"]] for item in selected],
        "hal_head_scores": selected,
        "contrastive_scores": selected,
        "score_sorted_heads": [[item["layer"], item["head"]] for item in sorted_items],
        "score_sorted_head_scores": sorted_items,
        "source": args.correlations_csv,
        "score_source": args.score_field,
        "top_k": args.top_k,
        "min_score": args.min_score,
        "positive_only": args.positive_only,
        "description": "Candidate heads ranked by correlation with early unsupported-component source heads.",
    }

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2)

    print("wrote:", args.output_path)
    print("score source:", args.score_field)
    print("num selected:", len(selected))
    if selected:
        print("score range:", selected[-1]["score"], selected[0]["score"])
        print("top heads:")
        for item in selected[:10]:
            print(item)


if __name__ == "__main__":
    main()

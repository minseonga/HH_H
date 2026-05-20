import argparse
import csv
import json
import os

from eval_scripts.soft_routing.head_prior_utils import head_key, parse_head_key


def parse_float(value, default=0.0):
    if value is None or value == "":
        return default
    return float(value)


def row_layer_head(row):
    if row.get("head_key"):
        return parse_head_key(row["head_key"])
    return int(row["layer"]), int(row["head"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overlap-summary", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--score-field", default="overlap_score")
    parser.add_argument("--min-score", type=float, default=None)
    args = parser.parse_args()

    rows = []
    with open(args.overlap_summary, newline="") as f:
        for row in csv.DictReader(f):
            if args.score_field not in row:
                raise ValueError(f"Missing score field {args.score_field} in {args.overlap_summary}")
            layer, head = row_layer_head(row)
            score = parse_float(row.get(args.score_field))
            if args.min_score is not None and score < args.min_score:
                continue
            rows.append({
                "layer": int(layer),
                "head": int(head),
                "score": float(score),
                "head_key": head_key(layer, head),
                "unsupported_text_value_norm": parse_float(row.get("unsupported_text_value_norm")),
                "should_suppress_minus_safe_unsupported_text_value_norm": parse_float(
                    row.get("should_suppress_minus_safe_unsupported_text_value_norm")
                ),
                "unsupported_norm01": parse_float(row.get("unsupported_norm01")),
                "positive_contrast_norm01": parse_float(row.get("positive_contrast_norm01")),
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
        "source": args.overlap_summary,
        "score_source": args.score_field,
        "top_k": args.top_k,
        "min_score": args.min_score,
        "description": "Candidate heads ranked by unsupported-value and positive-utility-overlap diagnostics.",
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

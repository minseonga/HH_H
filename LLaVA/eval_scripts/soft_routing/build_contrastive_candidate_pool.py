import argparse
import json
import os

from eval_scripts.soft_routing.head_prior_utils import head_key


def score_items_from_field(raw_scores):
    items = []
    if isinstance(raw_scores, dict):
        for key, score in raw_scores.items():
            layer, head = key.split(":")
            items.append({"layer": int(layer), "head": int(head), "score": float(score)})
    else:
        for item in raw_scores:
            if isinstance(item, dict):
                items.append({
                    "layer": int(item["layer"]),
                    "head": int(item["head"]),
                    "score": float(item["score"]),
                })
            else:
                items.append({"layer": int(item[0]), "head": int(item[1]), "score": float(item[2])})
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attribution-result", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--positive-only", action="store_true", default=False)
    args = parser.parse_args()

    with open(args.attribution_result, "r") as f:
        data = json.load(f)

    score_source = None
    raw_scores = None
    for field in ("contrastive_scores", "hal_head_scores"):
        if field in data:
            score_source = field
            raw_scores = data[field]
            break
    if raw_scores is None:
        raise ValueError(f"No contrastive_scores or hal_head_scores in {args.attribution_result}")

    dedup = {}
    for item in score_items_from_field(raw_scores):
        key = head_key(item["layer"], item["head"])
        if args.positive_only and item["score"] <= 0:
            continue
        dedup[key] = item
    sorted_items = sorted(dedup.values(), key=lambda item: item["score"], reverse=True)
    selected = sorted_items[:args.top_k]

    output = {
        "hal_heads": [[item["layer"], item["head"]] for item in selected],
        "hal_head_scores": selected,
        "contrastive_scores": selected,
        "score_sorted_heads": [[item["layer"], item["head"]] for item in sorted_items],
        "score_sorted_head_scores": sorted_items,
        "source": args.attribution_result,
        "score_source": score_source,
        "top_k": args.top_k,
        "positive_only": args.positive_only,
    }

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2)

    print("wrote:", args.output_path)
    print("score source:", score_source)
    print("num selected:", len(selected))
    if selected:
        print("score range:", selected[-1]["score"], selected[0]["score"])
        print("top heads:")
        for item in selected[:10]:
            print(item)


if __name__ == "__main__":
    main()

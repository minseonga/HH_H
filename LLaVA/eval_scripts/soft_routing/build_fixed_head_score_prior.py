import argparse
import json
import os

from eval_scripts.soft_routing.head_prior_utils import head_key, load_head_priors


def load_score_items(path):
    with open(path, "r") as f:
        data = json.load(f)
    raw_scores = data.get("contrastive_scores") or data.get("hal_head_scores")
    if not raw_scores:
        raise ValueError(f"No contrastive_scores or hal_head_scores found in {path}")

    score_map = {}
    for item in raw_scores:
        if isinstance(item, dict):
            key = head_key(item["layer"], item["head"])
            score_map[key] = float(item["score"])
        else:
            key = head_key(item[0], item[1])
            score_map[key] = float(item[2])
    return score_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-set-path", required=True)
    parser.add_argument("--score-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    heads, _, head_source = load_head_priors(args.head_set_path, top_k=args.top_k, prior_mode="rank")
    score_map = load_score_items(args.score_path)

    fixed_scores = []
    missing = []
    for layer_idx, head_idx in heads:
        key = head_key(layer_idx, head_idx)
        if key not in score_map:
            missing.append(key)
        fixed_scores.append({
            "layer": int(layer_idx),
            "head": int(head_idx),
            "score": float(score_map.get(key, 0.0)),
        })

    sorted_scores = sorted(fixed_scores, key=lambda item: item["score"], reverse=True)
    output = {
        "hal_heads": [[item["layer"], item["head"]] for item in fixed_scores],
        "hal_head_scores": fixed_scores,
        "contrastive_scores": fixed_scores,
        "score_sorted_heads": [[item["layer"], item["head"]] for item in sorted_scores],
        "score_sorted_head_scores": sorted_scores,
        "head_set_source": args.head_set_path,
        "head_set_prior_source": head_source,
        "score_source": args.score_path,
        "missing_score_heads": missing,
    }

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2)

    print("wrote:", args.output_path)
    print("num heads:", len(fixed_scores))
    print("missing score heads:", missing)
    print("score range:", min(item["score"] for item in fixed_scores), max(item["score"] for item in fixed_scores))
    print("top fixed-head scores:")
    for item in sorted_scores[:10]:
        print(item)


if __name__ == "__main__":
    main()

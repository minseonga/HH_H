import argparse
import json
import os
from collections import defaultdict

import numpy as np

from eval_scripts.soft_routing.head_prior_utils import head_key, load_head_priors


def load_allowed_heads(path, top_k):
    if not path:
        return None
    heads, _, _ = load_head_priors(path, top_k=top_k, prior_mode="uniform")
    return {head_key(*head) for head in heads}


def percentile(values, q):
    if not values:
        return 0.0
    return float(np.percentile(np.array(values, dtype=float), q))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records-jsonl", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--head-path", default="")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--norm-field", default="text_value_norm")
    parser.add_argument("--q-threshold", type=float, default=75.0)
    parser.add_argument("--q-low", type=float, default=50.0)
    parser.add_argument("--q-high", type=float, default=90.0)
    args = parser.parse_args()

    allowed = load_allowed_heads(args.head_path, args.top_k)
    values_by_head = defaultdict(list)
    with open(args.records_jsonl, "r") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            key = record.get("head_key") or head_key(record["layer"], record["head"])
            if allowed is not None and key not in allowed:
                continue
            values_by_head[key].append(float(record.get(args.norm_field, 0.0)))

    thresholds = {}
    for key, values in sorted(values_by_head.items()):
        thresholds[key] = {
            "threshold": percentile(values, args.q_threshold),
            "low": percentile(values, args.q_low),
            "high": percentile(values, args.q_high),
            "n": len(values),
        }

    output = {
        "records_jsonl": args.records_jsonl,
        "head_path": args.head_path,
        "top_k": args.top_k,
        "norm_field": args.norm_field,
        "q_threshold": args.q_threshold,
        "q_low": args.q_low,
        "q_high": args.q_high,
        "head_norm_thresholds": thresholds,
    }

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2)

    print("wrote:", args.output_path)
    print("num heads:", len(thresholds))
    if thresholds:
        flat = [item["threshold"] for item in thresholds.values()]
        print("threshold range:", min(flat), max(flat))


if __name__ == "__main__":
    main()

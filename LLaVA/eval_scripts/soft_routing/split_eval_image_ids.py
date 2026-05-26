import argparse
import json
import os
import random

from eval_scripts.soft_routing.analyze_query_direction_chair_alignment import image_id_key


def load_sentences(path):
    with open(path) as f:
        data = json.load(f)
    return data.get("sentences", data if isinstance(data, list) else []), data


def write_ids(path, ids):
    with open(path, "w") as f:
        for image_id in ids:
            f.write(f"{image_id}\n")


def write_eval_subset(path, sentences, image_ids):
    keep = set(image_ids)
    subset = [item for item in sentences if image_id_key(item) in keep]
    with open(path, "w") as f:
        json.dump({"sentences": subset, "overall_metrics": {}}, f, indent=2)
    return subset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-results", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--discovery-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    sentences, _ = load_sentences(args.eval_results)
    by_id = {}
    for sentence in sentences:
        image_id = image_id_key(sentence)
        if image_id and image_id not in by_id:
            by_id[image_id] = sentence
    image_ids = sorted(by_id)
    rng = random.Random(args.seed)
    rng.shuffle(image_ids)
    if args.n > 0:
        image_ids = image_ids[:args.n]
    if len(image_ids) < 2:
        raise ValueError(f"Need at least two image ids, got {len(image_ids)}")

    n_discovery = int(round(len(image_ids) * args.discovery_fraction))
    n_discovery = min(max(n_discovery, 1), len(image_ids) - 1)
    discovery_ids = sorted(image_ids[:n_discovery])
    heldout_ids = sorted(image_ids[n_discovery:])

    write_ids(os.path.join(args.output_dir, "all_image_ids.txt"), sorted(image_ids))
    write_ids(os.path.join(args.output_dir, "discovery_image_ids.txt"), discovery_ids)
    write_ids(os.path.join(args.output_dir, "heldout_image_ids.txt"), heldout_ids)
    discovery_subset = write_eval_subset(
        os.path.join(args.output_dir, "discovery_eval_results.json"),
        sentences,
        discovery_ids,
    )
    heldout_subset = write_eval_subset(
        os.path.join(args.output_dir, "heldout_eval_results.json"),
        sentences,
        heldout_ids,
    )
    summary = {
        "eval_results": args.eval_results,
        "n_requested": args.n,
        "n_all": len(image_ids),
        "n_discovery": len(discovery_ids),
        "n_heldout": len(heldout_ids),
        "seed": args.seed,
        "discovery_fraction": args.discovery_fraction,
        "n_discovery_sentences": len(discovery_subset),
        "n_heldout_sentences": len(heldout_subset),
        "discovery_examples": discovery_ids[:20],
        "heldout_examples": heldout_ids[:20],
    }
    with open(os.path.join(args.output_dir, "split_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

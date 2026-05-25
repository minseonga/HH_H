import argparse
import csv
import json
import os
import re

import numpy as np


def normalize_image_id(value):
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    basename = os.path.basename(text)
    match = re.search(r"COCO_(?:train|val)2014_(\d{12})\.jpg$", basename)
    if match:
        return str(int(match.group(1)))
    match = re.search(r"(\d{12})\.jpg$", basename)
    if match:
        return str(int(match.group(1)))
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def read_csv_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_json_sentences(path):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("sentences", [])
    if isinstance(data, list):
        return data
    return []


def ids_from_eval_results(path):
    ids = set()
    for row in load_json_sentences(path):
        image_id = row.get("image_id", row.get("question_id", row.get("image")))
        normalized = normalize_image_id(image_id)
        if normalized:
            ids.add(normalized)
    return ids


def ids_from_alignment_samples(path):
    ids = set()
    for row in read_csv_rows(path):
        image_id = row.get("image_id") or row.get("question_id") or row.get("image")
        normalized = normalize_image_id(image_id)
        if normalized:
            ids.add(normalized)
    return ids


def split_probe_ids(step_rows, test_fraction, seed):
    image_ids = sorted({
        normalize_image_id(row.get("image_id") or row.get("question_id") or row.get("image"))
        for row in step_rows
    })
    image_ids = [item for item in image_ids if item]
    rng = np.random.default_rng(seed)
    rng.shuffle(image_ids)
    if len(image_ids) <= 1 or test_fraction <= 0:
        test_ids = set()
    else:
        n_test = int(round(len(image_ids) * test_fraction))
        n_test = min(max(n_test, 1), len(image_ids) - 1)
        test_ids = set(image_ids[:n_test])
    all_ids = set(image_ids)
    train_ids = all_ids - test_ids
    return all_ids, train_ids, test_ids


def rate(numerator, denominator):
    return numerator / denominator if denominator else None


def sample_items(values, limit=20):
    return sorted(values)[:limit]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-probe-dir", default="")
    parser.add_argument("--probe-steps", default="")
    parser.add_argument("--eval-results", default="")
    parser.add_argument("--alignment-samples", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--test-fraction", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    probe_steps = args.probe_steps
    if not probe_steps:
        if not args.query_probe_dir:
            raise ValueError("Set --query-probe-dir or --probe-steps")
        probe_steps = os.path.join(args.query_probe_dir, "query_direction_steps.csv")
    if not os.path.exists(probe_steps):
        raise FileNotFoundError(probe_steps)

    test_fraction = args.test_fraction
    seed = args.seed
    summary_path = ""
    if args.query_probe_dir:
        summary_path = os.path.join(args.query_probe_dir, "query_direction_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                summary = json.load(f)
            if test_fraction is None:
                test_fraction = float(summary.get("test_fraction", 0.3))
            if seed is None:
                seed = int(summary.get("seed", 42))
    if test_fraction is None:
        test_fraction = 0.3
    if seed is None:
        seed = 42

    eval_ids = set()
    eval_sources = []
    if args.eval_results:
        eval_ids |= ids_from_eval_results(args.eval_results)
        eval_sources.append(args.eval_results)
    if args.alignment_samples:
        eval_ids |= ids_from_alignment_samples(args.alignment_samples)
        eval_sources.append(args.alignment_samples)
    if not eval_ids:
        raise ValueError("Set --eval-results and/or --alignment-samples")

    step_rows = read_csv_rows(probe_steps)
    probe_all, probe_train, probe_test = split_probe_ids(step_rows, test_fraction, seed)
    overlap_all = eval_ids & probe_all
    overlap_train = eval_ids & probe_train
    overlap_test = eval_ids & probe_test

    out = {
        "probe_steps": probe_steps,
        "probe_summary": summary_path if summary_path and os.path.exists(summary_path) else "",
        "eval_sources": eval_sources,
        "test_fraction": test_fraction,
        "seed": seed,
        "n_probe_steps": len(step_rows),
        "n_probe_images_all": len(probe_all),
        "n_probe_images_train": len(probe_train),
        "n_probe_images_test": len(probe_test),
        "n_eval_images": len(eval_ids),
        "n_overlap_all": len(overlap_all),
        "n_overlap_train": len(overlap_train),
        "n_overlap_test": len(overlap_test),
        "overlap_all_rate_vs_eval": rate(len(overlap_all), len(eval_ids)),
        "overlap_train_rate_vs_eval": rate(len(overlap_train), len(eval_ids)),
        "overlap_test_rate_vs_eval": rate(len(overlap_test), len(eval_ids)),
        "overlap_all_rate_vs_probe": rate(len(overlap_all), len(probe_all)),
        "overlap_train_rate_vs_probe_train": rate(len(overlap_train), len(probe_train)),
        "overlap_test_rate_vs_probe_test": rate(len(overlap_test), len(probe_test)),
        "overlap_all_examples": sample_items(overlap_all),
        "overlap_train_examples": sample_items(overlap_train),
        "overlap_test_examples": sample_items(overlap_test),
    }

    if args.output_json:
        output_dir = os.path.dirname(args.output_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()


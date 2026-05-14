import argparse
import csv
import json
import os
from collections import Counter, defaultdict


FEATURES_TO_BIN = [
    "diag_max_margin",
    "diag_mean_margin",
    "diag_triggered_head_frac",
    "diag_max_text_mass",
    "diag_mean_text_mass",
    "diag_max_soft_alpha",
    "diag_mean_soft_alpha",
    "original_entropy",
    "original_top1_top2_margin",
    "kl_original_to_hard",
    "kl_original_to_soft",
]


def load_records(path):
    records = []
    with open(path, "r") as f:
        for line in f:
            item = json.loads(line)
            if item.get("has_divergence", False):
                records.append(item)
    return records


def clean_token(token):
    return token.replace("\n", "\\n")


def token_key(item):
    return (
        item.get("original_next_token", ""),
        item.get("hard_next_token", ""),
        item.get("soft_next_token", ""),
    )


def record_row(item):
    return {
        "image_id": item.get("image_id"),
        "winner": item.get("winner"),
        "reason": item.get("reason"),
        "case_file": item.get("case_file"),
        "divergence_step": item.get("divergence_step"),
        "original_next_token": clean_token(item.get("original_next_token", "")),
        "hard_next_token": clean_token(item.get("hard_next_token", "")),
        "soft_next_token": clean_token(item.get("soft_next_token", "")),
        "original_next_token_type": item.get("original_next_token_type"),
        "hard_next_token_type": item.get("hard_next_token_type"),
        "soft_next_token_type": item.get("soft_next_token_type"),
        "hard_equals_original": item.get("hard_next_token_id") == item.get("original_next_token_id"),
        "soft_equals_original": item.get("soft_next_token_id") == item.get("original_next_token_id"),
        "hard_equals_soft": item.get("hard_next_token_id") == item.get("soft_next_token_id"),
        "diag_max_margin": item.get("diag_max_margin"),
        "diag_mean_margin": item.get("diag_mean_margin"),
        "diag_triggered_head_count": item.get("diag_triggered_head_count"),
        "diag_triggered_head_frac": item.get("diag_triggered_head_frac"),
        "diag_max_soft_alpha": item.get("diag_max_soft_alpha"),
        "diag_mean_soft_alpha": item.get("diag_mean_soft_alpha"),
        "original_entropy": item.get("original_entropy"),
        "original_top1_top2_margin": item.get("original_top1_top2_margin"),
        "kl_original_to_hard": item.get("kl_original_to_hard"),
        "kl_original_to_soft": item.get("kl_original_to_soft"),
        "prefix_text": item.get("prefix_text", ""),
        "hard_caption": item.get("hard_caption", ""),
        "soft_caption": item.get("soft_caption", ""),
    }


def write_csv(path, rows):
    if not rows:
        with open(path, "w") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def count_rows(counter, columns):
    rows = []
    for key, count in counter.most_common():
        if not isinstance(key, tuple):
            key = (key,)
        row = {col: val for col, val in zip(columns, key)}
        row["count"] = count
        rows.append(row)
    return rows


def numeric(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def make_bins(values, num_bins):
    values = sorted(v for v in values if v is not None)
    if not values:
        return []
    edges = []
    for i in range(1, num_bins):
        idx = round((len(values) - 1) * i / num_bins)
        edges.append(values[idx])
    return edges


def assign_bin(value, edges):
    if value is None:
        return "missing"
    lo = "-inf"
    for edge in edges:
        if value <= edge:
            return f"({lo},{edge:.4g}]"
        lo = f"{edge:.4g}"
    return f"({lo},inf)"


def feature_bin_rows(records, num_bins):
    rows = []
    for feature in FEATURES_TO_BIN:
        vals = [numeric(item.get(feature)) for item in records]
        edges = make_bins(vals, num_bins)
        groups = defaultdict(Counter)
        for item in records:
            label = item.get("winner") or item.get("case_file", "unknown")
            bin_name = assign_bin(numeric(item.get(feature)), edges)
            groups[bin_name][label] += 1
        for bin_name, counter in groups.items():
            total = sum(counter.values())
            row = {
                "feature": feature,
                "bin": bin_name,
                "total": total,
            }
            for label, count in sorted(counter.items()):
                row[label] = count
                row[f"{label}_rate"] = count / total if total else 0.0
            rows.append(row)
    return rows


def summarize(records, top_k):
    winner_counts = Counter(item.get("winner") or item.get("case_file", "unknown") for item in records)
    original_type = Counter((item.get("winner"), item.get("original_next_token_type")) for item in records)
    hard_type = Counter((item.get("winner"), item.get("hard_next_token_type")) for item in records)
    soft_type = Counter((item.get("winner"), item.get("soft_next_token_type")) for item in records)
    pair_counts = Counter(token_key(item) for item in records)

    hard_keeps_original = Counter()
    soft_keeps_original = Counter()
    for item in records:
        winner = item.get("winner") or item.get("case_file", "unknown")
        if item.get("hard_next_token_id") == item.get("original_next_token_id"):
            hard_keeps_original[winner] += 1
        if item.get("soft_next_token_id") == item.get("original_next_token_id"):
            soft_keeps_original[winner] += 1

    return {
        "num_records": len(records),
        "winner_counts": dict(winner_counts),
        "hard_keeps_original_counts": dict(hard_keeps_original),
        "soft_keeps_original_counts": dict(soft_keeps_original),
        "original_token_type_by_winner": count_rows(original_type, ["winner", "token_type"]),
        "hard_token_type_by_winner": count_rows(hard_type, ["winner", "token_type"]),
        "soft_token_type_by_winner": count_rows(soft_type, ["winner", "token_type"]),
        "top_token_triples": count_rows(pair_counts, ["original_token", "hard_token", "soft_token"])[:top_k],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--num-bins", type=int, default=5)
    args = parser.parse_args()

    records = load_records(args.diagnostics_jsonl)
    os.makedirs(args.output_dir, exist_ok=True)

    rows = [record_row(item) for item in records]
    write_csv(os.path.join(args.output_dir, "token_level_records.csv"), rows)

    summary = summarize(records, args.top_k)
    with open(os.path.join(args.output_dir, "token_level_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    write_csv(
        os.path.join(args.output_dir, "token_triples.csv"),
        summary["top_token_triples"],
    )
    write_csv(
        os.path.join(args.output_dir, "original_token_type_by_winner.csv"),
        summary["original_token_type_by_winner"],
    )
    write_csv(
        os.path.join(args.output_dir, "hard_token_type_by_winner.csv"),
        summary["hard_token_type_by_winner"],
    )
    write_csv(
        os.path.join(args.output_dir, "soft_token_type_by_winner.csv"),
        summary["soft_token_type_by_winner"],
    )
    write_csv(
        os.path.join(args.output_dir, "feature_bins_by_winner.csv"),
        feature_bin_rows(records, args.num_bins),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

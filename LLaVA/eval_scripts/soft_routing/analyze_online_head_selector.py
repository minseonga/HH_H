import argparse
import csv
import json
import math
import os
from collections import OrderedDict, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eval_scripts.soft_routing.head_prior_utils import head_key


RAW_FEATURES = [
    "text_mass",
    "img_mass",
    "text_ratio",
    "img_entropy_norm",
    "text_ratio_img_entropy",
    "text_img_log_ratio",
    "text_value_norm",
    "img_value_norm",
    "visual_mass_ratio",
    "visual_value_ratio",
    "question_attention",
    "output_attention",
    "recent_output_attention",
    "recent_output_ratio",
    "question_value_norm",
    "output_value_norm",
    "recent_output_value_norm",
]


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
                    "score": float(item.get("score", 0.0)),
                })
            else:
                items.append({"layer": int(item[0]), "head": int(item[1]), "score": float(item[2])})
    return items


def load_teacher_scores(path):
    with open(path, "r") as f:
        data = json.load(f)
    raw_scores = None
    for field in ("score_sorted_head_scores", "contrastive_scores", "hal_head_scores"):
        if field in data:
            raw_scores = data[field]
            break
    if raw_scores is None:
        heads = data.get("hal_heads", [])
        return {head_key(layer, head): float(len(heads) - idx) for idx, (layer, head) in enumerate(heads)}
    items = score_items_from_field(raw_scores)
    return {head_key(item["layer"], item["head"]): float(item["score"]) for item in items}


def sample_key(record):
    for field in ("image_id", "question_id", "image"):
        if field in record and record[field] is not None:
            return str(record[field])
    return "unknown"


def load_records(path, max_samples=0, min_layer=None, max_layer=None):
    samples = OrderedDict()
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            layer = int(record.get("layer", -1))
            if min_layer is not None and layer < min_layer:
                continue
            if max_layer is not None and layer > max_layer:
                continue
            sid = sample_key(record)
            if sid not in samples:
                if max_samples and max_samples > 0 and len(samples) >= max_samples:
                    continue
                samples[sid] = {
                    "sample_id": sid,
                    "image": record.get("image", ""),
                    "records": [],
                }
            if sid in samples:
                key = record.get("head_key") or head_key(record["layer"], record["head"])
                row = {
                    "sample_id": sid,
                    "image": record.get("image", samples[sid].get("image", "")),
                    "layer": layer,
                    "head": int(record.get("head", -1)),
                    "head_key": key,
                }
                for feature in RAW_FEATURES:
                    row[feature] = safe_float(record.get(feature, 0.0))
                samples[sid]["records"].append(row)
    return list(samples.values())


def safe_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(value):
        return 0.0
    return value


def add_normalized_features(samples):
    feature_names = list(RAW_FEATURES)
    for sample in samples:
        records = sample["records"]
        by_layer = defaultdict(list)
        for record in records:
            by_layer[int(record["layer"])].append(record)

        for feature in feature_names:
            values = np.array([record[feature] for record in records], dtype=float)
            add_minmax(records, feature, values, f"sample_norm_{feature}")

        for layer_records in by_layer.values():
            for feature in feature_names:
                values = np.array([record[feature] for record in layer_records], dtype=float)
                add_minmax(layer_records, feature, values, f"layer_norm_{feature}")

        for record in records:
            record["sample_norm_text_x_norm"] = (
                record["sample_norm_text_mass"] * record["sample_norm_text_value_norm"]
            )
            record["layer_norm_text_x_norm"] = (
                record["layer_norm_text_mass"] * record["layer_norm_text_value_norm"]
            )
            record["sample_norm_text_ratio_x_value"] = (
                record["sample_norm_text_ratio"] * record["sample_norm_text_value_norm"]
            )
            record["layer_norm_text_ratio_x_value"] = (
                record["layer_norm_text_ratio"] * record["layer_norm_text_value_norm"]
            )
            record["sample_norm_text_ratio_x_entropy"] = (
                record["sample_norm_text_ratio"] * record["sample_norm_img_entropy_norm"]
            )
            record["layer_norm_text_ratio_x_entropy"] = (
                record["layer_norm_text_ratio"] * record["layer_norm_img_entropy_norm"]
            )
    return samples


def add_minmax(records, feature, values, output_key):
    if len(values) == 0:
        return
    low = float(np.min(values))
    high = float(np.max(values))
    denom = high - low
    for record in records:
        if denom <= 1e-12:
            record[output_key] = 0.0
        else:
            record[output_key] = float((record[feature] - low) / denom)


def flatten(samples):
    rows = []
    for sample in samples:
        rows.extend(sample["records"])
    return rows


def average_ranks(values):
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    idx = 0
    while idx < len(values):
        end = idx + 1
        while end < len(values) and values[order[end]] == values[order[idx]]:
            end += 1
        rank = (idx + 1 + end) / 2.0
        ranks[order[idx:end]] = rank
        idx = end
    return ranks


def auroc(labels, scores):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = average_ranks(scores)
    rank_sum_pos = float(ranks[pos].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def percentile(values, q):
    if not values:
        return None
    return float(np.percentile(np.array(values, dtype=float), q))


def write_csv(path, rows):
    if not rows:
        with open(path, "w") as f:
            f.write("")
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def feature_auc_rows(rows, teacher_sets, selector_features, trigger_tau):
    output = []
    for top_k, positives in teacher_sets.items():
        for trigger_name, filtered_rows in [
            ("all_heads", rows),
            ("text_triggered", [row for row in rows if row["text_mass"] >= trigger_tau]),
        ]:
            labels = [1 if row["head_key"] in positives else 0 for row in filtered_rows]
            if not filtered_rows:
                continue
            for feature in selector_features:
                scores = [row.get(feature, 0.0) for row in filtered_rows]
                auc = auroc(labels, scores)
                if auc is None:
                    continue
                mean_pos = np.mean([score for label, score in zip(labels, scores) if label])
                mean_neg = np.mean([score for label, score in zip(labels, scores) if not label])
                output.append({
                    "teacher_top_k": top_k,
                    "subset": trigger_name,
                    "feature": feature,
                    "n": len(filtered_rows),
                    "n_positive": int(sum(labels)),
                    "auroc_high_predicts_teacher": auc,
                    "auroc_abs": max(auc, 1.0 - auc),
                    "direction": "high_predicts_teacher" if auc >= 0.5 else "low_predicts_teacher",
                    "mean_positive": float(mean_pos),
                    "mean_negative": float(mean_neg),
                })
    return sorted(output, key=lambda row: (row["teacher_top_k"], row["subset"], -row["auroc_abs"]))


def selector_overlap_rows(samples, teacher_sets, selector_features, eval_top_ks, trigger_tau):
    output = []
    for teacher_top_k, positives in teacher_sets.items():
        for selector in selector_features:
            for eval_top_k in eval_top_ks:
                for mode in ("all_heads", "text_triggered"):
                    stats = []
                    for sample in samples:
                        records = sample["records"]
                        if mode == "text_triggered":
                            records = [row for row in records if row["text_mass"] >= trigger_tau]
                        if not records:
                            selected = set()
                        else:
                            ranked = sorted(records, key=lambda row: row.get(selector, 0.0), reverse=True)
                            selected = {row["head_key"] for row in ranked[:eval_top_k]}
                        inter = selected & positives
                        union = selected | positives
                        stats.append({
                            "overlap": len(inter),
                            "precision": len(inter) / max(len(selected), 1),
                            "recall": len(inter) / max(len(positives), 1),
                            "jaccard": len(inter) / max(len(union), 1),
                            "selected_count": len(selected),
                        })
                    for metric in ("overlap", "precision", "recall", "jaccard", "selected_count"):
                        values = [row[metric] for row in stats]
                        output.append({
                            "teacher_top_k": teacher_top_k,
                            "selector": selector,
                            "mode": mode,
                            "eval_top_k": eval_top_k,
                            "metric": metric,
                            "mean": float(np.mean(values)),
                            "p10": percentile(values, 10),
                            "p50": percentile(values, 50),
                            "p90": percentile(values, 90),
                        })
    return output


def sample_selector_rows(samples, positives, selector, eval_top_k, trigger_tau):
    output = []
    for sample in samples:
        records = [row for row in sample["records"] if row["text_mass"] >= trigger_tau]
        ranked = sorted(records, key=lambda row: row.get(selector, 0.0), reverse=True)
        selected = {row["head_key"] for row in ranked[:eval_top_k]}
        inter = sorted(selected & positives)
        output.append({
            "sample_id": sample["sample_id"],
            "image": sample.get("image", ""),
            "selector": selector,
            "eval_top_k": eval_top_k,
            "triggered_count": len(records),
            "selected_count": len(selected),
            "overlap": len(inter),
            "precision": len(inter) / max(len(selected), 1),
            "teacher_recall": len(inter) / max(len(positives), 1),
            "overlap_heads": " ".join(inter),
        })
    return output


def teacher_feature_summary(rows, teacher_sets, selector_features):
    output = []
    for top_k, positives in teacher_sets.items():
        for group_name, group_rows in [
            ("teacher", [row for row in rows if row["head_key"] in positives]),
            ("non_teacher", [row for row in rows if row["head_key"] not in positives]),
        ]:
            record = {
                "teacher_top_k": top_k,
                "group": group_name,
                "n": len(group_rows),
            }
            for feature in selector_features:
                values = [row.get(feature, 0.0) for row in group_rows]
                record[f"{feature}_mean"] = float(np.mean(values)) if values else None
                record[f"{feature}_p50"] = percentile(values, 50)
            output.append(record)
    return output


def save_selector_bar(path, overlap_rows, teacher_top_k, metric="overlap", mode="text_triggered", eval_top_k=20):
    rows = [
        row for row in overlap_rows
        if row["teacher_top_k"] == teacher_top_k
        and row["metric"] == metric
        and row["mode"] == mode
        and row["eval_top_k"] == eval_top_k
    ]
    rows = sorted(rows, key=lambda row: row["mean"], reverse=True)[:20]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [row["selector"] for row in rows]
    values = [row["mean"] for row in rows]
    ax.bar(range(len(rows)), values, color="#386fa4")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(f"mean {metric}")
    ax.set_title(f"Online selector recovery of teacher top-{teacher_top_k} ({mode}, top-{eval_top_k})")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def parse_int_list(text):
    return [int(item) for item in text.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records-jsonl", required=True)
    parser.add_argument("--teacher-head-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--teacher-top-ks", default="20,40,60,100")
    parser.add_argument("--eval-top-ks", default="20,40,60,100")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--trigger-text-tau", type=float, default=0.4)
    parser.add_argument("--min-layer", type=int, default=None)
    parser.add_argument("--max-layer", type=int, default=None)
    parser.add_argument("--sample-detail-teacher-top-k", type=int, default=100)
    parser.add_argument("--sample-detail-selector", default="layer_norm_text_x_norm")
    parser.add_argument("--sample-detail-eval-top-k", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    teacher_scores = load_teacher_scores(args.teacher_head_path)
    teacher_ranked = sorted(teacher_scores.items(), key=lambda item: item[1], reverse=True)
    teacher_top_ks = parse_int_list(args.teacher_top_ks)
    eval_top_ks = parse_int_list(args.eval_top_ks)
    teacher_sets = {
        top_k: {key for key, _ in teacher_ranked[:top_k]}
        for top_k in teacher_top_ks
    }

    samples = add_normalized_features(load_records(
        args.records_jsonl,
        max_samples=args.max_samples,
        min_layer=args.min_layer,
        max_layer=args.max_layer,
    ))
    rows = flatten(samples)
    selector_features = [
        "text_mass",
        "text_ratio",
        "text_value_norm",
        "text_ratio_img_entropy",
        "text_img_log_ratio",
        "visual_value_ratio",
        "recent_output_ratio",
        "output_attention",
        "recent_output_attention",
        "sample_norm_text_mass",
        "sample_norm_text_value_norm",
        "sample_norm_text_x_norm",
        "sample_norm_text_ratio_x_value",
        "sample_norm_text_ratio_x_entropy",
        "layer_norm_text_mass",
        "layer_norm_text_value_norm",
        "layer_norm_text_x_norm",
        "layer_norm_text_ratio_x_value",
        "layer_norm_text_ratio_x_entropy",
        "layer_norm_img_entropy_norm",
    ]

    auc_rows = feature_auc_rows(rows, teacher_sets, selector_features, args.trigger_text_tau)
    overlap_rows = selector_overlap_rows(samples, teacher_sets, selector_features, eval_top_ks, args.trigger_text_tau)
    summary_rows = teacher_feature_summary(rows, teacher_sets, selector_features)
    detail_top = args.sample_detail_teacher_top_k
    detail_rows = sample_selector_rows(
        samples,
        teacher_sets[detail_top],
        args.sample_detail_selector,
        args.sample_detail_eval_top_k,
        args.trigger_text_tau,
    )

    write_csv(os.path.join(args.output_dir, "feature_auc_teacher_membership.csv"), auc_rows)
    write_csv(os.path.join(args.output_dir, "selector_overlap_summary.csv"), overlap_rows)
    write_csv(os.path.join(args.output_dir, "teacher_feature_summary.csv"), summary_rows)
    write_csv(os.path.join(args.output_dir, "sample_selector_detail.csv"), detail_rows)
    save_selector_bar(
        os.path.join(args.output_dir, "selector_overlap_bar_top100_triggered_top20.png"),
        overlap_rows,
        teacher_top_k=detail_top,
        metric="overlap",
        mode="text_triggered",
        eval_top_k=args.sample_detail_eval_top_k,
    )

    metadata = {
        "records_jsonl": args.records_jsonl,
        "teacher_head_path": args.teacher_head_path,
        "num_samples": len(samples),
        "num_records": len(rows),
        "teacher_top_ks": teacher_top_ks,
        "eval_top_ks": eval_top_ks,
        "trigger_text_tau": args.trigger_text_tau,
        "min_layer": args.min_layer,
        "max_layer": args.max_layer,
        "selector_features": selector_features,
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("wrote:", args.output_dir)
    print("num samples:", len(samples))
    print("num records:", len(rows))
    print("top selectors for teacher top-100, text-triggered, eval top-20:")
    for row in sorted([
        r for r in overlap_rows
        if r["teacher_top_k"] == 100 and r["mode"] == "text_triggered" and r["eval_top_k"] == 20 and r["metric"] == "overlap"
    ], key=lambda item: item["mean"], reverse=True)[:10]:
        print(row["selector"], row["mean"], row["p50"], row["p90"])


if __name__ == "__main__":
    main()

import argparse
import csv
import json
import math
import os
from collections import defaultdict

from eval_scripts.soft_routing.head_prior_utils import default_heads_for_model, head_key, parse_head_key


def safe_float(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    return value


def safe_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def mean(values):
    values = [value for value in values if value is not None and math.isfinite(value)]
    return float(sum(values) / len(values)) if values else None


def percentile(values, q):
    values = sorted(value for value in values if value is not None and math.isfinite(value))
    if not values:
        return None
    idx = int(round((len(values) - 1) * q / 100.0))
    return float(values[idx])


def positive_mean(values):
    values = [max(0.0, value) for value in values if value is not None and math.isfinite(value)]
    return float(sum(values) / len(values)) if values else None


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


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def label_name(row):
    for key in ("label_name", "label_family", "base_label_name"):
        value = str(row.get(key, "")).strip()
        if value in {"hallucinated", "grounded"}:
            return value
    return "hallucinated" if safe_int(row.get("label"), 0) == 1 else "grounded"


def step_key(row):
    return ":".join([
        str(row.get("image_id") or row.get("question_id") or row.get("image") or ""),
        str(row.get("mention_index") or row.get("occurrence_idx") or ""),
        str(row.get("node_word") or row.get("target_token") or ""),
        str(row.get("token_pos") or row.get("target_token_pos") or ""),
    ])


def score_heads(rows, teacher):
    grouped = defaultdict(list)
    for row in rows:
        key = row.get("head_key") or head_key(row.get("layer"), row.get("head"))
        grouped[key].append(row)

    scored = []
    for key, items in sorted(grouped.items()):
        layer, head = parse_head_key(key)
        values = [safe_float(row.get(teacher)) for row in items]
        values = [value for value in values if value is not None]
        positives = [value for value in values if value > 0.0]
        text_mass_values = [safe_float(row.get("text_mass")) for row in items]
        text_entropy_values = [safe_float(row.get("text_entropy_norm")) for row in items]
        concentrated_values = []
        for row in items:
            text_mass = safe_float(row.get("text_mass"))
            text_entropy = safe_float(row.get("text_entropy_norm"))
            if text_mass is None or text_entropy is None:
                continue
            concentrated_values.append(text_mass * max(0.0, 1.0 - text_entropy))
        scored.append({
            "layer": layer,
            "head": head,
            "head_key": key,
            "teacher_feature": teacher,
            "n_rows": len(items),
            "n_unique_mentions": len({step_key(row) for row in items}),
            "mean_teacher": mean(values),
            "positive_mean_teacher": positive_mean(values),
            "sum_positive_teacher": sum(max(0.0, value) for value in values) if values else None,
            "positive_fraction_teacher": float(len(positives) / len(values)) if values else None,
            "q50_teacher": percentile(values, 50),
            "q75_teacher": percentile(values, 75),
            "q90_teacher": percentile(values, 90),
            "max_teacher": max(values) if values else None,
            "mean_text_mass": mean(text_mass_values),
            "mean_text_entropy_norm": mean(text_entropy_values),
            "mean_concentrated_text_leverage": mean(concentrated_values),
        })

    by_positive = sorted(
        scored,
        key=lambda row: (
            -safe_float(row.get("positive_mean_teacher"), -1e30),
            -safe_float(row.get("mean_teacher"), -1e30),
            row["head_key"],
        ),
    )
    by_mean = sorted(
        scored,
        key=lambda row: (
            -safe_float(row.get("mean_teacher"), -1e30),
            -safe_float(row.get("positive_mean_teacher"), -1e30),
            row["head_key"],
        ),
    )
    for idx, row in enumerate(by_positive, start=1):
        row["rank_positive_mean"] = idx
    for idx, row in enumerate(by_mean, start=1):
        row["rank_mean"] = idx
    return sorted(scored, key=lambda row: int(row["rank_positive_mean"]))


def head_pool_rows(raw_rows, model_path):
    grouped = defaultdict(list)
    labels_by_head = defaultdict(lambda: defaultdict(int))
    mentions_by_head = defaultdict(set)
    layers = defaultdict(set)
    for row in raw_rows:
        key = row.get("head_key") or head_key(row.get("layer"), row.get("head"))
        grouped[key].append(row)
        labels_by_head[key][label_name(row)] += 1
        mentions_by_head[key].add(step_key(row))
        try:
            layer, head = parse_head_key(key)
        except Exception:
            continue
        layers[layer].add(head)

    default_set = {head_key(layer, head) for layer, head in default_heads_for_model(model_path)}
    output = []
    for key, items in sorted(grouped.items(), key=lambda item: parse_head_key(item[0])):
        layer, head = parse_head_key(key)
        output.append({
            "layer": layer,
            "head": head,
            "head_key": key,
            "n_rows": len(items),
            "n_unique_mentions": len(mentions_by_head[key]),
            "n_hallucinated_rows": labels_by_head[key].get("hallucinated", 0),
            "n_grounded_rows": labels_by_head[key].get("grounded", 0),
            "in_adhh_default": int(key in default_set),
        })
    layer_rows = []
    for layer in sorted(layers):
        keys = {head_key(layer, head) for head in layers[layer]}
        layer_rows.append({
            "layer": layer,
            "n_heads": len(layers[layer]),
            "n_adhh_default_heads": len(keys & default_set),
            "heads": ",".join(str(head) for head in sorted(layers[layer])),
        })
    return output, layer_rows


def head_pool_summary(raw_rows, model_path):
    head_rows, layer_rows = head_pool_rows(raw_rows, model_path)
    head_keys = {row["head_key"] for row in head_rows}
    default_set = {head_key(layer, head) for layer, head in default_heads_for_model(model_path)}
    label_counts = defaultdict(int)
    mentions_by_label = defaultdict(set)
    for row in raw_rows:
        label = label_name(row)
        label_counts[label] += 1
        mentions_by_label[label].add(step_key(row))
    row_counts = [safe_float(row["n_rows"]) for row in head_rows]
    mention_counts = [safe_float(row["n_unique_mentions"]) for row in head_rows]
    return {
        "n_rows": len(raw_rows),
        "n_candidate_heads": len(head_rows),
        "n_candidate_layers": len(layer_rows),
        "n_adhh_default_overlap": len(head_keys & default_set),
        "adhh_default_overlap": sorted(head_keys & default_set, key=parse_head_key),
        "label_row_counts": dict(label_counts),
        "label_unique_mentions": {label: len(values) for label, values in mentions_by_label.items()},
        "rows_per_head_min": min(row_counts) if row_counts else None,
        "rows_per_head_median": percentile(row_counts, 50),
        "rows_per_head_max": max(row_counts) if row_counts else None,
        "mentions_per_head_min": min(mention_counts) if mention_counts else None,
        "mentions_per_head_median": percentile(mention_counts, 50),
        "mentions_per_head_max": max(mention_counts) if mention_counts else None,
    }


def distribution_rows(scored):
    output = []
    for field in (
        "positive_mean_teacher",
        "mean_teacher",
        "sum_positive_teacher",
        "positive_fraction_teacher",
        "mean_text_mass",
        "mean_text_entropy_norm",
        "mean_concentrated_text_leverage",
    ):
        values = [safe_float(row.get(field)) for row in scored]
        values = [value for value in values if value is not None]
        output.append({
            "feature": field,
            "n_heads": len(values),
            "mean": mean(values),
            "min": min(values) if values else None,
            "q10": percentile(values, 10),
            "q25": percentile(values, 25),
            "q50": percentile(values, 50),
            "q75": percentile(values, 75),
            "q90": percentile(values, 90),
            "q95": percentile(values, 95),
            "max": max(values) if values else None,
        })
    return output


def write_prior(path, rows, top_k, score_field, metadata):
    selected = rows[: int(top_k)]
    payload = {
        "hal_heads": [[int(row["layer"]), int(row["head"])] for row in selected],
        "contrastive_scores": [
            {
                "layer": int(row["layer"]),
                "head": int(row["head"]),
                "head_key": row["head_key"],
                "score": safe_float(row.get(score_field), 0.0),
                "rank": idx + 1,
            }
            for idx, row in enumerate(rows)
        ],
        "metadata": metadata,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return payload


def selection_rows(scored, top_ks):
    by_positive = sorted(scored, key=lambda row: int(row["rank_positive_mean"]))
    by_mean = sorted(scored, key=lambda row: int(row["rank_mean"]))
    output = []
    for mode, rows, field in (
        ("positive_mean", by_positive, "positive_mean_teacher"),
        ("mean", by_mean, "mean_teacher"),
    ):
        for top_k in top_ks:
            selected = rows[: int(top_k)]
            output.append({
                "score_mode": mode,
                "top_k": int(top_k),
                "n_selected": len(selected),
                "mean_score": mean([safe_float(row.get(field)) for row in selected]),
                "mean_positive_teacher": mean([safe_float(row.get("positive_mean_teacher")) for row in selected]),
                "mean_signed_teacher": mean([safe_float(row.get("mean_teacher")) for row in selected]),
                "mean_text_mass": mean([safe_float(row.get("mean_text_mass")) for row in selected]),
                "mean_text_entropy_norm": mean([safe_float(row.get("mean_text_entropy_norm")) for row in selected]),
                "heads": ",".join(row["head_key"] for row in selected),
            })
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-rows", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--teacher-feature", default="target_text_logprob_drop")
    parser.add_argument("--label-filter", default="hallucinated", choices=["hallucinated", "grounded", "all"])
    parser.add_argument("--model-path", default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--top-ks", default="20,40")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    raw_rows = read_rows(args.head_rows)
    pool_rows, layer_rows = head_pool_rows(raw_rows, args.model_path)
    pool_summary = head_pool_summary(raw_rows, args.model_path)
    if args.label_filter == "all":
        rows = raw_rows
    else:
        rows = [row for row in raw_rows if label_name(row) == args.label_filter]
    scored = score_heads(rows, args.teacher_feature)
    top_ks = [
        safe_int(item, 0)
        for item in str(args.top_ks).replace(" ", ",").split(",")
        if item.strip()
    ]
    top_ks = [top_k for top_k in top_ks if top_k > 0]

    scores_path = os.path.join(args.output_dir, "hallucination_leverage_head_scores.csv")
    distribution_path = os.path.join(args.output_dir, "hallucination_leverage_head_distribution.csv")
    selection_path = os.path.join(args.output_dir, "hallucination_leverage_head_selection_summary.csv")
    pool_path = os.path.join(args.output_dir, "head_candidate_pool.csv")
    layer_pool_path = os.path.join(args.output_dir, "head_candidate_pool_by_layer.csv")
    pool_summary_path = os.path.join(args.output_dir, "head_candidate_pool_summary.json")
    write_csv(scores_path, scored)
    write_csv(distribution_path, distribution_rows(scored))
    write_csv(selection_path, selection_rows(scored, top_ks))
    write_csv(pool_path, pool_rows)
    write_csv(layer_pool_path, layer_rows)
    with open(pool_summary_path, "w") as f:
        json.dump(pool_summary, f, indent=2)

    metadata = {
        "head_rows": args.head_rows,
        "teacher_feature": args.teacher_feature,
        "label_filter": args.label_filter,
        "model_path": args.model_path,
        "n_rows_raw": len(raw_rows),
        "n_rows_used": len(rows),
        "n_heads_scored": len(scored),
        "head_pool_summary": pool_summary,
    }
    for mode, rank_field, score_field in (
        ("positive_mean", "rank_positive_mean", "positive_mean_teacher"),
        ("mean", "rank_mean", "mean_teacher"),
    ):
        ranked = sorted(scored, key=lambda row: int(row[rank_field]))
        for top_k in top_ks:
            prefix = f"hallucination_leverage_{mode}_top{top_k}"
            heads_txt = os.path.join(args.output_dir, f"{prefix}_heads.txt")
            prior_path = os.path.join(args.output_dir, f"{prefix}_head_prior.json")
            selected = ranked[: int(top_k)]
            with open(heads_txt, "w") as f:
                f.write(",".join(row["head_key"] for row in selected))
                f.write("\n")
            write_prior(
                prior_path,
                ranked,
                top_k,
                score_field,
                {**metadata, "score_mode": mode, "top_k": top_k},
            )

    with open(os.path.join(args.output_dir, "hallucination_leverage_head_discovery_summary.json"), "w") as f:
        json.dump({
            **metadata,
            "outputs": {
                "scores": scores_path,
                "distribution": distribution_path,
                "selection_summary": selection_path,
                "head_candidate_pool": pool_path,
                "head_candidate_pool_by_layer": layer_pool_path,
                "head_candidate_pool_summary": pool_summary_path,
            },
            "top_positive_mean": sorted(scored, key=lambda row: int(row["rank_positive_mean"]))[:20],
            "top_mean": sorted(scored, key=lambda row: int(row["rank_mean"]))[:20],
        }, f, indent=2)

    print("[summary] hallucination leverage head discovery")
    print(json.dumps({
        **metadata,
        "top_positive_mean": [
            {
                "rank": row["rank_positive_mean"],
                "head_key": row["head_key"],
                "positive_mean_teacher": row["positive_mean_teacher"],
                "mean_teacher": row["mean_teacher"],
                "n_rows": row["n_rows"],
            }
            for row in sorted(scored, key=lambda item: int(item["rank_positive_mean"]))[:10]
        ],
        "top_mean": [
            {
                "rank": row["rank_mean"],
                "head_key": row["head_key"],
                "mean_teacher": row["mean_teacher"],
                "positive_mean_teacher": row["positive_mean_teacher"],
                "n_rows": row["n_rows"],
            }
            for row in sorted(scored, key=lambda item: int(item["rank_mean"]))[:10]
        ],
    }, indent=2))


if __name__ == "__main__":
    main()

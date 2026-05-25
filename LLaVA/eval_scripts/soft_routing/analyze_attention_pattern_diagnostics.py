import argparse
import csv
import json
import math
import os
from collections import defaultdict


DEFAULT_FEATURES = [
    "text_mass",
    "img_mass",
    "text_max_attention",
    "text_top1_ratio",
    "text_entropy_norm",
    "text_concentration",
    "recent_text_mass",
    "recent_text_ratio",
    "first_text_attention",
    "last_text_attention",
    "img_max_attention",
    "img_top1_ratio",
    "img_entropy_norm",
    "img_concentration",
    "img_top1_offset",
    "full_max_attention",
    "full_entropy_norm",
    "full_concentration",
    "full_top1_index",
    "full_top1_is_img",
    "full_top1_is_text",
    "full_top1_is_prefix",
]


def safe_float(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    return value


def iter_jsonl(path):
    with open(path) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON in {path} at line {line_no}: {line[:120]!r}") from exc


def mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def variance(values):
    values = [value for value in values if value is not None]
    if len(values) < 2:
        return None
    mu = sum(values) / len(values)
    return sum((value - mu) ** 2 for value in values) / len(values)


def percentile(values, q):
    values = sorted(value for value in values if value is not None)
    if not values:
        return None
    idx = int(round((len(values) - 1) * q / 100.0))
    return values[idx]


def write_csv(path, rows):
    if not rows:
        with open(path, "w") as f:
            f.write("")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def layer_from_head_key(head_key):
    if not head_key or ":" not in head_key:
        return ""
    return head_key.split(":", 1)[0]


def bucket(value, width, lo=0.0, hi=1.0):
    value = min(max(value, lo), hi)
    bin_lo = math.floor((value - lo) / width) * width + lo
    bin_hi = min(bin_lo + width, hi)
    return f"{bin_lo:.2f}-{bin_hi:.2f}"


def normalize_head_rows(rows, features):
    output = []
    for row in rows:
        item = {
            "record_type": row.get("record_type", ""),
            "phase": row.get("phase", ""),
            "question_id": row.get("question_id", ""),
            "step_index": row.get("step_index", ""),
            "layer": str(row.get("layer", layer_from_head_key(row.get("head_key", "")))),
            "head_key": row.get("head_key", ""),
            "head": str(row.get("head", "")),
            "prefill_protected": bool(row.get("prefill_protected")),
            "active": bool(row.get("active")),
        }
        has_feature = False
        for feature in features:
            value = safe_float(row.get(feature))
            item[feature] = value
            has_feature = has_feature or value is not None
        if has_feature:
            output.append(item)
    return output


def summarize_values(values):
    values = [value for value in values if value is not None]
    if not values:
        return {
            "n": 0,
            "mean": None,
            "var": None,
            "p10": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "min": None,
            "max": None,
        }
    return {
        "n": len(values),
        "mean": mean(values),
        "var": variance(values),
        "p10": percentile(values, 10),
        "p25": percentile(values, 25),
        "p50": percentile(values, 50),
        "p75": percentile(values, 75),
        "p90": percentile(values, 90),
        "min": min(values),
        "max": max(values),
    }


def summarize_distribution(rows, group_keys, features):
    grouped = defaultdict(list)
    for row in rows:
        group = tuple(row.get(key, "") for key in group_keys)
        for feature in features:
            value = row.get(feature)
            if value is not None:
                grouped[group + (feature,)].append(value)

    output = []
    for key, values in grouped.items():
        group = key[:-1]
        feature = key[-1]
        item = {group_key: group[idx] for idx, group_key in enumerate(group_keys)}
        item["feature"] = feature
        item.update(summarize_values(values))
        output.append(item)
    output.sort(key=lambda row: tuple(str(row.get(key, "")) for key in group_keys) + (row["feature"],))
    return output


def summarize_histogram(rows, group_keys, features, width):
    counts = defaultdict(int)
    totals = defaultdict(int)
    for row in rows:
        group = tuple(row.get(key, "") for key in group_keys)
        for feature in features:
            value = row.get(feature)
            if value is None:
                continue
            bin_name = bucket(value, width)
            key = group + (feature,)
            counts[key + (bin_name,)] += 1
            totals[key] += 1

    output = []
    for key_with_bin, count in counts.items():
        group = key_with_bin[: len(group_keys)]
        feature = key_with_bin[len(group_keys)]
        bin_name = key_with_bin[-1]
        total_key = group + (feature,)
        item = {group_key: group[idx] for idx, group_key in enumerate(group_keys)}
        item.update({
            "feature": feature,
            "bin": bin_name,
            "n": count,
            "rate": count / max(totals[total_key], 1),
        })
        output.append(item)
    output.sort(key=lambda row: tuple(str(row.get(key, "")) for key in group_keys) + (row["feature"], row["bin"]))
    return output


def summarize_step_variance(rows, features):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (row["phase"], row["question_id"], row["step_index"])
        for feature in features:
            value = row.get(feature)
            if value is not None:
                grouped[key][feature].append(value)

    output = []
    for (phase, question_id, step_index), by_feature in grouped.items():
        for feature, values in by_feature.items():
            stats = summarize_values(values)
            output.append({
                "phase": phase,
                "question_id": question_id,
                "step_index": step_index,
                "feature": feature,
                "n_heads": stats["n"],
                "mean_across_heads": stats["mean"],
                "var_across_heads": stats["var"],
                "p10_across_heads": stats["p10"],
                "p50_across_heads": stats["p50"],
                "p90_across_heads": stats["p90"],
            })
    output.sort(key=lambda row: (str(row["phase"]), str(row["question_id"]), int(row["step_index"] or 0), row["feature"]))
    return output


def summarize_head_step_variance(rows, features):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (row["phase"], row["head_key"], row["layer"])
        for feature in features:
            value = row.get(feature)
            if value is not None:
                grouped[key][feature].append(value)

    output = []
    for (phase, head_key, layer), by_feature in grouped.items():
        for feature, values in by_feature.items():
            stats = summarize_values(values)
            output.append({
                "phase": phase,
                "head_key": head_key,
                "layer": layer,
                "feature": feature,
                "n_steps": stats["n"],
                "mean_across_steps": stats["mean"],
                "var_across_steps": stats["var"],
                "p10_across_steps": stats["p10"],
                "p50_across_steps": stats["p50"],
                "p90_across_steps": stats["p90"],
            })
    output.sort(
        key=lambda row: (
            str(row["phase"]),
            row["feature"],
            -(row["var_across_steps"] or 0.0),
            str(row["layer"]),
            row["head_key"],
        )
    )
    return output


def two_cluster_summary(values):
    values = sorted(value for value in values if value is not None)
    if len(values) < 4:
        return None
    left = percentile(values, 25)
    right = percentile(values, 75)
    if left is None or right is None:
        return None
    for _ in range(30):
        left_values = [value for value in values if abs(value - left) <= abs(value - right)]
        right_values = [value for value in values if abs(value - left) > abs(value - right)]
        if not left_values or not right_values:
            return None
        new_left = mean(left_values)
        new_right = mean(right_values)
        if abs(new_left - left) + abs(new_right - right) < 1e-8:
            left, right = new_left, new_right
            break
        left, right = new_left, new_right
    if left > right:
        left, right = right, left
        left_values, right_values = right_values, left_values
    total_var = variance(values) or 0.0
    separation = right - left
    valley_midpoint = (left + right) / 2.0
    between = [value for value in values if left <= value <= right]
    valley_hist = valley_midpoint
    if between:
        width = max((right - left) / 20.0, 1e-6)
        bins = defaultdict(int)
        for value in between:
            bins[bucket(value, width, left, right)] += 1
        valley_bin = min(bins.items(), key=lambda item: item[1])[0]
        valley_hist = sum(float(part) for part in valley_bin.split("-")) / 2.0
    return {
        "n": len(values),
        "cluster_low_mean": left,
        "cluster_high_mean": right,
        "cluster_separation": separation,
        "cluster_separation_over_std": separation / math.sqrt(total_var) if total_var > 0 else None,
        "cluster_low_weight": len(left_values) / len(values),
        "cluster_high_weight": len(right_values) / len(values),
        "valley_midpoint": valley_midpoint,
        "valley_histogram": valley_hist,
    }


def summarize_bimodality(rows, group_keys, features):
    grouped = defaultdict(list)
    for row in rows:
        group = tuple(row.get(key, "") for key in group_keys)
        for feature in features:
            value = row.get(feature)
            if value is not None:
                grouped[group + (feature,)].append(value)
    output = []
    for key, values in grouped.items():
        summary = two_cluster_summary(values)
        if summary is None:
            continue
        group = key[:-1]
        feature = key[-1]
        item = {group_key: group[idx] for idx, group_key in enumerate(group_keys)}
        item["feature"] = feature
        item.update(summary)
        output.append(item)
    output.sort(
        key=lambda row: (
            tuple(str(row.get(key, "")) for key in group_keys),
            -(row.get("cluster_separation_over_std") or 0.0),
            row["feature"],
        )
    )
    return output


def summarize_image_sink_offsets(rows):
    groups = defaultdict(lambda: defaultdict(int))
    totals = defaultdict(int)
    for row in rows:
        value = row.get("img_top1_offset")
        if value is None:
            continue
        offset = int(round(value))
        for group_keys in (("record_type", "phase"), ("record_type", "phase", "layer")):
            group = tuple(row.get(key, "") for key in group_keys)
            groups[(group_keys, group)][offset] += 1
            totals[(group_keys, group)] += 1
    output = []
    for (group_keys, group), counts in groups.items():
        total = totals[(group_keys, group)]
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        for rank, (offset, count) in enumerate(ranked[:20], start=1):
            item = {group_key: group[idx] for idx, group_key in enumerate(group_keys)}
            item.update({
                "rank": rank,
                "img_top1_offset": offset,
                "n": count,
                "rate": count / max(total, 1),
                "total": total,
            })
            output.append(item)
    output.sort(key=lambda row: tuple(str(row.get(key, "")) for key in ("record_type", "phase", "layer")) + (row["rank"],))
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--features", default=",".join(DEFAULT_FEATURES))
    parser.add_argument("--hist-bin-width", type=float, default=0.05)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    features = [item.strip() for item in args.features.split(",") if item.strip()]
    rows = list(iter_jsonl(args.diagnostics_jsonl))
    candidate_rows = normalize_head_rows((row for row in rows if row.get("record_type") == "candidate_head"), features)
    selected_rows = normalize_head_rows((row for row in rows if row.get("record_type") == "selected_head"), features)
    active_rows = normalize_head_rows(
        (row for row in rows if row.get("record_type") == "selected_head" and row.get("active")),
        features,
    )
    distribution_source = candidate_rows if candidate_rows else selected_rows
    distribution_source_name = "candidate_head" if candidate_rows else "selected_head"

    files = {
        "overall_summary": os.path.join(args.output_dir, "attention_pattern_overall_summary.csv"),
        "layer_summary": os.path.join(args.output_dir, "attention_pattern_layer_summary.csv"),
        "histogram": os.path.join(args.output_dir, "attention_pattern_histogram.csv"),
        "step_variance": os.path.join(args.output_dir, "attention_pattern_step_variance.csv"),
        "head_step_variance": os.path.join(args.output_dir, "attention_pattern_head_step_variance.csv"),
        "active_summary": os.path.join(args.output_dir, "attention_pattern_active_summary.csv"),
        "bimodality_summary": os.path.join(args.output_dir, "attention_pattern_bimodality_summary.csv"),
        "image_sink_offsets": os.path.join(args.output_dir, "image_sink_offset_summary.csv"),
    }
    write_csv(files["overall_summary"], summarize_distribution(distribution_source, ["record_type", "phase"], features))
    write_csv(files["layer_summary"], summarize_distribution(distribution_source, ["record_type", "phase", "layer"], features))
    write_csv(files["histogram"], summarize_histogram(distribution_source, ["record_type", "phase"], features, args.hist_bin_width))
    write_csv(files["step_variance"], summarize_step_variance(distribution_source, features))
    write_csv(files["head_step_variance"], summarize_head_step_variance(distribution_source, features))
    write_csv(files["active_summary"], summarize_distribution(active_rows, ["record_type", "phase", "prefill_protected"], features))
    write_csv(files["bimodality_summary"], summarize_bimodality(distribution_source, ["record_type", "phase"], features))
    write_csv(files["image_sink_offsets"], summarize_image_sink_offsets(distribution_source))

    config = {
        "diagnostics_jsonl": args.diagnostics_jsonl,
        "output_dir": args.output_dir,
        "features": features,
        "n_records": len(rows),
        "n_candidate_rows": len(candidate_rows),
        "n_selected_rows": len(selected_rows),
        "n_active_rows": len(active_rows),
        "distribution_source": distribution_source_name,
        "outputs": files,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(json.dumps(config, indent=2))
    print("[summary] overall")
    for row in summarize_distribution(distribution_source, ["record_type", "phase"], features):
        print(row)


if __name__ == "__main__":
    main()

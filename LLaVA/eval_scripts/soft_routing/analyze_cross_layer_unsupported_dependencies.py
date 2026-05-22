import argparse
import csv
import json
import math
import os
from collections import defaultdict


def safe_float(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    return value


def parse_int_ranges(value):
    output = set()
    if not value:
        return output
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            output.update(range(int(start), int(end) + 1))
        else:
            output.add(int(item))
    return output


def parse_heads(value):
    heads = set()
    for item in value.split(","):
        item = item.strip()
        if item:
            heads.add(item)
    return heads


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


def step_key(row):
    return (row.get("question_id"), row.get("step_index"), row.get("phase"))


def update_stats(stats, key, x, y):
    if x is None or y is None:
        return
    item = stats[key]
    item["n"] += 1
    item["sum_x"] += x
    item["sum_y"] += y
    item["sum_x2"] += x * x
    item["sum_y2"] += y * y
    item["sum_xy"] += x * y


def finish_stats(key, values):
    target, target_layer, target_head_key, feature, source_aggregate = key
    n = values["n"]
    if n <= 1:
        corr = None
    else:
        cov = values["sum_xy"] - values["sum_x"] * values["sum_y"] / n
        var_x = values["sum_x2"] - values["sum_x"] * values["sum_x"] / n
        var_y = values["sum_y2"] - values["sum_y"] * values["sum_y"] / n
        den = math.sqrt(max(var_x, 0.0) * max(var_y, 0.0))
        corr = cov / den if den > 0 else None
    return {
        "target": target,
        "target_layer": target_layer,
        "target_head_key": target_head_key,
        "feature": feature,
        "source_aggregate": source_aggregate,
        "n": n,
        "mean_source": values["sum_x"] / n if n else None,
        "mean_target": values["sum_y"] / n if n else None,
        "pearson": corr,
        "pearson_abs": abs(corr) if corr is not None else None,
    }


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--source-heads",
        default="0:25,0:15,5:15,5:13,10:1,10:3,11:6,11:20",
        help="Comma-separated layer:head keys used as the upstream source signal.",
    )
    parser.add_argument("--target-layers", default="12-31")
    parser.add_argument("--feature", default="unsupported_text_value_norm")
    parser.add_argument("--phase", default="decode", choices=["all", "prefill", "decode"])
    parser.add_argument("--source-aggregate", default="mean", choices=["mean", "max"])
    parser.add_argument("--min-n", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=30)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    source_heads = parse_heads(args.source_heads)
    target_layers = parse_int_ranges(args.target_layers)

    source_values = defaultdict(list)
    n_candidate_rows = 0
    n_source_rows = 0
    for row in iter_jsonl(args.diagnostics_jsonl):
        if row.get("record_type") != "candidate_head":
            continue
        if args.phase != "all" and row.get("phase") != args.phase:
            continue
        n_candidate_rows += 1
        if row.get("head_key") not in source_heads:
            continue
        value = safe_float(row.get(args.feature))
        if value is None:
            continue
        source_values[step_key(row)].append(value)
        n_source_rows += 1

    if args.source_aggregate == "max":
        source_by_step = {key: max(values) for key, values in source_values.items() if values}
    else:
        source_by_step = {key: sum(values) / len(values) for key, values in source_values.items() if values}

    head_stats = defaultdict(lambda: {"n": 0, "sum_x": 0.0, "sum_y": 0.0, "sum_x2": 0.0, "sum_y2": 0.0, "sum_xy": 0.0})
    layer_stats = defaultdict(lambda: {"n": 0, "sum_x": 0.0, "sum_y": 0.0, "sum_x2": 0.0, "sum_y2": 0.0, "sum_xy": 0.0})
    n_target_rows = 0
    n_joined_rows = 0
    for row in iter_jsonl(args.diagnostics_jsonl):
        if row.get("record_type") != "candidate_head":
            continue
        if args.phase != "all" and row.get("phase") != args.phase:
            continue
        layer = int(row.get("layer", -1))
        if target_layers and layer not in target_layers:
            continue
        n_target_rows += 1
        source = source_by_step.get(step_key(row))
        target = safe_float(row.get(args.feature))
        if source is None or target is None:
            continue
        n_joined_rows += 1
        head_key = row.get("head_key", "")
        update_stats(
            head_stats,
            ("head", layer, head_key, args.feature, args.source_aggregate),
            source,
            target,
        )
        update_stats(
            layer_stats,
            ("layer", layer, "", args.feature, args.source_aggregate),
            source,
            target,
        )

    head_rows = [finish_stats(key, values) for key, values in head_stats.items()]
    layer_rows = [finish_stats(key, values) for key, values in layer_stats.items()]
    head_rows = [row for row in head_rows if row["n"] >= args.min_n and row["pearson"] is not None]
    layer_rows = [row for row in layer_rows if row["n"] >= args.min_n and row["pearson"] is not None]
    head_rows.sort(key=lambda row: (-row["pearson"], row["target_layer"], row["target_head_key"]))
    layer_rows.sort(key=lambda row: (-row["pearson"], row["target_layer"]))

    head_path = os.path.join(args.output_dir, "cross_layer_head_correlations.csv")
    layer_path = os.path.join(args.output_dir, "cross_layer_layer_correlations.csv")
    write_csv(head_path, head_rows)
    write_csv(layer_path, layer_rows)

    summary = {
        "diagnostics_jsonl": args.diagnostics_jsonl,
        "source_heads": sorted(source_heads),
        "target_layers": sorted(target_layers),
        "feature": args.feature,
        "phase": args.phase,
        "source_aggregate": args.source_aggregate,
        "n_candidate_rows_seen_first_pass": n_candidate_rows,
        "n_source_rows": n_source_rows,
        "n_source_steps": len(source_by_step),
        "n_target_rows": n_target_rows,
        "n_joined_rows": n_joined_rows,
        "head_correlations": head_path,
        "layer_correlations": layer_path,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print("[summary] top positive head correlations")
    for row in head_rows[: args.top_k]:
        print(row)
    print("[summary] top positive layer correlations")
    for row in layer_rows[: args.top_k]:
        print(row)


if __name__ == "__main__":
    main()

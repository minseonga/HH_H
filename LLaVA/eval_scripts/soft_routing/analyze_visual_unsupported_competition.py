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


def clamp01(value):
    return min(1.0, max(0.0, value))


def add_pair(stats, x, y):
    if x is None or y is None:
        return
    stats["n_pair"] += 1
    stats["sum_x"] += x
    stats["sum_y"] += y
    stats["sum_x2"] += x * x
    stats["sum_y2"] += y * y
    stats["sum_xy"] += x * y


def pearson(stats):
    n = stats["n_pair"]
    if n <= 1:
        return None
    cov = stats["sum_xy"] - stats["sum_x"] * stats["sum_y"] / n
    var_x = stats["sum_x2"] - stats["sum_x"] * stats["sum_x"] / n
    var_y = stats["sum_y2"] - stats["sum_y"] * stats["sum_y"] / n
    den = math.sqrt(max(var_x, 0.0) * max(var_y, 0.0))
    return cov / den if den > 0 else None


def mean(stats, key):
    n = stats["n"]
    return stats[key] / n if n else None


def minmax(value, low, high):
    den = max(high - low, 1e-8)
    return clamp01((value - low) / den)


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
    parser.add_argument("--target-layers", default="12-21")
    parser.add_argument("--phase", default="decode", choices=["all", "prefill", "decode"])
    parser.add_argument("--min-n", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=40)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    target_layers = parse_int_ranges(args.target_layers)
    head_stats = defaultdict(lambda: defaultdict(float))
    layer_stats = defaultdict(lambda: defaultdict(float))
    n_candidate_rows = 0
    n_used_rows = 0

    for row in iter_jsonl(args.diagnostics_jsonl):
        if row.get("record_type") != "candidate_head":
            continue
        if args.phase != "all" and row.get("phase") != args.phase:
            continue
        layer = int(row.get("layer", -1))
        if target_layers and layer not in target_layers:
            continue
        n_candidate_rows += 1
        unsupported = safe_float(row.get("unsupported_text_value_norm"))
        img_mass = safe_float(row.get("img_mass"))
        visual_ratio = safe_float(row.get("visual_value_ratio"))
        text_mass = safe_float(row.get("text_mass"))
        if unsupported is None or img_mass is None or visual_ratio is None:
            continue
        n_used_rows += 1
        low_img_mass = 1.0 - clamp01(img_mass)
        low_visual_ratio = 1.0 - clamp01(visual_ratio)
        unsupported_x_low_img = unsupported * low_img_mass
        unsupported_x_low_visual = unsupported * low_visual_ratio
        key = (layer, row.get("head_key", ""))
        for stats in (head_stats[key], layer_stats[(layer, "")]):
            stats["n"] += 1
            stats["sum_unsupported"] += unsupported
            stats["sum_img_mass"] += img_mass
            stats["sum_low_img_mass"] += low_img_mass
            stats["sum_visual_ratio"] += visual_ratio
            stats["sum_low_visual_ratio"] += low_visual_ratio
            stats["sum_text_mass"] += text_mass if text_mass is not None else 0.0
            stats["sum_unsupported_x_low_img_mass"] += unsupported_x_low_img
            stats["sum_unsupported_x_low_visual_ratio"] += unsupported_x_low_visual
            add_pair(stats, unsupported, img_mass)

    raw_head_rows = []
    for (layer, head_key), stats in head_stats.items():
        if stats["n"] < args.min_n:
            continue
        corr = pearson(stats)
        row = {
            "target": "head",
            "target_layer": int(layer),
            "target_head_key": head_key,
            "n": int(stats["n"]),
            "mean_unsupported_text_value_norm": mean(stats, "sum_unsupported"),
            "mean_img_mass": mean(stats, "sum_img_mass"),
            "mean_low_img_mass": mean(stats, "sum_low_img_mass"),
            "mean_visual_value_ratio": mean(stats, "sum_visual_ratio"),
            "mean_low_visual_value_ratio": mean(stats, "sum_low_visual_ratio"),
            "mean_text_mass": mean(stats, "sum_text_mass"),
            "unsupported_x_low_img_mass": mean(stats, "sum_unsupported_x_low_img_mass"),
            "unsupported_x_low_visual_ratio": mean(stats, "sum_unsupported_x_low_visual_ratio"),
            "unsupported_img_mass_pearson": corr,
            "unsupported_img_mass_negative_pearson": -corr if corr is not None else None,
        }
        raw_head_rows.append(row)

    if raw_head_rows:
        unsupported_values = [row["mean_unsupported_text_value_norm"] for row in raw_head_rows]
        low_img_values = [row["mean_low_img_mass"] for row in raw_head_rows]
        low_visual_values = [row["mean_low_visual_value_ratio"] for row in raw_head_rows]
        min_unsupported, max_unsupported = min(unsupported_values), max(unsupported_values)
        min_low_img, max_low_img = min(low_img_values), max(low_img_values)
        min_low_visual, max_low_visual = min(low_visual_values), max(low_visual_values)
    else:
        min_unsupported = max_unsupported = 0.0
        min_low_img = max_low_img = 0.0
        min_low_visual = max_low_visual = 0.0

    head_rows = []
    for row in raw_head_rows:
        unsupported_norm01 = minmax(row["mean_unsupported_text_value_norm"], min_unsupported, max_unsupported)
        low_img_norm01 = minmax(row["mean_low_img_mass"], min_low_img, max_low_img)
        low_visual_norm01 = minmax(row["mean_low_visual_value_ratio"], min_low_visual, max_low_visual)
        row["unsupported_norm01"] = unsupported_norm01
        row["low_img_mass_norm01"] = low_img_norm01
        row["low_visual_value_ratio_norm01"] = low_visual_norm01
        row["visual_competition_score"] = unsupported_norm01 * low_img_norm01
        row["visual_value_competition_score"] = unsupported_norm01 * low_visual_norm01
        row["score"] = row["visual_competition_score"]
        head_rows.append(row)

    layer_rows = []
    for (layer, _), stats in layer_stats.items():
        if stats["n"] < args.min_n:
            continue
        corr = pearson(stats)
        layer_rows.append({
            "target": "layer",
            "target_layer": int(layer),
            "target_head_key": "",
            "n": int(stats["n"]),
            "mean_unsupported_text_value_norm": mean(stats, "sum_unsupported"),
            "mean_img_mass": mean(stats, "sum_img_mass"),
            "mean_low_img_mass": mean(stats, "sum_low_img_mass"),
            "mean_visual_value_ratio": mean(stats, "sum_visual_ratio"),
            "mean_text_mass": mean(stats, "sum_text_mass"),
            "unsupported_x_low_img_mass": mean(stats, "sum_unsupported_x_low_img_mass"),
            "unsupported_x_low_visual_ratio": mean(stats, "sum_unsupported_x_low_visual_ratio"),
            "unsupported_img_mass_pearson": corr,
            "unsupported_img_mass_negative_pearson": -corr if corr is not None else None,
        })

    head_rows.sort(
        key=lambda row: (
            -row["visual_competition_score"],
            -row["unsupported_x_low_img_mass"],
            row["target_layer"],
            row["target_head_key"],
        )
    )
    layer_rows.sort(key=lambda row: (-row["unsupported_x_low_img_mass"], row["target_layer"]))

    head_path = os.path.join(args.output_dir, "visual_unsupported_head_summary.csv")
    layer_path = os.path.join(args.output_dir, "visual_unsupported_layer_summary.csv")
    write_csv(head_path, head_rows)
    write_csv(layer_path, layer_rows)

    summary = {
        "diagnostics_jsonl": args.diagnostics_jsonl,
        "target_layers": sorted(target_layers),
        "phase": args.phase,
        "n_candidate_rows": n_candidate_rows,
        "n_used_rows": n_used_rows,
        "n_head_rows": len(head_rows),
        "n_layer_rows": len(layer_rows),
        "head_summary": head_path,
        "layer_summary": layer_path,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print("[summary] top visual-unsupported competition heads")
    for row in head_rows[: args.top_k]:
        print(row)
    print("[summary] layer summary")
    for row in layer_rows:
        print(row)


if __name__ == "__main__":
    main()

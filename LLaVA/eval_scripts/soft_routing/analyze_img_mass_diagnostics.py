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


def bucket(value, width):
    value = min(max(value, 0.0), 1.0)
    lo = math.floor(value / width) * width
    hi = min(lo + width, 1.0)
    return f"{lo:.2f}-{hi:.2f}"


def summarize_distribution(rows, group_keys, value_key):
    buckets = defaultdict(list)
    for row in rows:
        value = safe_float(row.get(value_key))
        if value is None:
            continue
        key = tuple(row.get(group_key, "") for group_key in group_keys)
        buckets[key].append(value)

    output = []
    for key, values in buckets.items():
        item = {group_key: key[idx] for idx, group_key in enumerate(group_keys)}
        item.update({
            "n": len(values),
            f"mean_{value_key}": mean(values),
            f"var_{value_key}": variance(values),
            f"p10_{value_key}": percentile(values, 10),
            f"p25_{value_key}": percentile(values, 25),
            f"p50_{value_key}": percentile(values, 50),
            f"p75_{value_key}": percentile(values, 75),
            f"p90_{value_key}": percentile(values, 90),
            f"min_{value_key}": min(values),
            f"max_{value_key}": max(values),
        })
        output.append(item)
    return output


def summarize_histogram(rows, group_keys, value_key, width):
    counts = defaultdict(int)
    totals = defaultdict(int)
    for row in rows:
        value = safe_float(row.get(value_key))
        if value is None:
            continue
        key = tuple(row.get(group_key, "") for group_key in group_keys)
        bin_name = bucket(value, width)
        counts[key + (bin_name,)] += 1
        totals[key] += 1

    output = []
    for key_with_bin, count in counts.items():
        key = key_with_bin[:-1]
        bin_name = key_with_bin[-1]
        item = {group_key: key[idx] for idx, group_key in enumerate(group_keys)}
        item.update({
            "bin": bin_name,
            "n": count,
            "rate": count / max(totals[key], 1),
        })
        output.append(item)
    output.sort(key=lambda row: tuple(str(row.get(group_key, "")) for group_key in group_keys) + (row["bin"],))
    return output


def layer_from_head_key(head_key):
    if not head_key or ":" not in head_key:
        return ""
    return head_key.split(":", 1)[0]


def normalize_head_rows(rows):
    output = []
    for row in rows:
        img_mass = safe_float(row.get("img_mass"))
        if img_mass is None:
            continue
        item = {
            "record_type": row.get("record_type", ""),
            "phase": row.get("phase", ""),
            "question_id": row.get("question_id", ""),
            "step_index": row.get("step_index", ""),
            "layer": str(row.get("layer", layer_from_head_key(row.get("head_key", "")))),
            "head_key": row.get("head_key", ""),
            "head": str(row.get("head", "")),
            "img_mass": img_mass,
            "low_img_mass": safe_float(row.get("low_img_mass"), 1.0 - min(max(img_mass, 0.0), 1.0)),
            "text_mass": safe_float(row.get("text_mass")),
            "strength": safe_float(row.get("strength")),
            "relative_head_output_delta": safe_float(row.get("relative_head_output_delta")),
            "prefill_protected": bool(row.get("prefill_protected")),
        }
        output.append(item)
    return output


def summarize_step_variance(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["phase"], row["question_id"], row["step_index"])].append(row["img_mass"])
    output = []
    for (phase, question_id, step_index), values in grouped.items():
        output.append({
            "phase": phase,
            "question_id": question_id,
            "step_index": step_index,
            "n_heads": len(values),
            "mean_img_mass": mean(values),
            "var_img_mass": variance(values),
            "p10_img_mass": percentile(values, 10),
            "p50_img_mass": percentile(values, 50),
            "p90_img_mass": percentile(values, 90),
        })
    output.sort(key=lambda row: (str(row["phase"]), str(row["question_id"]), int(row["step_index"] or 0)))
    return output


def summarize_head_step_variance(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["phase"], row["head_key"])].append(row["img_mass"])
    output = []
    for (phase, head_key), values in grouped.items():
        output.append({
            "phase": phase,
            "head_key": head_key,
            "layer": layer_from_head_key(head_key),
            "n_steps": len(values),
            "mean_img_mass": mean(values),
            "var_img_mass_across_steps": variance(values),
            "p10_img_mass": percentile(values, 10),
            "p50_img_mass": percentile(values, 50),
            "p90_img_mass": percentile(values, 90),
        })
    output.sort(key=lambda row: (row["phase"], row["layer"], row["head_key"]))
    return output


def summarize_protected(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["phase"], row["prefill_protected"])].append(row)
    output = []
    for (phase, protected), items in grouped.items():
        img_values = [row["img_mass"] for row in items]
        low_values = [row["low_img_mass"] for row in items]
        strengths = [row["strength"] for row in items]
        output.append({
            "phase": phase,
            "prefill_protected": protected,
            "n": len(items),
            "mean_img_mass": mean(img_values),
            "p50_img_mass": percentile(img_values, 50),
            "p90_img_mass": percentile(img_values, 90),
            "mean_low_img_mass": mean(low_values),
            "p50_low_img_mass": percentile(low_values, 50),
            "p90_low_img_mass": percentile(low_values, 90),
            "mean_strength": mean(strengths),
            "p50_strength": percentile(strengths, 50),
            "p90_strength": percentile(strengths, 90),
        })
    return output


def summarize_prefill_protect_records(layer_rows):
    output = []
    for row in layer_rows:
        if row.get("status") != "prefill_protect_recorded":
            continue
        output.append({
            "question_id": row.get("question_id", ""),
            "layer": row.get("layer", ""),
            "prefill_protect_top_k": row.get("prefill_protect_top_k", ""),
            "prefill_protected_n": row.get("prefill_protected_n", ""),
            "prefill_protected_heads": row.get("prefill_protected_heads", ""),
            "mean_prefill_protected_img_mass": row.get("mean_prefill_protected_img_mass", ""),
            "mean_prefill_unprotected_img_mass": row.get("mean_prefill_unprotected_img_mass", ""),
        })
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hist-bin-width", type=float, default=0.05)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rows = list(iter_jsonl(args.diagnostics_jsonl))
    candidate_rows = normalize_head_rows(row for row in rows if row.get("record_type") == "candidate_head")
    selected_rows = normalize_head_rows(row for row in rows if row.get("record_type") == "selected_head")
    active_rows = normalize_head_rows(
        row for row in rows if row.get("record_type") == "selected_head" and row.get("active")
    )
    layer_rows = [row for row in rows if row.get("record_type") == "layer_summary"]

    distribution_source = candidate_rows if candidate_rows else selected_rows
    distribution_source_name = "candidate_head" if candidate_rows else "selected_head"

    files = {
        "img_mass_overall_summary": os.path.join(args.output_dir, "img_mass_overall_summary.csv"),
        "img_mass_layer_summary": os.path.join(args.output_dir, "img_mass_layer_summary.csv"),
        "img_mass_head_step_variance": os.path.join(args.output_dir, "img_mass_head_step_variance.csv"),
        "img_mass_step_variance": os.path.join(args.output_dir, "img_mass_step_variance.csv"),
        "img_mass_histogram": os.path.join(args.output_dir, "img_mass_histogram.csv"),
        "low_img_mass_protected_summary": os.path.join(args.output_dir, "low_img_mass_protected_summary.csv"),
        "prefill_protect_summary": os.path.join(args.output_dir, "prefill_protect_summary.csv"),
    }

    write_csv(files["img_mass_overall_summary"], summarize_distribution(distribution_source, ["record_type", "phase"], "img_mass"))
    write_csv(files["img_mass_layer_summary"], summarize_distribution(distribution_source, ["record_type", "phase", "layer"], "img_mass"))
    write_csv(files["img_mass_head_step_variance"], summarize_head_step_variance(distribution_source))
    write_csv(files["img_mass_step_variance"], summarize_step_variance(distribution_source))
    write_csv(files["img_mass_histogram"], summarize_histogram(distribution_source, ["record_type", "phase"], "img_mass", args.hist_bin_width))
    write_csv(files["low_img_mass_protected_summary"], summarize_protected(active_rows))
    write_csv(files["prefill_protect_summary"], summarize_prefill_protect_records(layer_rows))

    config = {
        "diagnostics_jsonl": args.diagnostics_jsonl,
        "output_dir": args.output_dir,
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
    print("[summary] img_mass overall")
    for row in summarize_distribution(distribution_source, ["record_type", "phase"], "img_mass"):
        print(row)
    print("[summary] protected active")
    for row in summarize_protected(active_rows):
        print(row)


if __name__ == "__main__":
    main()

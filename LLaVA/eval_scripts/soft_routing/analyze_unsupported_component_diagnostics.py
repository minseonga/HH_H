import argparse
import csv
import json
import math
import os
from collections import defaultdict


def safe_float(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    return value


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


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


def mean(values):
    values = [safe_float(value) for value in values]
    return sum(values) / len(values) if values else None


def percentile(values, q):
    values = sorted(safe_float(value) for value in values)
    if not values:
        return None
    idx = int(round((len(values) - 1) * q / 100.0))
    return values[idx]


def summarize_layer_calls(rows):
    output = []
    groups = {"all": rows}
    for phase in sorted({row.get("phase", "") for row in rows}):
        groups[f"phase:{phase}"] = [row for row in rows if row.get("phase") == phase]
    for name, group_rows in groups.items():
        if not group_rows:
            continue
        candidate_rows = [row for row in group_rows if safe_float(row.get("candidate_n")) > 0]
        selected_rows = [row for row in group_rows if safe_float(row.get("selected_n")) > 0]
        active_rows = [row for row in group_rows if safe_float(row.get("active_n")) > 0]
        output.append({
            "group": name,
            "n_layer_calls": len(group_rows),
            "n_candidate_calls": len(candidate_rows),
            "candidate_call_rate": len(candidate_rows) / max(len(group_rows), 1),
            "n_selected_calls": len(selected_rows),
            "selected_call_rate": len(selected_rows) / max(len(group_rows), 1),
            "n_active_calls": len(active_rows),
            "active_call_rate": len(active_rows) / max(len(group_rows), 1),
            "mean_candidate_n": mean(row.get("candidate_n", 0) for row in group_rows),
            "mean_selected_n": mean(row.get("selected_n", 0) for row in group_rows),
            "mean_active_n": mean(row.get("active_n", 0) for row in group_rows),
            "mean_score_high": mean(row.get("score_high", 0) for row in candidate_rows),
            "mean_strength": mean(row.get("mean_strength", 0) for row in active_rows),
            "mean_relative_head_output_delta": mean(
                row.get("mean_relative_head_output_delta", 0) for row in active_rows
            ),
            "p90_relative_head_output_delta": percentile(
                [row.get("mean_relative_head_output_delta", 0) for row in active_rows],
                90,
            ),
        })
    return output


def summarize_selected_heads(rows):
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row.get("phase", ""), row.get("head_key", ""))].append(row)
        buckets[("all", row.get("head_key", ""))].append(row)
    output = []
    for (phase, head_key), items in buckets.items():
        if not head_key:
            continue
        active = [row for row in items if row.get("active")]
        output.append({
            "phase": phase,
            "head_key": head_key,
            "n_selected": len(items),
            "n_active": len(active),
            "active_rate": len(active) / max(len(items), 1),
            "mean_score": mean(row.get("score", 0) for row in items),
            "mean_normalized_score": mean(row.get("normalized_score", 0) for row in items),
            "mean_strength": mean(row.get("strength", 0) for row in active),
            "mean_delta_norm": mean(row.get("delta_norm", 0) for row in active),
            "mean_relative_head_output_delta": mean(row.get("relative_head_output_delta", 0) for row in active),
            "p90_relative_head_output_delta": percentile(
                [row.get("relative_head_output_delta", 0) for row in active],
                90,
            ),
            "mean_unsupported_text_value_norm": mean(
                row.get("unsupported_text_value_norm", 0) for row in active
            ),
            "mean_text_value_norm": mean(row.get("text_value_norm", 0) for row in active),
            "mean_text_mass": mean(row.get("text_mass", 0) for row in active),
            "mean_img_mass": mean(row.get("img_mass", 0) for row in active),
            "mean_text_img_value_cosine": mean(row.get("text_img_value_cosine", 0) for row in active),
        })
    output.sort(
        key=lambda row: (
            row["phase"] != "all",
            -safe_float(row.get("n_active")),
            -safe_float(row.get("mean_relative_head_output_delta")),
        )
    )
    return output


def summarize_steps(layer_rows):
    buckets = defaultdict(list)
    for row in layer_rows:
        key = (row.get("question_id"), row.get("step_index"), row.get("phase"))
        buckets[key].append(row)
    step_rows = []
    for (question_id, step_index, phase), items in buckets.items():
        candidate_calls = [row for row in items if safe_float(row.get("candidate_n")) > 0]
        active_calls = [row for row in items if safe_float(row.get("active_n")) > 0]
        step_rows.append({
            "question_id": question_id,
            "step_index": step_index,
            "phase": phase,
            "n_layer_calls": len(items),
            "n_candidate_layer_calls": len(candidate_calls),
            "n_active_layer_calls": len(active_calls),
            "candidate_layer_rate": len(candidate_calls) / max(len(items), 1),
            "active_layer_rate": len(active_calls) / max(len(items), 1),
            "total_active_heads": sum(int(safe_float(row.get("active_n"))) for row in active_calls),
            "mean_relative_head_output_delta": mean(
                row.get("mean_relative_head_output_delta", 0) for row in active_calls
            ),
        })
    return step_rows


def summarize_step_coverage(step_rows):
    output = []
    groups = {"all": step_rows}
    for phase in sorted({row.get("phase", "") for row in step_rows}):
        groups[f"phase:{phase}"] = [row for row in step_rows if row.get("phase") == phase]
    for name, rows in groups.items():
        active_steps = [row for row in rows if safe_float(row.get("total_active_heads")) > 0]
        output.append({
            "group": name,
            "n_steps": len(rows),
            "n_active_steps": len(active_steps),
            "active_step_rate": len(active_steps) / max(len(rows), 1),
            "mean_candidate_layer_rate": mean(row.get("candidate_layer_rate", 0) for row in rows),
            "mean_active_layer_rate": mean(row.get("active_layer_rate", 0) for row in rows),
            "mean_total_active_heads": mean(row.get("total_active_heads", 0) for row in rows),
            "mean_relative_head_output_delta": mean(
                row.get("mean_relative_head_output_delta", 0) for row in active_steps
            ),
            "p90_relative_head_output_delta": percentile(
                [row.get("mean_relative_head_output_delta", 0) for row in active_steps],
                90,
            ),
        })
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-heads", type=int, default=30)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rows = load_jsonl(args.diagnostics_jsonl)
    layer_rows = [row for row in rows if row.get("record_type") == "layer_summary"]
    selected_rows = [row for row in rows if row.get("record_type") == "selected_head"]
    active_selected_rows = [row for row in selected_rows if row.get("active")]
    step_rows = summarize_steps(layer_rows)

    layer_summary = summarize_layer_calls(layer_rows)
    selected_summary = summarize_selected_heads(selected_rows)
    step_summary = summarize_step_coverage(step_rows)

    write_csv(os.path.join(args.output_dir, "unsupported_component_layer_call_summary.csv"), layer_summary)
    write_csv(os.path.join(args.output_dir, "unsupported_component_selected_head_summary.csv"), selected_summary)
    write_csv(os.path.join(args.output_dir, "unsupported_component_step_rows.csv"), step_rows)
    write_csv(os.path.join(args.output_dir, "unsupported_component_step_summary.csv"), step_summary)

    config = {
        "diagnostics_jsonl": args.diagnostics_jsonl,
        "n_records": len(rows),
        "n_layer_summary_records": len(layer_rows),
        "n_selected_head_records": len(selected_rows),
        "n_active_selected_head_records": len(active_selected_rows),
        "n_steps": len(step_rows),
        "outputs": {
            "layer_call_summary": os.path.join(args.output_dir, "unsupported_component_layer_call_summary.csv"),
            "selected_head_summary": os.path.join(args.output_dir, "unsupported_component_selected_head_summary.csv"),
            "step_summary": os.path.join(args.output_dir, "unsupported_component_step_summary.csv"),
        },
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(json.dumps(config, indent=2))
    print("[summary] layer calls")
    for row in layer_summary:
        print(row)
    print("[summary] step coverage")
    for row in step_summary:
        print(row)
    print("[summary] top selected heads")
    for row in selected_summary[:args.top_heads]:
        if row["phase"] == "all":
            print(row)


if __name__ == "__main__":
    main()

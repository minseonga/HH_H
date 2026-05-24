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


def percentile(values, q):
    values = sorted(value for value in values if value is not None)
    if not values:
        return None
    idx = int(round((len(values) - 1) * q / 100.0))
    return values[idx]


def ratio(num, den, eps=1e-8):
    if num is None or den is None or abs(den) <= eps:
        return None
    return num / den


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


def parse_run(value):
    if "=" not in value:
        label = os.path.basename(os.path.dirname(value.rstrip("/"))) or "run"
        return label, value
    label, path = value.split("=", 1)
    return label, path


def action_delta_norm(row):
    delta = safe_float(row.get("delta_norm"))
    strength = safe_float(row.get("strength"))
    return ratio(delta, strength)


def selected_feature_row(row):
    strength = safe_float(row.get("strength"))
    delta_norm = safe_float(row.get("delta_norm"))
    action_norm = action_delta_norm(row)
    img_norm = safe_float(row.get("img_value_norm"))
    unsupported_norm = safe_float(row.get("unsupported_text_value_norm"))
    return {
        "strength": strength,
        "score": safe_float(row.get("score")),
        "normalized_score": safe_float(row.get("normalized_score")),
        "delta_norm": delta_norm,
        "action_delta_norm": action_norm,
        "relative_head_output_delta": safe_float(row.get("relative_head_output_delta")),
        "img_value_norm": img_norm,
        "unsupported_text_value_norm": unsupported_norm,
        "unsupported_head_output_ratio": safe_float(row.get("unsupported_head_output_ratio")),
        "text_value_norm": safe_float(row.get("text_value_norm")),
        "img_mass": safe_float(row.get("img_mass")),
        "low_img_mass": safe_float(row.get("low_img_mass")),
        "text_mass": safe_float(row.get("text_mass")),
        "text_head_agreement": safe_float(row.get("text_head_agreement")),
        "text_head_disagreement": safe_float(row.get("text_head_disagreement")),
        "text_mass_x_disagreement": safe_float(row.get("text_mass_x_disagreement")),
        "object_logit_agreement": safe_float(row.get("object_logit_agreement")),
        "object_logit_disagreement": safe_float(row.get("object_logit_disagreement")),
        "text_mass_x_object_logit_disagreement": safe_float(row.get("text_mass_x_object_logit_disagreement")),
        "visual_value_ratio": safe_float(row.get("visual_value_ratio")),
        "text_img_value_cosine": safe_float(row.get("text_img_value_cosine")),
        "action_over_unsupported_norm": ratio(action_norm, unsupported_norm),
        "action_over_img_norm": ratio(action_norm, img_norm),
        "img_over_unsupported_norm": ratio(img_norm, unsupported_norm),
    }


def summarize_values(prefix, rows, keys):
    output = {}
    for key in keys:
        values = [row.get(key) for row in rows]
        output[f"mean_{prefix}{key}"] = mean(values)
        output[f"p50_{prefix}{key}"] = percentile(values, 50)
        output[f"p90_{prefix}{key}"] = percentile(values, 90)
    return output


def summarize_run(label, path, rows):
    layer_rows = [row for row in rows if row.get("record_type") == "layer_summary"]
    selected_rows = [row for row in rows if row.get("record_type") == "selected_head"]
    active_rows = [row for row in selected_rows if row.get("active")]
    inactive_rows = [row for row in selected_rows if not row.get("active")]
    feature_rows = [selected_feature_row(row) for row in active_rows]

    steps = defaultdict(lambda: {"selected": 0, "active": 0})
    for row in selected_rows:
        key = (row.get("question_id"), row.get("step_index"), row.get("phase"))
        steps[key]["selected"] += 1
        if row.get("active"):
            steps[key]["active"] += 1

    active_steps = [item for item in steps.values() if item["active"] > 0]
    run_actions = sorted({row.get("action", "") for row in selected_rows if row.get("action")})
    run_risk = sorted({row.get("risk_feature", "") for row in selected_rows if row.get("risk_feature")})
    run_score_norm = sorted({row.get("score_norm", "") for row in selected_rows if row.get("score_norm")})
    run_modes = sorted({row.get("mode", "") for row in selected_rows if row.get("mode")})

    summary = {
        "run": label,
        "path": path,
        "action": "|".join(run_actions),
        "risk_feature": "|".join(run_risk),
        "score_norm": "|".join(run_score_norm),
        "mode": "|".join(run_modes),
        "n_records": len(rows),
        "n_layer_summary_records": len(layer_rows),
        "n_selected_records": len(selected_rows),
        "n_active_records": len(active_rows),
        "n_inactive_selected_records": len(inactive_rows),
        "selected_active_rate": len(active_rows) / max(len(selected_rows), 1),
        "n_selected_steps": len(steps),
        "n_active_steps": len(active_steps),
        "active_step_rate": len(active_steps) / max(len(steps), 1),
        "mean_active_heads_per_active_step": mean([item["active"] for item in active_steps]),
    }
    keys = [
        "strength",
        "score",
        "normalized_score",
        "delta_norm",
        "action_delta_norm",
        "relative_head_output_delta",
        "img_value_norm",
        "unsupported_text_value_norm",
        "unsupported_head_output_ratio",
        "action_over_unsupported_norm",
        "action_over_img_norm",
        "img_over_unsupported_norm",
        "img_mass",
        "low_img_mass",
        "text_mass",
        "text_head_agreement",
        "text_head_disagreement",
        "text_mass_x_disagreement",
        "object_logit_agreement",
        "object_logit_disagreement",
        "text_mass_x_object_logit_disagreement",
        "visual_value_ratio",
        "text_img_value_cosine",
    ]
    summary.update(summarize_values("", feature_rows, keys))
    return summary, selected_rows, active_rows


def summarize_heads(label, active_rows):
    buckets = defaultdict(list)
    for row in active_rows:
        buckets[row.get("head_key", "")].append(selected_feature_row(row))

    output = []
    keys = [
        "strength",
        "delta_norm",
        "action_delta_norm",
        "relative_head_output_delta",
        "img_value_norm",
        "unsupported_text_value_norm",
        "unsupported_head_output_ratio",
        "action_over_unsupported_norm",
        "action_over_img_norm",
        "img_over_unsupported_norm",
        "img_mass",
        "low_img_mass",
        "visual_value_ratio",
        "text_mass",
        "text_head_agreement",
        "text_head_disagreement",
        "text_mass_x_disagreement",
        "object_logit_agreement",
        "object_logit_disagreement",
        "text_mass_x_object_logit_disagreement",
    ]
    for head_key, rows in buckets.items():
        if not head_key:
            continue
        item = {
            "run": label,
            "head_key": head_key,
            "n_active": len(rows),
        }
        item.update(summarize_values("", rows, keys))
        output.append(item)
    output.sort(
        key=lambda row: (
            row["run"],
            -row["n_active"],
            -(row.get("mean_relative_head_output_delta") or 0.0),
        )
    )
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Run diagnostics as label=path entries.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-heads", type=int, default=30)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    run_summaries = []
    head_summaries = []
    for run_arg in args.runs:
        label, path = parse_run(run_arg)
        rows = list(iter_jsonl(path))
        summary, _, active_rows = summarize_run(label, path, rows)
        run_summaries.append(summary)
        head_summaries.extend(summarize_heads(label, active_rows))

    run_path = os.path.join(args.output_dir, "unsupported_action_run_summary.csv")
    head_path = os.path.join(args.output_dir, "unsupported_action_head_summary.csv")
    write_csv(run_path, run_summaries)
    write_csv(head_path, head_summaries)

    config = {
        "runs": args.runs,
        "run_summary": run_path,
        "head_summary": head_path,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(json.dumps(config, indent=2))
    print("[summary] runs")
    for row in run_summaries:
        print(row)
    print("[summary] top heads")
    for row in head_summaries[: args.top_heads]:
        print(row)


if __name__ == "__main__":
    main()

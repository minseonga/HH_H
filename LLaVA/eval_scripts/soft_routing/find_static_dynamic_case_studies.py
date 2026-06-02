#!/usr/bin/env python3
import argparse
import collections
import csv
import json
import os


def load_eval(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {str(row["image_id"]): row for row in data["sentences"]}, data.get("overall_metrics", {})


def chair_s(row):
    return int(row.get("metrics", {}).get("CHAIRs", 0))


def chair_i(row):
    return float(row.get("metrics", {}).get("CHAIRi", 0.0))


def words(row, key):
    out = []
    for item in row.get(key, []):
        if isinstance(item, (list, tuple)) and item:
            out.append(str(item[0]))
        else:
            out.append(str(item))
    return out


def word_nodes(row, key):
    out = []
    for item in row.get(key, []):
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            out.append(f"{item[0]}->{item[1]}")
        elif isinstance(item, (list, tuple)) and item:
            out.append(str(item[0]))
        else:
            out.append(str(item))
    return out


def compact(text):
    return " ".join(str(text).split())


def truncate(text, n=420):
    text = compact(text)
    return text if len(text) <= n else text[: n - 1].rstrip() + "..."


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields or ["empty"])
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, rows, top_n):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Static-Fail / Dynamic-Success Case Studies\n\n")
        f.write("Selection rule: static CHAIRs=1 and dynamic CHAIRs=0 on the same image.\n\n")
        f.write("Use these as cherry-picked qualitative cases, not aggregate evidence.\n\n")
        for idx, row in enumerate(rows[:top_n], 1):
            f.write(f"## Case {idx}: image {row['image_id']} ({row['case_type']})\n\n")
            f.write(f"- GT objects: `{row['gt_words']}`\n")
            f.write(f"- Greedy hallucinated: `{row['greedy_hall_words']}`\n")
            f.write(f"- Static hallucinated: `{row['static_hall_words']}`\n")
            f.write(f"- Dynamic hallucinated: `{row['dynamic_hall_words']}`\n")
            f.write(
                f"- Grounded mention counts G/S/D: {row['greedy_grounded_count']} / "
                f"{row['static_grounded_count']} / {row['dynamic_grounded_count']}\n\n"
            )
            f.write("| method | CHAIRs | CHAIRi | hallucinated words | caption |\n")
            f.write("|---|---:|---:|---|---|\n")
            for method in ["greedy", "static", "dynamic"]:
                f.write(
                    f"| {method} | {row[f'{method}_CHAIRs']} | {row[f'{method}_CHAIRi']:.3f} | "
                    f"`{row[f'{method}_hall_words']}` | {row[f'{method}_caption_short']} |\n"
                )
            f.write("\n")


def node_counts(row, key="mscoco_non_hallucinated_words"):
    counts = collections.Counter()
    for item in row.get(key, []):
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            counts[str(item[1])] += 1
        elif isinstance(item, (list, tuple)) and item:
            counts[str(item[0])] += 1
        else:
            counts[str(item)] += 1
    return counts


def write_grounded_rescue_markdown(path, rows, top_n):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Static Over-Suppression / Dynamic Grounded-Rescue Case Studies\n\n")
        f.write(
            "Selection rule: a grounded object node present in greedy is reduced or removed by static, "
            "then increased again by dynamic on the same image.\n\n"
        )
        f.write("Use these as cherry-picked qualitative cases, not aggregate evidence.\n\n")
        for idx, row in enumerate(rows[:top_n], 1):
            f.write(f"## Case {idx}: image {row['image_id']} ({row['case_type']})\n\n")
            f.write(f"- GT objects: `{row['gt_words']}`\n")
            f.write(f"- Rescued grounded nodes: `{row['rescued_nodes']}`\n")
            f.write(
                f"- Grounded mention counts G/S/D: {row['greedy_grounded_count']} / "
                f"{row['static_grounded_count']} / {row['dynamic_grounded_count']}\n"
            )
            f.write(f"- CHAIRs G/S/D: {row['greedy_CHAIRs']} / {row['static_CHAIRs']} / {row['dynamic_CHAIRs']}\n\n")
            f.write("| method | CHAIRs | CHAIRi | grounded node counts | hallucinated words | caption |\n")
            f.write("|---|---:|---:|---|---|---|\n")
            for method in ["greedy", "static", "dynamic"]:
                f.write(
                    f"| {method} | {row[f'{method}_CHAIRs']} | {row[f'{method}_CHAIRi']:.3f} | "
                    f"`{row[f'{method}_grounded_node_counts']}` | "
                    f"`{row[f'{method}_hall_words']}` | {row[f'{method}_caption_short']} |\n"
                )
            f.write("\n")


def classify_case(greedy_row, static_row):
    greedy_hall = chair_s(greedy_row) if greedy_row else None
    if greedy_hall == 1:
        return "static_retains_greedy_hallucination"
    return "static_induces_hallucination"


def build_rows(greedy, static, dynamic):
    rows = []
    common = sorted(set(static) & set(dynamic), key=lambda x: int(x) if x.isdigit() else x)
    for image_id in common:
        static_row = static[image_id]
        dynamic_row = dynamic[image_id]
        greedy_row = greedy.get(image_id)
        if chair_s(static_row) != 1 or chair_s(dynamic_row) != 0:
            continue
        greedy_hall_count = chair_s(greedy_row) if greedy_row else 0
        static_grounded = len(words(static_row, "mscoco_non_hallucinated_words"))
        dynamic_grounded = len(words(dynamic_row, "mscoco_non_hallucinated_words"))
        greedy_grounded = len(words(greedy_row, "mscoco_non_hallucinated_words")) if greedy_row else 0
        row = {
            "image_id": image_id,
            "image": static_row.get("image"),
            "case_type": classify_case(greedy_row, static_row) if greedy_row else "static_fail_dynamic_success",
            "gt_words": ", ".join(static_row.get("mscoco_gt_words", [])),
            "greedy_hall_words": ", ".join(word_nodes(greedy_row, "mscoco_hallucinated_words")) if greedy_row else "",
            "static_hall_words": ", ".join(word_nodes(static_row, "mscoco_hallucinated_words")),
            "dynamic_hall_words": ", ".join(word_nodes(dynamic_row, "mscoco_hallucinated_words")),
            "greedy_grounded_count": greedy_grounded,
            "static_grounded_count": static_grounded,
            "dynamic_grounded_count": dynamic_grounded,
            "grounded_count_delta_dynamic_minus_static": dynamic_grounded - static_grounded,
            "static_hall_count": len(words(static_row, "mscoco_hallucinated_words")),
            "greedy_hall_count": greedy_hall_count,
            "greedy_CHAIRs": chair_s(greedy_row) if greedy_row else "",
            "greedy_CHAIRi": chair_i(greedy_row) if greedy_row else 0.0,
            "static_CHAIRs": chair_s(static_row),
            "static_CHAIRi": chair_i(static_row),
            "dynamic_CHAIRs": chair_s(dynamic_row),
            "dynamic_CHAIRi": chair_i(dynamic_row),
            "greedy_caption": compact(greedy_row.get("caption", "")) if greedy_row else "",
            "static_caption": compact(static_row.get("caption", "")),
            "dynamic_caption": compact(dynamic_row.get("caption", "")),
            "greedy_caption_short": truncate(greedy_row.get("caption", "")) if greedy_row else "",
            "static_caption_short": truncate(static_row.get("caption", "")),
            "dynamic_caption_short": truncate(dynamic_row.get("caption", "")),
        }
        rows.append(row)
    rows.sort(
        key=lambda row: (
            row["case_type"] != "static_retains_greedy_hallucination",
            -row["grounded_count_delta_dynamic_minus_static"],
            -row["static_hall_count"],
            -row["static_CHAIRi"],
        )
    )
    return rows


def grounded_rescue_case_type(greedy_row, static_row, dynamic_row):
    if chair_s(greedy_row) == 0 and chair_s(static_row) == 0 and chair_s(dynamic_row) == 0:
        return "clean_grounded_rescue_no_hallucination"
    if chair_s(static_row) == 1 and chair_s(dynamic_row) == 0:
        return "grounded_rescue_plus_hallucination_fix"
    if chair_s(static_row) == 0 and chair_s(dynamic_row) == 0:
        return "grounded_rescue_after_static_hall_fix"
    return "grounded_rescue"


def build_grounded_rescue_rows(greedy, static, dynamic):
    rows = []
    common = sorted(set(greedy) & set(static) & set(dynamic), key=lambda x: int(x) if x.isdigit() else x)
    for image_id in common:
        greedy_row = greedy[image_id]
        static_row = static[image_id]
        dynamic_row = dynamic[image_id]
        greedy_counts = node_counts(greedy_row)
        static_counts = node_counts(static_row)
        dynamic_counts = node_counts(dynamic_row)

        rescued = []
        full_rescued = []
        for node, greedy_n in greedy_counts.items():
            static_n = static_counts.get(node, 0)
            dynamic_n = dynamic_counts.get(node, 0)
            if static_n < greedy_n and dynamic_n > static_n:
                rescued.append((node, greedy_n, static_n, dynamic_n))
                if dynamic_n >= greedy_n:
                    full_rescued.append((node, greedy_n, static_n, dynamic_n))
        if not rescued:
            continue

        row = {
            "image_id": image_id,
            "image": static_row.get("image"),
            "case_type": grounded_rescue_case_type(greedy_row, static_row, dynamic_row),
            "gt_words": ", ".join(greedy_row.get("mscoco_gt_words", [])),
            "rescued_nodes": ", ".join(f"{node}({g}->{s}->{d})" for node, g, s, d in rescued),
            "fully_rescued_nodes": ", ".join(f"{node}({g}->{s}->{d})" for node, g, s, d in full_rescued),
            "n_rescued_nodes": len(rescued),
            "n_fully_rescued_nodes": len(full_rescued),
            "rescued_mention_gain": sum(d - s for _, _, s, d in rescued),
            "static_loss_from_greedy": sum(g - s for _, g, s, _ in rescued),
            "greedy_grounded_count": sum(greedy_counts.values()),
            "static_grounded_count": sum(static_counts.values()),
            "dynamic_grounded_count": sum(dynamic_counts.values()),
            "greedy_grounded_node_counts": ", ".join(f"{k}:{v}" for k, v in sorted(greedy_counts.items())),
            "static_grounded_node_counts": ", ".join(f"{k}:{v}" for k, v in sorted(static_counts.items())),
            "dynamic_grounded_node_counts": ", ".join(f"{k}:{v}" for k, v in sorted(dynamic_counts.items())),
            "greedy_hall_words": ", ".join(word_nodes(greedy_row, "mscoco_hallucinated_words")),
            "static_hall_words": ", ".join(word_nodes(static_row, "mscoco_hallucinated_words")),
            "dynamic_hall_words": ", ".join(word_nodes(dynamic_row, "mscoco_hallucinated_words")),
            "greedy_CHAIRs": chair_s(greedy_row),
            "greedy_CHAIRi": chair_i(greedy_row),
            "static_CHAIRs": chair_s(static_row),
            "static_CHAIRi": chair_i(static_row),
            "dynamic_CHAIRs": chair_s(dynamic_row),
            "dynamic_CHAIRi": chair_i(dynamic_row),
            "greedy_caption": compact(greedy_row.get("caption", "")),
            "static_caption": compact(static_row.get("caption", "")),
            "dynamic_caption": compact(dynamic_row.get("caption", "")),
            "greedy_caption_short": truncate(greedy_row.get("caption", "")),
            "static_caption_short": truncate(static_row.get("caption", "")),
            "dynamic_caption_short": truncate(dynamic_row.get("caption", "")),
        }
        rows.append(row)
    case_priority = {
        "clean_grounded_rescue_no_hallucination": 0,
        "grounded_rescue_plus_hallucination_fix": 1,
        "grounded_rescue_after_static_hall_fix": 2,
        "grounded_rescue": 3,
    }
    rows.sort(
        key=lambda row: (
            case_priority.get(row["case_type"], 99),
            -row["n_fully_rescued_nodes"],
            -row["rescued_mention_gain"],
            -row["static_loss_from_greedy"],
        )
    )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--greedy-json", required=True)
    parser.add_argument("--static-json", required=True)
    parser.add_argument("--dynamic-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    greedy, greedy_metrics = load_eval(args.greedy_json)
    static, static_metrics = load_eval(args.static_json)
    dynamic, dynamic_metrics = load_eval(args.dynamic_json)
    rows = build_rows(greedy, static, dynamic)
    grounded_rescue_rows = build_grounded_rescue_rows(greedy, static, dynamic)
    os.makedirs(args.output_dir, exist_ok=True)
    write_csv(os.path.join(args.output_dir, "static_fail_dynamic_success_cases.csv"), rows)
    write_markdown(os.path.join(args.output_dir, "static_fail_dynamic_success_cases.md"), rows, args.top_n)
    write_csv(os.path.join(args.output_dir, "static_over_suppression_grounded_rescue_cases.csv"), grounded_rescue_rows)
    write_grounded_rescue_markdown(
        os.path.join(args.output_dir, "static_over_suppression_grounded_rescue_cases.md"),
        grounded_rescue_rows,
        args.top_n,
    )
    summary = {
        "greedy_json": args.greedy_json,
        "static_json": args.static_json,
        "dynamic_json": args.dynamic_json,
        "greedy_metrics": greedy_metrics,
        "static_metrics": static_metrics,
        "dynamic_metrics": dynamic_metrics,
        "n_cases": len(rows),
        "n_static_retains_greedy_hallucination": sum(
            row["case_type"] == "static_retains_greedy_hallucination" for row in rows
        ),
        "n_static_induces_hallucination": sum(row["case_type"] == "static_induces_hallucination" for row in rows),
        "n_grounded_rescue_cases": len(grounded_rescue_rows),
        "n_clean_grounded_rescue_no_hallucination": sum(
            row["case_type"] == "clean_grounded_rescue_no_hallucination" for row in grounded_rescue_rows
        ),
        "n_grounded_rescue_plus_hallucination_fix": sum(
            row["case_type"] == "grounded_rescue_plus_hallucination_fix" for row in grounded_rescue_rows
        ),
    }
    with open(os.path.join(args.output_dir, "static_fail_dynamic_success_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
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
    os.makedirs(args.output_dir, exist_ok=True)
    write_csv(os.path.join(args.output_dir, "static_fail_dynamic_success_cases.csv"), rows)
    write_markdown(os.path.join(args.output_dir, "static_fail_dynamic_success_cases.md"), rows, args.top_n)
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
    }
    with open(os.path.join(args.output_dir, "static_fail_dynamic_success_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

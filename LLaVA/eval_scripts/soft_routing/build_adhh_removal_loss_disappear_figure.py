#!/usr/bin/env python3
import argparse
import collections
import csv
import html
import json
import os


def load_sentences(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "sentences" in data:
        return data["sentences"], data.get("overall_metrics", {})
    raise ValueError(f"unsupported eval results file: {path}")


def row_key(row):
    for key in ("image_id", "question_id", "image"):
        if row.get(key) is not None:
            return str(row[key])
    return ""


def node_words(items):
    out = []
    for item in items or []:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            out.append(str(item[1]))
        else:
            out.append(str(item))
    return out


def counter_diff(base_words, target_words):
    base = collections.Counter(base_words)
    target = collections.Counter(target_words)
    removed = base - target
    added = target - base
    return removed, added


def caption(row):
    return row.get("caption") or row.get("text") or ""


def analyze(base_rows, target_rows):
    base_by_key = {row_key(row): row for row in base_rows if row_key(row)}
    target_by_key = {row_key(row): row for row in target_rows if row_key(row)}
    rows = []
    totals = collections.Counter()

    for key in sorted(set(base_by_key) & set(target_by_key)):
        base = base_by_key[key]
        target = target_by_key[key]
        base_hall = node_words(base.get("mscoco_hallucinated_words"))
        target_hall = node_words(target.get("mscoco_hallucinated_words"))
        base_ground = node_words(base.get("mscoco_non_hallucinated_words"))
        target_ground = node_words(target.get("mscoco_non_hallucinated_words"))
        removed_hall, added_hall = counter_diff(base_hall, target_hall)
        removed_ground, added_ground = counter_diff(base_ground, target_ground)
        base_ground_counts = collections.Counter(base_ground)
        target_ground_counts = collections.Counter(target_ground)
        reduced_ground_nodes = 0
        disappeared_ground_nodes = 0
        disappeared_ground_mentions = 0
        for word, base_count in base_ground_counts.items():
            target_count = target_ground_counts.get(word, 0)
            if target_count < base_count:
                reduced_ground_nodes += 1
            if target_count == 0:
                disappeared_ground_nodes += 1
                disappeared_ground_mentions += base_count

        row = {
            "key": key,
            "image": base.get("image") or target.get("image"),
            "base_hall_mentions": len(base_hall),
            "target_hall_mentions": len(target_hall),
            "removed_hall_mentions": sum(removed_hall.values()),
            "added_hall_mentions": sum(added_hall.values()),
            "base_ground_mentions": len(base_ground),
            "target_ground_mentions": len(target_ground),
            "removed_ground_mentions": sum(removed_ground.values()),
            "added_ground_mentions": sum(added_ground.values()),
            "base_ground_nodes": len(base_ground_counts),
            "reduced_ground_nodes": reduced_ground_nodes,
            "disappeared_ground_nodes": disappeared_ground_nodes,
            "disappeared_ground_mentions": disappeared_ground_mentions,
            "removed_hall_words": "; ".join(
                f"{word}x{count}" if count > 1 else word for word, count in sorted(removed_hall.items())
            ),
            "removed_ground_words": "; ".join(
                f"{word}x{count}" if count > 1 else word for word, count in sorted(removed_ground.items())
            ),
            "added_hall_words": "; ".join(
                f"{word}x{count}" if count > 1 else word for word, count in sorted(added_hall.items())
            ),
            "added_ground_words": "; ".join(
                f"{word}x{count}" if count > 1 else word for word, count in sorted(added_ground.items())
            ),
            "base_caption": caption(base),
            "target_caption": caption(target),
            "base_gt_words": ", ".join(str(item) for item in base.get("mscoco_gt_words") or []),
        }
        rows.append(row)
        for field in (
            "base_hall_mentions",
            "target_hall_mentions",
            "removed_hall_mentions",
            "added_hall_mentions",
            "base_ground_mentions",
            "target_ground_mentions",
            "removed_ground_mentions",
            "added_ground_mentions",
            "base_ground_nodes",
            "reduced_ground_nodes",
            "disappeared_ground_nodes",
            "disappeared_ground_mentions",
        ):
            totals[field] += row[field]
        totals["images"] += 1
        totals["images_with_hall_removed"] += int(row["removed_hall_mentions"] > 0)
        totals["images_with_ground_lost"] += int(row["removed_ground_mentions"] > 0)
    return rows, totals


def rate(num, den):
    return float(num) / float(den) if den else 0.0


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


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def svg_text(x, y, value, size=13, fill="#1f2937", anchor="start", weight="400"):
    return (
        f'<text x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" fill="{fill}" text-anchor="{anchor}" font-weight="{weight}">'
        f'{html.escape(str(value))}</text>'
    )


def svg_rect(x, y, w, h, fill, stroke="none", opacity=1.0, radius=0):
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" ry="{radius}" '
        f'fill="{fill}" stroke="{stroke}" opacity="{opacity}"/>'
    )


def svg_line(x1, y1, x2, y2, stroke="#d0d7de", width=1):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{width}"/>'


def draw_bar_figure(path, summary):
    colors = {
        "ground": "#2e8b57",
        "ground_light": "#8fd0ae",
        "dark": "#111827",
        "muted": "#667085",
        "grid": "#d0d7de",
        "panel": "#f8fafc",
    }
    ground_rate = summary["grounded_loss_rate"] * 100.0
    ground_node_reduction_rate = summary["grounded_reduced_node_rate"] * 100.0
    image_ground_rate = summary["images_with_ground_lost_rate"] * 100.0

    width, height = 820, 410
    chart_x, chart_y, chart_w, chart_h = 132, 98, 620, 205
    ymax = 55.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        svg_rect(0, 0, width, height, "white"),
        svg_text(width / 2, 34, "Grounded object reduction under AD-HH hard", 18, colors["dark"], "middle", "800"),
        svg_text(width / 2, 58, "Greedy -> AD-HH hard tau=0.4, COCO n=500; grounded-only comparison.", 12, colors["muted"], "middle"),
        svg_rect(48, 76, 735, 300, colors["panel"], "#e5e7eb", 1, 8),
    ]

    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        yy = chart_y + chart_h - chart_h * tick
        parts.append(svg_line(chart_x, yy, chart_x + chart_w, yy, colors["grid"], 1))
        parts.append(svg_text(chart_x - 12, yy + 4, f"{ymax * tick:.0f}%", 11, colors["muted"], "end"))
    parts.append(svg_line(chart_x, chart_y, chart_x, chart_y + chart_h, "#98a2b3", 1.2))
    parts.append(svg_line(chart_x, chart_y + chart_h, chart_x + chart_w, chart_y + chart_h, "#98a2b3", 1.2))

    bars = [
        (
            "grounded mentions\nwith count decrease",
            ground_rate,
            colors["ground"],
            f"{summary['removed_ground_mentions']}/{summary['base_ground_mentions']}",
        ),
        (
            "grounded object nodes\nwith count decrease",
            ground_node_reduction_rate,
            colors["ground_light"],
            f"{summary['reduced_ground_nodes']}/{summary['base_ground_nodes']}",
        ),
    ]
    for idx, (label, value, color, count_text) in enumerate(bars):
        cx = chart_x + chart_w * (0.30 + idx * 0.40)
        bar_w = 112
        bar_h = chart_h * value / ymax
        parts.append(svg_rect(cx - bar_w / 2, chart_y + chart_h - bar_h, bar_w, bar_h, color, opacity=0.88, radius=6))
        parts.append(svg_text(cx, chart_y + chart_h - bar_h - 12, f"{value:.1f}%", 18, colors["dark"], "middle", "800"))
        parts.append(svg_text(cx, chart_y + chart_h - bar_h - 32, count_text, 12, colors["muted"], "middle", "700"))
        for line_idx, text_line in enumerate(label.split("\n")):
            parts.append(svg_text(cx, chart_y + chart_h + 26 + line_idx * 17, text_line, 12, colors["muted"], "middle", "700"))

    parts.append(svg_text(70, 200, "grounded reduction rate", 12, colors["muted"], "middle", "700"))
    parts.append(svg_text(86, 348, f"Images with any grounded mention decrease: {image_ground_rate:.1f}%", 12, colors["ground"], "start", "700"))
    parts.append(svg_text(width / 2, 392, "Takeaway: hard suppression can reduce grounded object realization.", 12, colors["dark"], "middle", "700"))
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def choose_disappear_case(rows, preferred_key):
    if preferred_key:
        for row in rows:
            if row["key"] == str(preferred_key):
                return row
    candidates = [row for row in rows if row["removed_ground_mentions"] > 0 and row["added_hall_mentions"] == 0]
    if not candidates:
        candidates = [row for row in rows if row["removed_ground_mentions"] > 0]
    candidates.sort(
        key=lambda row: (
            row["removed_ground_mentions"],
            -row["added_hall_mentions"],
            -row["target_hall_mentions"],
            -len(row["target_caption"]),
        ),
        reverse=True,
    )
    return candidates[0] if candidates else {}


def write_case_markdown(path, case, base_name, target_name):
    text = f"""# Grounded Disappearance Case

Image: `{case.get('image')}`

GT objects: `{case.get('base_gt_words')}`

Removed grounded words under `{target_name}`: `{case.get('removed_ground_words')}`

Added hallucinated words under `{target_name}`: `{case.get('added_hall_words')}`

## {base_name}

{case.get('base_caption')}

## {target_name}

{case.get('target_caption')}
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--base-name", default="greedy")
    parser.add_argument("--target-name", default="adhh_hard")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--preferred-case-key", default="")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    base_rows, base_metrics = load_sentences(args.base)
    target_rows, target_metrics = load_sentences(args.target)
    rows, totals = analyze(base_rows, target_rows)
    summary = {
        "base": args.base,
        "target": args.target,
        "base_name": args.base_name,
        "target_name": args.target_name,
        "base_metrics": base_metrics,
        "target_metrics": target_metrics,
        "n_common": totals["images"],
        "base_hall_mentions": totals["base_hall_mentions"],
        "target_hall_mentions": totals["target_hall_mentions"],
        "removed_hall_mentions": totals["removed_hall_mentions"],
        "added_hall_mentions": totals["added_hall_mentions"],
        "hallucinated_removal_rate": rate(totals["removed_hall_mentions"], totals["base_hall_mentions"]),
        "base_ground_mentions": totals["base_ground_mentions"],
        "target_ground_mentions": totals["target_ground_mentions"],
        "removed_ground_mentions": totals["removed_ground_mentions"],
        "added_ground_mentions": totals["added_ground_mentions"],
        "grounded_loss_rate": rate(totals["removed_ground_mentions"], totals["base_ground_mentions"]),
        "base_ground_nodes": totals["base_ground_nodes"],
        "reduced_ground_nodes": totals["reduced_ground_nodes"],
        "grounded_reduced_node_rate": rate(totals["reduced_ground_nodes"], totals["base_ground_nodes"]),
        "partial_loss_ground_nodes": totals["reduced_ground_nodes"] - totals["disappeared_ground_nodes"],
        "grounded_partial_loss_node_rate": rate(
            totals["reduced_ground_nodes"] - totals["disappeared_ground_nodes"],
            totals["base_ground_nodes"],
        ),
        "disappeared_ground_nodes": totals["disappeared_ground_nodes"],
        "grounded_disappeared_node_rate": rate(totals["disappeared_ground_nodes"], totals["base_ground_nodes"]),
        "disappeared_ground_mentions": totals["disappeared_ground_mentions"],
        "grounded_disappeared_mention_rate": rate(totals["disappeared_ground_mentions"], totals["base_ground_mentions"]),
        "grounded_loss_due_to_disappearance_fraction": rate(
            totals["disappeared_ground_mentions"], totals["removed_ground_mentions"]
        ),
        "images_with_hall_removed": totals["images_with_hall_removed"],
        "images_with_hall_removed_rate": rate(totals["images_with_hall_removed"], totals["images"]),
        "images_with_ground_lost": totals["images_with_ground_lost"],
        "images_with_ground_lost_rate": rate(totals["images_with_ground_lost"], totals["images"]),
    }
    case = choose_disappear_case(rows, args.preferred_case_key)
    summary["disappear_case"] = {
        key: case.get(key)
        for key in (
            "key",
            "image",
            "base_gt_words",
            "removed_ground_words",
            "added_hall_words",
            "base_caption",
            "target_caption",
        )
    }

    write_csv(os.path.join(args.output_dir, "adhh_removal_loss_rows.csv"), rows)
    write_json(os.path.join(args.output_dir, "adhh_removal_loss_summary.json"), summary)
    write_csv(os.path.join(args.output_dir, "adhh_removal_loss_summary.csv"), [summary])
    draw_bar_figure(os.path.join(args.output_dir, "adhh_removal_loss_bar.svg"), summary)
    if case:
        write_case_markdown(
            os.path.join(args.output_dir, "grounded_disappear_case.md"),
            case,
            args.base_name,
            args.target_name,
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

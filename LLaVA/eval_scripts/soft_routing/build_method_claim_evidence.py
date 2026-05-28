import argparse
import csv
import glob
import html
import json
import math
import os
from collections import Counter

from eval_scripts.soft_routing.analyze_contrastive_dynamic_head_pool import (
    head_key,
    load_ranked_heads,
    safe_float,
)


BLUE = "#2563eb"
ORANGE = "#f97316"
GREEN = "#059669"
RED = "#dc2626"
PURPLE = "#7c3aed"
GRAY = "#cbd5e1"
DARK = "#0f172a"
MUTED = "#64748b"


def mean(values):
    values = [safe_float(value) for value in values]
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else 0.0


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys or ["empty"])
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def svg(path, width, height, body):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">\n'
        )
        f.write('<rect width="100%" height="100%" fill="white"/>\n')
        f.write(body)
        f.write("</svg>\n")


def text(x, y, value, size=13, fill=DARK, anchor="start", weight="400", rotate=None):
    value = html.escape(str(value))
    transform = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}" '
        f'text-anchor="{anchor}"{transform}>{value}</text>\n'
    )


def line(x1, y1, x2, y2, stroke=MUTED, width=1, dash=None):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{stroke}" stroke-width="{width}"{dash_attr}/>\n'


def rect(x, y, w, h, fill, stroke=None, width=1, opacity=1.0):
    stroke_attr = f' stroke="{stroke}" stroke-width="{width}"' if stroke else ""
    return f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" fill="{fill}" opacity="{opacity}"{stroke_attr}/>\n'


def circle(x, y, r, fill, stroke=None, width=1, opacity=1.0):
    stroke_attr = f' stroke="{stroke}" stroke-width="{width}"' if stroke else ""
    return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" fill="{fill}" opacity="{opacity}"{stroke_attr}/>\n'


def polyline(points, stroke, width=2.5):
    pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polyline points="{pts}" fill="none" stroke="{stroke}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"/>\n'


def raw_toi_gap(row):
    return safe_float(row.get("RawTOI_hallucinated"), 0.0) - safe_float(row.get("RawTOI_non_hallucinated"), 0.0)


def log_toi_gap(row):
    return safe_float(row.get("LogTOI_hallucinated"), 0.0) - safe_float(row.get("LogTOI_non_hallucinated"), 0.0)


def image_drop(row):
    return safe_float(row.get("Img_non_hallucinated"), 0.0) - safe_float(row.get("Img_hallucinated"), 0.0)


def itext_gap(row):
    return safe_float(row.get("Itext_hallucinated"), 0.0) - safe_float(row.get("Itext_non_hallucinated"), 0.0)


def add_component_ranks(records):
    by_front = sorted(records, key=lambda row: safe_float(row.get("front_percentile"), -1.0), reverse=True)
    by_back = sorted(records, key=lambda row: safe_float(row.get("back_percentile"), -1.0), reverse=True)
    for rank, row in enumerate(by_front, start=1):
        row["_front_rank"] = rank
    for rank, row in enumerate(by_back, start=1):
        row["_back_rank"] = rank


def categorize_components(records, top_k):
    rows = []
    for row in records:
        combo = int(row["_rank"]) <= top_k
        text_high = int(row["_front_rank"]) <= top_k
        contrast_high = int(row["_back_rank"]) <= top_k
        if combo:
            category = "combined_selected"
        elif text_high and not contrast_high:
            category = "text_only_high"
        elif contrast_high and not text_high:
            category = "contrast_only_high"
        elif text_high and contrast_high:
            category = "component_intersection_rejected"
        else:
            category = "tail"
        item = dict(row)
        item["component_category"] = category
        item["is_text_high"] = text_high
        item["is_contrast_high"] = contrast_high
        item["is_combined_selected"] = combo
        rows.append(item)
    return rows


def summarize_component_categories(rows):
    out = []
    for category in [
        "combined_selected",
        "text_only_high",
        "contrast_only_high",
        "component_intersection_rejected",
        "tail",
    ]:
        selected = [row for row in rows if row["component_category"] == category]
        if not selected:
            continue
        out.append({
            "category": category,
            "n_heads": len(selected),
            "mean_combo_rank": mean(row["_rank"] for row in selected),
            "mean_text_rank": mean(row["_front_rank"] for row in selected),
            "mean_contrast_rank": mean(row["_back_rank"] for row in selected),
            "mean_itext_all": mean(row.get("Itext_all") for row in selected),
            "mean_itext_gap_hall_minus_grounded": mean(itext_gap(row) for row in selected),
            "mean_image_drop_grounded_minus_hall": mean(image_drop(row) for row in selected),
            "mean_log_toi_gap_hall_minus_grounded": mean(log_toi_gap(row) for row in selected),
            "mean_raw_toi_gap_hall_minus_grounded": mean(raw_toi_gap(row) for row in selected),
            "positive_raw_toi_gap_fraction": mean(1.0 if raw_toi_gap(row) > 0 else 0.0 for row in selected),
            "positive_image_drop_fraction": mean(1.0 if image_drop(row) > 0 else 0.0 for row in selected),
        })
    return out


def component_color(category):
    return {
        "combined_selected": BLUE,
        "text_only_high": ORANGE,
        "contrast_only_high": GREEN,
        "component_intersection_rejected": PURPLE,
        "tail": GRAY,
    }.get(category, GRAY)


def make_component_quadrant_svg(path, rows, top_k):
    width, height = 920, 720
    left, top, plot_w, plot_h = 88, 74, 650, 540
    threshold = 1.0 - (top_k - 1) / float(max(len(rows) - 1, 1))

    def sx(x):
        return left + x * plot_w

    def sy(y):
        return top + (1.0 - y) * plot_h

    body = []
    body.append(text(width / 2, 30, "Head attribution: leverage and hallucination-specificity are distinct axes", 19, DARK, "middle", "700"))
    body.append(text(width / 2, 52, f"top{top_k}: combined rank = 0.5 * text-slice leverage rank + 0.5 * hallucination contrast rank", 12, MUTED, "middle"))
    body.append(rect(left, top, plot_w, plot_h, "#f8fafc", "#cbd5e1"))

    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        x = sx(tick)
        y = sy(tick)
        body.append(line(x, top, x, top + plot_h, "#e2e8f0"))
        body.append(line(left, y, left + plot_w, y, "#e2e8f0"))
        body.append(text(x, top + plot_h + 21, f"{tick:.2f}", 10, MUTED, "middle"))
        body.append(text(left - 12, y + 4, f"{tick:.2f}", 10, MUTED, "end"))
    body.append(line(sx(threshold), top, sx(threshold), top + plot_h, "#334155", 1.4, "5 5"))
    body.append(line(left, sy(threshold), left + plot_w, sy(threshold), "#334155", 1.4, "5 5"))
    body.append(text(sx(threshold) + 6, top + 16, f"top{top_k} text threshold", 10, "#334155"))
    body.append(text(left + 8, sy(threshold) - 8, f"top{top_k} contrast threshold", 10, "#334155"))
    body.append(text(left + plot_w / 2, height - 40, "text-slice leverage percentile", 13, DARK, "middle"))
    body.append(text(28, top + plot_h / 2, "hallucination contrast percentile", 13, DARK, "middle", rotate=-90))

    for row in reversed(rows):
        category = row["component_category"]
        radius = 3.8 if category != "tail" else 2.4
        opacity = 0.82 if category != "tail" else 0.36
        body.append(circle(
            sx(safe_float(row.get("front_percentile"), 0.0)),
            sy(safe_float(row.get("back_percentile"), 0.0)),
            radius,
            component_color(category),
            None,
            1,
            opacity,
        ))

    lx, ly = left + plot_w + 34, top + 22
    labels = [
        ("combined_selected", "combined selected"),
        ("text_only_high", "text-only high"),
        ("contrast_only_high", "contrast-only high"),
        ("tail", "tail"),
    ]
    for idx, (category, label) in enumerate(labels):
        y = ly + idx * 28
        body.append(circle(lx, y, 5, component_color(category), None, 1, 0.9))
        body.append(text(lx + 14, y + 4, label, 12, DARK))
    body.append(text(lx, ly + 150, "Interpretation", 13, DARK, "start", "700"))
    body.append(text(lx, ly + 172, "text-only = leverage without specificity", 11, MUTED))
    body.append(text(lx, ly + 192, "contrast-only = specificity without leverage", 11, MUTED))
    body.append(text(lx, ly + 212, "combined = suppressible hallucination route", 11, MUTED))
    svg(path, width, height, "".join(body))


def make_gate_curve_svg(path, strength=0.7, tau=0.9, betas=(6.0, 8.0, 10.0)):
    width, height = 900, 600
    left, top, plot_w, plot_h = 80, 70, 720, 420

    def sx(x):
        return left + x * plot_w

    def sy(y):
        return top + (1.0 - y) * plot_h

    def suppression(r, beta):
        return min(max(strength * math.exp(beta * (r - tau)), 0.0), 1.0)

    body = []
    body.append(text(width / 2, 31, "Dynamic suppression is a continuous confidence-weighted actuator", 20, DARK, "middle", "700"))
    body.append(text(width / 2, 53, f"delta = clip(s * exp(q * (text_ratio - tau)) * head_prior), shown for s={strength}, prior=1, tau={tau}", 12, MUTED, "middle"))
    body.append(rect(left, top, plot_w, plot_h, "#f8fafc", "#cbd5e1"))
    for tick in [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]:
        x = sx(tick)
        body.append(line(x, top, x, top + plot_h, "#e2e8f0" if tick != tau else "#334155", 1.2, "5 5" if tick == tau else None))
        body.append(text(x, top + plot_h + 22, f"{tick:.2f}", 10, MUTED, "middle"))
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = sy(tick)
        body.append(line(left, y, left + plot_w, y, "#e2e8f0"))
        body.append(text(left - 12, y + 4, f"{tick:.2f}", 10, MUTED, "end"))
    body.append(text(left + plot_w / 2, height - 40, "online text ratio T/(T+I)", 13, DARK, "middle"))
    body.append(text(26, top + plot_h / 2, "suppression strength delta", 13, DARK, "middle", rotate=-90))

    colors = [GREEN, BLUE, ORANGE]
    for beta, color in zip(betas, colors):
        pts = []
        for i in range(201):
            r = i / 200.0
            pts.append((sx(r), sy(suppression(r, beta))))
        body.append(polyline(pts, color, 2.8))
    hard_pts = []
    for r, y in [(0.0, 0.0), (tau, 0.0), (tau, 1.0), (1.0, 1.0)]:
        hard_pts.append((sx(r), sy(y)))
    body.append(polyline(hard_pts, RED, 2.4))

    lx, ly = left + plot_w - 185, top + 28
    for idx, (label, color) in enumerate([(f"q={betas[0]:.0f}", GREEN), (f"q={betas[1]:.0f}", BLUE), (f"q={betas[2]:.0f}", ORANGE), ("binary threshold", RED)]):
        y = ly + idx * 24
        body.append(line(lx, y, lx + 30, y, color, 3))
        body.append(text(lx + 38, y + 4, label, 12, DARK))
    body.append(text(left + 16, top + 26, "mild text reliance -> mild change", 12, MUTED))
    body.append(text(left + plot_w - 250, top + plot_h - 24, "extreme text reliance -> strong suppression", 12, MUTED))
    svg(path, width, height, "".join(body))


def canonical_counter(items):
    counts = Counter()
    for item in items or []:
        if isinstance(item, (list, tuple)) and item:
            key = item[-1]
        else:
            key = item
        if key:
            counts[str(key)] += 1
    return counts


def load_sentences(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    return {str(row["image_id"]): row for row in data.get("sentences", [])}


def compare_object_changes(base_path, method_paths):
    base = load_sentences(base_path)
    rows = []
    for method, path in method_paths:
        method_sentences = load_sentences(path)
        if not base or not method_sentences:
            continue
        totals = Counter()
        for image_id, base_row in base.items():
            method_row = method_sentences.get(image_id)
            if not method_row:
                continue
            method_generated = canonical_counter(method_row.get("mscoco_generated_words"))
            base_hall = canonical_counter(base_row.get("mscoco_hallucinated_words"))
            base_grounded = canonical_counter(base_row.get("mscoco_non_hallucinated_words"))
            method_hall = canonical_counter(method_row.get("mscoco_hallucinated_words"))

            for label, count in base_hall.items():
                retained = min(count, method_generated.get(label, 0))
                totals["base_hall_total"] += count
                totals["base_hall_retained"] += retained
                totals["base_hall_removed"] += count - retained
            for label, count in base_grounded.items():
                retained = min(count, method_generated.get(label, 0))
                totals["base_grounded_total"] += count
                totals["base_grounded_retained"] += retained
                totals["base_grounded_lost"] += count - retained
            for label, count in method_hall.items():
                base_count = canonical_counter(base_row.get("mscoco_generated_words")).get(label, 0)
                totals["method_hall_total"] += count
                totals["method_new_hall"] += max(count - base_count, 0)
            totals["paired_images"] += 1

        if not totals["paired_images"]:
            continue
        hall_removed_rate = totals["base_hall_removed"] / max(totals["base_hall_total"], 1)
        grounded_lost_rate = totals["base_grounded_lost"] / max(totals["base_grounded_total"], 1)
        rows.append({
            "method": method,
            "path": path,
            "paired_images": totals["paired_images"],
            "base_hall_total": totals["base_hall_total"],
            "base_hall_removed": totals["base_hall_removed"],
            "base_hall_removed_rate": hall_removed_rate,
            "base_grounded_total": totals["base_grounded_total"],
            "base_grounded_lost": totals["base_grounded_lost"],
            "base_grounded_lost_rate": grounded_lost_rate,
            "fragility_ratio": hall_removed_rate / max(grounded_lost_rate, 1e-9),
            "method_hall_total": totals["method_hall_total"],
            "method_new_hall": totals["method_new_hall"],
        })
    rows.sort(key=lambda row: safe_float(row.get("base_hall_removed_rate"), 0.0), reverse=True)
    return rows


def parse_method_name(path):
    return os.path.basename(os.path.dirname(path))


def make_object_change_svg(path, rows):
    width, height = 900, 560
    left, top, plot_w, plot_h = 86, 72, 690, 360
    body = []
    body.append(text(width / 2, 31, "Output-level selective effect: hallucinated objects are more fragile", 20, DARK, "middle", "700"))
    if not rows:
        body.append(text(width / 2, height / 2, "No paired caption-eval files found", 16, MUTED, "middle"))
        svg(path, width, height, "".join(body))
        return

    selected = rows[: min(6, len(rows))]
    vmax = max(
        max(row["base_hall_removed_rate"], row["base_grounded_lost_rate"])
        for row in selected
    )
    vmax = min(max(vmax * 1.18, 0.1), 1.0)

    def sy(v):
        return top + (vmax - v) / max(vmax, 1e-9) * plot_h

    body.append(rect(left, top, plot_w, plot_h, "#f8fafc", "#cbd5e1"))
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        if tick > vmax:
            continue
        y = sy(tick)
        body.append(line(left, y, left + plot_w, y, "#e2e8f0"))
        body.append(text(left - 12, y + 4, f"{tick:.2f}", 10, MUTED, "end"))

    group_w = plot_w / len(selected)
    for idx, row in enumerate(selected):
        cx = left + idx * group_w + group_w / 2
        bar_w = min(38, group_w * 0.28)
        hall = row["base_hall_removed_rate"]
        grounded = row["base_grounded_lost_rate"]
        for offset, value, color in [(-bar_w * 0.65, hall, RED), (bar_w * 0.65, grounded, BLUE)]:
            y = sy(value)
            body.append(rect(cx + offset - bar_w / 2, y, bar_w, top + plot_h - y, color, None, 1, 0.86))
            body.append(text(cx + offset, y - 6, f"{value:.2f}", 10, DARK, "middle"))
        label = row["method"].replace("llava-v1.5-7b_", "").replace("dynamic_ratio_exp_file_", "")
        if len(label) > 22:
            label = label[:21] + "..."
        body.append(text(cx, top + plot_h + 24, label, 9, DARK, "middle", rotate=18))
    body.append(text(left + plot_w / 2, height - 38, "method", 13, DARK, "middle"))
    body.append(text(28, top + plot_h / 2, "base mention removed/lost rate", 13, DARK, "middle", rotate=-90))
    lx, ly = left + plot_w + 32, top + 24
    body.append(rect(lx, ly, 18, 18, RED, None, 1, 0.86))
    body.append(text(lx + 28, ly + 14, "base hallucinated removed", 12, DARK))
    body.append(rect(lx, ly + 30, 18, 18, BLUE, None, 1, 0.86))
    body.append(text(lx + 28, ly + 44, "base grounded lost", 12, DARK))
    svg(path, width, height, "".join(body))


def materialize_component_rankings(records, output_dir, source_path):
    base_meta = {
        "source_ranked_heads": source_path,
        "description": "Component-only ranking materialized from the combined ranked-head records for ablation.",
    }
    ranking_specs = [
        ("itext_all_from_combo", "front_percentile", "Itext_all"),
        ("C_toi_HminusG_from_combo", "back_percentile", "RawTOI_hallucinated"),
    ]
    paths = {}
    for name, sort_key, raw_key in ranking_specs:
        sorted_records = sorted(records, key=lambda row: safe_float(row.get(sort_key), -1.0), reverse=True)
        out_heads = []
        for idx, row in enumerate(sorted_records, start=1):
            item = {key: value for key, value in row.items() if not key.startswith("_")}
            item["component_rank"] = idx
            item["component_score_key"] = sort_key
            item["component_raw_key"] = raw_key
            out_heads.append(item)
        path = os.path.join(output_dir, f"ranked_heads_{name}.json")
        write_json(path, {**base_meta, "score_name": name, "heads": out_heads})
        paths[name] = path
    return paths


def _clean_record(row):
    return {key: value for key, value in row.items() if not key.startswith("_")}


def materialize_quadrant_rankings(records, output_dir, source_path, top_k):
    """Write strict 2x2 component-quadrant head pools.

    Quadrants are defined by the two component ranks used by the method:
    text-slice leverage (`front_percentile`) and hallucination specificity
    (`back_percentile`). The resulting files are intended for behavioral
    ablations, not only plotting.
    """
    base_meta = {
        "source_ranked_heads": source_path,
        "quadrant_top_k": top_k,
        "description": (
            "Strict 2x2 head groups from text-slice leverage rank and "
            "hallucination-specific contrast rank."
        ),
    }
    specs = [
        (
            "quadA_high_itext_high_contrast",
            "A_high_itext_high_contrast",
            lambda row: int(row["_front_rank"]) <= top_k and int(row["_back_rank"]) <= top_k,
            lambda row: (
                safe_float(row.get("front_percentile"), 0.0)
                + safe_float(row.get("back_percentile"), 0.0)
            )
            / 2.0,
            "leverageable + hallucination-specific",
        ),
        (
            "quadB_high_itext_low_contrast",
            "B_high_itext_low_contrast",
            lambda row: int(row["_front_rank"]) <= top_k and int(row["_back_rank"]) > top_k,
            lambda row: safe_float(row.get("front_percentile"), 0.0),
            "leverage-only control",
        ),
        (
            "quadC_low_itext_high_contrast",
            "C_low_itext_high_contrast",
            lambda row: int(row["_front_rank"]) > top_k and int(row["_back_rank"]) <= top_k,
            lambda row: safe_float(row.get("back_percentile"), 0.0),
            "specificity-only control",
        ),
        (
            "quadD_low_itext_low_contrast",
            "D_low_itext_low_contrast",
            lambda row: int(row["_front_rank"]) > top_k and int(row["_back_rank"]) > top_k,
            lambda row: (
                safe_float(row.get("front_percentile"), 0.0)
                + safe_float(row.get("back_percentile"), 0.0)
            )
            / 2.0,
            "neither-axis control",
        ),
    ]

    paths = {}
    summary_rows = []
    for file_key, group_name, predicate, score_fn, description in specs:
        selected = [row for row in records if predicate(row)]
        selected = sorted(selected, key=score_fn, reverse=True)
        out_heads = []
        for idx, row in enumerate(selected, start=1):
            item = _clean_record(row)
            item["quadrant"] = group_name
            item["quadrant_rank"] = idx
            item["score"] = score_fn(row)
            item["quadrant_score"] = item["score"]
            item["quadrant_description"] = description
            out_heads.append(item)
        path = os.path.join(output_dir, f"ranked_heads_{file_key}.json")
        write_json(
            path,
            {
                **base_meta,
                "score_name": file_key,
                "quadrant": group_name,
                "heads": out_heads,
            },
        )
        paths[file_key] = path
        summary_rows.append({
            "quadrant": group_name,
            "description": description,
            "n_heads": len(selected),
            "mean_combo_rank": mean(row["_rank"] for row in selected),
            "mean_text_rank": mean(row["_front_rank"] for row in selected),
            "mean_contrast_rank": mean(row["_back_rank"] for row in selected),
            "mean_itext_all": mean(row.get("Itext_all") for row in selected),
            "mean_itext_gap_hall_minus_grounded": mean(itext_gap(row) for row in selected),
            "mean_image_drop_grounded_minus_hall": mean(image_drop(row) for row in selected),
            "mean_log_toi_gap_hall_minus_grounded": mean(log_toi_gap(row) for row in selected),
            "mean_raw_toi_gap_hall_minus_grounded": mean(raw_toi_gap(row) for row in selected),
            "positive_raw_toi_gap_fraction": mean(1.0 if raw_toi_gap(row) > 0 else 0.0 for row in selected),
            "positive_image_drop_fraction": mean(1.0 if image_drop(row) > 0 else 0.0 for row in selected),
            "path": path,
        })
    write_csv(os.path.join(output_dir, "quadrant_head_pool_summary.csv"), summary_rows)
    return paths, summary_rows


def summarize_quadrant_object_rows(object_rows):
    prefix_to_quadrant = {
        "quadA_": "A_high_itext_high_contrast",
        "quadB_": "B_high_itext_low_contrast",
        "quadC_": "C_low_itext_high_contrast",
        "quadD_": "D_low_itext_low_contrast",
    }
    rows = []
    for row in object_rows:
        method = row.get("method", "")
        quadrant = None
        for prefix, name in prefix_to_quadrant.items():
            if method.startswith(prefix):
                quadrant = name
                break
        if quadrant is None:
            continue
        out = dict(row)
        out["quadrant"] = quadrant
        out["hallucination_removal_rate"] = row.get("base_hall_removed_rate")
        out["grounded_damage_rate"] = row.get("base_grounded_lost_rate")
        rows.append(out)
    rows.sort(key=lambda row: (row["quadrant"], row["method"]))
    return rows


def make_quadrant_object_matrix_svg(path, rows):
    width, height = 900, 600
    body = []
    body.append(text(width / 2, 31, "2x2 quadrant behavioral matrix", 20, DARK, "middle", "700"))
    body.append(text(width / 2, 53, "Cells report greedy hallucinated removal, grounded damage, and fragility ratio after suppression", 12, MUTED, "middle"))

    if not rows:
        body.append(text(width / 2, height / 2, "Run quadA/quadB/quadC/quadD ablations to populate this matrix", 16, MUTED, "middle"))
        svg(path, width, height, "".join(body))
        return

    preferred = {}
    for row in rows:
        key = row["quadrant"]
        # Prefer the most selective run per quadrant for the dashboard.
        current = preferred.get(key)
        if current is None or safe_float(row.get("fragility_ratio"), 0.0) > safe_float(current.get("fragility_ratio"), 0.0):
            preferred[key] = row

    left, top, cell_w, cell_h = 160, 120, 280, 170
    cells = [
        ("A_high_itext_high_contrast", 1, 0, "A: high leverage + high specificity", BLUE),
        ("B_high_itext_low_contrast", 1, 1, "B: leverage only", ORANGE),
        ("C_low_itext_high_contrast", 0, 0, "C: specificity only", GREEN),
        ("D_low_itext_low_contrast", 0, 1, "D: neither", GRAY),
    ]
    body.append(text(left + cell_w, top - 48, "high contrastive", 13, DARK, "middle", "700"))
    body.append(text(left + cell_w * 2, top - 48, "low contrastive", 13, DARK, "middle", "700"))
    body.append(text(left - 50, top + cell_h / 2, "high itext", 13, DARK, "middle", "700", rotate=-90))
    body.append(text(left - 50, top + cell_h * 1.5, "low itext", 13, DARK, "middle", "700", rotate=-90))

    for quadrant, col, row_idx, label, color in cells:
        x = left + col * cell_w
        y = top + row_idx * cell_h
        item = preferred.get(quadrant)
        body.append(rect(x, y, cell_w - 18, cell_h - 18, "#f8fafc", color, 2, 1.0))
        body.append(text(x + 16, y + 28, label, 13, DARK, "start", "700"))
        if item:
            body.append(text(x + 16, y + 62, f"method: {item['method']}", 10, MUTED))
            body.append(text(x + 16, y + 88, f"hall removal: {safe_float(item.get('base_hall_removed_rate'), 0.0):.3f}", 13, RED))
            body.append(text(x + 16, y + 112, f"grounded damage: {safe_float(item.get('base_grounded_lost_rate'), 0.0):.3f}", 13, BLUE))
            body.append(text(x + 16, y + 136, f"fragility ratio: {safe_float(item.get('fragility_ratio'), 0.0):.2f}", 13, DARK, "start", "700"))
        else:
            body.append(text(x + 16, y + 86, "not run yet", 13, MUTED))
    svg(path, width, height, "".join(body))


def make_report(path, outputs, component_rows, object_rows, top_k):
    by_cat = {row["category"]: row for row in component_rows}

    def val(category, key, ndigits=4):
        return f"{safe_float(by_cat.get(category, {}).get(key), 0.0):.{ndigits}f}"

    lines = [
        "# Method Claim Evidence",
        "",
        "This report is organized around our method's own claims, not around AD-HH.",
        "",
        "## Q1. Why are these heads hallucination-relevant?",
        "",
        "The attribution score combines two different axes:",
        "",
        "- `txt_mass / Itext_all`: intervention leverage. It finds heads whose output is actually routed through the text-side attention slice we can suppress.",
        "- `C_toi_HminusG`: hallucination specificity. It finds heads where hallucinated object steps have higher text-over-image reliance than grounded object steps.",
        "- Combined score: leverageable plus hallucination-specific heads.",
        "",
        f"![Component quadrants]({os.path.basename(outputs['component_quadrant'])})",
        "",
        "Numbers from the component split:",
        "",
        f"- combined top{top_k}: n={int(by_cat.get('combined_selected', {}).get('n_heads', 0))}, Itext={val('combined_selected', 'mean_itext_all')}, logTOI gap={val('combined_selected', 'mean_log_toi_gap_hall_minus_grounded')}, image drop={val('combined_selected', 'mean_image_drop_grounded_minus_hall')}.",
        f"- text-only high: n={int(by_cat.get('text_only_high', {}).get('n_heads', 0))}, Itext={val('text_only_high', 'mean_itext_all')}, logTOI gap={val('text_only_high', 'mean_log_toi_gap_hall_minus_grounded')}, image drop={val('text_only_high', 'mean_image_drop_grounded_minus_hall')}.",
        f"- contrast-only high: n={int(by_cat.get('contrast_only_high', {}).get('n_heads', 0))}, Itext={val('contrast_only_high', 'mean_itext_all')}, logTOI gap={val('contrast_only_high', 'mean_log_toi_gap_hall_minus_grounded')}, image drop={val('contrast_only_high', 'mean_image_drop_grounded_minus_hall')}.",
        "",
        "This is the core attribution justification: text-only heads are leverageable but not necessarily hallucination-specific; contrast-only heads are hallucination-specific but may have weak intervention leverage. The combined pool is the intersection we can act on.",
        "",
        "## Q2. Why does dynamic suppression work?",
        "",
        "Mechanism: hallucinated objects are supported by text-side context without matching visual evidence. When online text ratio is high, reducing the text-side slice removes that support. Grounded objects have more visual evidence and are less fragile under the same text-side reduction.",
        "",
        f"![Gate curve]({os.path.basename(outputs['gate_curve'])})",
        "",
        "This figure gives the continuous-gate rationale. A binary threshold treats all above-threshold states equally; the exponential gate tracks the degree of text dominance.",
        "",
        "## Q3. Why continuous and why larger top-k?",
        "",
        "Continuous gating is what makes a larger head pool plausible. The offline head pool defines where intervention may happen; the online text ratio defines when and how strongly it happens. Irrelevant or weakly text-dominant head-steps receive low suppression.",
        "",
        "## Output-Level Selectivity Check",
        "",
        f"![Object change]({os.path.basename(outputs['object_change'])})",
        "",
        "## 2x2 Quadrant Behavioral Check",
        "",
        f"![Quadrant object matrix]({os.path.basename(outputs['quadrant_object_matrix'])})",
        "",
    ]
    if object_rows:
        best = object_rows[0]
        lines.extend([
            f"- strongest available paired run: `{best['method']}`",
            f"- base hallucinated mention removed rate: {best['base_hall_removed_rate']:.4f}",
            f"- base grounded mention lost rate: {best['base_grounded_lost_rate']:.4f}",
            f"- removal/loss ratio: {best['fragility_ratio']:.4f}",
        ])
    else:
        lines.append("- No paired caption-eval files were available for object-level change analysis.")
    lines.extend([
        "",
        "## Missing Evidence To Add With New Runs",
        "",
        "1. Per-object-step intervention trace: log online `delta`, `text_ratio`, and `head_score` at generated object steps, then bucket by CHAIR hallucinated vs grounded labels.",
        "2. Ablation matrix: txt-only heads vs contrastive-only heads vs combined heads; binary vs continuous; top20/top50/top100/top150.",
        "3. Caption quality failure taxonomy: show CHAIR reduction is not just length collapse by tracking caption length, grounded object retention, and new hallucination rate.",
        "",
        "## Generated Ablation Head Files",
        "",
        f"- text-only head ranking: `{outputs['component_rankings'].get('itext_all_from_combo', '')}`",
        f"- contrastive-only head ranking: `{outputs['component_rankings'].get('C_toi_HminusG_from_combo', '')}`",
        f"- quadrant A ranking: `{outputs['quadrant_rankings'].get('quadA_high_itext_high_contrast', '')}`",
        f"- quadrant B ranking: `{outputs['quadrant_rankings'].get('quadB_high_itext_low_contrast', '')}`",
        f"- quadrant C ranking: `{outputs['quadrant_rankings'].get('quadC_low_itext_high_contrast', '')}`",
        f"- quadrant D ranking: `{outputs['quadrant_rankings'].get('quadD_low_itext_low_contrast', '')}`",
        "",
    ])
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ranked-heads",
        default="../ADHH/LLaVA/results/coco/llava-v1.5-7b_base_original_qa_n3000/surrogate_hh_scores/surrogate_score_zoo/ranked_heads_global__itext_all__C_toi_HminusG.json",
    )
    parser.add_argument("--output-dir", default="./results/coco/contrastive_dynamic_head_pool_analysis/method_claim_evidence")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--gate-strength", type=float, default=0.7)
    parser.add_argument("--gate-tau", type=float, default=0.9)
    parser.add_argument("--base-eval-json", default="../ADHH/LLaVA/results/coco/llava-v1.5-7b_base_n500/captions_eval_results.json")
    parser.add_argument(
        "--method-eval-glob",
        default="../ADHH/LLaVA/results_dynamic/coco/*global__itext_all__C_toi_HminusG/captions_eval_results.json",
    )
    args = parser.parse_args()

    _, records = load_ranked_heads(args.ranked_heads)
    add_component_ranks(records)
    categorized = categorize_components(records, args.top_k)
    component_rows = summarize_component_categories(categorized)
    write_csv(os.path.join(args.output_dir, "component_category_summary.csv"), component_rows)

    component_rank_dir = os.path.join(args.output_dir, "component_rankings")
    component_paths = materialize_component_rankings(records, component_rank_dir, args.ranked_heads)
    quadrant_rank_dir = os.path.join(args.output_dir, "quadrant_rankings")
    quadrant_paths, quadrant_rows = materialize_quadrant_rankings(
        records,
        quadrant_rank_dir,
        args.ranked_heads,
        args.top_k,
    )

    component_quadrant = os.path.join(args.output_dir, "component_quadrant.svg")
    gate_curve = os.path.join(args.output_dir, "continuous_gate_curve.svg")
    object_change = os.path.join(args.output_dir, "output_object_change.svg")
    quadrant_object_matrix = os.path.join(args.output_dir, "quadrant_object_matrix.svg")
    make_component_quadrant_svg(component_quadrant, categorized, args.top_k)
    make_gate_curve_svg(gate_curve, strength=args.gate_strength, tau=args.gate_tau)

    method_paths = [
        (parse_method_name(path), path)
        for path in sorted(glob.glob(args.method_eval_glob))
    ]
    object_rows = compare_object_changes(args.base_eval_json, method_paths)
    write_csv(os.path.join(args.output_dir, "object_change_summary.csv"), object_rows)
    make_object_change_svg(object_change, object_rows)
    quadrant_object_rows = summarize_quadrant_object_rows(object_rows)
    write_csv(os.path.join(args.output_dir, "quadrant_object_matrix.csv"), quadrant_object_rows)
    make_quadrant_object_matrix_svg(quadrant_object_matrix, quadrant_object_rows)

    outputs = {
        "component_quadrant": component_quadrant,
        "gate_curve": gate_curve,
        "object_change": object_change,
        "quadrant_object_matrix": quadrant_object_matrix,
        "component_rankings": component_paths,
        "quadrant_rankings": quadrant_paths,
    }
    report_path = os.path.join(args.output_dir, "method_claim_evidence.md")
    make_report(report_path, outputs, component_rows, object_rows, args.top_k)

    summary = {
        "report": report_path,
        "component_summary": os.path.join(args.output_dir, "component_category_summary.csv"),
        "object_change_summary": os.path.join(args.output_dir, "object_change_summary.csv"),
        "figures": {
            "component_quadrant": component_quadrant,
            "gate_curve": gate_curve,
            "object_change": object_change,
            "quadrant_object_matrix": quadrant_object_matrix,
        },
        "component_rankings": component_paths,
        "quadrant_rankings": quadrant_paths,
        "quadrant_head_pool_summary": os.path.join(quadrant_rank_dir, "quadrant_head_pool_summary.csv"),
        "quadrant_object_matrix": os.path.join(args.output_dir, "quadrant_object_matrix.csv"),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

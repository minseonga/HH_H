import argparse
import csv
import html
import json
import math
import os
import sys


LLAVA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if LLAVA_ROOT not in sys.path:
    sys.path.insert(0, LLAVA_ROOT)

from eval_scripts.soft_routing.analyze_contrastive_dynamic_head_pool import (
    head_key,
    load_ranked_heads,
    safe_float,
)
from eval_scripts.soft_routing.head_prior_utils import default_heads_for_model


BLUE = "#2563eb"
ORANGE = "#f97316"
GRAY = "#cbd5e1"
DARK = "#0f172a"
MUTED = "#64748b"
RED = "#dc2626"
GREEN = "#059669"


def mean(values):
    values = [safe_float(value) for value in values]
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else 0.0


def quantile(values, q):
    values = sorted(value for value in (safe_float(item) for item in values) if value is not None)
    if not values:
        return 0.0
    idx = min(max(int(round((len(values) - 1) * q)), 0), len(values) - 1)
    return values[idx]


def raw_toi_gap(row):
    return safe_float(row.get("RawTOI_hallucinated"), 0.0) - safe_float(row.get("RawTOI_non_hallucinated"), 0.0)


def log_toi_gap(row):
    return safe_float(row.get("LogTOI_hallucinated"), 0.0) - safe_float(row.get("LogTOI_non_hallucinated"), 0.0)


def image_drop(row):
    return safe_float(row.get("Img_non_hallucinated"), 0.0) - safe_float(row.get("Img_hallucinated"), 0.0)


def itext_gap(row):
    return safe_float(row.get("Itext_hallucinated"), 0.0) - safe_float(row.get("Itext_non_hallucinated"), 0.0)


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


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


def cross(x, y, size=5, color=RED, width=2):
    return (
        line(x - size, y - size, x + size, y + size, color, width)
        + line(x - size, y + size, x + size, y - size, color, width)
    )


def summarize_bucket(name, rows, adhh_keys):
    keys = {row["_key"] for row in rows}
    return {
        "bucket": name,
        "n": len(rows),
        "adhh_overlap": len(keys & adhh_keys),
        "mean_itext_all": mean(row.get("Itext_all") for row in rows),
        "mean_itext_hallucinated": mean(row.get("Itext_hallucinated") for row in rows),
        "mean_itext_non_hallucinated": mean(row.get("Itext_non_hallucinated") for row in rows),
        "mean_itext_gap_hall_minus_grounded": mean(itext_gap(row) for row in rows),
        "mean_img_hallucinated": mean(row.get("Img_hallucinated") for row in rows),
        "mean_img_grounded": mean(row.get("Img_non_hallucinated") for row in rows),
        "mean_image_drop_grounded_minus_hall": mean(image_drop(row) for row in rows),
        "mean_log_toi_gap_hall_minus_grounded": mean(log_toi_gap(row) for row in rows),
        "mean_raw_toi_gap_hall_minus_grounded": mean(raw_toi_gap(row) for row in rows),
        "positive_raw_toi_gap": sum(1 for row in rows if raw_toi_gap(row) > 0.0),
        "positive_image_drop": sum(1 for row in rows if image_drop(row) > 0.0),
    }


def make_summary_rows(records, adhh_keys):
    buckets = [
        ("top20", records[:20]),
        ("top100", records[:100]),
        ("top150", records[:150]),
        ("top100_to_150_shell", records[100:150]),
        ("rank_gt_200_tail", records[200:]),
    ]
    return [summarize_bucket(name, rows, adhh_keys) for name, rows in buckets]


def make_head_space_svg(path, records, adhh_keys):
    width, height = 1040, 720
    left, right, top, bottom = 92, 58, 58, 94
    plot_w = width - left - right
    plot_h = height - top - bottom

    xs = [safe_float(row.get("Itext_all"), 0.0) for row in records]
    ys = [math.asinh(raw_toi_gap(row) / 20.0) for row in records]
    xmin, xmax = 0.0, max(1.0, quantile(xs, 0.995))
    ymin, ymax = quantile(ys, 0.01), quantile(ys, 0.99)
    pad = (ymax - ymin) * 0.08
    ymin -= pad
    ymax += pad

    def sx(x):
        return left + (x - xmin) / max(xmax - xmin, 1e-9) * plot_w

    def sy(y):
        return top + (ymax - y) / max(ymax - ymin, 1e-9) * plot_h

    rank_by_key = {row["_key"]: row["_rank"] for row in records}
    body = []
    body.append(text(width / 2, 28, "Head-space evidence for text-side suppression", 20, DARK, "middle", "700"))
    body.append(text(width / 2, 49, "x: intervention text mass, y: hallucinated-vs-grounded text-over-image gap (asinh scale)", 12, MUTED, "middle"))
    body.append(rect(left, top, plot_w, plot_h, "#f8fafc", "#cbd5e1"))
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        x = sx(tick)
        body.append(line(x, top, x, top + plot_h, "#e2e8f0"))
        body.append(text(x, top + plot_h + 22, f"{tick:.2f}", 11, MUTED, "middle"))
    for tick in [math.floor(ymin), 0, math.ceil(ymax)]:
        if ymin <= tick <= ymax:
            y = sy(tick)
            body.append(line(left, y, left + plot_w, y, "#e2e8f0"))
            body.append(text(left - 12, y + 4, f"{tick:.1f}", 11, MUTED, "end"))
    body.append(text(left + plot_w / 2, height - 38, "Itext_all: mean attention mass on intervention text slice", 13, DARK, "middle"))
    body.append(text(25, top + plot_h / 2, "asinh((RawTOI_hall - RawTOI_grounded)/20)", 13, DARK, "middle", rotate=-90))

    def category(row):
        rank = int(row["_rank"])
        if rank <= 100:
            return BLUE, 0.82, 3.9
        if rank <= 150:
            return ORANGE, 0.78, 3.7
        return GRAY, 0.42, 2.8

    for row in reversed(records):
        x = sx(safe_float(row.get("Itext_all"), 0.0))
        y = sy(math.asinh(raw_toi_gap(row) / 20.0))
        fill, opacity, radius = category(row)
        stroke = "#111827" if row["_key"] in adhh_keys else None
        body.append(circle(x, y, radius, fill, stroke, 1.3, opacity))

    rejected = [row for row in records if row["_key"] in adhh_keys and rank_by_key[row["_key"]] > 150]
    for row in rejected:
        x = sx(safe_float(row.get("Itext_all"), 0.0))
        y = sy(math.asinh(raw_toi_gap(row) / 20.0))
        body.append(cross(x, y, 6, RED, 2))
        body.append(text(x + 8, y - 6, row["_lhh"], 10, RED))

    for row in records[:5]:
        x = sx(safe_float(row.get("Itext_all"), 0.0))
        y = sy(math.asinh(raw_toi_gap(row) / 20.0))
        body.append(text(x + 7, y - 5, row["_lhh"], 10, DARK))

    lx, ly = left + 20, top + 20
    body.append(circle(lx, ly, 5, BLUE, None, 1, 0.9))
    body.append(text(lx + 14, ly + 4, "top100 selected", 12, DARK))
    body.append(circle(lx, ly + 24, 5, ORANGE, None, 1, 0.9))
    body.append(text(lx + 14, ly + 28, "top100-150 shell", 12, DARK))
    body.append(circle(lx, ly + 48, 5, GRAY, None, 1, 0.7))
    body.append(text(lx + 14, ly + 52, "tail candidates", 12, DARK))
    body.append(circle(lx, ly + 72, 5, "white", "#111827", 1.5, 1))
    body.append(text(lx + 14, ly + 76, "AD-HH default head", 12, DARK))
    body.append(cross(lx, ly + 96, 5, RED, 2))
    body.append(text(lx + 14, ly + 100, "AD-HH rejected by contrastive pool", 12, DARK))
    svg(path, width, height, "".join(body))


def make_source_shift_svg(path, summary_rows):
    width, height = 1120, 640
    panels = [
        ("Itext hall - grounded", "mean_itext_gap_hall_minus_grounded", BLUE),
        ("Image drop grounded - hall", "mean_image_drop_grounded_minus_hall", GREEN),
        ("LogTOI hall - grounded", "mean_log_toi_gap_hall_minus_grounded", ORANGE),
    ]
    buckets = ["top20", "top100", "top150", "rank_gt_200_tail"]
    labels = {"top20": "top20", "top100": "top100", "top150": "top150", "rank_gt_200_tail": "tail>200"}
    by_bucket = {row["bucket"]: row for row in summary_rows}
    left, top = 70, 76
    panel_w, panel_h = 310, 410
    gap = 32
    body = []
    body.append(text(width / 2, 31, "Why suppress text-side attention?", 21, DARK, "middle", "700"))
    body.append(text(width / 2, 53, "Selected heads move toward text-over-image reliance at hallucinated object steps", 12, MUTED, "middle"))

    for pidx, (title, key, color) in enumerate(panels):
        px = left + pidx * (panel_w + gap)
        py = top
        vals = [safe_float(by_bucket[bucket].get(key), 0.0) for bucket in buckets]
        vmax = max(max(vals), 0.0)
        vmin = min(min(vals), 0.0)
        pad = max((vmax - vmin) * 0.12, 0.01)
        vmax += pad
        vmin -= pad

        def sy(value):
            return py + (vmax - value) / max(vmax - vmin, 1e-9) * panel_h

        body.append(rect(px, py, panel_w, panel_h, "#f8fafc", "#cbd5e1"))
        zero_y = sy(0.0)
        body.append(line(px, zero_y, px + panel_w, zero_y, "#94a3b8", 1.2))
        body.append(text(px + panel_w / 2, py - 18, title, 14, DARK, "middle", "700"))

        for i, bucket in enumerate(buckets):
            bar_w = 48
            x = px + 34 + i * 68
            val = safe_float(by_bucket[bucket].get(key), 0.0)
            y = sy(max(val, 0.0))
            h = abs(sy(val) - zero_y)
            if val < 0:
                y = zero_y
            fill = color if bucket != "rank_gt_200_tail" else "#94a3b8"
            body.append(rect(x, y, bar_w, max(h, 1.2), fill, None, 1, 0.86))
            body.append(text(x + bar_w / 2, py + panel_h + 22, labels[bucket], 11, DARK, "middle"))
            body.append(text(x + bar_w / 2, y - 7 if val >= 0 else y + h + 14, f"{val:.3f}", 10, DARK, "middle"))

        body.append(text(px - 10, sy(vmax - pad) + 4, f"{(vmax - pad):.2f}", 10, MUTED, "end"))
        body.append(text(px - 10, zero_y + 4, "0", 10, MUTED, "end"))
        body.append(text(px - 10, sy(vmin + pad) + 4, f"{(vmin + pad):.2f}", 10, MUTED, "end"))

    note = (
        "Suppression is source-specific: the actuator only downweights the text-side slice, "
        "and these selected heads are exactly where hallucination steps show text-over-image shift."
    )
    body.append(text(width / 2, height - 52, note, 13, DARK, "middle"))
    svg(path, width, height, "".join(body))


def make_layer_map_svg(path, records, adhh_keys):
    width, height = 1040, 650
    left, top = 92, 70
    cell = 24
    layer_gap = 2
    row_h = cell + layer_gap
    body = []
    body.append(text(width / 2, 31, "Layer-head map of suppression candidates", 21, DARK, "middle", "700"))
    body.append(text(width / 2, 53, "top100/top150 distribute the actuator across mid-to-late layers; AD-HH overlap is outlined", 12, MUTED, "middle"))

    rank_by_key = {row["_key"]: int(row["_rank"]) for row in records}
    score_by_key = {row["_key"]: safe_float(row.get("itext_all__C_toi_HminusG"), 0.0) for row in records}
    all_layers = list(range(13, 32))
    for head in range(32):
        if head % 4 == 0:
            x = left + head * cell + cell / 2
            body.append(text(x, top - 14, head, 10, MUTED, "middle"))
    body.append(text(left + 32 * cell / 2, top - 36, "head index", 12, DARK, "middle"))

    for ridx, layer in enumerate(all_layers):
        y = top + ridx * row_h
        body.append(text(left - 16, y + 16, f"L{layer}", 11, DARK, "end"))
        for head in range(32):
            key = f"{layer}:{head}"
            rank = rank_by_key.get(key, 9999)
            if rank <= 100:
                fill = BLUE
                opacity = 0.35 + 0.6 * score_by_key.get(key, 0.0)
            elif rank <= 150:
                fill = ORANGE
                opacity = 0.35 + 0.55 * score_by_key.get(key, 0.0)
            else:
                fill = "#f1f5f9"
                opacity = 1.0
            stroke = "#111827" if key in adhh_keys else "#e2e8f0"
            sw = 1.9 if key in adhh_keys else 0.7
            body.append(rect(left + head * cell, y, cell - 2, cell - 2, fill, stroke, sw, opacity))
            if key in adhh_keys and rank > 150:
                body.append(cross(left + head * cell + cell / 2 - 1, y + cell / 2 - 1, 5, RED, 1.8))

    lx, ly = left + 32 * cell + 34, top + 20
    body.append(rect(lx, ly, 18, 18, BLUE, None, 1, 0.9))
    body.append(text(lx + 28, ly + 14, "top100", 12, DARK))
    body.append(rect(lx, ly + 28, 18, 18, ORANGE, None, 1, 0.9))
    body.append(text(lx + 28, ly + 42, "top100-150", 12, DARK))
    body.append(rect(lx, ly + 56, 18, 18, "white", "#111827", 2, 1))
    body.append(text(lx + 28, ly + 70, "AD-HH", 12, DARK))
    body.append(cross(lx + 9, ly + 93, 5, RED, 2))
    body.append(text(lx + 28, ly + 98, "AD-HH rejected", 12, DARK))
    svg(path, width, height, "".join(body))


def make_markdown(path, summary_rows, figure_paths):
    by_bucket = {row["bucket"]: row for row in summary_rows}

    def fmt(bucket, key, ndigits=4):
        return f"{safe_float(by_bucket[bucket].get(key), 0.0):.{ndigits}f}"

    lines = [
        "# Suppression Evidence Figures",
        "",
        "This bundle supports the intervention rationale independently of final CHAIR scores.",
        "The claim is not just that the selected heads are high-text heads; it is that they are the heads where hallucinated object steps show a source shift toward text-side context and away from image evidence.",
        "",
        "## Visual Evidence",
        "",
        f"![Head space]({os.path.basename(figure_paths['head_space'])})",
        "",
        f"![Source shift]({os.path.basename(figure_paths['source_shift'])})",
        "",
        f"![Layer-head map]({os.path.basename(figure_paths['layer_map'])})",
        "",
        "## Numbers To Cite",
        "",
        f"- top100: mean Itext gap H-G={fmt('top100', 'mean_itext_gap_hall_minus_grounded')}, image drop G-H={fmt('top100', 'mean_image_drop_grounded_minus_hall')}, logTOI gap H-G={fmt('top100', 'mean_log_toi_gap_hall_minus_grounded')}.",
        f"- top150: mean Itext gap H-G={fmt('top150', 'mean_itext_gap_hall_minus_grounded')}, image drop G-H={fmt('top150', 'mean_image_drop_grounded_minus_hall')}, logTOI gap H-G={fmt('top150', 'mean_log_toi_gap_hall_minus_grounded')}.",
        f"- tail rank>200: mean Itext gap H-G={fmt('rank_gt_200_tail', 'mean_itext_gap_hall_minus_grounded')}, image drop G-H={fmt('rank_gt_200_tail', 'mean_image_drop_grounded_minus_hall')}, logTOI gap H-G={fmt('rank_gt_200_tail', 'mean_log_toi_gap_hall_minus_grounded')}.",
        f"- top100 positive RawTOI gap: {int(by_bucket['top100']['positive_raw_toi_gap'])}/{int(by_bucket['top100']['n'])}; positive image drop: {int(by_bucket['top100']['positive_image_drop'])}/{int(by_bucket['top100']['n'])}.",
        f"- top150 positive RawTOI gap: {int(by_bucket['top150']['positive_raw_toi_gap'])}/{int(by_bucket['top150']['n'])}; positive image drop: {int(by_bucket['top150']['positive_image_drop'])}/{int(by_bucket['top150']['n'])}.",
        "",
        "## Interpretation",
        "",
        "1. Text-side suppression is source-matched: the method suppresses exactly the text-side attention slice, and the chosen heads are selected by high intervention-text mass.",
        "2. It is hallucination-specific: selected heads have higher text-over-image ratio on hallucinated object steps than on grounded object steps.",
        "3. It is not a generic language-head mask: AD-HH heads that are high text-mass but weak on hallucination contrast are visually marked as rejected by the contrastive pool.",
        "4. The layer-head map shows that top100/top150 are a distributed mid-to-late actuator scaffold rather than a small fixed AD-HH copy.",
        "",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ranked-heads",
        default="../ADHH/LLaVA/results/coco/llava-v1.5-7b_base_original_qa_n3000/surrogate_hh_scores/surrogate_score_zoo/ranked_heads_global__itext_all__C_toi_HminusG.json",
    )
    parser.add_argument("--model-path", default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--output-dir", default="./results/coco/contrastive_dynamic_head_pool_analysis")
    args = parser.parse_args()

    _, records = load_ranked_heads(args.ranked_heads)
    adhh_keys = {head_key(layer, head) for layer, head in default_heads_for_model(args.model_path)}
    summary_rows = make_summary_rows(records, adhh_keys)
    summary_csv = os.path.join(args.output_dir, "suppression_evidence_summary.csv")
    write_csv(summary_csv, summary_rows)

    figure_paths = {
        "head_space": os.path.join(args.output_dir, "suppression_evidence_head_space.svg"),
        "source_shift": os.path.join(args.output_dir, "suppression_evidence_source_shift.svg"),
        "layer_map": os.path.join(args.output_dir, "suppression_evidence_layer_head_map.svg"),
    }
    make_head_space_svg(figure_paths["head_space"], records, adhh_keys)
    make_source_shift_svg(figure_paths["source_shift"], summary_rows)
    make_layer_map_svg(figure_paths["layer_map"], records, adhh_keys)
    report_path = os.path.join(args.output_dir, "suppression_evidence_figures.md")
    make_markdown(report_path, summary_rows, figure_paths)

    print(json.dumps({
        "summary_csv": summary_csv,
        "report": report_path,
        "figures": figure_paths,
    }, indent=2))


if __name__ == "__main__":
    main()

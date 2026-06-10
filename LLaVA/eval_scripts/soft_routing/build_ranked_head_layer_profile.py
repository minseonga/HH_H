#!/usr/bin/env python3
import argparse
import csv
import html
import json
import os
from collections import Counter


DARK = "#0f172a"
MUTED = "#64748b"
GRID = "#e2e8f0"
PURPLE = "#7c3aed"
ORANGE = "#f97316"
BLUE = "#2563eb"


def safe_float(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value


def safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def read_ranked_heads(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    rows = data.get("heads", data if isinstance(data, list) else [])
    out = []
    for idx, row in enumerate(rows, start=1):
        if isinstance(row, dict):
            layer = safe_int(row.get("layer"))
            head = safe_int(row.get("head"))
            score = safe_float(row.get("score", row.get(data.get("score_name", ""), 0.0)))
            front = safe_float(row.get("front_percentile"))
            back = safe_float(row.get("back_percentile"))
            front_raw = safe_float(row.get("front_raw", row.get("Itext_all", 0.0)))
            back_raw = safe_float(row.get("back_raw", row.get("signed_toi_back_raw", 0.0)))
        else:
            layer = safe_int(row[0])
            head = safe_int(row[1])
            score = safe_float(row[2], 1.0) if len(row) > 2 else 1.0
            front = back = front_raw = back_raw = 0.0
        out.append(
            {
                "rank": idx,
                "layer": layer,
                "head": head,
                "score": score,
                "text_percentile": front,
                "contrast_percentile": back,
                "text_raw": front_raw,
                "contrast_raw": back_raw,
            }
        )
    return data if isinstance(data, dict) else {}, out


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


def contiguous_sum(values, start, width):
    return sum(values.get(layer, 0.0) for layer in range(start, start + width))


def recommend_windows(layer_rows, min_layer, max_layer, window_sizes):
    score_by_layer = {row["layer"]: safe_float(row["score_sum"]) for row in layer_rows}
    count_by_layer = {row["layer"]: safe_int(row["count"]) for row in layer_rows}
    rows = []
    for width in window_sizes:
        if width <= 0 or width > (max_layer - min_layer + 1):
            continue
        total_score = sum(score_by_layer.values()) or 1.0
        for start in range(min_layer, max_layer - width + 2):
            end = start + width - 1
            score_sum = contiguous_sum(score_by_layer, start, width)
            count = sum(count_by_layer.get(layer, 0) for layer in range(start, end + 1))
            rows.append(
                {
                    "window": f"L{start}-L{end}",
                    "start_layer": start,
                    "end_layer": end,
                    "width": width,
                    "score_sum": score_sum,
                    "score_share": score_sum / total_score,
                    "selected_count": count,
                    "score_per_layer": score_sum / width,
                }
            )
    rows.sort(key=lambda x: (x["score_sum"], x["selected_count"]), reverse=True)
    return rows


def esc(value):
    return html.escape(str(value))


def text(x, y, value, size=12, fill=DARK, anchor="start", weight="400"):
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{esc(value)}</text>\n'
    )


def line(x1, y1, x2, y2, stroke=GRID, width=1):
    return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{stroke}" stroke-width="{width}"/>\n'


def rect(x, y, w, h, fill, stroke=None, opacity=1.0, rx=0):
    stroke_attr = f' stroke="{stroke}" stroke-width="1"' if stroke else ""
    return f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" rx="{rx}" fill="{fill}" opacity="{opacity}"{stroke_attr}/>\n'


def circle(x, y, r, fill, stroke="white"):
    return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>\n'


def polyline(points, stroke=PURPLE, width=3):
    pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polyline points="{pts}" fill="none" stroke="{stroke}" stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"/>\n'


def build_svg(path, layer_rows, title, top_k, best_window=None):
    width, height = 980, 430
    left, right, top, bottom = 78, 32, 78, 76
    plot_w = width - left - right
    plot_h = height - top - bottom
    layers = [row["layer"] for row in layer_rows]
    if not layers:
        return
    min_layer, max_layer = min(layers), max(layers)
    max_score = max(safe_float(row["score_sum"]) for row in layer_rows) or 1.0
    max_count = max(safe_int(row["count"]) for row in layer_rows) or 1

    def x(layer):
        if max_layer == min_layer:
            return left + plot_w / 2
        return left + (layer - min_layer) / (max_layer - min_layer) * plot_w

    def y_score(value):
        return top + plot_h - (safe_float(value) / max_score) * plot_h

    def y_count(value):
        return top + plot_h - (safe_float(value) / max_count) * plot_h

    body = []
    body.append(rect(0, 0, width, height, "white"))
    body.append(text(width / 2, 34, title, 24, DARK, "middle", "700"))
    body.append(text(width / 2, 56, f"Layer profile from top-{top_k} ranked heads; no layer window is pre-selected.", 13, MUTED, "middle"))

    body.append(rect(left, top, plot_w, plot_h, "#f8fafc", "#cbd5e1", 1.0, 10))
    for i in range(5):
        yy = top + plot_h * i / 4
        body.append(line(left, yy, left + plot_w, yy, GRID, 1))
    body.append(line(left, top + plot_h, left + plot_w, top + plot_h, "#94a3b8", 1.2))

    if best_window:
        start = safe_int(best_window["start_layer"])
        end = safe_int(best_window["end_layer"])
        x0 = x(start) - plot_w / max((max_layer - min_layer + 1), 1) / 2
        x1 = x(end) + plot_w / max((max_layer - min_layer + 1), 1) / 2
        body.append(rect(max(left, x0), top, min(left + plot_w, x1) - max(left, x0), plot_h, "#ede9fe", "#c4b5fd", 0.55, 8))

    bar_w = max(7, min(18, plot_w / max((max_layer - min_layer + 1), 1) * 0.58))
    for row in layer_rows:
        layer = row["layer"]
        count = safe_int(row["count"])
        h = (count / max_count) * plot_h if max_count else 0
        body.append(rect(x(layer) - bar_w / 2, top + plot_h - h, bar_w, h, BLUE, None, 0.26, 2))

    points = [(x(row["layer"]), y_score(row["score_sum"])) for row in layer_rows]
    body.append(polyline(points, PURPLE, 3.2))
    for row in layer_rows:
        body.append(circle(x(row["layer"]), y_score(row["score_sum"]), 4.2, PURPLE))

    tick_step = 2 if (max_layer - min_layer) > 24 else 1
    for layer in range(min_layer, max_layer + 1, tick_step):
        body.append(text(x(layer), top + plot_h + 28, f"L{layer}", 10, MUTED, "middle", "700"))
    body.append(text(left + plot_w / 2, height - 20, "layer", 13, DARK, "middle", "700"))
    body.append(text(18, top + plot_h / 2, "summed fused score", 13, DARK, "middle", "700"))
    body[-1] = body[-1].replace(f'x="{18:.2f}" y="{(top + plot_h / 2):.2f}"', f'x="{18:.2f}" y="{(top + plot_h / 2):.2f}" transform="rotate(-90 18.00 {(top + plot_h / 2):.2f})"')

    body.append(rect(left + 18, top + 16, 26, 10, PURPLE, None, 1.0, 5))
    body.append(text(left + 52, top + 25, "score mass", 12, DARK, "start", "700"))
    body.append(rect(left + 150, top + 14, 24, 14, BLUE, None, 0.26, 3))
    body.append(text(left + 182, top + 25, "selected-head count", 12, DARK, "start", "700"))
    if best_window:
        body.append(rect(left + 340, top + 13, 24, 15, "#ede9fe", "#c4b5fd", 0.8, 4))
        body.append(text(left + 372, top + 25, f"top window: {best_window['window']}", 12, DARK, "start", "700"))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n')
        f.write("".join(body))
        f.write("</svg>\n")


def main():
    parser = argparse.ArgumentParser(description="Build a layer profile from a ranked head file without preselecting a layer window.")
    parser.add_argument("--ranked-heads", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--top-k-list", default="50,100,150,200")
    parser.add_argument("--window-sizes", default="6,8,10,12")
    parser.add_argument("--title", default="Layer-wise fused head score profile")
    args = parser.parse_args()

    meta, heads = read_ranked_heads(args.ranked_heads)
    if not heads:
        raise SystemExit(f"no heads found in {args.ranked_heads}")

    os.makedirs(args.output_dir, exist_ok=True)
    top_ks = [int(x) for x in args.top_k_list.replace(" ", ",").split(",") if x.strip()]
    window_sizes = [int(x) for x in args.window_sizes.replace(" ", ",").split(",") if x.strip()]
    max_layer = max(row["layer"] for row in heads)
    min_layer = min(row["layer"] for row in heads)

    profile_rows = []
    for top_k in top_ks:
        selected = heads[: min(top_k, len(heads))]
        score_by_layer = Counter()
        count_by_layer = Counter()
        text_by_layer = Counter()
        contrast_by_layer = Counter()
        for row in selected:
            layer = row["layer"]
            score_by_layer[layer] += max(row["score"], 0.0)
            count_by_layer[layer] += 1
            text_by_layer[layer] += row["text_percentile"]
            contrast_by_layer[layer] += row["contrast_percentile"]
        total_score = sum(score_by_layer.values()) or 1.0
        for layer in range(min_layer, max_layer + 1):
            count = count_by_layer[layer]
            score_sum = score_by_layer[layer]
            profile_rows.append(
                {
                    "top_k": top_k,
                    "layer": layer,
                    "count": count,
                    "score_sum": score_sum,
                    "score_share": score_sum / total_score,
                    "score_mean": score_sum / max(count, 1),
                    "text_percentile_mean": text_by_layer[layer] / max(count, 1),
                    "contrast_percentile_mean": contrast_by_layer[layer] / max(count, 1),
                }
            )

    profile_path = os.path.join(args.output_dir, "layer_profile_summary.csv")
    write_csv(profile_path, profile_rows)

    main_rows = [row for row in profile_rows if safe_int(row["top_k"]) == args.top_k]
    recommendations = recommend_windows(main_rows, min_layer, max_layer, window_sizes)
    rec_path = os.path.join(args.output_dir, "recommended_windows.csv")
    write_csv(rec_path, recommendations)

    best = recommendations[0] if recommendations else None
    svg_path = os.path.join(args.output_dir, f"layer_profile_top{args.top_k}.svg")
    build_svg(svg_path, main_rows, args.title, args.top_k, best_window=best)

    config = {
        "ranked_heads": args.ranked_heads,
        "score_name": meta.get("score_name", ""),
        "n_heads": len(heads),
        "layer_range": [min_layer, max_layer],
        "top_k": args.top_k,
        "top_k_list": top_ks,
        "window_sizes": window_sizes,
        "profile_csv": profile_path,
        "recommended_windows_csv": rec_path,
        "figure_svg": svg_path,
        "best_window": best,
    }
    config_path = os.path.join(args.output_dir, "layer_profile_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()

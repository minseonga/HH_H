import argparse
import csv
import html
import json
import os
from collections import Counter


BLUE = "#2563eb"
ORANGE = "#f97316"
GREEN = "#059669"
RED = "#dc2626"
PURPLE = "#7c3aed"
GRAY = "#94a3b8"
DARK = "#0f172a"
MUTED = "#64748b"
GRID = "#e2e8f0"


def safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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


def text(x, y, value, size=12, fill=DARK, anchor="start", weight="400", rotate=None):
    value = html.escape(str(value))
    transform = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}" '
        f'text-anchor="{anchor}"{transform}>{value}</text>\n'
    )


def rect(x, y, w, h, fill, stroke=None, width=1, opacity=1.0):
    stroke_attr = f' stroke="{stroke}" stroke-width="{width}"' if stroke else ""
    return f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" fill="{fill}" opacity="{opacity}"{stroke_attr}/>\n'


def line(x1, y1, x2, y2, stroke=GRID, width=1, dash=None):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{stroke}" stroke-width="{width}"{dash_attr}/>\n'


def polyline(points, stroke, width=2.5):
    pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polyline points="{pts}" fill="none" stroke="{stroke}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"/>\n'


def circle(x, y, r, fill, stroke=None, width=1, opacity=1.0):
    stroke_attr = f' stroke="{stroke}" stroke-width="{width}"' if stroke else ""
    return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" fill="{fill}" opacity="{opacity}"{stroke_attr}/>\n'


def load_heads(path):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        records = data.get("heads")
        if records is None:
            records = [
                {"layer": layer, "head": head}
                for layer, head in data.get("hal_heads", [])
            ]
    else:
        records = data
    heads = []
    for row in records:
        if isinstance(row, dict):
            heads.append({
                "layer": safe_int(row.get("layer")),
                "head": safe_int(row.get("head")),
                "score": safe_float(row.get("score", row.get("global__itext_all__C_toi_HminusG", row.get("itext_all__C_toi_HminusG", 0.0)))),
            })
        else:
            heads.append({"layer": safe_int(row[0]), "head": safe_int(row[1]), "score": 0.0})
    return data if isinstance(data, dict) else {}, heads


def band_name(layer):
    if layer <= 10:
        return "L0-10 early"
    if layer <= 20:
        return "L11-20 cross-modal"
    if layer <= 26:
        return "L21-26 bridge"
    return "L27-31 late"


def band_order():
    return ["L0-10 early", "L11-20 cross-modal", "L21-26 bridge", "L27-31 late"]


def summarize_layers(heads, top_ks, n_layers):
    rows = []
    band_rows = []
    for top_k in top_ks:
        selected = heads[: min(top_k, len(heads))]
        counts = Counter(row["layer"] for row in selected)
        band_counts = Counter(band_name(row["layer"]) for row in selected)
        by_layer_heads = {}
        for row in selected:
            by_layer_heads.setdefault(row["layer"], []).append(f"{row['layer']}:{row['head']}")
        for layer in range(n_layers):
            count = counts.get(layer, 0)
            rows.append({
                "top_k": top_k,
                "layer": layer,
                "count": count,
                "fraction": count / max(len(selected), 1),
                "heads": " ".join(by_layer_heads.get(layer, [])),
            })
        for band in band_order():
            count = band_counts.get(band, 0)
            band_rows.append({
                "top_k": top_k,
                "band": band,
                "count": count,
                "fraction": count / max(len(selected), 1),
            })
    return rows, band_rows


def color_scale(value, vmax):
    if vmax <= 0 or value <= 0:
        return "#f8fafc"
    t = min(max(value / vmax, 0.0), 1.0)
    # Blue ramp with enough contrast for low counts.
    palette = ["#dbeafe", "#93c5fd", "#60a5fa", "#2563eb", "#1d4ed8"]
    idx = min(int(t * len(palette)), len(palette) - 1)
    return palette[idx]


def make_layer_heatmap_svg(path, rows, top_ks, n_layers, title):
    width, height = 1180, 620
    left, top = 92, 86
    cell_w, cell_h = 30, 48
    plot_w = cell_w * n_layers
    plot_h = cell_h * len(top_ks)
    by = {(safe_int(row["top_k"]), safe_int(row["layer"])): safe_int(row["count"]) for row in rows}
    vmax = max(by.values()) if by else 1
    body = []
    body.append(text(width / 2, 32, title, 21, DARK, "middle", "700"))
    body.append(text(width / 2, 55, "Each cell is the number of selected heads in that layer for the top-k prefix.", 12, MUTED, "middle"))
    body.append(rect(left, top, plot_w, plot_h, "#f8fafc", "#cbd5e1"))

    for layer in range(n_layers + 1):
        x = left + layer * cell_w
        stroke = "#94a3b8" if layer in [11, 21, 27] else GRID
        body.append(line(x, top, x, top + plot_h, stroke, 1.1 if layer in [11, 21, 27] else 0.8))
    for idx in range(len(top_ks) + 1):
        y = top + idx * cell_h
        body.append(line(left, y, left + plot_w, y, GRID))

    for r_idx, top_k in enumerate(top_ks):
        y = top + r_idx * cell_h
        body.append(text(left - 16, y + cell_h / 2 + 4, f"top{top_k}", 12, DARK, "end", "700"))
        for layer in range(n_layers):
            count = by.get((top_k, layer), 0)
            x = left + layer * cell_w
            body.append(rect(x + 1, y + 1, cell_w - 2, cell_h - 2, color_scale(count, vmax), None))
            if count:
                body.append(text(x + cell_w / 2, y + cell_h / 2 + 4, count, 11, "white" if count / vmax > 0.55 else DARK, "middle", "700"))

    for layer in range(n_layers):
        x = left + layer * cell_w + cell_w / 2
        if layer % 2 == 0:
            body.append(text(x, top + plot_h + 20, f"L{layer}", 9, DARK, "middle"))
    for x, label in [
        (left + 5.5 * cell_w, "early"),
        (left + 15.5 * cell_w, "cross-modal"),
        (left + 23.5 * cell_w, "bridge"),
        (left + 29 * cell_w, "late"),
    ]:
        body.append(text(x, top - 14, label, 11, MUTED, "middle", "700"))
    body.append(text(left + plot_w / 2, height - 36, "layer", 13, DARK, "middle"))
    body.append(text(28, top + plot_h / 2, "rank prefix", 13, DARK, "middle", rotate=-90))

    lx, ly = left + plot_w + 34, top + 12
    body.append(text(lx, ly, "count", 12, DARK, "start", "700"))
    for idx in range(5):
        value = round(vmax * idx / 4)
        body.append(rect(lx, ly + 18 + idx * 30, 26, 20, color_scale(value, vmax), "#e2e8f0"))
        body.append(text(lx + 36, ly + 33 + idx * 30, value, 11, DARK))
    svg(path, width, height, "".join(body))


def make_band_svg(path, band_rows, top_ks, title):
    width, height = 900, 560
    left, top = 86, 80
    plot_w, plot_h = 690, 340
    colors = {
        "L0-10 early": GRAY,
        "L11-20 cross-modal": BLUE,
        "L21-26 bridge": ORANGE,
        "L27-31 late": PURPLE,
    }
    by = {(safe_int(row["top_k"]), row["band"]): safe_float(row["fraction"]) for row in band_rows}
    body = []
    body.append(text(width / 2, 32, title, 21, DARK, "middle", "700"))
    body.append(text(width / 2, 55, "Stacked bars show which architectural band supplies each top-k head pool.", 12, MUTED, "middle"))
    body.append(rect(left, top, plot_w, plot_h, "#f8fafc", "#cbd5e1"))
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = top + plot_h - tick * plot_h
        body.append(line(left, y, left + plot_w, y, GRID))
        body.append(text(left - 12, y + 4, f"{tick:.2f}", 10, MUTED, "end"))
    group_w = plot_w / len(top_ks)
    bar_w = min(62, group_w * 0.46)
    for idx, top_k in enumerate(top_ks):
        x = left + idx * group_w + group_w / 2 - bar_w / 2
        y_cursor = top + plot_h
        for band in band_order():
            frac = by.get((top_k, band), 0.0)
            h = frac * plot_h
            y_cursor -= h
            body.append(rect(x, y_cursor, bar_w, h, colors[band], None, 1, 0.88))
            if h > 22:
                body.append(text(x + bar_w / 2, y_cursor + h / 2 + 4, f"{frac:.2f}", 10, "white", "middle", "700"))
        body.append(text(x + bar_w / 2, top + plot_h + 24, f"top{top_k}", 11, DARK, "middle"))
    lx, ly = left + plot_w + 34, top + 20
    for idx, band in enumerate(band_order()):
        body.append(rect(lx, ly + idx * 28, 18, 18, colors[band], None, 1, 0.88))
        body.append(text(lx + 28, ly + idx * 28 + 14, band, 11, DARK))
    body.append(text(left + plot_w / 2, height - 38, "rank prefix", 13, DARK, "middle"))
    body.append(text(28, top + plot_h / 2, "fraction", 13, DARK, "middle", rotate=-90))
    svg(path, width, height, "".join(body))


def make_single_topk_layer_histogram_svg(path, rows, top_k, n_layers, title):
    width, height = 1120, 560
    left, top = 78, 78
    plot_w, plot_h = 930, 340
    by = {safe_int(row["layer"]): safe_int(row["count"]) for row in rows if safe_int(row["top_k"]) == top_k}
    vmax = max(max(by.values()) if by else 1, 1)
    body = []
    body.append(text(width / 2, 32, f"{title}: top{top_k}", 21, DARK, "middle", "700"))
    body.append(text(width / 2, 55, "One bar per transformer layer. Counts are selected heads from the ranked prefix.", 12, MUTED, "middle"))
    body.append(rect(left, top, plot_w, plot_h, "#f8fafc", "#cbd5e1"))

    for tick in range(0, vmax + 1):
        if vmax > 8 and tick % 2:
            continue
        y = top + plot_h - tick / vmax * plot_h
        body.append(line(left, y, left + plot_w, y, GRID))
        body.append(text(left - 12, y + 4, tick, 10, MUTED, "end"))

    layer_w = plot_w / n_layers
    bar_w = min(20, layer_w * 0.62)
    for layer in range(n_layers):
        count = by.get(layer, 0)
        h = count / vmax * plot_h
        x = left + layer * layer_w + layer_w / 2
        fill = BLUE if 11 <= layer <= 20 else ORANGE if 21 <= layer <= 26 else PURPLE if layer >= 27 else GRAY
        body.append(rect(x - bar_w / 2, top + plot_h - h, bar_w, h, fill, None, 1, 0.9))
        if count:
            body.append(text(x, top + plot_h - h - 7, count, 9, DARK, "middle", "700"))
        body.append(text(x, top + plot_h + 20, f"L{layer}", 8, DARK, "middle", rotate=35))

    for boundary in [11, 21, 27]:
        x = left + boundary * layer_w
        body.append(line(x, top, x, top + plot_h, "#334155", 1.2, "5 5"))
    body.append(text(left + 5.5 * layer_w, top - 13, "early", 11, MUTED, "middle", "700"))
    body.append(text(left + 15.5 * layer_w, top - 13, "cross-modal", 11, MUTED, "middle", "700"))
    body.append(text(left + 23.5 * layer_w, top - 13, "bridge", 11, MUTED, "middle", "700"))
    body.append(text(left + 29.0 * layer_w, top - 13, "late", 11, MUTED, "middle", "700"))
    body.append(text(left + plot_w / 2, height - 36, "layer", 13, DARK, "middle"))
    body.append(text(26, top + plot_h / 2, "head count", 13, DARK, "middle", rotate=-90))

    lx, ly = left + plot_w + 28, top + 28
    for idx, (label, color) in enumerate([
        ("L0-10", GRAY),
        ("L11-20", BLUE),
        ("L21-26", ORANGE),
        ("L27-31", PURPLE),
    ]):
        body.append(rect(lx, ly + idx * 28, 18, 18, color, None, 1, 0.9))
        body.append(text(lx + 27, ly + idx * 28 + 14, label, 11, DARK))
    svg(path, width, height, "".join(body))


def make_grouped_layer_histogram_svg(path, rows, top_ks, n_layers, title):
    width, height = 1320, 620
    left, top = 80, 82
    plot_w, plot_h = 1100, 390
    colors = [BLUE, GREEN, ORANGE, PURPLE, RED, GRAY]
    by = {(safe_int(row["top_k"]), safe_int(row["layer"])): safe_int(row["count"]) for row in rows}
    vmax = max(by.values()) if by else 1
    body = []
    body.append(text(width / 2, 32, title, 21, DARK, "middle", "700"))
    body.append(text(width / 2, 55, "Grouped bars compare exact layer counts across rank prefixes.", 12, MUTED, "middle"))
    body.append(rect(left, top, plot_w, plot_h, "#f8fafc", "#cbd5e1"))

    for tick in range(0, vmax + 1):
        if vmax > 8 and tick % 2:
            continue
        y = top + plot_h - tick / max(vmax, 1) * plot_h
        body.append(line(left, y, left + plot_w, y, GRID))
        body.append(text(left - 12, y + 4, tick, 10, MUTED, "end"))

    layer_w = plot_w / n_layers
    bar_w = min(5.5, layer_w / max(len(top_ks), 1) * 0.74)
    for layer in range(n_layers):
        cx = left + layer * layer_w + layer_w / 2
        for idx, top_k in enumerate(top_ks):
            count = by.get((top_k, layer), 0)
            h = count / max(vmax, 1) * plot_h
            x = cx + (idx - (len(top_ks) - 1) / 2.0) * (bar_w + 1.5)
            body.append(rect(x - bar_w / 2, top + plot_h - h, bar_w, h, colors[idx % len(colors)], None, 1, 0.86))
        body.append(text(cx, top + plot_h + 20, f"L{layer}", 8, DARK, "middle", rotate=35))

    for boundary in [11, 21, 27]:
        x = left + boundary * layer_w
        body.append(line(x, top, x, top + plot_h, "#334155", 1.2, "5 5"))
    lx, ly = left + plot_w + 30, top + 20
    for idx, top_k in enumerate(top_ks):
        body.append(rect(lx, ly + idx * 26, 18, 16, colors[idx % len(colors)], None, 1, 0.86))
        body.append(text(lx + 27, ly + idx * 26 + 13, f"top{top_k}", 11, DARK))
    body.append(text(left + plot_w / 2, height - 36, "layer", 13, DARK, "middle"))
    body.append(text(26, top + plot_h / 2, "head count", 13, DARK, "middle", rotate=-90))
    svg(path, width, height, "".join(body))


def make_layer_line_svg(path, rows, top_ks, n_layers, title):
    width, height = 1240, 620
    left, top = 78, 82
    plot_w, plot_h = 1010, 390
    colors = [BLUE, GREEN, ORANGE, PURPLE, RED, GRAY]
    by = {(safe_int(row["top_k"]), safe_int(row["layer"])): safe_int(row["count"]) for row in rows}
    vmax = max(by.values()) if by else 1

    def sx(layer):
        return left + layer / max(n_layers - 1, 1) * plot_w

    def sy(value):
        return top + plot_h - value / max(vmax, 1) * plot_h

    body = []
    body.append(text(width / 2, 32, title, 21, DARK, "middle", "700"))
    body.append(text(width / 2, 55, "X-axis is the exact transformer layer; each line is one ranked top-k prefix.", 12, MUTED, "middle"))
    body.append(rect(left, top, plot_w, plot_h, "#f8fafc", "#cbd5e1"))

    for tick in range(0, vmax + 1):
        if vmax > 8 and tick % 2:
            continue
        y = sy(tick)
        body.append(line(left, y, left + plot_w, y, GRID))
        body.append(text(left - 12, y + 4, tick, 10, MUTED, "end"))

    for layer in range(n_layers):
        x = sx(layer)
        if layer in [11, 21, 27]:
            body.append(line(x, top, x, top + plot_h, "#334155", 1.15, "5 5"))
        elif layer % 2 == 0:
            body.append(line(x, top, x, top + plot_h, "#edf2f7", 0.8))
        body.append(text(x, top + plot_h + 22, f"L{layer}", 8, DARK, "middle", rotate=35))

    body.append(text((sx(0) + sx(10)) / 2, top - 13, "early", 11, MUTED, "middle", "700"))
    body.append(text((sx(11) + sx(20)) / 2, top - 13, "cross-modal", 11, MUTED, "middle", "700"))
    body.append(text((sx(21) + sx(26)) / 2, top - 13, "bridge", 11, MUTED, "middle", "700"))
    body.append(text((sx(27) + sx(31)) / 2, top - 13, "late", 11, MUTED, "middle", "700"))

    for idx, top_k in enumerate(top_ks):
        color = colors[idx % len(colors)]
        points = [(sx(layer), sy(by.get((top_k, layer), 0))) for layer in range(n_layers)]
        body.append(polyline(points, color, 2.8))
        for layer, (x, y) in enumerate(points):
            count = by.get((top_k, layer), 0)
            if count:
                body.append(circle(x, y, 3.4, color, "white", 1.1, 0.95))

    lx, ly = left + plot_w + 32, top + 20
    for idx, top_k in enumerate(top_ks):
        color = colors[idx % len(colors)]
        y = ly + idx * 28
        body.append(line(lx, y, lx + 32, y, color, 3))
        body.append(circle(lx + 16, y, 3.4, color, "white", 1.0))
        body.append(text(lx + 42, y + 4, f"top{top_k}", 11, DARK))
    body.append(text(left + plot_w / 2, height - 36, "layer", 13, DARK, "middle"))
    body.append(text(26, top + plot_h / 2, "head count", 13, DARK, "middle", rotate=-90))
    svg(path, width, height, "".join(body))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ranked-heads",
        default="../ADHH/LLaVA/results/coco/llava-v1.5-7b_base_original_qa_n3000/surrogate_hh_scores/surrogate_score_zoo/ranked_heads_global__itext_all__C_toi_HminusG.json",
    )
    parser.add_argument("--top-ks", default="20,50,100,150,200")
    parser.add_argument("--n-layers", type=int, default=32)
    parser.add_argument("--output-dir", default="./results/coco/ranked_head_layer_distribution")
    parser.add_argument("--title", default="Layer distribution of ranked hallucination-suppression heads")
    args = parser.parse_args()

    meta, heads = load_heads(args.ranked_heads)
    top_ks = [int(item) for item in args.top_ks.replace(" ", "").split(",") if item]
    top_ks = [top_k for top_k in top_ks if top_k > 0]
    layer_rows, band_rows = summarize_layers(heads, top_ks, args.n_layers)

    os.makedirs(args.output_dir, exist_ok=True)
    layer_csv = os.path.join(args.output_dir, "topk_layer_distribution.csv")
    band_csv = os.path.join(args.output_dir, "topk_layer_band_distribution.csv")
    layer_svg = os.path.join(args.output_dir, "topk_layer_distribution.svg")
    band_svg = os.path.join(args.output_dir, "topk_layer_band_distribution.svg")
    grouped_hist_svg = os.path.join(args.output_dir, "topk_layer_histogram_grouped.svg")
    line_svg = os.path.join(args.output_dir, "topk_layer_distribution_lines.svg")
    write_csv(layer_csv, layer_rows)
    write_csv(band_csv, band_rows)
    make_layer_heatmap_svg(layer_svg, layer_rows, top_ks, args.n_layers, args.title)
    make_band_svg(band_svg, band_rows, top_ks, "Layer-band distribution of ranked head pools")
    make_grouped_layer_histogram_svg(grouped_hist_svg, layer_rows, top_ks, args.n_layers, "Exact per-layer head-count distribution")
    make_layer_line_svg(line_svg, layer_rows, top_ks, args.n_layers, "Exact per-layer head-count distribution")
    per_topk_histograms = {}
    for top_k in top_ks:
        path = os.path.join(args.output_dir, f"layer_histogram_top{top_k}.svg")
        make_single_topk_layer_histogram_svg(path, layer_rows, top_k, args.n_layers, "Exact per-layer head-count distribution")
        per_topk_histograms[str(top_k)] = path

    summary = {
        "ranked_heads": args.ranked_heads,
        "score_name": meta.get("score_name"),
        "n_heads": len(heads),
        "top_ks": top_ks,
        "outputs": {
            "layer_csv": layer_csv,
            "band_csv": band_csv,
            "layer_svg": layer_svg,
            "band_svg": band_svg,
            "grouped_hist_svg": grouped_hist_svg,
            "line_svg": line_svg,
            "per_topk_histograms": per_topk_histograms,
        },
    }
    with open(os.path.join(args.output_dir, "layer_distribution_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

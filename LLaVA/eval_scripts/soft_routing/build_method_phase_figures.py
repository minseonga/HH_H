import argparse
import csv
import html
import json
import math
import os


BLUE = "#2563eb"
ORANGE = "#f97316"
GREEN = "#059669"
RED = "#dc2626"
PURPLE = "#7c3aed"
GRAY = "#64748b"
LIGHT_GRAY = "#e2e8f0"
DARK = "#0f172a"
MUTED = "#64748b"
PANEL = "#f8fafc"


def safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        value = float(value)
        if math.isnan(value):
            return default
        return value
    except (TypeError, ValueError):
        return default


def mean(values):
    values = [safe_float(value, None) for value in values]
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else 0.0


def quantile(values, q):
    values = sorted(safe_float(value, None) for value in values)
    values = [value for value in values if value is not None]
    if not values:
        return 0.0
    idx = min(max(int(round((len(values) - 1) * q)), 0), len(values) - 1)
    return values[idx]


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


def write_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(value, f, indent=2)


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


def rect(x, y, w, h, fill, stroke=None, width=1, opacity=1.0, rx=0):
    stroke_attr = f' stroke="{stroke}" stroke-width="{width}"' if stroke else ""
    return (
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
        f'rx="{rx:.2f}" fill="{fill}" opacity="{opacity}"{stroke_attr}/>\n'
    )


def line(x1, y1, x2, y2, stroke=LIGHT_GRAY, width=1, dash=None):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{stroke}" stroke-width="{width}"{dash_attr}/>\n'


def circle(x, y, r, fill, stroke=None, width=1, opacity=1.0):
    stroke_attr = f' stroke="{stroke}" stroke-width="{width}"' if stroke else ""
    return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" fill="{fill}" opacity="{opacity}"{stroke_attr}/>\n'


def polyline(points, stroke, width=2.5, opacity=1.0):
    pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polyline points="{pts}" fill="none" stroke="{stroke}" stroke-width="{width}" opacity="{opacity}" stroke-linejoin="round" stroke-linecap="round"/>\n'


def polygon(points, fill, opacity=0.72):
    pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polygon points="{pts}" fill="{fill}" opacity="{opacity}"/>\n'


def arrow(x1, y1, x2, y2, color=GRAY, width=1.7):
    return (
        f'<defs><marker id="arrow-{abs(hash((x1, y1, x2, y2))) % 100000}" '
        'markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">'
        f'<polygon points="0 0, 10 3.5, 0 7" fill="{color}"/></marker></defs>\n'
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
        f'stroke="{color}" stroke-width="{width}" marker-end="url(#arrow-{abs(hash((x1, y1, x2, y2))) % 100000})"/>\n'
    )


def load_ranked_heads(path):
    with open(path) as f:
        data = json.load(f)
    records = list(data.get("heads", []))
    score_name = data.get("score_name")
    row_keys = list(records[0].keys()) if records else []
    score_key = score_name if score_name in row_keys else None
    if score_key is None and score_name:
        for prefix in ("global__", "local__"):
            stripped = score_name.removeprefix(prefix)
            if stripped in row_keys:
                score_key = stripped
                break
    if score_key is None:
        for key in row_keys:
            if "__" in key and key not in {"selection_method"}:
                score_key = key
                break
    score_key = score_key or "score"
    for idx, row in enumerate(records, start=1):
        row["_rank"] = idx
        row["_key"] = f"{int(row['layer'])}:{int(row['head'])}"
        row["_score"] = safe_float(row.get(score_key, row.get("score", 0.0)))
    return data, records, score_key


def add_component_ranks(records):
    for rank, row in enumerate(
        sorted(records, key=lambda item: safe_float(item.get("front_percentile")), reverse=True),
        start=1,
    ):
        row["_front_rank"] = rank
    for rank, row in enumerate(
        sorted(records, key=lambda item: safe_float(item.get("back_percentile")), reverse=True),
        start=1,
    ):
        row["_back_rank"] = rank


def online_ratio(row, label):
    text_key = f"Itext_{label}"
    img_key = f"Img_{label}"
    t = safe_float(row.get(text_key))
    i = safe_float(row.get(img_key))
    return t / max(t + i, 1e-12)


def gate_delta(r, score, strength, beta, tau):
    return min(max(strength * math.exp(beta * (r - tau)) * score, 0.0), 1.0)


def color_scale(value):
    value = min(max(value, 0.0), 1.0)
    # Light blue to red, readable on white.
    r1, g1, b1 = 226, 232, 240
    r2, g2, b2 = 220, 38, 38
    r = int(r1 + (r2 - r1) * value)
    g = int(g1 + (g2 - g1) * value)
    b = int(b1 + (b2 - b1) * value)
    return f"#{r:02x}{g:02x}{b:02x}"


def summarize_buckets(records, top_ks):
    rows = []
    max_top = max(top_ks)
    specs = [(f"top{k}", records[:k]) for k in top_ks]
    specs.append((f"rank>{max_top}", records[max_top:]))
    for label, rows_in_bucket in specs:
        rows.append({
            "bucket": label,
            "n_heads": len(rows_in_bucket),
            "mean_score": mean(row["_score"] for row in rows_in_bucket),
            "mean_itext_all": mean(row.get("Itext_all") for row in rows_in_bucket),
            "mean_itext_hallucinated": mean(row.get("Itext_hallucinated") for row in rows_in_bucket),
            "mean_itext_grounded": mean(row.get("Itext_non_hallucinated") for row in rows_in_bucket),
            "mean_log_toi_hallucinated": mean(row.get("LogTOI_hallucinated") for row in rows_in_bucket),
            "mean_log_toi_grounded": mean(row.get("LogTOI_non_hallucinated") for row in rows_in_bucket),
            "mean_log_toi_gap": mean(
                safe_float(row.get("LogTOI_hallucinated")) - safe_float(row.get("LogTOI_non_hallucinated"))
                for row in rows_in_bucket
            ),
            "positive_raw_toi_gap_fraction": mean(
                1.0
                if safe_float(row.get("RawTOI_hallucinated")) > safe_float(row.get("RawTOI_non_hallucinated"))
                else 0.0
                for row in rows_in_bucket
            ),
            "mean_image_drop_grounded_minus_hall": mean(
                safe_float(row.get("Img_non_hallucinated")) - safe_float(row.get("Img_hallucinated"))
                for row in rows_in_bucket
            ),
        })
    return rows


def make_text_mass_bars(path, bucket_rows):
    width, height = 900, 520
    left, top, plot_w, plot_h = 82, 78, 680, 330
    vmax = max(max(row["mean_itext_all"], row["mean_itext_hallucinated"], row["mean_itext_grounded"]) for row in bucket_rows)
    vmax = min(max(vmax * 1.16, 0.1), 1.0)

    def sy(v):
        return top + plot_h - v / vmax * plot_h

    body = []
    body.append(text(width / 2, 30, "Phase 1: text-side mass is the leverage signal", 20, DARK, "middle", "700"))
    body.append(text(width / 2, 53, "Bars use the actual ranked-head records; higher Itext means the suppressible text slice carries more mass.", 12, MUTED, "middle"))
    body.append(rect(left, top, plot_w, plot_h, PANEL, "#cbd5e1"))
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        if tick > vmax:
            continue
        y = sy(tick)
        body.append(line(left, y, left + plot_w, y))
        body.append(text(left - 12, y + 4, f"{tick:.2f}", 10, MUTED, "end"))
    group_w = plot_w / max(len(bucket_rows), 1)
    colors = [ORANGE, RED, BLUE]
    labels = ["all steps", "hall object", "grounded object"]
    keys = ["mean_itext_all", "mean_itext_hallucinated", "mean_itext_grounded"]
    for idx, row in enumerate(bucket_rows):
        cx = left + idx * group_w + group_w / 2
        bar_w = min(30, group_w * 0.18)
        for j, key in enumerate(keys):
            x = cx + (j - 1) * (bar_w + 4) - bar_w / 2
            y = sy(row[key])
            body.append(rect(x, y, bar_w, top + plot_h - y, colors[j], opacity=0.86))
        body.append(text(cx, top + plot_h + 22, row["bucket"], 10, DARK, "middle"))
    lx, ly = left + plot_w + 32, top + 22
    for idx, label in enumerate(labels):
        body.append(rect(lx, ly + idx * 28 - 12, 20, 16, colors[idx], opacity=0.86))
        body.append(text(lx + 30, ly + idx * 28 + 1, label, 11, DARK))
    body.append(text(left + plot_w / 2, height - 38, "head bucket", 13, DARK, "middle"))
    body.append(text(26, top + plot_h / 2, "mean text-side mass", 13, DARK, "middle", rotate=-90))
    svg(path, width, height, "".join(body))


def make_contrast_distribution(path, records, top_k):
    selected = records[:top_k]
    hall = [safe_float(row.get("LogTOI_hallucinated")) for row in selected]
    grounded = [safe_float(row.get("LogTOI_non_hallucinated")) for row in selected]
    values = hall + grounded
    lo, hi = min(values), max(values)
    pad = max((hi - lo) * 0.08, 0.1)
    lo -= pad
    hi += pad
    bins = 24

    def hist(vals):
        counts = [0] * bins
        for value in vals:
            idx = int((value - lo) / max(hi - lo, 1e-9) * bins)
            idx = min(max(idx, 0), bins - 1)
            counts[idx] += 1
        return counts

    h_counts = hist(hall)
    g_counts = hist(grounded)
    ymax = max(max(h_counts), max(g_counts), 1)
    width, height = 940, 540
    left, top, plot_w, plot_h = 82, 78, 700, 340

    def sx(v):
        return left + (v - lo) / max(hi - lo, 1e-9) * plot_w

    def sy(v):
        return top + plot_h - v / ymax * plot_h

    body = []
    body.append(text(width / 2, 30, f"Phase 1: contrastive bias distribution for top{top_k} heads", 20, DARK, "middle", "700"))
    body.append(text(width / 2, 53, "Log text-over-image ratio is compared at hallucinated vs grounded object steps.", 12, MUTED, "middle"))
    body.append(rect(left, top, plot_w, plot_h, PANEL, "#cbd5e1"))
    for tick in range(0, ymax + 1, max(1, math.ceil(ymax / 4))):
        y = sy(tick)
        body.append(line(left, y, left + plot_w, y))
        body.append(text(left - 12, y + 4, str(tick), 10, MUTED, "end"))
    bar_w = plot_w / bins
    for idx in range(bins):
        x = left + idx * bar_w
        y = sy(h_counts[idx])
        body.append(rect(x + 1, y, bar_w * 0.48, top + plot_h - y, RED, opacity=0.68))
        y = sy(g_counts[idx])
        body.append(rect(x + bar_w * 0.50, y, bar_w * 0.48, top + plot_h - y, BLUE, opacity=0.62))
    for tick in [quantile(values, q) for q in [0.0, 0.25, 0.5, 0.75, 1.0]]:
        x = sx(tick)
        body.append(line(x, top + plot_h, x, top + plot_h + 5, MUTED))
        body.append(text(x, top + plot_h + 21, f"{tick:.1f}", 10, MUTED, "middle"))
    hall_mean = mean(hall)
    grounded_mean = mean(grounded)
    body.append(line(sx(hall_mean), top, sx(hall_mean), top + plot_h, RED, 1.8, "5 5"))
    body.append(line(sx(grounded_mean), top, sx(grounded_mean), top + plot_h, BLUE, 1.8, "5 5"))
    lx, ly = left + plot_w + 30, top + 30
    body.append(rect(lx, ly - 13, 20, 16, RED, opacity=0.68))
    body.append(text(lx + 30, ly, "hallucinated object", 11, DARK))
    body.append(rect(lx, ly + 18, 20, 16, BLUE, opacity=0.62))
    body.append(text(lx + 30, ly + 31, "grounded object", 11, DARK))
    body.append(text(lx, ly + 76, f"mean gap: {hall_mean - grounded_mean:.3f}", 12, DARK, "start", "700"))
    body.append(text(lx, ly + 98, f"positive gap: {sum(h > g for h, g in zip(hall, grounded))}/{len(selected)}", 12, DARK))
    body.append(text(left + plot_w / 2, height - 42, "log text-over-image ratio", 13, DARK, "middle"))
    body.append(text(26, top + plot_h / 2, "head count", 13, DARK, "middle", rotate=-90))
    svg(path, width, height, "".join(body))


def make_head_score_heatmap(path, records, top_k, layer_min, layer_max, head_count, highlight_start, highlight_end):
    by_key = {(int(row["layer"]), int(row["head"])): row for row in records}
    selected = {(int(row["layer"]), int(row["head"])) for row in records[:top_k]}
    scored_layers = sorted({int(row["layer"]) for row in records})
    width, height = 1180, 760
    left, top = 86, 84
    cell = min(26, 840 / max(layer_max - layer_min + 1, 1), 520 / head_count)
    plot_w = cell * (layer_max - layer_min + 1)
    plot_h = cell * head_count
    body = []
    body.append(text(width / 2, 30, f"Phase 1: layer x head score heatmap with top{top_k} selection", 20, DARK, "middle", "700"))
    body.append(text(width / 2, 53, "Cell color is fused head score; top-k heads are outlined. The L9-L16 intervention window is shaded.", 12, MUTED, "middle"))
    body.append(rect(left, top, plot_w, plot_h, "#f1f5f9", "#cbd5e1"))
    sx0 = left + (highlight_start - layer_min) * cell
    sx1 = left + (highlight_end - layer_min + 1) * cell
    body.append(rect(sx0, top, max(sx1 - sx0, 0), plot_h, "#fef3c7", opacity=0.42))
    for layer in range(layer_min, layer_max + 1):
        x = left + (layer - layer_min) * cell
        if layer not in scored_layers:
            body.append(rect(x, top, cell, plot_h, "#e5e7eb", opacity=0.38))
        if layer % 2 == 0:
            body.append(line(x, top, x, top + plot_h, "#e2e8f0"))
        body.append(text(x + cell / 2, top + plot_h + 18, f"L{layer}", 8, DARK, "middle", rotate=35))
    for head in range(head_count):
        y = top + head * cell
        if head % 4 == 0:
            body.append(line(left, y, left + plot_w, y, "#e2e8f0"))
            body.append(text(left - 10, y + cell * 0.72, f"H{head}", 8, MUTED, "end"))
        for layer in range(layer_min, layer_max + 1):
            x = left + (layer - layer_min) * cell
            row = by_key.get((layer, head))
            if row is None:
                continue
            score = safe_float(row.get("_score"))
            body.append(rect(x + 1, y + 1, cell - 2, cell - 2, color_scale(score)))
            if (layer, head) in selected:
                body.append(rect(x + 1.2, y + 1.2, cell - 2.4, cell - 2.4, "none", DARK, 1.3))
    body.append(rect(sx0, top, max(sx1 - sx0, 0), plot_h, "none", "#f59e0b", 2.0))
    body.append(text((sx0 + sx1) / 2, top - 12, "best empirical window L9-L16", 11, "#92400e", "middle", "700"))
    lx, ly = left + plot_w + 42, top + 20
    for idx, value in enumerate([0.0, 0.25, 0.5, 0.75, 1.0]):
        body.append(rect(lx, ly + idx * 25, 26, 18, color_scale(value)))
        body.append(text(lx + 36, ly + idx * 25 + 14, f"{value:.2f}", 10, DARK))
    body.append(text(lx, ly + 150, "not scored", 11, DARK, "start", "700"))
    body.append(rect(lx, ly + 162, 26, 18, "#e5e7eb", opacity=0.55))
    body.append(text(lx + 36, ly + 176, "outside ranked pool", 10, DARK))
    body.append(text(left + plot_w / 2, height - 38, "transformer layer", 13, DARK, "middle"))
    body.append(text(28, top + plot_h / 2, "attention head", 13, DARK, "middle", rotate=-90))
    svg(path, width, height, "".join(body))


def make_rank_fusion_flow(path, records, top_n):
    selected = records[:top_n]
    width, height = 1180, 620
    left, top, row_h = 80, 96, 22
    col_x = [150, 420, 690, 930]
    max_rows = min(top_n, 18)
    body = []
    body.append(text(width / 2, 30, "Phase 1: rank fusion flow", 20, DARK, "middle", "700"))
    body.append(text(width / 2, 53, "Each row is one selected head: text leverage percentile and contrast percentile are averaged into the fused score.", 12, MUTED, "middle"))
    headers = ["head", "Itext percentile", "contrast percentile", "fused score"]
    for x, header in zip(col_x, headers):
        body.append(text(x, top - 26, header, 12, DARK, "middle", "700"))
    body.append(line(col_x[1] + 98, top - 19, col_x[2] - 98, top - 19, MUTED, 1.5))
    body.append(text((col_x[1] + col_x[2]) / 2, top - 25, "+", 15, MUTED, "middle", "700"))
    body.append(line(col_x[2] + 102, top - 19, col_x[3] - 102, top - 19, MUTED, 1.5))
    body.append(text((col_x[2] + col_x[3]) / 2, top - 25, "/ 2", 13, MUTED, "middle", "700"))
    for idx, row in enumerate(selected[:max_rows]):
        y = top + idx * row_h
        if idx % 2 == 0:
            body.append(rect(left, y - 14, 1000, row_h, "#f8fafc"))
        body.append(text(col_x[0], y, f"L{row['layer']}H{row['head']}", 11, DARK, "middle"))
        values = [
            safe_float(row.get("front_percentile")),
            safe_float(row.get("back_percentile")),
            safe_float(row.get("_score")),
        ]
        colors = [ORANGE, GREEN, RED]
        for j, value in enumerate(values):
            x = col_x[j + 1] - 75
            body.append(rect(x, y - 10, 150, 12, "#e2e8f0"))
            body.append(rect(x, y - 10, 150 * value, 12, colors[j], opacity=0.82))
            body.append(text(x + 158, y, f"{value:.3f}", 10, DARK))
        body.append(line(col_x[1] + 90, y - 4, col_x[2] - 90, y - 4, "#cbd5e1"))
        body.append(line(col_x[2] + 90, y - 4, col_x[3] - 90, y - 4, "#cbd5e1"))
    body.append(text(left, height - 48, "Data source: ranked_heads JSON. This panel is useful for explaining why the pool is not text-mass-only.", 12, MUTED))
    svg(path, width, height, "".join(body))


def make_gate_curve(path, records, top_k, strength, beta, tau):
    selected = records[:top_k]
    hall_ratios = [online_ratio(row, "hallucinated") for row in selected]
    grounded_ratios = [online_ratio(row, "non_hallucinated") for row in selected]
    width, height = 940, 600
    left, top, plot_w, plot_h = 86, 82, 700, 360

    def sx(r):
        return left + r * plot_w

    def sy(v):
        return top + plot_h - v * plot_h

    body = []
    body.append(text(width / 2, 30, f"Phase 2: online text ratio drives an exponential gate (top{top_k})", 20, DARK, "middle", "700"))
    body.append(text(width / 2, 53, f"delta = clip(s * exp(q(r - tau)) * S(l,h)); shown with s={strength}, q={beta}, tau={tau}, S=1.", 12, MUTED, "middle"))
    body.append(rect(left, top, plot_w, plot_h, PANEL, "#cbd5e1"))
    for tick in [0, 0.25, 0.5, 0.75, tau, 1.0]:
        x = sx(tick)
        body.append(line(x, top, x, top + plot_h, "#334155" if abs(tick - tau) < 1e-9 else LIGHT_GRAY, 1.2, "5 5" if abs(tick - tau) < 1e-9 else None))
        body.append(text(x, top + plot_h + 21, f"{tick:.2f}", 10, MUTED, "middle"))
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        y = sy(tick)
        body.append(line(left, y, left + plot_w, y))
        body.append(text(left - 12, y + 4, f"{tick:.2f}", 10, MUTED, "end"))
    pts = []
    for i in range(301):
        r = i / 300
        pts.append((sx(r), sy(min(max(strength * math.exp(beta * (r - tau)), 0.0), 1.0))))
    body.append(polyline(pts, RED, 3.0))
    # Rug plot from actual top-k head records.
    for r in hall_ratios:
        body.append(line(sx(r), top + plot_h + 34, sx(r), top + plot_h + 46, RED, 0.8, None))
    for r in grounded_ratios:
        body.append(line(sx(r), top + plot_h + 50, sx(r), top + plot_h + 62, BLUE, 0.8, None))
    lx, ly = left + plot_w + 32, top + 28
    body.append(text(lx, ly, "actual ratio summary", 12, DARK, "start", "700"))
    body.append(text(lx, ly + 28, f"hall q50: {quantile(hall_ratios, 0.5):.3f}", 12, RED))
    body.append(text(lx, ly + 50, f"ground q50: {quantile(grounded_ratios, 0.5):.3f}", 12, BLUE))
    body.append(text(lx, ly + 72, f"hall q75: {quantile(hall_ratios, 0.75):.3f}", 12, RED))
    body.append(text(lx, ly + 94, f"ground q75: {quantile(grounded_ratios, 0.75):.3f}", 12, BLUE))
    body.append(text(lx, ly + 130, "rug marks below x-axis", 11, MUTED))
    body.append(text(lx, ly + 151, "red = hallucinated", 11, RED))
    body.append(text(lx, ly + 172, "blue = grounded", 11, BLUE))
    body.append(text(left + plot_w / 2, height - 42, "online text ratio r = T / (T + I)", 13, DARK, "middle"))
    body.append(text(28, top + plot_h / 2, "gate value before head score", 13, DARK, "middle", rotate=-90))
    svg(path, width, height, "".join(body))


def redistribute(row, label, strength, beta, tau):
    t = safe_float(row.get(f"Itext_{label}"))
    i = safe_float(row.get(f"Img_{label}"))
    r = t / max(t + i, 1e-12)
    delta = gate_delta(r, safe_float(row.get("_score")), strength, beta, tau)
    t_after_raw = (1.0 - delta) * t
    denom = max(t_after_raw + i, 1e-12)
    return {
        "text_before": t / max(t + i, 1e-12),
        "image_before": i / max(t + i, 1e-12),
        "text_after": t_after_raw / denom,
        "image_after": i / denom,
        "delta": delta,
        "ratio": r,
    }


def make_redistribution_bar(path, records, top_k, strength, beta, tau):
    selected = records[:top_k]
    hall = [redistribute(row, "hallucinated", strength, beta, tau) for row in selected]
    grounded = [redistribute(row, "non_hallucinated", strength, beta, tau) for row in selected]
    rows = [
        ("hall before", mean(x["text_before"] for x in hall), mean(x["image_before"] for x in hall)),
        ("hall after", mean(x["text_after"] for x in hall), mean(x["image_after"] for x in hall)),
        ("ground before", mean(x["text_before"] for x in grounded), mean(x["image_before"] for x in grounded)),
        ("ground after", mean(x["text_after"] for x in grounded), mean(x["image_after"] for x in grounded)),
    ]
    width, height = 860, 520
    left, top, plot_w, plot_h = 90, 80, 600, 320
    body = []
    body.append(text(width / 2, 30, f"Phase 2: suppression redistributes attention away from text-side (top{top_k})", 20, DARK, "middle", "700"))
    body.append(text(width / 2, 53, "Before/after is computed from actual selected-head object-step T/I records using the method gate.", 12, MUTED, "middle"))
    body.append(rect(left, top, plot_w, plot_h, PANEL, "#cbd5e1"))
    group_w = plot_w / len(rows)
    for idx, (label, text_value, image_value) in enumerate(rows):
        x = left + idx * group_w + group_w * 0.25
        w = group_w * 0.50
        text_h = text_value * plot_h
        image_h = image_value * plot_h
        y0 = top + plot_h
        body.append(rect(x, y0 - image_h, w, image_h, BLUE, opacity=0.82))
        body.append(rect(x, y0 - image_h - text_h, w, text_h, ORANGE, opacity=0.82))
        body.append(text(x + w / 2, top + plot_h + 24, label, 10, DARK, "middle", rotate=15))
        body.append(text(x + w / 2, y0 - image_h - text_h - 7, f"T {text_value:.2f}", 10, ORANGE, "middle"))
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        y = top + plot_h - tick * plot_h
        body.append(line(left, y, left + plot_w, y))
        body.append(text(left - 12, y + 4, f"{tick:.2f}", 10, MUTED, "end"))
    lx, ly = left + plot_w + 32, top + 28
    body.append(rect(lx, ly - 13, 20, 16, ORANGE, opacity=0.82))
    body.append(text(lx + 30, ly, "text-side", 12, DARK))
    body.append(rect(lx, ly + 18, 20, 16, BLUE, opacity=0.82))
    body.append(text(lx + 30, ly + 31, "image", 12, DARK))
    body.append(text(lx, ly + 74, f"mean delta hall: {mean(x['delta'] for x in hall):.3f}", 12, RED))
    body.append(text(lx, ly + 96, f"mean delta ground: {mean(x['delta'] for x in grounded):.3f}", 12, BLUE))
    body.append(text(left + plot_w / 2, height - 42, "object-step bucket", 13, DARK, "middle"))
    body.append(text(28, top + plot_h / 2, "share within T/I slice", 13, DARK, "middle", rotate=-90))
    svg(path, width, height, "".join(body))
    return hall, grounded


def make_delta_flow(path, records, top_n, strength, beta, tau):
    selected = records[:top_n]
    width, height = 1080, 520
    left, top, row_h = 84, 100, 36
    columns = [130, 295, 460, 625, 790, 940]
    headers = ["head", "S(l,h)", "r_hall", "exp(q(r-tau))", "delta", "action"]
    body = []
    body.append(text(width / 2, 30, "Phase 2: head score and online ratio determine per-head suppression", 20, DARK, "middle", "700"))
    body.append(text(width / 2, 53, "Values use hallucinated object-step records from the selected head pool.", 12, MUTED, "middle"))
    for x, header in zip(columns, headers):
        body.append(text(x, top - 28, header, 12, DARK, "middle", "700"))
    for idx, row in enumerate(selected):
        y = top + idx * row_h
        if idx % 2 == 0:
            body.append(rect(left, y - 20, 920, row_h, "#f8fafc"))
        r = online_ratio(row, "hallucinated")
        gate = math.exp(beta * (r - tau))
        delta = gate_delta(r, safe_float(row.get("_score")), strength, beta, tau)
        values = [
            f"L{row['layer']}H{row['head']}",
            f"{safe_float(row.get('_score')):.3f}",
            f"{r:.3f}",
            f"{gate:.2f}",
            f"{delta:.3f}",
            "T *= (1-delta), renorm",
        ]
        fills = [DARK, RED, ORANGE, GREEN, PURPLE, MUTED]
        for x, value, fill in zip(columns, values, fills):
            body.append(text(x, y, value, 11, fill, "middle"))
        for x1, x2 in zip(columns[1:-1], columns[2:]):
            body.append(line(x1 + 60, y - 4, x2 - 60, y - 4, "#cbd5e1"))
    svg(path, width, height, "".join(body))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ranked-heads",
        default="../ADHH/LLaVA/results/coco/llava-v1.5-7b_base_original_qa_n3000/surrogate_hh_scores/surrogate_score_zoo/ranked_heads_global__itext_all__C_toi_HminusG.json",
    )
    parser.add_argument("--output-dir", default="./results/coco/method_phase_figures")
    parser.add_argument("--top-k", type=int, default=150)
    parser.add_argument("--top-ks", default="20,50,100,150,200")
    parser.add_argument("--layer-min", type=int, default=0)
    parser.add_argument("--layer-max", type=int, default=31)
    parser.add_argument("--head-count", type=int, default=32)
    parser.add_argument("--highlight-layer-start", type=int, default=9)
    parser.add_argument("--highlight-layer-end", type=int, default=16)
    parser.add_argument("--gate-strength", type=float, default=0.7)
    parser.add_argument("--gate-beta", type=float, default=10.0)
    parser.add_argument("--gate-tau", type=float, default=0.9)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    data, records, score_name = load_ranked_heads(args.ranked_heads)
    add_component_ranks(records)
    top_ks = [int(item) for item in args.top_ks.split(",") if item.strip()]

    bucket_rows = summarize_buckets(records, top_ks)
    write_csv(os.path.join(args.output_dir, "phase1_bucket_summary.csv"), bucket_rows)

    figure_paths = {
        "phase1_text_mass_bars": os.path.join(args.output_dir, "phase1_text_mass_bars.svg"),
        "phase1_contrastive_bias_distribution": os.path.join(args.output_dir, "phase1_contrastive_bias_distribution.svg"),
        "phase1_head_score_heatmap": os.path.join(args.output_dir, "phase1_head_score_heatmap.svg"),
        "phase1_rank_fusion_flow": os.path.join(args.output_dir, "phase1_rank_fusion_flow.svg"),
        "phase2_gate_curve": os.path.join(args.output_dir, "phase2_gate_curve.svg"),
        "phase2_attention_redistribution": os.path.join(args.output_dir, "phase2_attention_redistribution.svg"),
        "phase2_delta_flow": os.path.join(args.output_dir, "phase2_delta_flow.svg"),
    }

    make_text_mass_bars(figure_paths["phase1_text_mass_bars"], bucket_rows)
    make_contrast_distribution(figure_paths["phase1_contrastive_bias_distribution"], records, args.top_k)
    make_head_score_heatmap(
        figure_paths["phase1_head_score_heatmap"],
        records,
        args.top_k,
        args.layer_min,
        args.layer_max,
        args.head_count,
        args.highlight_layer_start,
        args.highlight_layer_end,
    )
    make_rank_fusion_flow(figure_paths["phase1_rank_fusion_flow"], records, min(18, args.top_k))
    make_gate_curve(
        figure_paths["phase2_gate_curve"],
        records,
        args.top_k,
        args.gate_strength,
        args.gate_beta,
        args.gate_tau,
    )
    hall_redist, grounded_redist = make_redistribution_bar(
        figure_paths["phase2_attention_redistribution"],
        records,
        args.top_k,
        args.gate_strength,
        args.gate_beta,
        args.gate_tau,
    )
    make_delta_flow(
        figure_paths["phase2_delta_flow"],
        records,
        min(8, args.top_k),
        args.gate_strength,
        args.gate_beta,
        args.gate_tau,
    )

    phase2_rows = [{
        "top_k": args.top_k,
        "gate_strength": args.gate_strength,
        "gate_beta": args.gate_beta,
        "gate_tau": args.gate_tau,
        "hall_mean_ratio": mean(x["ratio"] for x in hall_redist),
        "grounded_mean_ratio": mean(x["ratio"] for x in grounded_redist),
        "hall_mean_delta": mean(x["delta"] for x in hall_redist),
        "grounded_mean_delta": mean(x["delta"] for x in grounded_redist),
        "hall_text_before": mean(x["text_before"] for x in hall_redist),
        "hall_text_after": mean(x["text_after"] for x in hall_redist),
        "grounded_text_before": mean(x["text_before"] for x in grounded_redist),
        "grounded_text_after": mean(x["text_after"] for x in grounded_redist),
    }]
    write_csv(os.path.join(args.output_dir, "phase2_gate_redistribution_summary.csv"), phase2_rows)

    summary = {
        "ranked_heads": args.ranked_heads,
        "score_name": score_name,
        "source_layer_range": data.get("layer_range"),
        "n_heads": len(records),
        "top_k": args.top_k,
        "highlight_layer_window": [args.highlight_layer_start, args.highlight_layer_end],
        "note": (
            "If source_layer_range excludes L9-L12, those heatmap cells are intentionally shown as unscored. "
            "Use a ranked-head file generated over L9-L16 or L9-L31 for a complete L9-L16 heatmap."
        ),
        "figures": figure_paths,
        "tables": {
            "phase1_bucket_summary": os.path.join(args.output_dir, "phase1_bucket_summary.csv"),
            "phase2_gate_redistribution_summary": os.path.join(args.output_dir, "phase2_gate_redistribution_summary.csv"),
        },
    }
    write_json(os.path.join(args.output_dir, "method_phase_figures_summary.json"), summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

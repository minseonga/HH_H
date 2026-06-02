#!/usr/bin/env python3
import argparse
import collections
import csv
import json
import os


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def exposure_object_fractions(path):
    rows = read_csv(path)
    by = {row.get("bucket"): row for row in rows}
    hall = fnum(by.get("hallucinated_object", {}).get("token_steps"))
    ground = fnum(by.get("grounded_object", {}).get("token_steps"))
    total = hall + ground
    return {
        "hall_steps": hall,
        "ground_steps": ground,
        "hall_fraction": hall / total if total else 0.0,
        "ground_fraction": ground / total if total else 0.0,
    }


def percentile(values, q):
    if not values:
        return 0.0
    values = sorted(values)
    idx = int(round((len(values) - 1) * q / 100.0))
    return float(values[idx])


def deact_stats(path):
    rows = read_csv(path)
    by_label = collections.defaultdict(list)
    by_step = collections.defaultdict(lambda: collections.defaultdict(list))
    for row in rows:
        label = row.get("label")
        if label not in {"grounded", "hallucinated"}:
            continue
        delta = fnum(row.get("delta"))
        by_label[label].append(delta)
        key = (row.get("question_id"), row.get("step_idx"), row.get("token_text"))
        by_step[label][key].append(delta)

    out = {}
    for label in ["grounded", "hallucinated"]:
        vals = by_label[label]
        step_vals = [sum(items) / len(items) for items in by_step[label].values() if items]
        out[label] = {
            "n_head_steps": len(vals),
            "n_object_steps": len(step_vals),
            "mean_delta_head_step": sum(vals) / len(vals) if vals else 0.0,
            "median_delta_head_step": percentile(vals, 50),
            "frac_head_step_delta_ge_0p5": sum(v >= 0.5 for v in vals) / len(vals) if vals else 0.0,
            "frac_head_step_delta_ge_0p8": sum(v >= 0.8 for v in vals) / len(vals) if vals else 0.0,
            "sum_delta_head_step": sum(vals),
            "mean_delta_object_step": sum(step_vals) / len(step_vals) if step_vals else 0.0,
            "median_delta_object_step": percentile(step_vals, 50),
            "q90_delta_object_step": percentile(step_vals, 90),
            "frac_object_step_mean_delta_ge_0p45": (
                sum(v >= 0.45 for v in step_vals) / len(step_vals) if step_vals else 0.0
            ),
            "frac_object_step_mean_delta_ge_0p5": (
                sum(v >= 0.5 for v in step_vals) / len(step_vals) if step_vals else 0.0
            ),
        }
    total_mass = out["grounded"]["sum_delta_head_step"] + out["hallucinated"]["sum_delta_head_step"]
    out["suppression_mass_share"] = {
        "grounded": out["grounded"]["sum_delta_head_step"] / total_mass if total_mass else 0.0,
        "hallucinated": out["hallucinated"]["sum_delta_head_step"] / total_mass if total_mass else 0.0,
    }
    return out


def text(x, y, value, size=13, fill="#1f2937", anchor="start", weight="400"):
    return (
        f'<text x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" fill="{fill}" text-anchor="{anchor}" font-weight="{weight}">{value}</text>'
    )


def rect(x, y, w, h, fill, stroke="none", opacity=1, radius=0):
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" ry="{radius}" '
        f'fill="{fill}" stroke="{stroke}" opacity="{opacity}"/>'
    )


def line(x1, y1, x2, y2, stroke="#d0d7de", width=1):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{width}"/>'


def draw_grouped_bars(parts, x0, y0, w, h, title, ylabel, groups, ymax, value_fmt):
    colors = {"ground": "#2e8b57", "hall": "#d95f02", "grid": "#d0d7de", "text": "#1f2937", "muted": "#667085"}
    parts.append(text(x0, y0 - 18, title, 14, colors["text"], "start", "800"))
    chart_x, chart_y = x0 + 42, y0
    chart_w, chart_h = w - 58, h - 48
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        yy = chart_y + chart_h - chart_h * tick
        parts.append(line(chart_x, yy, chart_x + chart_w, yy, colors["grid"], 1))
        parts.append(text(chart_x - 8, yy + 4, value_fmt(ymax * tick), 9.5, colors["muted"], "end"))
    parts.append(line(chart_x, chart_y, chart_x, chart_y + chart_h, "#98a2b3", 1.2))
    parts.append(line(chart_x, chart_y + chart_h, chart_x + chart_w, chart_y + chart_h, "#98a2b3", 1.2))
    group_w = chart_w / len(groups)
    bar_w = min(36, group_w / 4)
    for gi, (name, ground, hall) in enumerate(groups):
        cx = chart_x + group_w * gi + group_w / 2
        for offset, value, color, label in [(-bar_w / 1.7, ground, colors["ground"], "ground"), (bar_w / 1.7, hall, colors["hall"], "hall")]:
            bh = chart_h * value / ymax if ymax else 0
            bx = cx + offset - bar_w / 2
            parts.append(rect(bx, chart_y + chart_h - bh, bar_w, bh, color, opacity=0.86, radius=3))
            parts.append(text(bx + bar_w / 2, chart_y + chart_h - bh - 7, value_fmt(value), 9.5, colors["text"], "middle", "800"))
        parts.append(text(cx, chart_y + chart_h + 22, name, 10.5, colors["muted"], "middle", "700"))
    parts.append(text(chart_x - 36, chart_y + chart_h / 2, ylabel, 10.5, colors["muted"], "middle", "700"))
    parts.append(rect(chart_x + chart_w - 98, chart_y + 4, 10, 10, colors["ground"], radius=2))
    parts.append(text(chart_x + chart_w - 82, chart_y + 13, "ground", 10, colors["muted"]))
    parts.append(rect(chart_x + chart_w - 98, chart_y + 20, 10, 10, colors["hall"], radius=2))
    parts.append(text(chart_x + chart_w - 82, chart_y + 29, "hall", 10, colors["muted"]))


def build_svg(path, exposure, deact):
    colors = {
        "ground": "#2e8b57",
        "hall": "#d95f02",
        "muted": "#667085",
        "text": "#1f2937",
        "dark": "#111827",
        "panel": "#f8fafc",
    }
    width, height = 1180, 500
    static_hall = exposure["hall_fraction"] * 100
    static_ground = exposure["ground_fraction"] * 100
    deact_hall_mass = deact["suppression_mass_share"]["hallucinated"] * 100
    deact_ground_mass = deact["suppression_mass_share"]["grounded"] * 100

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        rect(0, 0, width, height, "white"),
        text(590, 34, "Does DEACT reduce AD-HH static over-touch?", 20, colors["dark"], "middle", "800"),
        text(590, 58, "DEACT does not change which object steps exist; it changes the intervention strength from hard δ=1 to a continuous gate.", 12, colors["muted"], "middle"),
    ]

    # Panel A: exposure/mass share.
    x0, y0, pw, ph = 45, 92, 330, 320
    parts.append(rect(x0, y0, pw, ph, colors["panel"], "#e5e7eb", radius=8))
    parts.append(text(x0 + 18, y0 + 30, "A. Static exposure vs DEACT mass", 14, colors["dark"], "start", "800"))
    bar_x, bar_y, bar_w, bar_h = x0 + 34, y0 + 83, pw - 68, 36
    for idx, (label, ground, hall) in enumerate([
        ("static head-step exposure", static_ground, static_hall),
        ("DEACT suppression mass", deact_ground_mass, deact_hall_mass),
    ]):
        yy = bar_y + idx * 92
        gw = bar_w * ground / 100
        hw = bar_w - gw
        parts.append(text(bar_x, yy - 12, label, 11, colors["muted"], "start", "700"))
        parts.append(rect(bar_x, yy, gw, bar_h, colors["ground"], opacity=0.86, radius=4))
        parts.append(rect(bar_x + gw, yy, hw, bar_h, colors["hall"], opacity=0.9, radius=4))
        parts.append(text(bar_x + gw / 2, yy + 23, f"ground {ground:.1f}%", 10.5, "white", "middle", "800"))
        parts.append(text(bar_x + gw + hw / 2, yy + 23, f"hall {hall:.1f}%", 10.5, "white", "middle", "800"))
    parts.append(text(bar_x, y0 + 266, "Mass share shifts only slightly because grounded object steps are much more frequent.", 10.5, colors["muted"]))
    parts.append(text(bar_x, y0 + 288, "This does not by itself prove selectivity; inspect strength per step.", 10.5, colors["muted"]))

    # Panel B: mean delta / strong delta.
    x1, y1, pw2, ph2 = 420, 92, 345, 320
    parts.append(rect(x1, y1, pw2, ph2, colors["panel"], "#e5e7eb", radius=8))
    groups = [
        (
            "mean δ",
            deact["grounded"]["mean_delta_object_step"],
            deact["hallucinated"]["mean_delta_object_step"],
        ),
        (
            "median δ",
            deact["grounded"]["median_delta_object_step"],
            deact["hallucinated"]["median_delta_object_step"],
        ),
    ]
    draw_grouped_bars(parts, x1 + 18, y1 + 75, pw2 - 32, 230, "B. DEACT strength per object step", "δ", groups, 0.7, lambda v: f"{v:.2f}")
    parts.append(text(x1 + 24, y1 + 292, f"hall mean δ / ground mean δ = {deact['hallucinated']['mean_delta_object_step'] / max(deact['grounded']['mean_delta_object_step'], 1e-9):.2f}×", 11, colors["muted"]))

    # Panel C: thresholded strong object steps.
    x2, y2, pw3, ph3 = 810, 92, 345, 320
    parts.append(rect(x2, y2, pw3, ph3, colors["panel"], "#e5e7eb", radius=8))
    groups = [
        (
            "mean δ≥0.45",
            deact["grounded"]["frac_object_step_mean_delta_ge_0p45"] * 100,
            deact["hallucinated"]["frac_object_step_mean_delta_ge_0p45"] * 100,
        ),
        (
            "mean δ≥0.50",
            deact["grounded"]["frac_object_step_mean_delta_ge_0p5"] * 100,
            deact["hallucinated"]["frac_object_step_mean_delta_ge_0p5"] * 100,
        ),
    ]
    draw_grouped_bars(parts, x2 + 18, y2 + 75, pw3 - 32, 230, "C. Strong DEACT object steps", "%", groups, 65, lambda v: f"{v:.1f}")
    parts.append(text(x2 + 24, y2 + 292, "DEACT gives hallucinated objects a higher strong-suppression rate.", 11, colors["muted"]))

    parts.append(text(590, 455, "Takeaway: DEACT partially mitigates static over-suppression by reducing ground-step strength and slightly increasing hall-step strength, but it does not eliminate grounded collateral exposure.", 12, colors["dark"], "middle", "800"))
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-exposure-csv", required=True)
    parser.add_argument("--deact-ratio-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    exposure = exposure_object_fractions(args.static_exposure_csv)
    deact = deact_stats(args.deact_ratio_csv)
    summary = {
        "sources": {
            "static_exposure_csv": args.static_exposure_csv,
            "deact_ratio_csv": args.deact_ratio_csv,
        },
        "static_object_exposure": exposure,
        "deact": deact,
        "interpretation": (
            "DEACT softens the intervention relative to static AD-HH: mean object-step delta is below 0.5 for both labels, "
            "and hallucinated objects receive slightly stronger average and thresholded suppression. However, because grounded "
            "object steps are much more frequent, total suppression mass is still dominated by grounded steps."
        ),
    }
    svg_path = os.path.join(args.output_dir, "deact_vs_static_hall_ground_touch.svg")
    build_svg(svg_path, exposure, deact)
    summary["figure_path"] = svg_path
    with open(os.path.join(args.output_dir, "deact_vs_static_hall_ground_touch_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

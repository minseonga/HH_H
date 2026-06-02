#!/usr/bin/env python3
import argparse
import csv
import json
import os


def read_csv(path):
    if not path or not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def exposure_from_csv(path):
    rows = read_csv(path)
    by_bucket = {row.get("bucket"): row for row in rows}
    hall = as_float(by_bucket.get("hallucinated_object", {}).get("token_steps"))
    ground = as_float(by_bucket.get("grounded_object", {}).get("token_steps"))
    total = hall + ground
    return {
        "hall_steps": hall,
        "ground_steps": ground,
        "hall_object_fraction": hall / total if total else 0.0,
        "ground_object_fraction": ground / total if total else 0.0,
    }


def logprob_from_csv(path):
    rows = read_csv(path)
    out = {}
    for row in rows:
        label = row.get("label")
        if label in {"grounded_object", "hallucinated_object"}:
            out[label] = {
                "positive_drop_fraction": as_float(row.get("positive_drop_fraction")),
                "top1_loss_fraction": 1.0 - as_float(row.get("static_target_top1_fraction")),
                "mean_delta_logprob": as_float(row.get("mean_delta_logprob")),
                "median_delta_logprob": as_float(row.get("median_delta_logprob")),
            }
    return out


def write_summary(path, exposure, logprob, sources):
    summary = {
        "sources": sources,
        "object_step_exposure": exposure,
        "logprob_perturbation": logprob,
        "interpretation": (
            "Object-only static exposure is dominated by grounded object steps because hard/static "
            "AD-HH-style suppression is keyed by head identity, not by whether the current object is hallucinated. "
            "The log-probability diagnostic then shows that grounded objects are measurably perturbed as well."
        ),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def svg_text(x, y, text, size=13, fill="#1f2937", anchor="start", weight="400"):
    return (
        f'<text x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" fill="{fill}" text-anchor="{anchor}" font-weight="{weight}">{text}</text>'
    )


def svg_line(x1, y1, x2, y2, stroke="#d0d7de", width=1, dash=""):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{width}"{dash_attr}/>'


def svg_rect(x, y, w, h, fill, stroke="none", width=1, opacity=1.0, radius=0):
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" ry="{radius}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{width}" opacity="{opacity}"/>'
    )


def plot(output_dir, exposure, logprob, formats):
    colors = {
        "hall": "#d95f02",
        "ground": "#2e8b57",
        "muted": "#667085",
        "grid": "#d0d7de",
        "text": "#1f2937",
        "dark": "#111827",
        "panel": "#f8fafc",
    }

    hall_exp = exposure["hall_object_fraction"] * 100.0
    ground_exp = exposure["ground_object_fraction"] * 100.0
    ground_log = logprob.get("grounded_object", {})
    hall_log = logprob.get("hallucinated_object", {})
    pos_drop = [
        ground_log.get("positive_drop_fraction", 0.0) * 100.0,
        hall_log.get("positive_drop_fraction", 0.0) * 100.0,
    ]
    top1_loss = [
        ground_log.get("top1_loss_fraction", 0.0) * 100.0,
        hall_log.get("top1_loss_fraction", 0.0) * 100.0,
    ]
    mean_drop = [
        ground_log.get("mean_delta_logprob", 0.0),
        hall_log.get("mean_delta_logprob", 0.0),
    ]

    width, height = 1200, 430
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        svg_rect(0, 0, width, height, "white"),
        svg_text(600, 34, "AD-HH-style static suppression touches grounded objects more often than hallucinated objects", 18, colors["dark"], "middle", "800"),
        svg_text(600, 58, "Object-only view: static is keyed by head identity, not by whether the current object is hallucinated.", 12, colors["muted"], "middle", "400"),
    ]

    panel_y, panel_h = 86, 285
    panel_w = 340
    panel_xs = [45, 430, 815]
    titles = ["A. Static exposure", "B. Positive log-prob drop", "C. Top-1 loss and mean drop"]
    for x0, title in zip(panel_xs, titles):
        parts.append(svg_rect(x0, panel_y, panel_w, panel_h, colors["panel"], "#e5e7eb", 1, 1.0, 8))
        parts.append(svg_text(x0 + 18, panel_y + 30, title, 14, colors["dark"], "start", "800"))

    # Panel A: object-step exposure.
    x0 = panel_xs[0]
    bar_x, bar_y, bar_w, bar_h = x0 + 28, panel_y + 120, panel_w - 56, 42
    ground_w = bar_w * ground_exp / 100.0
    hall_w = bar_w - ground_w
    parts.append(svg_rect(bar_x, bar_y, bar_w, bar_h, "#eef2f7", "none", radius=5))
    parts.append(svg_rect(bar_x, bar_y, ground_w, bar_h, colors["ground"], "none", opacity=0.88, radius=5))
    parts.append(svg_rect(bar_x + ground_w, bar_y, hall_w, bar_h, colors["hall"], "none", opacity=0.9, radius=5))
    parts.append(svg_text(bar_x + ground_w / 2, bar_y + 17, "grounded", 12, "white", "middle", "800"))
    parts.append(svg_text(bar_x + ground_w / 2, bar_y + 34, f"{ground_exp:.1f}%", 12, "white", "middle", "800"))
    parts.append(svg_text(bar_x + ground_w + hall_w / 2, bar_y + 17, "hall", 12, "white", "middle", "800"))
    parts.append(svg_text(bar_x + ground_w + hall_w / 2, bar_y + 34, f"{hall_exp:.1f}%", 12, "white", "middle", "800"))
    parts.append(svg_text(bar_x, bar_y + 76, f"steps: grounded {int(exposure['ground_steps'])}, hallucinated {int(exposure['hall_steps'])}", 11, colors["muted"]))
    parts.append(svg_text(bar_x, bar_y + 98, "Interpretation: most object-step interventions are collateral opportunities.", 11, colors["muted"]))

    # Common mini bar chart helper.
    def mini_bar_panel(x0, values, ymax, ylabel, value_fmt, subtitle=None):
        chart_x, chart_y = x0 + 62, panel_y + 74
        chart_w, chart_h = panel_w - 105, 150
        for tick in [0, 0.25, 0.5, 0.75, 1.0]:
            y = chart_y + chart_h - chart_h * tick
            parts.append(svg_line(chart_x, y, chart_x + chart_w, y, colors["grid"], 1))
            parts.append(svg_text(chart_x - 8, y + 4, f"{int(ymax * tick)}", 10, colors["muted"], "end"))
        parts.append(svg_line(chart_x, chart_y, chart_x, chart_y + chart_h, "#98a2b3", 1.2))
        parts.append(svg_line(chart_x, chart_y + chart_h, chart_x + chart_w, chart_y + chart_h, "#98a2b3", 1.2))
        bar_w2 = 58
        xs = [chart_x + 48, chart_x + 145]
        for idx, value in enumerate(values):
            h = chart_h * value / ymax if ymax else 0
            color = colors["ground"] if idx == 0 else colors["hall"]
            parts.append(svg_rect(xs[idx], chart_y + chart_h - h, bar_w2, h, color, "none", opacity=0.86, radius=4))
            parts.append(svg_text(xs[idx] + bar_w2 / 2, chart_y + chart_h - h - 8, value_fmt(value), 12, colors["text"], "middle", "800"))
            parts.append(svg_text(xs[idx] + bar_w2 / 2, chart_y + chart_h + 24, "ground" if idx == 0 else "hall", 11, colors["muted"], "middle"))
        parts.append(svg_text(chart_x - 42, chart_y + chart_h / 2, ylabel, 11, colors["muted"], "middle", "700"))
        if subtitle:
            parts.append(svg_text(x0 + 20, panel_y + 256, subtitle, 11, colors["muted"]))

    mini_bar_panel(
        panel_xs[1],
        pos_drop,
        100,
        "percent",
        lambda value: f"{value:.1f}%",
        "Grounded objects are affected too, but hallucinated objects are more fragile.",
    )

    # Panel C: two-scale compact comparison.
    x0 = panel_xs[2]
    chart_x, chart_y = x0 + 55, panel_y + 76
    chart_w, chart_h = panel_w - 105, 145
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        y = chart_y + chart_h - chart_h * tick
        parts.append(svg_line(chart_x, y, chart_x + chart_w, y, colors["grid"], 1))
        parts.append(svg_text(chart_x - 8, y + 4, f"{int(40 * tick)}", 10, colors["muted"], "end"))
    parts.append(svg_text(chart_x - 36, chart_y + chart_h / 2, "top-1 lost %", 11, colors["muted"], "middle", "700"))
    bar_w3 = 46
    xs = [chart_x + 45, chart_x + 140]
    for idx, value in enumerate(top1_loss):
        h = chart_h * value / 40.0
        color = colors["ground"] if idx == 0 else colors["hall"]
        parts.append(svg_rect(xs[idx], chart_y + chart_h - h, bar_w3, h, color, "none", opacity=0.84, radius=4))
        parts.append(svg_text(xs[idx] + bar_w3 / 2, chart_y + chart_h - h - 8, f"{value:.1f}%", 11, colors["text"], "middle", "800"))
        parts.append(svg_text(xs[idx] + bar_w3 / 2, chart_y + chart_h + 24, "ground" if idx == 0 else "hall", 11, colors["muted"], "middle"))
    # Mean drop line uses a 0.45 max scale.
    line_points = []
    for idx, value in enumerate(mean_drop):
        px = xs[idx] + bar_w3 + 18
        py = chart_y + chart_h - chart_h * min(value / 0.45, 1.0)
        line_points.append((px, py))
        parts.append(f'<circle cx="{px}" cy="{py}" r="5" fill="#344054"/>')
        parts.append(svg_text(px, py - 12, f"{value:.3f}", 11, "#344054", "middle", "800"))
    parts.append(svg_line(line_points[0][0], line_points[0][1], line_points[1][0], line_points[1][1], "#344054", 2))
    parts.append(svg_text(x0 + panel_w - 20, chart_y + 12, "dot: mean drop", 11, "#344054", "end", "700"))
    parts.append(svg_text(x0 + 20, panel_y + 256, "Static can demote hallucinated objects, but it also demotes grounded ones.", 11, colors["muted"]))

    parts.append(svg_text(600, 405, "Takeaway: AD-HH hard/static suppression is effective but coarse: hallucinated objects are the desired target, grounded objects are frequent collateral targets.", 12, colors["dark"], "middle", "700"))
    parts.append("</svg>")
    svg = "\n".join(parts)
    paths = {}
    for ext in formats:
        if ext != "svg":
            continue
        path = os.path.join(output_dir, "adhh_static_hall_ground_touch.svg")
        with open(path, "w", encoding="utf-8") as f:
            f.write(svg)
        paths[ext] = path
    if "svg" not in paths:
        path = os.path.join(output_dir, "adhh_static_hall_ground_touch.svg")
        with open(path, "w", encoding="utf-8") as f:
            f.write(svg)
        paths["svg"] = path
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exposure-summary-csv", required=True)
    parser.add_argument("--logprob-summary-csv", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--formats", default="png,svg,pdf")
    parser.add_argument("--grounded-positive-drop-fraction", type=float, default=0.625)
    parser.add_argument("--hall-positive-drop-fraction", type=float, default=0.805)
    parser.add_argument("--grounded-static-target-top1-fraction", type=float, default=0.865)
    parser.add_argument("--hall-static-target-top1-fraction", type=float, default=0.655)
    parser.add_argument("--grounded-mean-delta-logprob", type=float, default=0.08092865523305591)
    parser.add_argument("--hall-mean-delta-logprob", type=float, default=0.3864174883562373)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    exposure = exposure_from_csv(args.exposure_summary_csv)
    logprob = logprob_from_csv(args.logprob_summary_csv)
    logprob.setdefault(
        "grounded_object",
        {
            "positive_drop_fraction": args.grounded_positive_drop_fraction,
            "top1_loss_fraction": 1.0 - args.grounded_static_target_top1_fraction,
            "mean_delta_logprob": args.grounded_mean_delta_logprob,
        },
    )
    logprob.setdefault(
        "hallucinated_object",
        {
            "positive_drop_fraction": args.hall_positive_drop_fraction,
            "top1_loss_fraction": 1.0 - args.hall_static_target_top1_fraction,
            "mean_delta_logprob": args.hall_mean_delta_logprob,
        },
    )
    formats = [item.strip() for item in args.formats.split(",") if item.strip()]
    figure_paths = plot(args.output_dir, exposure, logprob, formats)
    summary = write_summary(
        os.path.join(args.output_dir, "adhh_static_hall_ground_touch_summary.json"),
        exposure,
        logprob,
        {
            "exposure_summary_csv": args.exposure_summary_csv,
            "logprob_summary_csv": args.logprob_summary_csv,
        },
    )
    summary["figure_paths"] = figure_paths
    with open(os.path.join(args.output_dir, "adhh_static_hall_ground_touch_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

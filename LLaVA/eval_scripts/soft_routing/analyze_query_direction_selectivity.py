import argparse
import csv
import html
import json
import math
import os
import re


BLUE = "#2563eb"
ORANGE = "#f97316"
GREEN = "#059669"
RED = "#dc2626"
DARK = "#0f172a"
MUTED = "#64748b"
GRID = "#e2e8f0"


def safe_float(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    return value


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


def load_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def parse_csv_list(text_value):
    out = []
    for item in str(text_value or "").replace(" ", ",").split(","):
        item = item.strip()
        if item:
            out.append(item)
    return out


def infer_features(rows):
    if not rows:
        return []
    pattern = re.compile(
        r"^(top\d+(?:_offset_[mp]\d+)?_(?:score_mean|score_max|margin_mean|margin_max|hard_rate|sigmoid_mean)|"
        r"\d+:\d+_(?:score|margin)(?:_offset_[mp]\d+)?)$"
    )
    fields = rows[0].keys()
    features = []
    for key in fields:
        if not pattern.match(key):
            continue
        values = [safe_float(row.get(key)) for row in rows]
        values = [value for value in values if value is not None]
        if len(values) >= 2 and max(values) > min(values):
            features.append(key)
    priority = {
        "sigmoid_mean": 0,
        "margin_mean": 1,
        "margin_max": 2,
        "hard_rate": 3,
        "score_mean": 4,
        "score_max": 5,
    }

    def sort_key(feature):
        m = re.match(r"top(\d+)", feature)
        top_k = int(m.group(1)) if m else 9999
        suffix_rank = min((rank for suffix, rank in priority.items() if suffix in feature), default=99)
        return top_k, suffix_rank, feature

    return sorted(features, key=sort_key)


def thresholds_for(values):
    unique = sorted(set(values), reverse=True)
    if not unique:
        return []
    thresholds = [unique[0] + 1e-12]
    for idx in range(len(unique) - 1):
        thresholds.append(0.5 * (unique[idx] + unique[idx + 1]))
    thresholds.append(unique[-1] - 1e-12)
    return thresholds


def confusion_at(values, labels, threshold):
    tp = fp = tn = fn = 0
    for value, label in zip(values, labels):
        pred = value >= threshold
        if label == 1 and pred:
            tp += 1
        elif label == 1:
            fn += 1
        elif pred:
            fp += 1
        else:
            tn += 1
    n_pos = tp + fn
    n_neg = tn + fp
    flagged = tp + fp
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "flagged": flagged,
        "hall_recall": tp / max(n_pos, 1),
        "ground_fpr": fp / max(n_neg, 1),
        "ground_specificity": tn / max(n_neg, 1),
        "precision": tp / max(flagged, 1),
        "flagged_rate": flagged / max(len(labels), 1),
    }


def feature_curves(rows, feature):
    values = []
    labels = []
    for row in rows:
        value = safe_float(row.get(feature))
        label = safe_float(row.get("label"))
        if value is None or label is None:
            continue
        values.append(value)
        labels.append(1 if int(label) == 1 else 0)
    if not values:
        return []
    curve = []
    for threshold in thresholds_for(values):
        item = confusion_at(values, labels, threshold)
        item["feature"] = feature
        curve.append(item)
    return curve


def best_under_fpr(curve, fpr_budget):
    candidates = [row for row in curve if row["ground_fpr"] <= fpr_budget]
    if not candidates:
        return None
    return max(candidates, key=lambda row: (row["hall_recall"], row["precision"], -row["flagged_rate"]))


def best_for_recall(curve, recall_target):
    candidates = [row for row in curve if row["hall_recall"] >= recall_target]
    if not candidates:
        return None
    return min(candidates, key=lambda row: (row["ground_fpr"], -row["precision"], row["flagged_rate"]))


def top_budget_point(rows, feature, budget):
    scored = []
    for row in rows:
        value = safe_float(row.get(feature))
        label = safe_float(row.get("label"))
        if value is None or label is None:
            continue
        scored.append((value, 1 if int(label) == 1 else 0))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    n = max(1, int(round(len(scored) * budget)))
    threshold = scored[min(n - 1, len(scored) - 1)][0]
    values = [value for value, _ in scored]
    labels = [label for _, label in scored]
    return confusion_at(values, labels, threshold)


def summarize_features(rows, features, fpr_budgets, recall_targets, budget_rates):
    curve_rows = []
    operating_rows = []
    for feature in features:
        curve = feature_curves(rows, feature)
        curve_rows.extend(curve)
        if not curve:
            continue
        labels = [row["label"] for row in rows if safe_float(row.get(feature)) is not None]
        base_rate = sum(int(float(label)) for label in labels) / max(len(labels), 1)
        best_selective = max(
            curve,
            key=lambda row: (
                row["hall_recall"] - row["ground_fpr"],
                row["precision"],
                -row["flagged_rate"],
            ),
        )
        best_selective = dict(best_selective)
        best_selective.update({
            "mode": "max_recall_minus_ground_fpr",
            "base_hall_rate": base_rate,
            "precision_lift": best_selective["precision"] / max(base_rate, 1e-12),
        })
        operating_rows.append(best_selective)
        for budget in fpr_budgets:
            item = best_under_fpr(curve, budget)
            if item is None:
                continue
            item = dict(item)
            item.update({
                "mode": f"max_hall_recall_at_ground_fpr_le_{budget:g}",
                "base_hall_rate": base_rate,
                "precision_lift": item["precision"] / max(base_rate, 1e-12),
            })
            operating_rows.append(item)
        for target in recall_targets:
            item = best_for_recall(curve, target)
            if item is None:
                continue
            item = dict(item)
            item.update({
                "mode": f"min_ground_fpr_at_hall_recall_ge_{target:g}",
                "base_hall_rate": base_rate,
                "precision_lift": item["precision"] / max(base_rate, 1e-12),
            })
            operating_rows.append(item)
        for budget in budget_rates:
            item = top_budget_point(rows, feature, budget)
            if item is None:
                continue
            item = dict(item)
            item.update({
                "feature": feature,
                "mode": f"top_{budget:g}_flagged_budget",
                "base_hall_rate": base_rate,
                "precision_lift": item["precision"] / max(base_rate, 1e-12),
            })
            operating_rows.append(item)
    operating_rows.sort(
        key=lambda row: (
            row["mode"],
            -(row["hall_recall"] - row["ground_fpr"]),
            -row["hall_recall"],
            row["ground_fpr"],
        )
    )
    return curve_rows, operating_rows


def choose_plot_features(operating_rows, max_features):
    best_by_feature = {}
    for row in operating_rows:
        if row["mode"] != "max_recall_minus_ground_fpr":
            continue
        feature = row["feature"]
        current = best_by_feature.get(feature)
        if current is None or (row["hall_recall"] - row["ground_fpr"]) > (current["hall_recall"] - current["ground_fpr"]):
            best_by_feature[feature] = row
    ranked = sorted(
        best_by_feature.values(),
        key=lambda row: (row["hall_recall"] - row["ground_fpr"], row["precision"]),
        reverse=True,
    )
    return [row["feature"] for row in ranked[:max_features]]


def make_recall_fpr_svg(path, curve_rows, features):
    width, height = 900, 620
    left, top = 78, 78
    plot_w, plot_h = 640, 420
    colors = [BLUE, GREEN, ORANGE, RED, "#7c3aed", "#0891b2"]
    by_feature = {}
    for row in curve_rows:
        if row["feature"] in features:
            by_feature.setdefault(row["feature"], []).append(row)

    def sx(value):
        return left + value * plot_w

    def sy(value):
        return top + (1.0 - value) * plot_h

    body = []
    body.append(text(width / 2, 32, "Q-direction selectivity: hallucination recall vs grounded false positive", 19, DARK, "middle", "700"))
    body.append(text(width / 2, 55, "Good gate features move toward the upper-left: high hall capture with low grounded capture.", 12, MUTED, "middle"))
    body.append(rect(left, top, plot_w, plot_h, "#f8fafc", "#cbd5e1"))
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        x = sx(tick)
        y = sy(tick)
        body.append(line(x, top, x, top + plot_h, GRID))
        body.append(line(left, y, left + plot_w, y, GRID))
        body.append(text(x, top + plot_h + 22, f"{tick:.2f}", 10, MUTED, "middle"))
        body.append(text(left - 12, y + 4, f"{tick:.2f}", 10, MUTED, "end"))
    body.append(line(sx(0), sy(0), sx(1), sy(1), "#94a3b8", 1.2, "5 5"))

    for idx, feature in enumerate(features):
        rows = sorted(by_feature.get(feature, []), key=lambda row: row["ground_fpr"])
        if not rows:
            continue
        pts = [(sx(row["ground_fpr"]), sy(row["hall_recall"])) for row in rows]
        color = colors[idx % len(colors)]
        body.append(polyline(pts, color, 2.6))
        for row in rows[::max(1, len(rows) // 12)]:
            body.append(circle(sx(row["ground_fpr"]), sy(row["hall_recall"]), 2.3, color, None, 1, 0.85))
    body.append(text(left + plot_w / 2, height - 42, "grounded false positive rate", 13, DARK, "middle"))
    body.append(text(26, top + plot_h / 2, "hallucinated recall", 13, DARK, "middle", rotate=-90))
    lx, ly = left + plot_w + 34, top + 14
    for idx, feature in enumerate(features):
        y = ly + idx * 38
        color = colors[idx % len(colors)]
        label = feature if len(feature) <= 28 else feature[:27] + "..."
        body.append(line(lx, y, lx + 28, y, color, 3))
        body.append(text(lx + 38, y + 4, label, 10, DARK))
    svg(path, width, height, "".join(body))


def make_budget_svg(path, operating_rows, mode, max_rows=10):
    rows = [row for row in operating_rows if row["mode"] == mode]
    rows.sort(key=lambda row: (row["hall_recall"], -row["ground_fpr"], row["precision"]), reverse=True)
    rows = rows[:max_rows]
    width, height = 980, 620
    left, top = 90, 78
    plot_w, plot_h = 720, 390
    body = []
    title = mode.replace("_", " ")
    body.append(text(width / 2, 32, f"Q-direction operating point: {title}", 18, DARK, "middle", "700"))
    if not rows:
        body.append(text(width / 2, height / 2, "No operating rows available", 14, MUTED, "middle"))
        svg(path, width, height, "".join(body))
        return
    vmax = max(max(row["hall_recall"], row["ground_fpr"]) for row in rows)
    vmax = max(min(vmax * 1.15, 1.0), 0.1)

    def sy(value):
        return top + (vmax - value) / vmax * plot_h

    body.append(rect(left, top, plot_w, plot_h, "#f8fafc", "#cbd5e1"))
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        if tick > vmax:
            continue
        y = sy(tick)
        body.append(line(left, y, left + plot_w, y, GRID))
        body.append(text(left - 12, y + 4, f"{tick:.2f}", 10, MUTED, "end"))
    group_w = plot_w / len(rows)
    bar_w = min(30, group_w * 0.28)
    for idx, row in enumerate(rows):
        cx = left + idx * group_w + group_w / 2
        for offset, key, color in [
            (-bar_w * 0.65, "hall_recall", RED),
            (bar_w * 0.65, "ground_fpr", BLUE),
        ]:
            value = row[key]
            y = sy(value)
            body.append(rect(cx + offset - bar_w / 2, y, bar_w, top + plot_h - y, color, None, 1, 0.85))
            body.append(text(cx + offset, y - 6, f"{value:.2f}", 9, DARK, "middle"))
        label = row["feature"]
        if len(label) > 24:
            label = label[:23] + "..."
        body.append(text(cx, top + plot_h + 24, label, 8, DARK, "middle", rotate=23))
    lx, ly = left + plot_w + 34, top + 24
    body.append(rect(lx, ly, 18, 18, RED, None, 1, 0.85))
    body.append(text(lx + 28, ly + 14, "hall recall", 11, DARK))
    body.append(rect(lx, ly + 30, 18, 18, BLUE, None, 1, 0.85))
    body.append(text(lx + 28, ly + 44, "ground FPR", 11, DARK))
    body.append(text(left + plot_w / 2, height - 36, "feature", 13, DARK, "middle"))
    body.append(text(28, top + plot_h / 2, "rate", 13, DARK, "middle", rotate=-90))
    svg(path, width, height, "".join(body))


def make_report(path, summary, top_rows, figures):
    lines = [
        "# Q-Direction Selectivity",
        "",
        "This report evaluates Q-direction as a gate, not only as an AUROC-ranked probe.",
        "",
        f"- mentions: {summary['n_mentions']}",
        f"- hallucinated mentions: {summary['n_hallucinated']}",
        f"- grounded mentions: {summary['n_grounded']}",
        f"- base hallucination rate: {summary['base_hall_rate']:.4f}",
        "",
        "## Recall/FPR Curve",
        "",
        f"![Recall vs FPR]({os.path.basename(figures['recall_fpr'])})",
        "",
        "## Operating Points",
        "",
    ]
    for figure in figures.get("budget_figures", []):
        lines.extend(["", f"![{os.path.basename(figure)}]({os.path.basename(figure)})"])
    lines.extend([
        "",
        "## Top Rows",
        "",
        "| mode | feature | hall recall | ground FPR | precision | precision lift |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in top_rows[:20]:
        lines.append(
            f"| {row['mode']} | `{row['feature']}` | "
            f"{row['hall_recall']:.3f} | {row['ground_fpr']:.3f} | "
            f"{row['precision']:.3f} | {row['precision_lift']:.2f} |"
        )
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mentions-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--features", default="auto")
    parser.add_argument("--fpr-budgets", default="0.05,0.10,0.20,0.30,0.50")
    parser.add_argument("--recall-targets", default="0.50,0.70,0.80,0.90")
    parser.add_argument("--budget-rates", default="0.05,0.10,0.20,0.30")
    parser.add_argument("--plot-top-n", type=int, default=6)
    args = parser.parse_args()

    rows = load_rows(args.mentions_csv)
    if args.features == "auto":
        features = infer_features(rows)
    else:
        features = parse_csv_list(args.features)
    if not features:
        raise ValueError("No numeric Q-direction features found. Check --features or mentions CSV.")
    fpr_budgets = [float(item) for item in parse_csv_list(args.fpr_budgets)]
    recall_targets = [float(item) for item in parse_csv_list(args.recall_targets)]
    budget_rates = [float(item) for item in parse_csv_list(args.budget_rates)]
    curve_rows, operating_rows = summarize_features(rows, features, fpr_budgets, recall_targets, budget_rates)

    os.makedirs(args.output_dir, exist_ok=True)
    curve_csv = os.path.join(args.output_dir, "query_direction_selectivity_curve.csv")
    operating_csv = os.path.join(args.output_dir, "query_direction_selectivity_operating_points.csv")
    write_csv(curve_csv, curve_rows)
    write_csv(operating_csv, operating_rows)

    plot_features = choose_plot_features(operating_rows, args.plot_top_n)
    recall_fpr_svg = os.path.join(args.output_dir, "query_direction_recall_vs_ground_fpr.svg")
    make_recall_fpr_svg(recall_fpr_svg, curve_rows, plot_features)

    budget_figures = []
    for budget in fpr_budgets:
        mode = f"max_hall_recall_at_ground_fpr_le_{budget:g}"
        figure = os.path.join(args.output_dir, f"operating_{mode}.svg")
        make_budget_svg(figure, operating_rows, mode)
        budget_figures.append(figure)

    labels = [int(float(row["label"])) for row in rows if row.get("label", "") != ""]
    summary = {
        "mentions_csv": args.mentions_csv,
        "n_mentions": len(labels),
        "n_hallucinated": sum(labels),
        "n_grounded": len(labels) - sum(labels),
        "base_hall_rate": sum(labels) / max(len(labels), 1),
        "n_features": len(features),
        "features": features,
        "plot_features": plot_features,
        "outputs": {
            "curve_csv": curve_csv,
            "operating_csv": operating_csv,
            "recall_fpr_svg": recall_fpr_svg,
            "budget_figures": budget_figures,
        },
    }
    write_json(os.path.join(args.output_dir, "query_direction_selectivity_summary.json"), summary)
    top_rows = sorted(
        [
            row for row in operating_rows
            if row["mode"].startswith("max_hall_recall_at_ground_fpr")
        ],
        key=lambda row: (row["mode"], -row["hall_recall"], row["ground_fpr"], -row["precision"]),
    )
    make_report(
        os.path.join(args.output_dir, "query_direction_selectivity.md"),
        summary,
        top_rows,
        {"recall_fpr": recall_fpr_svg, "budget_figures": budget_figures[:2]},
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

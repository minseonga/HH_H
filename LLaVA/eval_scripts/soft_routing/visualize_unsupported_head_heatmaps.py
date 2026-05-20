import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_FEATURES = [
    "unsupported_text_value_norm",
    "text_value_norm",
    "supported_text_value_norm",
    "text_img_value_cosine",
    "visual_value_ratio",
    "text_mass",
]


def safe_float(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    return value


def parse_csv_list(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def sanitize(value):
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value)).strip("_")


def normalize_label(value):
    value = str(value)
    if value.startswith("hallucinated"):
        return "hallucinated"
    if value in {"kept_grounded", "lost_grounded"}:
        return value
    return value


def load_rows(path, utility_threshold):
    rows = []
    if path.endswith(".csv"):
        with open(path, "r", newline="") as f:
            rows.extend(dict(row) for row in csv.DictReader(f))
    else:
        with open(path, "r") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))

    for row in rows:
        row["layer"] = int(row.get("layer", -1))
        row["head"] = int(row.get("head", -1))
        row["head_key"] = row.get("head_key") or f'{row["layer"]}:{row["head"]}'
        family = row.get("label_family") or row.get("label", "")
        row["label_family"] = normalize_label(family)
        if "suppression_utility" not in row and "causal_effect" in row:
            effect = safe_float(row.get("causal_effect"))
            row["suppression_utility"] = effect if row["label_family"] == "hallucinated" else -effect
        if "suppression_utility" in row:
            utility = safe_float(row.get("suppression_utility"))
            row["suppression_utility"] = utility
            row["should_suppress"] = utility > utility_threshold
        elif "should_suppress" in row:
            row["should_suppress"] = str(row.get("should_suppress")).lower() == "true"
        for feature in DEFAULT_FEATURES:
            if feature in row:
                row[feature] = safe_float(row.get(feature))
    return rows


def write_csv(path, rows):
    if not rows:
        with open(path, "w") as f:
            f.write("")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean_or_nan(values):
    values = [safe_float(value) for value in values]
    return float(np.mean(values)) if values else np.nan


def p50_or_nan(values):
    values = [safe_float(value) for value in values]
    return float(np.percentile(values, 50)) if values else np.nan


def rows_for_group(rows, group):
    if group == "all":
        return rows
    if group == "grounded":
        return [row for row in rows if row["label_family"] in {"kept_grounded", "lost_grounded"}]
    return [row for row in rows if row["label_family"] == group]


def infer_shape(rows, layers, heads):
    max_layer = max([row["layer"] for row in rows if row["layer"] >= 0], default=-1)
    max_head = max([row["head"] for row in rows if row["head"] >= 0], default=-1)
    n_layers = int(layers) if layers and layers > 0 else max_layer + 1
    n_heads = int(heads) if heads and heads > 0 else max_head + 1
    return n_layers, n_heads


def aggregate_matrix(rows, n_layers, n_heads, feature):
    buckets = defaultdict(list)
    for row in rows:
        layer = row["layer"]
        head = row["head"]
        if 0 <= layer < n_layers and 0 <= head < n_heads:
            buckets[(layer, head)].append(row)
    matrix = np.full((n_layers, n_heads), np.nan, dtype=float)
    counts = np.zeros((n_layers, n_heads), dtype=float)
    for (layer, head), items in buckets.items():
        counts[layer, head] = len(items)
        if feature == "count":
            matrix[layer, head] = len(items)
        elif feature == "should_suppress_rate":
            positives = [bool(item.get("should_suppress")) for item in items if "should_suppress" in item]
            if positives:
                matrix[layer, head] = float(np.mean(positives))
        else:
            values = [item.get(feature) for item in items if feature in item]
            if values:
                matrix[layer, head] = mean_or_nan(values)
    return matrix, counts


def write_matrix_csv(path, matrix):
    rows = []
    for layer in range(matrix.shape[0]):
        row = {"layer": layer}
        for head in range(matrix.shape[1]):
            value = matrix[layer, head]
            row[f"head_{head}"] = "" if np.isnan(value) else float(value)
        rows.append(row)
    write_csv(path, rows)


def finite_minmax_normalize(matrix):
    output = np.full_like(matrix, np.nan, dtype=float)
    finite = np.isfinite(matrix)
    if not bool(finite.any()):
        return output
    values = matrix[finite]
    low = float(np.min(values))
    high = float(np.max(values))
    den = high - low
    output[finite] = 0.0 if den <= 1e-12 else (matrix[finite] - low) / den
    return output


def positive_minmax_normalize(matrix):
    positive = np.where(np.isfinite(matrix), np.maximum(matrix, 0.0), np.nan)
    return finite_minmax_normalize(positive)


def write_overlap_summary(path, unsupported, contrast, overlap):
    rows = []
    unsupported_norm = finite_minmax_normalize(unsupported)
    contrast_norm = positive_minmax_normalize(contrast)
    for layer in range(overlap.shape[0]):
        for head in range(overlap.shape[1]):
            value = overlap[layer, head]
            if not np.isfinite(value):
                continue
            rows.append({
                "layer": layer,
                "head": head,
                "head_key": f"{layer}:{head}",
                "unsupported_text_value_norm": float(unsupported[layer, head])
                if np.isfinite(unsupported[layer, head]) else None,
                "should_suppress_minus_safe_unsupported_text_value_norm": float(contrast[layer, head])
                if np.isfinite(contrast[layer, head]) else None,
                "unsupported_norm01": float(unsupported_norm[layer, head])
                if np.isfinite(unsupported_norm[layer, head]) else None,
                "positive_contrast_norm01": float(contrast_norm[layer, head])
                if np.isfinite(contrast_norm[layer, head]) else None,
                "overlap_score": float(value),
            })
    rows.sort(key=lambda row: row["overlap_score"], reverse=True)
    write_csv(path, rows)


def save_heatmap(path, matrix, title, cmap="viridis", center_zero=False):
    masked = np.ma.masked_invalid(matrix)
    height = max(5.0, min(12.0, matrix.shape[0] * 0.35))
    width = max(7.0, min(16.0, matrix.shape[1] * 0.35))
    fig, ax = plt.subplots(figsize=(width, height))
    vmin = vmax = None
    if center_zero:
        finite = matrix[np.isfinite(matrix)]
        scale = float(np.max(np.abs(finite))) if finite.size else 1.0
        vmin, vmax = -scale, scale
    im = ax.imshow(masked, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.tick_params(axis="both", labelsize=7)
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def head_summary(rows, features, n_layers, n_heads, utility_threshold):
    output = []
    groups = ["all", "hallucinated", "grounded", "kept_grounded", "lost_grounded"]
    for group in groups:
        group_rows = rows_for_group(rows, group)
        by_head = defaultdict(list)
        for row in group_rows:
            if 0 <= row["layer"] < n_layers and 0 <= row["head"] < n_heads:
                by_head[(row["layer"], row["head"])].append(row)
        for layer in range(n_layers):
            for head in range(n_heads):
                items = by_head.get((layer, head), [])
                if not items:
                    continue
                row = {
                    "group": group,
                    "layer": layer,
                    "head": head,
                    "head_key": f"{layer}:{head}",
                    "n": len(items),
                    "n_steps": len({item.get("step_id", "") for item in items}),
                    "n_should_suppress": sum(1 for item in items if item.get("should_suppress")),
                    "should_suppress_rate": (
                        np.mean([bool(item.get("should_suppress")) for item in items])
                        if any("should_suppress" in item for item in items) else None
                    ),
                    "mean_suppression_utility": (
                        mean_or_nan([item.get("suppression_utility") for item in items if "suppression_utility" in item])
                        if any("suppression_utility" in item for item in items) else None
                    ),
                    "utility_threshold": utility_threshold,
                }
                for feature in features:
                    values = [item.get(feature) for item in items if feature in item]
                    row[f"mean_{feature}"] = mean_or_nan(values) if values else None
                    row[f"p50_{feature}"] = p50_or_nan(values) if values else None
                output.append(row)
    return output


def contrast_matrix(rows_a, rows_b, n_layers, n_heads, feature):
    mat_a, _ = aggregate_matrix(rows_a, n_layers, n_heads, feature)
    mat_b, _ = aggregate_matrix(rows_b, n_layers, n_heads, feature)
    return mat_a - mat_b


def build_unsupported_positive_overlap(rows, n_layers, n_heads):
    unsupported, _ = aggregate_matrix(rows, n_layers, n_heads, "unsupported_text_value_norm")
    pos_rows = [row for row in rows if row.get("should_suppress")]
    safe_rows = [row for row in rows if not row.get("should_suppress")]
    contrast = contrast_matrix(pos_rows, safe_rows, n_layers, n_heads, "unsupported_text_value_norm")
    unsupported_norm = finite_minmax_normalize(unsupported)
    contrast_norm = positive_minmax_normalize(contrast)
    overlap = unsupported_norm * contrast_norm
    return unsupported, contrast, overlap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--features", default=",".join(DEFAULT_FEATURES))
    parser.add_argument("--layers", type=int, default=0)
    parser.add_argument("--heads", type=int, default=0)
    parser.add_argument("--utility-threshold", type=float, default=0.02)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rows = load_rows(args.input_jsonl, args.utility_threshold)
    if not rows:
        raise ValueError("No rows found")

    features = parse_csv_list(args.features)
    n_layers, n_heads = infer_shape(rows, args.layers, args.heads)
    summary_rows = head_summary(rows, features, n_layers, n_heads, args.utility_threshold)
    write_csv(os.path.join(args.output_dir, "unsupported_head_summary.csv"), summary_rows)

    groups = ["all", "hallucinated", "grounded"]
    matrix_specs = []
    for group in groups:
        group_rows = rows_for_group(rows, group)
        for feature in features:
            matrix_specs.append((group, feature, aggregate_matrix(group_rows, n_layers, n_heads, feature)[0], False))
        matrix_specs.append((group, "count", aggregate_matrix(group_rows, n_layers, n_heads, "count")[0], False))
        if any("suppression_utility" in row for row in rows):
            matrix_specs.append((
                group,
                "suppression_utility",
                aggregate_matrix(group_rows, n_layers, n_heads, "suppression_utility")[0],
                True,
            ))
        if any("should_suppress" in row for row in rows):
            matrix_specs.append((
                group,
                "should_suppress_rate",
                aggregate_matrix(group_rows, n_layers, n_heads, "should_suppress_rate")[0],
                False,
            ))

    if "unsupported_text_value_norm" in features:
        hallucinated_rows = rows_for_group(rows, "hallucinated")
        grounded_rows = rows_for_group(rows, "grounded")
        matrix_specs.append((
            "contrast",
            "hallucinated_minus_grounded_unsupported_text_value_norm",
            contrast_matrix(hallucinated_rows, grounded_rows, n_layers, n_heads, "unsupported_text_value_norm"),
            True,
        ))
        if any("should_suppress" in row for row in rows):
            pos_rows = [row for row in rows if row.get("should_suppress")]
            safe_rows = [row for row in rows if not row.get("should_suppress")]
            matrix_specs.append((
                "contrast",
                "should_suppress_minus_safe_unsupported_text_value_norm",
                contrast_matrix(pos_rows, safe_rows, n_layers, n_heads, "unsupported_text_value_norm"),
                True,
            ))
            unsupported, contrast, overlap = build_unsupported_positive_overlap(rows, n_layers, n_heads)
            matrix_specs.append((
                "overlap",
                "unsupported_x_positive_utility_contrast",
                overlap,
                False,
            ))
            write_overlap_summary(
                os.path.join(args.output_dir, "unsupported_positive_overlap_summary.csv"),
                unsupported,
                contrast,
                overlap,
            )

    manifest = []
    for group, feature, matrix, center_zero in matrix_specs:
        safe_group = sanitize(group)
        safe_feature = sanitize(feature)
        matrix_csv = os.path.join(args.output_dir, f"matrix_{safe_group}_{safe_feature}.csv")
        heatmap_png = os.path.join(args.output_dir, f"heatmap_{safe_group}_{safe_feature}.png")
        write_matrix_csv(matrix_csv, matrix)
        cmap = "coolwarm" if center_zero else "viridis"
        save_heatmap(
            heatmap_png,
            matrix,
            f"{feature} ({group})",
            cmap=cmap,
            center_zero=center_zero,
        )
        manifest.append({
            "group": group,
            "feature": feature,
            "matrix_csv": matrix_csv,
            "heatmap_png": heatmap_png,
        })

    write_csv(os.path.join(args.output_dir, "manifest.csv"), manifest)
    config = {
        "input_jsonl": args.input_jsonl,
        "output_dir": args.output_dir,
        "n_rows": len(rows),
        "n_layers": n_layers,
        "n_heads": n_heads,
        "features": features,
        "utility_threshold": args.utility_threshold,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(json.dumps(config, indent=2))
    overlap_path = os.path.join(args.output_dir, "unsupported_positive_overlap_summary.csv")
    if os.path.exists(overlap_path):
        print("top heads by unsupported-positive overlap:")
        with open(overlap_path, "r", newline="") as f:
            for idx, row in enumerate(csv.DictReader(f)):
                if idx >= 20:
                    break
                print(
                    row["head_key"],
                    row["overlap_score"],
                    row["unsupported_text_value_norm"],
                    row["should_suppress_minus_safe_unsupported_text_value_norm"],
                )
    print("top heads by mean unsupported_text_value_norm:")
    for row in sorted(
        [row for row in summary_rows if row["group"] == "all"],
        key=lambda item: safe_float(item.get("mean_unsupported_text_value_norm")),
        reverse=True,
    )[:20]:
        print(row["head_key"], row.get("mean_unsupported_text_value_norm"), row.get("n"), row.get("should_suppress_rate"))


if __name__ == "__main__":
    main()

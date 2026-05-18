import argparse
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "hallucinated": "#d95f02",
    "kept_grounded": "#1b9e77",
    "lost_grounded": "#7570b3",
    "grounded": "#1b9e77",
}


def safe_float(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(value):
        return default
    return value


def load_csv(path):
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


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


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def direction_key(row):
    return f"{int(row['layer'])}:{int(row['head'])}"


def select_top_directions(auc_rows, top_k, direction_filter):
    rows = []
    for row in auc_rows:
        if direction_filter == "high" and row.get("test_direction") != "high_predicts_hallucinated":
            continue
        rows.append(row)
    rows.sort(key=lambda row: safe_float(row.get("test_auroc_abs")), reverse=True)
    return rows[:top_k] if top_k > 0 else rows


def plot_auc_heatmap(auc_rows, output_dir):
    if not auc_rows:
        return None
    max_layer = max(int(row["layer"]) for row in auc_rows)
    max_head = max(int(row["head"]) for row in auc_rows)
    matrix = np.full((max_layer + 1, max_head + 1), np.nan, dtype=float)
    for row in auc_rows:
        matrix[int(row["layer"]), int(row["head"])] = safe_float(row.get("test_auroc_high_predicts_hallucinated"), np.nan)

    fig, ax = plt.subplots(figsize=(12, 7))
    image = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=0.35, vmax=0.80, interpolation="nearest")
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    ax.set_title("Query Direction Separability: AUROC(high score = hallucinated)")
    fig.colorbar(image, ax=ax, label="test AUROC")

    top = select_top_directions(auc_rows, 15, "all")
    for row in top:
        layer = int(row["layer"])
        head = int(row["head"])
        ax.scatter([head], [layer], s=45, facecolors="none", edgecolors="black", linewidths=1.1)
    fig.tight_layout()
    path = os.path.join(output_dir, "query_direction_auc_heatmap.png")
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def load_calibration(path):
    data = np.load(path)
    out = {}
    layers = data["layers"].astype(int)
    heads = data["heads"].astype(int)
    directions = data["directions"].astype(np.float32)
    thresholds = data["threshold_midpoint"].astype(np.float32)
    for idx, (layer, head) in enumerate(zip(layers, heads)):
        key = f"{int(layer)}:{int(head)}"
        direction = directions[idx]
        norm = np.linalg.norm(direction)
        if norm > 1e-12:
            direction = direction / norm
        out[key] = {
            "direction": direction.astype(np.float32),
            "threshold": float(thresholds[idx]),
        }
    return out


def label_family(row):
    value = row.get("label_family") or row.get("label", "")
    if str(value).startswith("hallucinated"):
        return "hallucinated"
    if value in {"kept_grounded", "lost_grounded"}:
        return value
    return str(value)


def l2_normalize(x, eps=1e-12):
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(denom, eps)


def direction_scores_for_head(vectors, layers, heads, step_ids, step_rows_by_id, layer, head, direction, query_normalization):
    mask = (layers == layer) & (heads == head)
    if not np.any(mask):
        return [], None, None, None
    x = vectors[mask].astype(np.float32)
    if query_normalization == "l2":
        x = l2_normalize(x)
    direction = direction.astype(np.float32)
    direction = direction / max(np.linalg.norm(direction), 1e-12)
    scores = x @ direction
    local_step_ids = step_ids[mask]

    rows = []
    for score, step_id in zip(scores.tolist(), local_step_ids.tolist()):
        step = step_rows_by_id.get(int(step_id), {})
        rows.append({
            "step_id": int(step_id),
            "score": float(score),
            "label": step.get("label", ""),
            "label_family": label_family(step),
            "image_id": step.get("image_id", ""),
            "image": step.get("image", ""),
            "object_word": step.get("object_word", ""),
            "target_token": step.get("target_token", ""),
        })
    return rows, x, scores, local_step_ids


def plot_score_distribution(score_rows, head_key, threshold, output_dir):
    groups = ["hallucinated", "lost_grounded", "kept_grounded"]
    values_by_group = {
        group: [safe_float(row["score"]) for row in score_rows if row.get("label_family") == group]
        for group in groups
    }
    if sum(len(values) for values in values_by_group.values()) < 3:
        return None
    all_values = [value for values in values_by_group.values() for value in values]
    low, high = np.percentile(all_values, [1, 99])
    if high <= low:
        low, high = min(all_values), max(all_values)
    bins = np.linspace(low, high, 35)

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for group in groups:
        values = values_by_group[group]
        if len(values) < 2:
            continue
        ax.hist(values, bins=bins, density=True, alpha=0.45, color=COLORS[group], label=f"{group} (n={len(values)})")
        ax.axvline(np.mean(values), color=COLORS[group], linewidth=1.6)
    ax.axvline(threshold, color="#333333", linestyle="--", linewidth=1.3, label="calibrated threshold")
    ax.set_xlabel("<normalized query, hallucination direction>")
    ax.set_ylabel("density")
    ax.set_title(f"Query Direction Score Distribution ({head_key})")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    path = os.path.join(output_dir, f"query_direction_score_distribution_{head_key.replace(':', '_')}.png")
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def residual_pc_axis(x, scores, direction):
    residual = x - scores[:, None] * direction[None, :]
    residual = residual - residual.mean(axis=0, keepdims=True)
    if residual.shape[0] < 2:
        return np.zeros((residual.shape[0],), dtype=float)
    _, _, vh = np.linalg.svd(residual, full_matrices=False)
    return residual @ vh[0]


def plot_direction_plane(score_rows, x, scores, head_key, direction, threshold, output_dir):
    if x is None or x.shape[0] < 4:
        return None
    y = residual_pc_axis(x, scores, direction)
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for group in ["kept_grounded", "lost_grounded", "hallucinated"]:
        indices = [idx for idx, row in enumerate(score_rows) if row.get("label_family") == group]
        if not indices:
            continue
        ax.scatter(scores[indices], y[indices], s=20, alpha=0.72, color=COLORS[group], label=group)
    ax.axvline(threshold, color="#333333", linestyle="--", linewidth=1.2)
    ax.set_xlabel("hallucination direction score")
    ax.set_ylabel("residual PC1")
    ax.set_title(f"Query Space View Around Direction ({head_key})")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = os.path.join(output_dir, f"query_direction_plane_{head_key.replace(':', '_')}.png")
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def summarize_direction_scores(score_rows, head_key, threshold):
    output = []
    for group in ["hallucinated", "lost_grounded", "kept_grounded"]:
        values = [safe_float(row["score"]) for row in score_rows if row.get("label_family") == group]
        if not values:
            continue
        output.append({
            "head_key": head_key,
            "label_family": group,
            "n": len(values),
            "mean_score": float(np.mean(values)),
            "median_score": float(np.median(values)),
            "std_score": float(np.std(values)),
            "threshold": threshold,
            "above_threshold_rate": float(np.mean(np.asarray(values) >= threshold)),
        })
    return output


def top_score_examples(score_rows, head_key, n_examples):
    rows = []
    for rank, row in enumerate(sorted(score_rows, key=lambda item: safe_float(item["score"]), reverse=True)[:n_examples], start=1):
        rows.append({"head_key": head_key, "rank_high": rank, **row})
    for rank, row in enumerate(sorted(score_rows, key=lambda item: safe_float(item["score"]))[:n_examples], start=1):
        rows.append({"head_key": head_key, "rank_low": rank, **row})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-probe-dir", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--direction-filter", choices=["high", "all"], default="high")
    parser.add_argument("--query-normalization", choices=["l2", "none"], default="l2")
    parser.add_argument("--n-examples", type=int, default=12)
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.query_probe_dir, "query_direction_viz")
    ensure_dir(output_dir)

    auc_path = os.path.join(args.query_probe_dir, "query_direction_auc.csv")
    steps_path = os.path.join(args.query_probe_dir, "query_direction_steps.csv")
    calibration_path = os.path.join(args.query_probe_dir, "query_direction_calibration.npz")
    vectors_path = os.path.join(args.query_probe_dir, "query_vectors.npz")

    auc_rows = load_csv(auc_path)
    top_rows = select_top_directions(auc_rows, args.top_k, args.direction_filter)
    write_csv(os.path.join(output_dir, "selected_top_query_directions.csv"), top_rows)
    heatmap = plot_auc_heatmap(auc_rows, output_dir)

    generated = [path for path in [heatmap] if path]
    if not os.path.exists(vectors_path):
        with open(os.path.join(output_dir, "README.txt"), "w") as f:
            f.write(
                "Only the AUROC heatmap was generated because query_vectors.npz is missing.\n"
                "Re-run query calibration with SAVE_QUERY_VECTORS=1 to generate score distributions and 2D plots.\n"
            )
        print(json.dumps({"generated": generated, "missing_query_vectors": vectors_path}, indent=2))
        return

    calibration = load_calibration(calibration_path)
    step_rows = {int(row["step_id"]): row for row in load_csv(steps_path)}
    data = np.load(vectors_path)
    vectors = data["vectors"].astype(np.float32)
    layers = data["layers"].astype(int)
    heads = data["heads"].astype(int)
    step_ids = data["step_ids"].astype(int)

    score_summaries = []
    score_examples = []
    all_score_rows = []
    for row in top_rows:
        layer = int(row["layer"])
        head = int(row["head"])
        head_key = f"{layer}:{head}"
        item = calibration.get(head_key)
        if item is None:
            continue
        score_rows, x, scores, _ = direction_scores_for_head(
            vectors,
            layers,
            heads,
            step_ids,
            step_rows,
            layer,
            head,
            item["direction"],
            args.query_normalization,
        )
        if not score_rows:
            continue
        for score_row in score_rows:
            all_score_rows.append({"head_key": head_key, **score_row})
        score_summaries.extend(summarize_direction_scores(score_rows, head_key, item["threshold"]))
        score_examples.extend(top_score_examples(score_rows, head_key, args.n_examples))
        generated.append(plot_score_distribution(score_rows, head_key, item["threshold"], output_dir))
        generated.append(plot_direction_plane(score_rows, x, scores, head_key, item["direction"], item["threshold"], output_dir))

    write_csv(os.path.join(output_dir, "query_direction_score_summary.csv"), score_summaries)
    write_csv(os.path.join(output_dir, "query_direction_score_examples.csv"), score_examples)
    write_csv(os.path.join(output_dir, "query_direction_scores.csv"), all_score_rows)
    generated = [path for path in generated if path]
    print(json.dumps({"generated": generated}, indent=2))


if __name__ == "__main__":
    main()

import argparse
import csv
import json
import math
import os
from collections import OrderedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eval_scripts.soft_routing.head_prior_utils import head_key


def score_items_from_field(raw_scores):
    items = []
    if isinstance(raw_scores, dict):
        for key, score in raw_scores.items():
            layer, head = key.split(":")
            items.append({"layer": int(layer), "head": int(head), "score": float(score)})
    else:
        for item in raw_scores:
            if isinstance(item, dict):
                items.append({
                    "layer": int(item["layer"]),
                    "head": int(item["head"]),
                    "score": float(item.get("score", 0.0)),
                })
            else:
                items.append({"layer": int(item[0]), "head": int(item[1]), "score": float(item[2])})
    return items


def load_candidate_heads(path, top_k):
    with open(path, "r") as f:
        data = json.load(f)

    score_items = []
    for field in ("score_sorted_head_scores", "contrastive_scores", "hal_head_scores"):
        if field in data:
            score_items = score_items_from_field(data[field])
            break
    score_by_key = {head_key(item["layer"], item["head"]): float(item["score"]) for item in score_items}

    heads = data.get("hal_heads")
    if not heads:
        heads = [[item["layer"], item["head"]] for item in sorted(score_items, key=lambda x: x["score"], reverse=True)]
    heads = heads[:top_k]

    output = []
    for idx, (layer, head) in enumerate(heads, start=1):
        key = head_key(layer, head)
        output.append({
            "rank": idx,
            "layer": int(layer),
            "head": int(head),
            "head_key": key,
            "score": float(score_by_key.get(key, 0.0)),
        })
    return output


def load_norm_thresholds(path):
    if not path:
        return {}
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("head_norm_thresholds", data)


def sample_key(record):
    for field in ("image_id", "question_id", "image"):
        if field in record and record[field] is not None:
            return str(record[field])
    return "unknown"


def load_records(path, allowed_keys, max_samples):
    samples = OrderedDict()
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            key = record.get("head_key") or head_key(record["layer"], record["head"])
            if key not in allowed_keys:
                continue
            sid = sample_key(record)
            if sid not in samples:
                if max_samples and max_samples > 0 and len(samples) >= max_samples:
                    continue
                samples[sid] = {
                    "sample_id": sid,
                    "image": record.get("image", ""),
                    "heads": {},
                }
            if sid in samples:
                samples[sid]["heads"][key] = record
    return list(samples.values())


def clamp01(value):
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def build_matrices(samples, heads, thresholds, text_tau, text_high, norm_field, default_norm_threshold):
    n_samples = len(samples)
    n_heads = len(heads)
    text_mass = np.zeros((n_samples, n_heads), dtype=float)
    norm_value = np.zeros((n_samples, n_heads), dtype=float)
    text_gate = np.zeros((n_samples, n_heads), dtype=float)
    norm_gate = np.zeros((n_samples, n_heads), dtype=float)
    text_norm_gate = np.zeros((n_samples, n_heads), dtype=float)
    strength = np.zeros((n_samples, n_heads), dtype=float)
    present = np.zeros((n_samples, n_heads), dtype=float)

    text_den = max(float(text_high) - float(text_tau), 1e-6)
    for sample_idx, sample in enumerate(samples):
        records = sample["heads"]
        for head_idx, head in enumerate(heads):
            key = head["head_key"]
            record = records.get(key)
            if record is None:
                continue
            present[sample_idx, head_idx] = 1.0
            tmass = float(record.get("text_mass", 0.0))
            nval = float(record.get(norm_field, record.get("text_value_norm", 0.0)))
            threshold = thresholds.get(key, {})
            nth = float(threshold.get("threshold", default_norm_threshold))
            nlow = float(threshold.get("low", nth))
            nhigh = float(threshold.get("high", max(nlow + 1e-6, 1.0)))
            if nhigh <= nlow:
                nhigh = nlow + 1e-6

            texcess = clamp01((tmass - text_tau) / text_den)
            nexcess = clamp01((nval - nlow) / (nhigh - nlow))
            text_mass[sample_idx, head_idx] = tmass
            norm_value[sample_idx, head_idx] = nval
            text_gate[sample_idx, head_idx] = float(tmass >= text_tau)
            norm_gate[sample_idx, head_idx] = float(nval >= nth)
            text_norm_gate[sample_idx, head_idx] = float(tmass >= text_tau and nval >= nth)
            strength[sample_idx, head_idx] = texcess * nexcess
    return {
        "text_mass": text_mass,
        "norm_value": norm_value,
        "text_gate": text_gate,
        "norm_gate": norm_gate,
        "text_norm_gate": text_norm_gate,
        "strength": strength,
        "present": present,
    }


def write_csv(path, rows):
    if not rows:
        with open(path, "w") as f:
            f.write("")
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_matrix_csv(path, samples, heads, matrix):
    rows = []
    for sample_idx, sample in enumerate(samples):
        row = {"sample_id": sample["sample_id"], "image": sample.get("image", "")}
        for head_idx, head in enumerate(heads):
            row[head["head_key"]] = float(matrix[sample_idx, head_idx])
        rows.append(row)
    write_csv(path, rows)


def summarize_samples(samples, heads, matrices):
    rows = []
    active = matrices["text_norm_gate"]
    strength = matrices["strength"]
    text_gate = matrices["text_gate"]
    norm_gate = matrices["norm_gate"]
    for sample_idx, sample in enumerate(samples):
        active_indices = np.where(active[sample_idx] > 0.5)[0]
        active_heads = [heads[idx]["head_key"] for idx in active_indices[:20]]
        rows.append({
            "sample_id": sample["sample_id"],
            "image": sample.get("image", ""),
            "text_active_count": int(text_gate[sample_idx].sum()),
            "norm_active_count": int(norm_gate[sample_idx].sum()),
            "text_norm_active_count": int(active[sample_idx].sum()),
            "text_norm_active_frac": float(active[sample_idx].mean()) if active.shape[1] else 0.0,
            "continuous_strength_sum": float(strength[sample_idx].sum()),
            "continuous_strength_mean": float(strength[sample_idx].mean()) if strength.shape[1] else 0.0,
            "top_active_heads": " ".join(active_heads),
        })
    return rows


def summarize_heads(samples, heads, matrices):
    rows = []
    n = max(len(samples), 1)
    for head_idx, head in enumerate(heads):
        rows.append({
            "rank": head["rank"],
            "layer": head["layer"],
            "head": head["head"],
            "head_key": head["head_key"],
            "contrastive_score": head["score"],
            "text_trigger_rate": float(matrices["text_gate"][:, head_idx].sum() / n),
            "norm_trigger_rate": float(matrices["norm_gate"][:, head_idx].sum() / n),
            "text_norm_trigger_rate": float(matrices["text_norm_gate"][:, head_idx].sum() / n),
            "mean_text_mass": float(matrices["text_mass"][:, head_idx].mean()),
            "mean_norm_value": float(matrices["norm_value"][:, head_idx].mean()),
            "mean_continuous_strength": float(matrices["strength"][:, head_idx].mean()),
            "present_count": int(matrices["present"][:, head_idx].sum()),
        })
    return rows


def jaccard_matrix(binary):
    binary = binary.astype(bool)
    n = binary.shape[0]
    output = np.zeros((n, n), dtype=float)
    for i in range(n):
        inter = np.logical_and(binary[i], binary).sum(axis=1)
        union = np.logical_or(binary[i], binary).sum(axis=1)
        output[i] = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    return output


def write_jaccard_summary(path, matrix):
    if matrix.shape[0] <= 1:
        values = np.array([], dtype=float)
    else:
        values = matrix[np.triu_indices(matrix.shape[0], k=1)]
    rows = [{
        "n_pairs": int(values.size),
        "mean_jaccard": float(values.mean()) if values.size else None,
        "std_jaccard": float(values.std()) if values.size else None,
        "p10_jaccard": float(np.percentile(values, 10)) if values.size else None,
        "p50_jaccard": float(np.percentile(values, 50)) if values.size else None,
        "p90_jaccard": float(np.percentile(values, 90)) if values.size else None,
    }]
    write_csv(path, rows)


def save_heatmap(path, matrix, title, xlabel, ylabel, cmap="viridis", vmin=None, vmax=None, ytick_labels=None):
    height = max(4.0, min(12.0, matrix.shape[0] * 0.12))
    width = max(7.0, min(18.0, matrix.shape[1] * 0.035))
    fig, ax = plt.subplots(figsize=(width, height))
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks([])
    if ytick_labels:
        step = max(1, len(ytick_labels) // 20)
        ticks = list(range(0, len(ytick_labels), step))
        ax.set_yticks(ticks)
        ax.set_yticklabels([ytick_labels[idx] for idx in ticks], fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_sample_hist(path, sample_rows):
    counts = [int(row["text_norm_active_count"]) for row in sample_rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(counts, bins=min(30, max(5, len(set(counts)))), color="#3a7ca5", edgecolor="white")
    ax.set_title("Active top-100 candidate heads per sample")
    ax.set_xlabel("active heads, text_mass gate AND norm gate")
    ax.set_ylabel("sample count")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_head_rates(path, head_rows):
    ranks = [int(row["rank"]) for row in head_rows]
    rates = [float(row["text_norm_trigger_rate"]) for row in head_rows]
    scores = [float(row["contrastive_score"]) for row in head_rows]
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.bar(ranks, rates, color="#4f6d7a", width=0.85)
    ax1.set_xlabel("AD-HH contrastive rank in top-100")
    ax1.set_ylabel("sample activation rate")
    ax1.set_ylim(0, 1)
    ax2 = ax1.twinx()
    ax2.plot(ranks, scores, color="#c44536", linewidth=1.2)
    ax2.set_ylabel("contrastive score")
    ax1.set_title("Sample activation rate by AD-HH rank")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records-jsonl", required=True)
    parser.add_argument("--candidate-head-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--head-norm-thresholds-path", default="")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--text-tau", type=float, default=0.4)
    parser.add_argument("--text-high", type=float, default=0.9)
    parser.add_argument("--norm-field", default="text_value_norm")
    parser.add_argument("--default-norm-threshold", type=float, default=0.0)
    parser.add_argument("--max-jaccard-samples", type=int, default=200)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    heads = load_candidate_heads(args.candidate_head_path, args.top_k)
    thresholds = load_norm_thresholds(args.head_norm_thresholds_path)
    allowed = {head["head_key"] for head in heads}
    samples = load_records(args.records_jsonl, allowed, args.max_samples)
    if not samples:
        raise ValueError(f"No matching records in {args.records_jsonl}")

    matrices = build_matrices(
        samples,
        heads,
        thresholds,
        args.text_tau,
        args.text_high,
        args.norm_field,
        args.default_norm_threshold,
    )
    sample_rows = summarize_samples(samples, heads, matrices)
    head_rows = summarize_heads(samples, heads, matrices)

    write_csv(os.path.join(args.output_dir, "sample_gate_summary.csv"), sample_rows)
    write_csv(os.path.join(args.output_dir, "head_gate_summary.csv"), head_rows)
    write_matrix_csv(os.path.join(args.output_dir, "sample_head_text_norm_gate_matrix.csv"), samples, heads, matrices["text_norm_gate"])
    write_matrix_csv(os.path.join(args.output_dir, "sample_head_continuous_strength_matrix.csv"), samples, heads, matrices["strength"])
    write_matrix_csv(os.path.join(args.output_dir, "sample_head_text_mass_matrix.csv"), samples, heads, matrices["text_mass"])

    active_counts = matrices["text_norm_gate"].sum(axis=1)
    sample_order = np.argsort(-active_counts)
    head_labels = [f'{head["rank"]}:{head["head_key"]}' for head in heads]
    save_heatmap(
        os.path.join(args.output_dir, "sample_head_text_norm_gate_heatmap.png"),
        matrices["text_norm_gate"][sample_order].T,
        "Top-100 AD-HH candidate activation by sample",
        "samples sorted by active candidate count",
        "AD-HH rank:layer:head",
        cmap="Greys",
        vmin=0,
        vmax=1,
        ytick_labels=head_labels,
    )
    save_heatmap(
        os.path.join(args.output_dir, "sample_head_continuous_strength_heatmap.png"),
        matrices["strength"][sample_order].T,
        "Continuous text_norm strength by sample",
        "samples sorted by active candidate count",
        "AD-HH rank:layer:head",
        cmap="magma",
        vmin=0,
        vmax=max(1e-6, float(np.max(matrices["strength"]))),
        ytick_labels=head_labels,
    )
    save_sample_hist(os.path.join(args.output_dir, "sample_active_head_count_hist.png"), sample_rows)
    save_head_rates(os.path.join(args.output_dir, "head_activation_rate_by_rank.png"), head_rows)

    jaccard_input = matrices["text_norm_gate"][sample_order]
    if args.max_jaccard_samples and jaccard_input.shape[0] > args.max_jaccard_samples:
        jaccard_input = jaccard_input[:args.max_jaccard_samples]
    jm = jaccard_matrix(jaccard_input)
    write_jaccard_summary(os.path.join(args.output_dir, "sample_pairwise_jaccard_summary.csv"), jm)
    save_heatmap(
        os.path.join(args.output_dir, "sample_pairwise_jaccard_heatmap.png"),
        jm,
        "Pairwise Jaccard similarity of active head sets",
        "samples",
        "samples",
        cmap="viridis",
        vmin=0,
        vmax=1,
    )

    metadata = {
        "records_jsonl": args.records_jsonl,
        "candidate_head_path": args.candidate_head_path,
        "head_norm_thresholds_path": args.head_norm_thresholds_path,
        "top_k": args.top_k,
        "num_samples": len(samples),
        "text_tau": args.text_tau,
        "text_high": args.text_high,
        "norm_field": args.norm_field,
        "outputs": [
            "sample_gate_summary.csv",
            "head_gate_summary.csv",
            "sample_head_text_norm_gate_matrix.csv",
            "sample_head_continuous_strength_matrix.csv",
            "sample_head_text_mass_matrix.csv",
            "sample_head_text_norm_gate_heatmap.png",
            "sample_head_continuous_strength_heatmap.png",
            "sample_active_head_count_hist.png",
            "head_activation_rate_by_rank.png",
            "sample_pairwise_jaccard_summary.csv",
            "sample_pairwise_jaccard_heatmap.png",
        ],
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("wrote:", args.output_dir)
    print("num samples:", len(samples))
    print("num heads:", len(heads))
    print("active count mean:", float(active_counts.mean()))
    print("active count p10/p50/p90:", np.percentile(active_counts, [10, 50, 90]).tolist())


if __name__ == "__main__":
    main()

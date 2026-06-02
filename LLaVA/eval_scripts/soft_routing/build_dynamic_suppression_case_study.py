#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "dark": "#111827",
    "muted": "#64748b",
    "grid": "#e2e8f0",
    "hall": "#dc2626",
    "ratio": "#7c3aed",
    "strong": "#f97316",
}


def setup_style():
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 320,
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 6.4,
            "legend.fontsize": 7.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#cbd5e1",
            "axes.linewidth": 0.8,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def as_float(row, key, default=0.0):
    try:
        value = float(row.get(key, default))
        return value if math.isfinite(value) else default
    except Exception:
        return default


def as_int(row, key, default=0):
    try:
        return int(float(row.get(key, default)))
    except Exception:
        return default


def mean(values):
    values = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(values)) if values else 0.0


def quantile(values, q):
    values = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not values:
        return 0.0
    idx = min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))
    return values[idx]


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save(fig, output_dir, name, formats):
    paths = {}
    for fmt in formats:
        path = os.path.join(output_dir, f"{name}.{fmt}")
        fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
        paths[fmt] = path
    plt.close(fig)
    return paths


def is_readable_token(token):
    token = token.strip()
    return bool(re.match(r"^[A-Za-z][A-Za-z-]{2,}$", token))


def normalize_object_key(text):
    text = str(text or "").strip().lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    parts = []
    for part in text.split():
        if len(part) > 3 and part.endswith("s"):
            part = part[:-1]
        parts.append(part)
    return " ".join(parts)


def parse_label_map(spec):
    mapping = {}
    for item in str(spec or "").split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"label map entries must be old=new: {item}")
        old, new = item.split("=", 1)
        mapping[normalize_object_key(old)] = new.strip()
    return mapping


def object_text(row):
    return (row.get("object_words") or row.get("token_text") or "").strip()


def display_text(row, label_map=None):
    raw = object_text(row)
    key = normalize_object_key(raw)
    if label_map and key in label_map:
        return label_map[key]
    return raw or row.get("token_text", "")


def token_label(row, label_map=None):
    return f"{display_text(row, label_map)}@{row['step_idx']}"


def aggregate_mentions(trace_rows, label_map=None):
    grouped = defaultdict(list)
    for row in trace_rows:
        if row.get("label") != "hallucinated":
            continue
        key = (
            row["question_id"],
            row.get("image", ""),
            row["step_idx"],
            row["token_text"],
            row.get("object_words", ""),
        )
        grouped[key].append(row)

    mentions = []
    for (qid, image, step_idx, token_text, words), rows in grouped.items():
        mention_object_text = words or token_text
        normalized = normalize_object_key(mention_object_text)
        shown = label_map.get(normalized, mention_object_text) if label_map else mention_object_text
        deltas = [as_float(row, "delta") for row in rows]
        ratios = [as_float(row, "bounded_ratio") for row in rows]
        gates = [as_float(row, "gate") for row in rows]
        scores = [as_float(row, "score") for row in rows]
        mentions.append(
            {
                "question_id": qid,
                "image": image,
                "step_idx": as_int({"x": step_idx}, "x"),
                "token_text": token_text,
                "object_words": words,
                "object_key": normalized,
                "display_text": shown,
                "token_label": f"{shown}@{step_idx}",
                "n_heads": len(rows),
                "mean_delta": mean(deltas),
                "median_delta": quantile(deltas, 0.5),
                "p90_delta": quantile(deltas, 0.9),
                "max_delta": max(deltas) if deltas else 0.0,
                "strong_delta_rate": sum(delta >= 0.8 for delta in deltas) / len(deltas),
                "weak_delta_rate": sum(delta <= 0.05 for delta in deltas) / len(deltas),
                "mean_ratio": mean(ratios),
                "max_ratio": max(ratios) if ratios else 0.0,
                "mean_gate": mean(gates),
                "mean_head_score": mean(scores),
                "readable_token": int(is_readable_token(shown)),
            }
        )
    return sorted(mentions, key=lambda row: (row["question_id"], row["step_idx"]))


def sample_candidates(mentions, sample_lookup, prefer_unique=False, min_unique=0):
    by_qid = defaultdict(list)
    for mention in mentions:
        by_qid[mention["question_id"]].append(mention)

    rows = []
    for qid, group in by_qid.items():
        if len(group) < 2:
            continue
        means = [row["mean_delta"] for row in group]
        ratios = [row["mean_ratio"] for row in group]
        readable = sum(row["readable_token"] for row in group)
        unique_keys = []
        unique_readable_keys = []
        for row in group:
            key = row["object_key"] or normalize_object_key(row["token_text"])
            if key and key not in unique_keys:
                unique_keys.append(key)
            if row["readable_token"] and key and key not in unique_readable_keys:
                unique_readable_keys.append(key)
        sample = sample_lookup.get(qid, {})
        rows.append(
            {
                "question_id": qid,
                "image": sample.get("image", group[0]["image"]),
                "n_hallucinated_mentions": len(group),
                "n_readable_tokens": readable,
                "n_unique_object_tokens": len(unique_keys),
                "n_unique_readable_object_tokens": len(unique_readable_keys),
                "unique_object_tokens": " ".join(unique_keys),
                "mean_delta_min": min(means),
                "mean_delta_max": max(means),
                "mean_delta_spread": max(means) - min(means),
                "mean_ratio_min": min(ratios),
                "mean_ratio_max": max(ratios),
                "mean_ratio_spread": max(ratios) - min(ratios),
                "tokens": " ".join(row["token_label"] for row in group),
                "caption": sample.get("caption", ""),
            }
        )
    if min_unique:
        rows = [row for row in rows if int(row["n_unique_readable_object_tokens"]) >= int(min_unique)]
    if prefer_unique:
        rows.sort(
            key=lambda row: (
                int(row["n_unique_readable_object_tokens"]),
                float(row["mean_delta_spread"]),
                int(row["n_readable_tokens"]),
                int(row["n_hallucinated_mentions"]),
            ),
            reverse=True,
        )
    else:
        rows.sort(key=lambda row: (row["n_readable_tokens"], row["mean_delta_spread"], row["n_hallucinated_mentions"]), reverse=True)
    return rows


def choose_sample(candidates, requested_qid=None):
    if requested_qid:
        for row in candidates:
            if row["question_id"] == requested_qid:
                return row
        raise ValueError(f"requested question_id not found among multi-hall candidates: {requested_qid}")
    if not candidates:
        raise ValueError("no samples with at least two hallucinated mentions")
    return candidates[0]


def collapse_unique_mentions(mentions):
    by_key = {}
    for row in mentions:
        key = row["object_key"] or normalize_object_key(row["token_text"])
        current = by_key.get(key)
        if current is None or float(row["mean_delta"]) > float(current["mean_delta"]):
            by_key[key] = row
    return sorted(by_key.values(), key=lambda row: row["step_idx"])


def head_token_matrix(trace_rows, qid, mentions, max_heads):
    mention_keys = {(str(m["step_idx"]), m["token_text"], m.get("object_words", "")) for m in mentions}
    by_head = defaultdict(dict)
    head_scores = {}
    for row in trace_rows:
        if row.get("question_id") != qid or row.get("label") != "hallucinated":
            continue
        key = (row["step_idx"], row["token_text"], row.get("object_words", ""))
        if key not in mention_keys:
            continue
        head = row.get("head_key") or f"{row.get('layer')}:{row.get('head')}"
        by_head[head][key] = as_float(row, "delta")
        head_scores[head] = as_float(row, "score")

    token_keys = [(str(m["step_idx"]), m["token_text"], m.get("object_words", "")) for m in mentions]
    head_order = []
    for head, vals in by_head.items():
        vector = [vals.get(key, 0.0) for key in token_keys]
        head_order.append(
            (
                max(vector) - min(vector),
                mean(vector),
                head_scores.get(head, 0.0),
                head,
            )
        )
    head_order.sort(reverse=True)
    head_order = head_order[:max_heads]

    matrix = []
    rows = []
    for _, _, score, head in head_order:
        vals = [by_head[head].get(key, 0.0) for key in token_keys]
        matrix.append(vals)
        rows.append({"head_key": head, "head_score": score, **{f"delta_{token}@{step}": val for (step, token, _), val in zip(token_keys, vals)}})
    return token_keys, [item[3] for item in head_order], np.asarray(matrix, dtype=np.float64), rows


def plot_case(output_dir, formats, chosen, mentions, token_keys, head_order, matrix):
    setup_style()
    fig = plt.figure(figsize=(7.4, 4.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.05, 2.2])
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[1, 0])

    x = np.arange(len(mentions))
    mean_delta = [row["mean_delta"] for row in mentions]
    strong = [row["strong_delta_rate"] for row in mentions]
    ratios = [row["mean_ratio"] for row in mentions]
    labels = [row["token_label"] for row in mentions]

    width = 0.36
    ax_bar.bar(x - width / 2, mean_delta, width, color=COLORS["hall"], alpha=0.82, label="mean suppression")
    ax_bar.bar(x + width / 2, strong, width, color=COLORS["strong"], alpha=0.75, label="strong-head rate")
    ax_bar.plot(x, ratios, color=COLORS["ratio"], marker="o", linewidth=1.4, label="mean text ratio")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels)
    ax_bar.set_ylim(0, 1.02)
    ax_bar.set_ylabel("value")
    ax_bar.grid(axis="y")
    ax_bar.legend(frameon=False, ncol=3, loc="upper left")
    ax_bar.set_title(f"One caption, multiple hallucinated tokens: qid {chosen['question_id']}")

    if matrix.size:
        im = ax_heat.imshow(matrix, aspect="auto", cmap="magma", vmin=0, vmax=1)
        ax_heat.set_xticks(np.arange(len(labels)))
        ax_heat.set_xticklabels(labels)
        ax_heat.set_yticks(np.arange(len(head_order)))
        ax_heat.set_yticklabels(head_order)
        ax_heat.set_xlabel("hallucinated token")
        ax_heat.set_ylabel("selected heads with largest token-to-token variation")
        cbar = fig.colorbar(im, ax=ax_heat, shrink=0.86)
        cbar.set_label("suppression strength δ")
    else:
        ax_heat.text(0.5, 0.5, "no head-token matrix", ha="center", va="center")
        ax_heat.axis("off")

    return save(fig, output_dir, "dynamic_suppression_multi_hall_case", formats)


def write_notes(path, chosen, mentions):
    lines = [
        "# Dynamic Suppression Case Study",
        "",
        f"Question id: `{chosen['question_id']}`",
        f"Image: `{chosen['image']}`",
        "",
        "This case contains multiple hallucinated object-token steps in the same generated caption. The selected head pool does not apply a fixed suppression level to all hallucinated tokens. Instead, suppression changes with the online text-over-image ratio at each decoding step.",
        "",
        "## Token-Level Summary",
        "",
        "| step | token | mean delta | p90 delta | strong-head rate | weak-head rate | mean ratio |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in mentions:
        lines.append(
            f"| {row['step_idx']} | {row['token_text']} | {row['mean_delta']:.3f} | {row['p90_delta']:.3f} | {row['strong_delta_rate']:.3f} | {row['weak_delta_rate']:.3f} | {row['mean_ratio']:.3f} |"
        )
    lines += [
        "",
        "## Paper-Ready Interpretation",
        "",
        "Within a single caption, hallucinated object tokens can receive substantially different intervention strengths. This supports the need for dynamic suppression: the offline head pool identifies where intervention is plausible, but the online text-over-image ratio determines how strongly each selected head is suppressed at a particular token step.",
        "",
        "This case should not be interpreted as token-level hallucination detection. All listed tokens are hallucinated under CHAIR, but the intervention strength varies because each token is generated under a different attention-routing state.",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-csv", required=True)
    parser.add_argument("--samples-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--question-id", default="")
    parser.add_argument("--max-heads", type=int, default=35)
    parser.add_argument("--formats", default="png,pdf,svg")
    parser.add_argument("--prefer-unique-object-tokens", action="store_true")
    parser.add_argument("--unique-object-tokens", action="store_true")
    parser.add_argument("--min-unique-readable-object-tokens", type=int, default=0)
    parser.add_argument("--token-label-map", default="")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    formats = [fmt.strip() for fmt in args.formats.split(",") if fmt.strip()]

    trace_rows = read_csv(args.trace_csv)
    samples = read_csv(args.samples_csv)
    sample_lookup = {row["question_id"]: row for row in samples}

    label_map = parse_label_map(args.token_label_map)
    mentions = aggregate_mentions(trace_rows, label_map)
    candidates = sample_candidates(
        mentions,
        sample_lookup,
        prefer_unique=args.prefer_unique_object_tokens,
        min_unique=args.min_unique_readable_object_tokens,
    )
    chosen = choose_sample(candidates, args.question_id or None)
    chosen_mentions = [row for row in mentions if row["question_id"] == chosen["question_id"]]
    chosen_mentions.sort(key=lambda row: row["step_idx"])
    if args.unique_object_tokens:
        chosen_mentions = collapse_unique_mentions(chosen_mentions)

    token_keys, head_order, matrix, matrix_rows = head_token_matrix(
        trace_rows, chosen["question_id"], chosen_mentions, args.max_heads
    )

    write_csv(os.path.join(args.output_dir, "multi_hall_sample_candidates.csv"), candidates)
    write_csv(os.path.join(args.output_dir, "chosen_case_token_summary.csv"), chosen_mentions)
    write_csv(os.path.join(args.output_dir, "chosen_case_head_token_delta_matrix.csv"), matrix_rows)
    figure_paths = plot_case(args.output_dir, formats, chosen, chosen_mentions, token_keys, head_order, matrix)

    summary = {
        "trace_csv": args.trace_csv,
        "samples_csv": args.samples_csv,
        "chosen_question_id": chosen["question_id"],
        "chosen_image": chosen["image"],
        "chosen_caption": chosen.get("caption", ""),
        "n_candidates": len(candidates),
        "chosen_candidate": chosen,
        "token_summary": chosen_mentions,
        "figure_paths": figure_paths,
    }
    write_json(os.path.join(args.output_dir, "dynamic_suppression_case_summary.json"), summary)
    write_notes(os.path.join(args.output_dir, "dynamic_suppression_case_notes.md"), chosen, chosen_mentions)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

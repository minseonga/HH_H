import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict

from eval_scripts.soft_routing.analyze_text_heavy_object_alignment import auc_score


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


def safe_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def mean(values):
    clean = [value for value in values if value is not None and math.isfinite(value)]
    return sum(clean) / len(clean) if clean else None


def percentile(values, q):
    clean = sorted(value for value in values if value is not None and math.isfinite(value))
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    pos = (len(clean) - 1) * float(q)
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return clean[low]
    frac = pos - low
    return clean[low] * (1.0 - frac) + clean[high] * frac


def max_or_none(values):
    clean = [value for value in values if value is not None and math.isfinite(value)]
    return max(clean) if clean else None


def mention_key(row):
    return (
        str(row.get("question_id", "")),
        str(row.get("image_id", "")),
        str(row.get("image", "")),
        str(row.get("word", "")),
        str(row.get("node_word", "")),
        str(row.get("token_pos", "")),
        str(row.get("label", "")),
    )


def load_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def head_stats(rows):
    scores = [safe_float(row.get("score")) for row in rows]
    margins = [safe_float(row.get("margin")) for row in rows]
    scores = [value for value in scores if value is not None]
    margins = [value for value in margins if value is not None]
    hard = [1.0 if value > 0 else 0.0 for value in margins]
    return {
        "score_mean": mean(scores),
        "score_max": max_or_none(scores),
        "score_p90": percentile(scores, 0.9),
        "margin_mean": mean(margins),
        "margin_max": max_or_none(margins),
        "margin_p90": percentile(margins, 0.9),
        "hard_rate": mean(hard),
        "n_mentions": len(rows),
    }


def auc_rows(rows, features, label_key="label"):
    output = []
    total_labels = [int(row[label_key]) for row in rows]
    total_pos = sum(total_labels)
    total_neg = len(total_labels) - total_pos
    for feature in features:
        values = []
        labels = []
        for row in rows:
            value = safe_float(row.get(feature))
            if value is None:
                continue
            values.append(value)
            labels.append(int(row[label_key]))
        pos_values = [value for value, label in zip(values, labels) if label == 1]
        neg_values = [value for value, label in zip(values, labels) if label == 0]
        auc = auc_score(labels, values)
        output.append({
            "feature": feature,
            "n": len(values),
            "n_pos": sum(labels),
            "n_neg": len(labels) - sum(labels),
            "total_pos": total_pos,
            "total_neg": total_neg,
            "pos_mean": mean(pos_values),
            "neg_mean": mean(neg_values),
            "pos_minus_neg": (
                mean(pos_values) - mean(neg_values)
                if mean(pos_values) is not None and mean(neg_values) is not None
                else None
            ),
            "auroc_high_predicts_hallucinated": auc,
            "auroc_low_predicts_hallucinated": (1.0 - auc) if auc is not None else None,
        })
    return output


def build_adaptive_rows(head_score_rows, top_ks, select_features):
    by_sample_head = defaultdict(list)
    by_sample_mention = defaultdict(dict)
    sample_labels = {}
    sample_meta = {}

    for row in head_score_rows:
        qid = str(row.get("question_id", ""))
        head_key = str(row.get("head_key", ""))
        rank = int(float(row.get("rank", 10**9)))
        label = int(float(row.get("label", 0)))
        by_sample_head[(qid, head_key)].append(row)
        by_sample_mention[(qid, mention_key(row))][head_key] = row
        sample_labels[qid] = max(sample_labels.get(qid, 0), label)
        sample_meta.setdefault(qid, {
            "question_id": qid,
            "image_id": row.get("image_id", ""),
            "image": row.get("image", ""),
        })

    head_ranks = {}
    for row in head_score_rows:
        head_ranks[str(row.get("head_key", ""))] = int(float(row.get("rank", 10**9)))

    mention_rows = []
    sample_rows = []
    selection_rows = []
    for top_k in top_ks:
        allowed_heads = {
            head for head, rank in head_ranks.items()
            if rank <= top_k
        }
        for select_feature in select_features:
            selection_name = f"top{top_k}_{select_feature}"
            for qid, label in sample_labels.items():
                candidates = []
                for head in allowed_heads:
                    rows = by_sample_head.get((qid, head), [])
                    if not rows:
                        continue
                    stats = head_stats(rows)
                    value = stats.get(select_feature)
                    if value is None:
                        continue
                    candidates.append((float(value), head, stats))
                if not candidates:
                    continue
                candidates.sort(key=lambda item: item[0], reverse=True)
                selected_value, selected_head, selected_stats = candidates[0]

                sample_row = {
                    **sample_meta.get(qid, {"question_id": qid}),
                    "label": label,
                    "label_name": "CHAIRs1" if label else "CHAIRs0",
                    "selection": selection_name,
                    "top_k": top_k,
                    "select_feature": select_feature,
                    "selected_head": selected_head,
                    "selected_head_rank": head_ranks.get(selected_head),
                    "selection_value": selected_value,
                }
                for key, value in selected_stats.items():
                    sample_row[f"selected_{key}"] = value
                sample_rows.append(sample_row)
                selection_rows.append({
                    "selection": selection_name,
                    "question_id": qid,
                    "label": label,
                    "selected_head": selected_head,
                    "selected_head_rank": head_ranks.get(selected_head),
                    "selection_value": selected_value,
                })

                for (sample_qid, mkey), per_head in by_sample_mention.items():
                    if sample_qid != qid:
                        continue
                    selected = per_head.get(selected_head)
                    if selected is None:
                        continue
                    score = safe_float(selected.get("score"))
                    margin = safe_float(selected.get("margin"))
                    mention_rows.append({
                        "selection": selection_name,
                        "question_id": qid,
                        "image_id": selected.get("image_id", ""),
                        "image": selected.get("image", ""),
                        "word": selected.get("word", ""),
                        "node_word": selected.get("node_word", ""),
                        "token_pos": selected.get("token_pos", ""),
                        "label": int(float(selected.get("label", 0))),
                        "label_name": selected.get("label_name", ""),
                        "selected_head": selected_head,
                        "selected_head_rank": head_ranks.get(selected_head),
                        "selected_score": score,
                        "selected_margin": margin,
                        "selected_hard": 1.0 if margin is not None and margin > 0 else 0.0,
                    })
    return mention_rows, sample_rows, selection_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-scores", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-ks", default="1,3,5,10,20")
    parser.add_argument(
        "--select-features",
        default="score_max,score_p90,margin_max,margin_p90,hard_rate",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    top_ks = [int(item) for item in args.top_ks.split(",") if item.strip()]
    select_features = [item.strip() for item in args.select_features.split(",") if item.strip()]

    head_score_rows = load_rows(args.head_scores)
    mention_rows, sample_rows, selection_rows = build_adaptive_rows(
        head_score_rows,
        top_ks=top_ks,
        select_features=select_features,
    )

    mention_features = ["selected_score", "selected_margin", "selected_hard"]
    sample_features = [
        "selection_value",
        "selected_score_mean",
        "selected_score_max",
        "selected_score_p90",
        "selected_margin_mean",
        "selected_margin_max",
        "selected_margin_p90",
        "selected_hard_rate",
    ]

    mention_auc_rows = []
    sample_auc_rows = []
    for selection in sorted({row["selection"] for row in mention_rows}):
        rows = [row for row in mention_rows if row["selection"] == selection]
        for auc_row in auc_rows(rows, mention_features):
            auc_row["selection"] = selection
            mention_auc_rows.append(auc_row)
    for selection in sorted({row["selection"] for row in sample_rows}):
        rows = [row for row in sample_rows if row["selection"] == selection]
        for auc_row in auc_rows(rows, sample_features):
            auc_row["selection"] = selection
            sample_auc_rows.append(auc_row)

    selected_head_counts = []
    grouped_counts = Counter((row["selection"], row["selected_head"]) for row in selection_rows)
    grouped_totals = Counter(row["selection"] for row in selection_rows)
    for (selection, head), count in sorted(grouped_counts.items()):
        selected_head_counts.append({
            "selection": selection,
            "selected_head": head,
            "n_samples": count,
            "sample_rate": count / grouped_totals[selection] if grouped_totals[selection] else None,
        })

    write_csv(os.path.join(args.output_dir, "adaptive_query_head_mentions.csv"), mention_rows)
    write_csv(os.path.join(args.output_dir, "adaptive_query_head_samples.csv"), sample_rows)
    write_csv(os.path.join(args.output_dir, "adaptive_query_head_selection_rows.csv"), selection_rows)
    write_csv(os.path.join(args.output_dir, "adaptive_query_head_mention_auc.csv"), mention_auc_rows)
    write_csv(os.path.join(args.output_dir, "adaptive_query_head_sample_auc.csv"), sample_auc_rows)
    write_csv(os.path.join(args.output_dir, "adaptive_query_head_selected_counts.csv"), selected_head_counts)

    summary = {
        "head_scores": args.head_scores,
        "n_head_score_rows": len(head_score_rows),
        "n_mention_rows": len(mention_rows),
        "n_sample_rows": len(sample_rows),
        "top_ks": top_ks,
        "select_features": select_features,
        "top_mention_auc": sorted(
            mention_auc_rows,
            key=lambda row: max(
                row["auroc_high_predicts_hallucinated"] or 0.0,
                row["auroc_low_predicts_hallucinated"] or 0.0,
            ),
            reverse=True,
        )[:20],
        "top_sample_auc": sorted(
            sample_auc_rows,
            key=lambda row: max(
                row["auroc_high_predicts_hallucinated"] or 0.0,
                row["auroc_low_predicts_hallucinated"] or 0.0,
            ),
            reverse=True,
        )[:20],
    }
    with open(os.path.join(args.output_dir, "adaptive_query_head_selection_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


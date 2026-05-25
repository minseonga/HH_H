import argparse
import csv
import json
import math
import os
from collections import defaultdict


def safe_float(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    return value


def safe_int(value, default=None):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def iter_jsonl(path):
    with open(path) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON in {path} at line {line_no}: {line[:120]!r}") from exc


def load_eval_sentences(path):
    with open(path) as f:
        data = json.load(f)
    return data.get("sentences", [])


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


def mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def percentile(values, q):
    values = sorted(value for value in values if value is not None)
    if not values:
        return None
    idx = int(round((len(values) - 1) * q / 100.0))
    return values[idx]


def two_cluster_threshold(values, max_iter=50):
    values = [value for value in values if value is not None]
    if not values:
        return None
    lo = percentile(values, 25)
    hi = percentile(values, 75)
    if lo is None or hi is None or abs(hi - lo) < 1e-12:
        return {
            "threshold": mean(values),
            "low_mean": mean(values),
            "high_mean": mean(values),
            "low_weight": 1.0,
            "high_weight": 0.0,
        }
    c0, c1 = lo, hi
    low = []
    high = []
    for _ in range(max_iter):
        low = []
        high = []
        midpoint = (c0 + c1) / 2.0
        for value in values:
            (low if value <= midpoint else high).append(value)
        if not low or not high:
            break
        n0 = sum(low) / len(low)
        n1 = sum(high) / len(high)
        if abs(n0 - c0) < 1e-12 and abs(n1 - c1) < 1e-12:
            break
        c0, c1 = n0, n1
    if c0 > c1:
        c0, c1 = c1, c0
        low, high = high, low
    return {
        "threshold": (c0 + c1) / 2.0,
        "low_mean": c0,
        "high_mean": c1,
        "low_weight": len(low) / len(values) if values else None,
        "high_weight": len(high) / len(values) if values else None,
    }


def auc_score(labels, scores):
    pairs = [(float(score), int(label)) for label, score in zip(labels, scores) if score is not None]
    if not pairs:
        return None
    positives = sum(label for _, label in pairs)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return None
    pairs.sort(key=lambda item: item[0])
    rank_sum = 0.0
    idx = 0
    while idx < len(pairs):
        j = idx + 1
        while j < len(pairs) and pairs[j][0] == pairs[idx][0]:
            j += 1
        avg_rank = (idx + 1 + j) / 2.0
        rank_sum += avg_rank * sum(label for _, label in pairs[idx:j])
        idx = j
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def node_pairs(sentence, key):
    pairs = []
    for item in sentence.get(key, []):
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            pairs.append((str(item[0]), str(item[1])))
    return pairs


def object_mentions(sentence):
    mentions = []
    nonhall_idxs = sentence.get("non_hallucination_idxs", [])
    hall_idxs = sentence.get("hallucination_idxs", [])
    for idx, (word, node_word) in zip(nonhall_idxs, node_pairs(sentence, "mscoco_non_hallucinated_words")):
        mentions.append({
            "word": word,
            "node_word": node_word,
            "word_idx": safe_int(idx, -1),
            "label": 0,
            "label_name": "grounded",
        })
    for idx, (word, node_word) in zip(hall_idxs, node_pairs(sentence, "mscoco_hallucinated_words")):
        mentions.append({
            "word": word,
            "node_word": node_word,
            "word_idx": safe_int(idx, -1),
            "label": 1,
            "label_name": "hallucinated",
        })
    mentions.sort(key=lambda item: (item.get("word_idx", -1) if item.get("word_idx", -1) >= 0 else 10**9))
    return mentions


def token_id_candidates_for_word(tokenizer, word):
    candidates = []
    seen = set()
    variants = [word, " " + word, word.lower(), " " + word.lower()]
    for text in variants:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        key = tuple(ids)
        if ids and key not in seen:
            candidates.append(ids)
            seen.add(key)
    return candidates


def find_next_subsequence(sequence, subsequence, start):
    if not subsequence:
        return None
    limit = len(sequence) - len(subsequence) + 1
    for idx in range(max(0, start), max(0, limit)):
        if sequence[idx:idx + len(subsequence)] == subsequence:
            return idx
    return None


def align_mentions(tokenizer, caption, mentions):
    caption_ids = tokenizer(caption, add_special_tokens=False)["input_ids"]
    cursor = 0
    aligned = []
    for mention in mentions:
        pos = None
        word_ids = []
        for candidate_ids in token_id_candidates_for_word(tokenizer, mention["word"]):
            pos = find_next_subsequence(caption_ids, candidate_ids, cursor)
            if pos is None and candidate_ids:
                pos = find_next_subsequence(caption_ids, candidate_ids[-1:], cursor)
            if pos is not None:
                word_ids = candidate_ids
                break
        if pos is None:
            continue
        aligned.append({
            **mention,
            "token_pos": int(pos),
            "token_len": int(max(1, len(word_ids))),
            "token_ids": " ".join(str(item) for item in word_ids),
        })
        cursor = max(cursor, pos + 1)
    return caption_ids, aligned


def load_tokenizer(path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, use_fast=False)


def band_name(layer):
    if layer < 12:
        return "early"
    if layer < 22:
        return "middle"
    return "late"


def add_head_record(agg, record, threshold):
    text_mass = safe_float(record.get("text_mass"))
    if text_mass is None:
        return
    layer = safe_int(record.get("layer"))
    if layer is None:
        head_key = str(record.get("head_key", ""))
        if ":" in head_key:
            layer = safe_int(head_key.split(":", 1)[0], -1)
        else:
            layer = -1
    band = band_name(layer)
    is_heavy = text_mass >= threshold
    agg["n_heads"] += 1
    agg["text_mass_sum"] += text_mass
    agg["text_mass_values"].append(text_mass)
    agg["text_mass_max"] = text_mass if agg["text_mass_max"] is None else max(agg["text_mass_max"], text_mass)
    if is_heavy:
        agg["text_heavy_count"] += 1
        agg["text_heavy_head_keys"].append(str(record.get("head_key", "")))
    for key in ("early", "middle", "late"):
        agg[f"{key}_n_heads"] += 1 if band == key else 0
        if band == key and is_heavy:
            agg[f"{key}_text_heavy_count"] += 1


def empty_step_agg():
    return {
        "n_heads": 0,
        "text_heavy_count": 0,
        "text_mass_sum": 0.0,
        "text_mass_values": [],
        "text_mass_max": None,
        "text_heavy_head_keys": [],
        "early_n_heads": 0,
        "early_text_heavy_count": 0,
        "middle_n_heads": 0,
        "middle_text_heavy_count": 0,
        "late_n_heads": 0,
        "late_text_heavy_count": 0,
    }


def finalize_step_row(question_id, token_pos, agg):
    n_heads = agg["n_heads"]
    row = {
        "question_id": question_id,
        "token_pos": int(token_pos),
        "n_heads": n_heads,
        "text_heavy_count": agg["text_heavy_count"],
        "text_heavy_rate": agg["text_heavy_count"] / n_heads if n_heads else None,
        "text_mass_mean": agg["text_mass_sum"] / n_heads if n_heads else None,
        "text_mass_max": agg["text_mass_max"],
        "text_mass_p90": percentile(agg["text_mass_values"], 90),
        "early_text_heavy_rate": (
            agg["early_text_heavy_count"] / agg["early_n_heads"] if agg["early_n_heads"] else None
        ),
        "middle_text_heavy_rate": (
            agg["middle_text_heavy_count"] / agg["middle_n_heads"] if agg["middle_n_heads"] else None
        ),
        "late_text_heavy_rate": (
            agg["late_text_heavy_count"] / agg["late_n_heads"] if agg["late_n_heads"] else None
        ),
        "midlate_text_heavy_rate": (
            (agg["middle_text_heavy_count"] + agg["late_text_heavy_count"])
            / (agg["middle_n_heads"] + agg["late_n_heads"])
            if (agg["middle_n_heads"] + agg["late_n_heads"]) else None
        ),
    }
    return row


def score_at_position(step_rows_by_qid, question_id, token_pos, feature, window):
    rows = step_rows_by_qid.get(str(question_id), {})
    values = []
    for pos in range(int(token_pos) - window, int(token_pos) + window + 1):
        row = rows.get(pos)
        if row is None:
            continue
        value = safe_float(row.get(feature))
        if value is not None:
            values.append(value)
    if not values:
        return None
    return max(values) if window > 0 else values[0]


def mention_feature_rows(sentences, tokenizer, step_rows_by_qid, features, window, mark_all_tokens):
    rows = []
    aligned_n = 0
    total_mentions = 0
    for sentence in sentences:
        qid = str(sentence.get("image_id"))
        mentions = object_mentions(sentence)
        total_mentions += len(mentions)
        _, aligned = align_mentions(tokenizer, sentence.get("caption", ""), mentions)
        aligned_n += len(aligned)
        for mention in aligned:
            positions = [mention["token_pos"]]
            if mark_all_tokens:
                positions = list(range(mention["token_pos"], mention["token_pos"] + mention["token_len"]))
            row = {
                "question_id": qid,
                "image": sentence.get("image", ""),
                "word": mention["word"],
                "node_word": mention["node_word"],
                "word_idx": mention.get("word_idx", ""),
                "label": mention["label"],
                "label_name": mention["label_name"],
                "token_pos": mention["token_pos"],
                "token_len": mention["token_len"],
                "token_ids": mention["token_ids"],
            }
            for feature in features:
                vals = [
                    score_at_position(step_rows_by_qid, qid, pos, feature, window)
                    for pos in positions
                ]
                vals = [value for value in vals if value is not None]
                row[feature] = max(vals) if vals else None
            rows.append(row)
    return rows, {"total_mentions": total_mentions, "aligned_mentions": aligned_n}


def step_label_rows(sentences, tokenizer, step_rows_by_qid):
    object_labels = defaultdict(dict)
    for sentence in sentences:
        qid = str(sentence.get("image_id"))
        _, aligned = align_mentions(tokenizer, sentence.get("caption", ""), object_mentions(sentence))
        for mention in aligned:
            current = object_labels[qid].get(mention["token_pos"])
            label = int(mention["label"])
            if current is None or label > current["label"]:
                object_labels[qid][mention["token_pos"]] = {
                    "label": label,
                    "label_name": mention["label_name"],
                    "word_idx": mention.get("word_idx", ""),
                    "word": mention["word"],
                    "node_word": mention["node_word"],
                }
    rows = []
    for qid, step_rows in step_rows_by_qid.items():
        for token_pos, row in step_rows.items():
            label_info = object_labels[qid].get(token_pos)
            out = dict(row)
            if label_info is None:
                out.update({
                    "is_object_step": 0,
                    "is_hallucinated_object_step": 0,
                    "is_grounded_object_step": 0,
                    "object_label": "non_object",
                    "word_idx": "",
                    "word": "",
                    "node_word": "",
                })
            else:
                out.update({
                    "is_object_step": 1,
                    "is_hallucinated_object_step": int(label_info["label"] == 1),
                    "is_grounded_object_step": int(label_info["label"] == 0),
                    "object_label": label_info["label_name"],
                    "word_idx": label_info.get("word_idx", ""),
                    "word": label_info["word"],
                    "node_word": label_info["node_word"],
                })
            rows.append(out)
    rows.sort(key=lambda item: (str(item["question_id"]), int(item["token_pos"])))
    return rows


def feature_contrast(rows, label_key, positive_value, negative_filter, features):
    selected = [row for row in rows if row.get(label_key) == positive_value or negative_filter(row)]
    labels = [1 if row.get(label_key) == positive_value else 0 for row in selected]
    output = []
    for feature in features:
        pos_vals = [safe_float(row.get(feature)) for row, label in zip(selected, labels) if label == 1]
        neg_vals = [safe_float(row.get(feature)) for row, label in zip(selected, labels) if label == 0]
        pos_vals = [value for value in pos_vals if value is not None]
        neg_vals = [value for value in neg_vals if value is not None]
        scores = [safe_float(row.get(feature)) for row in selected]
        output.append({
            "feature": feature,
            "n_pos": len(pos_vals),
            "n_neg": len(neg_vals),
            "pos_mean": mean(pos_vals),
            "neg_mean": mean(neg_vals),
            "pos_minus_neg": (
                mean(pos_vals) - mean(neg_vals) if pos_vals and neg_vals else None
            ),
            "auroc_high_predicts_pos": auc_score(labels, scores),
        })
    return output


def load_step_rows(diagnostics_jsonl, threshold, phase):
    values = []
    if threshold is None:
        for record in iter_jsonl(diagnostics_jsonl):
            if record.get("record_type") != "candidate_head":
                continue
            if phase != "all" and record.get("phase") != phase:
                continue
            value = safe_float(record.get("text_mass"))
            if value is not None:
                values.append(value)
        threshold_info = two_cluster_threshold(values)
        threshold = threshold_info["threshold"] if threshold_info else 0.35
    else:
        threshold_info = {"threshold": threshold}

    raw = defaultdict(lambda: defaultdict(empty_step_agg))
    first_step_by_qid = {}
    for record in iter_jsonl(diagnostics_jsonl):
        if record.get("record_type") != "candidate_head":
            continue
        if phase != "all" and record.get("phase") != phase:
            continue
        qid = str(record.get("question_id"))
        step_index = safe_int(record.get("step_index"))
        if not qid or step_index is None:
            continue
        first = first_step_by_qid.get(qid)
        if first is None or step_index < first:
            first_step_by_qid[qid] = step_index
        add_head_record(raw[qid][step_index], record, threshold)

    by_qid = defaultdict(dict)
    flat_rows = []
    for qid, steps in raw.items():
        first = first_step_by_qid.get(qid, 0)
        for step_index, agg in steps.items():
            token_pos = int(step_index - first)
            row = finalize_step_row(qid, token_pos, agg)
            row["step_index"] = int(step_index)
            row["first_decode_step_index"] = int(first)
            by_qid[qid][token_pos] = row
            flat_rows.append(row)
    flat_rows.sort(key=lambda item: (str(item["question_id"]), int(item["token_pos"])))
    return by_qid, flat_rows, threshold_info


def infer_eval_path(diagnostics_jsonl):
    parent = os.path.dirname(diagnostics_jsonl)
    candidate = os.path.join(parent, "captions_eval_results.json")
    return candidate if os.path.exists(candidate) else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics-jsonl", required=True)
    parser.add_argument("--eval-results", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer-path", default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--phase", default="decode", choices=["all", "decode", "prefill"])
    parser.add_argument("--text-mass-threshold", type=float, default=None)
    parser.add_argument("--window", type=int, default=0)
    parser.add_argument("--mark-all-object-tokens", action="store_true", default=False)
    args = parser.parse_args()

    eval_path = args.eval_results or infer_eval_path(args.diagnostics_jsonl)
    if not eval_path:
        raise ValueError("--eval-results is required when captions_eval_results.json is not next to diagnostics")

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = load_tokenizer(args.tokenizer_path)
    step_rows_by_qid, step_rows, threshold_info = load_step_rows(
        args.diagnostics_jsonl,
        threshold=args.text_mass_threshold,
        phase=args.phase,
    )
    sentences = load_eval_sentences(eval_path)

    features = [
        "text_heavy_rate",
        "text_heavy_count",
        "text_mass_mean",
        "text_mass_max",
        "text_mass_p90",
        "early_text_heavy_rate",
        "middle_text_heavy_rate",
        "late_text_heavy_rate",
        "midlate_text_heavy_rate",
    ]
    mention_rows, align_summary = mention_feature_rows(
        sentences,
        tokenizer,
        step_rows_by_qid,
        features,
        window=max(0, args.window),
        mark_all_tokens=args.mark_all_object_tokens,
    )
    labeled_step_rows = step_label_rows(sentences, tokenizer, step_rows_by_qid)

    mention_contrast = feature_contrast(
        mention_rows,
        label_key="label",
        positive_value=1,
        negative_filter=lambda row: row.get("label") == 0,
        features=features,
    )
    step_hall_vs_grounded = feature_contrast(
        labeled_step_rows,
        label_key="object_label",
        positive_value="hallucinated",
        negative_filter=lambda row: row.get("object_label") == "grounded",
        features=features,
    )
    step_hall_vs_nonhall = feature_contrast(
        labeled_step_rows,
        label_key="is_hallucinated_object_step",
        positive_value=1,
        negative_filter=lambda row: row.get("is_hallucinated_object_step") == 0,
        features=features,
    )

    write_csv(os.path.join(args.output_dir, "text_heavy_step_rows.csv"), labeled_step_rows)
    write_csv(os.path.join(args.output_dir, "text_heavy_object_mentions.csv"), mention_rows)
    write_csv(os.path.join(args.output_dir, "mention_hallucinated_vs_grounded_auc.csv"), mention_contrast)
    write_csv(os.path.join(args.output_dir, "step_hallucinated_vs_grounded_object_auc.csv"), step_hall_vs_grounded)
    write_csv(os.path.join(args.output_dir, "step_hallucinated_vs_nonhall_auc.csv"), step_hall_vs_nonhall)

    summary = {
        "diagnostics_jsonl": args.diagnostics_jsonl,
        "eval_results": eval_path,
        "output_dir": args.output_dir,
        "phase": args.phase,
        "window": args.window,
        "mark_all_object_tokens": args.mark_all_object_tokens,
        "text_mass_threshold": threshold_info,
        "n_step_rows": len(labeled_step_rows),
        "n_questions": len(step_rows_by_qid),
        "n_object_mentions": len(mention_rows),
        **align_summary,
        "mention_hallucinated_vs_grounded_auc": mention_contrast,
        "step_hallucinated_vs_grounded_object_auc": step_hall_vs_grounded,
        "step_hallucinated_vs_nonhall_auc": step_hall_vs_nonhall,
    }
    with open(os.path.join(args.output_dir, "text_heavy_object_alignment_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

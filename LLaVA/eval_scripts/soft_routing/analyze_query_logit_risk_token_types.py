import argparse
import csv
import json
import math
import os
import re
from collections import Counter, defaultdict

from eval_scripts.soft_routing.analyze_query_direction_chair_alignment import (
    image_id_key,
    prepare_mentions,
)
from eval_scripts.soft_routing.analyze_text_heavy_object_alignment import (
    load_eval_sentences,
    load_tokenizer,
    safe_float,
    safe_int,
)


FUNCTION_WORDS = {
    "a", "an", "the", "and", "or", "but", "of", "in", "on", "at", "to", "from",
    "with", "without", "by", "for", "as", "into", "over", "under", "near", "next",
    "is", "are", "was", "were", "be", "being", "been", "has", "have", "had",
    "do", "does", "did", "can", "could", "would", "should", "will", "may", "might",
    "this", "that", "these", "those", "there", "it", "its", "they", "their", "them",
    "he", "she", "his", "her", "him", "who", "which", "what", "where", "when",
    "one", "two", "some", "several", "many", "few", "both", "all", "also", "not",
    "while", "along", "through", "around", "behind", "front", "side", "top", "bottom",
}


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
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(values) / len(values) if values else None


def percentile(values, q):
    values = sorted(float(value) for value in values if value is not None and math.isfinite(float(value)))
    if not values:
        return None
    idx = int(round((len(values) - 1) * q / 100.0))
    return values[idx]


def decode_one(tokenizer, token_id):
    if token_id is None:
        return ""
    try:
        return tokenizer.decode([int(token_id)], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    except TypeError:
        return tokenizer.decode([int(token_id)], skip_special_tokens=False)


def lexical_category(token_text):
    if token_text is None:
        return "unknown"
    text = str(token_text)
    if text == "":
        return "empty"
    stripped = text.strip()
    if stripped == "":
        return "whitespace"
    if re.fullmatch(r"[^\w]+", stripped, flags=re.UNICODE):
        return "punctuation"
    lowered = stripped.lower()
    if lowered in FUNCTION_WORDS:
        return "function_word"
    if re.search(r"\d", stripped):
        return "number_or_mixed"
    if re.search(r"[A-Za-z]", stripped):
        return "content_word"
    return "other"


def build_sentence_index(eval_results, tokenizer):
    index = {}
    for sentence in load_eval_sentences(eval_results):
        key = image_id_key(sentence)
        if not key:
            continue
        caption = sentence.get("caption") or sentence.get("text") or ""
        caption_ids = tokenizer(caption, add_special_tokens=False)["input_ids"]
        span_by_pos = {}
        for mention in prepare_mentions(tokenizer, sentence, score_span="all"):
            token_pos = int(mention["token_pos"])
            token_len = int(mention.get("token_len", 1))
            label = str(mention.get("label_name", ""))
            for pos in range(token_pos, token_pos + token_len):
                if pos < 0 or pos >= len(caption_ids):
                    continue
                previous = span_by_pos.get(pos)
                if previous is None or label == "hallucinated":
                    span_by_pos[pos] = {
                        "label_name": label,
                        "word": mention.get("word", ""),
                        "node_word": mention.get("node_word", ""),
                        "mention_token_pos": token_pos,
                        "mention_token_len": token_len,
                    }
        item = {
            "sentence": sentence,
            "caption": caption,
            "caption_ids": caption_ids,
            "span_by_pos": span_by_pos,
        }
        index[key] = item
        qid = str(sentence.get("question_id", ""))
        if qid:
            index[qid] = item
    return index


def token_position(record, default_decode_offset):
    predicted = safe_int(record.get("predicted_token_pos"))
    if predicted is not None:
        return predicted
    step = safe_int(record.get("step_index"))
    if step is None:
        step = safe_int(record.get("call_index"))
    if step is None:
        return None
    if record.get("phase") == "decode":
        return step + int(default_decode_offset)
    return step


def build_rows(args, tokenizer):
    sentence_index = build_sentence_index(args.eval_results, tokenizer)
    rows = []
    for record in iter_jsonl(args.diagnostics_jsonl):
        if record.get("kind") != "query_logit_correction_logits":
            continue
        if not args.include_prefill and record.get("phase") != "decode":
            continue
        key = image_id_key(record)
        item = sentence_index.get(key) or sentence_index.get(str(record.get("question_id", "")))
        if item is None:
            continue
        pos = token_position(record, args.decode_token_offset)
        if pos is None:
            continue
        caption_ids = item["caption_ids"]
        caption_token_id = int(caption_ids[pos]) if 0 <= pos < len(caption_ids) else None
        top1_token_id = safe_int(record.get("top1_token_id"))
        top1_text = decode_one(tokenizer, top1_token_id)
        caption_text = decode_one(tokenizer, caption_token_id)
        span = item["span_by_pos"].get(pos)
        if span:
            position_category = f"object_{span['label_name']}"
            object_label = span["label_name"]
            object_word = span["word"]
            object_node_word = span["node_word"]
            is_object_first_token = int(pos == int(span["mention_token_pos"]))
        else:
            position_category = "non_object"
            object_label = ""
            object_word = ""
            object_node_word = ""
            is_object_first_token = 0
        risk = safe_float(record.get("risk"), 0.0)
        rows.append({
            "question_id": record.get("question_id", ""),
            "image": record.get("image", ""),
            "phase": record.get("phase", ""),
            "call_index": safe_int(record.get("call_index")),
            "step_index": safe_int(record.get("step_index")),
            "predicted_token_pos": pos,
            "risk": risk,
            "active_risk": safe_float(record.get("active_risk"), 0.0),
            "penalty_mean": safe_float(record.get("penalty_mean"), 0.0),
            "top1_token_id": top1_token_id,
            "top1_token": top1_text.replace("\n", "\\n"),
            "top1_lexical_category": lexical_category(top1_text),
            "caption_token_id": caption_token_id,
            "caption_token": caption_text.replace("\n", "\\n"),
            "top1_matches_caption_token": int(top1_token_id == caption_token_id) if caption_token_id is not None else 0,
            "position_category": position_category,
            "object_label": object_label,
            "object_word": object_word,
            "object_node_word": object_node_word,
            "is_object_first_token": is_object_first_token,
            "caption_len_tokens": len(caption_ids),
            "top1_logit_before": safe_float(record.get("top1_logit_before")),
            "top1_logit_after": safe_float(record.get("top1_logit_after")),
            "strength": safe_float(record.get("strength")),
            "top_k": safe_int(record.get("top_k")),
            "detector_aggregation": record.get("detector_aggregation", ""),
            "global_aggregation": record.get("global_aggregation", ""),
            "rank_weight": record.get("rank_weight", ""),
        })
    return rows


def summarize_group(name, rows, cutoff=None):
    n = len(rows)
    position_counts = Counter(row["position_category"] for row in rows)
    lexical_counts = Counter(row["top1_lexical_category"] for row in rows)
    object_n = position_counts["object_grounded"] + position_counts["object_hallucinated"]
    summary = {
        "group": name,
        "risk_cutoff": cutoff,
        "n": n,
        "mean_risk": mean(row["risk"] for row in rows),
        "p50_risk": percentile([row["risk"] for row in rows], 50),
        "p90_risk": percentile([row["risk"] for row in rows], 90),
        "top1_caption_token_match_rate": mean(row["top1_matches_caption_token"] for row in rows),
        "object_position_rate": object_n / n if n else None,
        "hallucinated_object_position_rate": position_counts["object_hallucinated"] / n if n else None,
        "grounded_object_position_rate": position_counts["object_grounded"] / n if n else None,
        "non_object_position_rate": position_counts["non_object"] / n if n else None,
        "object_first_token_rate": mean(row["is_object_first_token"] for row in rows),
        "top1_function_word_rate": lexical_counts["function_word"] / n if n else None,
        "top1_punctuation_rate": lexical_counts["punctuation"] / n if n else None,
        "top1_content_word_rate": lexical_counts["content_word"] / n if n else None,
        "top1_number_or_mixed_rate": lexical_counts["number_or_mixed"] / n if n else None,
    }
    for key, value in position_counts.items():
        summary[f"position_count:{key}"] = value
    for key, value in lexical_counts.items():
        summary[f"top1_lexical_count:{key}"] = value
    return summary


def build_summary(rows, quantiles):
    summary = [summarize_group("all", rows)]
    positive_rows = [row for row in rows if float(row["risk"] or 0.0) > 0.0]
    summary.append(summarize_group("risk_positive", positive_rows, 0.0))
    risks = [row["risk"] for row in rows]
    for q in quantiles:
        cutoff = percentile(risks, q)
        if cutoff is None:
            continue
        group_rows = [row for row in rows if float(row["risk"] or 0.0) >= cutoff]
        summary.append(summarize_group(f"risk_top_q{q:g}", group_rows, cutoff))
    return summary


def build_top_tokens(rows, quantiles, top_n):
    groups = {"all": rows}
    risks = [row["risk"] for row in rows]
    for q in quantiles:
        cutoff = percentile(risks, q)
        if cutoff is not None:
            groups[f"risk_top_q{q:g}"] = [row for row in rows if float(row["risk"] or 0.0) >= cutoff]
    output = []
    for group, group_rows in groups.items():
        by_token = defaultdict(list)
        for row in group_rows:
            by_token[(row["top1_token"], row["top1_token_id"], row["top1_lexical_category"])].append(row)
        ranked = sorted(by_token.items(), key=lambda item: (-len(item[1]), str(item[0][0])))[:top_n]
        for rank, ((token, token_id, lexical), token_rows) in enumerate(ranked, start=1):
            position_counts = Counter(row["position_category"] for row in token_rows)
            output.append({
                "group": group,
                "rank": rank,
                "top1_token": token,
                "top1_token_id": token_id,
                "top1_lexical_category": lexical,
                "count": len(token_rows),
                "rate_within_group": len(token_rows) / len(group_rows) if group_rows else None,
                "mean_risk": mean(row["risk"] for row in token_rows),
                "object_hallucinated_count": position_counts["object_hallucinated"],
                "object_grounded_count": position_counts["object_grounded"],
                "non_object_count": position_counts["non_object"],
            })
    return output


def parse_quantiles(text):
    output = []
    for item in str(text).replace(",", " ").split():
        if item.strip():
            output.append(float(item))
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics-jsonl", required=True)
    parser.add_argument("--eval-results", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--risk-quantiles", default="50 75 90 95")
    parser.add_argument("--top-n-tokens", type=int, default=50)
    parser.add_argument("--decode-token-offset", type=int, default=1)
    parser.add_argument("--include-prefill", action="store_true", default=False)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = load_tokenizer(args.model_path)
    rows = build_rows(args, tokenizer)
    quantiles = parse_quantiles(args.risk_quantiles)
    summary = build_summary(rows, quantiles)
    top_tokens = build_top_tokens(rows, quantiles, args.top_n_tokens)

    write_csv(os.path.join(args.output_dir, "query_logit_risk_token_rows.csv"), rows)
    write_csv(os.path.join(args.output_dir, "query_logit_risk_token_summary.csv"), summary)
    write_csv(os.path.join(args.output_dir, "query_logit_risk_top_tokens.csv"), top_tokens)
    with open(os.path.join(args.output_dir, "query_logit_risk_token_summary.json"), "w") as f:
        json.dump({
            "diagnostics_jsonl": args.diagnostics_jsonl,
            "eval_results": args.eval_results,
            "n_rows": len(rows),
            "risk_quantiles": quantiles,
            "summary": summary,
        }, f, indent=2)
    print("[summary] query-logit risk token types")
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()

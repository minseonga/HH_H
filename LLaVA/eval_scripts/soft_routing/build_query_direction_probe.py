import argparse
import csv
import json
import os
from collections import Counter, defaultdict

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from eval_scripts.soft_routing.analyze_object_retention_steps import (
    build_prompt_inputs,
    configure_model,
    load_sentences,
    one_step,
    select_rows,
)


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
    return float(np.mean(values)) if values else None


def l2_normalize(x, eps=1e-12):
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(denom, eps)


def normalize_queries(x, mode):
    if mode == "l2":
        return l2_normalize(x)
    if mode == "none":
        return x
    raise ValueError(f"Unknown query normalization mode: {mode}")


def label_family(label):
    if str(label).startswith("hallucinated_object"):
        return "hallucinated"
    if label in {"lost_grounded", "kept_grounded"}:
        return label
    return str(label)


def split_images(step_rows, test_fraction, seed):
    images = sorted({row["image_id"] for row in step_rows})
    rng = np.random.default_rng(seed)
    rng.shuffle(images)
    if len(images) <= 1 or test_fraction <= 0:
        test_images = set()
    else:
        n_test = int(round(len(images) * test_fraction))
        n_test = min(max(n_test, 1), len(images) - 1)
        test_images = set(images[:n_test])
    return {
        int(row["step_id"]): ("test" if row["image_id"] in test_images else "train")
        for row in step_rows
    }


def empty_calibration():
    return {
        "layers": [],
        "heads": [],
        "directions": [],
        "threshold_midpoint": [],
        "train_mean_positive": [],
        "train_mean_negative": [],
        "test_auroc": [],
        "test_auprc": [],
    }


def collect_query_vectors(args, tokenizer, model, image_processor, selected):
    vectors = []
    layers = []
    heads = []
    step_ids = []
    step_rows = []

    if args.surface == "residual":
        vector_key = "residual"
        diagnostics_attr = "residual_diagnostics"
        model.config.record_residual_diagnostics = True
        model.config.residual_record_min_layer = args.min_layer
        model.config.residual_record_max_layer = args.max_layer
        model.config.residual_record_batch_index = 0
    elif args.surface == "head_output":
        vector_key = "head_output"
        diagnostics_attr = "head_output_diagnostics"
        model.config.record_head_output_diagnostics = True
        model.config.head_output_record_all_heads = True
        model.config.head_output_record_min_layer = args.min_layer
        model.config.head_output_record_max_layer = args.max_layer
        model.config.head_output_record_batch_index = 0
    else:
        vector_key = "query"
        diagnostics_attr = "query_diagnostics"
        model.config.record_query_diagnostics = True
        model.config.query_record_all_heads = True
        model.config.query_record_min_layer = args.min_layer
        model.config.query_record_max_layer = args.max_layer
        model.config.query_record_batch_index = 0

    for step_id, row in enumerate(tqdm(selected, desc=f"{args.surface} steps")):
        prompt_ids, image_tensor, image_size = build_prompt_inputs(
            row,
            args.image_folder,
            tokenizer,
            image_processor,
            model.config,
            args.conv_mode,
        )
        prompt_ids = prompt_ids.to(device="cuda", non_blocking=True)
        image_tensor = image_tensor.to(dtype=torch.float16, device="cuda", non_blocking=True)
        prefix_ids = row["probe_caption_ids"][:row["target_token_pos"]]
        target_token_id = int(row["target_token_id"])

        setattr(model.config, diagnostics_attr, [])
        one_step(
            model,
            tokenizer,
            prompt_ids,
            prefix_ids,
            image_tensor,
            image_size,
            target_token_id,
            "none",
            record=False,
        )

        records = list(getattr(model.config, diagnostics_attr, []) or [])
        family = label_family(row["label"])
        binary_label = 1 if family == "hallucinated" else 0
        step_rows.append({
            "step_id": step_id,
            "image_id": row["image_id"],
            "image": row["image"],
            "label": row["label"],
            "label_family": family,
            "binary_label_hallucinated": binary_label,
            "caption_source": row["caption_source"],
            "object_node": row["object_node"],
            "object_word": row["object_word"],
            "target_token": tokenizer.decode([target_token_id]),
            "target_token_id": target_token_id,
            "target_token_pos": int(row["target_token_pos"]),
            "surface": args.surface,
            "num_vector_records": len(records),
            "num_query_records": len(records) if args.surface == "query" else 0,
            "num_head_output_records": len(records) if args.surface == "head_output" else 0,
            "num_residual_records": len(records) if args.surface == "residual" else 0,
        })

        for record in records:
            vector = record[vector_key]
            if isinstance(vector, torch.Tensor):
                vector = vector.numpy()
            vectors.append(np.asarray(vector, dtype=np.float32))
            layers.append(int(record["layer"]))
            heads.append(int(record["head"]))
            step_ids.append(step_id)

    if args.surface == "residual":
        model.config.record_residual_diagnostics = False
        model.config.residual_diagnostics = None
    elif args.surface == "head_output":
        model.config.record_head_output_diagnostics = False
        model.config.head_output_diagnostics = None
    else:
        model.config.record_query_diagnostics = False
        model.config.query_diagnostics = None
    return {
        "vectors": np.stack(vectors, axis=0) if vectors else np.zeros((0, 0), dtype=np.float32),
        "layers": np.asarray(layers, dtype=np.int32),
        "heads": np.asarray(heads, dtype=np.int32),
        "step_ids": np.asarray(step_ids, dtype=np.int32),
        "step_rows": step_rows,
    }


def calibrate_directions(query_data, step_rows, args):
    vectors = query_data["vectors"]
    layers = query_data["layers"]
    heads = query_data["heads"]
    step_ids = query_data["step_ids"]
    if vectors.size == 0:
        return [], empty_calibration()

    step_labels = {
        int(row["step_id"]): int(row["binary_label_hallucinated"])
        for row in step_rows
    }
    step_families = {
        int(row["step_id"]): row["label_family"]
        for row in step_rows
    }
    step_split = split_images(step_rows, args.test_fraction, args.seed)

    labels = np.asarray([step_labels[int(step_id)] for step_id in step_ids], dtype=np.int32)
    splits = np.asarray([step_split[int(step_id)] for step_id in step_ids])
    families = np.asarray([step_families[int(step_id)] for step_id in step_ids])

    rows = []
    calibration = empty_calibration()
    for layer in sorted(set(layers.tolist())):
        for head in sorted(set(heads[layers == layer].tolist())):
            mask = (layers == layer) & (heads == head)
            x = normalize_queries(vectors[mask], args.query_normalization)
            y = labels[mask]
            split = splits[mask]
            family = families[mask]

            train = split == "train"
            test = split == "test"
            train_pos = train & (y == 1)
            train_neg = train & (y == 0)
            test_pos = test & (y == 1)
            test_neg = test & (y == 0)
            if train_pos.sum() < args.min_train_per_class or train_neg.sum() < args.min_train_per_class:
                continue
            if test_pos.sum() < args.min_test_per_class or test_neg.sum() < args.min_test_per_class:
                continue

            direction = x[train_pos].mean(axis=0) - x[train_neg].mean(axis=0)
            direction_norm = float(np.linalg.norm(direction))
            if direction_norm <= 1e-12:
                continue
            direction = direction / direction_norm

            train_scores = x[train] @ direction
            test_scores = x[test] @ direction
            train_labels = y[train]
            test_labels = y[test]
            test_auc = float(roc_auc_score(test_labels, test_scores))
            test_ap = float(average_precision_score(test_labels, test_scores))
            train_auc = float(roc_auc_score(train_labels, train_scores))
            train_ap = float(average_precision_score(train_labels, train_scores))
            train_mean_pos = float(np.mean(train_scores[train_labels == 1]))
            train_mean_neg = float(np.mean(train_scores[train_labels == 0]))
            test_mean_pos = float(np.mean(test_scores[test_labels == 1]))
            test_mean_neg = float(np.mean(test_scores[test_labels == 0]))
            threshold_midpoint = 0.5 * (train_mean_pos + train_mean_neg)

            row = {
                "layer": int(layer),
                "head": int(head),
                "head_key": f"{int(layer)}:{int(head)}",
                "surface": args.surface,
                "query_normalization": args.query_normalization,
                "n_train": int(train.sum()),
                "n_train_positive": int(train_pos.sum()),
                "n_train_negative": int(train_neg.sum()),
                "n_test": int(test.sum()),
                "n_test_positive": int(test_pos.sum()),
                "n_test_negative": int(test_neg.sum()),
                "train_auroc_high_predicts_hallucinated": train_auc,
                "train_auprc_high_predicts_hallucinated": train_ap,
                "test_auroc_high_predicts_hallucinated": test_auc,
                "test_auroc_abs": max(test_auc, 1.0 - test_auc),
                "test_direction": "high_predicts_hallucinated" if test_auc >= 0.5 else "low_predicts_hallucinated",
                "test_auprc_high_predicts_hallucinated": test_ap,
                "direction_norm_before_unit": direction_norm,
                "threshold_midpoint": threshold_midpoint,
                "train_mean_positive_score": train_mean_pos,
                "train_mean_negative_score": train_mean_neg,
                "test_mean_positive_score": test_mean_pos,
                "test_mean_negative_score": test_mean_neg,
                "test_score_gap_pos_minus_neg": test_mean_pos - test_mean_neg,
                "test_mean_kept_grounded_score": mean(test_scores[family[test] == "kept_grounded"].tolist()),
                "test_mean_lost_grounded_score": mean(test_scores[family[test] == "lost_grounded"].tolist()),
                "test_mean_hallucinated_score": mean(test_scores[family[test] == "hallucinated"].tolist()),
            }
            rows.append(row)

            calibration["layers"].append(int(layer))
            calibration["heads"].append(int(head))
            calibration["directions"].append(direction.astype(np.float32))
            calibration["threshold_midpoint"].append(threshold_midpoint)
            calibration["train_mean_positive"].append(train_mean_pos)
            calibration["train_mean_negative"].append(train_mean_neg)
            calibration["test_auroc"].append(test_auc)
            calibration["test_auprc"].append(test_ap)

    rows.sort(key=lambda item: item["test_auroc_abs"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank_by_abs_auroc"] = rank
    return rows, calibration


def layer_summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["layer"])].append(row)
    output = []
    for layer, items in sorted(grouped.items()):
        best = max(items, key=lambda item: item["test_auroc_abs"])
        output.append({
            "layer": layer,
            "n_heads": len(items),
            "mean_test_auroc": mean([item["test_auroc_high_predicts_hallucinated"] for item in items]),
            "mean_test_auroc_abs": mean([item["test_auroc_abs"] for item in items]),
            "max_test_auroc_abs": best["test_auroc_abs"],
            "best_head": best["head_key"],
            "heads_abs_auc_ge_0p60": sum(1 for item in items if item["test_auroc_abs"] >= 0.60),
            "heads_abs_auc_ge_0p65": sum(1 for item in items if item["test_auroc_abs"] >= 0.65),
            "heads_abs_auc_ge_0p70": sum(1 for item in items if item["test_auroc_abs"] >= 0.70),
        })
    return output


def save_calibration(path, calibration):
    if not calibration["directions"]:
        np.savez(
            path,
            layers=np.asarray([], dtype=np.int32),
            heads=np.asarray([], dtype=np.int32),
            directions=np.zeros((0, 0), dtype=np.float32),
        )
        return
    np.savez(
        path,
        layers=np.asarray(calibration["layers"], dtype=np.int32),
        heads=np.asarray(calibration["heads"], dtype=np.int32),
        directions=np.stack(calibration["directions"], axis=0).astype(np.float32),
        threshold_midpoint=np.asarray(calibration["threshold_midpoint"], dtype=np.float32),
        train_mean_positive=np.asarray(calibration["train_mean_positive"], dtype=np.float32),
        train_mean_negative=np.asarray(calibration["train_mean_negative"], dtype=np.float32),
        test_auroc=np.asarray(calibration["test_auroc"], dtype=np.float32),
        test_auprc=np.asarray(calibration["test_auprc"], dtype=np.float32),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hard-results", required=True)
    parser.add_argument("--soft-results", required=True)
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prior-path", default="")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--model-path", default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--conv-mode", default="vicuna_v1")
    parser.add_argument("--max-per-label", type=int, default=100)
    parser.add_argument("--hallucinated-source", type=str, default="both", choices=["soft", "hard", "both"])
    parser.add_argument("--adhh-threshold", type=float, default=0.4)
    parser.add_argument("--soft-gamma", type=float, default=0.75)
    parser.add_argument("--soft-temperature", type=float, default=0.05)
    parser.add_argument("--min-layer", type=int, default=13)
    parser.add_argument("--max-layer", type=int, default=31)
    parser.add_argument("--surface", choices=["query", "head_output", "residual"], default="query")
    parser.add_argument("--query-normalization", choices=["none", "l2"], default="l2")
    parser.add_argument("--test-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-train-per-class", type=int, default=2)
    parser.add_argument("--min-test-per-class", type=int, default=1)
    parser.add_argument("--save-query-vectors", action="store_true", default=False)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    disable_torch_init()
    model_name = get_model_name_from_path(os.path.expanduser(args.model_path))
    tokenizer, model, image_processor, _ = load_pretrained_model(args.model_path, args.model_base, model_name)
    heads, _, prior_source = configure_model(
        model,
        args.model_path,
        args.prior_path,
        args.top_k,
        args.adhh_threshold,
        args.soft_gamma,
        args.soft_temperature,
    )

    hard_by_id = load_sentences(args.hard_results)
    soft_by_id = load_sentences(args.soft_results)
    selected = select_rows(hard_by_id, soft_by_id, tokenizer, args.max_per_label, args.hallucinated_source)
    query_data = collect_query_vectors(args, tokenizer, model, image_processor, selected)
    step_rows = query_data["step_rows"]
    direction_rows, calibration = calibrate_directions(query_data, step_rows, args)
    layer_rows = layer_summary(direction_rows)

    write_csv(os.path.join(args.output_dir, "query_direction_steps.csv"), step_rows)
    write_csv(os.path.join(args.output_dir, "query_direction_auc.csv"), direction_rows)
    write_csv(os.path.join(args.output_dir, "query_direction_layer_summary.csv"), layer_rows)
    save_calibration(os.path.join(args.output_dir, "query_direction_calibration.npz"), calibration)

    if args.save_query_vectors:
        np.savez_compressed(
            os.path.join(args.output_dir, "query_vectors.npz"),
        vectors=query_data["vectors"].astype(np.float32),
            layers=query_data["layers"],
            heads=query_data["heads"],
            step_ids=query_data["step_ids"],
        )

    summary = {
        "num_steps": len(step_rows),
        "num_vectors": int(query_data["vectors"].shape[0]),
        "num_query_vectors": int(query_data["vectors"].shape[0]),
        "head_dim": int(query_data["vectors"].shape[1]) if query_data["vectors"].ndim == 2 and query_data["vectors"].shape[0] else 0,
        "surface": args.surface,
        "label_counts": dict(Counter(row["label"] for row in step_rows)),
        "label_family_counts": dict(Counter(row["label_family"] for row in step_rows)),
        "min_layer": args.min_layer,
        "max_layer": args.max_layer,
        "query_normalization": args.query_normalization,
        "test_fraction": args.test_fraction,
        "seed": args.seed,
        "prior_source": prior_source,
        "heads": heads,
        "top_directions": direction_rows[:20],
    }
    with open(os.path.join(args.output_dir, "query_direction_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({
        "num_steps": summary["num_steps"],
        "num_query_vectors": summary["num_query_vectors"],
        "label_family_counts": summary["label_family_counts"],
        "top_directions": direction_rows[:10],
    }, indent=2))


if __name__ == "__main__":
    main()

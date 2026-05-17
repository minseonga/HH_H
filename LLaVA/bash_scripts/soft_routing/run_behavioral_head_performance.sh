#!/bin/bash

set -euo pipefail

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-True}"

GPU_ID="${GPU_ID:-0}"
DATASET="${DATASET:-coco}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
ANNOTATION_DIR="${ANNOTATION_DIR:-/home/kms/data/images/mscoco/annotations}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
CAPTION_FILE_PATH="${CAPTION_FILE_PATH:-${ANNOTATION_DIR}/captions_val2014.json}"

NUM_SAMPLES="${NUM_SAMPLES:-500}"
SEED="${SEED:-42}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
ADHH_THRESHOLD="${ADHH_THRESHOLD:-0.4}"
SOFT_TEMPERATURE="${SOFT_TEMPERATURE:-0.05}"
TOP_K="${TOP_K:-20}"
FORCE="${FORCE:-0}"

BASE_RESULT_PATH="${BASE_RESULT_PATH:-./results/${DATASET}/soft_routing_smoke_n${NUM_SAMPLES}_seed${SEED}_tau${ADHH_THRESHOLD}_T${SOFT_TEMPERATURE}}"
HEAD_DIR="${HEAD_DIR:-${BASE_RESULT_PATH}/behavioral_head_overlap_prefill_n500_minlayer13}"
PERF_DIR="${PERF_DIR:-${BASE_RESULT_PATH}/behavioral_head_performance_$(basename "${HEAD_DIR}")}"
LOG_DIR="${LOG_DIR:-./logs/soft_routing}"

DEFAULT_ATTENTION_HEAD_PATH="${BASE_RESULT_PATH}/fixed_adhh20_score_prior/attribution_result.json"
if [ ! -f "${DEFAULT_ATTENTION_HEAD_PATH}" ]; then
    DEFAULT_ATTENTION_HEAD_PATH="./results/coco/llava_3000/identify_attention_head/attribution_result.json"
fi
ATTENTION_HEAD_PATH="${ATTENTION_HEAD_PATH:-${DEFAULT_ATTENTION_HEAD_PATH}}"

SELECTORS="${SELECTORS:-adhh_reference text_ratio image_entropy text_ratio_x_image_entropy layer_norm_image_entropy layer_norm_text_ratio_x_image_entropy}"

mkdir -p "${PERF_DIR}" "${LOG_DIR}"

echo "[info] head dir: ${HEAD_DIR}"
echo "[info] performance dir: ${PERF_DIR}"
echo "[info] selectors: ${SELECTORS}"
echo "[info] AD-HH reference: ${ATTENTION_HEAD_PATH}"

head_path_for_selector() {
    local selector="$1"
    if [ "${selector}" = "adhh_reference" ]; then
        echo "${ATTENTION_HEAD_PATH}"
    else
        echo "${HEAD_DIR}/${selector}_top${TOP_K}.json"
    fi
}

for selector in ${SELECTORS}; do
    head_path="$(head_path_for_selector "${selector}")"
    if [ ! -f "${head_path}" ]; then
        echo "[warn] missing head file for ${selector}: ${head_path}"
        continue
    fi

    result_dir="${PERF_DIR}/${selector}_hard_tau${ADHH_THRESHOLD}"
    answers_file="${result_dir}/captions.jsonl"
    eval_file="${result_dir}/captions_eval_results.json"
    mkdir -p "${result_dir}"

    if [ "${FORCE}" != "1" ] && [ -f "${eval_file}" ]; then
        echo "[skip] ${selector}: ${eval_file}"
        continue
    fi

    echo "[run] ${selector}: hard-zero AD-HH action with heads ${head_path}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python -m eval_scripts.eval_caption_adhh \
        --model-path "${MODEL_PATH}" \
        --image-folder "${IMAGE_FOLDER}" \
        --caption_file_path "${CAPTION_FILE_PATH}" \
        --answers-file "${answers_file}" \
        --dataset "${DATASET}" \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --num_samples "${NUM_SAMPLES}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --seed "${SEED}" \
        --adaptive_deactivate \
        --adhh_threshold "${ADHH_THRESHOLD}" \
        --attention_head_path "${head_path}" \
        --head_prior_mode uniform \
        --top_k "${TOP_K}" \
        2>&1 | tee "${LOG_DIR}/behavioral_perf_${selector}_hard_tau${ADHH_THRESHOLD}.log"

    python eval_scripts/eval_utils/eval_chair.py \
        --annotation-dir "${ANNOTATION_DIR}" \
        --answers-file "${answers_file}" \
        --caption_file "$(basename "${CAPTION_FILE_PATH}")"
done

python - <<PY
import csv
import json
import os

perf_dir = "${PERF_DIR}"
rows = []
for selector in "${SELECTORS}".split():
    path = os.path.join(perf_dir, f"{selector}_hard_tau${ADHH_THRESHOLD}", "captions_eval_results.json")
    if not os.path.exists(path):
        continue
    metrics = json.load(open(path))["overall_metrics"]
    rows.append({
        "selector": selector,
        "CHAIRs": metrics.get("CHAIRs"),
        "CHAIRi": metrics.get("CHAIRi"),
        "Bleu1": (metrics.get("Bleu") or [None, None, None, None])[0],
        "Bleu2": (metrics.get("Bleu") or [None, None, None, None])[1],
        "Bleu3": (metrics.get("Bleu") or [None, None, None, None])[2],
        "Bleu4": (metrics.get("Bleu") or [None, None, None, None])[3],
        "avg_caption_length": metrics.get("avg_caption_length"),
    })

out = os.path.join(perf_dir, "behavioral_head_performance_summary.csv")
if rows:
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
print(out)
PY

echo "[summary] behavioral head performance"
if [ -f "${PERF_DIR}/behavioral_head_performance_summary.csv" ]; then
    column -s, -t "${PERF_DIR}/behavioral_head_performance_summary.csv"
fi

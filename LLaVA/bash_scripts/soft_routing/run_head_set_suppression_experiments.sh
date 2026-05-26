#!/bin/bash

set -euo pipefail

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-True}"

GPU_ID="${GPU_ID:-0}"
DATASET="${DATASET:-coco}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
ANNOTATION_DIR="${ANNOTATION_DIR:-/home/kms/data/images/mscoco/annotations}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
CAPTION_FILE_PATH="${CAPTION_FILE_PATH:-${ANNOTATION_DIR}/captions_val2014.json}"

NUM_SAMPLES="${NUM_SAMPLES:-0}"
SEED="${SEED:-42}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
ADHH_THRESHOLD="${ADHH_THRESHOLD:-0.4}"
SOFT_GAMMA="${SOFT_GAMMA:-0.5}"
SOFT_TEMPERATURE="${SOFT_TEMPERATURE:-0.05}"
TOP_K="${TOP_K:-20}"
FORCE="${FORCE:-0}"

BASE_RESULT_PATH="${BASE_RESULT_PATH:-./results/${DATASET}/soft_routing_smoke_n500_seed42_tau0.4_T0.05}"
OUTPUT_DIR="${OUTPUT_DIR:-${BASE_RESULT_PATH}/head_set_suppression_n${NUM_SAMPLES}}"
LOG_DIR="${LOG_DIR:-./logs/soft_routing}"
GREEDY_EVAL="${GREEDY_EVAL:-${BASE_RESULT_PATH}/greedy/captions_eval_results.json}"
INCLUDE_IMAGE_IDS_PATH="${INCLUDE_IMAGE_IDS_PATH:-}"
EXCLUDE_IMAGE_IDS_PATH="${EXCLUDE_IMAGE_IDS_PATH:-}"
CUSTOM_HEAD_PATHS="${CUSTOM_HEAD_PATHS:-}"
RUN_METHODS="${RUN_METHODS:-greedy adhh_hard adhh_soft custom_hard custom_soft}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

echo "[info] output dir: ${OUTPUT_DIR}"
echo "[info] include ids: ${INCLUDE_IMAGE_IDS_PATH:-none}"
echo "[info] exclude ids: ${EXCLUDE_IMAGE_IDS_PATH:-none}"
echo "[info] custom head paths: ${CUSTOM_HEAD_PATHS:-none}"
echo "[info] run methods: ${RUN_METHODS}"
echo "[info] num samples: ${NUM_SAMPLES}"

include_args=()
if [ -n "${INCLUDE_IMAGE_IDS_PATH}" ]; then
    include_args=(--include_image_ids_path "${INCLUDE_IMAGE_IDS_PATH}")
fi
exclude_args=()
if [ -n "${EXCLUDE_IMAGE_IDS_PATH}" ]; then
    exclude_args=(--exclude_image_ids_path "${EXCLUDE_IMAGE_IDS_PATH}")
fi

should_run() {
    local tag="$1"
    for method in ${RUN_METHODS}; do
        if [ "${method}" = "${tag}" ]; then
            return 0
        fi
    done
    return 1
}

eval_chair() {
    local answers_file="$1"
    python eval_scripts/eval_utils/eval_chair.py \
        --annotation-dir "${ANNOTATION_DIR}" \
        --answers-file "${answers_file}" \
        --caption_file "$(basename "${CAPTION_FILE_PATH}")"
}

run_eval() {
    local tag="$1"
    shift
    local result_dir="${OUTPUT_DIR}/${tag}"
    local answers_file="${result_dir}/captions.jsonl"
    local eval_file="${result_dir}/captions_eval_results.json"
    mkdir -p "${result_dir}"
    if [ "${FORCE}" != "1" ] && [ -f "${eval_file}" ]; then
        echo "[skip] ${tag}: ${eval_file}"
        return
    fi
    echo "[run] ${tag}"
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
        "${include_args[@]}" \
        "${exclude_args[@]}" \
        "$@" \
        2>&1 | tee "${LOG_DIR}/head_set_${tag}.log"
    eval_chair "${answers_file}"
}

if should_run "greedy"; then
    run_eval "greedy"
else
    echo "[skip] greedy: not in RUN_METHODS"
fi

if should_run "adhh_hard"; then
    run_eval "adhh_hard" \
        --adaptive_deactivate \
        --adhh_threshold "${ADHH_THRESHOLD}" \
        --head_prior_mode uniform \
        --top_k "${TOP_K}"
else
    echo "[skip] adhh_hard: not in RUN_METHODS"
fi

if should_run "adhh_soft"; then
    run_eval "adhh_soft" \
        --soft_deactivate \
        --adhh_threshold "${ADHH_THRESHOLD}" \
        --soft_gamma "${SOFT_GAMMA}" \
        --soft_temperature "${SOFT_TEMPERATURE}" \
        --head_prior_mode uniform \
        --top_k "${TOP_K}"
else
    echo "[skip] adhh_soft: not in RUN_METHODS"
fi

for head_path in ${CUSTOM_HEAD_PATHS}; do
    if [ ! -f "${head_path}" ]; then
        echo "[error] missing custom head path: ${head_path}" >&2
        exit 1
    fi
    head_tag="$(basename "${head_path}" .json)"
    if should_run "custom_hard"; then
        run_eval "${head_tag}_hard" \
            --adaptive_deactivate \
            --adhh_threshold "${ADHH_THRESHOLD}" \
            --attention_head_path "${head_path}" \
            --head_prior_mode uniform \
            --top_k "${TOP_K}"
    else
        echo "[skip] ${head_tag}_hard: custom_hard not in RUN_METHODS"
    fi
    if should_run "custom_soft"; then
        run_eval "${head_tag}_soft" \
            --soft_deactivate \
            --adhh_threshold "${ADHH_THRESHOLD}" \
            --soft_gamma "${SOFT_GAMMA}" \
            --soft_temperature "${SOFT_TEMPERATURE}" \
            --attention_head_path "${head_path}" \
            --head_prior_mode uniform \
            --top_k "${TOP_K}"
    else
        echo "[skip] ${head_tag}_soft: custom_soft not in RUN_METHODS"
    fi
done

python - <<PY
import csv
import glob
import json
import os

rows = []
for path in sorted(glob.glob("${OUTPUT_DIR}/*/captions_eval_results.json")):
    tag = os.path.basename(os.path.dirname(path))
    metrics = json.load(open(path))["overall_metrics"]
    bleu = metrics.get("Bleu") or [None, None, None, None]
    rows.append({
        "method": tag,
        "CHAIRs": metrics.get("CHAIRs"),
        "CHAIRi": metrics.get("CHAIRi"),
        "Bleu1": bleu[0],
        "Bleu2": bleu[1],
        "Bleu3": bleu[2],
        "Bleu4": bleu[3],
        "avg_caption_length": metrics.get("avg_caption_length"),
    })

out = "${OUTPUT_DIR}/head_set_suppression_summary.csv"
if rows:
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
print(out)
PY

if [ -f "${GREEDY_EVAL}" ]; then
    mapfile -t eval_files < <(find "${OUTPUT_DIR}" -mindepth 2 -maxdepth 2 -name captions_eval_results.json | sort)
    if [ "${#eval_files[@]}" -gt 0 ]; then
        python -m eval_scripts.soft_routing.analyze_hallucination_position \
            --eval-results "${eval_files[@]}" \
            --names $(for path in "${eval_files[@]}"; do basename "$(dirname "${path}")"; done) \
            --match-common-image-ids \
            --output-dir "${OUTPUT_DIR}/position_summary"
    fi
fi

echo "[summary] head-set suppression metrics"
if [ -f "${OUTPUT_DIR}/head_set_suppression_summary.csv" ]; then
    column -s, -t "${OUTPUT_DIR}/head_set_suppression_summary.csv"
fi

#!/bin/bash

set -euo pipefail

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-True}"

GPU_ID="${GPU_ID:-0}"
DATASET="${DATASET:-coco}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
NUM_SAMPLES="${NUM_SAMPLES:-500}"
SEED="${SEED:-42}"
ADHH_THRESHOLD="${ADHH_THRESHOLD:-0.4}"
SOFT_TEMPERATURE="${SOFT_TEMPERATURE:-0.05}"
SOFT_GAMMA="${SOFT_GAMMA:-0.75}"
TOP_K="${TOP_K:-20}"
MAX_PER_LABEL="${MAX_PER_LABEL:-10}"
HALLUCINATED_SOURCE="${HALLUCINATED_SOURCE:-both}"
RESUME="${RESUME:-1}"

BASE_RESULT_PATH="${BASE_RESULT_PATH:-./results/${DATASET}/soft_routing_smoke_n${NUM_SAMPLES}_seed${SEED}_tau${ADHH_THRESHOLD}_T${SOFT_TEMPERATURE}}"
HARD_RESULTS="${HARD_RESULTS:-${BASE_RESULT_PATH}/hard_tau${ADHH_THRESHOLD}/captions_eval_results.json}"
SOFT_RESULTS="${SOFT_RESULTS:-${BASE_RESULT_PATH}/soft_gamma${SOFT_GAMMA}/captions_eval_results.json}"
PRIOR_PATH="${PRIOR_PATH:-${BASE_RESULT_PATH}/fixed_adhh20_score_prior/attribution_result.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${BASE_RESULT_PATH}/all_head_feature_diagnostics_max${MAX_PER_LABEL}}"
LOG_DIR="${LOG_DIR:-./logs/soft_routing}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

if [ ! -f "${HARD_RESULTS}" ]; then
    echo "[error] missing hard results: ${HARD_RESULTS}" >&2
    exit 1
fi
if [ ! -f "${SOFT_RESULTS}" ]; then
    echo "[error] missing soft results: ${SOFT_RESULTS}" >&2
    exit 1
fi
if [ -n "${PRIOR_PATH}" ] && [ ! -f "${PRIOR_PATH}" ]; then
    echo "[warn] missing prior path: ${PRIOR_PATH}"
    echo "[warn] falling back to built-in AD-HH heads with rank priors"
    PRIOR_PATH=""
fi

resume_args=()
if [ "${RESUME}" = "1" ]; then
    resume_args=(--resume)
fi

echo "[info] hard results: ${HARD_RESULTS}"
echo "[info] soft results: ${SOFT_RESULTS}"
echo "[info] output dir: ${OUTPUT_DIR}"
echo "[info] max per label: ${MAX_PER_LABEL}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" python -m eval_scripts.soft_routing.build_all_head_feature_diagnostics \
    --hard-results "${HARD_RESULTS}" \
    --soft-results "${SOFT_RESULTS}" \
    --image-folder "${IMAGE_FOLDER}" \
    --output-dir "${OUTPUT_DIR}" \
    --prior-path "${PRIOR_PATH}" \
    --top-k "${TOP_K}" \
    --model-path "${MODEL_PATH}" \
    --conv-mode vicuna_v1 \
    --max-per-label "${MAX_PER_LABEL}" \
    --hallucinated-source "${HALLUCINATED_SOURCE}" \
    --adhh-threshold "${ADHH_THRESHOLD}" \
    --soft-gamma "${SOFT_GAMMA}" \
    --soft-temperature "${SOFT_TEMPERATURE}" \
    "${resume_args[@]}" \
    2>&1 | tee "${LOG_DIR}/all_head_feature_diagnostics_max${MAX_PER_LABEL}.log"

INPUT_JSONL="${OUTPUT_DIR}/all_head_feature_diagnostics.jsonl" \
OUTPUT_DIR="${OUTPUT_DIR}/unsupported_head_heatmaps" \
bash bash_scripts/soft_routing/run_unsupported_head_heatmaps.sh

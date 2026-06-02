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
TOP_K="${TOP_K:-150}"
MAX_PER_LABEL="${MAX_PER_LABEL:-200}"
SOFT_GAMMA="${SOFT_GAMMA:-0.75}"
SOFT_TEMPERATURE="${SOFT_TEMPERATURE:-0.05}"

BASE_RESULT_PATH="${BASE_RESULT_PATH:-./results/${DATASET}/soft_routing_smoke_n${NUM_SAMPLES}_seed${SEED}_tau${ADHH_THRESHOLD}_T${SOFT_TEMPERATURE}}"
BASE_RESULTS="${BASE_RESULTS:-${BASE_RESULT_PATH}/greedy/captions_eval_results.json}"
DEFAULT_PRIOR_PATH="../ADHH/LLaVA/results_summary/coco/ranked_heads_global__itext_all__C_toi_HminusG.json"
if [ ! -f "${DEFAULT_PRIOR_PATH}" ]; then
    DEFAULT_PRIOR_PATH="./results/${DATASET}/llava-v1.5-7b_base_original_qa_n3000/surrogate_hh_scores/surrogate_score_zoo/ranked_heads_global__itext_all__C_toi_HminusG.json"
fi
PRIOR_PATH="${PRIOR_PATH:-${DEFAULT_PRIOR_PATH}}"
OUTPUT_DIR="${OUTPUT_DIR:-${BASE_RESULT_PATH}/static_object_logprob_drop_top${TOP_K}}"
LOG_DIR="${LOG_DIR:-./logs/soft_routing}"

mkdir -p "${LOG_DIR}"

if [ ! -f "${BASE_RESULTS}" ]; then
    echo "[error] missing base results: ${BASE_RESULTS}" >&2
    exit 1
fi
if [ -n "${PRIOR_PATH}" ] && [ ! -f "${PRIOR_PATH}" ]; then
    echo "[warn] missing prior path: ${PRIOR_PATH}" >&2
    echo "[warn] falling back to built-in AD-HH heads" >&2
    PRIOR_PATH=""
fi

echo "[info] base results: ${BASE_RESULTS}"
echo "[info] prior path: ${PRIOR_PATH:-built-in AD-HH heads}"
echo "[info] top k: ${TOP_K}"
echo "[info] AD-HH threshold: ${ADHH_THRESHOLD}"
echo "[info] max per label: ${MAX_PER_LABEL}"
echo "[info] output dir: ${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" python -m eval_scripts.soft_routing.analyze_static_object_logprob_drop \
    --base-results "${BASE_RESULTS}" \
    --image-folder "${IMAGE_FOLDER}" \
    --output-dir "${OUTPUT_DIR}" \
    --prior-path "${PRIOR_PATH}" \
    --top-k "${TOP_K}" \
    --model-path "${MODEL_PATH}" \
    --conv-mode vicuna_v1 \
    --max-per-label "${MAX_PER_LABEL}" \
    --adhh-threshold "${ADHH_THRESHOLD}" \
    --soft-gamma "${SOFT_GAMMA}" \
    --soft-temperature "${SOFT_TEMPERATURE}" \
    2>&1 | tee "${LOG_DIR}/static_object_logprob_drop_top${TOP_K}.log"

echo "[summary] static object logprob drop"
if [ -f "${OUTPUT_DIR}/static_object_logprob_drop_summary.csv" ]; then
    column -s, -t "${OUTPUT_DIR}/static_object_logprob_drop_summary.csv"
fi

echo "[done] ${OUTPUT_DIR}"

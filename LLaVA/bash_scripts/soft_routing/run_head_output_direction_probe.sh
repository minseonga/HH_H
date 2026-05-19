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
MAX_PER_LABEL="${MAX_PER_LABEL:-100}"
HALLUCINATED_SOURCE="${HALLUCINATED_SOURCE:-both}"
MIN_LAYER="${MIN_LAYER:-13}"
MAX_LAYER="${MAX_LAYER:-31}"
HEAD_OUTPUT_NORMALIZATION="${HEAD_OUTPUT_NORMALIZATION:-l2}"
TEST_FRACTION="${TEST_FRACTION:-0.3}"
SAVE_HEAD_OUTPUT_VECTORS="${SAVE_HEAD_OUTPUT_VECTORS:-0}"

BASE_RESULT_PATH="${BASE_RESULT_PATH:-./results/${DATASET}/soft_routing_smoke_n${NUM_SAMPLES}_seed${SEED}_tau${ADHH_THRESHOLD}_T${SOFT_TEMPERATURE}}"
HARD_RESULTS="${HARD_RESULTS:-${BASE_RESULT_PATH}/hard_tau${ADHH_THRESHOLD}/captions_eval_results.json}"
SOFT_RESULTS="${SOFT_RESULTS:-${BASE_RESULT_PATH}/soft_gamma${SOFT_GAMMA}/captions_eval_results.json}"
PRIOR_PATH="${PRIOR_PATH:-${BASE_RESULT_PATH}/fixed_adhh20_score_prior/attribution_result.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${BASE_RESULT_PATH}/head_output_direction_probe_${HEAD_OUTPUT_NORMALIZATION}_l${MIN_LAYER}_${MAX_LAYER}_hall${HALLUCINATED_SOURCE}_max${MAX_PER_LABEL}}"
LOG_DIR="${LOG_DIR:-./logs/soft_routing}"

mkdir -p "${LOG_DIR}"

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

save_vector_args=()
if [ "${SAVE_HEAD_OUTPUT_VECTORS}" = "1" ]; then
    save_vector_args=(--save-query-vectors)
fi

echo "[info] hard results: ${HARD_RESULTS}"
echo "[info] soft results: ${SOFT_RESULTS}"
echo "[info] prior path: ${PRIOR_PATH:-built-in rank prior}"
echo "[info] output dir: ${OUTPUT_DIR}"
echo "[info] max per label: ${MAX_PER_LABEL}"
echo "[info] hallucinated source: ${HALLUCINATED_SOURCE}"
echo "[info] layers: ${MIN_LAYER}-${MAX_LAYER}"
echo "[info] head-output normalization: ${HEAD_OUTPUT_NORMALIZATION}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" python -m eval_scripts.soft_routing.build_query_direction_probe \
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
    --min-layer "${MIN_LAYER}" \
    --max-layer "${MAX_LAYER}" \
    --surface head_output \
    --query-normalization "${HEAD_OUTPUT_NORMALIZATION}" \
    --test-fraction "${TEST_FRACTION}" \
    --seed "${SEED}" \
    "${save_vector_args[@]}" \
    2>&1 | tee "${LOG_DIR}/head_output_direction_probe_${HEAD_OUTPUT_NORMALIZATION}_l${MIN_LAYER}_${MAX_LAYER}_hall${HALLUCINATED_SOURCE}_max${MAX_PER_LABEL}.log"

echo "[summary] top head-output directions"
if [ -f "${OUTPUT_DIR}/query_direction_auc.csv" ]; then
    column -s, -t "${OUTPUT_DIR}/query_direction_auc.csv" | head -40
fi

echo "[summary] layer summary"
if [ -f "${OUTPUT_DIR}/query_direction_layer_summary.csv" ]; then
    column -s, -t "${OUTPUT_DIR}/query_direction_layer_summary.csv" | head -40
fi

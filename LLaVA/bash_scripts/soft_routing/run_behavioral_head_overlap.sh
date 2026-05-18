#!/bin/bash

set -euo pipefail

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-True}"

GPU_ID="${GPU_ID:-0}"
DATASET="${DATASET:-coco}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
NUM_SAMPLES="${NUM_SAMPLES:-500}"
MAX_SAMPLES="${MAX_SAMPLES:-200}"
SEED="${SEED:-42}"
ADHH_THRESHOLD="${ADHH_THRESHOLD:-0.4}"
SOFT_TEMPERATURE="${SOFT_TEMPERATURE:-0.05}"
TOP_K="${TOP_K:-20}"
WARMUP_TOKENS="${WARMUP_TOKENS:-0}"
HEAD_PRIOR_MODE="${HEAD_PRIOR_MODE:-auto}"
MIN_LAYER="${MIN_LAYER:-}"
MAX_LAYER="${MAX_LAYER:-}"

BASE_RESULT_PATH="${BASE_RESULT_PATH:-./results/${DATASET}/soft_routing_smoke_n${NUM_SAMPLES}_seed${SEED}_tau${ADHH_THRESHOLD}_T${SOFT_TEMPERATURE}}"
EVAL_RESULTS="${EVAL_RESULTS:-${BASE_RESULT_PATH}/greedy/captions_eval_results.json}"

DEFAULT_ATTENTION_HEAD_PATH="${BASE_RESULT_PATH}/fixed_adhh20_score_prior/attribution_result.json"
if [ ! -f "${DEFAULT_ATTENTION_HEAD_PATH}" ]; then
    DEFAULT_ATTENTION_HEAD_PATH="./results/coco/llava_3000/identify_attention_head/attribution_result.json"
fi
ATTENTION_HEAD_PATH="${ATTENTION_HEAD_PATH:-${DEFAULT_ATTENTION_HEAD_PATH}}"

OUTPUT_DIR="${OUTPUT_DIR:-${BASE_RESULT_PATH}/behavioral_head_overlap_warmup${WARMUP_TOKENS}_n${MAX_SAMPLES}}"
LOG_DIR="${LOG_DIR:-./logs/soft_routing}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

if [ ! -f "${EVAL_RESULTS}" ]; then
    echo "[error] missing eval results: ${EVAL_RESULTS}" >&2
    exit 1
fi

if [ -n "${ATTENTION_HEAD_PATH}" ] && [ ! -f "${ATTENTION_HEAD_PATH}" ]; then
    echo "[warn] missing AD-HH head path: ${ATTENTION_HEAD_PATH}"
    echo "[warn] falling back to built-in AD-HH heads"
    ATTENTION_HEAD_PATH=""
fi

echo "[info] eval results: ${EVAL_RESULTS}"
echo "[info] image folder: ${IMAGE_FOLDER}"
echo "[info] AD-HH head path: ${ATTENTION_HEAD_PATH:-built-in default heads}"
echo "[info] output dir: ${OUTPUT_DIR}"
echo "[info] max samples: ${MAX_SAMPLES}"
echo "[info] warmup tokens: ${WARMUP_TOKENS}"
echo "[info] top K: ${TOP_K}"
echo "[info] layer filter: ${MIN_LAYER:-none}..${MAX_LAYER:-none}"

layer_args=()
if [ -n "${MIN_LAYER}" ]; then
    layer_args+=(--min-layer "${MIN_LAYER}")
fi
if [ -n "${MAX_LAYER}" ]; then
    layer_args+=(--max-layer "${MAX_LAYER}")
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" python -m eval_scripts.soft_routing.score_behavioral_heads \
    --eval-results "${EVAL_RESULTS}" \
    --image-folder "${IMAGE_FOLDER}" \
    --output-dir "${OUTPUT_DIR}" \
    --model-path "${MODEL_PATH}" \
    --conv-mode vicuna_v1 \
    --attention-head-path "${ATTENTION_HEAD_PATH}" \
    --head-prior-mode "${HEAD_PRIOR_MODE}" \
    --top-k "${TOP_K}" \
    --max-samples "${MAX_SAMPLES}" \
    --warmup-tokens "${WARMUP_TOKENS}" \
    "${layer_args[@]}" \
    --adhh-threshold "${ADHH_THRESHOLD}" \
    2>&1 | tee "${LOG_DIR}/behavioral_head_overlap_warmup${WARMUP_TOKENS}_n${MAX_SAMPLES}.log"

echo "[summary] overlap with AD-HH top-${TOP_K}"
if [ -f "${OUTPUT_DIR}/overlap_summary.csv" ]; then
    column -s, -t "${OUTPUT_DIR}/overlap_summary.csv"
fi

echo "[summary] top behavioral heads"
if [ -f "${OUTPUT_DIR}/head_scores.csv" ]; then
    column -s, -t "${OUTPUT_DIR}/head_scores.csv" | head -30
fi

echo "[summary] AD-HH reference head ranks under behavioral scores"
if [ -f "${OUTPUT_DIR}/reference_head_ranks.csv" ]; then
    column -s, -t "${OUTPUT_DIR}/reference_head_ranks.csv" | head -30
fi

echo "[summary] AD-HH reference rank distribution"
if [ -f "${OUTPUT_DIR}/reference_rank_summary.csv" ]; then
    column -s, -t "${OUTPUT_DIR}/reference_rank_summary.csv"
fi

echo "[summary] head group score comparison"
if [ -f "${OUTPUT_DIR}/head_group_score_summary.csv" ]; then
    column -s, -t "${OUTPUT_DIR}/head_group_score_summary.csv" | head -40
fi

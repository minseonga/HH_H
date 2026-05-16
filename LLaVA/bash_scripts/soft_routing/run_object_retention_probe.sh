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
HALLUCINATED_SOURCE="${HALLUCINATED_SOURCE:-soft}"
COMPUTE_VISUAL_SUPPORT="${COMPUTE_VISUAL_SUPPORT:-0}"
VISUAL_ABLATION="${VISUAL_ABLATION:-zero}"
SUPPRESSION_SWEEP="${SUPPRESSION_SWEEP:-}"
SWEEP_BIN_FEATURE="${SWEEP_BIN_FEATURE:-max_norm_excess}"
SWEEP_BIN_EDGES="${SWEEP_BIN_EDGES:-0,0.2,0.4,0.6,0.8,1.0}"
SWEEP_TAU_LOW="${SWEEP_TAU_LOW:-${ADHH_THRESHOLD}}"
SWEEP_TAU_HIGH="${SWEEP_TAU_HIGH:-0.9}"
SWEEP_SAFE_DROP="${SWEEP_SAFE_DROP:-0.1}"
SWEEP_HALL_DROP_FRACTION="${SWEEP_HALL_DROP_FRACTION:-0.8}"

BASE_RESULT_PATH="${BASE_RESULT_PATH:-./results/${DATASET}/soft_routing_smoke_n${NUM_SAMPLES}_seed${SEED}_tau${ADHH_THRESHOLD}_T${SOFT_TEMPERATURE}}"
HARD_RESULTS="${HARD_RESULTS:-${BASE_RESULT_PATH}/hard_tau${ADHH_THRESHOLD}/captions_eval_results.json}"
SOFT_RESULTS="${SOFT_RESULTS:-${BASE_RESULT_PATH}/soft_gamma${SOFT_GAMMA}/captions_eval_results.json}"
PRIOR_PATH="${PRIOR_PATH:-${BASE_RESULT_PATH}/fixed_adhh20_score_prior/attribution_result.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${BASE_RESULT_PATH}/object_retention_probe_soft${SOFT_GAMMA}_max${MAX_PER_LABEL}}"
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

echo "[info] hard results: ${HARD_RESULTS}"
echo "[info] soft results: ${SOFT_RESULTS}"
echo "[info] prior path: ${PRIOR_PATH:-built-in rank prior}"
echo "[info] output dir: ${OUTPUT_DIR}"
echo "[info] max per label: ${MAX_PER_LABEL}"
echo "[info] hallucinated source: ${HALLUCINATED_SOURCE}"
echo "[info] compute visual support: ${COMPUTE_VISUAL_SUPPORT}"
echo "[info] suppression sweep: ${SUPPRESSION_SWEEP:-disabled}"
echo "[info] sweep bin feature: ${SWEEP_BIN_FEATURE}"

visual_support_args=()
if [ "${COMPUTE_VISUAL_SUPPORT}" = "1" ]; then
    visual_support_args=(--compute-visual-support --visual-ablation "${VISUAL_ABLATION}")
fi

sweep_args=()
if [ -n "${SUPPRESSION_SWEEP}" ]; then
    sweep_args=(
        --suppression-sweep "${SUPPRESSION_SWEEP}"
        --sweep-bin-feature "${SWEEP_BIN_FEATURE}"
        --sweep-bin-edges "${SWEEP_BIN_EDGES}"
        --sweep-tau-low "${SWEEP_TAU_LOW}"
        --sweep-tau-high "${SWEEP_TAU_HIGH}"
        --sweep-safe-drop "${SWEEP_SAFE_DROP}"
        --sweep-hall-drop-fraction "${SWEEP_HALL_DROP_FRACTION}"
    )
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" python -m eval_scripts.soft_routing.analyze_object_retention_steps \
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
    "${visual_support_args[@]}" \
    "${sweep_args[@]}" \
    --adhh-threshold "${ADHH_THRESHOLD}" \
    --soft-gamma "${SOFT_GAMMA}" \
    --soft-temperature "${SOFT_TEMPERATURE}" \
    2>&1 | tee "${LOG_DIR}/object_retention_probe_soft${SOFT_GAMMA}_max${MAX_PER_LABEL}.log"

if [ -f "${OUTPUT_DIR}/object_retention_features.csv" ]; then
    python -m eval_scripts.soft_routing.summarize_object_retention_features \
        --features-csv "${OUTPUT_DIR}/object_retention_features.csv"
fi

echo "[summary] group means"
if [ -f "${OUTPUT_DIR}/group_feature_means.csv" ]; then
    column -s, -t "${OUTPUT_DIR}/group_feature_means.csv" | head -40
fi

echo "[summary] lost-grounded AUC"
if [ -f "${OUTPUT_DIR}/lost_grounded_auc.csv" ]; then
    column -s, -t "${OUTPUT_DIR}/lost_grounded_auc.csv" | head -30
fi

echo "[summary] lost-grounded vs hallucinated-object AUC"
if [ -f "${OUTPUT_DIR}/lost_vs_hallucinated_auc.csv" ]; then
    column -s, -t "${OUTPUT_DIR}/lost_vs_hallucinated_auc.csv" | head -30
fi

echo "[summary] suppression sweep by group/bin"
if [ -f "${OUTPUT_DIR}/suppression_sweep_by_group_bin.csv" ]; then
    column -s, -t "${OUTPUT_DIR}/suppression_sweep_by_group_bin.csv" | head -80
fi

echo "[summary] suppression sweep policy table"
if [ -f "${OUTPUT_DIR}/suppression_sweep_policy_table.csv" ]; then
    column -s, -t "${OUTPUT_DIR}/suppression_sweep_policy_table.csv" | head -40
fi

echo "[summary] oracle visual support correlations"
if [ -f "${OUTPUT_DIR}/oracle_visual_support_correlations.csv" ]; then
    column -s, -t "${OUTPUT_DIR}/oracle_visual_support_correlations.csv" | head -30
fi

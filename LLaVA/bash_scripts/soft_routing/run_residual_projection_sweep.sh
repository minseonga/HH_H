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
RESIDUAL_NORMALIZATION="${RESIDUAL_NORMALIZATION:-l2}"
MIN_LAYER="${MIN_LAYER:-13}"
MAX_LAYER="${MAX_LAYER:-31}"
DIRECTION_TOP_K="${DIRECTION_TOP_K:-10}"
MIN_DIRECTION_AUROC="${MIN_DIRECTION_AUROC:-0.65}"
GATE_MODE="${GATE_MODE:-none}"
RESIDUAL_DIRECTION_TEMPERATURE="${RESIDUAL_DIRECTION_TEMPERATURE:-0.05}"
PROJECTION_STRENGTHS="${PROJECTION_STRENGTHS:--1.0,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1.0}"

BASE_RESULT_PATH="${BASE_RESULT_PATH:-./results/${DATASET}/soft_routing_smoke_n${NUM_SAMPLES}_seed${SEED}_tau${ADHH_THRESHOLD}_T${SOFT_TEMPERATURE}}"
HARD_RESULTS="${HARD_RESULTS:-${BASE_RESULT_PATH}/hard_tau${ADHH_THRESHOLD}/captions_eval_results.json}"
SOFT_RESULTS="${SOFT_RESULTS:-${BASE_RESULT_PATH}/soft_gamma${SOFT_GAMMA}/captions_eval_results.json}"
PRIOR_PATH="${PRIOR_PATH:-${BASE_RESULT_PATH}/fixed_adhh20_score_prior/attribution_result.json}"
RESIDUAL_PROBE_DIR="${RESIDUAL_PROBE_DIR:-${BASE_RESULT_PATH}/residual_direction_probe_${RESIDUAL_NORMALIZATION}_l${MIN_LAYER}_${MAX_LAYER}_hall${HALLUCINATED_SOURCE}_max${MAX_PER_LABEL}}"
CALIBRATION_NPZ="${CALIBRATION_NPZ:-${RESIDUAL_PROBE_DIR}/query_direction_calibration.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-${BASE_RESULT_PATH}/residual_projection_sweep_${RESIDUAL_NORMALIZATION}_top${DIRECTION_TOP_K}_${GATE_MODE}_max${MAX_PER_LABEL}}"
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
if [ ! -f "${CALIBRATION_NPZ}" ]; then
    echo "[error] missing residual direction calibration: ${CALIBRATION_NPZ}" >&2
    echo "[hint] run bash_scripts/soft_routing/run_residual_direction_probe.sh first" >&2
    exit 1
fi
if [ -n "${PRIOR_PATH}" ] && [ ! -f "${PRIOR_PATH}" ]; then
    echo "[warn] missing prior path: ${PRIOR_PATH}"
    echo "[warn] falling back to built-in AD-HH heads with rank priors"
    PRIOR_PATH=""
fi

echo "[info] hard results: ${HARD_RESULTS}"
echo "[info] soft results: ${SOFT_RESULTS}"
echo "[info] calibration: ${CALIBRATION_NPZ}"
echo "[info] output dir: ${OUTPUT_DIR}"
echo "[info] direction top-k: ${DIRECTION_TOP_K}"
echo "[info] min direction AUROC: ${MIN_DIRECTION_AUROC}"
echo "[info] gate mode: ${GATE_MODE}"
echo "[info] projection strengths: ${PROJECTION_STRENGTHS}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" python -m eval_scripts.soft_routing.query_projection_sweep \
    --hard-results "${HARD_RESULTS}" \
    --soft-results "${SOFT_RESULTS}" \
    --image-folder "${IMAGE_FOLDER}" \
    --calibration-npz "${CALIBRATION_NPZ}" \
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
    --surface residual \
    --projection-strengths="${PROJECTION_STRENGTHS}" \
    --direction-top-k "${DIRECTION_TOP_K}" \
    --min-direction-auroc "${MIN_DIRECTION_AUROC}" \
    --gate-mode "${GATE_MODE}" \
    --query-direction-temperature "${RESIDUAL_DIRECTION_TEMPERATURE}" \
    2>&1 | tee "${LOG_DIR}/residual_projection_sweep_${RESIDUAL_NORMALIZATION}_top${DIRECTION_TOP_K}_${GATE_MODE}_max${MAX_PER_LABEL}.log"

echo "[summary] selected residual directions"
if [ -f "${OUTPUT_DIR}/selected_query_directions.csv" ]; then
    column -s, -t "${OUTPUT_DIR}/selected_query_directions.csv" | head -40
fi

echo "[summary] projection sweep by group"
if [ -f "${OUTPUT_DIR}/query_projection_sweep_by_group.csv" ]; then
    column -s, -t "${OUTPUT_DIR}/query_projection_sweep_by_group.csv" | head -80
fi

echo "[summary] projection policy summary"
if [ -f "${OUTPUT_DIR}/query_projection_sweep_policy_summary.csv" ]; then
    column -s, -t "${OUTPUT_DIR}/query_projection_sweep_policy_summary.csv" | head -80
fi

echo "[summary] projection diagnostic summary"
if [ -f "${OUTPUT_DIR}/query_projection_diagnostics_by_group.csv" ]; then
    column -s, -t "${OUTPUT_DIR}/query_projection_diagnostics_by_group.csv" | head -120
fi

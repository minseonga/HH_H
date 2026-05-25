#!/bin/bash

set -euo pipefail

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-True}"

GPU_ID="${GPU_ID:-0}"
DATASET="${DATASET:-coco}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
IMAGE_SPLIT="${IMAGE_SPLIT:-val2014}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
SEED="${SEED:-42}"
ADHH_THRESHOLD="${ADHH_THRESHOLD:-0.4}"
SOFT_TEMPERATURE="${SOFT_TEMPERATURE:-0.05}"

BASE_RESULT_PATH="${BASE_RESULT_PATH:-./results/${DATASET}/soft_routing_smoke_n500_seed42_tau0.4_T0.05}"
EVAL_RESULTS="${EVAL_RESULTS:-${BASE_RESULT_PATH}/greedy/captions_eval_results.json}"
CALIBRATION_NPZ="${CALIBRATION_NPZ:-${BASE_RESULT_PATH}/query_direction_probe_l2_l13_31_hallboth_max100/query_direction_calibration.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-${BASE_RESULT_PATH}/query_direction_chair_alignment_n${NUM_SAMPLES}}"

DIRECTION_TOP_K="${DIRECTION_TOP_K:-20}"
MIN_DIRECTION_AUROC="${MIN_DIRECTION_AUROC:-0.0}"
SELECT_BY="${SELECT_BY:-high}"
QUERY_NORMALIZATION="${QUERY_NORMALIZATION:-l2}"
ENSEMBLE_TOP_KS="${ENSEMBLE_TOP_KS:-1,3,5,10,20}"
SCORE_SPAN="${SCORE_SPAN:-first}"
SPAN_AGGREGATION="${SPAN_AGGREGATION:-max}"
MATCH_EVAL_RESULTS="${MATCH_EVAL_RESULTS:-}"
EXCLUDE_QUERY_PROBE_DIR="${EXCLUDE_QUERY_PROBE_DIR:-}"
EXCLUDE_PROBE_STEPS="${EXCLUDE_PROBE_STEPS:-}"
EXCLUDE_EVAL_RESULTS="${EXCLUDE_EVAL_RESULTS:-}"
EXCLUDE_IMAGE_IDS="${EXCLUDE_IMAGE_IDS:-}"

mkdir -p "${OUTPUT_DIR}"

if [ ! -f "${EVAL_RESULTS}" ]; then
    echo "[error] missing eval results: ${EVAL_RESULTS}" >&2
    exit 1
fi
if [ ! -f "${CALIBRATION_NPZ}" ]; then
    echo "[error] missing query calibration: ${CALIBRATION_NPZ}" >&2
    exit 1
fi

match_args=()
if [ -n "${MATCH_EVAL_RESULTS}" ]; then
    match_args=(--match-eval-results "${MATCH_EVAL_RESULTS}")
fi

exclude_args=()
if [ -n "${EXCLUDE_QUERY_PROBE_DIR}" ]; then
    exclude_args+=(--exclude-query-probe-dir "${EXCLUDE_QUERY_PROBE_DIR}")
fi
if [ -n "${EXCLUDE_PROBE_STEPS}" ]; then
    exclude_args+=(--exclude-probe-steps ${EXCLUDE_PROBE_STEPS})
fi
if [ -n "${EXCLUDE_EVAL_RESULTS}" ]; then
    exclude_args+=(--exclude-eval-results ${EXCLUDE_EVAL_RESULTS})
fi
if [ -n "${EXCLUDE_IMAGE_IDS}" ]; then
    exclude_args+=(--exclude-image-ids ${EXCLUDE_IMAGE_IDS})
fi

echo "[info] eval results: ${EVAL_RESULTS}"
echo "[info] calibration: ${CALIBRATION_NPZ}"
echo "[info] output dir: ${OUTPUT_DIR}"
echo "[info] image folder: ${IMAGE_FOLDER}"
echo "[info] num samples: ${NUM_SAMPLES}"
echo "[info] exclude query probe dir: ${EXCLUDE_QUERY_PROBE_DIR:-none}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" python -m eval_scripts.soft_routing.analyze_query_direction_chair_alignment \
    --eval-results "${EVAL_RESULTS}" \
    --calibration-npz "${CALIBRATION_NPZ}" \
    --image-folder "${IMAGE_FOLDER}" \
    --image-split "${IMAGE_SPLIT}" \
    --output-dir "${OUTPUT_DIR}" \
    --model-path "${MODEL_PATH}" \
    --conv-mode vicuna_v1 \
    --direction-top-k "${DIRECTION_TOP_K}" \
    --min-direction-auroc "${MIN_DIRECTION_AUROC}" \
    --select-by "${SELECT_BY}" \
    --query-normalization "${QUERY_NORMALIZATION}" \
    --ensemble-top-ks "${ENSEMBLE_TOP_KS}" \
    --score-span "${SCORE_SPAN}" \
    --span-aggregation "${SPAN_AGGREGATION}" \
    --max-sentences "${NUM_SAMPLES}" \
    "${match_args[@]}" \
    "${exclude_args[@]}"

echo "[summary] mention-level AUROC"
if [ -f "${OUTPUT_DIR}/query_direction_chair_auc.csv" ]; then
    column -s, -t "${OUTPUT_DIR}/query_direction_chair_auc.csv" | head -30
fi

echo "[summary] sample-level AUROC"
if [ -f "${OUTPUT_DIR}/query_direction_chair_sample_auc.csv" ]; then
    column -s, -t "${OUTPUT_DIR}/query_direction_chair_sample_auc.csv" | head -30
fi

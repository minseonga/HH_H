#!/bin/bash

set -euo pipefail

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-True}"

GPU_ID="${GPU_ID:-0}"
DATASET="${DATASET:-coco}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/pope/val2014}"

BASE_RESULT_PATH="${BASE_RESULT_PATH:-./results/${DATASET}/soft_routing_smoke_n500_seed42_tau0.4_T0.05}"
EVAL_DIR="${EVAL_DIR:-${BASE_RESULT_PATH}/adhh_robustness_n100}"
GREEDY_EVAL="${GREEDY_EVAL:-${EVAL_DIR}/greedy/captions_eval_results.json}"
ADHH_EVAL="${ADHH_EVAL:-${EVAL_DIR}/adhh_hard/captions_eval_results.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${BASE_RESULT_PATH}/contribution_gap_validation}"

MAX_MENTIONS="${MAX_MENTIONS:-80}"
MAX_PER_LABEL="${MAX_PER_LABEL:-40}"
INCLUDE_ADHH_TOP_K="${INCLUDE_ADHH_TOP_K:-20}"
CANDIDATE_HEADS="${CANDIDATE_HEADS:-}"
QUERY_CALIBRATION="${QUERY_CALIBRATION:-}"
QUERY_TOP_K="${QUERY_TOP_K:-0}"
QUERY_MIN_AUROC="${QUERY_MIN_AUROC:-0.0}"
RESUME="${RESUME:-0}"
AGGREGATE_ONLY="${AGGREGATE_ONLY:-0}"
SKIP_FULL_HEAD_ABLATION="${SKIP_FULL_HEAD_ABLATION:-1}"
SKIP_TEXT_ABLATION="${SKIP_TEXT_ABLATION:-0}"

mkdir -p "${OUTPUT_DIR}"

echo "[info] output dir: ${OUTPUT_DIR}"
echo "[info] greedy eval: ${GREEDY_EVAL}"
echo "[info] AD-HH eval: ${ADHH_EVAL}"
echo "[info] max mentions: ${MAX_MENTIONS}"
echo "[info] max per label: ${MAX_PER_LABEL}"
echo "[info] include AD-HH top-k: ${INCLUDE_ADHH_TOP_K}"
echo "[info] explicit candidate heads: ${CANDIDATE_HEADS:-<none>}"
echo "[info] skip text ablation: ${SKIP_TEXT_ABLATION}"

extra_args=()
if [ -n "${CANDIDATE_HEADS}" ]; then
    extra_args+=(--candidate-heads "${CANDIDATE_HEADS}")
fi
if [ -n "${QUERY_CALIBRATION}" ]; then
    extra_args+=(
        --query-calibration "${QUERY_CALIBRATION}"
        --query-top-k "${QUERY_TOP_K}"
        --query-min-auroc "${QUERY_MIN_AUROC}"
    )
fi
if [ "${RESUME}" = "1" ]; then
    extra_args+=(--resume)
fi
if [ "${AGGREGATE_ONLY}" = "1" ]; then
    extra_args+=(--aggregate-only)
fi
if [ "${SKIP_FULL_HEAD_ABLATION}" = "1" ]; then
    extra_args+=(--skip-full-head-ablation)
fi
if [ "${SKIP_TEXT_ABLATION}" = "1" ]; then
    extra_args+=(--skip-text-ablation)
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" python -m eval_scripts.soft_routing.validate_head_logit_contribution_proxy \
    --eval-results "${GREEDY_EVAL}" \
    --match-eval-results "${ADHH_EVAL}" \
    --image-folder "${IMAGE_FOLDER}" \
    --model-path "${MODEL_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --max-mentions "${MAX_MENTIONS}" \
    --max-per-label "${MAX_PER_LABEL}" \
    --include-adhh-top-k "${INCLUDE_ADHH_TOP_K}" \
    "${extra_args[@]}"

python -m eval_scripts.soft_routing.analyze_adhh_fragility_features \
    --base-eval "${GREEDY_EVAL}" \
    --target-eval "${ADHH_EVAL}" \
    --object-step-features "${OUTPUT_DIR}/contribution_gap_mention_features.csv" \
    --base-name greedy \
    --target-name adhh_hard \
    --output-dir "${OUTPUT_DIR}/fragility_outcomes"

echo "[summary] proxy vs text-side ablation"
column -s, -t "${OUTPUT_DIR}/head_logit_proxy_ablation_correlations.csv" | head -60

echo "[summary] mention-level proxy vs text-side teacher"
column -s, -t "${OUTPUT_DIR}/mention_proxy_teacher_correlations.csv" | head -80

echo "[summary] contribution-gap outcome AUC"
column -s, -t "${OUTPUT_DIR}/fragility_outcomes/adhh_fragility_feature_auc.csv" | head -80

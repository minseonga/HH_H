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
MAX_PER_LABEL="${MAX_PER_LABEL:-20}"
HALLUCINATED_SOURCE="${HALLUCINATED_SOURCE:-both}"
CANDIDATE_POLICY="${CANDIDATE_POLICY:-text_topk}"
CANDIDATE_MAX_HEADS="${CANDIDATE_MAX_HEADS:-32}"
CANDIDATE_TEXT_TAU="${CANDIDATE_TEXT_TAU:-0.4}"
ABLATION_THRESHOLD="${ABLATION_THRESHOLD:-0.0}"
POSITIVE_EFFECT_THRESHOLD="${POSITIVE_EFFECT_THRESHOLD:-0.02}"
SELECTOR_TOP_K="${SELECTOR_TOP_K:-8}"
RESUME="${RESUME:-1}"

BASE_RESULT_PATH="${BASE_RESULT_PATH:-./results/${DATASET}/soft_routing_smoke_n${NUM_SAMPLES}_seed${SEED}_tau${ADHH_THRESHOLD}_T${SOFT_TEMPERATURE}}"
HARD_RESULTS="${HARD_RESULTS:-${BASE_RESULT_PATH}/hard_tau${ADHH_THRESHOLD}/captions_eval_results.json}"
SOFT_RESULTS="${SOFT_RESULTS:-${BASE_RESULT_PATH}/soft_gamma${SOFT_GAMMA}/captions_eval_results.json}"
PRIOR_PATH="${PRIOR_PATH:-${BASE_RESULT_PATH}/fixed_adhh20_score_prior/attribution_result.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${BASE_RESULT_PATH}/online_causal_head_teacher_${CANDIDATE_POLICY}_h${CANDIDATE_MAX_HEADS}_max${MAX_PER_LABEL}}"
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
echo "[info] prior path: ${PRIOR_PATH:-built-in rank prior}"
echo "[info] output dir: ${OUTPUT_DIR}"
echo "[info] max per label: ${MAX_PER_LABEL}"
echo "[info] hallucinated source: ${HALLUCINATED_SOURCE}"
echo "[info] candidate policy: ${CANDIDATE_POLICY}"
echo "[info] candidate max heads: ${CANDIDATE_MAX_HEADS}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" python -m eval_scripts.soft_routing.build_online_causal_head_teacher \
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
    --candidate-policy "${CANDIDATE_POLICY}" \
    --candidate-max-heads "${CANDIDATE_MAX_HEADS}" \
    --candidate-text-tau "${CANDIDATE_TEXT_TAU}" \
    --ablation-threshold "${ABLATION_THRESHOLD}" \
    --positive-effect-threshold "${POSITIVE_EFFECT_THRESHOLD}" \
    --selector-top-k "${SELECTOR_TOP_K}" \
    --adhh-threshold "${ADHH_THRESHOLD}" \
    --soft-gamma "${SOFT_GAMMA}" \
    --soft-temperature "${SOFT_TEMPERATURE}" \
    "${resume_args[@]}" \
    2>&1 | tee "${LOG_DIR}/online_causal_head_teacher_${CANDIDATE_POLICY}_h${CANDIDATE_MAX_HEADS}_max${MAX_PER_LABEL}.log"

echo "[summary] positive-effect AUC"
if [ -f "${OUTPUT_DIR}/feature_positive_effect_auc.csv" ]; then
    column -s, -t "${OUTPUT_DIR}/feature_positive_effect_auc.csv" | head -50
fi

echo "[summary] effect correlations"
if [ -f "${OUTPUT_DIR}/feature_effect_correlations.csv" ]; then
    column -s, -t "${OUTPUT_DIR}/feature_effect_correlations.csv" | head -50
fi

echo "[summary] selector recovery"
if [ -f "${OUTPUT_DIR}/selector_recovery_summary.csv" ]; then
    column -s, -t "${OUTPUT_DIR}/selector_recovery_summary.csv" | head -80
fi

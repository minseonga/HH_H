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

MAX_MENTIONS="${MAX_MENTIONS:-80}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
MAX_PER_LABEL="${MAX_PER_LABEL:-0}"
LABEL_FILTER="${LABEL_FILTER:-hallucinated}"
LAYER_START="${LAYER_START:-13}"
LAYER_END="${LAYER_END:-31}"
HEAD_START="${HEAD_START:-0}"
HEAD_END="${HEAD_END:-31}"
CANDIDATE_HEADS="${CANDIDATE_HEADS:-}"
TOP_KS="${TOP_KS:-20,40}"

TEACHER_HEAD_ROWS="${TEACHER_HEAD_ROWS:-}"
TEACHER_FEATURE="${TEACHER_FEATURE:-proxy_text_target_logit}"
TEACHER_LABEL_FILTER="${TEACHER_LABEL_FILTER:-hallucinated}"
TEACHER_SCORE_MODE="${TEACHER_SCORE_MODE:-positive_mean}"

if [ -z "${TEACHER_HEAD_ROWS}" ]; then
    default_teacher="${BASE_RESULT_PATH}/proxy_hallucination_leverage_l0_31_h0_31_n${MAX_MENTIONS}/head_logit_proxy_ablation_rows.csv"
    if [ -f "${default_teacher}" ]; then
        TEACHER_HEAD_ROWS="${default_teacher}"
    fi
fi

if [ -z "${CANDIDATE_HEADS}" ]; then
    HEAD_GRID_NAME="l${LAYER_START}_${LAYER_END}_h${HEAD_START}_${HEAD_END}"
else
    HEAD_GRID_NAME="custom_heads"
fi

OUTPUT_DIR="${OUTPUT_DIR:-${BASE_RESULT_PATH}/prefill_head_selection_${HEAD_GRID_NAME}_n${MAX_MENTIONS}}"
mkdir -p "${OUTPUT_DIR}"

echo "[info] output dir: ${OUTPUT_DIR}"
echo "[info] greedy eval: ${GREEDY_EVAL}"
echo "[info] AD-HH eval: ${ADHH_EVAL}"
echo "[info] label filter: ${LABEL_FILTER}"
echo "[info] max mentions: ${MAX_MENTIONS}"
echo "[info] max samples: ${MAX_SAMPLES}"
echo "[info] candidate head grid: ${HEAD_GRID_NAME}"
echo "[info] top ks: ${TOP_KS}"
echo "[info] teacher rows: ${TEACHER_HEAD_ROWS:-<none>}"
echo "[info] teacher feature: ${TEACHER_FEATURE}"

extra_args=()
if [ -n "${CANDIDATE_HEADS}" ]; then
    extra_args+=(--candidate-heads "${CANDIDATE_HEADS}")
fi
if [ -n "${TEACHER_HEAD_ROWS}" ]; then
    extra_args+=(
        --teacher-head-rows "${TEACHER_HEAD_ROWS}"
        --teacher-feature "${TEACHER_FEATURE}"
        --teacher-label-filter "${TEACHER_LABEL_FILTER}"
        --teacher-score-mode "${TEACHER_SCORE_MODE}"
    )
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" python -m eval_scripts.soft_routing.analyze_prefill_head_selection \
    --eval-results "${GREEDY_EVAL}" \
    --match-eval-results "${ADHH_EVAL}" \
    --image-folder "${IMAGE_FOLDER}" \
    --model-path "${MODEL_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --max-mentions "${MAX_MENTIONS}" \
    --max-samples "${MAX_SAMPLES}" \
    --max-per-label "${MAX_PER_LABEL}" \
    --label-filter "${LABEL_FILTER}" \
    --layer-start "${LAYER_START}" \
    --layer-end "${LAYER_END}" \
    --head-start "${HEAD_START}" \
    --head-end "${HEAD_END}" \
    --top-ks "${TOP_KS}" \
    "${extra_args[@]}"

echo "[summary] prefill vs AD-HH overlay"
column -s, -t "${OUTPUT_DIR}/prefill_adhh_overlay_summary.csv"

if [ -s "${OUTPUT_DIR}/prefill_teacher_overlap_summary.csv" ]; then
    echo "[summary] prefill vs later object-step teacher overlap"
    column -s, -t "${OUTPUT_DIR}/prefill_teacher_overlap_summary.csv"
fi

echo "[summary] most frequently selected prefill heads"
column -s, -t "${OUTPUT_DIR}/prefill_selected_head_frequency.csv" | head -80

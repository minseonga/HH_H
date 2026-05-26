#!/bin/bash

set -euo pipefail

DATASET="${DATASET:-coco}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"

BASE_RESULT_PATH="${BASE_RESULT_PATH:-./results/${DATASET}/soft_routing_smoke_n500_seed42_tau0.4_T0.05}"
EVAL_DIR="${EVAL_DIR:-${BASE_RESULT_PATH}/adhh_robustness_n100}"
GREEDY_EVAL="${GREEDY_EVAL:-${EVAL_DIR}/greedy/captions_eval_results.json}"
ADHH_EVAL="${ADHH_EVAL:-${EVAL_DIR}/adhh_hard/captions_eval_results.json}"
CONTRIBUTION_DIR="${CONTRIBUTION_DIR:-${BASE_RESULT_PATH}/contribution_gap_validation_n80}"
HEAD_ROWS="${HEAD_ROWS:-${CONTRIBUTION_DIR}/head_logit_proxy_ablation_rows.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-${CONTRIBUTION_DIR}/residual_hallucination_head_discovery}"

TEACHER_FEATURE="${TEACHER_FEATURE:-target_text_logprob_drop}"
SECONDARY_TEACHER_FEATURE="${SECONDARY_TEACHER_FEATURE:-target_logprob_drop}"
PRIMARY_POSITIVE_OUTCOMES="${PRIMARY_POSITIVE_OUTCOMES:-hallucinated_retained}"
GROUNDED_PENALTY_ALPHA="${GROUNDED_PENALTY_ALPHA:-0.15}"
GROUNDED_LOST_PENALTY_ALPHA="${GROUNDED_LOST_PENALTY_ALPHA:-0.05}"
PRIOR_TOP_K="${PRIOR_TOP_K:-20}"
FILTER_COLUMN="${FILTER_COLUMN:-}"
FILTER_MIN="${FILTER_MIN:-}"
FILTER_MAX="${FILTER_MAX:-}"

mkdir -p "${OUTPUT_DIR}"

echo "[info] output dir: ${OUTPUT_DIR}"
echo "[info] greedy eval: ${GREEDY_EVAL}"
echo "[info] AD-HH eval: ${ADHH_EVAL}"
echo "[info] head rows: ${HEAD_ROWS}"
echo "[info] teacher: ${TEACHER_FEATURE}"
echo "[info] primary positives: ${PRIMARY_POSITIVE_OUTCOMES}"
echo "[info] grounded penalty alpha: ${GROUNDED_PENALTY_ALPHA}"

extra_args=()
if [ -n "${FILTER_COLUMN}" ]; then
    extra_args+=(--filter-column "${FILTER_COLUMN}")
fi
if [ -n "${FILTER_MIN}" ]; then
    extra_args+=(--filter-min "${FILTER_MIN}")
fi
if [ -n "${FILTER_MAX}" ]; then
    extra_args+=(--filter-max "${FILTER_MAX}")
fi

python -m eval_scripts.soft_routing.discover_residual_hallucination_heads \
    --base-eval "${GREEDY_EVAL}" \
    --target-eval "${ADHH_EVAL}" \
    --head-rows "${HEAD_ROWS}" \
    --output-dir "${OUTPUT_DIR}" \
    --model-path "${MODEL_PATH}" \
    --teacher-feature "${TEACHER_FEATURE}" \
    --secondary-teacher-feature "${SECONDARY_TEACHER_FEATURE}" \
    --primary-positive-outcomes "${PRIMARY_POSITIVE_OUTCOMES}" \
    --grounded-penalty-alpha "${GROUNDED_PENALTY_ALPHA}" \
    --grounded-lost-penalty-alpha "${GROUNDED_LOST_PENALTY_ALPHA}" \
    --prior-top-k "${PRIOR_TOP_K}" \
    "${extra_args[@]}"

echo "[summary] residual hallucination head scores"
column -s, -t "${OUTPUT_DIR}/residual_hallucination_head_scores.csv" | head -80

echo "[summary] selected heads"
cat "${OUTPUT_DIR}/residual_hallucination_heads.txt"

echo "[summary] selection"
column -s, -t "${OUTPUT_DIR}/residual_hallucination_head_selection_summary.csv"

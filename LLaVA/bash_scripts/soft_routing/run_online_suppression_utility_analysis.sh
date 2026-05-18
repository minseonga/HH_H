#!/bin/bash

set -euo pipefail

DATASET="${DATASET:-coco}"
NUM_SAMPLES="${NUM_SAMPLES:-500}"
SEED="${SEED:-42}"
ADHH_THRESHOLD="${ADHH_THRESHOLD:-0.4}"
SOFT_TEMPERATURE="${SOFT_TEMPERATURE:-0.05}"
CANDIDATE_POLICY="${CANDIDATE_POLICY:-text_topk}"
CANDIDATE_MAX_HEADS="${CANDIDATE_MAX_HEADS:-32}"
MAX_PER_LABEL="${MAX_PER_LABEL:-10}"
POSITIVE_UTILITY_THRESHOLD="${POSITIVE_UTILITY_THRESHOLD:-0.02}"
SELECTOR_TOP_K="${SELECTOR_TOP_K:-8}"

BASE_RESULT_PATH="${BASE_RESULT_PATH:-./results/${DATASET}/soft_routing_smoke_n${NUM_SAMPLES}_seed${SEED}_tau${ADHH_THRESHOLD}_T${SOFT_TEMPERATURE}}"
TEACHER_DIR="${TEACHER_DIR:-${BASE_RESULT_PATH}/online_causal_head_teacher_${CANDIDATE_POLICY}_h${CANDIDATE_MAX_HEADS}_max${MAX_PER_LABEL}}"
TEACHER_JSONL="${TEACHER_JSONL:-${TEACHER_DIR}/online_causal_head_teacher.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${TEACHER_DIR}/suppression_utility_t${POSITIVE_UTILITY_THRESHOLD}_top${SELECTOR_TOP_K}}"

if [ ! -f "${TEACHER_JSONL}" ]; then
    echo "[error] missing teacher jsonl: ${TEACHER_JSONL}" >&2
    echo "[hint] run bash_scripts/soft_routing/run_online_causal_head_teacher.sh first" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "[info] teacher jsonl: ${TEACHER_JSONL}"
echo "[info] output dir: ${OUTPUT_DIR}"

python -m eval_scripts.soft_routing.analyze_online_suppression_utility \
    --teacher-jsonl "${TEACHER_JSONL}" \
    --output-dir "${OUTPUT_DIR}" \
    --positive-utility-threshold "${POSITIVE_UTILITY_THRESHOLD}" \
    --selector-top-k "${SELECTOR_TOP_K}" \
    --seed "${SEED}"

echo "[summary] suppression utility AUC"
if [ -f "${OUTPUT_DIR}/feature_suppression_utility_auc.csv" ]; then
    column -s, -t "${OUTPUT_DIR}/feature_suppression_utility_auc.csv" | head -80
fi

echo "[summary] selector utility recovery"
if [ -f "${OUTPUT_DIR}/selector_suppression_utility_recovery.csv" ]; then
    column -s, -t "${OUTPUT_DIR}/selector_suppression_utility_recovery.csv" | head -100
fi

echo "[summary] learned utility model"
if [ -f "${OUTPUT_DIR}/learned_suppression_utility_metrics.csv" ]; then
    column -s, -t "${OUTPUT_DIR}/learned_suppression_utility_metrics.csv"
fi

echo "[summary] learned selector recovery"
if [ -f "${OUTPUT_DIR}/learned_selector_recovery.csv" ]; then
    column -s, -t "${OUTPUT_DIR}/learned_selector_recovery.csv" | head -80
fi

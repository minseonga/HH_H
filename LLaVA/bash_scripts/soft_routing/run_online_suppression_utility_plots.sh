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
PLOT_FEATURE="${PLOT_FEATURE:-text_value_norm}"
PLOT_BINS="${PLOT_BINS:-10}"

BASE_RESULT_PATH="${BASE_RESULT_PATH:-./results/${DATASET}/soft_routing_smoke_n${NUM_SAMPLES}_seed${SEED}_tau${ADHH_THRESHOLD}_T${SOFT_TEMPERATURE}}"
TEACHER_DIR="${TEACHER_DIR:-${BASE_RESULT_PATH}/online_causal_head_teacher_${CANDIDATE_POLICY}_h${CANDIDATE_MAX_HEADS}_max${MAX_PER_LABEL}}"
UTILITY_DIR="${UTILITY_DIR:-${TEACHER_DIR}/suppression_utility_t${POSITIVE_UTILITY_THRESHOLD}_top${SELECTOR_TOP_K}}"
OUTPUT_DIR="${OUTPUT_DIR:-${UTILITY_DIR}/figures}"

if [ ! -f "${UTILITY_DIR}/suppression_utility_rows.csv" ]; then
    echo "[error] missing utility rows: ${UTILITY_DIR}/suppression_utility_rows.csv" >&2
    echo "[hint] run bash_scripts/soft_routing/run_online_suppression_utility_analysis.sh first" >&2
    exit 1
fi

python -m eval_scripts.soft_routing.plot_online_suppression_utility \
    --utility-dir "${UTILITY_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --feature "${PLOT_FEATURE}" \
    --bins "${PLOT_BINS}"

echo "[summary] figure outputs"
find "${OUTPUT_DIR}" -maxdepth 1 -type f | sort

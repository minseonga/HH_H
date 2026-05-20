#!/bin/bash

set -euo pipefail

DATASET="${DATASET:-coco}"
NUM_SAMPLES="${NUM_SAMPLES:-500}"
SEED="${SEED:-42}"
ADHH_THRESHOLD="${ADHH_THRESHOLD:-0.4}"
SOFT_TEMPERATURE="${SOFT_TEMPERATURE:-0.05}"
CANDIDATE_POLICY="${CANDIDATE_POLICY:-anchor_mixed}"
CANDIDATE_MAX_HEADS="${CANDIDATE_MAX_HEADS:-48}"
MAX_PER_LABEL="${MAX_PER_LABEL:-10}"
POSITIVE_UTILITY_THRESHOLD="${POSITIVE_UTILITY_THRESHOLD:-0.02}"
PARALLEL_COS_THRESHOLD="${PARALLEL_COS_THRESHOLD:-0.3}"
ORTHOGONAL_ABS_COS_THRESHOLD="${ORTHOGONAL_ABS_COS_THRESHOLD:-0.1}"

BASE_RESULT_PATH="${BASE_RESULT_PATH:-./results/${DATASET}/soft_routing_smoke_n${NUM_SAMPLES}_seed${SEED}_tau${ADHH_THRESHOLD}_T${SOFT_TEMPERATURE}}"
TEACHER_DIR="${TEACHER_DIR:-${BASE_RESULT_PATH}/online_causal_head_teacher_${CANDIDATE_POLICY}_h${CANDIDATE_MAX_HEADS}_max${MAX_PER_LABEL}}"
VISUAL_ANALYSIS_DIR="${VISUAL_ANALYSIS_DIR:-${TEACHER_DIR}/visual_anchor_cases_t${POSITIVE_UTILITY_THRESHOLD}_p${PARALLEL_COS_THRESHOLD}_o${ORTHOGONAL_ABS_COS_THRESHOLD}}"
OVERLAP_SUMMARY="${OVERLAP_SUMMARY:-${TEACHER_DIR}/unsupported_head_heatmaps/unsupported_positive_overlap_summary.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-${TEACHER_DIR}/unsupported_feature_report}"
TOP_OVERLAP_K="${TOP_OVERLAP_K:-20}"

eval_args=()
if [ -n "${EVAL_RESULTS:-}" ]; then
    for item in ${EVAL_RESULTS}; do
        eval_args+=(--eval-result "${item}")
    done
fi

if [ ! -f "${VISUAL_ANALYSIS_DIR}/visual_feature_suppression_utility_contrast.csv" ]; then
    echo "[error] missing visual analysis output: ${VISUAL_ANALYSIS_DIR}" >&2
    echo "[hint] run bash_scripts/soft_routing/run_visual_anchor_case_analysis.sh first" >&2
    exit 1
fi

if [ ! -f "${OVERLAP_SUMMARY}" ]; then
    echo "[error] missing overlap summary: ${OVERLAP_SUMMARY}" >&2
    echo "[hint] run bash_scripts/soft_routing/run_unsupported_head_heatmaps.sh first" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${OUTPUT_DIR}/.matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

echo "[info] visual analysis dir: ${VISUAL_ANALYSIS_DIR}"
echo "[info] overlap summary: ${OVERLAP_SUMMARY}"
echo "[info] output dir: ${OUTPUT_DIR}"

python -m eval_scripts.soft_routing.build_unsupported_feature_report \
    --visual-analysis-dir "${VISUAL_ANALYSIS_DIR}" \
    --overlap-summary "${OVERLAP_SUMMARY}" \
    --output-dir "${OUTPUT_DIR}" \
    --top-overlap-k "${TOP_OVERLAP_K}" \
    "${eval_args[@]}"

echo "[summary] feature vs AD-HH"
column -s, -t "${OUTPUT_DIR}/feature_vs_adhh_summary.csv"

echo "[summary] AD-HH failure cases"
column -s, -t "${OUTPUT_DIR}/adhh_failure_case_summary.csv"

echo "[summary] overlap candidates"
column -s, -t "${OUTPUT_DIR}/overlap_candidate_head_summary.csv" | head -30

echo "[summary] report markdown: ${OUTPUT_DIR}/unsupported_feature_report.md"

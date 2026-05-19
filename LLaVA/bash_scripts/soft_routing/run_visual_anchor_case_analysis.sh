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
PARALLEL_COS_THRESHOLD="${PARALLEL_COS_THRESHOLD:-0.3}"
ORTHOGONAL_ABS_COS_THRESHOLD="${ORTHOGONAL_ABS_COS_THRESHOLD:-0.1}"
CASE1_RHO_SWEEP="${CASE1_RHO_SWEEP:-0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7}"

BASE_RESULT_PATH="${BASE_RESULT_PATH:-./results/${DATASET}/soft_routing_smoke_n${NUM_SAMPLES}_seed${SEED}_tau${ADHH_THRESHOLD}_T${SOFT_TEMPERATURE}}"
TEACHER_DIR="${TEACHER_DIR:-${BASE_RESULT_PATH}/online_causal_head_teacher_${CANDIDATE_POLICY}_h${CANDIDATE_MAX_HEADS}_max${MAX_PER_LABEL}}"
TEACHER_JSONL="${TEACHER_JSONL:-${TEACHER_DIR}/online_causal_head_teacher.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${TEACHER_DIR}/visual_anchor_cases_t${POSITIVE_UTILITY_THRESHOLD}_p${PARALLEL_COS_THRESHOLD}_o${ORTHOGONAL_ABS_COS_THRESHOLD}}"

if [ ! -f "${TEACHER_JSONL}" ]; then
    echo "[error] missing teacher jsonl: ${TEACHER_JSONL}" >&2
    echo "[hint] run bash_scripts/soft_routing/run_online_causal_head_teacher.sh first" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "[info] teacher jsonl: ${TEACHER_JSONL}"
echo "[info] output dir: ${OUTPUT_DIR}"
echo "[info] AD-HH threshold: ${ADHH_THRESHOLD}"
echo "[info] positive utility threshold: ${POSITIVE_UTILITY_THRESHOLD}"
echo "[info] parallel cosine threshold: ${PARALLEL_COS_THRESHOLD}"
echo "[info] orthogonal abs cosine threshold: ${ORTHOGONAL_ABS_COS_THRESHOLD}"

python -m eval_scripts.soft_routing.analyze_visual_anchor_cases \
    --teacher-jsonl "${TEACHER_JSONL}" \
    --output-dir "${OUTPUT_DIR}" \
    --adhh-threshold "${ADHH_THRESHOLD}" \
    --positive-utility-threshold "${POSITIVE_UTILITY_THRESHOLD}" \
    --parallel-cos-threshold "${PARALLEL_COS_THRESHOLD}" \
    --orthogonal-abs-cos-threshold "${ORTHOGONAL_ABS_COS_THRESHOLD}" \
    --case1-rho-sweep "${CASE1_RHO_SWEEP}"

echo "[summary] visual anchor cases"
if [ -f "${OUTPUT_DIR}/visual_anchor_case_summary.csv" ]; then
    column -s, -t "${OUTPUT_DIR}/visual_anchor_case_summary.csv" | head -100
fi

echo "[summary] threshold sweep"
if [ -f "${OUTPUT_DIR}/visual_anchor_case_threshold_sweep.csv" ]; then
    column -s, -t "${OUTPUT_DIR}/visual_anchor_case_threshold_sweep.csv" | head -80
fi

echo "[summary] case1 grounded-vs-hallucinated alignment"
if [ -f "${OUTPUT_DIR}/case1_alignment_grounded_vs_hallucinated.csv" ]; then
    column -s, -t "${OUTPUT_DIR}/case1_alignment_grounded_vs_hallucinated.csv" | head -80
fi

echo "[summary] case1 rho protection sweep"
if [ -f "${OUTPUT_DIR}/case1_alignment_threshold_sweep.csv" ]; then
    column -s, -t "${OUTPUT_DIR}/case1_alignment_threshold_sweep.csv" | head -120
fi

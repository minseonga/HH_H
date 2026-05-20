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
FORCE="${FORCE:-0}"

OVERLAP_TOP_K="${OVERLAP_TOP_K:-12}"
OVERLAP_SCORE_FIELD="${OVERLAP_SCORE_FIELD:-overlap_score}"
UNSUPPORTED_COMPONENT_GAMMA="${UNSUPPORTED_COMPONENT_GAMMA:-0.75}"
UNSUPPORTED_COMPONENT_RISK_FEATURE="${UNSUPPORTED_COMPONENT_RISK_FEATURE:-unsupported_norm}"

BASE_RESULT_PATH="${BASE_RESULT_PATH:-./results/${DATASET}/soft_routing_smoke_n${NUM_SAMPLES}_seed${SEED}_tau${ADHH_THRESHOLD}_T${SOFT_TEMPERATURE}}"
TEACHER_DIR="${TEACHER_DIR:-${BASE_RESULT_PATH}/online_causal_head_teacher_${CANDIDATE_POLICY}_h${CANDIDATE_MAX_HEADS}_max${MAX_PER_LABEL}}"
HEATMAP_DIR="${HEATMAP_DIR:-${TEACHER_DIR}/unsupported_head_heatmaps}"
OVERLAP_SUMMARY="${OVERLAP_SUMMARY:-${HEATMAP_DIR}/unsupported_positive_overlap_summary.csv}"
OVERLAP_POOL_DIR="${OVERLAP_POOL_DIR:-${BASE_RESULT_PATH}/overlap_candidate_pools}"
OVERLAP_POOL_PATH="${OVERLAP_POOL_PATH:-${OVERLAP_POOL_DIR}/overlap_top${OVERLAP_TOP_K}_${OVERLAP_SCORE_FIELD}.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${BASE_RESULT_PATH}/online_unsupported_overlap_top${OVERLAP_TOP_K}_${UNSUPPORTED_COMPONENT_RISK_FEATURE}_g${UNSUPPORTED_COMPONENT_GAMMA}}"

if [ ! -f "${OVERLAP_SUMMARY}" ]; then
    echo "[error] missing overlap summary: ${OVERLAP_SUMMARY}" >&2
    echo "[hint] run bash_scripts/soft_routing/run_unsupported_head_heatmaps.sh first, or set OVERLAP_SUMMARY" >&2
    exit 1
fi

mkdir -p "${OVERLAP_POOL_DIR}"

echo "[info] overlap summary: ${OVERLAP_SUMMARY}"
echo "[info] overlap pool: ${OVERLAP_POOL_PATH}"
echo "[info] overlap top-k: ${OVERLAP_TOP_K}"
echo "[info] score field: ${OVERLAP_SCORE_FIELD}"

if [ "${FORCE}" = "1" ] || [ ! -f "${OVERLAP_POOL_PATH}" ]; then
    python -m eval_scripts.soft_routing.build_overlap_candidate_pool \
        --overlap-summary "${OVERLAP_SUMMARY}" \
        --output-path "${OVERLAP_POOL_PATH}" \
        --top-k "${OVERLAP_TOP_K}" \
        --score-field "${OVERLAP_SCORE_FIELD}"
else
    echo "[skip] overlap pool exists: ${OVERLAP_POOL_PATH}"
fi

DATASET="${DATASET}" \
NUM_SAMPLES="${NUM_SAMPLES}" \
SEED="${SEED}" \
ADHH_THRESHOLD="${ADHH_THRESHOLD}" \
SOFT_TEMPERATURE="${SOFT_TEMPERATURE}" \
FORCE="${FORCE}" \
UNSUPPORTED_COMPONENT_HEAD_PATH="${OVERLAP_POOL_PATH}" \
UNSUPPORTED_COMPONENT_HEAD_TOP_K="${OVERLAP_TOP_K}" \
UNSUPPORTED_COMPONENT_RISK_FEATURE="${UNSUPPORTED_COMPONENT_RISK_FEATURE}" \
UNSUPPORTED_COMPONENT_GAMMA="${UNSUPPORTED_COMPONENT_GAMMA}" \
UNSUPPORTED_COMPONENT_ALL_HEADS=0 \
OUTPUT_DIR="${OUTPUT_DIR}" \
bash bash_scripts/soft_routing/run_online_unsupported_component_experiments.sh

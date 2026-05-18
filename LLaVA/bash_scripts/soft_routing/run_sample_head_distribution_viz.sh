#!/bin/bash

set -euo pipefail

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-True}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-user}}"

GPU_ID="${GPU_ID:-0}"
DATASET="${DATASET:-coco}"
NUM_SAMPLES="${NUM_SAMPLES:-500}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SEED="${SEED:-42}"
ADHH_THRESHOLD="${ADHH_THRESHOLD:-0.4}"
SOFT_TEMPERATURE="${SOFT_TEMPERATURE:-0.05}"
TOP_POOL_K="${TOP_POOL_K:-100}"
FORCE="${FORCE:-0}"

NORM_Q_THRESHOLD="${NORM_Q_THRESHOLD:-75}"
NORM_Q_LOW="${NORM_Q_LOW:-50}"
NORM_Q_HIGH="${NORM_Q_HIGH:-90}"
NORM_FIELD="${NORM_FIELD:-text_value_norm}"
WIDE_GATE_TEXT_HIGH="${WIDE_GATE_TEXT_HIGH:-0.9}"

BASE_RESULT_PATH="${BASE_RESULT_PATH:-./results/${DATASET}/soft_routing_smoke_n${NUM_SAMPLES}_seed${SEED}_tau${ADHH_THRESHOLD}_T${SOFT_TEMPERATURE}}"
ATTRIBUTION_RESULT="${ATTRIBUTION_RESULT:-${BASE_RESULT_PATH}/identify_attention_head_val_calib200_full1024/attribution_result.json}"
if [ ! -f "${ATTRIBUTION_RESULT}" ]; then
    ATTRIBUTION_RESULT="${ATTRIBUTION_RESULT_FALLBACK:-./results/coco/llava_3000/identify_attention_head/attribution_result.json}"
fi

POOL_DIR="${POOL_DIR:-${BASE_RESULT_PATH}/contrastive_candidate_pools}"
CANDIDATE_HEAD_PATH="${CANDIDATE_HEAD_PATH:-${POOL_DIR}/contrastive_top${TOP_POOL_K}.json}"
BEHAVIORAL_RECORDS="${BEHAVIORAL_RECORDS:-${BASE_RESULT_PATH}/behavioral_head_overlap_prefill_n500_minlayer13/all_head_records.jsonl}"
NORM_THRESHOLDS_PATH="${NORM_THRESHOLDS_PATH:-${POOL_DIR}/text_value_norm_q${NORM_Q_THRESHOLD}_top${TOP_POOL_K}.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${BASE_RESULT_PATH}/sample_head_distribution_top${TOP_POOL_K}_normq${NORM_Q_THRESHOLD}}"

mkdir -p "${POOL_DIR}" "${OUTPUT_DIR}" "${MPLCONFIGDIR}"

if [ ! -f "${ATTRIBUTION_RESULT}" ]; then
    echo "[error] missing attribution result: ${ATTRIBUTION_RESULT}" >&2
    exit 1
fi

echo "[info] attribution result: ${ATTRIBUTION_RESULT}"
echo "[info] candidate heads: ${CANDIDATE_HEAD_PATH}"
echo "[info] behavioral records: ${BEHAVIORAL_RECORDS}"
echo "[info] norm thresholds: ${NORM_THRESHOLDS_PATH}"
echo "[info] output dir: ${OUTPUT_DIR}"

if [ "${FORCE}" = "1" ] || [ ! -f "${CANDIDATE_HEAD_PATH}" ]; then
    python -m eval_scripts.soft_routing.build_contrastive_candidate_pool \
        --attribution-result "${ATTRIBUTION_RESULT}" \
        --output-path "${CANDIDATE_HEAD_PATH}" \
        --top-k "${TOP_POOL_K}"
else
    echo "[skip] candidate pool exists: ${CANDIDATE_HEAD_PATH}"
fi

if [ ! -f "${BEHAVIORAL_RECORDS}" ]; then
    echo "[error] missing behavioral records: ${BEHAVIORAL_RECORDS}" >&2
    echo "[hint] run behavioral logging first, or let wide-pool experiment script build it:" >&2
    echo "       bash bash_scripts/soft_routing/run_wide_pool_gate_experiments.sh" >&2
    exit 1
fi

if [ "${FORCE}" = "1" ] || [ ! -f "${NORM_THRESHOLDS_PATH}" ]; then
    python -m eval_scripts.soft_routing.build_head_norm_thresholds \
        --records-jsonl "${BEHAVIORAL_RECORDS}" \
        --output-path "${NORM_THRESHOLDS_PATH}" \
        --head-path "${CANDIDATE_HEAD_PATH}" \
        --top-k "${TOP_POOL_K}" \
        --norm-field "${NORM_FIELD}" \
        --q-threshold "${NORM_Q_THRESHOLD}" \
        --q-low "${NORM_Q_LOW}" \
        --q-high "${NORM_Q_HIGH}"
else
    echo "[skip] norm thresholds exist: ${NORM_THRESHOLDS_PATH}"
fi

python -m eval_scripts.soft_routing.visualize_sample_head_distribution \
    --records-jsonl "${BEHAVIORAL_RECORDS}" \
    --candidate-head-path "${CANDIDATE_HEAD_PATH}" \
    --head-norm-thresholds-path "${NORM_THRESHOLDS_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --top-k "${TOP_POOL_K}" \
    --max-samples "${MAX_SAMPLES}" \
    --text-tau "${ADHH_THRESHOLD}" \
    --text-high "${WIDE_GATE_TEXT_HIGH}" \
    --norm-field "${NORM_FIELD}"

echo "[outputs] visualization files"
find "${OUTPUT_DIR}" -maxdepth 1 \( -name "*.png" -o -name "*.csv" -o -name "metadata.json" \) -print | sort

echo "[summary] per-sample active head distribution"
if [ -f "${OUTPUT_DIR}/sample_gate_summary.csv" ]; then
    column -s, -t "${OUTPUT_DIR}/sample_gate_summary.csv" | head -30
fi

echo "[summary] per-head activation rates"
if [ -f "${OUTPUT_DIR}/head_gate_summary.csv" ]; then
    column -s, -t "${OUTPUT_DIR}/head_gate_summary.csv" | head -40
fi

echo "[summary] sample pairwise active-set similarity"
if [ -f "${OUTPUT_DIR}/sample_pairwise_jaccard_summary.csv" ]; then
    column -s, -t "${OUTPUT_DIR}/sample_pairwise_jaccard_summary.csv"
fi

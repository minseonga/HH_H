#!/bin/bash

set -euo pipefail

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-user}}"
mkdir -p "${MPLCONFIGDIR}"

DATASET="${DATASET:-coco}"
NUM_SAMPLES="${NUM_SAMPLES:-500}"
SEED="${SEED:-42}"
ADHH_THRESHOLD="${ADHH_THRESHOLD:-0.4}"
SOFT_TEMPERATURE="${SOFT_TEMPERATURE:-0.05}"
MAX_PER_LABEL="${MAX_PER_LABEL:-100}"
HALLUCINATED_SOURCE="${HALLUCINATED_SOURCE:-both}"
QUERY_NORMALIZATION="${QUERY_NORMALIZATION:-l2}"
MIN_LAYER="${MIN_LAYER:-13}"
MAX_LAYER="${MAX_LAYER:-31}"
VIZ_TOP_K="${VIZ_TOP_K:-6}"
DIRECTION_FILTER="${DIRECTION_FILTER:-high}"

BASE_RESULT_PATH="${BASE_RESULT_PATH:-./results/${DATASET}/soft_routing_smoke_n${NUM_SAMPLES}_seed${SEED}_tau${ADHH_THRESHOLD}_T${SOFT_TEMPERATURE}}"
QUERY_PROBE_DIR="${QUERY_PROBE_DIR:-${BASE_RESULT_PATH}/query_direction_probe_${QUERY_NORMALIZATION}_l${MIN_LAYER}_${MAX_LAYER}_hall${HALLUCINATED_SOURCE}_max${MAX_PER_LABEL}}"
OUTPUT_DIR="${OUTPUT_DIR:-${QUERY_PROBE_DIR}/query_direction_viz}"

if [ ! -f "${QUERY_PROBE_DIR}/query_direction_auc.csv" ]; then
    echo "[error] missing query direction probe outputs: ${QUERY_PROBE_DIR}" >&2
    echo "[hint] run bash_scripts/soft_routing/run_query_direction_probe.sh first" >&2
    exit 1
fi

if [ ! -f "${QUERY_PROBE_DIR}/query_vectors.npz" ]; then
    echo "[warn] missing ${QUERY_PROBE_DIR}/query_vectors.npz"
    echo "[warn] heatmap will be generated, but score distributions need SAVE_QUERY_VECTORS=1 during calibration"
fi

python -m eval_scripts.soft_routing.visualize_query_directions \
    --query-probe-dir "${QUERY_PROBE_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --top-k "${VIZ_TOP_K}" \
    --direction-filter "${DIRECTION_FILTER}" \
    --query-normalization "${QUERY_NORMALIZATION}"

echo "[summary] output dir: ${OUTPUT_DIR}"
if [ -f "${OUTPUT_DIR}/selected_top_query_directions.csv" ]; then
    column -s, -t "${OUTPUT_DIR}/selected_top_query_directions.csv" | head -30
fi
if [ -f "${OUTPUT_DIR}/query_direction_score_summary.csv" ]; then
    echo "[summary] score distribution summary"
    column -s, -t "${OUTPUT_DIR}/query_direction_score_summary.csv" | head -60
fi
if [ -f "${OUTPUT_DIR}/query_direction_score_examples.csv" ]; then
    echo "[summary] high/low score examples"
    column -s, -t "${OUTPUT_DIR}/query_direction_score_examples.csv" | head -60
fi

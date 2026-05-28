#!/bin/bash

set -euo pipefail

QUERY_ALIGNMENT_DIR="${QUERY_ALIGNMENT_DIR:-}"
MENTIONS_CSV="${MENTIONS_CSV:-${QUERY_ALIGNMENT_DIR}/query_direction_chair_mentions.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-${QUERY_ALIGNMENT_DIR}/query_direction_selectivity}"
FEATURES="${FEATURES:-auto}"
FPR_BUDGETS="${FPR_BUDGETS:-0.05,0.10,0.20,0.30,0.50}"
RECALL_TARGETS="${RECALL_TARGETS:-0.50,0.70,0.80,0.90}"
BUDGET_RATES="${BUDGET_RATES:-0.05,0.10,0.20,0.30}"
PLOT_TOP_N="${PLOT_TOP_N:-6}"

if [ -z "${QUERY_ALIGNMENT_DIR}" ] && [ -z "${MENTIONS_CSV}" ]; then
    echo "[error] set QUERY_ALIGNMENT_DIR or MENTIONS_CSV" >&2
    exit 2
fi

if [ ! -f "${MENTIONS_CSV}" ]; then
    echo "[error] missing mentions CSV: ${MENTIONS_CSV}" >&2
    echo "[hint] run bash_scripts/soft_routing/run_query_direction_chair_alignment.sh first" >&2
    exit 2
fi

python -m eval_scripts.soft_routing.analyze_query_direction_selectivity \
    --mentions-csv "${MENTIONS_CSV}" \
    --output-dir "${OUTPUT_DIR}" \
    --features "${FEATURES}" \
    --fpr-budgets "${FPR_BUDGETS}" \
    --recall-targets "${RECALL_TARGETS}" \
    --budget-rates "${BUDGET_RATES}" \
    --plot-top-n "${PLOT_TOP_N}"

echo "[summary] operating points"
column -s, -t "${OUTPUT_DIR}/query_direction_selectivity_operating_points.csv" | head -80

echo "[done] ${OUTPUT_DIR}/query_direction_selectivity.md"

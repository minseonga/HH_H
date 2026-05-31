#!/bin/bash

set -euo pipefail

export MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

SUMMARY_CSV="${SUMMARY_CSV:-./results/coco/method_claim_ablation_n500_seed42_core6/method_claim_ablation_summary.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "${SUMMARY_CSV}")/method_axis_outcome_figure}"
FIGURE_FORMATS="${FIGURE_FORMATS:-png,pdf,svg}"

if [ ! -f "${SUMMARY_CSV}" ]; then
    echo "[error] missing method summary CSV: ${SUMMARY_CSV}" >&2
    echo "[hint] set SUMMARY_CSV to method_claim_ablation_summary.csv" >&2
    exit 1
fi

python -m eval_scripts.soft_routing.build_method_axis_outcome_figure \
    --summary-csv "${SUMMARY_CSV}" \
    --output-dir "${OUTPUT_DIR}" \
    --formats "${FIGURE_FORMATS}"

echo "[summary] selected methods"
column -s, -t "${OUTPUT_DIR}/method_axis_outcome_selected.csv"

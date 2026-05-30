#!/bin/bash

set -euo pipefail

export MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

SOURCE_DIR="${SOURCE_DIR:-./results/coco/method_figure_source_trace_n100_k150_l9_16}"
OUTPUT_DIR="${OUTPUT_DIR:-${SOURCE_DIR}/figures}"
TOP_K="${TOP_K:-0}"
SELECTION_LAYERS="${SELECTION_LAYERS:-}"
RATIO_SOURCE="${RATIO_SOURCE:-selected}"
FIGURE_FORMATS="${FIGURE_FORMATS:-png,pdf,svg}"
CAPTION_TEXT="${CAPTION_TEXT:-}"

if [ ! -f "${SOURCE_DIR}/head_scores_all.csv" ]; then
    echo "[error] missing source trace files in: ${SOURCE_DIR}" >&2
    echo "[hint] run bash_scripts/soft_routing/run_method_figure_source_trace.sh first" >&2
    exit 1
fi

top_k_args=()
if [ "${TOP_K}" != "0" ]; then
    top_k_args=(--top-k "${TOP_K}")
fi

selection_layer_args=()
if [ -n "${SELECTION_LAYERS}" ]; then
    selection_layer_args=(--selection-layers "${SELECTION_LAYERS}")
fi

python -m eval_scripts.soft_routing.visualize_method_figure_source_data \
    --source-dir "${SOURCE_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    "${top_k_args[@]}" \
    "${selection_layer_args[@]}" \
    --ratio-source "${RATIO_SOURCE}" \
    --formats "${FIGURE_FORMATS}" \
    --caption-text "${CAPTION_TEXT}"

echo "[summary] visualization outputs"
cat "${OUTPUT_DIR}/method_figure_visualization_manifest.json"

echo
echo "[summary] numeric summary"
column -s, -t "${OUTPUT_DIR}/method_figure_visualization_numeric_summary.csv"

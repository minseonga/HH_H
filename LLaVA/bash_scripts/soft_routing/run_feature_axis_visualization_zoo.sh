#!/bin/bash

set -euo pipefail

export MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

SOURCE_DIR="${SOURCE_DIR:-./results/coco/method_figure_source_trace_n100_k150_l9_16}"
OUTPUT_DIR="${OUTPUT_DIR:-${SOURCE_DIR}/feature_axis_visualization_zoo}"
SELECTION_LAYERS="${SELECTION_LAYERS:-}"
FEATURE_TOP_K="${FEATURE_TOP_K:-0}"
FIGURE_FORMATS="${FIGURE_FORMATS:-png,pdf,svg}"

if [ ! -f "${SOURCE_DIR}/head_scores_all.csv" ]; then
    echo "[error] missing head_scores_all.csv in: ${SOURCE_DIR}" >&2
    echo "[hint] run bash_scripts/soft_routing/run_method_figure_source_trace.sh first" >&2
    exit 1
fi

selection_layer_args=()
if [ -n "${SELECTION_LAYERS}" ]; then
    selection_layer_args=(--selection-layers "${SELECTION_LAYERS}")
fi

python -m eval_scripts.soft_routing.build_feature_axis_visualization_zoo \
    --source-dir "${SOURCE_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    "${selection_layer_args[@]}" \
    --feature-top-k "${FEATURE_TOP_K}" \
    --formats "${FIGURE_FORMATS}"

echo "[summary] feature-axis visualization zoo"
cat "${OUTPUT_DIR}/feature_axis_visualization_zoo_manifest.json"

echo
echo "[summary] feature-axis groups"
column -s, -t "${OUTPUT_DIR}/feature_axis_visualization_zoo_summary.csv"

echo
echo "[summary] independent top-k feature sets"
column -s, -t "${OUTPUT_DIR}/feature_axis_independent_topk_summary.csv"

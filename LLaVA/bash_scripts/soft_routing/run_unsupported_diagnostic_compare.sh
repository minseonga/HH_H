#!/bin/bash

set -euo pipefail

RUNS="${RUNS:?set RUNS to space-separated name=diagnostic_summary_dir entries}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/unsupported_component_diagnostic_compare}"

args=()
for item in ${RUNS}; do
    args+=(--run "${item}")
done

mkdir -p "${OUTPUT_DIR}"

python -m eval_scripts.soft_routing.compare_unsupported_component_diagnostics \
    "${args[@]}" \
    --output-dir "${OUTPUT_DIR}"

echo "[summary] combined layer calls"
column -s, -t "${OUTPUT_DIR}/combined_layer_call_summary.csv"

echo "[summary] combined step summary"
column -s, -t "${OUTPUT_DIR}/combined_step_summary.csv"

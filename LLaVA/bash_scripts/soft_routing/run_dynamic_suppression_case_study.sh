#!/usr/bin/env bash
set -euo pipefail

TRACE_CSV=${TRACE_CSV:-../experiments_in_server/method_figure_source_trace_n100_k150_l9_16/selected_head_object_ratio_distribution.csv}
SAMPLES_CSV=${SAMPLES_CSV:-../experiments_in_server/method_figure_source_trace_n100_k150_l9_16/samples.csv}
OUTPUT_DIR=${OUTPUT_DIR:-./results/coco/dynamic_suppression_case_study_l9_l16_k150}
QUESTION_ID=${QUESTION_ID:-}
MAX_HEADS=${MAX_HEADS:-35}
FIGURE_FORMATS=${FIGURE_FORMATS:-png,pdf,svg}

args=(
  --trace-csv "${TRACE_CSV}"
  --samples-csv "${SAMPLES_CSV}"
  --output-dir "${OUTPUT_DIR}"
  --max-heads "${MAX_HEADS}"
  --formats "${FIGURE_FORMATS}"
)

if [[ -n "${QUESTION_ID}" ]]; then
  args+=(--question-id "${QUESTION_ID}")
fi

python eval_scripts/soft_routing/build_dynamic_suppression_case_study.py "${args[@]}"

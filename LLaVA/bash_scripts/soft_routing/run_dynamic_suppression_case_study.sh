#!/usr/bin/env bash
set -euo pipefail

TRACE_CSV=${TRACE_CSV:-../experiments_in_server/method_figure_source_trace_n100_k150_l9_16/selected_head_object_ratio_distribution.csv}
SAMPLES_CSV=${SAMPLES_CSV:-../experiments_in_server/method_figure_source_trace_n100_k150_l9_16/samples.csv}
OUTPUT_DIR=${OUTPUT_DIR:-./results/coco/dynamic_suppression_case_study_l9_l16_k150}
QUESTION_ID=${QUESTION_ID:-}
MAX_HEADS=${MAX_HEADS:-35}
FIGURE_FORMATS=${FIGURE_FORMATS:-png,pdf,svg}
PREFER_UNIQUE_OBJECT_TOKENS=${PREFER_UNIQUE_OBJECT_TOKENS:-0}
UNIQUE_OBJECT_TOKENS=${UNIQUE_OBJECT_TOKENS:-0}
MIN_UNIQUE_READABLE_OBJECT_TOKENS=${MIN_UNIQUE_READABLE_OBJECT_TOKENS:-0}
TOKEN_LABEL_MAP=${TOKEN_LABEL_MAP:-}

args=(
  --trace-csv "${TRACE_CSV}"
  --samples-csv "${SAMPLES_CSV}"
  --output-dir "${OUTPUT_DIR}"
  --max-heads "${MAX_HEADS}"
  --formats "${FIGURE_FORMATS}"
  --min-unique-readable-object-tokens "${MIN_UNIQUE_READABLE_OBJECT_TOKENS}"
)

if [[ -n "${QUESTION_ID}" ]]; then
  args+=(--question-id "${QUESTION_ID}")
fi
if [[ "${PREFER_UNIQUE_OBJECT_TOKENS}" == "1" ]]; then
  args+=(--prefer-unique-object-tokens)
fi
if [[ "${UNIQUE_OBJECT_TOKENS}" == "1" ]]; then
  args+=(--unique-object-tokens)
fi
if [[ -n "${TOKEN_LABEL_MAP}" ]]; then
  args+=(--token-label-map "${TOKEN_LABEL_MAP}")
fi

python eval_scripts/soft_routing/build_dynamic_suppression_case_study.py "${args[@]}"

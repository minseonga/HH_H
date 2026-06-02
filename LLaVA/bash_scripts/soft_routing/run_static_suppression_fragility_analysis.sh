#!/usr/bin/env bash
set -euo pipefail

SAMPLES_CSV=${SAMPLES_CSV:-../experiments_in_server/method_figure_source_trace_n100_k150_l9_16/samples.csv}
OBJECT_RATIO_CSV=${OBJECT_RATIO_CSV:-../experiments_in_server/method_figure_source_trace_n100_k150_l9_16/selected_head_object_ratio_distribution.csv}
TEACHER_JSONL=${TEACHER_JSONL:-../experiments_in_server/soft_routing_smoke_n500_seed42_tau0.4_T0.05/online_causal_head_teacher_text_topk_h64_max30/online_causal_head_teacher.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-./results/coco/static_suppression_fragility_analysis_l9_l16_k150}
FIGURE_FORMATS=${FIGURE_FORMATS:-png,pdf,svg}

python eval_scripts/soft_routing/analyze_static_suppression_fragility.py \
  --samples-csv "${SAMPLES_CSV}" \
  --object-ratio-csv "${OBJECT_RATIO_CSV}" \
  --teacher-jsonl "${TEACHER_JSONL}" \
  --output-dir "${OUTPUT_DIR}" \
  --formats "${FIGURE_FORMATS}"

#!/usr/bin/env bash
set -euo pipefail

RANKED_HEADS_JSON=${RANKED_HEADS_JSON:-../ADHH/LLaVA/results_summary/coco/ranked_heads_global__itext_all__C_toi_HminusG.json}
TOP_K=${TOP_K:-100}
OUTPUT_DIR=${OUTPUT_DIR:-./results/coco/selected_head_actuator_analysis_l9_l16_k100}
FIGURE_FORMATS=${FIGURE_FORMATS:-png,pdf,svg}

python eval_scripts/soft_routing/analyze_selected_head_actuators.py \
  --ranked-heads-json "${RANKED_HEADS_JSON}" \
  --top-k "${TOP_K}" \
  --output-dir "${OUTPUT_DIR}" \
  --formats "${FIGURE_FORMATS}"

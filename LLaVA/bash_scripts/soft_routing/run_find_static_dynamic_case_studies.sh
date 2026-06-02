#!/usr/bin/env bash
set -euo pipefail

BASE_DIR=${BASE_DIR:-../experiments_in_server/soft_routing_smoke_n500_seed42_tau0.4_T0.05}
GREEDY_JSON=${GREEDY_JSON:-${BASE_DIR}/greedy/captions_eval_results.json}
STATIC_JSON=${STATIC_JSON:-${BASE_DIR}/hard_tau0.4/captions_eval_results.json}
DYNAMIC_JSON=${DYNAMIC_JSON:-${BASE_DIR}/dynamic_v1_g1.0_m1.0_r0.0_c0.0_b0.0/captions_eval_results.json}
OUTPUT_DIR=${OUTPUT_DIR:-./results/coco/static_fail_dynamic_success_cases}
TOP_N=${TOP_N:-20}

python eval_scripts/soft_routing/find_static_dynamic_case_studies.py \
  --greedy-json "${GREEDY_JSON}" \
  --static-json "${STATIC_JSON}" \
  --dynamic-json "${DYNAMIC_JSON}" \
  --output-dir "${OUTPUT_DIR}" \
  --top-n "${TOP_N}"

#!/bin/bash

set -euo pipefail

DATASET="${DATASET:-coco}"
BASE_RESULT_PATH="${BASE_RESULT_PATH:-./results/${DATASET}/soft_routing_smoke_n500_seed42_tau0.4_T0.05}"
CONTRIBUTION_DIR="${CONTRIBUTION_DIR:-${BASE_RESULT_PATH}/contribution_gap_validation_n80}"
HEAD_ROWS="${HEAD_ROWS:-${CONTRIBUTION_DIR}/head_logit_proxy_ablation_rows.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-${CONTRIBUTION_DIR}/hallucination_leverage_head_discovery}"

TEACHER_FEATURE="${TEACHER_FEATURE:-target_text_logprob_drop}"
TOP_KS="${TOP_KS:-20,40}"
LABEL_FILTER="${LABEL_FILTER:-hallucinated}"

mkdir -p "${OUTPUT_DIR}"

echo "[info] output dir: ${OUTPUT_DIR}"
echo "[info] head rows: ${HEAD_ROWS}"
echo "[info] teacher: ${TEACHER_FEATURE}"
echo "[info] label filter: ${LABEL_FILTER}"
echo "[info] top ks: ${TOP_KS}"

python -m eval_scripts.soft_routing.discover_hallucination_leverage_heads \
    --head-rows "${HEAD_ROWS}" \
    --output-dir "${OUTPUT_DIR}" \
    --teacher-feature "${TEACHER_FEATURE}" \
    --label-filter "${LABEL_FILTER}" \
    --top-ks "${TOP_KS}"

echo "[summary] score distribution"
column -s, -t "${OUTPUT_DIR}/hallucination_leverage_head_distribution.csv"

echo "[summary] top heads by positive mean drop"
column -s, -t "${OUTPUT_DIR}/hallucination_leverage_head_scores.csv" | head -60

echo "[summary] selections"
column -s, -t "${OUTPUT_DIR}/hallucination_leverage_head_selection_summary.csv"

echo "[summary] top20 positive-mean heads"
cat "${OUTPUT_DIR}/hallucination_leverage_positive_mean_top20_heads.txt"

if [ -f "${OUTPUT_DIR}/hallucination_leverage_positive_mean_top40_heads.txt" ]; then
    echo "[summary] top40 positive-mean heads"
    cat "${OUTPUT_DIR}/hallucination_leverage_positive_mean_top40_heads.txt"
fi

#!/bin/bash

set -euo pipefail

# Recompute Section III-D causal fragility on the exact DEACT head pool:
# top-100, L9-L16, global__itext_all__C_toi_HminusG.
#
# This is intentionally a diagnostic hard/static text-side suppression probe.
# It is not the proposed DEACT decoding run and not the original AD-HH fixed
# head set.

GPU_ID="${GPU_ID:-0}"
DATASET="${DATASET:-coco}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
TOP_K="${TOP_K:-100}"
MAX_PER_LABEL="${MAX_PER_LABEL:-200}"
ADHH_THRESHOLD="${ADHH_THRESHOLD:-0.4}"
SOFT_GAMMA="${SOFT_GAMMA:-0.75}"
SOFT_TEMPERATURE="${SOFT_TEMPERATURE:-0.05}"

BASE_RESULTS="${BASE_RESULTS:-./results/coco/verify_0288_dynamic_l9_l16_k100_s1_q8_tau0p90_n500_seed42/greedy/captions_eval_results.json}"
PRIOR_PATH="${PRIOR_PATH:-../ADHH/LLaVA/results_l9_l16/coco/llava-v1.5-7b_base_original_qa_n500_txtattn_l9_l16_allheads/surrogate_score_zoo/ranked_heads_global__itext_all__C_toi_HminusG.json}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/coco/section_d_exact_top100_l9_l16_fragility_m${MAX_PER_LABEL}}"
LOG_DIR="${LOG_DIR:-./logs/soft_routing}"

if [ ! -f "${BASE_RESULTS}" ]; then
    echo "[error] missing BASE_RESULTS: ${BASE_RESULTS}" >&2
    echo "[hint] set BASE_RESULTS to the exact greedy n500 captions_eval_results.json used by the 0.288 run" >&2
    exit 1
fi

if [ ! -f "${PRIOR_PATH}" ]; then
    echo "[error] missing PRIOR_PATH: ${PRIOR_PATH}" >&2
    echo "[hint] set PRIOR_PATH to ranked_heads_global__itext_all__C_toi_HminusG.json for L9-L16" >&2
    exit 1
fi

mkdir -p "${LOG_DIR}"

echo "[info] base results: ${BASE_RESULTS}"
echo "[info] prior path: ${PRIOR_PATH}"
echo "[info] top k: ${TOP_K}"
echo "[info] hard/static threshold: ${ADHH_THRESHOLD}"
echo "[info] output dir: ${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" python -m eval_scripts.soft_routing.analyze_static_object_logprob_drop \
    --base-results "${BASE_RESULTS}" \
    --image-folder "${IMAGE_FOLDER}" \
    --output-dir "${OUTPUT_DIR}" \
    --prior-path "${PRIOR_PATH}" \
    --top-k "${TOP_K}" \
    --model-path "${MODEL_PATH}" \
    --conv-mode vicuna_v1 \
    --max-per-label "${MAX_PER_LABEL}" \
    --adhh-threshold "${ADHH_THRESHOLD}" \
    --soft-gamma "${SOFT_GAMMA}" \
    --soft-temperature "${SOFT_TEMPERATURE}" \
    2>&1 | tee "${LOG_DIR}/section_d_exact_top${TOP_K}_l9_l16_fragility.log"

echo "[summary] Section D exact top-${TOP_K} L9-L16 fragility"
if [ -f "${OUTPUT_DIR}/static_object_logprob_drop_summary.csv" ]; then
    column -s, -t "${OUTPUT_DIR}/static_object_logprob_drop_summary.csv"
fi

echo "[done] ${OUTPUT_DIR}"

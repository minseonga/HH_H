#!/bin/bash

set -euo pipefail

RANKED_HEADS="${RANKED_HEADS:-../ADHH/LLaVA/results/coco/llava-v1.5-7b_base_original_qa_n3000/surrogate_hh_scores/surrogate_score_zoo/ranked_heads_global__itext_all__C_toi_HminusG.json}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/coco/contrastive_dynamic_head_pool_analysis}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
TOP_KS="${TOP_KS:-20,30,50,100,150,200}"
BASE_EVAL_JSON="${BASE_EVAL_JSON:-../ADHH/LLaVA/results/coco/llava-v1.5-7b_base_n500/captions_eval_results.json}"
ADHH_EVAL_JSON="${ADHH_EVAL_JSON:-../ADHH/LLaVA/results/coco/llava-v1.5-7b_adhh_file_k20_tau0.4_n500_real/captions_eval_results.json}"
DYNAMIC_EVAL_GLOB="${DYNAMIC_EVAL_GLOB:-../ADHH/LLaVA/results_dynamic/coco/*global__itext_all__C_toi_HminusG/captions_eval_results.json}"

python -m eval_scripts.soft_routing.analyze_contrastive_dynamic_head_pool \
    --ranked-heads "${RANKED_HEADS}" \
    --output-dir "${OUTPUT_DIR}" \
    --model-path "${MODEL_PATH}" \
    --top-ks "${TOP_KS}" \
    --base-eval-json "${BASE_EVAL_JSON}" \
    --adhh-eval-json "${ADHH_EVAL_JSON}" \
    --dynamic-eval-glob "${DYNAMIC_EVAL_GLOB}"

echo "[summary] head-pool rank buckets"
column -s, -t "${OUTPUT_DIR}/head_pool_rank_bucket_summary.csv"

echo "[summary] AD-HH overlay"
column -s, -t "${OUTPUT_DIR}/head_pool_adhh_overlay.csv"

echo "[summary] local eval metrics"
column -s, -t "${OUTPUT_DIR}/local_eval_metrics.csv"

echo "[done] report: ${OUTPUT_DIR}/method_justification_report.md"

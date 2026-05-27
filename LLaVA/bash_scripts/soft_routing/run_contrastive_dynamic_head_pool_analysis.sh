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

python -m eval_scripts.soft_routing.build_suppression_evidence_figures \
    --ranked-heads "${RANKED_HEADS}" \
    --output-dir "${OUTPUT_DIR}" \
    --model-path "${MODEL_PATH}"

python -m eval_scripts.soft_routing.build_method_claim_evidence \
    --ranked-heads "${RANKED_HEADS}" \
    --output-dir "${OUTPUT_DIR}/method_claim_evidence" \
    --base-eval-json "${BASE_EVAL_JSON}" \
    --method-eval-glob "${DYNAMIC_EVAL_GLOB}"

echo "[summary] head-pool rank buckets"
column -s, -t "${OUTPUT_DIR}/head_pool_rank_bucket_summary.csv"

echo "[summary] AD-HH overlay"
column -s, -t "${OUTPUT_DIR}/head_pool_adhh_overlay.csv"

echo "[summary] architecture findings"
column -s, -t "${OUTPUT_DIR}/head_pool_architecture_findings.csv"

echo "[summary] rejected AD-HH heads"
column -s, -t "${OUTPUT_DIR}/rejected_adhh_heads_by_contrastive_pool.csv"

echo "[summary] local eval metrics"
column -s, -t "${OUTPUT_DIR}/local_eval_metrics.csv"

echo "[summary] suppression evidence"
column -s, -t "${OUTPUT_DIR}/suppression_evidence_summary.csv"

echo "[summary] method claim component split"
column -s, -t "${OUTPUT_DIR}/method_claim_evidence/component_category_summary.csv"

echo "[summary] method claim object changes"
column -s, -t "${OUTPUT_DIR}/method_claim_evidence/object_change_summary.csv"

echo "[done] report: ${OUTPUT_DIR}/method_justification_report.md"
echo "[done] evidence figures: ${OUTPUT_DIR}/suppression_evidence_figures.md"
echo "[done] method claim evidence: ${OUTPUT_DIR}/method_claim_evidence/method_claim_evidence.md"

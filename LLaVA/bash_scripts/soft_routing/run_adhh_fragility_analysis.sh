#!/bin/bash

set -euo pipefail

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-True}"

GPU_ID="${GPU_ID:-0}"
DATASET="${DATASET:-coco}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
ANNOTATION_DIR="${ANNOTATION_DIR:-/home/kms/data/images/mscoco/annotations}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
CAPTION_FILE_PATH="${CAPTION_FILE_PATH:-${ANNOTATION_DIR}/captions_val2014.json}"

NUM_SAMPLES="${NUM_SAMPLES:-100}"
SEED="${SEED:-42}"
ADHH_THRESHOLD="${ADHH_THRESHOLD:-0.4}"
TOP_K="${TOP_K:-20}"
FORCE="${FORCE:-0}"

BASE_RESULT_PATH="${BASE_RESULT_PATH:-./results/${DATASET}/soft_routing_smoke_n500_seed42_tau0.4_T0.05}"
OUTPUT_DIR="${OUTPUT_DIR:-${BASE_RESULT_PATH}/adhh_fragility_n${NUM_SAMPLES}}"
LOG_DIR="${LOG_DIR:-./logs/soft_routing}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

echo "[info] output dir: ${OUTPUT_DIR}"
echo "[info] num samples: ${NUM_SAMPLES}"
echo "[info] image folder: ${IMAGE_FOLDER}"

FORCE="${FORCE}" GPU_ID="${GPU_ID}" NUM_SAMPLES="${NUM_SAMPLES}" \
BASE_RESULT_PATH="${BASE_RESULT_PATH}" \
IMAGE_FOLDER="${IMAGE_FOLDER}" \
CAPTION_FILE_PATH="${CAPTION_FILE_PATH}" \
ANNOTATION_DIR="${ANNOTATION_DIR}" \
RUN_METHODS="greedy adhh_hard" \
OUTPUT_DIR="${OUTPUT_DIR}/caption_runs" \
ADHH_THRESHOLD="${ADHH_THRESHOLD}" \
TOP_K="${TOP_K}" \
bash bash_scripts/soft_routing/run_head_set_suppression_experiments.sh

FEATURE_DIR="${OUTPUT_DIR}/greedy_adhh_object_step_features"
FEATURE_CSV="${FEATURE_DIR}/object_step_features.csv"
if [ "${FORCE}" = "1" ] || [ ! -f "${FEATURE_CSV}" ]; then
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python -m eval_scripts.soft_routing.log_object_step_features \
        --eval-results "${OUTPUT_DIR}/caption_runs/greedy/captions_eval_results.json" \
        --image-folder "${IMAGE_FOLDER}" \
        --model-path "${MODEL_PATH}" \
        --max-samples "${NUM_SAMPLES}" \
        --adhh-threshold "${ADHH_THRESHOLD}" \
        --top-k "${TOP_K}" \
        --head-prior-mode uniform \
        --output-dir "${FEATURE_DIR}" \
        2>&1 | tee "${LOG_DIR}/adhh_fragility_object_step_features.log"
else
    echo "[skip] object-step features: ${FEATURE_CSV}"
fi

python -m eval_scripts.soft_routing.analyze_adhh_fragility_features \
    --base-eval "${OUTPUT_DIR}/caption_runs/greedy/captions_eval_results.json" \
    --target-eval "${OUTPUT_DIR}/caption_runs/adhh_hard/captions_eval_results.json" \
    --object-step-features "${FEATURE_CSV}" \
    --base-name greedy \
    --target-name adhh_hard \
    --output-dir "${OUTPUT_DIR}/fragility_analysis"

echo "[summary] AD-HH caption metrics"
column -s, -t "${OUTPUT_DIR}/caption_runs/head_set_suppression_summary.csv"

echo "[summary] AD-HH fragility"
column -s, -t "${OUTPUT_DIR}/fragility_analysis/adhh_fragility_summary.csv"

echo "[summary] feature coverage"
column -s, -t "${OUTPUT_DIR}/fragility_analysis/adhh_fragility_feature_coverage.csv"

echo "[top AUC] hallucinated removed/retained and grounded lost/retained"
column -s, -t "${OUTPUT_DIR}/fragility_analysis/adhh_fragility_feature_auc.csv" | head -80

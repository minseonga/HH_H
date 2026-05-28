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
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/${DATASET}/vanilla_region_attention_n${NUM_SAMPLES}_seed${SEED}}"
LOG_DIR="${LOG_DIR:-./logs/soft_routing}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

echo "[info] output dir: ${OUTPUT_DIR}"
echo "[info] num samples: ${NUM_SAMPLES}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" python -m eval_scripts.soft_routing.analyze_vanilla_region_attention \
    --model-path "${MODEL_PATH}" \
    --image-folder "${IMAGE_FOLDER}" \
    --caption-file-path "${CAPTION_FILE_PATH}" \
    --dataset "${DATASET}" \
    --num-samples "${NUM_SAMPLES}" \
    --seed "${SEED}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --conv-mode vicuna_v1 \
    --output-dir "${OUTPUT_DIR}" \
    2>&1 | tee "${LOG_DIR}/vanilla_region_attention_n${NUM_SAMPLES}_seed${SEED}.log"

echo "[summary] layer-wise vanilla attention region shares"
if [ -f "${OUTPUT_DIR}/vanilla_region_attention_by_layer.csv" ]; then
    column -s, -t "${OUTPUT_DIR}/vanilla_region_attention_by_layer.csv" | head -40
fi

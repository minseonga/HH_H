#!/bin/bash

set -euo pipefail

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-True}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

GPU_ID="${GPU_ID:-0}"
DATASET="${DATASET:-coco}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
MODEL_BASE="${MODEL_BASE:-}"
ANNOTATION_DIR="${ANNOTATION_DIR:-/home/kms/data/images/mscoco/annotations}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
CAPTION_FILE_PATH="${CAPTION_FILE_PATH:-${ANNOTATION_DIR}/captions_val2014.json}"

NUM_SAMPLES="${NUM_SAMPLES:-100}"
SEED="${SEED:-42}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
TOP_K="${TOP_K:-150}"
SELECTION_LAYERS="${SELECTION_LAYERS:-9-16}"
GATE_STRENGTH="${GATE_STRENGTH:-1.0}"
GATE_BETA="${GATE_BETA:-10}"
GATE_TAU="${GATE_TAU:-0.9}"
SAVE_ALL_HEAD_OBJECT_TRACE="${SAVE_ALL_HEAD_OBJECT_TRACE:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/${DATASET}/method_figure_source_trace_n${NUM_SAMPLES}_seed${SEED}_k${TOP_K}_layers${SELECTION_LAYERS//,/_}}"
BUILD_FIGURES="${BUILD_FIGURES:-1}"
FIGURE_OUTPUT_DIR="${FIGURE_OUTPUT_DIR:-${OUTPUT_DIR}/figures}"
FIGURE_RATIO_SOURCE="${FIGURE_RATIO_SOURCE:-selected}"
FIGURE_FORMATS="${FIGURE_FORMATS:-png,pdf,svg}"
LOG_DIR="${LOG_DIR:-./logs/soft_routing}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

echo "[info] output dir: ${OUTPUT_DIR}"
echo "[info] samples: ${NUM_SAMPLES}"
echo "[info] top_k: ${TOP_K}"
echo "[info] selection layers: ${SELECTION_LAYERS}"

model_base_args=()
if [ -n "${MODEL_BASE}" ]; then
    model_base_args=(--model-base "${MODEL_BASE}")
fi

all_head_args=()
if [ "${SAVE_ALL_HEAD_OBJECT_TRACE}" = "0" ]; then
    all_head_args=(--no-save-all-head-object-trace)
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" python -m eval_scripts.soft_routing.extract_method_figure_source_data \
    --model-path "${MODEL_PATH}" \
    "${model_base_args[@]}" \
    --image-folder "${IMAGE_FOLDER}" \
    --caption-file-path "${CAPTION_FILE_PATH}" \
    --annotation-dir "${ANNOTATION_DIR}" \
    --num-samples "${NUM_SAMPLES}" \
    --seed "${SEED}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --top-k "${TOP_K}" \
    --selection-layers "${SELECTION_LAYERS}" \
    --gate-strength "${GATE_STRENGTH}" \
    --gate-beta "${GATE_BETA}" \
    --gate-tau "${GATE_TAU}" \
    "${all_head_args[@]}" \
    --output-dir "${OUTPUT_DIR}" \
    2>&1 | tee "${LOG_DIR}/method_figure_source_trace_n${NUM_SAMPLES}_seed${SEED}_k${TOP_K}.log"

echo "[summary] source data"
cat "${OUTPUT_DIR}/method_figure_source_summary.json"

echo
echo "[summary] redistribution"
column -s, -t "${OUTPUT_DIR}/attention_redistribution_summary.csv"

if [ "${BUILD_FIGURES}" = "1" ]; then
    echo
    echo "[run] method figure visualization"
    python -m eval_scripts.soft_routing.visualize_method_figure_source_data \
        --source-dir "${OUTPUT_DIR}" \
        --output-dir "${FIGURE_OUTPUT_DIR}" \
        --top-k "${TOP_K}" \
        --selection-layers "${SELECTION_LAYERS}" \
        --ratio-source "${FIGURE_RATIO_SOURCE}" \
        --formats "${FIGURE_FORMATS}"

    echo
    echo "[summary] figure numeric summary"
    column -s, -t "${FIGURE_OUTPUT_DIR}/method_figure_visualization_numeric_summary.csv"
fi

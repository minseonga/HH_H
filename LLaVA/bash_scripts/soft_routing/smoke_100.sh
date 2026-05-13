#!/bin/bash

set -e
set -u

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-True}"

GPU_ID="${GPU_ID:-${1:-0}}"
MODEL_NAME="${MODEL_NAME:-llava-v1.5-7b}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
DATASET="${DATASET:-coco}"
ANNOTATION_DIR="${ANNOTATION_DIR:-/home/kms/data/images/mscoco/annotations}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
CAPTION_FILE_PATH="${CAPTION_FILE_PATH:-${ANNOTATION_DIR}/captions_val2014.json}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
SEED="${SEED:-42}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
ADHH_THRESHOLD="${ADHH_THRESHOLD:-0.4}"
SOFT_TEMPERATURE="${SOFT_TEMPERATURE:-0.05}"
RUNS="${RUNS:-greedy hard soft025 soft050 soft075}"
BASE_RESULT_PATH="${BASE_RESULT_PATH:-./results/${DATASET}/soft_routing_smoke_n${NUM_SAMPLES}_seed${SEED}_tau${ADHH_THRESHOLD}_T${SOFT_TEMPERATURE}}"

run_chair_eval() {
    local answers_file="$1"
    python eval_scripts/eval_utils/eval_chair.py \
        --annotation-dir "${ANNOTATION_DIR}" \
        --answers-file "${answers_file}" \
        --caption_file "$(basename "${CAPTION_FILE_PATH}")"
}

run_greedy() {
    local result_path="${BASE_RESULT_PATH}/greedy"
    local answers_file="${result_path}/captions.jsonl"
    mkdir -p "${result_path}"

    CUDA_VISIBLE_DEVICES="${GPU_ID}" python -m eval_scripts.eval_caption \
        --model-path "${MODEL_PATH}" \
        --image-folder "${IMAGE_FOLDER}" \
        --caption_file_path "${CAPTION_FILE_PATH}" \
        --answers-file "${answers_file}" \
        --dataset "${DATASET}" \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --num_samples "${NUM_SAMPLES}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --seed "${SEED}"

    run_chair_eval "${answers_file}"
}

run_hard() {
    local result_path="${BASE_RESULT_PATH}/hard_tau${ADHH_THRESHOLD}"
    local answers_file="${result_path}/captions.jsonl"
    mkdir -p "${result_path}"

    CUDA_VISIBLE_DEVICES="${GPU_ID}" python -m eval_scripts.eval_caption_adhh \
        --model-path "${MODEL_PATH}" \
        --image-folder "${IMAGE_FOLDER}" \
        --caption_file_path "${CAPTION_FILE_PATH}" \
        --answers-file "${answers_file}" \
        --dataset "${DATASET}" \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --num_samples "${NUM_SAMPLES}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --seed "${SEED}" \
        --adaptive_deactivate \
        --adhh_threshold "${ADHH_THRESHOLD}"

    run_chair_eval "${answers_file}"
}

run_soft() {
    local gamma="$1"
    local tag="$2"
    local result_path="${BASE_RESULT_PATH}/soft_gamma${tag}"
    local answers_file="${result_path}/captions.jsonl"
    mkdir -p "${result_path}"

    CUDA_VISIBLE_DEVICES="${GPU_ID}" python -m eval_scripts.eval_caption_adhh \
        --model-path "${MODEL_PATH}" \
        --image-folder "${IMAGE_FOLDER}" \
        --caption_file_path "${CAPTION_FILE_PATH}" \
        --answers-file "${answers_file}" \
        --dataset "${DATASET}" \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --num_samples "${NUM_SAMPLES}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --seed "${SEED}" \
        --soft_deactivate \
        --adhh_threshold "${ADHH_THRESHOLD}" \
        --soft_gamma "${gamma}" \
        --soft_temperature "${SOFT_TEMPERATURE}"

    run_chair_eval "${answers_file}"
}

for run_name in ${RUNS}; do
    case "${run_name}" in
        greedy)
            run_greedy
            ;;
        hard)
            run_hard
            ;;
        soft025)
            run_soft 0.25 0.25
            ;;
        soft050)
            run_soft 0.50 0.50
            ;;
        soft075)
            run_soft 0.75 0.75
            ;;
        *)
            echo "Unknown run: ${run_name}" >&2
            exit 1
            ;;
    esac
done

#!/bin/bash

set -euo pipefail

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-True}"

GPU_ID="${GPU_ID:-0}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
IMAGE_SPLIT="${IMAGE_SPLIT:-val2014}"
BASE_RESULT_PATH="${BASE_RESULT_PATH:-./results/coco/soft_routing_smoke_n500_seed42_tau0.4_T0.05}"
ANSWERS_FILE="${ANSWERS_FILE:-${BASE_RESULT_PATH}/greedy/captions_eval_results_calib200.json}"
CANDIDATE_HEAD_PATH="${CANDIDATE_HEAD_PATH:-results/coco/llava_3000/identify_attention_head/attribution_result.json}"
OUTPUT_PATH="${OUTPUT_PATH:-${BASE_RESULT_PATH}/identify_attention_head_val_calib200_candidates20_children_h1n1}"
LOG_DIR="${LOG_DIR:-./logs/soft_routing/candidate_attribution_children}"

START_IDX="${START_IDX:-0}"
END_IDX="${END_IDX:-}"
CANDIDATE_TOPK="${CANDIDATE_TOPK:-20}"
MAX_HALL_EVENTS="${MAX_HALL_EVENTS:-1}"
MAX_NONHALL_EVENTS="${MAX_NONHALL_EVENTS:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
INFLUENCE_SCORE="${INFLUENCE_SCORE:-prob_diff}"
LAYER_NUM="${LAYER_NUM:-32}"
HEAD_NUM="${HEAD_NUM:-32}"
SLEEP_BETWEEN="${SLEEP_BETWEEN:-2}"

mkdir -p "${OUTPUT_PATH}" "${LOG_DIR}"

if [ -z "${END_IDX}" ]; then
    END_IDX="$(python - <<PY
import json
path = "${ANSWERS_FILE}"
data = json.load(open(path))
print(sum(1 for s in data["sentences"] if s["metrics"]["CHAIRs"] == 1))
PY
)"
fi

echo "[info] answers: ${ANSWERS_FILE}"
echo "[info] output: ${OUTPUT_PATH}"
echo "[info] hallucinated-sample idx range: ${START_IDX}..$((END_IDX - 1))"
echo "[info] candidate heads: ${CANDIDATE_HEAD_PATH} top ${CANDIDATE_TOPK}"
echo "[info] event cap: hall=${MAX_HALL_EVENTS}, nonhall=${MAX_NONHALL_EVENTS}"

for idx in $(seq "${START_IDX}" $((END_IDX - 1))); do
    next_idx=$((idx + 1))
    echo "[child] idx ${idx}/${END_IDX}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python -m eval_scripts.identify_attention_head \
        --model-path "${MODEL_PATH}" \
        --image-folder "${IMAGE_FOLDER}" \
        --image-split "${IMAGE_SPLIT}" \
        --output-path "${OUTPUT_PATH}" \
        --answers-file "${ANSWERS_FILE}" \
        --candidate-head-path "${CANDIDATE_HEAD_PATH}" \
        --candidate-topk "${CANDIDATE_TOPK}" \
        --max-hall-events-per-sample "${MAX_HALL_EVENTS}" \
        --max-nonhall-events-per-sample "${MAX_NONHALL_EVENTS}" \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --topk "${CANDIDATE_TOPK}" \
        --layer_num "${LAYER_NUM}" \
        --head_num "${HEAD_NUM}" \
        --influence_score "${INFLUENCE_SCORE}" \
        --start_idx "${idx}" \
        --end_idx "${next_idx}" \
        --resume \
        --skip-aggregate \
        2>&1 | tee "${LOG_DIR}/child_${idx}.log"
    sleep "${SLEEP_BETWEEN}"
done

echo "[aggregate] building attribution_result.json"
python -m eval_scripts.identify_attention_head \
    --output-path "${OUTPUT_PATH}" \
    --answers-file "${ANSWERS_FILE}" \
    --candidate-head-path "${CANDIDATE_HEAD_PATH}" \
    --candidate-topk "${CANDIDATE_TOPK}" \
    --topk "${CANDIDATE_TOPK}" \
    --layer_num "${LAYER_NUM}" \
    --head_num "${HEAD_NUM}" \
    --aggregate-only

echo "[done] ${OUTPUT_PATH}/attribution_result.json"

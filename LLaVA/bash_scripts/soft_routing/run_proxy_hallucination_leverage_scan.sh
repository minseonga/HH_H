#!/bin/bash

set -euo pipefail

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-True}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPU_ID="${GPU_ID:-0}"
DATASET="${DATASET:-coco}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/pope/val2014}"

BASE_RESULT_PATH="${BASE_RESULT_PATH:-./results/${DATASET}/soft_routing_smoke_n500_seed42_tau0.4_T0.05}"
EVAL_DIR="${EVAL_DIR:-${BASE_RESULT_PATH}/adhh_robustness_n100}"
GREEDY_EVAL="${GREEDY_EVAL:-${EVAL_DIR}/greedy/captions_eval_results.json}"
ADHH_EVAL="${ADHH_EVAL:-${EVAL_DIR}/adhh_hard/captions_eval_results.json}"

MAX_MENTIONS="${MAX_MENTIONS:-80}"
MAX_PER_LABEL="${MAX_PER_LABEL:-0}"
LABEL_FILTER="${LABEL_FILTER:-hallucinated}"
LAYER_START="${LAYER_START:-0}"
LAYER_END="${LAYER_END:-31}"
HEAD_START="${HEAD_START:-0}"
HEAD_END="${HEAD_END:-31}"
CANDIDATE_HEADS="${CANDIDATE_HEADS:-}"
RESUME="${RESUME:-0}"

TEACHER_FEATURE="${TEACHER_FEATURE:-proxy_text_target_logit}"
TOP_KS="${TOP_KS:-20,40}"

if [ -z "${CANDIDATE_HEADS}" ]; then
    heads=()
    for ((layer=LAYER_START; layer<=LAYER_END; layer++)); do
        for ((head=HEAD_START; head<=HEAD_END; head++)); do
            heads+=("${layer}:${head}")
        done
    done
    CANDIDATE_HEADS="$(IFS=,; echo "${heads[*]}")"
    N_CANDIDATE_HEADS="${#heads[@]}"
    HEAD_GRID_NAME="l${LAYER_START}_${LAYER_END}_h${HEAD_START}_${HEAD_END}"
else
    N_CANDIDATE_HEADS="$(python - "${CANDIDATE_HEADS}" <<'PY'
import sys
items = [x for x in sys.argv[1].replace(" ", ",").split(",") if x.strip()]
print(len(items))
PY
)"
    HEAD_GRID_NAME="custom_heads"
fi

OUTPUT_DIR="${OUTPUT_DIR:-${BASE_RESULT_PATH}/proxy_hallucination_leverage_${HEAD_GRID_NAME}_n${MAX_MENTIONS}}"
DISCOVERY_DIR="${DISCOVERY_DIR:-${OUTPUT_DIR}/hallucination_leverage_head_discovery}"

mkdir -p "${OUTPUT_DIR}" "${DISCOVERY_DIR}"

echo "[info] output dir: ${OUTPUT_DIR}"
echo "[info] discovery dir: ${DISCOVERY_DIR}"
echo "[info] greedy eval: ${GREEDY_EVAL}"
echo "[info] AD-HH eval: ${ADHH_EVAL}"
echo "[info] label filter: ${LABEL_FILTER}"
echo "[info] max mentions: ${MAX_MENTIONS}"
echo "[info] max per label: ${MAX_PER_LABEL}"
echo "[info] candidate heads: ${N_CANDIDATE_HEADS}"
echo "[info] teacher feature for discovery: ${TEACHER_FEATURE}"
echo "[info] skip full/text ablation: 1/1"

extra_args=()
if [ "${RESUME}" = "1" ]; then
    extra_args+=(--resume)
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" python -m eval_scripts.soft_routing.validate_head_logit_contribution_proxy \
    --eval-results "${GREEDY_EVAL}" \
    --match-eval-results "${ADHH_EVAL}" \
    --image-folder "${IMAGE_FOLDER}" \
    --model-path "${MODEL_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --max-mentions "${MAX_MENTIONS}" \
    --max-per-label "${MAX_PER_LABEL}" \
    --label-filter "${LABEL_FILTER}" \
    --include-adhh-top-k 0 \
    --candidate-heads "${CANDIDATE_HEADS}" \
    --skip-full-head-ablation \
    --skip-text-ablation \
    "${extra_args[@]}"

CONTRIBUTION_DIR="${OUTPUT_DIR}" \
OUTPUT_DIR="${DISCOVERY_DIR}" \
HEAD_ROWS="${OUTPUT_DIR}/head_logit_proxy_ablation_rows.csv" \
TEACHER_FEATURE="${TEACHER_FEATURE}" \
TOP_KS="${TOP_KS}" \
LABEL_FILTER="${LABEL_FILTER}" \
MODEL_PATH="${MODEL_PATH}" \
bash "${SCRIPT_DIR}/run_hallucination_leverage_head_discovery.sh"

echo "[done] proxy head rows: ${OUTPUT_DIR}/head_logit_proxy_ablation_rows.csv"
echo "[done] positive-mean top20 prior: ${DISCOVERY_DIR}/hallucination_leverage_positive_mean_top20_head_prior.json"
echo "[done] signed-mean top20 prior: ${DISCOVERY_DIR}/hallucination_leverage_mean_top20_head_prior.json"

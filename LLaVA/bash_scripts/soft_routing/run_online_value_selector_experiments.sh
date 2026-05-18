#!/bin/bash

set -euo pipefail

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-True}"

GPU_ID="${GPU_ID:-0}"
DATASET="${DATASET:-coco}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
ANNOTATION_DIR="${ANNOTATION_DIR:-/home/kms/data/images/mscoco/annotations}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
CAPTION_FILE_PATH="${CAPTION_FILE_PATH:-${ANNOTATION_DIR}/captions_val2014.json}"

NUM_SAMPLES="${NUM_SAMPLES:-500}"
SEED="${SEED:-42}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
ADHH_THRESHOLD="${ADHH_THRESHOLD:-0.4}"
SOFT_TEMPERATURE="${SOFT_TEMPERATURE:-0.05}"
TOP_POOL_K="${TOP_POOL_K:-100}"
FORCE="${FORCE:-0}"
RUN_BASELINES="${RUN_BASELINES:-0}"

NORM_Q_THRESHOLD="${NORM_Q_THRESHOLD:-75}"
NORM_Q_LOW="${NORM_Q_LOW:-50}"
NORM_Q_HIGH="${NORM_Q_HIGH:-90}"
NORM_FIELD="${NORM_FIELD:-text_value_norm}"

ONLINE_VALUE_GAMMA="${ONLINE_VALUE_GAMMA:-1.0}"
ONLINE_VALUE_NORM_SOURCE="${ONLINE_VALUE_NORM_SOURCE:-text_value}"

BASE_RESULT_PATH="${BASE_RESULT_PATH:-./results/${DATASET}/soft_routing_smoke_n${NUM_SAMPLES}_seed${SEED}_tau${ADHH_THRESHOLD}_T${SOFT_TEMPERATURE}}"
ATTRIBUTION_RESULT="${ATTRIBUTION_RESULT:-${BASE_RESULT_PATH}/identify_attention_head_val_calib200_full1024/attribution_result.json}"
if [ ! -f "${ATTRIBUTION_RESULT}" ]; then
    ATTRIBUTION_RESULT="${ATTRIBUTION_RESULT_FALLBACK:-./results/coco/llava_3000/identify_attention_head/attribution_result.json}"
fi

POOL_DIR="${POOL_DIR:-${BASE_RESULT_PATH}/contrastive_candidate_pools}"
POOL_PATH="${POOL_PATH:-${POOL_DIR}/contrastive_top${TOP_POOL_K}.json}"
BEHAVIORAL_RECORDS="${BEHAVIORAL_RECORDS:-${BASE_RESULT_PATH}/behavioral_head_overlap_prefill_n500_minlayer13/all_head_records.jsonl}"
NORM_THRESHOLDS_PATH="${NORM_THRESHOLDS_PATH:-${POOL_DIR}/text_value_norm_q${NORM_Q_THRESHOLD}_top${TOP_POOL_K}.json}"

OUTPUT_DIR="${OUTPUT_DIR:-${BASE_RESULT_PATH}/online_value_selector_top${TOP_POOL_K}_normq${NORM_Q_THRESHOLD}}"
LOG_DIR="${LOG_DIR:-./logs/soft_routing}"

mkdir -p "${POOL_DIR}" "${OUTPUT_DIR}" "${LOG_DIR}"

if [ ! -f "${ATTRIBUTION_RESULT}" ]; then
    echo "[error] missing attribution result: ${ATTRIBUTION_RESULT}" >&2
    exit 1
fi

echo "[info] attribution result: ${ATTRIBUTION_RESULT}"
echo "[info] candidate pool: ${POOL_PATH}"
echo "[info] norm thresholds: ${NORM_THRESHOLDS_PATH}"
echo "[info] output dir: ${OUTPUT_DIR}"

if [ "${FORCE}" = "1" ] || [ ! -f "${POOL_PATH}" ]; then
    python -m eval_scripts.soft_routing.build_contrastive_candidate_pool \
        --attribution-result "${ATTRIBUTION_RESULT}" \
        --output-path "${POOL_PATH}" \
        --top-k "${TOP_POOL_K}"
else
    echo "[skip] candidate pool exists: ${POOL_PATH}"
fi

if [ ! -f "${BEHAVIORAL_RECORDS}" ]; then
    echo "[warn] missing behavioral records for norm calibration: ${BEHAVIORAL_RECORDS}"
    echo "[warn] running prefill all-head logging first"
    HF_HUB_OFFLINE="${HF_HUB_OFFLINE}" \
    GPU_ID="${GPU_ID}" \
    NUM_SAMPLES="${NUM_SAMPLES}" \
    MAX_SAMPLES="${NUM_SAMPLES}" \
    WARMUP_TOKENS=0 \
    MIN_LAYER=13 \
    TOP_K=20 \
    ATTENTION_HEAD_PATH="${POOL_PATH}" \
    OUTPUT_DIR="$(dirname "${BEHAVIORAL_RECORDS}")" \
    bash bash_scripts/soft_routing/run_behavioral_head_overlap.sh
fi

if [ "${FORCE}" = "1" ] || [ ! -f "${NORM_THRESHOLDS_PATH}" ]; then
    python -m eval_scripts.soft_routing.build_head_norm_thresholds \
        --records-jsonl "${BEHAVIORAL_RECORDS}" \
        --output-path "${NORM_THRESHOLDS_PATH}" \
        --head-path "${POOL_PATH}" \
        --top-k "${TOP_POOL_K}" \
        --norm-field "${NORM_FIELD}" \
        --q-threshold "${NORM_Q_THRESHOLD}" \
        --q-low "${NORM_Q_LOW}" \
        --q-high "${NORM_Q_HIGH}"
else
    echo "[skip] norm thresholds exist: ${NORM_THRESHOLDS_PATH}"
fi

run_eval() {
    local tag="$1"
    local mode="$2"
    local layer_top_k="$3"
    local result_dir="${OUTPUT_DIR}/${tag}"
    local answers_file="${result_dir}/captions.jsonl"
    local eval_file="${result_dir}/captions_eval_results.json"
    mkdir -p "${result_dir}"

    if [ "${FORCE}" != "1" ] && [ -f "${eval_file}" ]; then
        echo "[skip] ${tag}: ${eval_file}"
        return
    fi

    echo "[run] ${tag}"
    if [ "${mode}" = "adhh" ]; then
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
            --adhh_threshold "${ADHH_THRESHOLD}" \
            --attention_head_path "${POOL_PATH}" \
            --head_prior_mode uniform \
            --top_k "${layer_top_k}" \
            2>&1 | tee "${LOG_DIR}/online_value_${tag}.log"
    elif [ "${mode}" = "wide_text" ]; then
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
            --wide_gate_deactivate \
            --wide_gate_mode hard \
            --wide_gate_feature text \
            --wide_gate_text_tau "${ADHH_THRESHOLD}" \
            --attention_head_path "${POOL_PATH}" \
            --head_prior_mode uniform \
            --top_k "${TOP_POOL_K}" \
            2>&1 | tee "${LOG_DIR}/online_value_${tag}.log"
    else
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
            --online_value_selector_deactivate \
            --online_value_selector_mode "${mode}" \
            --online_value_selector_layer_top_k "${layer_top_k}" \
            --online_value_selector_text_tau "${ADHH_THRESHOLD}" \
            --online_value_selector_gamma "${ONLINE_VALUE_GAMMA}" \
            --online_value_selector_norm_source "${ONLINE_VALUE_NORM_SOURCE}" \
            --head_norm_thresholds_path "${NORM_THRESHOLDS_PATH}" \
            --attention_head_path "${POOL_PATH}" \
            --head_prior_mode uniform \
            --top_k "${TOP_POOL_K}" \
            2>&1 | tee "${LOG_DIR}/online_value_${tag}.log"
    fi

    python eval_scripts/eval_utils/eval_chair.py \
        --annotation-dir "${ANNOTATION_DIR}" \
        --answers-file "${answers_file}" \
        --caption_file "$(basename "${CAPTION_FILE_PATH}")"
}

if [ "${RUN_BASELINES}" = "1" ]; then
    run_eval "adhh_top20_hard" "adhh" 20
    run_eval "top100_text_hard" "wide_text" 100
fi
run_eval "online_value_layer_top1_hard" "hard" 1
run_eval "online_value_layer_top2_hard" "hard" 2
run_eval "online_value_layer_top1_continuous" "continuous" 1
run_eval "online_value_layer_top2_continuous" "continuous" 2

python - <<PY
import csv
import glob
import json
import os

rows = []
for path in sorted(glob.glob("${OUTPUT_DIR}/*/captions_eval_results.json")):
    tag = os.path.basename(os.path.dirname(path))
    metrics = json.load(open(path))["overall_metrics"]
    bleu = metrics.get("Bleu") or [None, None, None, None]
    rows.append({
        "method": tag,
        "CHAIRs": metrics.get("CHAIRs"),
        "CHAIRi": metrics.get("CHAIRi"),
        "Bleu1": bleu[0],
        "Bleu2": bleu[1],
        "Bleu3": bleu[2],
        "Bleu4": bleu[3],
        "avg_caption_length": metrics.get("avg_caption_length"),
    })

out = "${OUTPUT_DIR}/online_value_selector_summary.csv"
if rows:
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
print(out)
PY

echo "[summary] online value selector metrics"
if [ -f "${OUTPUT_DIR}/online_value_selector_summary.csv" ]; then
    column -s, -t "${OUTPUT_DIR}/online_value_selector_summary.csv"
fi

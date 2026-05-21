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
RUN_METHODS="${RUN_METHODS:-online_unsupported_layer_top1_continuous online_unsupported_layer_top2_continuous}"

UNSUPPORTED_COMPONENT_GAMMA="${UNSUPPORTED_COMPONENT_GAMMA:-0.5}"
UNSUPPORTED_COMPONENT_RISK_FEATURE="${UNSUPPORTED_COMPONENT_RISK_FEATURE:-unsupported_norm_x_low_anchor}"
UNSUPPORTED_COMPONENT_SOFT_THRESHOLD="${UNSUPPORTED_COMPONENT_SOFT_THRESHOLD:-0.25}"
UNSUPPORTED_COMPONENT_HARD_THRESHOLD="${UNSUPPORTED_COMPONENT_HARD_THRESHOLD:-0.75}"
UNSUPPORTED_COMPONENT_SCORE_NORM="${UNSUPPORTED_COMPONENT_SCORE_NORM:-candidate_minmax}"
UNSUPPORTED_COMPONENT_SCORE_LOW="${UNSUPPORTED_COMPONENT_SCORE_LOW:-0.0}"
UNSUPPORTED_COMPONENT_SCORE_HIGH="${UNSUPPORTED_COMPONENT_SCORE_HIGH:-1.0}"
UNSUPPORTED_COMPONENT_PHASE="${UNSUPPORTED_COMPONENT_PHASE:-all}"
UNSUPPORTED_COMPONENT_LAYERS="${UNSUPPORTED_COMPONENT_LAYERS:-}"
UNSUPPORTED_COMPONENT_ALL_HEADS="${UNSUPPORTED_COMPONENT_ALL_HEADS:-0}"
UNSUPPORTED_COMPONENT_HEAD_PATH="${UNSUPPORTED_COMPONENT_HEAD_PATH:-}"
UNSUPPORTED_COMPONENT_HEAD_TOP_K="${UNSUPPORTED_COMPONENT_HEAD_TOP_K:-${TOP_POOL_K}}"
RECORD_UNSUPPORTED_COMPONENT_DIAGNOSTICS="${RECORD_UNSUPPORTED_COMPONENT_DIAGNOSTICS:-0}"
UNSUPPORTED_COMPONENT_DIAGNOSTICS_MAX_RECORDS="${UNSUPPORTED_COMPONENT_DIAGNOSTICS_MAX_RECORDS:-0}"

BASE_RESULT_PATH="${BASE_RESULT_PATH:-./results/${DATASET}/soft_routing_smoke_n${NUM_SAMPLES}_seed${SEED}_tau${ADHH_THRESHOLD}_T${SOFT_TEMPERATURE}}"
ATTRIBUTION_RESULT="${ATTRIBUTION_RESULT:-${BASE_RESULT_PATH}/identify_attention_head_val_calib200_full1024/attribution_result.json}"
if [ ! -f "${ATTRIBUTION_RESULT}" ]; then
    ATTRIBUTION_RESULT="${ATTRIBUTION_RESULT_FALLBACK:-./results/coco/llava_3000/identify_attention_head/attribution_result.json}"
fi

POOL_DIR="${POOL_DIR:-${BASE_RESULT_PATH}/contrastive_candidate_pools}"
POOL_PATH="${POOL_PATH:-${POOL_DIR}/contrastive_top${TOP_POOL_K}.json}"
if [ -n "${UNSUPPORTED_COMPONENT_HEAD_PATH}" ]; then
    HEAD_PATH="${UNSUPPORTED_COMPONENT_HEAD_PATH}"
    HEAD_TOP_K="${UNSUPPORTED_COMPONENT_HEAD_TOP_K}"
    HEAD_TAG="$(basename "${HEAD_PATH}" .json)"
    OUTPUT_DIR="${OUTPUT_DIR:-${BASE_RESULT_PATH}/online_unsupported_component_${HEAD_TAG}_${UNSUPPORTED_COMPONENT_RISK_FEATURE}_g${UNSUPPORTED_COMPONENT_GAMMA}}"
else
    HEAD_PATH="${POOL_PATH}"
    HEAD_TOP_K="${TOP_POOL_K}"
    OUTPUT_DIR="${OUTPUT_DIR:-${BASE_RESULT_PATH}/online_unsupported_component_top${TOP_POOL_K}_${UNSUPPORTED_COMPONENT_RISK_FEATURE}_g${UNSUPPORTED_COMPONENT_GAMMA}}"
fi
LOG_DIR="${LOG_DIR:-./logs/soft_routing}"

mkdir -p "${POOL_DIR}" "${OUTPUT_DIR}" "${LOG_DIR}"

if [ -z "${UNSUPPORTED_COMPONENT_HEAD_PATH}" ] && [ ! -f "${ATTRIBUTION_RESULT}" ]; then
    echo "[error] missing attribution result: ${ATTRIBUTION_RESULT}" >&2
    exit 1
fi

if [ -f "${ATTRIBUTION_RESULT}" ]; then
    echo "[info] attribution result: ${ATTRIBUTION_RESULT}"
else
    echo "[info] attribution result: ${ATTRIBUTION_RESULT} (not required for custom head path)"
fi
echo "[info] candidate pool: ${POOL_PATH}"
echo "[info] active head path: ${HEAD_PATH}"
echo "[info] active head top-k: ${HEAD_TOP_K}"
echo "[info] output dir: ${OUTPUT_DIR}"
echo "[info] risk feature: ${UNSUPPORTED_COMPONENT_RISK_FEATURE}"
echo "[info] gamma: ${UNSUPPORTED_COMPONENT_GAMMA}"
echo "[info] score norm: ${UNSUPPORTED_COMPONENT_SCORE_NORM}"
echo "[info] score low/high: ${UNSUPPORTED_COMPONENT_SCORE_LOW}/${UNSUPPORTED_COMPONENT_SCORE_HIGH}"
echo "[info] phase: ${UNSUPPORTED_COMPONENT_PHASE}"
echo "[info] layers: ${UNSUPPORTED_COMPONENT_LAYERS:-all}"
echo "[info] all heads: ${UNSUPPORTED_COMPONENT_ALL_HEADS}"
echo "[info] record unsupported diagnostics: ${RECORD_UNSUPPORTED_COMPONENT_DIAGNOSTICS}"

if [ -z "${UNSUPPORTED_COMPONENT_HEAD_PATH}" ] && { [ "${FORCE}" = "1" ] || [ ! -f "${POOL_PATH}" ]; }; then
    python -m eval_scripts.soft_routing.build_contrastive_candidate_pool \
        --attribution-result "${ATTRIBUTION_RESULT}" \
        --output-path "${POOL_PATH}" \
        --top-k "${TOP_POOL_K}"
elif [ -n "${UNSUPPORTED_COMPONENT_HEAD_PATH}" ]; then
    if [ ! -f "${HEAD_PATH}" ]; then
        echo "[error] missing custom head path: ${HEAD_PATH}" >&2
        exit 1
    fi
else
    echo "[skip] candidate pool exists: ${POOL_PATH}"
fi

all_head_args=()
if [ "${UNSUPPORTED_COMPONENT_ALL_HEADS}" = "1" ]; then
    all_head_args=(--unsupported_component_all_heads)
fi

diagnostic_args=()
if [ "${RECORD_UNSUPPORTED_COMPONENT_DIAGNOSTICS}" = "1" ]; then
    diagnostic_args=(
        --record_unsupported_component_diagnostics
        --unsupported_component_diagnostics_max_records "${UNSUPPORTED_COMPONENT_DIAGNOSTICS_MAX_RECORDS}"
    )
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
        --unsupported_component_deactivate \
        --unsupported_component_mode "${mode}" \
        --unsupported_component_layer_top_k "${layer_top_k}" \
        --unsupported_component_gamma "${UNSUPPORTED_COMPONENT_GAMMA}" \
        --unsupported_component_soft_threshold "${UNSUPPORTED_COMPONENT_SOFT_THRESHOLD}" \
        --unsupported_component_hard_threshold "${UNSUPPORTED_COMPONENT_HARD_THRESHOLD}" \
        --unsupported_component_score_norm "${UNSUPPORTED_COMPONENT_SCORE_NORM}" \
        --unsupported_component_score_low "${UNSUPPORTED_COMPONENT_SCORE_LOW}" \
        --unsupported_component_score_high "${UNSUPPORTED_COMPONENT_SCORE_HIGH}" \
        --unsupported_component_phase "${UNSUPPORTED_COMPONENT_PHASE}" \
        --unsupported_component_layers "${UNSUPPORTED_COMPONENT_LAYERS}" \
        --unsupported_component_risk_feature "${UNSUPPORTED_COMPONENT_RISK_FEATURE}" \
        "${all_head_args[@]}" \
        "${diagnostic_args[@]}" \
        --attention_head_path "${HEAD_PATH}" \
        --head_prior_mode uniform \
        --top_k "${HEAD_TOP_K}" \
        2>&1 | tee "${LOG_DIR}/online_unsupported_${tag}.log"

    python eval_scripts/eval_utils/eval_chair.py \
        --annotation-dir "${ANNOTATION_DIR}" \
        --answers-file "${answers_file}" \
        --caption_file "$(basename "${CAPTION_FILE_PATH}")"

    local diagnostics_file="${result_dir}/unsupported_component_diagnostics.jsonl"
    if [ "${RECORD_UNSUPPORTED_COMPONENT_DIAGNOSTICS}" = "1" ] && [ -f "${diagnostics_file}" ]; then
        python -m eval_scripts.soft_routing.analyze_unsupported_component_diagnostics \
            --diagnostics-jsonl "${diagnostics_file}" \
            --output-dir "${result_dir}/unsupported_component_diagnostic_summary"
    fi
}

should_run() {
    local tag="$1"
    for method in ${RUN_METHODS}; do
        if [ "${method}" = "${tag}" ]; then
            return 0
        fi
    done
    return 1
}

maybe_run_eval() {
    local tag="$1"
    local mode="$2"
    local layer_top_k="$3"
    if should_run "${tag}"; then
        run_eval "${tag}" "${mode}" "${layer_top_k}"
    else
        echo "[skip] ${tag}: not in RUN_METHODS"
    fi
}

echo "[info] run methods: ${RUN_METHODS}"

maybe_run_eval "online_unsupported_layer_top1_continuous" "continuous" 1
maybe_run_eval "online_unsupported_layer_top2_continuous" "continuous" 2
maybe_run_eval "online_unsupported_layer_top1_hybrid" "hybrid" 1
maybe_run_eval "online_unsupported_layer_top2_hybrid" "hybrid" 2

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

out = "${OUTPUT_DIR}/online_unsupported_component_summary.csv"
if rows:
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
print(out)
PY

echo "[summary] online unsupported component metrics"
if [ -f "${OUTPUT_DIR}/online_unsupported_component_summary.csv" ]; then
    column -s, -t "${OUTPUT_DIR}/online_unsupported_component_summary.csv"
fi

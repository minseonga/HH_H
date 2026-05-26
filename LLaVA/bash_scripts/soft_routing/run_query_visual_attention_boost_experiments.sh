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
FORCE="${FORCE:-0}"

BASE_RESULT_PATH="${BASE_RESULT_PATH:-./results/${DATASET}/soft_routing_smoke_n500_seed42_tau0.4_T0.05}"
CALIBRATION_NPZ="${CALIBRATION_NPZ:-${BASE_RESULT_PATH}/query_direction_probe_l2_l13_31_hallboth_max100/query_direction_calibration.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-${BASE_RESULT_PATH}/query_visual_attention_boost_n${NUM_SAMPLES}}"
LOG_DIR="${LOG_DIR:-./logs/soft_routing}"
GREEDY_EVAL="${GREEDY_EVAL:-${BASE_RESULT_PATH}/greedy/captions_eval_results.json}"

QUERY_VISUAL_BOOST_DETECTOR_TOP_KS="${QUERY_VISUAL_BOOST_DETECTOR_TOP_KS:-1 5 10}"
QUERY_VISUAL_BOOST_ALPHAS="${QUERY_VISUAL_BOOST_ALPHAS:-0.5 1 2 4}"
QUERY_VISUAL_BOOST_TEXT_BETAS="${QUERY_VISUAL_BOOST_TEXT_BETAS:-0.0}"
QUERY_VISUAL_BOOST_RISK_MODES="${QUERY_VISUAL_BOOST_RISK_MODES:-z_softplus}"
QUERY_VISUAL_BOOST_TARGET_HEADS="${QUERY_VISUAL_BOOST_TARGET_HEADS:-all}"
QUERY_VISUAL_BOOST_DETECTOR_AGGREGATION="${QUERY_VISUAL_BOOST_DETECTOR_AGGREGATION:-max}"
QUERY_VISUAL_BOOST_ACCUMULATION="${QUERY_VISUAL_BOOST_ACCUMULATION:-max}"
QUERY_VISUAL_BOOST_TEMPERATURE="${QUERY_VISUAL_BOOST_TEMPERATURE:-1.0}"
QUERY_VISUAL_BOOST_RISK_SCALE="${QUERY_VISUAL_BOOST_RISK_SCALE:-1.0}"
QUERY_VISUAL_BOOST_MAX_RISK="${QUERY_VISUAL_BOOST_MAX_RISK:-0.0}"
QUERY_VISUAL_BOOST_MAX_FACTOR="${QUERY_VISUAL_BOOST_MAX_FACTOR:-0.0}"
QUERY_VISUAL_BOOST_MIN_AUROC="${QUERY_VISUAL_BOOST_MIN_AUROC:-0.0}"
QUERY_VISUAL_BOOST_DETECTOR_PHASE="${QUERY_VISUAL_BOOST_DETECTOR_PHASE:-decode}"
QUERY_VISUAL_BOOST_PHASE="${QUERY_VISUAL_BOOST_PHASE:-decode}"
QUERY_VISUAL_BOOST_MIN_LAYER="${QUERY_VISUAL_BOOST_MIN_LAYER:-}"
QUERY_VISUAL_BOOST_MAX_LAYER="${QUERY_VISUAL_BOOST_MAX_LAYER:-}"
RECORD_QUERY_VISUAL_BOOST_DIAGNOSTICS="${RECORD_QUERY_VISUAL_BOOST_DIAGNOSTICS:-1}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

if [ ! -f "${CALIBRATION_NPZ}" ]; then
    echo "[error] missing query calibration: ${CALIBRATION_NPZ}" >&2
    exit 1
fi

echo "[info] output dir: ${OUTPUT_DIR}"
echo "[info] calibration: ${CALIBRATION_NPZ}"
echo "[info] detector top ks: ${QUERY_VISUAL_BOOST_DETECTOR_TOP_KS}"
echo "[info] alphas: ${QUERY_VISUAL_BOOST_ALPHAS}"
echo "[info] text betas: ${QUERY_VISUAL_BOOST_TEXT_BETAS}"
echo "[info] risk modes: ${QUERY_VISUAL_BOOST_RISK_MODES}"
echo "[info] target heads: ${QUERY_VISUAL_BOOST_TARGET_HEADS}"
echo "[info] image folder: ${IMAGE_FOLDER}"
echo "[info] num samples: ${NUM_SAMPLES}"

diagnostic_args=()
if [ "${RECORD_QUERY_VISUAL_BOOST_DIAGNOSTICS}" = "1" ]; then
    diagnostic_args=(--record_query_visual_attention_boost_diagnostics)
fi

min_layer_args=()
if [ -n "${QUERY_VISUAL_BOOST_MIN_LAYER}" ]; then
    min_layer_args=(--query_visual_attention_boost_min_layer "${QUERY_VISUAL_BOOST_MIN_LAYER}")
fi

max_layer_args=()
if [ -n "${QUERY_VISUAL_BOOST_MAX_LAYER}" ]; then
    max_layer_args=(--query_visual_attention_boost_max_layer "${QUERY_VISUAL_BOOST_MAX_LAYER}")
fi

run_visual_boost() {
    local detector_top_k="$1"
    local alpha="$2"
    local beta="$3"
    local risk_mode="$4"
    local alpha_tag beta_tag
    alpha_tag="$(printf "%s" "${alpha}" | tr "." "p")"
    beta_tag="$(printf "%s" "${beta}" | tr "." "p")"
    local tag="qvboost_${risk_mode}_det${detector_top_k}_a${alpha_tag}_b${beta_tag}_${QUERY_VISUAL_BOOST_TARGET_HEADS}"
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
        --query_visual_attention_boost \
        --query_visual_attention_boost_calibration "${CALIBRATION_NPZ}" \
        --query_visual_attention_boost_detector_top_k "${detector_top_k}" \
        --query_visual_attention_boost_min_auroc "${QUERY_VISUAL_BOOST_MIN_AUROC}" \
        --query_visual_attention_boost_alpha "${alpha}" \
        --query_visual_attention_boost_text_beta "${beta}" \
        --query_visual_attention_boost_risk_mode "${risk_mode}" \
        --query_visual_attention_boost_detector_aggregation "${QUERY_VISUAL_BOOST_DETECTOR_AGGREGATION}" \
        --query_visual_attention_boost_accumulation "${QUERY_VISUAL_BOOST_ACCUMULATION}" \
        --query_visual_attention_boost_target_heads "${QUERY_VISUAL_BOOST_TARGET_HEADS}" \
        --query_visual_attention_boost_temperature "${QUERY_VISUAL_BOOST_TEMPERATURE}" \
        --query_visual_attention_boost_risk_scale "${QUERY_VISUAL_BOOST_RISK_SCALE}" \
        --query_visual_attention_boost_max_risk "${QUERY_VISUAL_BOOST_MAX_RISK}" \
        --query_visual_attention_boost_max_factor "${QUERY_VISUAL_BOOST_MAX_FACTOR}" \
        --query_visual_attention_boost_detector_phase "${QUERY_VISUAL_BOOST_DETECTOR_PHASE}" \
        --query_visual_attention_boost_phase "${QUERY_VISUAL_BOOST_PHASE}" \
        "${min_layer_args[@]}" \
        "${max_layer_args[@]}" \
        "${diagnostic_args[@]}" \
        2>&1 | tee "${LOG_DIR}/${tag}.log"

    python eval_scripts/eval_utils/eval_chair.py \
        --annotation-dir "${ANNOTATION_DIR}" \
        --answers-file "${answers_file}" \
        --caption_file "$(basename "${CAPTION_FILE_PATH}")"
}

for detector_top_k in ${QUERY_VISUAL_BOOST_DETECTOR_TOP_KS}; do
    for alpha in ${QUERY_VISUAL_BOOST_ALPHAS}; do
        for beta in ${QUERY_VISUAL_BOOST_TEXT_BETAS}; do
            for risk_mode in ${QUERY_VISUAL_BOOST_RISK_MODES}; do
                run_visual_boost "${detector_top_k}" "${alpha}" "${beta}" "${risk_mode}"
            done
        done
    done
done

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

out = "${OUTPUT_DIR}/query_visual_attention_boost_summary.csv"
if rows:
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
print(out)
PY

if [ -f "${GREEDY_EVAL}" ]; then
    mapfile -t eval_files < <(find "${OUTPUT_DIR}" -mindepth 2 -maxdepth 2 -name captions_eval_results.json | sort)
    if [ "${#eval_files[@]}" -gt 0 ]; then
        python -m eval_scripts.soft_routing.analyze_hallucination_position \
            --eval-results "${GREEDY_EVAL}" "${eval_files[@]}" \
            --names greedy $(for path in "${eval_files[@]}"; do basename "$(dirname "${path}")"; done) \
            --match-common-image-ids \
            --output-dir "${OUTPUT_DIR}/position_vs_greedy"
    fi
else
    echo "[warn] missing greedy eval for position analysis: ${GREEDY_EVAL}" >&2
fi

echo "[summary] query visual-attention boost metrics"
if [ -f "${OUTPUT_DIR}/query_visual_attention_boost_summary.csv" ]; then
    column -s, -t "${OUTPUT_DIR}/query_visual_attention_boost_summary.csv"
fi

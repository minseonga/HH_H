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
OUTPUT_DIR="${OUTPUT_DIR:-${BASE_RESULT_PATH}/query_gated_head_output_caption_n${NUM_SAMPLES}}"
LOG_DIR="${LOG_DIR:-./logs/soft_routing}"
GREEDY_EVAL="${GREEDY_EVAL:-${BASE_RESULT_PATH}/greedy/captions_eval_results.json}"

QUERY_GATED_HEAD_OUTPUT_TOP_KS="${QUERY_GATED_HEAD_OUTPUT_TOP_KS:-1}"
QUERY_GATED_HEAD_OUTPUT_STRENGTHS="${QUERY_GATED_HEAD_OUTPUT_STRENGTHS:-1 3 5 10}"
QUERY_GATED_HEAD_OUTPUT_PHASES="${QUERY_GATED_HEAD_OUTPUT_PHASES:-decode}"
QUERY_GATED_HEAD_OUTPUT_GATE_MODES="${QUERY_GATED_HEAD_OUTPUT_GATE_MODES:-margin}"
QUERY_GATED_HEAD_OUTPUT_MARGIN_SCALES="${QUERY_GATED_HEAD_OUTPUT_MARGIN_SCALES:-0.1}"
QUERY_GATED_HEAD_OUTPUT_MIN_AUROC="${QUERY_GATED_HEAD_OUTPUT_MIN_AUROC:-0.0}"
QUERY_GATED_HEAD_OUTPUT_TEMPERATURE="${QUERY_GATED_HEAD_OUTPUT_TEMPERATURE:-0.05}"
QUERY_GATED_HEAD_OUTPUT_MIN_GATE="${QUERY_GATED_HEAD_OUTPUT_MIN_GATE:-0.0}"
QUERY_GATED_HEAD_OUTPUT_MAX_GATE="${QUERY_GATED_HEAD_OUTPUT_MAX_GATE:-1.0}"
RECORD_QUERY_GATED_HEAD_OUTPUT_DIAGNOSTICS="${RECORD_QUERY_GATED_HEAD_OUTPUT_DIAGNOSTICS:-1}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

if [ ! -f "${CALIBRATION_NPZ}" ]; then
    echo "[error] missing query calibration: ${CALIBRATION_NPZ}" >&2
    exit 1
fi

echo "[info] output dir: ${OUTPUT_DIR}"
echo "[info] calibration: ${CALIBRATION_NPZ}"
echo "[info] top ks: ${QUERY_GATED_HEAD_OUTPUT_TOP_KS}"
echo "[info] strengths: ${QUERY_GATED_HEAD_OUTPUT_STRENGTHS}"
echo "[info] phases: ${QUERY_GATED_HEAD_OUTPUT_PHASES}"
echo "[info] gate modes: ${QUERY_GATED_HEAD_OUTPUT_GATE_MODES}"
echo "[info] margin scales: ${QUERY_GATED_HEAD_OUTPUT_MARGIN_SCALES}"
echo "[info] image folder: ${IMAGE_FOLDER}"
echo "[info] num samples: ${NUM_SAMPLES}"

diagnostic_args=()
if [ "${RECORD_QUERY_GATED_HEAD_OUTPUT_DIAGNOSTICS}" = "1" ]; then
    diagnostic_args=(--record_query_gated_head_output_diagnostics)
fi

run_qgated_head_output() {
    local phase="$1"
    local top_k="$2"
    local strength="$3"
    local gate_mode="$4"
    local margin_scale="$5"
    local strength_tag
    local margin_tag
    strength_tag="$(printf "%s" "${strength}" | tr "." "p")"
    margin_tag="$(printf "%s" "${margin_scale}" | tr "." "p")"
    local tag="qgated_headout_${phase}_top${top_k}_${gate_mode}_m${margin_tag}_s${strength_tag}"
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
        --query_gated_head_output_suppress \
        --query_gated_head_output_calibration "${CALIBRATION_NPZ}" \
        --query_gated_head_output_top_k "${top_k}" \
        --query_gated_head_output_min_auroc "${QUERY_GATED_HEAD_OUTPUT_MIN_AUROC}" \
        --query_gated_head_output_strength "${strength}" \
        --query_gated_head_output_gate_mode "${gate_mode}" \
        --query_gated_head_output_temperature "${QUERY_GATED_HEAD_OUTPUT_TEMPERATURE}" \
        --query_gated_head_output_margin_scale "${margin_scale}" \
        --query_gated_head_output_min_gate "${QUERY_GATED_HEAD_OUTPUT_MIN_GATE}" \
        --query_gated_head_output_max_gate "${QUERY_GATED_HEAD_OUTPUT_MAX_GATE}" \
        --query_gated_head_output_phase "${phase}" \
        "${diagnostic_args[@]}" \
        2>&1 | tee "${LOG_DIR}/${tag}.log"

    python eval_scripts/eval_utils/eval_chair.py \
        --annotation-dir "${ANNOTATION_DIR}" \
        --answers-file "${answers_file}" \
        --caption_file "$(basename "${CAPTION_FILE_PATH}")"
}

for phase in ${QUERY_GATED_HEAD_OUTPUT_PHASES}; do
    for top_k in ${QUERY_GATED_HEAD_OUTPUT_TOP_KS}; do
        for strength in ${QUERY_GATED_HEAD_OUTPUT_STRENGTHS}; do
            for gate_mode in ${QUERY_GATED_HEAD_OUTPUT_GATE_MODES}; do
                for margin_scale in ${QUERY_GATED_HEAD_OUTPUT_MARGIN_SCALES}; do
                    run_qgated_head_output "${phase}" "${top_k}" "${strength}" "${gate_mode}" "${margin_scale}"
                done
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

out = "${OUTPUT_DIR}/query_gated_head_output_caption_summary.csv"
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

echo "[summary] query-gated head-output caption metrics"
if [ -f "${OUTPUT_DIR}/query_gated_head_output_caption_summary.csv" ]; then
    column -s, -t "${OUTPUT_DIR}/query_gated_head_output_caption_summary.csv"
fi

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
ADHH_THRESHOLD="${ADHH_THRESHOLD:-0.4}"
SOFT_TEMPERATURE="${SOFT_TEMPERATURE:-0.05}"
FORCE="${FORCE:-0}"

BASE_RESULT_PATH="${BASE_RESULT_PATH:-./results/${DATASET}/soft_routing_smoke_n500_seed42_tau0.4_T0.05}"
CALIBRATION_NPZ="${CALIBRATION_NPZ:-${BASE_RESULT_PATH}/query_direction_probe_l2_l13_31_hallboth_max100/query_direction_calibration.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-${BASE_RESULT_PATH}/query_projection_caption_n${NUM_SAMPLES}}"
LOG_DIR="${LOG_DIR:-./logs/soft_routing}"
GREEDY_EVAL="${GREEDY_EVAL:-${BASE_RESULT_PATH}/greedy/captions_eval_results.json}"

QUERY_DIRECTION_TOP_KS="${QUERY_DIRECTION_TOP_KS:-1 5 10}"
QUERY_DIRECTION_PHASES="${QUERY_DIRECTION_PHASES:-decode all}"
QUERY_DIRECTION_STRENGTHS="${QUERY_DIRECTION_STRENGTHS:-0.5}"
QUERY_DIRECTION_MIN_AUROC="${QUERY_DIRECTION_MIN_AUROC:-0.0}"
QUERY_DIRECTION_GATE_MODE="${QUERY_DIRECTION_GATE_MODE:-none}"
QUERY_DIRECTION_TEMPERATURE="${QUERY_DIRECTION_TEMPERATURE:-0.05}"
QUERY_DIRECTION_ALLOW_NEGATIVE="${QUERY_DIRECTION_ALLOW_NEGATIVE:-0}"
QUERY_DIRECTION_PREFILL_POSITIONS="${QUERY_DIRECTION_PREFILL_POSITIONS:-last}"
QUERY_DIRECTION_SAMPLE_GATE_MODES="${QUERY_DIRECTION_SAMPLE_GATE_MODES:-off}"
QUERY_DIRECTION_SAMPLE_GATE_POSITIONS="${QUERY_DIRECTION_SAMPLE_GATE_POSITIONS:-image}"
QUERY_DIRECTION_SAMPLE_GATE_SCALE="${QUERY_DIRECTION_SAMPLE_GATE_SCALE:-0.1}"
QUERY_DIRECTION_SAMPLE_GATE_MIN="${QUERY_DIRECTION_SAMPLE_GATE_MIN:-0.0}"
QUERY_DIRECTION_SAMPLE_GATE_MAX="${QUERY_DIRECTION_SAMPLE_GATE_MAX:-1.0}"
RECORD_QUERY_PROJECTION_DIAGNOSTICS="${RECORD_QUERY_PROJECTION_DIAGNOSTICS:-1}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

if [ ! -f "${CALIBRATION_NPZ}" ]; then
    echo "[error] missing query calibration: ${CALIBRATION_NPZ}" >&2
    exit 1
fi

echo "[info] output dir: ${OUTPUT_DIR}"
echo "[info] calibration: ${CALIBRATION_NPZ}"
echo "[info] top ks: ${QUERY_DIRECTION_TOP_KS}"
echo "[info] phases: ${QUERY_DIRECTION_PHASES}"
echo "[info] strengths: ${QUERY_DIRECTION_STRENGTHS}"
echo "[info] gate mode: ${QUERY_DIRECTION_GATE_MODE}"
echo "[info] sample gate modes: ${QUERY_DIRECTION_SAMPLE_GATE_MODES}"
echo "[info] image folder: ${IMAGE_FOLDER}"
echo "[info] num samples: ${NUM_SAMPLES}"

allow_negative_args=()
if [ "${QUERY_DIRECTION_ALLOW_NEGATIVE}" = "1" ]; then
    allow_negative_args=(--query_direction_allow_negative)
fi

diagnostic_args=()
if [ "${RECORD_QUERY_PROJECTION_DIAGNOSTICS}" = "1" ]; then
    diagnostic_args=(--record_query_projection_diagnostics)
fi

run_projection() {
    local phase="$1"
    local top_k="$2"
    local strength="$3"
    local sample_gate_mode="$4"
    local strength_tag
    strength_tag="$(printf "%s" "${strength}" | tr "." "p")"
    local gate_tag="${sample_gate_mode}"
    if [ "${sample_gate_mode}" = "off" ]; then
        gate_tag="token"
    fi
    local tag="qproj_${gate_tag}_${phase}_top${top_k}_s${strength_tag}"
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
        --query_direction_project \
        --query_direction_calibration "${CALIBRATION_NPZ}" \
        --query_direction_top_k "${top_k}" \
        --query_direction_min_auroc "${QUERY_DIRECTION_MIN_AUROC}" \
        --query_direction_strength "${strength}" \
        --query_direction_gate_mode "${QUERY_DIRECTION_GATE_MODE}" \
        --query_direction_temperature "${QUERY_DIRECTION_TEMPERATURE}" \
        --query_direction_phase "${phase}" \
        --query_direction_prefill_positions "${QUERY_DIRECTION_PREFILL_POSITIONS}" \
        --query_direction_sample_gate_mode "${sample_gate_mode}" \
        --query_direction_sample_gate_positions "${QUERY_DIRECTION_SAMPLE_GATE_POSITIONS}" \
        --query_direction_sample_gate_scale "${QUERY_DIRECTION_SAMPLE_GATE_SCALE}" \
        --query_direction_sample_gate_min "${QUERY_DIRECTION_SAMPLE_GATE_MIN}" \
        --query_direction_sample_gate_max "${QUERY_DIRECTION_SAMPLE_GATE_MAX}" \
        "${allow_negative_args[@]}" \
        "${diagnostic_args[@]}" \
        2>&1 | tee "${LOG_DIR}/${tag}.log"

    python eval_scripts/eval_utils/eval_chair.py \
        --annotation-dir "${ANNOTATION_DIR}" \
        --answers-file "${answers_file}" \
        --caption_file "$(basename "${CAPTION_FILE_PATH}")"
}

for phase in ${QUERY_DIRECTION_PHASES}; do
    for top_k in ${QUERY_DIRECTION_TOP_KS}; do
        for strength in ${QUERY_DIRECTION_STRENGTHS}; do
            for sample_gate_mode in ${QUERY_DIRECTION_SAMPLE_GATE_MODES}; do
                run_projection "${phase}" "${top_k}" "${strength}" "${sample_gate_mode}"
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

out = "${OUTPUT_DIR}/query_projection_caption_summary.csv"
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

echo "[summary] query projection caption metrics"
if [ -f "${OUTPUT_DIR}/query_projection_caption_summary.csv" ]; then
    column -s, -t "${OUTPUT_DIR}/query_projection_caption_summary.csv"
fi

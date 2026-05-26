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
TOP_K="${TOP_K:-20}"
FORCE="${FORCE:-0}"

BASE_RESULT_PATH="${BASE_RESULT_PATH:-./results/${DATASET}/soft_routing_smoke_n500_seed42_tau0.4_T0.05}"
CALIBRATION_NPZ="${CALIBRATION_NPZ:-${BASE_RESULT_PATH}/query_direction_probe_l2_l13_31_hallboth_max100/query_direction_calibration.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-${BASE_RESULT_PATH}/entropy_aware_adhh_n${NUM_SAMPLES}}"
LOG_DIR="${LOG_DIR:-./logs/soft_routing}"

ENTROPY_AWARE_GAMMAS="${ENTROPY_AWARE_GAMMAS:-1 2 4}"
ENTROPY_AWARE_ENTROPY_SOURCES="${ENTROPY_AWARE_ENTROPY_SOURCES:-full text}"
ENTROPY_AWARE_STRENGTH_CAPS="${ENTROPY_AWARE_STRENGTH_CAPS:-1}"
ENTROPY_AWARE_RENORMALIZE="${ENTROPY_AWARE_RENORMALIZE:-0}"
ENTROPY_AWARE_QUERY_GATED="${ENTROPY_AWARE_QUERY_GATED:-0 1}"

QUERY_GATE_TOP_K="${QUERY_GATE_TOP_K:-1}"
QUERY_GATE_MIN_AUROC="${QUERY_GATE_MIN_AUROC:-0.0}"
QUERY_GATE_RISK_MODE="${QUERY_GATE_RISK_MODE:-z_softplus}"
QUERY_GATE_SOURCE="${QUERY_GATE_SOURCE:-previous}"
QUERY_GATE_RISK_SCALE="${QUERY_GATE_RISK_SCALE:-1.0}"
QUERY_GATE_MAX_RISK="${QUERY_GATE_MAX_RISK:-1.0}"
QUERY_GATE_MIN_SCALE="${QUERY_GATE_MIN_SCALE:-0.0}"
QUERY_GATE_MAX_SCALE="${QUERY_GATE_MAX_SCALE:-1.0}"

RUN_BASELINES="${RUN_BASELINES:-1}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

echo "[info] output dir: ${OUTPUT_DIR}"
echo "[info] num samples: ${NUM_SAMPLES}"
echo "[info] gammas: ${ENTROPY_AWARE_GAMMAS}"
echo "[info] entropy sources: ${ENTROPY_AWARE_ENTROPY_SOURCES}"
echo "[info] strength caps: ${ENTROPY_AWARE_STRENGTH_CAPS}"
echo "[info] renormalize modes: ${ENTROPY_AWARE_RENORMALIZE}"
echo "[info] q-gated modes: ${ENTROPY_AWARE_QUERY_GATED}"
echo "[info] calibration: ${CALIBRATION_NPZ}"
echo "[info] q-gate risk mode: ${QUERY_GATE_RISK_MODE}"

eval_chair() {
    local answers_file="$1"
    python eval_scripts/eval_utils/eval_chair.py \
        --annotation-dir "${ANNOTATION_DIR}" \
        --answers-file "${answers_file}" \
        --caption_file "$(basename "${CAPTION_FILE_PATH}")"
}

run_eval() {
    local tag="$1"
    shift
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
        "$@" \
        2>&1 | tee "${LOG_DIR}/entropy_aware_adhh_${tag}.log"
    eval_chair "${answers_file}"
}

if [ "${RUN_BASELINES}" = "1" ]; then
    run_eval "greedy"
    run_eval "adhh_hard" \
        --adaptive_deactivate \
        --adhh_threshold "${ADHH_THRESHOLD}" \
        --head_prior_mode uniform \
        --top_k "${TOP_K}"
fi

for gamma in ${ENTROPY_AWARE_GAMMAS}; do
    for source in ${ENTROPY_AWARE_ENTROPY_SOURCES}; do
        for cap in ${ENTROPY_AWARE_STRENGTH_CAPS}; do
            for renorm in ${ENTROPY_AWARE_RENORMALIZE}; do
                for qgated in ${ENTROPY_AWARE_QUERY_GATED}; do
                    gamma_tag="$(printf "%s" "${gamma}" | tr "." "p")"
                    cap_tag="$(printf "%s" "${cap}" | tr "." "p")"
                    tag="eahh_${source}_g${gamma_tag}_cap${cap_tag}"
                    extra_args=()
                    if [ "${renorm}" = "1" ]; then
                        tag="${tag}_renorm"
                        extra_args+=(--entropy_aware_renormalize)
                    fi
                    if [ "${qgated}" = "1" ]; then
                        if [ ! -f "${CALIBRATION_NPZ}" ]; then
                            echo "[error] missing query calibration: ${CALIBRATION_NPZ}" >&2
                            exit 1
                        fi
                        tag="${tag}_qgate"
                        extra_args+=(
                            --entropy_aware_use_query_gate
                            --adhh_query_gate_calibration "${CALIBRATION_NPZ}"
                            --adhh_query_gate_top_k "${QUERY_GATE_TOP_K}"
                            --adhh_query_gate_min_auroc "${QUERY_GATE_MIN_AUROC}"
                            --adhh_query_gate_risk_mode "${QUERY_GATE_RISK_MODE}"
                            --adhh_query_gate_source "${QUERY_GATE_SOURCE}"
                            --adhh_query_gate_risk_scale "${QUERY_GATE_RISK_SCALE}"
                            --adhh_query_gate_max_risk "${QUERY_GATE_MAX_RISK}"
                            --adhh_query_gate_min_scale "${QUERY_GATE_MIN_SCALE}"
                            --adhh_query_gate_max_scale "${QUERY_GATE_MAX_SCALE}"
                        )
                    fi
                    run_eval "${tag}" \
                        --entropy_aware_deactivate \
                        --entropy_aware_gamma "${gamma}" \
                        --entropy_aware_entropy_source "${source}" \
                        --entropy_aware_strength_cap "${cap}" \
                        --entropy_aware_phase decode \
                        --adhh_threshold "${ADHH_THRESHOLD}" \
                        --head_prior_mode uniform \
                        --top_k "${TOP_K}" \
                        "${extra_args[@]}"
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

out = "${OUTPUT_DIR}/entropy_aware_adhh_summary.csv"
if rows:
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
print(out)
PY

echo "[summary] entropy-aware AD-HH metrics"
column -s, -t "${OUTPUT_DIR}/entropy_aware_adhh_summary.csv"

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
OUTPUT_DIR="${OUTPUT_DIR:-${BASE_RESULT_PATH}/query_routed_adhh_n${NUM_SAMPLES}}"
LOG_DIR="${LOG_DIR:-./logs/soft_routing}"
GREEDY_EVAL="${GREEDY_EVAL:-${BASE_RESULT_PATH}/greedy/captions_eval_results.json}"

QUERY_ROUTED_ADHH_TOP_KS="${QUERY_ROUTED_ADHH_TOP_KS:-1}"
QUERY_ROUTED_ADHH_MIN_SCALES="${QUERY_ROUTED_ADHH_MIN_SCALES:-0.25 0.5}"
QUERY_ROUTED_ADHH_RISK_SCALES="${QUERY_ROUTED_ADHH_RISK_SCALES:-0.03}"
QUERY_ROUTED_ADHH_RISK_MODES="${QUERY_ROUTED_ADHH_RISK_MODES:-margin}"
QUERY_ROUTED_ADHH_SOURCES="${QUERY_ROUTED_ADHH_SOURCES:-current_or_previous}"
QUERY_ROUTED_ADHH_MIN_AUROC="${QUERY_ROUTED_ADHH_MIN_AUROC:-0.0}"
QUERY_ROUTED_ADHH_DETECTOR_AGGREGATION="${QUERY_ROUTED_ADHH_DETECTOR_AGGREGATION:-max}"
QUERY_ROUTED_ADHH_GLOBAL_AGGREGATION="${QUERY_ROUTED_ADHH_GLOBAL_AGGREGATION:-max}"
QUERY_ROUTED_ADHH_MAX_RISK="${QUERY_ROUTED_ADHH_MAX_RISK:-1.0}"

RUN_BASELINES="${RUN_BASELINES:-1}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

if [ ! -f "${CALIBRATION_NPZ}" ]; then
    echo "[error] missing query calibration: ${CALIBRATION_NPZ}" >&2
    exit 1
fi

echo "[info] output dir: ${OUTPUT_DIR}"
echo "[info] calibration: ${CALIBRATION_NPZ}"
echo "[info] detector top ks: ${QUERY_ROUTED_ADHH_TOP_KS}"
echo "[info] min scales: ${QUERY_ROUTED_ADHH_MIN_SCALES}"
echo "[info] risk scales: ${QUERY_ROUTED_ADHH_RISK_SCALES}"
echo "[info] risk modes: ${QUERY_ROUTED_ADHH_RISK_MODES}"
echo "[info] sources: ${QUERY_ROUTED_ADHH_SOURCES}"
echo "[info] num samples: ${NUM_SAMPLES}"

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
        2>&1 | tee "${LOG_DIR}/query_routed_adhh_${tag}.log"
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

for detector_top_k in ${QUERY_ROUTED_ADHH_TOP_KS}; do
    for min_scale in ${QUERY_ROUTED_ADHH_MIN_SCALES}; do
        for risk_scale in ${QUERY_ROUTED_ADHH_RISK_SCALES}; do
            for risk_mode in ${QUERY_ROUTED_ADHH_RISK_MODES}; do
                for source in ${QUERY_ROUTED_ADHH_SOURCES}; do
                    min_tag="$(printf "%s" "${min_scale}" | tr "." "p")"
                    risk_tag="$(printf "%s" "${risk_scale}" | tr "." "p")"
                    tag="qadhh_${risk_mode}_det${detector_top_k}_${source}_min${min_tag}_rs${risk_tag}"
                    run_eval "${tag}" \
                        --adaptive_deactivate \
                        --adhh_threshold "${ADHH_THRESHOLD}" \
                        --head_prior_mode uniform \
                        --top_k "${TOP_K}" \
                        --adhh_query_gate_calibration "${CALIBRATION_NPZ}" \
                        --adhh_query_gate_top_k "${detector_top_k}" \
                        --adhh_query_gate_min_auroc "${QUERY_ROUTED_ADHH_MIN_AUROC}" \
                        --adhh_query_gate_risk_mode "${risk_mode}" \
                        --adhh_query_gate_detector_aggregation "${QUERY_ROUTED_ADHH_DETECTOR_AGGREGATION}" \
                        --adhh_query_gate_global_aggregation "${QUERY_ROUTED_ADHH_GLOBAL_AGGREGATION}" \
                        --adhh_query_gate_source "${source}" \
                        --adhh_query_gate_min_scale "${min_scale}" \
                        --adhh_query_gate_max_scale 1.0 \
                        --adhh_query_gate_risk_scale "${risk_scale}" \
                        --adhh_query_gate_max_risk "${QUERY_ROUTED_ADHH_MAX_RISK}"
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

out = "${OUTPUT_DIR}/query_routed_adhh_summary.csv"
if rows:
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
print(out)
PY

mapfile -t eval_files < <(find "${OUTPUT_DIR}" -mindepth 2 -maxdepth 2 -name captions_eval_results.json | sort)
if [ "${#eval_files[@]}" -gt 0 ]; then
    python -m eval_scripts.soft_routing.analyze_hallucination_position \
        --eval-results "${eval_files[@]}" \
        --names $(for path in "${eval_files[@]}"; do basename "$(dirname "${path}")"; done) \
        --match-common-image-ids \
        --output-dir "${OUTPUT_DIR}/position_summary"
fi

if [ -f "${OUTPUT_DIR}/greedy/captions_eval_results.json" ]; then
    for path in "${eval_files[@]}"; do
        method="$(basename "$(dirname "${path}")")"
        if [ "${method}" = "greedy" ]; then
            continue
        fi
        python -m eval_scripts.soft_routing.analyze_pairwise_object_inventory \
            --base "${OUTPUT_DIR}/greedy/captions_eval_results.json" \
            --target "${path}" \
            --base-name greedy \
            --target-name "${method}" \
            --output-dir "${OUTPUT_DIR}/pairwise_object_inventory/greedy_vs_${method}"
    done
fi

echo "[summary] query-routed AD-HH metrics"
if [ -f "${OUTPUT_DIR}/query_routed_adhh_summary.csv" ]; then
    column -s, -t "${OUTPUT_DIR}/query_routed_adhh_summary.csv"
fi

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
TOP_K="${TOP_K:-20}"
HEAD_PRIOR_MODE="${HEAD_PRIOR_MODE:-score}"

VISUAL_GATE_PROXIES="${VISUAL_GATE_PROXIES:-value mass}"
VISUAL_GATE_BETAS="${VISUAL_GATE_BETAS:-0.5 0.75 1.0}"
VISUAL_GATE_V0S="${VISUAL_GATE_V0S:-0.35 0.5}"
VISUAL_GATE_TEMPERATURE="${VISUAL_GATE_TEMPERATURE:-0.15}"
VISUAL_GATE_GAMMA="${VISUAL_GATE_GAMMA:-1.0}"
VISUAL_GATE_RECENT_WEIGHT="${VISUAL_GATE_RECENT_WEIGHT:-0.0}"
VISUAL_GATE_RECENT_WINDOW="${VISUAL_GATE_RECENT_WINDOW:-16}"
VISUAL_GATE_TAU_LOW="${VISUAL_GATE_TAU_LOW:-0.4}"
VISUAL_GATE_TAU_HIGH="${VISUAL_GATE_TAU_HIGH:-0.9}"

BASE_RESULT_PATH="${BASE_RESULT_PATH:-./results/${DATASET}/soft_routing_smoke_n${NUM_SAMPLES}_seed${SEED}_tau${ADHH_THRESHOLD}_T${SOFT_TEMPERATURE}}"
PRIOR_PATH="${PRIOR_PATH:-${BASE_RESULT_PATH}/fixed_adhh20_score_prior/attribution_result.json}"
HEAD_THRESHOLDS_PATH="${HEAD_THRESHOLDS_PATH:-}"
LOG_DIR="${LOG_DIR:-./logs/soft_routing}"
FORCE="${FORCE:-0}"

mkdir -p "${LOG_DIR}"

if [ -n "${PRIOR_PATH}" ] && [ ! -f "${PRIOR_PATH}" ]; then
    echo "[warn] missing prior path: ${PRIOR_PATH}"
    echo "[warn] falling back to built-in AD-HH heads with rank priors"
    PRIOR_PATH=""
    HEAD_PRIOR_MODE="rank"
fi

run_chair_eval() {
    local answers_file="$1"
    python eval_scripts/eval_utils/eval_chair.py \
        --annotation-dir "${ANNOTATION_DIR}" \
        --answers-file "${answers_file}" \
        --caption_file "$(basename "${CAPTION_FILE_PATH}")"
}

run_policy() {
    local proxy="$1"
    local beta="$2"
    local v0="$3"
    local beta_tag v0_tag
    beta_tag=$(echo "${beta}" | tr '.' 'p' | tr '-' 'm')
    v0_tag=$(echo "${v0}" | tr '.' 'p' | tr '-' 'm')
    local tag="visual_gate_${proxy}_b${beta_tag}_v${v0_tag}_g${VISUAL_GATE_GAMMA}"
    local result_path="${BASE_RESULT_PATH}/${tag}"
    local answers_file="${result_path}/captions.jsonl"
    local eval_file="${result_path}/captions_eval_results.json"
    mkdir -p "${result_path}"
    local threshold_args=()
    if [ -n "${HEAD_THRESHOLDS_PATH}" ]; then
        threshold_args=(--head_thresholds_path "${HEAD_THRESHOLDS_PATH}")
    fi

    if [ "${FORCE}" != "1" ] && [ -f "${eval_file}" ]; then
        echo "[skip] ${tag}"
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
        --visual_gate_deactivate \
        --attention_head_path "${PRIOR_PATH}" \
        --top_k "${TOP_K}" \
        --head_prior_mode "${HEAD_PRIOR_MODE}" \
        --adhh_threshold "${ADHH_THRESHOLD}" \
        --visual_gate_proxy "${proxy}" \
        --visual_gate_gamma "${VISUAL_GATE_GAMMA}" \
        --visual_gate_beta "${beta}" \
        --visual_gate_v0 "${v0}" \
        --visual_gate_temperature "${VISUAL_GATE_TEMPERATURE}" \
        --visual_gate_recent_weight "${VISUAL_GATE_RECENT_WEIGHT}" \
        --visual_gate_recent_window "${VISUAL_GATE_RECENT_WINDOW}" \
        --visual_gate_tau_low "${VISUAL_GATE_TAU_LOW}" \
        --visual_gate_tau_high "${VISUAL_GATE_TAU_HIGH}" \
        "${threshold_args[@]}" \
        2>&1 | tee "${LOG_DIR}/${tag}.log"

    run_chair_eval "${answers_file}"
}

for proxy in ${VISUAL_GATE_PROXIES}; do
    for beta in ${VISUAL_GATE_BETAS}; do
        for v0 in ${VISUAL_GATE_V0S}; do
            run_policy "${proxy}" "${beta}" "${v0}"
        done
    done
done

echo "[summary] visual gate policy metrics"
python - <<PY
import csv, glob, json, os
base = "${BASE_RESULT_PATH}"
rows = []
for path in sorted(glob.glob(base + "/visual_gate_*/captions_eval_results.json")):
    method = os.path.basename(os.path.dirname(path))
    metrics = json.load(open(path))["overall_metrics"]
    print("\\n" + path)
    print(metrics)
    rows.append({
        "method": method,
        "CHAIRs": metrics.get("CHAIRs"),
        "CHAIRi": metrics.get("CHAIRi"),
        "Bleu1": metrics.get("Bleu", [None, None, None, None])[0],
        "Bleu2": metrics.get("Bleu", [None, None, None, None])[1],
        "Bleu3": metrics.get("Bleu", [None, None, None, None])[2],
        "Bleu4": metrics.get("Bleu", [None, None, None, None])[3],
        "avg_caption_length": metrics.get("avg_caption_length"),
    })
out_path = os.path.join(base, "visual_gate_policy_metrics_summary.csv")
if rows:
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("\\nwrote", out_path)
PY

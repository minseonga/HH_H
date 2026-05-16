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
SOFT_GAMMA="${SOFT_GAMMA:-0.75}"
TOP_K="${TOP_K:-20}"
HEAD_PRIOR_MODE="${HEAD_PRIOR_MODE:-score}"

RETENTION_POLICY_MODE="${RETENTION_POLICY_MODE:-hard_or_soft}"
RETENTION_FEATURES="${RETENTION_FEATURES:-mean_prior_text_mass}"
RETENTION_RHOS="${RETENTION_RHOS:-0.08 0.10 0.12 0.14}"
RETENTION_LAMBDA="${RETENTION_LAMBDA:-1.0}"

BASE_RESULT_PATH="${BASE_RESULT_PATH:-./results/${DATASET}/soft_routing_smoke_n${NUM_SAMPLES}_seed${SEED}_tau${ADHH_THRESHOLD}_T${SOFT_TEMPERATURE}}"
PRIOR_PATH="${PRIOR_PATH:-${BASE_RESULT_PATH}/fixed_adhh20_score_prior/attribution_result.json}"
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
    local feature="$1"
    local rho="$2"
    local rho_tag
    rho_tag=$(echo "${rho}" | tr '.' 'p' | tr '-' 'm')
    local tag="retention_${RETENTION_POLICY_MODE}_${feature}_rho${rho_tag}_sg${SOFT_GAMMA}"
    local result_path="${BASE_RESULT_PATH}/${tag}"
    local answers_file="${result_path}/captions.jsonl"
    local eval_file="${result_path}/captions_eval_results.json"
    mkdir -p "${result_path}"

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
        --retention_aware_deactivate \
        --attention_head_path "${PRIOR_PATH}" \
        --top_k "${TOP_K}" \
        --head_prior_mode "${HEAD_PRIOR_MODE}" \
        --adhh_threshold "${ADHH_THRESHOLD}" \
        --retention_policy_mode "${RETENTION_POLICY_MODE}" \
        --retention_feature "${feature}" \
        --retention_rho "${rho}" \
        --retention_lambda "${RETENTION_LAMBDA}" \
        --retention_soft_gamma "${SOFT_GAMMA}" \
        --retention_soft_temperature "${SOFT_TEMPERATURE}" \
        2>&1 | tee "${LOG_DIR}/${tag}.log"

    run_chair_eval "${answers_file}"
}

for feature in ${RETENTION_FEATURES}; do
    for rho in ${RETENTION_RHOS}; do
        run_policy "${feature}" "${rho}"
    done
done

echo "[summary] retention policy metrics"
python - <<PY
import csv, glob, json, os
base = "${BASE_RESULT_PATH}"
rows = []
for path in sorted(glob.glob(base + "/retention_*/captions_eval_results.json")):
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
out_path = os.path.join(base, "retention_policy_metrics_summary.csv")
if rows:
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("\\nwrote", out_path)
PY

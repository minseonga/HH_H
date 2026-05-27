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
FORCE="${FORCE:-0}"

BASE_RESULT_PATH="${BASE_RESULT_PATH:-./results/${DATASET}/soft_routing_smoke_n500_seed42_tau0.4_T0.05}"
OUTPUT_DIR="${OUTPUT_DIR:-${BASE_RESULT_PATH}/contrastive_dynamic_n${NUM_SAMPLES}}"
LOG_DIR="${LOG_DIR:-./logs/soft_routing}"

HEAD_PRIOR_MODE="${HEAD_PRIOR_MODE:-auto}"
ATTENTION_HEAD_SCORE_KEY="${ATTENTION_HEAD_SCORE_KEY:-global__itext_all__C_toi_HminusG}"
ATTENTION_HEAD_SCORE_NORMALIZE="${ATTENTION_HEAD_SCORE_NORMALIZE:-rank_percentile}"
DEFAULT_TEAM_HEAD_PATH="../ADHH/LLaVA/results/coco/llava-v1.5-7b_base_original_qa_n3000/surrogate_hh_scores/surrogate_score_zoo/ranked_heads_${ATTENTION_HEAD_SCORE_KEY}.json"
ATTENTION_HEAD_PATH="${ATTENTION_HEAD_PATH:-}"
if [ -z "${ATTENTION_HEAD_PATH}" ] && [ -f "${DEFAULT_TEAM_HEAD_PATH}" ]; then
    ATTENTION_HEAD_PATH="${DEFAULT_TEAM_HEAD_PATH}"
fi
if [ -n "${ATTENTION_HEAD_PATH}" ] && [ ! -f "${ATTENTION_HEAD_PATH}" ]; then
    if [ "${ATTENTION_HEAD_PATH}" = "/path/to/team_top_heads.json" ] && [ -f "${DEFAULT_TEAM_HEAD_PATH}" ]; then
        echo "[warn] ATTENTION_HEAD_PATH was left as the placeholder; using ${DEFAULT_TEAM_HEAD_PATH}"
        ATTENTION_HEAD_PATH="${DEFAULT_TEAM_HEAD_PATH}"
    else
        echo "[error] attention head file not found: ${ATTENTION_HEAD_PATH}" >&2
        echo "[hint] unset ATTENTION_HEAD_PATH to auto-use: ${DEFAULT_TEAM_HEAD_PATH}" >&2
        exit 2
    fi
fi
TOP_KS="${TOP_KS:-100 150 200}"
STRENGTHS="${STRENGTHS:-0.7 1.0}"
BETAS="${BETAS:-8 10}"
TAU="${TAU:-0.9}"
CONCENTRATION_MODES="${CONCENTRATION_MODES:-none inverse_text_entropy sqrt_inverse_text_entropy}"
CONCENTRATION_POWER="${CONCENTRATION_POWER:-1.0}"
RENORMALIZE="${RENORMALIZE:-1}"
RUN_BASELINES="${RUN_BASELINES:-1}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

echo "[info] output dir: ${OUTPUT_DIR}"
echo "[info] num samples: ${NUM_SAMPLES}"
echo "[info] attention head path: ${ATTENTION_HEAD_PATH:-<default>}"
echo "[info] head prior mode: ${HEAD_PRIOR_MODE}"
echo "[info] head score: ${ATTENTION_HEAD_SCORE_KEY}/${ATTENTION_HEAD_SCORE_NORMALIZE}"
echo "[info] top ks: ${TOP_KS}"
echo "[info] strengths: ${STRENGTHS}"
echo "[info] betas: ${BETAS}"
echo "[info] tau: ${TAU}"
echo "[info] concentration modes: ${CONCENTRATION_MODES}"
echo "[info] renormalize: ${RENORMALIZE}"

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
        2>&1 | tee "${LOG_DIR}/contrastive_dynamic_${tag}.log"
    eval_chair "${answers_file}"
}

head_path_arg=()
if [ -n "${ATTENTION_HEAD_PATH}" ]; then
    head_path_arg=(
        --attention_head_path "${ATTENTION_HEAD_PATH}"
        --attention_head_score_key "${ATTENTION_HEAD_SCORE_KEY}"
        --attention_head_score_normalize "${ATTENTION_HEAD_SCORE_NORMALIZE}"
    )
fi
renorm_arg=()
if [ "${RENORMALIZE}" = "1" ]; then
    renorm_arg=(--contrastive_dynamic_renormalize)
fi

if [ "${RUN_BASELINES}" = "1" ]; then
    run_eval "greedy"
    run_eval "adhh_hard_k20" \
        --adaptive_deactivate \
        --adhh_threshold "${ADHH_THRESHOLD}" \
        --head_prior_mode uniform \
        --top_k 20
fi

for top_k in ${TOP_KS}; do
    for strength in ${STRENGTHS}; do
        strength_tag="$(printf "%s" "${strength}" | tr "." "p")"
        for beta in ${BETAS}; do
            beta_tag="$(printf "%s" "${beta}" | tr "." "p")"
            for concentration_mode in ${CONCENTRATION_MODES}; do
                tag="contrastive_dynamic_k${top_k}_s${strength_tag}_q${beta_tag}_${concentration_mode}"
                if [ "${RENORMALIZE}" = "1" ]; then
                    tag="${tag}_renorm"
                fi
                run_eval "${tag}" \
                    --contrastive_dynamic_deactivate \
                    --contrastive_dynamic_strength "${strength}" \
                    --contrastive_dynamic_beta "${beta}" \
                    --contrastive_dynamic_tau "${TAU}" \
                    --contrastive_dynamic_concentration_mode "${concentration_mode}" \
                    --contrastive_dynamic_concentration_power "${CONCENTRATION_POWER}" \
                    "${renorm_arg[@]}" \
                    --adhh_threshold "${ADHH_THRESHOLD}" \
                    "${head_path_arg[@]}" \
                    --head_prior_mode "${HEAD_PRIOR_MODE}" \
                    --top_k "${top_k}"
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
        "METEOR": metrics.get("METEOR"),
        "CIDEr": metrics.get("CIDEr"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1": metrics.get("f1"),
        "avg_caption_length": metrics.get("avg_caption_length"),
    })
out = "${OUTPUT_DIR}/contrastive_dynamic_summary.csv"
with open(out, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["method"])
    writer.writeheader()
    writer.writerows(rows)
print(out)
PY

column -s, -t "${OUTPUT_DIR}/contrastive_dynamic_summary.csv"

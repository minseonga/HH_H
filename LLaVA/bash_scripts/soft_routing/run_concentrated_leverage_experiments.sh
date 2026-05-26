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
OUTPUT_DIR="${OUTPUT_DIR:-${BASE_RESULT_PATH}/concentrated_leverage_n${NUM_SAMPLES}}"
LOG_DIR="${LOG_DIR:-./logs/soft_routing}"

RUN_BASELINES="${RUN_BASELINES:-1}"

TEXT_MASS_GAMMAS="${TEXT_MASS_GAMMAS:-0.5}"
CONCENTRATED_GAMMAS="${CONCENTRATED_GAMMAS:-4}"
VISUAL_BOOST_GAMMAS="${VISUAL_BOOST_GAMMAS:-4}"
HYBRID_GAMMAS="${HYBRID_GAMMAS:-4}"
STRENGTH_CAPS="${STRENGTH_CAPS:-1}"

TEXT_SUPPRESS_TARGET_HEADS="${TEXT_SUPPRESS_TARGET_HEADS:-all}"
TEXT_SUPPRESS_RENORMALIZE="${TEXT_SUPPRESS_RENORMALIZE:-0}"

FCCT_MIN_LAYER="${FCCT_MIN_LAYER:-11}"
FCCT_MAX_LAYER="${FCCT_MAX_LAYER:-20}"
FCCT_TARGET_HEADS="${FCCT_TARGET_HEADS:-all}"
VISUAL_BETA="${VISUAL_BETA:-1}"
TEXT_ALPHA="${TEXT_ALPHA:-1}"
FCCT_RENORMALIZE="${FCCT_RENORMALIZE:-1}"
FCCT_MAX_IMAGE_FACTOR="${FCCT_MAX_IMAGE_FACTOR:-0}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

echo "[info] output dir: ${OUTPUT_DIR}"
echo "[info] num samples: ${NUM_SAMPLES}"
echo "[info] text-mass gammas: ${TEXT_MASS_GAMMAS}"
echo "[info] concentrated gammas: ${CONCENTRATED_GAMMAS}"
echo "[info] visual-boost gammas: ${VISUAL_BOOST_GAMMAS}"
echo "[info] hybrid gammas: ${HYBRID_GAMMAS}"
echo "[info] strength caps: ${STRENGTH_CAPS}"
echo "[info] text suppress target heads: ${TEXT_SUPPRESS_TARGET_HEADS}"
echo "[info] FCCT layers: ${FCCT_MIN_LAYER}-${FCCT_MAX_LAYER}"
echo "[info] FCCT target heads: ${FCCT_TARGET_HEADS}"

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
        2>&1 | tee "${LOG_DIR}/concentrated_leverage_${tag}.log"
    eval_chair "${answers_file}"
}

renorm_arg=()
if [ "${TEXT_SUPPRESS_RENORMALIZE}" = "1" ]; then
    renorm_arg=(--entropy_aware_renormalize)
fi

fcct_renorm_arg=()
if [ "${FCCT_RENORMALIZE}" = "1" ]; then
    fcct_renorm_arg=(--concentrated_visual_boost_renormalize)
fi

if [ "${RUN_BASELINES}" = "1" ]; then
    run_eval "greedy"
    run_eval "adhh_hard" \
        --adaptive_deactivate \
        --adhh_threshold "${ADHH_THRESHOLD}" \
        --head_prior_mode uniform \
        --top_k "${TOP_K}"
fi

for cap in ${STRENGTH_CAPS}; do
    cap_tag="$(printf "%s" "${cap}" | tr "." "p")"

    for gamma in ${TEXT_MASS_GAMMAS}; do
        gamma_tag="$(printf "%s" "${gamma}" | tr "." "p")"
        tag="continuous_text_mass_all_g${gamma_tag}_cap${cap_tag}"
        if [ "${TEXT_SUPPRESS_TARGET_HEADS}" != "all" ]; then
            tag="continuous_text_mass_${TEXT_SUPPRESS_TARGET_HEADS}_g${gamma_tag}_cap${cap_tag}"
        fi
        if [ "${TEXT_SUPPRESS_RENORMALIZE}" = "1" ]; then
            tag="${tag}_renorm"
        fi
        run_eval "${tag}" \
            --entropy_aware_deactivate \
            --entropy_aware_gamma "${gamma}" \
            --entropy_aware_entropy_source text \
            --entropy_aware_entropy_transform none \
            --entropy_aware_target_heads "${TEXT_SUPPRESS_TARGET_HEADS}" \
            --entropy_aware_strength_cap "${cap}" \
            --entropy_aware_phase decode \
            --adhh_threshold "${ADHH_THRESHOLD}" \
            --head_prior_mode uniform \
            --top_k "${TOP_K}" \
            "${renorm_arg[@]}"
    done

    for gamma in ${CONCENTRATED_GAMMAS}; do
        gamma_tag="$(printf "%s" "${gamma}" | tr "." "p")"
        tag="continuous_concentrated_all_g${gamma_tag}_cap${cap_tag}"
        if [ "${TEXT_SUPPRESS_TARGET_HEADS}" != "all" ]; then
            tag="continuous_concentrated_${TEXT_SUPPRESS_TARGET_HEADS}_g${gamma_tag}_cap${cap_tag}"
        fi
        if [ "${TEXT_SUPPRESS_RENORMALIZE}" = "1" ]; then
            tag="${tag}_renorm"
        fi
        run_eval "${tag}" \
            --entropy_aware_deactivate \
            --entropy_aware_gamma "${gamma}" \
            --entropy_aware_entropy_source text \
            --entropy_aware_entropy_transform inverse \
            --entropy_aware_target_heads "${TEXT_SUPPRESS_TARGET_HEADS}" \
            --entropy_aware_strength_cap "${cap}" \
            --entropy_aware_phase decode \
            --adhh_threshold "${ADHH_THRESHOLD}" \
            --head_prior_mode uniform \
            --top_k "${TOP_K}" \
            "${renorm_arg[@]}"
    done

    for gamma in ${VISUAL_BOOST_GAMMAS}; do
        gamma_tag="$(printf "%s" "${gamma}" | tr "." "p")"
        tag="fcct_l${FCCT_MIN_LAYER}_${FCCT_MAX_LAYER}_visual_boost_g${gamma_tag}_cap${cap_tag}"
        if [ "${FCCT_RENORMALIZE}" = "1" ]; then
            tag="${tag}_renorm"
        fi
        run_eval "${tag}" \
            --concentrated_visual_boost \
            --concentrated_visual_boost_mode boost \
            --concentrated_visual_boost_gamma "${gamma}" \
            --concentrated_visual_boost_visual_beta "${VISUAL_BETA}" \
            --concentrated_visual_boost_strength_cap "${cap}" \
            --concentrated_visual_boost_max_image_factor "${FCCT_MAX_IMAGE_FACTOR}" \
            --concentrated_visual_boost_min_layer "${FCCT_MIN_LAYER}" \
            --concentrated_visual_boost_max_layer "${FCCT_MAX_LAYER}" \
            --concentrated_visual_boost_target_heads "${FCCT_TARGET_HEADS}" \
            --concentrated_visual_boost_phase decode \
            --adhh_threshold "${ADHH_THRESHOLD}" \
            --head_prior_mode uniform \
            --top_k "${TOP_K}" \
            "${fcct_renorm_arg[@]}"
    done

    for gamma in ${HYBRID_GAMMAS}; do
        gamma_tag="$(printf "%s" "${gamma}" | tr "." "p")"
        tag="fcct_l${FCCT_MIN_LAYER}_${FCCT_MAX_LAYER}_hybrid_g${gamma_tag}_cap${cap_tag}"
        if [ "${FCCT_RENORMALIZE}" = "1" ]; then
            tag="${tag}_renorm"
        fi
        run_eval "${tag}" \
            --concentrated_visual_boost \
            --concentrated_visual_boost_mode hybrid \
            --concentrated_visual_boost_gamma "${gamma}" \
            --concentrated_visual_boost_text_alpha "${TEXT_ALPHA}" \
            --concentrated_visual_boost_visual_beta "${VISUAL_BETA}" \
            --concentrated_visual_boost_strength_cap "${cap}" \
            --concentrated_visual_boost_max_image_factor "${FCCT_MAX_IMAGE_FACTOR}" \
            --concentrated_visual_boost_min_layer "${FCCT_MIN_LAYER}" \
            --concentrated_visual_boost_max_layer "${FCCT_MAX_LAYER}" \
            --concentrated_visual_boost_target_heads "${FCCT_TARGET_HEADS}" \
            --concentrated_visual_boost_phase decode \
            --adhh_threshold "${ADHH_THRESHOLD}" \
            --head_prior_mode uniform \
            --top_k "${TOP_K}" \
            "${fcct_renorm_arg[@]}"
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

def sort_key(row):
    chairs = row["CHAIRs"]
    chairi = row["CHAIRi"]
    return (
        999.0 if chairs is None else float(chairs),
        999.0 if chairi is None else float(chairi),
        row["method"],
    )

rows.sort(key=sort_key)
out = "${OUTPUT_DIR}/concentrated_leverage_summary.csv"
if rows:
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
print(out)
PY

echo "[summary] concentrated leverage metrics"
column -s, -t "${OUTPUT_DIR}/concentrated_leverage_summary.csv"

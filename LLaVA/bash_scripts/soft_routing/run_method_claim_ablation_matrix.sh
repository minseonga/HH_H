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

RANKED_HEADS="${RANKED_HEADS:-../ADHH/LLaVA/results/coco/llava-v1.5-7b_base_original_qa_n3000/surrogate_hh_scores/surrogate_score_zoo/ranked_heads_global__itext_all__C_toi_HminusG.json}"
BASE_EVAL_JSON="${BASE_EVAL_JSON:-../ADHH/LLaVA/results/coco/llava-v1.5-7b_base_n500/captions_eval_results.json}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/${DATASET}/method_claim_ablation_n${NUM_SAMPLES}_seed${SEED}}"
EVIDENCE_DIR="${EVIDENCE_DIR:-${OUTPUT_DIR}/method_claim_evidence}"
LOG_DIR="${LOG_DIR:-./logs/soft_routing}"

SELECTIONS="${SELECTIONS:-combined itext contrast}"
POLICIES="${POLICIES:-continuous hard}"
TOP_KS="${TOP_KS:-20 50 100 150}"
STRENGTHS="${STRENGTHS:-0.7}"
BETAS="${BETAS:-8 10}"
TAU="${TAU:-0.9}"
RENORMALIZE="${RENORMALIZE:-1}"
HEAD_PRIOR_MODE="${HEAD_PRIOR_MODE:-auto}"
ATTENTION_HEAD_SCORE_NORMALIZE="${ATTENTION_HEAD_SCORE_NORMALIZE:-rank_percentile}"
RUN_BASELINE="${RUN_BASELINE:-1}"

mkdir -p "${OUTPUT_DIR}" "${EVIDENCE_DIR}" "${LOG_DIR}"

python -m eval_scripts.soft_routing.build_method_claim_evidence \
    --ranked-heads "${RANKED_HEADS}" \
    --output-dir "${EVIDENCE_DIR}" \
    --base-eval-json "${BASE_EVAL_JSON}"

COMBINED_HEADS="${RANKED_HEADS}"
ITEXT_HEADS="${EVIDENCE_DIR}/component_rankings/ranked_heads_itext_all_from_combo.json"
CONTRAST_HEADS="${EVIDENCE_DIR}/component_rankings/ranked_heads_C_toi_HminusG_from_combo.json"

head_path_for_selection() {
    case "$1" in
        combined) echo "${COMBINED_HEADS}" ;;
        itext) echo "${ITEXT_HEADS}" ;;
        contrast) echo "${CONTRAST_HEADS}" ;;
        *) echo "[error] unknown selection: $1" >&2; exit 2 ;;
    esac
}

score_key_for_selection() {
    case "$1" in
        combined) echo "global__itext_all__C_toi_HminusG" ;;
        itext) echo "front_percentile" ;;
        contrast) echo "back_percentile" ;;
        *) echo "score" ;;
    esac
}

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
        2>&1 | tee "${LOG_DIR}/method_claim_ablation_${tag}.log"
    eval_chair "${answers_file}"
}

if [ "${RUN_BASELINE}" = "1" ]; then
    run_eval "greedy"
fi

for selection in ${SELECTIONS}; do
    head_path="$(head_path_for_selection "${selection}")"
    score_key="$(score_key_for_selection "${selection}")"
    if [ ! -f "${head_path}" ]; then
        echo "[error] missing head path for ${selection}: ${head_path}" >&2
        exit 2
    fi

    for top_k in ${TOP_KS}; do
        for policy in ${POLICIES}; do
            if [ "${policy}" = "hard" ]; then
                tag="${selection}_hard_k${top_k}"
                run_eval "${tag}" \
                    --adaptive_deactivate \
                    --adhh_threshold "${ADHH_THRESHOLD}" \
                    --attention_head_path "${head_path}" \
                    --attention_head_score_key "${score_key}" \
                    --attention_head_score_normalize "${ATTENTION_HEAD_SCORE_NORMALIZE}" \
                    --head_prior_mode uniform \
                    --top_k "${top_k}"
            elif [ "${policy}" = "continuous" ]; then
                for strength in ${STRENGTHS}; do
                    strength_tag="$(printf "%s" "${strength}" | tr "." "p")"
                    for beta in ${BETAS}; do
                        beta_tag="$(printf "%s" "${beta}" | tr "." "p")"
                        tag="${selection}_continuous_k${top_k}_s${strength_tag}_q${beta_tag}"
                        if [ "${RENORMALIZE}" = "1" ]; then
                            tag="${tag}_renorm"
                            renorm_arg=(--contrastive_dynamic_renormalize)
                        else
                            renorm_arg=()
                        fi
                        run_eval "${tag}" \
                            --contrastive_dynamic_deactivate \
                            --contrastive_dynamic_strength "${strength}" \
                            --contrastive_dynamic_beta "${beta}" \
                            --contrastive_dynamic_tau "${TAU}" \
                            "${renorm_arg[@]}" \
                            --adhh_threshold "${ADHH_THRESHOLD}" \
                            --attention_head_path "${head_path}" \
                            --attention_head_score_key "${score_key}" \
                            --attention_head_score_normalize "${ATTENTION_HEAD_SCORE_NORMALIZE}" \
                            --head_prior_mode "${HEAD_PRIOR_MODE}" \
                            --top_k "${top_k}"
                    done
                done
            else
                echo "[error] unknown policy: ${policy}" >&2
                exit 2
            fi
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
out = "${OUTPUT_DIR}/method_claim_ablation_summary.csv"
with open(out, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["method"])
    writer.writeheader()
    writer.writerows(rows)
print(out)
PY

column -s, -t "${OUTPUT_DIR}/method_claim_ablation_summary.csv"

#!/bin/bash

set -euo pipefail

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-True}"

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

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
DYNAMIC_GPU="${DYNAMIC_GPU:-${GPU0}}"
PARALLEL_GENERATION="${PARALLEL_GENERATION:-1}"
RUN_GENERATION="${RUN_GENERATION:-1}"
RUN_STRICT_ANALYSIS="${RUN_STRICT_ANALYSIS:-1}"
RUN_DIVERGENCE="${RUN_DIVERGENCE:-1}"
RUN_AUC="${RUN_AUC:-1}"
RUN_TOKEN_ANALYSIS="${RUN_TOKEN_ANALYSIS:-1}"
RUN_DYNAMIC_EVAL="${RUN_DYNAMIC_EVAL:-1}"
FORCE="${FORCE:-0}"

MIN_RECALL_DELTA="${MIN_RECALL_DELTA:-0.05}"
MIN_BLEU4_DELTA="${MIN_BLEU4_DELTA:-0.005}"
BEST_SOFT_GAMMA="${BEST_SOFT_GAMMA:-0.75}"
MAX_PER_CASE="${MAX_PER_CASE:-30}"

DYNAMIC_GAMMA="${DYNAMIC_GAMMA:-1.0}"
DYNAMIC_TEMPERATURE="${DYNAMIC_TEMPERATURE:-0.05}"
DYNAMIC_MARGIN_WEIGHT="${DYNAMIC_MARGIN_WEIGHT:-1.0}"
DYNAMIC_RATIO_WEIGHT="${DYNAMIC_RATIO_WEIGHT:-0.25}"
DYNAMIC_CONSENSUS_WEIGHT="${DYNAMIC_CONSENSUS_WEIGHT:-0.5}"
DYNAMIC_BIAS="${DYNAMIC_BIAS:-0.0}"

BASE_RESULT_PATH="${BASE_RESULT_PATH:-./results/${DATASET}/soft_routing_smoke_n${NUM_SAMPLES}_seed${SEED}_tau${ADHH_THRESHOLD}_T${SOFT_TEMPERATURE}}"
LOG_DIR="${LOG_DIR:-./logs/soft_routing}"
mkdir -p "${LOG_DIR}"

expected_generation_outputs=(
    "${BASE_RESULT_PATH}/greedy/captions_eval_results.json"
    "${BASE_RESULT_PATH}/hard_tau${ADHH_THRESHOLD}/captions_eval_results.json"
    "${BASE_RESULT_PATH}/soft_gamma0.25/captions_eval_results.json"
    "${BASE_RESULT_PATH}/soft_gamma0.50/captions_eval_results.json"
    "${BASE_RESULT_PATH}/soft_gamma0.75/captions_eval_results.json"
)

all_exist() {
    for path in "$@"; do
        if [ ! -f "${path}" ]; then
            return 1
        fi
    done
    return 0
}

run_generation() {
    if [ "${FORCE}" != "1" ] && all_exist "${expected_generation_outputs[@]}"; then
        echo "[skip] generation outputs already exist under ${BASE_RESULT_PATH}"
        return
    fi

    echo "[run] greedy/hard/soft generation and CHAIR evaluation"
    if [ "${PARALLEL_GENERATION}" = "1" ]; then
        GPU_ID="${GPU0}" NUM_SAMPLES="${NUM_SAMPLES}" SEED="${SEED}" MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
        MODEL_PATH="${MODEL_PATH}" DATASET="${DATASET}" ANNOTATION_DIR="${ANNOTATION_DIR}" \
        IMAGE_FOLDER="${IMAGE_FOLDER}" CAPTION_FILE_PATH="${CAPTION_FILE_PATH}" \
        ADHH_THRESHOLD="${ADHH_THRESHOLD}" SOFT_TEMPERATURE="${SOFT_TEMPERATURE}" \
        BASE_RESULT_PATH="${BASE_RESULT_PATH}" RUNS="greedy hard soft025" \
        bash bash_scripts/soft_routing/smoke_100.sh \
        2>&1 | tee "${LOG_DIR}/pipeline_gpu${GPU0}_greedy_hard_soft025.log" &
        pid0=$!

        GPU_ID="${GPU1}" NUM_SAMPLES="${NUM_SAMPLES}" SEED="${SEED}" MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
        MODEL_PATH="${MODEL_PATH}" DATASET="${DATASET}" ANNOTATION_DIR="${ANNOTATION_DIR}" \
        IMAGE_FOLDER="${IMAGE_FOLDER}" CAPTION_FILE_PATH="${CAPTION_FILE_PATH}" \
        ADHH_THRESHOLD="${ADHH_THRESHOLD}" SOFT_TEMPERATURE="${SOFT_TEMPERATURE}" \
        BASE_RESULT_PATH="${BASE_RESULT_PATH}" RUNS="soft050 soft075" \
        bash bash_scripts/soft_routing/smoke_100.sh \
        2>&1 | tee "${LOG_DIR}/pipeline_gpu${GPU1}_soft050_soft075.log" &
        pid1=$!

        wait "${pid0}"
        wait "${pid1}"
    else
        GPU_ID="${GPU0}" NUM_SAMPLES="${NUM_SAMPLES}" SEED="${SEED}" MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
        MODEL_PATH="${MODEL_PATH}" DATASET="${DATASET}" ANNOTATION_DIR="${ANNOTATION_DIR}" \
        IMAGE_FOLDER="${IMAGE_FOLDER}" CAPTION_FILE_PATH="${CAPTION_FILE_PATH}" \
        ADHH_THRESHOLD="${ADHH_THRESHOLD}" SOFT_TEMPERATURE="${SOFT_TEMPERATURE}" \
        BASE_RESULT_PATH="${BASE_RESULT_PATH}" RUNS="greedy hard soft025 soft050 soft075" \
        bash bash_scripts/soft_routing/smoke_100.sh \
        2>&1 | tee "${LOG_DIR}/pipeline_gpu${GPU0}_all_generation.log"
    fi
}

run_strict_analysis() {
    echo "[run] strict sample-level hard-vs-soft case mining"
    for gamma in 0.25 0.50 0.75; do
        python -m eval_scripts.soft_routing.analyze_sample_wins \
            --greedy-results "${BASE_RESULT_PATH}/greedy/captions_eval_results.json" \
            --hard-results "${BASE_RESULT_PATH}/hard_tau${ADHH_THRESHOLD}/captions_eval_results.json" \
            --soft-results "${BASE_RESULT_PATH}/soft_gamma${gamma}/captions_eval_results.json" \
            --annotation-file "${CAPTION_FILE_PATH}" \
            --output-dir "${BASE_RESULT_PATH}/analysis_hard_vs_soft${gamma}_strict" \
            --min-recall-delta "${MIN_RECALL_DELTA}" \
            --min-bleu4-delta "${MIN_BLEU4_DELTA}"
    done
}

run_divergence() {
    local analysis_dir="${BASE_RESULT_PATH}/analysis_hard_vs_soft${BEST_SOFT_GAMMA}_strict"
    local diag_dir="${BASE_RESULT_PATH}/diagnostics_hard_vs_soft${BEST_SOFT_GAMMA}"
    mkdir -p "${diag_dir}"

    echo "[run] first-divergence diagnostics for ${analysis_dir}"
    CUDA_VISIBLE_DEVICES="${DYNAMIC_GPU}" python -m eval_scripts.soft_routing.run_first_divergence_diagnostics \
        --case-files \
            "${analysis_dir}/hard_win.jsonl" \
            "${analysis_dir}/soft_win.jsonl" \
            "${analysis_dir}/hard_tradeoff.jsonl" \
        --output-jsonl "${diag_dir}/first_divergence.jsonl" \
        --model-path "${MODEL_PATH}" \
        --image-folder "${IMAGE_FOLDER}" \
        --caption-file-path "${CAPTION_FILE_PATH}" \
        --max-new-tokens "${MAX_NEW_TOKENS}" \
        --max-per-case "${MAX_PER_CASE}" \
        --adhh-threshold "${ADHH_THRESHOLD}" \
        --soft-gamma "${BEST_SOFT_GAMMA}" \
        --soft-temperature "${SOFT_TEMPERATURE}" \
        2>&1 | tee "${LOG_DIR}/pipeline_divergence_soft${BEST_SOFT_GAMMA}.log"
}

run_auc() {
    local diag_dir="${BASE_RESULT_PATH}/diagnostics_hard_vs_soft${BEST_SOFT_GAMMA}"
    echo "[run] feature AUC for hard-vs-weak decision points"
    python -m eval_scripts.soft_routing.analyze_feature_auc \
        --diagnostics-jsonl "${diag_dir}/first_divergence.jsonl" \
        --output-dir "${diag_dir}/feature_auc"
}

run_token_analysis() {
    local diag_dir="${BASE_RESULT_PATH}/diagnostics_hard_vs_soft${BEST_SOFT_GAMMA}"
    echo "[run] token-level first-divergence analysis"
    python -m eval_scripts.soft_routing.analyze_token_diagnostics \
        --diagnostics-jsonl "${diag_dir}/first_divergence.jsonl" \
        --output-dir "${diag_dir}/token_analysis"
}

run_dynamic_eval() {
    local dyn_tag="dynamic_v1_g${DYNAMIC_GAMMA}_m${DYNAMIC_MARGIN_WEIGHT}_r${DYNAMIC_RATIO_WEIGHT}_c${DYNAMIC_CONSENSUS_WEIGHT}_b${DYNAMIC_BIAS}"
    local result_path="${BASE_RESULT_PATH}/${dyn_tag}"
    local answers_file="${result_path}/captions.jsonl"
    mkdir -p "${result_path}"

    if [ "${FORCE}" != "1" ] && [ -f "${answers_file/.jsonl/_eval_results.json}" ]; then
        echo "[skip] dynamic eval already exists at ${result_path}"
        return
    fi

    echo "[run] training-free dynamic suppression evaluation"
    CUDA_VISIBLE_DEVICES="${DYNAMIC_GPU}" python -m eval_scripts.eval_caption_adhh \
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
        --dynamic_deactivate \
        --adhh_threshold "${ADHH_THRESHOLD}" \
        --dynamic_gamma "${DYNAMIC_GAMMA}" \
        --dynamic_temperature "${DYNAMIC_TEMPERATURE}" \
        --dynamic_margin_weight "${DYNAMIC_MARGIN_WEIGHT}" \
        --dynamic_ratio_weight "${DYNAMIC_RATIO_WEIGHT}" \
        --dynamic_consensus_weight "${DYNAMIC_CONSENSUS_WEIGHT}" \
        --dynamic_bias "${DYNAMIC_BIAS}" \
        2>&1 | tee "${LOG_DIR}/pipeline_dynamic_eval.log"

    python eval_scripts/eval_utils/eval_chair.py \
        --annotation-dir "${ANNOTATION_DIR}" \
        --answers-file "${answers_file}" \
        --caption_file "$(basename "${CAPTION_FILE_PATH}")"
}

print_summary() {
    echo "[summary] available overall metrics"
    python - <<PY
import glob, json
base = "${BASE_RESULT_PATH}"
for path in sorted(glob.glob(base + "/*/captions_eval_results.json")):
    print("\\n" + path)
    print(json.load(open(path))["overall_metrics"])
for path in sorted(glob.glob(base + "/analysis_*_strict/summary.json")):
    print("\\n" + path)
    print(json.load(open(path))["winner_counts"])
for path in sorted(glob.glob(base + "/diagnostics_*/feature_auc/feature_auc.csv")):
    print("\\nfeature_auc:", path)
    with open(path) as f:
        for idx, line in zip(range(6), f):
            print(line.rstrip())
PY
}

if [ "${RUN_GENERATION}" = "1" ]; then
    run_generation
fi
if [ "${RUN_STRICT_ANALYSIS}" = "1" ]; then
    run_strict_analysis
fi
if [ "${RUN_DIVERGENCE}" = "1" ]; then
    run_divergence
fi
if [ "${RUN_AUC}" = "1" ]; then
    run_auc
fi
if [ "${RUN_TOKEN_ANALYSIS}" = "1" ]; then
    run_token_analysis
fi
if [ "${RUN_DYNAMIC_EVAL}" = "1" ]; then
    run_dynamic_eval
fi
print_summary

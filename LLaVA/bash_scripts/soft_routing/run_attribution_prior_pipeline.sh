#!/bin/bash

set -euo pipefail

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-True}"

DATASET="${DATASET:-coco}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
ANNOTATION_DIR="${ANNOTATION_DIR:-/home/kms/data/images/mscoco/annotations}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
CAPTION_FILE_PATH="${CAPTION_FILE_PATH:-${ANNOTATION_DIR}/captions_val2014.json}"

NUM_SAMPLES="${NUM_SAMPLES:-500}"
CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-200}"
SEED="${SEED:-42}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
ADHH_THRESHOLD="${ADHH_THRESHOLD:-0.4}"
SOFT_TEMPERATURE="${SOFT_TEMPERATURE:-0.05}"
BASE_SOFT_GAMMA="${BASE_SOFT_GAMMA:-0.75}"

GPU_ID="${GPU_ID:-0}"
FEATURE_GPU="${FEATURE_GPU:-${GPU_ID}}"
POLICY_GPU="${POLICY_GPU:-${GPU_ID}}"
RUN_BASELINES="${RUN_BASELINES:-1}"
RUN_FEATURE_VALIDATION="${RUN_FEATURE_VALIDATION:-1}"
RUN_POLICY_EVAL="${RUN_POLICY_EVAL:-1}"
FORCE="${FORCE:-0}"

TOP_K="${TOP_K:-20}"
Q_LOW="${Q_LOW:-60}"
Q_HIGH="${Q_HIGH:-90}"
ATTR_GAMMA="${ATTR_GAMMA:-1.0}"
NO_PRIOR_MODES="${NO_PRIOR_MODES:-linear}"
PRIOR_MODES="${PRIOR_MODES:-linear sqrt quadratic budget}"
HEAD_PRIOR_MODE="${HEAD_PRIOR_MODE:-auto}"
ATTENTION_HEAD_PATH="${ATTENTION_HEAD_PATH:-results/coco/llava_3000/identify_attention_head/attribution_result.json}"

TEST_NUM_CHUNKS="${TEST_NUM_CHUNKS:-1}"
TEST_CHUNK_IDX="${TEST_CHUNK_IDX:-0}"

BASE_RESULT_PATH="${BASE_RESULT_PATH:-./results/${DATASET}/soft_routing_smoke_n${NUM_SAMPLES}_seed${SEED}_tau${ADHH_THRESHOLD}_T${SOFT_TEMPERATURE}}"
FEATURE_DIR="${FEATURE_DIR:-${BASE_RESULT_PATH}/object_feature_validation_calib${CALIBRATION_SAMPLES}_q${Q_LOW}_${Q_HIGH}}"
LOG_DIR="${LOG_DIR:-./logs/soft_routing}"
mkdir -p "${LOG_DIR}"

RESOLVED_ATTENTION_HEAD_PATH="${ATTENTION_HEAD_PATH}"
if [ -n "${RESOLVED_ATTENTION_HEAD_PATH}" ] && [ ! -f "${RESOLVED_ATTENTION_HEAD_PATH}" ]; then
    echo "[warn] attention head file not found: ${RESOLVED_ATTENTION_HEAD_PATH}"
    echo "[warn] falling back to built-in LLaVA-1.5 AD-HH head list with rank priors"
    RESOLVED_ATTENTION_HEAD_PATH=""
fi

run_baselines() {
    local greedy_eval="${BASE_RESULT_PATH}/greedy/captions_eval_results.json"
    local hard_eval="${BASE_RESULT_PATH}/hard_tau${ADHH_THRESHOLD}/captions_eval_results.json"
    local soft_eval="${BASE_RESULT_PATH}/soft_gamma${BASE_SOFT_GAMMA}/captions_eval_results.json"
    if [ "${FORCE}" != "1" ] && [ -f "${greedy_eval}" ] && [ -f "${hard_eval}" ] && [ -f "${soft_eval}" ]; then
        echo "[skip] baseline outputs already exist under ${BASE_RESULT_PATH}"
        return
    fi

    echo "[run] baseline greedy / hard AD-HH / fixed soft gamma=${BASE_SOFT_GAMMA}"
    GPU_ID="${GPU_ID}" NUM_SAMPLES="${NUM_SAMPLES}" SEED="${SEED}" MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
    MODEL_PATH="${MODEL_PATH}" DATASET="${DATASET}" ANNOTATION_DIR="${ANNOTATION_DIR}" \
    IMAGE_FOLDER="${IMAGE_FOLDER}" CAPTION_FILE_PATH="${CAPTION_FILE_PATH}" \
    ADHH_THRESHOLD="${ADHH_THRESHOLD}" SOFT_TEMPERATURE="${SOFT_TEMPERATURE}" \
    BASE_RESULT_PATH="${BASE_RESULT_PATH}" RUNS="greedy hard soft075" \
    bash bash_scripts/soft_routing/smoke_100.sh \
    2>&1 | tee "${LOG_DIR}/attribution_prior_baselines.log"
}

run_feature_validation() {
    local greedy_eval="${BASE_RESULT_PATH}/greedy/captions_eval_results.json"
    local summary="${FEATURE_DIR}/feature_validation_summary.json"
    if [ "${FORCE}" != "1" ] && [ -f "${summary}" ]; then
        echo "[skip] feature validation exists at ${FEATURE_DIR}"
        return
    fi
    if [ ! -f "${greedy_eval}" ]; then
        echo "[error] missing greedy eval results: ${greedy_eval}" >&2
        exit 1
    fi

    echo "[run] logging-only object-step feature validation"
    CUDA_VISIBLE_DEVICES="${FEATURE_GPU}" python -m eval_scripts.soft_routing.log_object_step_features \
        --eval-results "${greedy_eval}" \
        --image-folder "${IMAGE_FOLDER}" \
        --model-path "${MODEL_PATH}" \
        --conv-mode vicuna_v1 \
        --output-dir "${FEATURE_DIR}" \
        --attention-head-path "${RESOLVED_ATTENTION_HEAD_PATH}" \
        --top-k "${TOP_K}" \
        --head-prior-mode "${HEAD_PRIOR_MODE}" \
        --max-samples "${CALIBRATION_SAMPLES}" \
        --adhh-threshold "${ADHH_THRESHOLD}" \
        --soft-gamma "${BASE_SOFT_GAMMA}" \
        --soft-temperature "${SOFT_TEMPERATURE}" \
        --q-low "${Q_LOW}" \
        --q-high "${Q_HIGH}" \
        2>&1 | tee "${LOG_DIR}/attribution_prior_feature_validation.log"
}

eval_attr_policy() {
    local mode="$1"
    local prior_mode="$2"
    local tag="$3"
    local result_path="${BASE_RESULT_PATH}/${tag}"
    local answers_file="${result_path}/captions.jsonl"
    local eval_file="${result_path}/captions_eval_results.json"
    mkdir -p "${result_path}"

    if [ "${FORCE}" != "1" ] && [ -f "${eval_file}" ]; then
        echo "[skip] ${tag} already exists"
        return
    fi

    echo "[run] ${tag}"
    CUDA_VISIBLE_DEVICES="${POLICY_GPU}" python -m eval_scripts.eval_caption_adhh \
        --model-path "${MODEL_PATH}" \
        --image-folder "${IMAGE_FOLDER}" \
        --caption_file_path "${CAPTION_FILE_PATH}" \
        --answers-file "${answers_file}" \
        --dataset "${DATASET}" \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --num_samples "${NUM_SAMPLES}" \
        --num-chunks "${TEST_NUM_CHUNKS}" \
        --chunk-idx "${TEST_CHUNK_IDX}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --seed "${SEED}" \
        --attribution_soft_deactivate \
        --attention_head_path "${RESOLVED_ATTENTION_HEAD_PATH}" \
        --top_k "${TOP_K}" \
        --head_prior_mode "${prior_mode}" \
        --head_thresholds_path "${FEATURE_DIR}/head_thresholds.json" \
        --adhh_threshold "${ADHH_THRESHOLD}" \
        --attribution_soft_gamma "${ATTR_GAMMA}" \
        --attribution_soft_mode "${mode}" \
        --attribution_tau_low "${ADHH_THRESHOLD}" \
        --attribution_tau_high 0.9 \
        2>&1 | tee "${LOG_DIR}/attribution_prior_${tag}.log"

    python eval_scripts/eval_utils/eval_chair.py \
        --annotation-dir "${ANNOTATION_DIR}" \
        --answers-file "${answers_file}" \
        --caption_file "$(basename "${CAPTION_FILE_PATH}")"
}

run_policy_eval() {
    if [ ! -f "${FEATURE_DIR}/head_thresholds.json" ]; then
        echo "[error] missing head thresholds: ${FEATURE_DIR}/head_thresholds.json" >&2
        exit 1
    fi

    for mode in ${NO_PRIOR_MODES}; do
        eval_attr_policy "${mode}" "uniform" "attr_no_prior_${mode}_g${ATTR_GAMMA}_q${Q_LOW}_${Q_HIGH}"
    done
    for mode in ${PRIOR_MODES}; do
        eval_attr_policy "${mode}" "${HEAD_PRIOR_MODE}" "attr_prior_${mode}_g${ATTR_GAMMA}_q${Q_LOW}_${Q_HIGH}"
    done
}

print_summary() {
    echo "[summary] object-step feature validation"
    python - <<PY
import json, os
summary_path = "${FEATURE_DIR}/feature_validation_summary.json"
if os.path.exists(summary_path):
    data = json.load(open(summary_path))
    print("prior_source:", data.get("prior_source"))
    print("num_object_steps:", data.get("num_object_steps"), "hall:", data.get("num_hallucinated"), "grounded:", data.get("num_grounded"))
    for row in data.get("feature_summary", []):
        if row["feature"] in {"max_i_text", "mean_i_text", "max_prior_i_text", "mean_prior_i_text", "sum_prior_percentile_excess", "weighted_trigger_count"}:
            print(row)
else:
    print("missing", summary_path)
PY

    echo "[summary] available overall metrics"
    python - <<PY
import csv, glob, json, os
base = "${BASE_RESULT_PATH}"
rows = []
for path in sorted(glob.glob(base + "/*/captions_eval_results.json")):
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
out_path = os.path.join(base, "attribution_prior_metrics_summary.csv")
if rows:
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("\\nwrote", out_path)
PY
}

if [ "${RUN_BASELINES}" = "1" ]; then
    run_baselines
fi
if [ "${RUN_FEATURE_VALIDATION}" = "1" ]; then
    run_feature_validation
fi
if [ "${RUN_POLICY_EVAL}" = "1" ]; then
    run_policy_eval
fi
print_summary

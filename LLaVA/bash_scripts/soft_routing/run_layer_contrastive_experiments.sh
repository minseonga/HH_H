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
FORCE="${FORCE:-0}"

LAYER_CONTRASTIVE_LAYERS="${LAYER_CONTRASTIVE_LAYERS:-16}"
LAYER_CONTRASTIVE_ALPHA="${LAYER_CONTRASTIVE_ALPHA:-0.5}"
LAYER_CONTRASTIVE_GATE_FEATURE="${LAYER_CONTRASTIVE_GATE_FEATURE:-js_divergence}"
LAYER_CONTRASTIVE_GATE_POWER="${LAYER_CONTRASTIVE_GATE_POWER:-1.0}"
LAYER_CONTRASTIVE_MARGIN_TEMPERATURE="${LAYER_CONTRASTIVE_MARGIN_TEMPERATURE:-1.0}"
LAYER_CONTRASTIVE_PHASE="${LAYER_CONTRASTIVE_PHASE:-decode}"
RECORD_LAYER_CONTRASTIVE_DIAGNOSTICS="${RECORD_LAYER_CONTRASTIVE_DIAGNOSTICS:-0}"
LAYER_CONTRASTIVE_DIAGNOSTICS_MAX_RECORDS="${LAYER_CONTRASTIVE_DIAGNOSTICS_MAX_RECORDS:-0}"

BASE_RESULT_PATH="${BASE_RESULT_PATH:-./results/${DATASET}/soft_routing_smoke_n${NUM_SAMPLES}_seed${SEED}_tau${ADHH_THRESHOLD}_T${SOFT_TEMPERATURE}}"
TAG="${TAG:-layer_contrastive_l${LAYER_CONTRASTIVE_LAYERS}_${LAYER_CONTRASTIVE_GATE_FEATURE}_a${LAYER_CONTRASTIVE_ALPHA}_${LAYER_CONTRASTIVE_PHASE}}"
OUTPUT_DIR="${OUTPUT_DIR:-${BASE_RESULT_PATH}/layer_contrastive}"
LOG_DIR="${LOG_DIR:-./logs/soft_routing}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

echo "[info] output dir: ${OUTPUT_DIR}"
echo "[info] tag: ${TAG}"
echo "[info] layers: ${LAYER_CONTRASTIVE_LAYERS}"
echo "[info] alpha: ${LAYER_CONTRASTIVE_ALPHA}"
echo "[info] gate feature: ${LAYER_CONTRASTIVE_GATE_FEATURE}"
echo "[info] gate power: ${LAYER_CONTRASTIVE_GATE_POWER}"
echo "[info] margin temperature: ${LAYER_CONTRASTIVE_MARGIN_TEMPERATURE}"
echo "[info] phase: ${LAYER_CONTRASTIVE_PHASE}"
echo "[info] record diagnostics: ${RECORD_LAYER_CONTRASTIVE_DIAGNOSTICS}"

diagnostic_args=()
if [ "${RECORD_LAYER_CONTRASTIVE_DIAGNOSTICS}" = "1" ]; then
    diagnostic_args=(
        --record_layer_contrastive_diagnostics
        --layer_contrastive_diagnostics_max_records "${LAYER_CONTRASTIVE_DIAGNOSTICS_MAX_RECORDS}"
    )
fi

result_dir="${OUTPUT_DIR}/${TAG}"
answers_file="${result_dir}/captions.jsonl"
eval_file="${result_dir}/captions_eval_results.json"
mkdir -p "${result_dir}"

if [ "${FORCE}" != "1" ] && [ -f "${eval_file}" ]; then
    echo "[skip] ${TAG}: ${eval_file}"
else
    echo "[run] ${TAG}"
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
        --layer_contrastive_deactivate \
        --layer_contrastive_layers "${LAYER_CONTRASTIVE_LAYERS}" \
        --layer_contrastive_alpha "${LAYER_CONTRASTIVE_ALPHA}" \
        --layer_contrastive_gate_feature "${LAYER_CONTRASTIVE_GATE_FEATURE}" \
        --layer_contrastive_gate_power "${LAYER_CONTRASTIVE_GATE_POWER}" \
        --layer_contrastive_margin_temperature "${LAYER_CONTRASTIVE_MARGIN_TEMPERATURE}" \
        --layer_contrastive_phase "${LAYER_CONTRASTIVE_PHASE}" \
        "${diagnostic_args[@]}" \
        2>&1 | tee "${LOG_DIR}/${TAG}.log"

    python eval_scripts/eval_utils/eval_chair.py \
        --annotation-dir "${ANNOTATION_DIR}" \
        --answers-file "${answers_file}" \
        --caption_file "$(basename "${CAPTION_FILE_PATH}")"
fi

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

out = "${OUTPUT_DIR}/layer_contrastive_summary.csv"
if rows:
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
print(out)
PY

echo "[summary] layer contrastive metrics"
if [ -f "${OUTPUT_DIR}/layer_contrastive_summary.csv" ]; then
    column -s, -t "${OUTPUT_DIR}/layer_contrastive_summary.csv"
fi

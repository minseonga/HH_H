#!/bin/bash

set -euo pipefail

DATASET="${DATASET:-coco}"
NUM_SAMPLES="${NUM_SAMPLES:-500}"
SEED="${SEED:-42}"
ADHH_THRESHOLD="${ADHH_THRESHOLD:-0.4}"
SOFT_TEMPERATURE="${SOFT_TEMPERATURE:-0.05}"

BASE_RESULT_PATH="${BASE_RESULT_PATH:-./results/${DATASET}/soft_routing_smoke_n${NUM_SAMPLES}_seed${SEED}_tau${ADHH_THRESHOLD}_T${SOFT_TEMPERATURE}}"
OUTPUT_DIR="${OUTPUT_DIR:-${BASE_RESULT_PATH}/caption_repetition_analysis}"

python -m eval_scripts.soft_routing.analyze_caption_repetition \
    --base-dir "${BASE_RESULT_PATH}" \
    --output-dir "${OUTPUT_DIR}"

echo "[summary] repetition metrics"
column -s, -t "${OUTPUT_DIR}/caption_repetition_summary.csv"

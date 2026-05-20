#!/bin/bash

set -euo pipefail

DATASET="${DATASET:-coco}"
NUM_SAMPLES="${NUM_SAMPLES:-500}"
SEED="${SEED:-42}"
ADHH_THRESHOLD="${ADHH_THRESHOLD:-0.4}"
SOFT_TEMPERATURE="${SOFT_TEMPERATURE:-0.05}"
BASE_RESULT_PATH="${BASE_RESULT_PATH:-./results/${DATASET}/soft_routing_smoke_n${NUM_SAMPLES}_seed${SEED}_tau${ADHH_THRESHOLD}_T${SOFT_TEMPERATURE}}"

BASE_EVAL="${BASE_EVAL:-${BASE_RESULT_PATH}/greedy/captions_eval_results.json}"
TARGET_EVAL="${TARGET_EVAL:?set TARGET_EVAL to captions_eval_results.json or captions.jsonl}"
BASE_NAME="${BASE_NAME:-greedy}"
TARGET_NAME="${TARGET_NAME:-$(basename "$(dirname "${TARGET_EVAL}")")}"
OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "${TARGET_EVAL}")/caption_pairwise_vs_${BASE_NAME}}"
MAX_EXAMPLES="${MAX_EXAMPLES:-50}"

if [ ! -f "${BASE_EVAL}" ]; then
    echo "[error] missing BASE_EVAL: ${BASE_EVAL}" >&2
    exit 1
fi
if [ ! -f "${TARGET_EVAL}" ]; then
    echo "[error] missing TARGET_EVAL: ${TARGET_EVAL}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "[info] base eval: ${BASE_EVAL}"
echo "[info] target eval: ${TARGET_EVAL}"
echo "[info] output dir: ${OUTPUT_DIR}"

python -m eval_scripts.soft_routing.compare_caption_pairs \
    --base "${BASE_EVAL}" \
    --target "${TARGET_EVAL}" \
    --base-name "${BASE_NAME}" \
    --target-name "${TARGET_NAME}" \
    --output-dir "${OUTPUT_DIR}" \
    --max-examples "${MAX_EXAMPLES}"

echo "[summary] pairwise"
column -s, -t "${OUTPUT_DIR}/caption_pairwise_summary.csv"

echo "[summary] examples: ${OUTPUT_DIR}/caption_pairwise_examples.csv"

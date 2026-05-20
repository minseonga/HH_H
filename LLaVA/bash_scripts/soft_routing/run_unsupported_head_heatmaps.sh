#!/bin/bash

set -euo pipefail

DATASET="${DATASET:-coco}"
NUM_SAMPLES="${NUM_SAMPLES:-500}"
SEED="${SEED:-42}"
ADHH_THRESHOLD="${ADHH_THRESHOLD:-0.4}"
SOFT_TEMPERATURE="${SOFT_TEMPERATURE:-0.05}"
CANDIDATE_POLICY="${CANDIDATE_POLICY:-anchor_mixed}"
CANDIDATE_MAX_HEADS="${CANDIDATE_MAX_HEADS:-48}"
MAX_PER_LABEL="${MAX_PER_LABEL:-10}"
POSITIVE_UTILITY_THRESHOLD="${POSITIVE_UTILITY_THRESHOLD:-0.02}"
LAYERS="${LAYERS:-32}"
HEADS="${HEADS:-32}"
FEATURES="${FEATURES:-unsupported_text_value_norm,text_value_norm,supported_text_value_norm,text_img_value_cosine,visual_value_ratio,text_mass}"

BASE_RESULT_PATH="${BASE_RESULT_PATH:-./results/${DATASET}/soft_routing_smoke_n${NUM_SAMPLES}_seed${SEED}_tau${ADHH_THRESHOLD}_T${SOFT_TEMPERATURE}}"
TEACHER_DIR="${TEACHER_DIR:-${BASE_RESULT_PATH}/online_causal_head_teacher_${CANDIDATE_POLICY}_h${CANDIDATE_MAX_HEADS}_max${MAX_PER_LABEL}}"
INPUT_JSONL="${INPUT_JSONL:-${TEACHER_DIR}/online_causal_head_teacher.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${TEACHER_DIR}/unsupported_head_heatmaps}"

if [ ! -f "${INPUT_JSONL}" ]; then
    echo "[error] missing input jsonl: ${INPUT_JSONL}" >&2
    echo "[hint] set INPUT_JSONL to online_causal_head_teacher.jsonl or all_head_feature_diagnostics.jsonl" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "[info] input jsonl: ${INPUT_JSONL}"
echo "[info] output dir: ${OUTPUT_DIR}"
echo "[info] features: ${FEATURES}"

python -m eval_scripts.soft_routing.visualize_unsupported_head_heatmaps \
    --input-jsonl "${INPUT_JSONL}" \
    --output-dir "${OUTPUT_DIR}" \
    --features "${FEATURES}" \
    --layers "${LAYERS}" \
    --heads "${HEADS}" \
    --utility-threshold "${POSITIVE_UTILITY_THRESHOLD}"

echo "[summary] top unsupported heads"
if [ -f "${OUTPUT_DIR}/unsupported_head_summary.csv" ]; then
    python - <<PY
import csv
path = "${OUTPUT_DIR}/unsupported_head_summary.csv"
rows = list(csv.DictReader(open(path)))
rows = [r for r in rows if r.get("group") == "all"]
rows.sort(key=lambda r: float(r.get("mean_unsupported_text_value_norm") or 0), reverse=True)
for row in rows[:30]:
    print(
        row["head_key"],
        "n", row["n"],
        "unsupported", row.get("mean_unsupported_text_value_norm"),
        "utility", row.get("mean_suppression_utility"),
        "suppress_rate", row.get("should_suppress_rate"),
    )
PY
fi

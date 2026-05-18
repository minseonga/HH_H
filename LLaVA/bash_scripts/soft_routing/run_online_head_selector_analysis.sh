#!/bin/bash

set -euo pipefail

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-True}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-user}}"

DATASET="${DATASET:-coco}"
NUM_SAMPLES="${NUM_SAMPLES:-500}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SEED="${SEED:-42}"
ADHH_THRESHOLD="${ADHH_THRESHOLD:-0.4}"
SOFT_TEMPERATURE="${SOFT_TEMPERATURE:-0.05}"
TOP_POOL_K="${TOP_POOL_K:-100}"

BASE_RESULT_PATH="${BASE_RESULT_PATH:-./results/${DATASET}/soft_routing_smoke_n${NUM_SAMPLES}_seed${SEED}_tau${ADHH_THRESHOLD}_T${SOFT_TEMPERATURE}}"
ATTRIBUTION_RESULT="${ATTRIBUTION_RESULT:-${BASE_RESULT_PATH}/identify_attention_head_val_calib200_full1024/attribution_result.json}"
if [ ! -f "${ATTRIBUTION_RESULT}" ]; then
    ATTRIBUTION_RESULT="${ATTRIBUTION_RESULT_FALLBACK:-./results/coco/llava_3000/identify_attention_head/attribution_result.json}"
fi

POOL_DIR="${POOL_DIR:-${BASE_RESULT_PATH}/contrastive_candidate_pools}"
TEACHER_HEAD_PATH="${TEACHER_HEAD_PATH:-${POOL_DIR}/contrastive_top${TOP_POOL_K}.json}"
BEHAVIORAL_RECORDS="${BEHAVIORAL_RECORDS:-${BASE_RESULT_PATH}/behavioral_head_overlap_prefill_n500_minlayer13/all_head_records.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${BASE_RESULT_PATH}/online_head_selector_analysis_top${TOP_POOL_K}}"

mkdir -p "${POOL_DIR}" "${OUTPUT_DIR}" "${MPLCONFIGDIR}"

if [ ! -f "${ATTRIBUTION_RESULT}" ]; then
    echo "[error] missing attribution result: ${ATTRIBUTION_RESULT}" >&2
    exit 1
fi
if [ ! -f "${TEACHER_HEAD_PATH}" ]; then
    python -m eval_scripts.soft_routing.build_contrastive_candidate_pool \
        --attribution-result "${ATTRIBUTION_RESULT}" \
        --output-path "${TEACHER_HEAD_PATH}" \
        --top-k "${TOP_POOL_K}"
fi
if [ ! -f "${BEHAVIORAL_RECORDS}" ]; then
    echo "[error] missing behavioral records: ${BEHAVIORAL_RECORDS}" >&2
    echo "[hint] run bash_scripts/soft_routing/run_behavioral_head_overlap.sh first" >&2
    exit 1
fi

echo "[info] teacher heads: ${TEACHER_HEAD_PATH}"
echo "[info] behavioral records: ${BEHAVIORAL_RECORDS}"
echo "[info] output dir: ${OUTPUT_DIR}"

python -m eval_scripts.soft_routing.analyze_online_head_selector \
    --records-jsonl "${BEHAVIORAL_RECORDS}" \
    --teacher-head-path "${TEACHER_HEAD_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --teacher-top-ks "20,40,60,100" \
    --eval-top-ks "20,40,60,100" \
    --max-samples "${MAX_SAMPLES}" \
    --trigger-text-tau "${ADHH_THRESHOLD}"

echo "[summary] best feature AUCs for teacher top-100 under text-triggered heads"
python - <<PY
import csv
path = "${OUTPUT_DIR}/feature_auc_teacher_membership.csv"
rows = list(csv.DictReader(open(path)))
rows = [
    r for r in rows
    if r["teacher_top_k"] == "100" and r["subset"] == "text_triggered"
]
for r in sorted(rows, key=lambda x: float(x["auroc_abs"]), reverse=True)[:20]:
    print(r["feature"], r["direction"], r["auroc_high_predicts_teacher"], r["mean_positive"], r["mean_negative"])
PY

echo "[summary] online selector recovery of teacher top-100, text-triggered, eval top-20"
python - <<PY
import csv
path = "${OUTPUT_DIR}/selector_overlap_summary.csv"
rows = list(csv.DictReader(open(path)))
rows = [
    r for r in rows
    if r["teacher_top_k"] == "100"
    and r["mode"] == "text_triggered"
    and r["eval_top_k"] == "20"
    and r["metric"] in {"overlap", "precision", "jaccard"}
]
for metric in ["overlap", "precision", "jaccard"]:
    print("\\n", metric)
    sub = [r for r in rows if r["metric"] == metric]
    for r in sorted(sub, key=lambda x: float(x["mean"]), reverse=True)[:12]:
        print(r["selector"], r["mean"], r["p50"], r["p90"])
PY

echo "[outputs]"
find "${OUTPUT_DIR}" -maxdepth 1 -type f -print | sort

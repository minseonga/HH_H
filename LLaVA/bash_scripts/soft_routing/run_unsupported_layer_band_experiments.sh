#!/bin/bash

set -euo pipefail

DATASET="${DATASET:-coco}"
NUM_SAMPLES="${NUM_SAMPLES:-50}"
SEED="${SEED:-42}"
ADHH_THRESHOLD="${ADHH_THRESHOLD:-0.4}"
SOFT_TEMPERATURE="${SOFT_TEMPERATURE:-0.05}"
GPU_ID="${GPU_ID:-0}"
FORCE="${FORCE:-0}"
RUN_METHODS="${RUN_METHODS:-online_unsupported_layer_top1_continuous online_unsupported_layer_top2_continuous}"

UNSUPPORTED_COMPONENT_GAMMA="${UNSUPPORTED_COMPONENT_GAMMA:-1.0}"
UNSUPPORTED_COMPONENT_RISK_FEATURE="${UNSUPPORTED_COMPONENT_RISK_FEATURE:-unsupported_norm}"
UNSUPPORTED_COMPONENT_SCORE_NORM="${UNSUPPORTED_COMPONENT_SCORE_NORM:-candidate_minmax}"
UNSUPPORTED_COMPONENT_SCORE_LOW="${UNSUPPORTED_COMPONENT_SCORE_LOW:-0.0}"
UNSUPPORTED_COMPONENT_SCORE_HIGH="${UNSUPPORTED_COMPONENT_SCORE_HIGH:-1.0}"
UNSUPPORTED_COMPONENT_PHASE="${UNSUPPORTED_COMPONENT_PHASE:-decode}"
UNSUPPORTED_COMPONENT_ALL_HEADS="${UNSUPPORTED_COMPONENT_ALL_HEADS:-1}"
RECORD_UNSUPPORTED_COMPONENT_DIAGNOSTICS="${RECORD_UNSUPPORTED_COMPONENT_DIAGNOSTICS:-1}"
UNSUPPORTED_COMPONENT_DIAGNOSTICS_MAX_RECORDS="${UNSUPPORTED_COMPONENT_DIAGNOSTICS_MAX_RECORDS:-10000}"

BASE_RESULT_PATH="${BASE_RESULT_PATH:-./results/${DATASET}/soft_routing_smoke_n${NUM_SAMPLES}_seed${SEED}_tau${ADHH_THRESHOLD}_T${SOFT_TEMPERATURE}}"
BAND_SPECS="${BAND_SPECS:-early=0-11 middle=12-21 late=22-31}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${BASE_RESULT_PATH}}"

echo "[info] base result path: ${BASE_RESULT_PATH}"
echo "[info] phase: ${UNSUPPORTED_COMPONENT_PHASE}"
echo "[info] band specs: ${BAND_SPECS}"
echo "[info] run methods: ${RUN_METHODS}"

for item in ${BAND_SPECS}; do
    band_name="${item%%=*}"
    band_layers="${item#*=}"
    if [ -z "${band_name}" ] || [ -z "${band_layers}" ] || [ "${band_name}" = "${band_layers}" ]; then
        echo "[error] invalid band spec: ${item}; expected name=layers, e.g. late=22-31" >&2
        exit 1
    fi
    band_tag="$(printf '%s' "${band_name}" | tr -c '[:alnum:]_-' '_')"
    output_dir="${OUTPUT_ROOT}/diag_allheads_${UNSUPPORTED_COMPONENT_SCORE_NORM}_g${UNSUPPORTED_COMPONENT_GAMMA}_${UNSUPPORTED_COMPONENT_PHASE}_${band_tag}_n${NUM_SAMPLES}"

    echo "[run] band=${band_name} layers=${band_layers} output=${output_dir}"
    FORCE="${FORCE}" \
    GPU_ID="${GPU_ID}" \
    DATASET="${DATASET}" \
    NUM_SAMPLES="${NUM_SAMPLES}" \
    SEED="${SEED}" \
    ADHH_THRESHOLD="${ADHH_THRESHOLD}" \
    SOFT_TEMPERATURE="${SOFT_TEMPERATURE}" \
    BASE_RESULT_PATH="${BASE_RESULT_PATH}" \
    OUTPUT_DIR="${output_dir}" \
    UNSUPPORTED_COMPONENT_ALL_HEADS="${UNSUPPORTED_COMPONENT_ALL_HEADS}" \
    UNSUPPORTED_COMPONENT_GAMMA="${UNSUPPORTED_COMPONENT_GAMMA}" \
    UNSUPPORTED_COMPONENT_RISK_FEATURE="${UNSUPPORTED_COMPONENT_RISK_FEATURE}" \
    UNSUPPORTED_COMPONENT_SCORE_NORM="${UNSUPPORTED_COMPONENT_SCORE_NORM}" \
    UNSUPPORTED_COMPONENT_SCORE_LOW="${UNSUPPORTED_COMPONENT_SCORE_LOW}" \
    UNSUPPORTED_COMPONENT_SCORE_HIGH="${UNSUPPORTED_COMPONENT_SCORE_HIGH}" \
    UNSUPPORTED_COMPONENT_PHASE="${UNSUPPORTED_COMPONENT_PHASE}" \
    UNSUPPORTED_COMPONENT_LAYERS="${band_layers}" \
    RECORD_UNSUPPORTED_COMPONENT_DIAGNOSTICS="${RECORD_UNSUPPORTED_COMPONENT_DIAGNOSTICS}" \
    UNSUPPORTED_COMPONENT_DIAGNOSTICS_MAX_RECORDS="${UNSUPPORTED_COMPONENT_DIAGNOSTICS_MAX_RECORDS}" \
    RUN_METHODS="${RUN_METHODS}" \
    bash bash_scripts/soft_routing/run_online_unsupported_component_experiments.sh
done

echo "[summary] layer band metrics"
for item in ${BAND_SPECS}; do
    band_name="${item%%=*}"
    band_tag="$(printf '%s' "${band_name}" | tr -c '[:alnum:]_-' '_')"
    summary="${OUTPUT_ROOT}/diag_allheads_${UNSUPPORTED_COMPONENT_SCORE_NORM}_g${UNSUPPORTED_COMPONENT_GAMMA}_${UNSUPPORTED_COMPONENT_PHASE}_${band_tag}_n${NUM_SAMPLES}/online_unsupported_component_summary.csv"
    if [ -f "${summary}" ]; then
        echo "=== ${band_name} ==="
        column -s, -t "${summary}"
    fi
done

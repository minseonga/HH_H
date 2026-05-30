#!/bin/bash

set -euo pipefail

RANKED_HEADS="${RANKED_HEADS:-../ADHH/LLaVA/results/coco/llava-v1.5-7b_base_original_qa_n3000/surrogate_hh_scores/surrogate_score_zoo/ranked_heads_global__itext_all__C_toi_HminusG.json}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/coco/method_phase_figures}"
TOP_K="${TOP_K:-150}"
TOP_KS="${TOP_KS:-20,50,100,150,200}"
LAYER_MIN="${LAYER_MIN:-0}"
LAYER_MAX="${LAYER_MAX:-31}"
HEAD_COUNT="${HEAD_COUNT:-32}"
HIGHLIGHT_LAYER_START="${HIGHLIGHT_LAYER_START:-9}"
HIGHLIGHT_LAYER_END="${HIGHLIGHT_LAYER_END:-16}"
GATE_STRENGTH="${GATE_STRENGTH:-0.7}"
GATE_BETA="${GATE_BETA:-10}"
GATE_TAU="${GATE_TAU:-0.9}"

if [ ! -f "${RANKED_HEADS}" ]; then
    echo "[error] missing ranked heads: ${RANKED_HEADS}" >&2
    exit 1
fi

python -m eval_scripts.soft_routing.build_method_phase_figures \
    --ranked-heads "${RANKED_HEADS}" \
    --output-dir "${OUTPUT_DIR}" \
    --top-k "${TOP_K}" \
    --top-ks "${TOP_KS}" \
    --layer-min "${LAYER_MIN}" \
    --layer-max "${LAYER_MAX}" \
    --head-count "${HEAD_COUNT}" \
    --highlight-layer-start "${HIGHLIGHT_LAYER_START}" \
    --highlight-layer-end "${HIGHLIGHT_LAYER_END}" \
    --gate-strength "${GATE_STRENGTH}" \
    --gate-beta "${GATE_BETA}" \
    --gate-tau "${GATE_TAU}"

echo "[summary] phase figure outputs"
if [ -f "${OUTPUT_DIR}/method_phase_figures_summary.json" ]; then
    cat "${OUTPUT_DIR}/method_phase_figures_summary.json"
fi

echo
echo "[summary] phase 1 bucket table"
if [ -f "${OUTPUT_DIR}/phase1_bucket_summary.csv" ]; then
    column -s, -t "${OUTPUT_DIR}/phase1_bucket_summary.csv"
fi

echo
echo "[summary] phase 2 gate redistribution table"
if [ -f "${OUTPUT_DIR}/phase2_gate_redistribution_summary.csv" ]; then
    column -s, -t "${OUTPUT_DIR}/phase2_gate_redistribution_summary.csv"
fi

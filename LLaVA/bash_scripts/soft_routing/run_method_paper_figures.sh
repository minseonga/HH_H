#!/bin/bash

set -euo pipefail

export MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

RANKED_HEADS="${RANKED_HEADS:-../ADHH/LLaVA/results/coco/llava-v1.5-7b_base_original_qa_n3000/surrogate_hh_scores/surrogate_score_zoo/ranked_heads_global__itext_all__C_toi_HminusG.json}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/coco/method_paper_figures_top150_l9_16}"
TOP_K="${TOP_K:-150}"
TAIL_START="${TAIL_START:-200}"
LAYER_PROFILE_MIN="${LAYER_PROFILE_MIN:-13}"
LAYER_PROFILE_MAX="${LAYER_PROFILE_MAX:-31}"
HEATMAP_LAYER_MIN="${HEATMAP_LAYER_MIN:-9}"
HEATMAP_LAYER_MAX="${HEATMAP_LAYER_MAX:-20}"
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

python -m eval_scripts.soft_routing.build_method_paper_figures \
    --ranked-heads "${RANKED_HEADS}" \
    --output-dir "${OUTPUT_DIR}" \
    --top-k "${TOP_K}" \
    --tail-start "${TAIL_START}" \
    --layer-profile-min "${LAYER_PROFILE_MIN}" \
    --layer-profile-max "${LAYER_PROFILE_MAX}" \
    --heatmap-layer-min "${HEATMAP_LAYER_MIN}" \
    --heatmap-layer-max "${HEATMAP_LAYER_MAX}" \
    --head-count "${HEAD_COUNT}" \
    --highlight-layer-start "${HIGHLIGHT_LAYER_START}" \
    --highlight-layer-end "${HIGHLIGHT_LAYER_END}" \
    --gate-strength "${GATE_STRENGTH}" \
    --gate-beta "${GATE_BETA}" \
    --gate-tau "${GATE_TAU}"

echo "[summary] paper figure outputs"
cat "${OUTPUT_DIR}/paper_figures_summary.json"

echo
echo "[summary] paper figure numbers"
column -s, -t "${OUTPUT_DIR}/paper_figure_summary.csv"

#!/bin/bash

set -euo pipefail

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-True}"

GPU_ID="${GPU_ID:-0}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
MODEL_BASE="${MODEL_BASE:-}"
CONV_MODE="${CONV_MODE:-vicuna_v1}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
IMAGE_SPLIT="${IMAGE_SPLIT:-val2014}"
EVAL_RESULTS="${EVAL_RESULTS:-}"

RANKED_HEADS="${RANKED_HEADS:-../ADHH/LLaVA/results/coco/llava-v1.5-7b_base_original_qa_n3000/surrogate_hh_scores/surrogate_score_zoo/ranked_heads_global__itext_all__C_toi_HminusG.json}"
TOP_K="${TOP_K:-100}"
MAX_MENTIONS="${MAX_MENTIONS:-80}"
MAX_PER_LABEL="${MAX_PER_LABEL:-0}"
MAX_SENTENCES="${MAX_SENTENCES:-0}"
LABEL_FILTER="${LABEL_FILTER:-all}"
SKIP_FULL_HEAD_ABLATION="${SKIP_FULL_HEAD_ABLATION:-1}"
SKIP_TEXT_ABLATION="${SKIP_TEXT_ABLATION:-0}"
CANDIDATE_SELECTIONS="${CANDIDATE_SELECTIONS:-combined itext contrast}"
CANDIDATE_HEAD_LIMIT="${CANDIDATE_HEAD_LIMIT:-0}"

OUTPUT_DIR="${OUTPUT_DIR:-./results/coco/itext_language_prior_channel_top${TOP_K}_m${MAX_MENTIONS}}"
ROWS_DIR="${ROWS_DIR:-${OUTPUT_DIR}/head_logit_rows}"
HEAD_ROWS="${HEAD_ROWS:-${ROWS_DIR}/head_logit_proxy_ablation_rows.csv}"

mkdir -p "${OUTPUT_DIR}" "${ROWS_DIR}"

if [ ! -f "${RANKED_HEADS}" ]; then
    echo "[error] ranked heads not found: ${RANKED_HEADS}" >&2
    exit 2
fi

CANDIDATE_HEADS="${CANDIDATE_HEADS:-$(python - <<PY
import json

path = "${RANKED_HEADS}"
top_k = int("${TOP_K}")
head_limit = int("${CANDIDATE_HEAD_LIMIT}")
selections = set("${CANDIDATE_SELECTIONS}".replace(",", " ").split())
data = json.load(open(path))
records = data["heads"]
by_itext = sorted(records, key=lambda row: float(row.get("front_percentile", -1.0)), reverse=True)
by_contrast = sorted(records, key=lambda row: float(row.get("back_percentile", -1.0)), reverse=True)
heads = []
seen = set()
ranking_rows = []
if "combined" in selections:
    ranking_rows.append(records[:top_k])
if "itext" in selections:
    ranking_rows.append(by_itext[:top_k])
if "contrast" in selections:
    ranking_rows.append(by_contrast[:top_k])
for rows in ranking_rows:
    for row in rows:
        key = f"{int(row['layer'])}:{int(row['head'])}"
        if key in seen:
            continue
        heads.append(key)
        seen.add(key)
if head_limit > 0:
    heads = heads[:head_limit]
print(",".join(heads))
PY
)}"

echo "[info] output dir: ${OUTPUT_DIR}"
echo "[info] ranked heads: ${RANKED_HEADS}"
echo "[info] top k: ${TOP_K}"
echo "[info] candidate selections: ${CANDIDATE_SELECTIONS}"
echo "[info] candidate head limit: ${CANDIDATE_HEAD_LIMIT}"
echo "[info] candidate heads: $(python - <<PY
print(len([x for x in "${CANDIDATE_HEADS}".split(",") if x]))
PY
)"
echo "[info] head rows: ${HEAD_ROWS}"
echo "[info] skip full-head ablation: ${SKIP_FULL_HEAD_ABLATION}"
echo "[info] skip text ablation: ${SKIP_TEXT_ABLATION}"

model_base_arg=()
if [ -n "${MODEL_BASE}" ]; then
    model_base_arg=(--model-base "${MODEL_BASE}")
fi
full_ablation_arg=()
if [ "${SKIP_FULL_HEAD_ABLATION}" = "1" ]; then
    full_ablation_arg=(--skip-full-head-ablation)
fi
text_ablation_arg=()
if [ "${SKIP_TEXT_ABLATION}" = "1" ]; then
    text_ablation_arg=(--skip-text-ablation)
fi

if [ ! -f "${HEAD_ROWS}" ]; then
    if [ -z "${EVAL_RESULTS}" ]; then
        echo "[error] HEAD_ROWS does not exist and EVAL_RESULTS was not provided." >&2
        echo "[hint] set EVAL_RESULTS=/path/to/captions_eval_results.json or HEAD_ROWS=/path/to/head_logit_proxy_ablation_rows.csv" >&2
        exit 2
    fi
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python -m eval_scripts.soft_routing.validate_head_logit_contribution_proxy \
        --eval-results "${EVAL_RESULTS}" \
        --image-folder "${IMAGE_FOLDER}" \
        --output-dir "${ROWS_DIR}" \
        --model-path "${MODEL_PATH}" \
        "${model_base_arg[@]}" \
        --conv-mode "${CONV_MODE}" \
        --image-split "${IMAGE_SPLIT}" \
        --label-filter "${LABEL_FILTER}" \
        --max-sentences "${MAX_SENTENCES}" \
        --max-mentions "${MAX_MENTIONS}" \
        --max-per-label "${MAX_PER_LABEL}" \
        --include-adhh-top-k 0 \
        --candidate-heads "${CANDIDATE_HEADS}" \
        "${full_ablation_arg[@]}" \
        "${text_ablation_arg[@]}" \
        --resume
fi

python -m eval_scripts.soft_routing.analyze_itext_language_prior_channel \
    --head-rows "${HEAD_ROWS}" \
    --ranked-heads "${RANKED_HEADS}" \
    --output-dir "${OUTPUT_DIR}" \
    --top-k "${TOP_K}"

echo "[summary] itext language-prior groups"
column -s, -t "${OUTPUT_DIR}/itext_language_prior_group_summary.csv"

echo "[summary] itext layer distribution"
column -s, -t "${OUTPUT_DIR}/itext_layer_distribution.csv"

echo "[done] report: ${OUTPUT_DIR}/itext_language_prior_channel.md"

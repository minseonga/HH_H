#!/usr/bin/env bash
set -euo pipefail

# Build an all-layer LLaVA-1.5-13B head-score profile before choosing a
# layer window. This intentionally does not run DEACT decoding. It first
# builds/rereuses an all-layer txt-attention calibration and rank-fused head
# file, then summarizes where the high-scoring heads concentrate by layer.
#
# Example:
#   GPU_LIST="6" \
#   NUM_CHUNKS=1 \
#   TRAIN_NUM_SAMPLES=500 \
#   CALIB_IMAGE_FOLDER="/home/kms/data/pope/val2014" \
#   CALIB_CAPTION_FILE_PATH="/home/kms/data/images/mscoco/annotations/captions_val2014.json" \
#   bash LLaVA/bash_scripts/soft_routing/run_llava13b_layer_profile.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BASE_RUNNER="${SCRIPT_DIR}/run_llava13b_deact_calib_to_test.sh"

if [[ ! -x "${BASE_RUNNER}" ]]; then
  echo "[error] missing base runner: ${BASE_RUNNER}" >&2
  exit 1
fi

python_bin="${PYTHON_BIN:-python}"
if ! command -v "${python_bin}" >/dev/null 2>&1; then
  python_bin=python3
fi

ADHH_ROOT="${ADHH_ROOT:-${REPO_ROOT}/ADHH}"
ADHH_LLAVA_DIR="${ADHH_LLAVA_DIR:-${ADHH_ROOT}/LLaVA}"
RESULTS_ROOT="${RESULTS_ROOT:-${ADHH_LLAVA_DIR}/results_deact}"
DATASET="${DATASET:-coco}"
MODEL_NAME="${MODEL_NAME:-llava-v1.5-13b}"
TRAIN_NUM_SAMPLES="${TRAIN_NUM_SAMPLES:-500}"
TOPK="${TOPK:-100}"
HEAD_SCORE_KEY="${HEAD_SCORE_KEY:-global__itext_all__C_toi_HminusG_signed}"

PROFILE_LAYERS="${PROFILE_LAYERS:-0:39}"
PROFILE_TOPK_LIST="${PROFILE_TOPK_LIST:-50,100,150,200}"
PROFILE_WINDOW_SIZES="${PROFILE_WINDOW_SIZES:-6,8,10,12}"

export ADHH_ROOT ADHH_LLAVA_DIR RESULTS_ROOT DATASET MODEL_NAME TRAIN_NUM_SAMPLES TOPK HEAD_SCORE_KEY

slug_layers() {
  "${python_bin}" - "$1" <<'PY'
import sys
spec = sys.argv[1]
layers = []
for part in spec.replace(";", ",").split(","):
    part = part.strip()
    if not part:
        continue
    if ":" in part:
        a, b = map(int, part.split(":", 1))
        step = 1 if b >= a else -1
        layers.extend(range(a, b + step, step))
    elif "-" in part and not part.startswith("-"):
        a, b = map(int, part.split("-", 1))
        step = 1 if b >= a else -1
        layers.extend(range(a, b + step, step))
    else:
        layers.append(int(part))
seen, out = set(), []
for x in layers:
    if x not in seen:
        seen.add(x)
        out.append(x)
if len(out) > 1 and out == list(range(out[0], out[-1] + 1)):
    print(f"l{out[0]}_l{out[-1]}")
else:
    print("l" + "_l".join(map(str, out)))
PY
}

profile_slug="$(slug_layers "${PROFILE_LAYERS}")"
model_root="${RESULTS_ROOT}/${DATASET}/${MODEL_NAME}"
resources_root="${model_root}/resources/${profile_slug}_train_n${TRAIN_NUM_SAMPLES}"
ranked_heads="${resources_root}/surrogate_score_zoo/ranked_heads_${HEAD_SCORE_KEY}.json"
profile_dir="${model_root}/layer_profiles/${profile_slug}_train_n${TRAIN_NUM_SAMPLES}_top${TOPK}"

echo "[config] profile layers=${PROFILE_LAYERS} (${profile_slug})"
echo "[config] resources=${resources_root}"
echo "[config] ranked_heads=${ranked_heads}"
echo "[config] profile_dir=${profile_dir}"

SELECTION_LAYERS="${PROFILE_LAYERS}" \
RUN_TEST=0 \
RUN_GREEDY=0 \
RUN_DEACT=0 \
RUN_CALIBRATION="${RUN_CALIBRATION:-1}" \
RUN_HEAD_BUILD="${RUN_HEAD_BUILD:-1}" \
DELETE_CALIB_TRACE_AFTER_SUMMARY="${DELETE_CALIB_TRACE_AFTER_SUMMARY:-1}" \
KEEP_MERGED_TRACE="${KEEP_MERGED_TRACE:-false}" \
CALIB_ENABLE_ATTENTION_ANALYSIS="${CALIB_ENABLE_ATTENTION_ANALYSIS:-0}" \
bash "${BASE_RUNNER}"

if [[ ! -f "${ranked_heads}" ]]; then
  echo "[error] missing ranked heads after calibration/profile build: ${ranked_heads}" >&2
  exit 1
fi

"${python_bin}" "${REPO_ROOT}/LLaVA/eval_scripts/soft_routing/build_ranked_head_layer_profile.py" \
  --ranked-heads "${ranked_heads}" \
  --output-dir "${profile_dir}" \
  --top-k "${TOPK}" \
  --top-k-list "${PROFILE_TOPK_LIST}" \
  --window-sizes "${PROFILE_WINDOW_SIZES}" \
  --title "LLaVA-1.5-13B all-layer head-score profile"

echo "[done] profile outputs"
echo "  summary: ${profile_dir}/layer_profile_summary.csv"
echo "  windows: ${profile_dir}/recommended_windows.csv"
echo "  figure:  ${profile_dir}/layer_profile_top${TOPK}.svg"

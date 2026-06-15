#!/usr/bin/env bash
set -euo pipefail

# Run LLaVA-1.5-13B greedy and/or paper-style AD-HH on the exact same
# validation sample-id file used by the 13B DEACT runs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

ADHH_LLAVA_DIR="${ADHH_LLAVA_DIR:-${ROOT_DIR}/ADHH/LLaVA}"
PYTHON_BIN="${PYTHON_BIN:-python}"

MODEL_NAME="${MODEL_NAME:-llava-v1.5-13b}"
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-13b}"
DATASET="${DATASET:-coco}"
GPU_ID="${GPU_ID:-6}"
NUM_SAMPLES="${NUM_SAMPLES:-500}"
SEED="${SEED:-42}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"

IMAGE_FOLDER="${IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
CAPTION_FILE_PATH="${CAPTION_FILE_PATH:-/home/kms/data/images/mscoco/annotations/captions_val2014.json}"
ANNOTATION_DIR="${ANNOTATION_DIR:-/home/kms/data/images/mscoco/annotations}"

RUN_GREEDY="${RUN_GREEDY:-0}"
RUN_ADHH="${RUN_ADHH:-1}"
RESUME="${RESUME:-0}"
STREAM_LOGS="${STREAM_LOGS:-1}"

ADHH_TOPK="${ADHH_TOPK:-20}"
ADHH_THRESHOLD="${ADHH_THRESHOLD:-0.4}"
ADHH_HEAD_SOURCE="${ADHH_HEAD_SOURCE:-default}"
ADHH_HEAD_FILE="${ADHH_HEAD_FILE:-}"

RESULT_ROOT="${RESULT_ROOT:-${ADHH_LLAVA_DIR}/results_deact/${DATASET}/${MODEL_NAME}}"
SAMPLE_ID_FILE="${SAMPLE_ID_FILE:-${RESULT_ROOT}/shared_samples/val_seed${SEED}_n${NUM_SAMPLES}.json}"

GREEDY_DIR="${GREEDY_DIR:-${RESULT_ROOT}/baselines/greedy/tok${MAX_NEW_TOKENS}/n${NUM_SAMPLES}_seed${SEED}}"
ADHH_DIR="${ADHH_DIR:-${RESULT_ROOT}/baselines/adhh/default_k${ADHH_TOPK}_tau${ADHH_THRESHOLD}/tok${MAX_NEW_TOKENS}/n${NUM_SAMPLES}_seed${SEED}}"

bool_true() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|y|Y) return 0 ;;
    *) return 1 ;;
  esac
}

run_logged() {
  local log_file=$1
  shift
  if bool_true "${STREAM_LOGS}"; then
    "$@" 2>&1 | tee "${log_file}"
  else
    "$@" > "${log_file}" 2>&1
  fi
}

ensure_sample_ids() {
  if [[ -f "${SAMPLE_ID_FILE}" ]]; then
    echo "[sample] using existing sample ids: ${SAMPLE_ID_FILE}"
    return
  fi

  echo "[sample] creating sample ids: ${SAMPLE_ID_FILE}"
  mkdir -p "$(dirname "${SAMPLE_ID_FILE}")"
  "${PYTHON_BIN}" - "${CAPTION_FILE_PATH}" "${SEED}" "${NUM_SAMPLES}" "${SAMPLE_ID_FILE}" <<'PY'
import json, os, random, sys
from pycocotools.coco import COCO

caption_file, seed, num_samples, out_file = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
random.seed(seed)
coco = COCO(caption_file)
sampled = random.sample(coco.getImgIds(), num_samples)
os.makedirs(os.path.dirname(out_file), exist_ok=True)
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(sampled, f, indent=2)
print(f"saved sample ids -> {out_file}")
PY
}

run_chair_eval() {
  local result_dir=$1
  (
    cd "${ADHH_LLAVA_DIR}"
    run_logged "${result_dir}/chair.log" \
      "${PYTHON_BIN}" eval_scripts/eval_utils/eval_chair.py \
        --annotation-dir "${ANNOTATION_DIR}" \
        --answers-file "${result_dir}/captions.jsonl" \
        --caption_file captions_val2014.json
  )
}

resume_args=()
if bool_true "${RESUME}"; then
  resume_args+=(--resume)
fi

ensure_sample_ids

echo "[config] model=${MODEL_NAME} path=${MODEL_PATH}"
echo "[config] sample_id_file=${SAMPLE_ID_FILE}"
echo "[config] greedy=${GREEDY_DIR}"
echo "[config] adhh=${ADHH_DIR}"

if bool_true "${RUN_GREEDY}"; then
  mkdir -p "${GREEDY_DIR}"
  echo "[greedy] start -> ${GREEDY_DIR}"
  (
    cd "${ADHH_LLAVA_DIR}"
    run_logged "${GREEDY_DIR}/decode.log" \
      env CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -m eval_scripts.eval_caption_dynamic \
        --model-path "${MODEL_PATH}" \
        --image-folder "${IMAGE_FOLDER}" \
        --caption_file_path "${CAPTION_FILE_PATH}" \
        --answers-file "${GREEDY_DIR}/captions.jsonl" \
        --dataset "${DATASET}" \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --num_samples "${NUM_SAMPLES}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --seed "${SEED}" \
        --intervention none \
        --sample-id-file "${SAMPLE_ID_FILE}" \
        "${resume_args[@]}"
  )
  run_chair_eval "${GREEDY_DIR}"
fi

if bool_true "${RUN_ADHH}"; then
  mkdir -p "${ADHH_DIR}"
  extra_head_args=()
  if [[ "${ADHH_HEAD_SOURCE}" == "file" ]]; then
    if [[ -z "${ADHH_HEAD_FILE}" ]]; then
      echo "[error] ADHH_HEAD_SOURCE=file requires ADHH_HEAD_FILE" >&2
      exit 1
    fi
    extra_head_args+=(--head-file "${ADHH_HEAD_FILE}")
  fi

  echo "[adhh] start source=${ADHH_HEAD_SOURCE} topk=${ADHH_TOPK} tau=${ADHH_THRESHOLD} -> ${ADHH_DIR}"
  (
    cd "${ADHH_LLAVA_DIR}"
    run_logged "${ADHH_DIR}/decode.log" \
      env CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -m eval_scripts.eval_caption_adhh \
        --model-path "${MODEL_PATH}" \
        --image-folder "${IMAGE_FOLDER}" \
        --caption_file_path "${CAPTION_FILE_PATH}" \
        --answers-file "${ADHH_DIR}/captions.jsonl" \
        --dataset "${DATASET}" \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --num_samples "${NUM_SAMPLES}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --seed "${SEED}" \
        --num-workers 4 \
        --intervention adhh \
        --topk "${ADHH_TOPK}" \
        --text-threshold "${ADHH_THRESHOLD}" \
        --head-source "${ADHH_HEAD_SOURCE}" \
        "${extra_head_args[@]}" \
        --sample-id-file "${SAMPLE_ID_FILE}" \
        --log-intervention-stats \
        "${resume_args[@]}"
  )
  run_chair_eval "${ADHH_DIR}"
fi

echo "[done]"
echo "  sample ids: ${SAMPLE_ID_FILE}"
echo "  greedy:     ${GREEDY_DIR}/captions_eval_results.json"
echo "  adhh:       ${ADHH_DIR}/captions_eval_results.json"

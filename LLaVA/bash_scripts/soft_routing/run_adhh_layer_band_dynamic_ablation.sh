#!/usr/bin/env bash
set -euo pipefail

# Run the teammate ADHH dynamic captioning code with layer-band-filtered head
# pools stored in this repository. This keeps ADHH as an external clone while
# preserving the reproducible experiment command in HH_H.
#
# Run from Hallucination-Attribution:
#   GPU_LIST="0 1 2 3" bash LLaVA/bash_scripts/soft_routing/run_adhh_layer_band_dynamic_ablation.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

ADHH_ROOT="${ADHH_ROOT:-${REPO_ROOT}/ADHH}"
ADHH_LLAVA_DIR="${ADHH_LLAVA_DIR:-${ADHH_ROOT}/LLaVA}"

if [[ ! -d "${ADHH_LLAVA_DIR}" ]]; then
  echo "[error] missing ADHH LLaVA clone: ${ADHH_LLAVA_DIR}" >&2
  echo "[hint] clone or set ADHH_LLAVA_DIR=/path/to/ADHH/LLaVA" >&2
  exit 1
fi

model_name="${MODEL_NAME:-llava-v1.5-7b}"
model_path="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
dataset="${DATASET:-coco}"
data_path="${DATA_PATH:-../dataset}"
seed="${SEED:-42}"
num_samples="${NUM_SAMPLES:-500}"

head_dir="${HEAD_DIR:-${REPO_ROOT}/LLaVA/results/coco/layer_band_dynamic_ablation_head_files}"
head_score_key="${HEAD_SCORE_KEY:-global__itext_all__C_toi_HminusG}"
head_score_normalize="${HEAD_SCORE_NORMALIZE:-rank_percentile}"
topk="${TOPK:-100}"

dynamic_context_mode="${DYNAMIC_CONTEXT_MODE:-ratio_exp}"
dynamic_strength="${DYNAMIC_STRENGTH:-1.0}"
dynamic_exp_sharpness="${DYNAMIC_EXP_SHARPNESS:-8.0}"
dynamic_tau="${DYNAMIC_TAU:-0.90}"
dynamic_score_power="${DYNAMIC_SCORE_POWER:-1.0}"
dynamic_redistribute="${DYNAMIC_REDISTRIBUTE:-renorm}"
use_head_scores="${USE_HEAD_SCORES:-true}"
resume="${RESUME:-true}"

read -r -a gpu_list <<< "${GPU_LIST:-0 1}"
read -r -a band_list <<< "${BANDS:-l0_l8 l9_l16 l17_l24 l25_l31}"

output_root="${OUTPUT_ROOT:-${REPO_ROOT}/LLaVA/results/coco/adhh_layer_band_dynamic_ablation_n${num_samples}_k${topk}_s${dynamic_strength}_q${dynamic_exp_sharpness}_tau${dynamic_tau}}"
sample_id_file="${SAMPLE_ID_FILE:-${ADHH_LLAVA_DIR}/results/${dataset}/shared_samples/val_seed${seed}_n${num_samples}.json}"
mkdir -p "$(dirname "${sample_id_file}")" "${output_root}"

export PYTHONUNBUFFERED=1
python_bin="${PYTHON_BIN:-python}"
if ! command -v "${python_bin}" >/dev/null 2>&1; then
  python_bin=python3
fi

if [[ ! -f "${head_dir}/layer_band_head_files_summary.csv" ]]; then
  echo "[error] missing layer-band head files under: ${head_dir}" >&2
  exit 1
fi

if [[ ! -f "${sample_id_file}" ]]; then
  echo "[sample] creating fixed sample ids: ${sample_id_file}"
  (
    cd "${ADHH_LLAVA_DIR}"
    "${python_bin}" - <<PY
import json, random
from pycocotools.coco import COCO

caption_file = "${data_path}/coco/annotations/captions_val2014.json"
seed = ${seed}
num_samples = ${num_samples}
out_file = "${sample_id_file}"

random.seed(seed)
coco = COCO(caption_file)
sampled = random.sample(coco.getImgIds(), num_samples)
with open(out_file, "w") as f:
    json.dump(sampled, f, indent=2)
print(f"saved sample ids -> {out_file}")
PY
  )
else
  echo "[sample] using existing sample ids: ${sample_id_file}"
fi

run_band() {
  local band=$1
  local gpu=$2
  local head_file="${head_dir}/ranked_heads_${head_score_key}_${band}.json"
  local result_path="${output_root}/${model_name}_dynamic_${band}_${dynamic_context_mode}_k${topk}_s${dynamic_strength}_q${dynamic_exp_sharpness}_tau${dynamic_tau}_p${dynamic_score_power}_${dynamic_redistribute}"

  if [[ ! -f "${head_file}" ]]; then
    echo "[error] missing head file: ${head_file}" >&2
    return 1
  fi

  mkdir -p "${result_path}"

  local score_args=()
  if [[ "${use_head_scores}" == "true" ]]; then
    score_args+=(--use-head-scores)
  fi

  local resume_args=()
  if [[ "${resume}" == "true" ]]; then
    resume_args+=(--resume)
  fi

  echo "[GPU ${gpu}] start band=${band}"
  (
    cd "${ADHH_LLAVA_DIR}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -m eval_scripts.eval_caption_dynamic \
      --model-path "${model_path}" \
      --image-folder "${data_path}/coco/val2014" \
      --caption_file_path "${data_path}/coco/annotations/captions_val2014.json" \
      --answers-file "${result_path}/captions.jsonl" \
      --dataset "${dataset}" \
      --temperature 0 \
      --conv-mode vicuna_v1 \
      --num_samples "${num_samples}" \
      --seed "${seed}" \
      --intervention dynamic \
      --head-source file \
      --head-file "${head_file}" \
      --head-score-key "${head_score_key}" \
      --head-score-normalize "${head_score_normalize}" \
      --topk "${topk}" \
      --dynamic-strength "${dynamic_strength}" \
      --dynamic-context-mode "${dynamic_context_mode}" \
      --dynamic-tau "${dynamic_tau}" \
      --dynamic-exp-sharpness "${dynamic_exp_sharpness}" \
      --dynamic-score-power "${dynamic_score_power}" \
      --dynamic-redistribute "${dynamic_redistribute}" \
      "${score_args[@]}" \
      --log-intervention-stats \
      --sample-id-file "${sample_id_file}" \
      "${resume_args[@]}" \
      > "${result_path}/decode.log" 2>&1

    "${python_bin}" eval_scripts/eval_utils/eval_chair.py \
      --annotation-dir "${data_path}/coco/annotations" \
      --answers-file "${result_path}/captions.jsonl" \
      --caption_file captions_val2014.json \
      > "${result_path}/chair.log" 2>&1
  )
  echo "[GPU ${gpu}] done band=${band} result=${result_path}"
}

pids=()
job_idx=0
for band in "${band_list[@]}"; do
  gpu="${gpu_list[$((job_idx % ${#gpu_list[@]}))]}"
  run_band "${band}" "${gpu}" &
  pids+=("$!")
  job_idx=$((job_idx + 1))

  if (( ${#pids[@]} == ${#gpu_list[@]} )); then
    wait "${pids[@]}"
    pids=()
  fi
done

if (( ${#pids[@]} > 0 )); then
  wait "${pids[@]}"
fi

"${python_bin}" - <<PY
import csv, json
from pathlib import Path

root = Path("${output_root}")
rows = []
for path in sorted(root.glob("*/captions_eval_results.json")):
    data = json.load(open(path))
    metrics = data.get("overall_metrics", {})
    name = path.parent.name
    marker = "_dynamic_"
    band = name.split(marker, 1)[1].split("_ratio_exp", 1)[0] if marker in name else name
    bleu = metrics.get("Bleu", [""])
    rows.append({
        "band": band,
        "CHAIRs": metrics.get("CHAIRs", ""),
        "CHAIRi": metrics.get("CHAIRi", ""),
        "Bleu1": bleu[0] if isinstance(bleu, list) and bleu else bleu,
        "METEOR": metrics.get("METEOR", ""),
        "CIDEr": metrics.get("CIDEr", ""),
        "result_dir": str(path.parent),
    })

out = root / "layer_band_dynamic_ablation_summary.csv"
with open(out, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["band", "CHAIRs", "CHAIRi", "Bleu1", "METEOR", "CIDEr", "result_dir"])
    writer.writeheader()
    writer.writerows(rows)
print(f"[summary] {out}")
for row in rows:
    print(row)
PY

echo "[done] ${output_root}"

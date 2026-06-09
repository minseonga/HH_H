#!/usr/bin/env bash
set -euo pipefail

# End-to-end LLaVA-1.5-13B DEACT run using the teammate ADHH clone:
#   1) build a 13B all-head txt-attention calibration summary
#   2) filter the summary to the target layer window
#   3) build the rank-fused signed head pool
#   4) run greedy and DEACT late-boost captioning on the same COCO-val samples
#   5) run CHAIR and write a compact summary
#
# Run from Hallucination-Attribution on the server:
#
#   GPU_LIST="6 7" \
#   NUM_SAMPLES=500 \
#   TRAIN_NUM_SAMPLES=500 \
#   bash LLaVA/bash_scripts/soft_routing/run_llava13b_deact_calib_to_test.sh
#
# Useful controls:
#   RUN_CALIBRATION=0  reuse existing txtattn_summary.json
#   RUN_HEAD_BUILD=0   reuse existing filtered summary/head pool
#   RUN_TEST=0         stop after calibration/head-pool build
#   RESUME=0           restart decoding outputs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

ADHH_ROOT="${ADHH_ROOT:-${REPO_ROOT}/ADHH}"
ADHH_LLAVA_DIR="${ADHH_LLAVA_DIR:-${ADHH_ROOT}/LLaVA}"

if [[ ! -d "${ADHH_LLAVA_DIR}" ]]; then
  echo "[error] missing ADHH LLaVA clone: ${ADHH_LLAVA_DIR}" >&2
  echo "[hint] set ADHH_LLAVA_DIR=/path/to/ADHH/LLaVA" >&2
  exit 1
fi

python_bin="${PYTHON_BIN:-python}"
if ! command -v "${python_bin}" >/dev/null 2>&1; then
  python_bin=python3
fi

model_name="${MODEL_NAME:-llava-v1.5-13b}"
model_path="${MODEL_PATH:-liuhaotian/llava-v1.5-13b}"
dataset="${DATASET:-coco}"
seed="${SEED:-42}"
num_samples="${NUM_SAMPLES:-500}"
train_num_samples="${TRAIN_NUM_SAMPLES:-500}"
max_new_tokens="${MAX_NEW_TOKENS:-128}"

# Server paths used in the 7B runs. Override if your dataset symlinks differ.
annotation_dir="${ANNOTATION_DIR:-/home/kms/data/images/mscoco/annotations}"
val_image_folder="${IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
val_caption_file="${CAPTION_FILE_PATH:-${annotation_dir}/captions_val2014.json}"
train_image_folder="${CALIB_IMAGE_FOLDER:-/home/kms/data/images/mscoco/train2014}"
train_caption_file="${CALIB_CAPTION_FILE_PATH:-${annotation_dir}/captions_train2014.json}"

num_layers="${NUM_LAYERS:-40}"
num_heads="${NUM_HEADS:-40}"
all_layer_spec="${ALL_LAYER_SPEC:-0:39}"
selection_layers="${SELECTION_LAYERS:-9:16}"

topk="${TOPK:-100}"
head_score_key="${HEAD_SCORE_KEY:-global__itext_all__C_toi_HminusG_signed}"
head_score_normalize="${HEAD_SCORE_NORMALIZE:-rank_percentile}"
min_head_back_raw="${MIN_HEAD_BACK_RAW:-0.0}"
use_head_scores="${USE_HEAD_SCORES:-true}"

intervention="${INTERVENTION:-late_boost}"
dynamic_context_mode="${DYNAMIC_CONTEXT_MODE:-ratio_exp}"
dynamic_strength="${DYNAMIC_STRENGTH:-1.0}"
dynamic_exp_sharpness="${DYNAMIC_EXP_SHARPNESS:-8.0}"
dynamic_tau="${DYNAMIC_TAU:-0.90}"
dynamic_score_power="${DYNAMIC_SCORE_POWER:-1.0}"
dynamic_redistribute="${DYNAMIC_REDISTRIBUTE:-none}"
dynamic_renorm="${DYNAMIC_RENORM:-false}"
dynamic_late_boost_start="${DYNAMIC_LATE_BOOST_START:-0}"
dynamic_late_boost_end="${DYNAMIC_LATE_BOOST_END:-${max_new_tokens}}"
dynamic_late_boost_mode="${DYNAMIC_LATE_BOOST_MODE:-linear}"
dynamic_late_tau="${DYNAMIC_LATE_TAU:-0.80}"
log_dynamic_trace="${LOG_DYNAMIC_TRACE:-true}"
dynamic_trace_topn="${DYNAMIC_TRACE_TOPN:-10}"
dynamic_trace_every="${DYNAMIC_TRACE_EVERY:-5}"

results_root="${RESULTS_ROOT:-${ADHH_LLAVA_DIR}/results_deact}"
calib_result_path="${CALIB_RESULT_PATH:-${ADHH_LLAVA_DIR}/results/${dataset}/${model_name}_base_original_qa_n${train_num_samples}_txtattn_l0_l$((num_layers - 1))_allheads}"
calib_existing_sample_file="${CALIB_EXISTING_SAMPLE_FILE:-}"
txtattn_trace_mode="${TXTATTN_TRACE_MODE:-last_row}"
keep_merged_trace="${KEEP_MERGED_TRACE:-false}"

run_calibration="${RUN_CALIBRATION:-1}"
run_head_build="${RUN_HEAD_BUILD:-1}"
run_greedy="${RUN_GREEDY:-1}"
run_deact="${RUN_DEACT:-1}"
run_test="${RUN_TEST:-1}"
resume="${RESUME:-1}"
dry_run="${DRY_RUN:-0}"

read -r -a gpu_list <<< "${GPU_LIST:-0 1}"
num_chunks="${NUM_CHUNKS:-${#gpu_list[@]}}"

export PYTHONUNBUFFERED=1

bool_true() {
  [[ "$1" == "1" || "$1" == "true" || "$1" == "TRUE" || "$1" == "yes" ]]
}

run_cmd() {
  echo "+ $*"
  if ! bool_true "${dry_run}"; then
    "$@"
  fi
}

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

slug_float() {
  "${python_bin}" - "$1" <<'PY'
import sys
x = float(sys.argv[1])
if 0 < x < 1:
    s = f"{x:.4f}".rstrip("0")
    decimals = s.split(".", 1)[1] if "." in s else ""
    if len(decimals) < 2:
        s = f"{x:.2f}"
else:
    s = f"{x:.4f}".rstrip("0").rstrip(".")
print((s or "0").replace(".", ""))
PY
}

selection_slug="$(slug_layers "${selection_layers}")"
q_slug="$(slug_float "${dynamic_exp_sharpness}")"
tau_hi_slug="$(slug_float "${dynamic_tau}")"
tau_lo_slug="$(slug_float "${dynamic_late_tau}")"

model_root="${results_root}/${dataset}/${model_name}"
resources_root="${model_root}/resources/${selection_slug}_train_n${train_num_samples}"
filtered_summary="${resources_root}/txtattn_summary.json"
surrogate_dir="${resources_root}/surrogate_score_zoo"
head_file="${HEAD_FILE:-${surrogate_dir}/ranked_heads_${head_score_key}.json}"
tau_file="${resources_root}/dynamic_tau_estimate.json"
candidate_head_file="${calib_result_path}/surrogate_hh_scores/candidate_heads_l0_l$((num_layers - 1)).json"

sample_dir="${model_root}/shared_samples"
sample_id_file="${SAMPLE_ID_FILE:-${sample_dir}/val_seed${seed}_n${num_samples}.json}"
greedy_dir="${GREEDY_DIR:-${model_root}/baselines/greedy/tok${max_new_tokens}/n${num_samples}_seed${seed}}"
update_name="direct"
if [[ "${dynamic_redistribute}" != "none" ]]; then
  update_name="redir_${dynamic_redistribute}"
elif bool_true "${dynamic_renorm}"; then
  update_name="renorm"
fi
deact_dir="${DEACT_DIR:-${model_root}/main/${selection_slug}/k${topk}/${update_name}/tok${max_new_tokens}/q${q_slug}_tau${tau_hi_slug}-${tau_lo_slug}_n${num_samples}_seed${seed}}"

mkdir -p "${resources_root}" "${surrogate_dir}" "${sample_dir}" "$(dirname "${candidate_head_file}")"

echo "[config] model=${model_name} path=${model_path}"
echo "[config] calibration=${calib_result_path}"
echo "[config] resources=${resources_root}"
echo "[config] head_file=${head_file}"
echo "[config] greedy=${greedy_dir}"
echo "[config] deact=${deact_dir}"

if [[ ! -f "${candidate_head_file}" ]]; then
  echo "[heads] creating 13B all-head candidate file: ${candidate_head_file}"
  run_cmd "${python_bin}" - "${candidate_head_file}" "${num_layers}" "${num_heads}" <<'PY'
import json, os, sys
path, num_layers, num_heads = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
heads = [[l, h] for l in range(num_layers) for h in range(num_heads)]
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, "w", encoding="utf-8") as f:
    json.dump(heads, f, indent=2)
print(f"wrote {len(heads)} heads -> {path}")
PY
else
  echo "[heads] using existing candidate file: ${candidate_head_file}"
fi

if bool_true "${run_calibration}"; then
  if [[ -f "${calib_result_path}/txtattn_summary.json" ]]; then
    echo "[calibration] reusing existing summary: ${calib_result_path}/txtattn_summary.json"
  else
    mkdir -p "${calib_result_path}/analysis"
    echo "[calibration] running ${num_chunks} chunks on GPU_LIST=${GPU_LIST:-0 1}"

    resume_args=()
    if bool_true "${resume}"; then
      resume_args+=(--resume)
    fi

    existing_sample_args=()
    if [[ -n "${calib_existing_sample_file}" && -f "${calib_existing_sample_file}" ]]; then
      existing_sample_args+=(--use-existing-sample-file --existing-sample-file "${calib_existing_sample_file}")
      echo "[calibration] using existing calibration samples: ${calib_existing_sample_file}"
    fi

    pids=()
    for ((chunk_idx=0; chunk_idx<num_chunks; chunk_idx++)); do
      gpu="${gpu_list[$((chunk_idx % ${#gpu_list[@]}))]}"
      chunk_analysis="${calib_result_path}/analysis/chunk${chunk_idx}"
      mkdir -p "${chunk_analysis}"
      echo "[calibration][GPU ${gpu}] start chunk ${chunk_idx}/${num_chunks}"
      (
        cd "${ADHH_LLAVA_DIR}"
        CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -m eval_scripts.eval_caption \
          --model-path "${model_path}" \
          --image-folder "${train_image_folder}" \
          --caption_file_path "${train_caption_file}" \
          --annotation-dir "${annotation_dir}" \
          --answers-file "${calib_result_path}/captions.chunk${chunk_idx}.jsonl" \
          --output-path "${chunk_analysis}" \
          --dataset "${dataset}" \
          --temperature 0 \
          --conv-mode vicuna_v1 \
          --num_samples "${train_num_samples}" \
          --save-sample-ids "${calib_result_path}/sample_ids.chunk${chunk_idx}.json" \
          --max_new_tokens "${max_new_tokens}" \
          ${existing_sample_args[@]+"${existing_sample_args[@]}"} \
          --enable-attention-analysis \
          --enable-pre-token-analysis \
          --enable-txtattn-trace \
          --txtattn-trace-mode "${txtattn_trace_mode}" \
          --txtattn-head-file "${candidate_head_file}" \
          --txtattn-topk 0 \
          --txtattn-output-file "${calib_result_path}/txtattn_trace.chunk${chunk_idx}.jsonl" \
          --txtattn-summary-file "${calib_result_path}/txtattn_summary.chunk${chunk_idx}.json" \
          --num-chunks "${num_chunks}" \
          --chunk-idx "${chunk_idx}" \
          ${resume_args[@]+"${resume_args[@]}"} \
          > "${calib_result_path}/decode.chunk${chunk_idx}.log" 2>&1
      ) &
      pids+=("$!")
    done
    wait ${pids[@]+"${pids[@]}"}

    echo "[calibration] merging captions/traces"
    : > "${calib_result_path}/captions.jsonl"
    trace_files=()
    if bool_true "${keep_merged_trace}"; then
      : > "${calib_result_path}/txtattn_trace.jsonl"
    fi
    for ((chunk_idx=0; chunk_idx<num_chunks; chunk_idx++)); do
      if [[ -f "${calib_result_path}/captions.chunk${chunk_idx}.jsonl" ]]; then
        cat "${calib_result_path}/captions.chunk${chunk_idx}.jsonl" >> "${calib_result_path}/captions.jsonl"
      fi
      if [[ -f "${calib_result_path}/txtattn_trace.chunk${chunk_idx}.jsonl" ]]; then
        trace_files+=("${calib_result_path}/txtattn_trace.chunk${chunk_idx}.jsonl")
        if bool_true "${keep_merged_trace}"; then
          cat "${calib_result_path}/txtattn_trace.chunk${chunk_idx}.jsonl" >> "${calib_result_path}/txtattn_trace.jsonl"
        fi
      fi
    done
    if (( ${#trace_files[@]} == 0 )); then
      echo "[error] no txtattn trace chunks were produced under ${calib_result_path}" >&2
      exit 1
    fi
    (
      cd "${ADHH_LLAVA_DIR}"
      "${python_bin}" -m eval_scripts.summarize_txtattn_trace \
        --trace-file ${trace_files[@]+"${trace_files[@]}"} \
        --head-file "${candidate_head_file}" \
        --topk 0 \
        --summary-file "${calib_result_path}/txtattn_summary.json"
    )
  fi
fi

if bool_true "${run_head_build}"; then
  if [[ ! -f "${calib_result_path}/txtattn_summary.json" ]]; then
    echo "[error] missing calibration summary: ${calib_result_path}/txtattn_summary.json" >&2
    echo "[hint] run calibration first, or set CALIB_RESULT_PATH to an existing 13B calibration output" >&2
    exit 1
  fi

  echo "[head-build] filtering ${selection_layers} -> ${filtered_summary}"
  run_cmd "${python_bin}" "${ADHH_LLAVA_DIR}/eval_scripts/filter_txtattn_summary.py" \
    --summary-file "${calib_result_path}/txtattn_summary.json" \
    --output-file "${filtered_summary}" \
    --layers "${selection_layers}"

  echo "[head-build] computing surrogate score zoo -> ${surrogate_dir}"
  run_cmd "${python_bin}" "${ADHH_LLAVA_DIR}/eval_scripts/compute_surrogate_score_zoo.py" \
    --summary-file "${filtered_summary}" \
    --output-dir "${surrogate_dir}"

  if [[ ! -f "${head_file}" ]]; then
    echo "[error] missing requested head file after build: ${head_file}" >&2
    exit 1
  fi

  echo "[head-build] estimating tau -> ${tau_file}"
  run_cmd "${python_bin}" "${ADHH_LLAVA_DIR}/eval_scripts/estimate_dynamic_tau.py" \
    --summary-file "${filtered_summary}" \
    --head-file "${head_file}" \
    --topk "${topk}" \
    --topk-list "${topk}" \
    --calibration-scope selected_head \
    --calibration-bucket all \
    --hi-quantile q66 \
    --lo-quantile q33 \
    --output-file "${tau_file}" \
    --round-step 0.01 \
    --round-mode floor
fi

if ! bool_true "${run_test}"; then
  echo "[done] stopped before test stage"
  exit 0
fi

if bool_true "${run_deact}" && [[ ! -f "${head_file}" ]]; then
  echo "[error] missing DEACT head file: ${head_file}" >&2
  exit 1
fi

if [[ ! -f "${sample_id_file}" ]]; then
  echo "[sample] creating fixed validation sample ids: ${sample_id_file}"
  run_cmd "${python_bin}" - "${val_caption_file}" "${seed}" "${num_samples}" "${sample_id_file}" <<'PY'
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
else
  echo "[sample] using existing validation sample ids: ${sample_id_file}"
fi

resume_args=()
if bool_true "${resume}"; then
  resume_args+=(--resume)
fi

run_chair_eval() {
  local result_dir=$1
  (
    cd "${ADHH_LLAVA_DIR}"
    "${python_bin}" eval_scripts/eval_utils/eval_chair.py \
      --annotation-dir "${annotation_dir}" \
      --answers-file "${result_dir}/captions.jsonl" \
      --caption_file captions_val2014.json \
      > "${result_dir}/chair.log" 2>&1
  )
}

if bool_true "${run_greedy}"; then
  mkdir -p "${greedy_dir}"
  echo "[test] greedy -> ${greedy_dir}"
  (
    cd "${ADHH_LLAVA_DIR}"
    CUDA_VISIBLE_DEVICES="${gpu_list[0]}" "${python_bin}" -m eval_scripts.eval_caption_dynamic \
      --model-path "${model_path}" \
      --image-folder "${val_image_folder}" \
      --caption_file_path "${val_caption_file}" \
      --answers-file "${greedy_dir}/captions.jsonl" \
      --dataset "${dataset}" \
      --temperature 0 \
      --conv-mode vicuna_v1 \
      --num_samples "${num_samples}" \
      --max_new_tokens "${max_new_tokens}" \
      --seed "${seed}" \
      --intervention none \
      --sample-id-file "${sample_id_file}" \
      ${resume_args[@]+"${resume_args[@]}"} \
      > "${greedy_dir}/decode.log" 2>&1
  )
  run_chair_eval "${greedy_dir}"
fi

if bool_true "${run_deact}"; then
  mkdir -p "${deact_dir}"
  score_args=()
  if bool_true "${use_head_scores}"; then
    score_args+=(--use-head-scores)
  fi
  renorm_args=()
  if ! bool_true "${dynamic_renorm}"; then
    renorm_args+=(--no-dynamic-renorm)
  fi
  trace_args=()
  if bool_true "${log_dynamic_trace}"; then
    trace_args+=(--log-dynamic-trace --dynamic-trace-topn "${dynamic_trace_topn}" --dynamic-trace-every "${dynamic_trace_every}")
  fi

  echo "[test] DEACT ${intervention} -> ${deact_dir}"
  (
    cd "${ADHH_LLAVA_DIR}"
    CUDA_VISIBLE_DEVICES="${gpu_list[0]}" "${python_bin}" -m eval_scripts.eval_caption_dynamic \
      --model-path "${model_path}" \
      --image-folder "${val_image_folder}" \
      --caption_file_path "${val_caption_file}" \
      --answers-file "${deact_dir}/captions.jsonl" \
      --dataset "${dataset}" \
      --temperature 0 \
      --conv-mode vicuna_v1 \
      --num_samples "${num_samples}" \
      --max_new_tokens "${max_new_tokens}" \
      --seed "${seed}" \
      --intervention "${intervention}" \
      --head-source file \
      --head-file "${head_file}" \
      --head-score-key "${head_score_key}" \
      --head-score-normalize "${head_score_normalize}" \
      --min-head-back-raw "${min_head_back_raw}" \
      --topk "${topk}" \
      --dynamic-strength "${dynamic_strength}" \
      --dynamic-context-mode "${dynamic_context_mode}" \
      --dynamic-tau "${dynamic_tau}" \
      --dynamic-exp-sharpness "${dynamic_exp_sharpness}" \
      --dynamic-late-boost-start "${dynamic_late_boost_start}" \
      --dynamic-late-boost-end "${dynamic_late_boost_end}" \
      --dynamic-late-boost-mode "${dynamic_late_boost_mode}" \
      --dynamic-late-tau "${dynamic_late_tau}" \
      --dynamic-score-power "${dynamic_score_power}" \
      --dynamic-redistribute "${dynamic_redistribute}" \
      ${renorm_args[@]+"${renorm_args[@]}"} \
      ${score_args[@]+"${score_args[@]}"} \
      ${trace_args[@]+"${trace_args[@]}"} \
      --log-intervention-stats \
      --sample-id-file "${sample_id_file}" \
      ${resume_args[@]+"${resume_args[@]}"} \
      > "${deact_dir}/decode.log" 2>&1
  )
  run_chair_eval "${deact_dir}"
fi

"${python_bin}" - <<PY
import csv, json
from pathlib import Path

rows = []
for method, path in [
    ("greedy", Path("${greedy_dir}") / "captions_eval_results.json"),
    ("deact_${intervention}_${selection_slug}_k${topk}", Path("${deact_dir}") / "captions_eval_results.json"),
]:
    if not path.exists():
        continue
    data = json.load(open(path))
    metrics = data.get("overall_metrics", {})
    bleu = metrics.get("Bleu", [""])
    rows.append({
        "method": method,
        "CHAIRs": metrics.get("CHAIRs", ""),
        "CHAIRi": metrics.get("CHAIRi", ""),
        "Bleu1": bleu[0] if isinstance(bleu, list) and bleu else bleu,
        "METEOR": metrics.get("METEOR", ""),
        "CIDEr": metrics.get("CIDEr", ""),
        "ObjectPrecision": metrics.get("ObjectPrecision", ""),
        "ObjectRecall": metrics.get("ObjectRecall", ""),
        "ObjectF1": metrics.get("ObjectF1", ""),
        "result_dir": str(path.parent),
    })

out = Path("${model_root}") / "llava13b_deact_calib_to_test_summary.csv"
out.parent.mkdir(parents=True, exist_ok=True)
fields = ["method", "CHAIRs", "CHAIRi", "Bleu1", "METEOR", "CIDEr", "ObjectPrecision", "ObjectRecall", "ObjectF1", "result_dir"]
with open(out, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print(f"[summary] {out}")
for row in rows:
    print(row)
PY

echo "[done]"
echo "  calibration summary: ${calib_result_path}/txtattn_summary.json"
echo "  filtered summary:    ${filtered_summary}"
echo "  head file:           ${head_file}"
echo "  greedy result:       ${greedy_dir}/captions_eval_results.json"
echo "  deact result:        ${deact_dir}/captions_eval_results.json"

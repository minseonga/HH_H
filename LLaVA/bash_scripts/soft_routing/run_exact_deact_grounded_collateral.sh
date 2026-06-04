#!/usr/bin/env bash
set -euo pipefail

# Run the exact teammate DEACT setting and measure grounded-object collateral
# against the same greedy sample. Intended to be launched from the repository
# root on the server:
#
#   cd ~/Hallucination-Attribution
#   GPU_ID=6 NUM_SAMPLES=500 bash LLaVA/bash_scripts/soft_routing/run_exact_deact_grounded_collateral.sh
#
# Optional:
#   STATIC_EVAL_JSON=/path/to/static/captions_eval_results.json ...

repo_root="${REPO_ROOT:-$(pwd)}"
llava_dir="${LLAVA_DIR:-${repo_root}/LLaVA}"
adhh_llava_dir="${ADHH_LLAVA_DIR:-${repo_root}/ADHH/LLaVA}"
python_bin="${PYTHON_BIN:-python}"
gpu_id="${GPU_ID:-0}"

model_path="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
dataset="${DATASET:-coco}"
num_samples="${NUM_SAMPLES:-500}"
seed="${SEED:-42}"
image_folder="${IMAGE_FOLDER:-/home/kms/data/pope/val2014}"
annotation_dir="${ANNOTATION_DIR:-/home/kms/data/images/mscoco/annotations}"
caption_file_path="${CAPTION_FILE_PATH:-${annotation_dir}/captions_val2014.json}"

topk="${TOPK:-100}"
dynamic_strength="${DYNAMIC_STRENGTH:-1.0}"
dynamic_context_mode="${DYNAMIC_CONTEXT_MODE:-ratio_exp}"
dynamic_exp_sharpness="${DYNAMIC_EXP_SHARPNESS:-8.0}"
dynamic_tau="${DYNAMIC_TAU:-0.90}"
dynamic_redistribute="${DYNAMIC_REDISTRIBUTE:-renorm}"
dynamic_score_power="${DYNAMIC_SCORE_POWER:-1.0}"
head_score_key="${HEAD_SCORE_KEY:-global__itext_all__C_toi_HminusG}"
head_score_normalize="${HEAD_SCORE_NORMALIZE:-rank_percentile}"
head_file="${HEAD_FILE:-${adhh_llava_dir}/results_summary/coco/ranked_heads_global__itext_all__C_toi_HminusG.json}"

tag="${TAG:-exact_deact_l9_l16_k${topk}_${dynamic_context_mode}_s${dynamic_strength}_q${dynamic_exp_sharpness}_tau${dynamic_tau}_n${num_samples}_seed${seed}}"
output_root="${OUTPUT_ROOT:-${llava_dir}/results/coco/${tag}}"
sample_id_file="${SAMPLE_ID_FILE:-${output_root}/sample_ids_seed${seed}_n${num_samples}.json}"
greedy_dir="${output_root}/greedy"
deact_dir="${output_root}/dynamic_l9_l16_k${topk}_${dynamic_context_mode}_s${dynamic_strength}_q${dynamic_exp_sharpness}_tau${dynamic_tau}"
collateral_dir="${output_root}/grounded_collateral_dynamic"
unique_dir="${output_root}/unique_object_nodes_dynamic"

mkdir -p "${greedy_dir}" "${deact_dir}" "${collateral_dir}" "${unique_dir}"

if [[ ! -d "${adhh_llava_dir}" ]]; then
  echo "[error] missing ADHH LLaVA directory: ${adhh_llava_dir}" >&2
  exit 1
fi
if [[ ! -f "${head_file}" ]]; then
  echo "[error] missing exact head file: ${head_file}" >&2
  exit 1
fi

run_chair_eval() {
  local result_dir=$1
  (
    cd "${adhh_llava_dir}"
    "${python_bin}" eval_scripts/eval_utils/eval_chair.py \
      --annotation-dir "${annotation_dir}" \
      --answers-file "${result_dir}/captions.jsonl" \
      --caption_file captions_val2014.json \
      > "${result_dir}/chair.log" 2>&1
  )
}

echo "[run] greedy baseline -> ${greedy_dir}"
(
  cd "${adhh_llava_dir}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" "${python_bin}" -m eval_scripts.eval_caption_dynamic \
    --model-path "${model_path}" \
    --image-folder "${image_folder}" \
    --caption_file_path "${caption_file_path}" \
    --answers-file "${greedy_dir}/captions.jsonl" \
    --dataset "${dataset}" \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --num_samples "${num_samples}" \
    --seed "${seed}" \
    --intervention none \
    --save-sample-id-file "${sample_id_file}" \
    --resume \
    > "${greedy_dir}/decode.log" 2>&1
)
run_chair_eval "${greedy_dir}"

echo "[run] exact DEACT dynamic -> ${deact_dir}"
(
  cd "${adhh_llava_dir}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" "${python_bin}" -m eval_scripts.eval_caption_dynamic \
    --model-path "${model_path}" \
    --image-folder "${image_folder}" \
    --caption_file_path "${caption_file_path}" \
    --answers-file "${deact_dir}/captions.jsonl" \
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
    --use-head-scores \
    --log-intervention-stats \
    --sample-id-file "${sample_id_file}" \
    --resume \
    > "${deact_dir}/decode.log" 2>&1
)
run_chair_eval "${deact_dir}"

echo "[analyze] greedy -> exact DEACT grounded collateral"
(
  cd "${repo_root}"
  "${python_bin}" LLaVA/eval_scripts/soft_routing/build_adhh_removal_loss_disappear_figure.py \
    --base "${greedy_dir}/captions_eval_results.json" \
    --target "${deact_dir}/captions_eval_results.json" \
    --base-name "greedy" \
    --target-name "exact_deact_l9_l16_k${topk}_${dynamic_context_mode}_s${dynamic_strength}_q${dynamic_exp_sharpness}" \
    --output-dir "${collateral_dir}"

  "${python_bin}" LLaVA/eval_scripts/soft_routing/analyze_unique_object_node_transitions.py \
    --base "${greedy_dir}/captions_eval_results.json" \
    --target "${deact_dir}/captions_eval_results.json" \
    --base-name "greedy" \
    --target-name "exact_deact_l9_l16_k${topk}_${dynamic_context_mode}_s${dynamic_strength}_q${dynamic_exp_sharpness}" \
    --output-dir "${unique_dir}"
)

static_eval_json="${STATIC_EVAL_JSON:-}"
if [[ -n "${static_eval_json}" ]]; then
  static_dir="${output_root}/grounded_collateral_static"
  static_unique_dir="${output_root}/unique_object_nodes_static"
  compare_dir="${output_root}/grounded_node_outcome_static_vs_exact_deact"
  mkdir -p "${static_dir}" "${static_unique_dir}" "${compare_dir}"
  echo "[analyze] greedy -> static grounded collateral"
  (
    cd "${repo_root}"
    "${python_bin}" LLaVA/eval_scripts/soft_routing/build_adhh_removal_loss_disappear_figure.py \
      --base "${greedy_dir}/captions_eval_results.json" \
      --target "${static_eval_json}" \
      --base-name "greedy" \
      --target-name "static_hard" \
      --output-dir "${static_dir}"

    "${python_bin}" LLaVA/eval_scripts/soft_routing/analyze_unique_object_node_transitions.py \
      --base "${greedy_dir}/captions_eval_results.json" \
      --target "${static_eval_json}" \
      --base-name "greedy" \
      --target-name "static_hard" \
      --output-dir "${static_unique_dir}"

    "${python_bin}" LLaVA/eval_scripts/soft_routing/build_grounded_node_outcome_comparison.py \
      --summary "static:${static_dir}/adhh_removal_loss_summary.json" \
      --summary "exact DEACT:${collateral_dir}/adhh_removal_loss_summary.json" \
      --output-dir "${compare_dir}"
  )
fi

echo "[done] outputs"
echo "  greedy:   ${greedy_dir}/captions_eval_results.json"
echo "  deact:    ${deact_dir}/captions_eval_results.json"
echo "  summary:  ${collateral_dir}/adhh_removal_loss_summary.json"
echo "  rows:     ${collateral_dir}/adhh_removal_loss_rows.csv"
echo "  unique:   ${unique_dir}/unique_object_node_transition_summary.json"

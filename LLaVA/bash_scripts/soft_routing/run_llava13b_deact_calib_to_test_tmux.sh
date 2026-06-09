#!/usr/bin/env bash
set -euo pipefail

# Tmux launcher for the 13B calibration-to-test runner.
# It defaults to a single visible calibration job instead of hidden chunk logs.
#
# Run from Hallucination-Attribution on the server:
#   GPU_LIST="6" NUM_SAMPLES=500 TRAIN_NUM_SAMPLES=500 \
#     bash LLaVA/bash_scripts/soft_routing/run_llava13b_deact_calib_to_test_tmux.sh
#
# Attach later:
#   tmux attach -t llava13b_deact

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MAIN_SCRIPT="${REPO_ROOT}/LLaVA/bash_scripts/soft_routing/run_llava13b_deact_calib_to_test.sh"

if ! command -v tmux >/dev/null 2>&1; then
  echo "[error] tmux is not installed or not on PATH" >&2
  exit 1
fi

if [[ ! -f "${MAIN_SCRIPT}" ]]; then
  echo "[error] missing runner: ${MAIN_SCRIPT}" >&2
  exit 1
fi

session="${TMUX_SESSION:-llava13b_deact}"
attach="${TMUX_ATTACH:-1}"
job_dir="${TMUX_JOB_DIR:-${REPO_ROOT}/LLaVA/results/coco/tmux_jobs}"
job_script="${job_dir}/${session}_run.sh"
mkdir -p "${job_dir}"

if tmux has-session -t "${session}" 2>/dev/null; then
  echo "[error] tmux session already exists: ${session}" >&2
  echo "[hint] attach with: tmux attach -t ${session}" >&2
  echo "[hint] or choose another name: TMUX_SESSION=${session}_2 bash ${BASH_SOURCE[0]}" >&2
  exit 1
fi

write_export_if_set() {
  local name=$1
  if [[ -n "${!name+x}" ]]; then
    printf 'export %s=%q\n' "${name}" "${!name}" >> "${job_script}"
  fi
}

cat > "${job_script}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"

# Tmux-visible defaults. Override these by passing env vars to this launcher.
export STREAM_LOGS=1
export NUM_CHUNKS="\${NUM_CHUNKS:-1}"
export GPU_LIST="\${GPU_LIST:-6}"
EOF

for var in \
  ADHH_ROOT ADHH_LLAVA_DIR PYTHON_BIN \
  MODEL_NAME MODEL_PATH DATASET SEED NUM_SAMPLES TRAIN_NUM_SAMPLES MAX_NEW_TOKENS \
  ANNOTATION_DIR IMAGE_FOLDER CAPTION_FILE_PATH CALIB_IMAGE_FOLDER CALIB_CAPTION_FILE_PATH \
  NUM_LAYERS NUM_HEADS ALL_LAYER_SPEC SELECTION_LAYERS \
  TOPK HEAD_SCORE_KEY HEAD_SCORE_NORMALIZE MIN_HEAD_BACK_RAW USE_HEAD_SCORES \
  INTERVENTION DYNAMIC_CONTEXT_MODE DYNAMIC_STRENGTH DYNAMIC_EXP_SHARPNESS DYNAMIC_TAU \
  DYNAMIC_SCORE_POWER DYNAMIC_REDISTRIBUTE DYNAMIC_RENORM DYNAMIC_LATE_BOOST_START \
  DYNAMIC_LATE_BOOST_END DYNAMIC_LATE_BOOST_MODE DYNAMIC_LATE_TAU \
  LOG_DYNAMIC_TRACE DYNAMIC_TRACE_TOPN DYNAMIC_TRACE_EVERY \
  RESULTS_ROOT CALIB_RESULT_PATH CALIB_EXISTING_SAMPLE_FILE TXTATTN_TRACE_MODE KEEP_MERGED_TRACE \
  RUN_CALIBRATION RUN_HEAD_BUILD RUN_GREEDY RUN_DEACT RUN_TEST RESUME DRY_RUN \
  SAMPLE_ID_FILE GREEDY_DIR DEACT_DIR HEAD_FILE
do
  write_export_if_set "${var}"
done

cat >> "${job_script}" <<EOF

echo "[tmux] session=${session}"
echo "[tmux] GPU_LIST=\${GPU_LIST} NUM_CHUNKS=\${NUM_CHUNKS}"
echo "[tmux] running: ${MAIN_SCRIPT}"
bash "${MAIN_SCRIPT}"
status=\$?
echo
echo "[tmux] finished with status \${status}"
echo "[tmux] press Ctrl-b d to detach, or exit this shell to close the session"
exec bash
EOF

chmod +x "${job_script}"

tmux new-session -d -s "${session}" -n run "bash '${job_script}'"

echo "[tmux] started session: ${session}"
echo "[tmux] job script: ${job_script}"
echo "[tmux] attach: tmux attach -t ${session}"

if [[ "${attach}" == "1" || "${attach}" == "true" ]]; then
  tmux attach -t "${session}"
fi

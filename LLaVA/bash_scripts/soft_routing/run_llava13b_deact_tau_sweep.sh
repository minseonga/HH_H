#!/usr/bin/env bash
set -euo pipefail

# Tau sweep for LLaVA-1.5-13B DEACT with a fixed layer window.
# The sweep sets DYNAMIC_TAU == DYNAMIC_LATE_TAU by default so that the
# experiment isolates the threshold value rather than late-threshold decay.
#
# Example:
#   GPU_LIST="6" \
#   NUM_SAMPLES=500 \
#   TRAIN_NUM_SAMPLES=500 \
#   SELECTION_LAYERS="9:16" \
#   TAUS="0.78 0.80 0.82 0.84 0.86" \
#   bash LLaVA/bash_scripts/soft_routing/run_llava13b_deact_tau_sweep.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BASE_RUNNER="${SCRIPT_DIR}/run_llava13b_deact_calib_to_test.sh"

if [[ ! -x "${BASE_RUNNER}" ]]; then
  echo "[error] missing base runner: ${BASE_RUNNER}" >&2
  exit 1
fi

ADHH_ROOT="${ADHH_ROOT:-${REPO_ROOT}/ADHH}"
ADHH_LLAVA_DIR="${ADHH_LLAVA_DIR:-${ADHH_ROOT}/LLaVA}"
RESULTS_ROOT="${RESULTS_ROOT:-${ADHH_LLAVA_DIR}/results_deact}"
DATASET="${DATASET:-coco}"
MODEL_NAME="${MODEL_NAME:-llava-v1.5-13b}"
NUM_SAMPLES="${NUM_SAMPLES:-500}"
TRAIN_NUM_SAMPLES="${TRAIN_NUM_SAMPLES:-500}"
SEED="${SEED:-42}"
TOPK="${TOPK:-100}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
SELECTION_LAYERS="${SELECTION_LAYERS:-9:16}"
TAUS="${TAUS:-0.78 0.80 0.82 0.84 0.86 0.88 0.90}"
DYNAMIC_EXP_SHARPNESS="${DYNAMIC_EXP_SHARPNESS:-8.0}"
SWEEP_RUN_CALIBRATION="${SWEEP_RUN_CALIBRATION:-1}"
SWEEP_RUN_HEAD_BUILD="${SWEEP_RUN_HEAD_BUILD:-1}"

export ADHH_ROOT ADHH_LLAVA_DIR RESULTS_ROOT DATASET MODEL_NAME
export NUM_SAMPLES TRAIN_NUM_SAMPLES SEED TOPK MAX_NEW_TOKENS SELECTION_LAYERS
export DYNAMIC_EXP_SHARPNESS

MODEL_ROOT="${RESULTS_ROOT}/${DATASET}/${MODEL_NAME}"
GREEDY_EVAL="${MODEL_ROOT}/baselines/greedy/tok${MAX_NEW_TOKENS}/n${NUM_SAMPLES}_seed${SEED}/captions_eval_results.json"
SUMMARY_CSV="${MODEL_ROOT}/llava13b_deact_tau_sweep_${SELECTION_LAYERS//:/_}.csv"

echo "[config] selection layers: ${SELECTION_LAYERS}"
echo "[config] taus: ${TAUS}"
echo "[config] summary: ${SUMMARY_CSV}"

first=1
for tau in ${TAUS}; do
  echo
  echo "========== [tau-sweep] tau=${tau}, late_tau=${tau} =========="
  if [[ "${first}" == "1" ]]; then
    run_calibration="${SWEEP_RUN_CALIBRATION}"
    run_head_build="${SWEEP_RUN_HEAD_BUILD}"
  else
    run_calibration=0
    run_head_build=0
  fi

  if [[ -f "${GREEDY_EVAL}" ]]; then
    run_greedy=0
    echo "[tau-sweep] reusing greedy: ${GREEDY_EVAL}"
  else
    run_greedy=1
    echo "[tau-sweep] greedy not found; will run greedy once"
  fi

  DYNAMIC_TAU="${tau}" \
  DYNAMIC_LATE_TAU="${tau}" \
  RUN_CALIBRATION="${run_calibration}" \
  RUN_HEAD_BUILD="${run_head_build}" \
  RUN_GREEDY="${run_greedy}" \
  RUN_DEACT="${RUN_DEACT:-1}" \
  RUN_TEST="${RUN_TEST:-1}" \
  RESUME="${RESUME:-1}" \
  bash "${BASE_RUNNER}"

  first=0
done

python - <<'PY'
import csv
import json
import os
from pathlib import Path

model_root = Path(os.environ["RESULTS_ROOT"]) / os.environ.get("DATASET", "coco") / os.environ.get("MODEL_NAME", "llava-v1.5-13b")
num_samples = os.environ.get("NUM_SAMPLES", "500")
seed = os.environ.get("SEED", "42")
topk = os.environ.get("TOPK", "100")
max_new_tokens = os.environ.get("MAX_NEW_TOKENS", "128")
selection_layers = os.environ.get("SELECTION_LAYERS", "9:16")
selection_slug = "l" + selection_layers.replace(":", "_l").replace("-", "_l")
summary_csv = model_root / f"llava13b_deact_tau_sweep_{selection_layers.replace(':', '_')}.csv"

rows = []
greedy_path = model_root / "baselines" / "greedy" / f"tok{max_new_tokens}" / f"n{num_samples}_seed{seed}" / "captions_eval_results.json"
if greedy_path.exists():
    data = json.load(open(greedy_path, encoding="utf-8"))
    m = data.get("overall_metrics", {})
    bleu = m.get("Bleu", [""])
    rows.append({
        "tau": "greedy",
        "method": "greedy",
        "CHAIRs": m.get("CHAIRs", ""),
        "CHAIRi": m.get("CHAIRi", ""),
        "Bleu1": bleu[0] if isinstance(bleu, list) and bleu else bleu,
        "METEOR": m.get("METEOR", ""),
        "CIDEr": m.get("CIDEr", ""),
        "ObjectPrecision": m.get("ObjectPrecision", ""),
        "ObjectRecall": m.get("ObjectRecall", ""),
        "ObjectF1": m.get("ObjectF1", ""),
        "result_dir": str(greedy_path.parent),
    })

pattern = f"{selection_slug}/k{topk}/*/tok*/q*_tau*-*_n{num_samples}_seed{seed}/captions_eval_results.json"
for path in sorted((model_root / "main").glob(pattern)):
    try:
        data = json.load(open(path, encoding="utf-8"))
    except Exception:
        continue
    m = data.get("overall_metrics", {})
    bleu = m.get("Bleu", [""])
    run_name = path.parent.name
    rows.append({
        "tau": run_name,
        "method": f"deact_{selection_slug}_k{topk}_{run_name}",
        "CHAIRs": m.get("CHAIRs", ""),
        "CHAIRi": m.get("CHAIRi", ""),
        "Bleu1": bleu[0] if isinstance(bleu, list) and bleu else bleu,
        "METEOR": m.get("METEOR", ""),
        "CIDEr": m.get("CIDEr", ""),
        "ObjectPrecision": m.get("ObjectPrecision", ""),
        "ObjectRecall": m.get("ObjectRecall", ""),
        "ObjectF1": m.get("ObjectF1", ""),
        "result_dir": str(path.parent),
    })

fields = ["tau", "method", "CHAIRs", "CHAIRi", "Bleu1", "METEOR", "CIDEr", "ObjectPrecision", "ObjectRecall", "ObjectF1", "result_dir"]
with open(summary_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print(f"[summary] {summary_csv}")
for row in rows:
    print(row)
PY

echo "[done] tau sweep"

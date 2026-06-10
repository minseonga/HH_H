#!/usr/bin/env bash
set -euo pipefail

# Coarse layer-window search for LLaVA-1.5-13B DEACT.
# This wrapper calls run_llava13b_deact_calib_to_test.sh once per window,
# building a window-local txt-attention calibration/head pool and evaluating
# DEACT on the same fixed validation sample IDs.
#
# Example:
#   GPU_LIST="6 7" \
#   NUM_CHUNKS=2 \
#   NUM_SAMPLES=500 \
#   TRAIN_NUM_SAMPLES=500 \
#   LAYER_WINDOWS="9:16 11:20 13:20 17:24" \
#   bash LLaVA/bash_scripts/soft_routing/run_llava13b_deact_layer_search.sh

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
LAYER_WINDOWS="${LAYER_WINDOWS:-9:16 11:20 13:20 17:24 21:28}"

export ADHH_ROOT ADHH_LLAVA_DIR RESULTS_ROOT DATASET MODEL_NAME
export NUM_SAMPLES TRAIN_NUM_SAMPLES SEED TOPK MAX_NEW_TOKENS

MODEL_ROOT="${RESULTS_ROOT}/${DATASET}/${MODEL_NAME}"
GREEDY_EVAL="${MODEL_ROOT}/baselines/greedy/tok${MAX_NEW_TOKENS}/n${NUM_SAMPLES}_seed${SEED}/captions_eval_results.json"
SUMMARY_CSV="${MODEL_ROOT}/llava13b_deact_layer_search_summary.csv"

echo "[config] layer windows: ${LAYER_WINDOWS}"
echo "[config] model root: ${MODEL_ROOT}"
echo "[config] summary: ${SUMMARY_CSV}"

for window in ${LAYER_WINDOWS}; do
  echo
  echo "========== [layer-search] SELECTION_LAYERS=${window} =========="
  if [[ -f "${GREEDY_EVAL}" ]]; then
    run_greedy=0
    echo "[layer-search] reusing greedy: ${GREEDY_EVAL}"
  else
    run_greedy=1
    echo "[layer-search] greedy not found; will run greedy once"
  fi

  SELECTION_LAYERS="${window}" \
  RUN_GREEDY="${run_greedy}" \
  RUN_CALIBRATION="${RUN_CALIBRATION:-1}" \
  RUN_HEAD_BUILD="${RUN_HEAD_BUILD:-1}" \
  RUN_DEACT="${RUN_DEACT:-1}" \
  RUN_TEST="${RUN_TEST:-1}" \
  RESUME="${RESUME:-1}" \
  bash "${BASE_RUNNER}"
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
summary_csv = model_root / "llava13b_deact_layer_search_summary.csv"

rows = []
greedy_path = model_root / "baselines" / "greedy" / f"tok{os.environ.get('MAX_NEW_TOKENS', '128')}" / f"n{num_samples}_seed{seed}" / "captions_eval_results.json"
if greedy_path.exists():
    data = json.load(open(greedy_path, encoding="utf-8"))
    m = data.get("overall_metrics", {})
    bleu = m.get("Bleu", [""])
    rows.append({
        "window": "greedy",
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

for path in sorted((model_root / "main").glob(f"*/k{topk}/*/tok*/q*_n{num_samples}_seed{seed}/captions_eval_results.json")):
    try:
        data = json.load(open(path, encoding="utf-8"))
    except Exception:
        continue
    m = data.get("overall_metrics", {})
    bleu = m.get("Bleu", [""])
    parts = path.parts
    try:
        main_idx = parts.index("main")
        window = parts[main_idx + 1]
    except Exception:
        window = path.parent.name
    rows.append({
        "window": window,
        "method": f"deact_{window}_k{topk}",
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

fields = ["window", "method", "CHAIRs", "CHAIRi", "Bleu1", "METEOR", "CIDEr", "ObjectPrecision", "ObjectRecall", "ObjectF1", "result_dir"]
summary_csv.parent.mkdir(parents=True, exist_ok=True)
with open(summary_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print(f"[summary] {summary_csv}")
for row in rows:
    print(row)
PY

echo "[done] layer search"

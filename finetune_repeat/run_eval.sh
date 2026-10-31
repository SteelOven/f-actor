#!/usr/bin/env bash
# Generate the micro-run test dialogues with one model (loads it once): run_eval.sh <tag> [model dir; default: the released model]
# TAIL=0.1 LOOKAHEAD=1.5 run_eval.sh <tag> ... changes the prefix tail / look-ahead (see make_eval_configs.sh).
# 18 dialogues x ~2 min on the laptop CPU. Resumable (finished dialogues are skipped).
#   finetune_repeat/run_eval.sh base
#   finetune_repeat/run_eval.sh tuned ../oov-data/finetune/micro_lora/final
set -euo pipefail
TAG=${1:?usage: run_eval.sh <tag> [model dir]}
cd "$(dirname "$0")/.."
[[ -n ${2:-} ]] && export FACTOR_MODEL_DIR=$(readlink -f "$2")
BASE=../oov-data/finetune/micro_eval
[[ -d $BASE/eval_confs/$TAG ]] || finetune_repeat/make_eval_configs.sh "$TAG" "${TAIL:-0.3}" "${LOOKAHEAD:-0}"
.venv/bin/python training/run_oov_batch.py --configs-glob "$BASE/eval_confs/$TAG/*/*_r*.json" 2>&1 | tee "$BASE/run_$TAG.log" | grep -E "^\[|Done|Error|Traceback"

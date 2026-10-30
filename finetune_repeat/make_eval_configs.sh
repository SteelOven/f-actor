#!/usr/bin/env bash
# Test configs for the micro-run, one set per model tag (base / tuned): the prefix is Tom's request only, in the training prompt format,
# with the same plain narratives as in training. Groups: train (2 of the 5 training words), heldout (3 words never in training), oov
# (4 OOV words with a P4 request clip). 2 repeats, 150 frames.
#   finetune_repeat/make_eval_configs.sh base
set -euo pipefail
TAG=${1:?usage: make_eval_configs.sh <tag>}
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
BASE=../oov-data/finetune/micro_eval
P5=../oov-data/checks/p5_demo_request
P4=../oov-data/checks/p4_repeat
gen() {  # group run-dir words...
  local group=$1 run=$2; shift 2
  $PY scripts/analysis/generate_oov_configs.py --words "$@" --repeats 2 --speakers Tom,Gweneth --backchannels 0 --interruptions 0 \
    --prompt-format training --max-length 150 --prefix-run-dir "$run" --prefix-tail-s 0.3 \
    --narrative-explainer "You ask {partner} to repeat words after you, one word at a time." \
    --narrative-listener "{partner} will say a word. Listen to it and say the same word back to {partner}." \
    --outdir "$BASE/eval_confs/$TAG/$group" --audio-dir "$BASE/eval_out/$TAG/$group" | tail -1
}
gen train  $P5 doctor music
gen heldout $P5 water kitchen hello
gen oov    $P4 archband adherer appalachia anthropophagous

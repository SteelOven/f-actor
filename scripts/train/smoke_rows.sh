#!/usr/bin/env bash
# Smoke test of the request -> reply fine-tune, inside the container (interactive: ~/personaplex/cluster/cluster_shell.sh, GPU node).
# Stages (run one, several or all; a finished stage is cheap to rerun):
#   align   forced alignment of the reply clips (CPU, ~4 s per clip, needs $WS/align-venv; resumes, skips unalignable clips)
#   build   N words -> rows (train/validation); the validation words are never trained on
#   train   LoRA-free full fine-tune, few steps (finetune_rows.sh)
#   eval    request + 0.3 s tail as prefix on the validation words, base model and tuned model, then the answer score
#
#   cd /workspace/factor-code && git checkout answer-finetune
#   bash scripts/train/smoke_rows.sh align build          # first, no GPU needed
#   bash scripts/train/smoke_rows.sh train eval
#   N=50 VAL=0.2 STEPS=20 LR=1e-5 TAIL=0.3 NAME=smoke40 bash scripts/train/smoke_rows.sh ...
# Defaults: 50 words, 20 % validation = 40 training rows (10 steps per epoch at batch 1 x 4 accumulation) and 10 unseen words
# for the eval; 20 steps = 2 epochs (the earlier run's eval loss was best at ~1.6 epochs and rose after).
set -euo pipefail
cd "$(dirname "$0")/../.."
WS=${WS_DIR:-/workspace/ws}
N=${N:-50}; VAL=${VAL:-0.2}; STEPS=${STEPS:-20}; LR=${LR:-1e-5}; TAIL=${TAIL:-0.3}; NAME=${NAME:-smoke40}
REQ=$WS/checks/p6_full_request_s0_ok
REPLY=$WS/checks/p7_reply_short_s0_ok
DATA=finetune/$NAME
GATE=/workspace/oov-shared/wordlists/lists/prefix_gate_p4.csv
PY=${FACTOR_PYTHON:-python3}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

stage_align() {
  (cd /workspace/oov-shared && export TORCH_HOME=$WS/torch-hub WS \
    && $WS/align-venv/bin/python3 checks/asr_roundtrip.py align checks/configs/p7_reply_short_s0.yaml 2>&1 | grep -v Warning | tail -3)
  [[ -f $WS/checks/p7_reply_short_s0/alignment.jsonl ]] || { echo "error: align wrote no alignment.jsonl" >&2; exit 2; }
}
stage_build() {
  $PY scripts/train/build_rows_dataset.py --request-dir "$REQ" --reply-dir "$REPLY" --exclude "$GATE" \
      --limit "$N" --val-frac "$VAL" --out-dir "$WS/$DATA"
}
stage_train() { bash scripts/train/finetune_rows.sh "$NAME" "$STEPS" "$DATA" "$LR"; }
stage_eval() {
  local conf=confs/rows_test_$NAME out=outputs/rows_test_$NAME
  for who in baseline tuned; do
    rm -rf "$conf/$who"; mkdir -p "$out/$who"
    $PY scripts/analysis/make_rows_test_configs.py --rows-dir "$WS/$DATA" --request-dir "$REQ" --timings "$WS/$DATA.timings.json" \
        --out-dir "$conf/$who" --output-dir "$out/$who" --tail-s "$TAIL"
  done
  $PY training/run_oov_batch.py --configs-glob "$conf/baseline/*.json"
  FACTOR_MODEL_DIR=$WS/finetune/runs/$NAME/best_model $PY training/run_oov_batch.py --configs-glob "$conf/tuned/*.json"
  for who in baseline tuned; do
    echo "== ended_by / frames: $who"
    $PY - "$out/$who" <<'PYEOF'
import glob, json, sys
r = sorted((json.load(open(f))["generation"] for f in glob.glob(sys.argv[1] + "/*.json")), key=lambda g: g["n_generated_frames"])
print(len(r), "dialogues; ended_by:", {k: sum(g["ended_by"] == k for g in r) for k in {g["ended_by"] for g in r}},
      "| immediate stops (eos, <=2 frames):", sum(g["ended_by"] == "eos" and g["n_generated_frames"] <= 2 for g in r),
      "| frames:", [g["n_generated_frames"] for g in r])
PYEOF
  done
  for who in baseline tuned; do echo "== answer score: $who"; $PY scripts/analysis/score_rows_test.py --configs-glob "$conf/$who/*.json" | tail -15; done
}

[[ $# -gt 0 ]] || { sed -n 2,14p "$0"; exit 1; }
for st in "$@"; do echo "=== $st ($(date +%T))"; "stage_$st"; done

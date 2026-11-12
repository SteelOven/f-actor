#!/usr/bin/env bash
# Smoke test of the request -> reply fine-tune, inside the container (interactive: ~/personaplex/cluster/cluster_shell.sh, GPU node).
# Stages (run one, several or all; a finished stage is cheap to rerun):
#   align   forced alignment of the reply clips (CPU, ~4 s per clip, needs $WS/align-venv; resumes, skips unalignable clips)
#   build   N words -> rows (train/validation); the validation words are never trained on (tts_rows.build_dataset: needs the tts_rows repo at $WS/tts_rows, or TTS_ROWS=<dir>)
#   train   LoRA-free full fine-tune, few steps (finetune_rows.sh)
#   probe   P(EOS) at the first frame per word, base and tuned, with and without prefix look-ahead (after eval)
#   tf      teacher-forced check of the tuned model on the unseen (validation) words, narrative as trained / without the word / with another word
#           (does it copy the word from the request audio, or read it from the narrative, or recall a trained word?); TFMODEL=baseline for the base model, TFSPLIT=train for the trained words
#   eval    request + 0.3 s tail as prefix on the validation words, base model and tuned model, then the answer score
#           (superseded 2026-11-12 by oov-shared run_experiment.sh with factor.model + factor.rows, which records the run and
#           grades with analyze_oov.py: oov-shared README.md, "Evaluating a fine-tuned model". Kept to reproduce old numbers.)
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
TAG=${TAG:-$NAME}               # eval/probe output names (confs/rows_test_<TAG>), so a rerun does not overwrite
SPLIT=${SPLIT:-validation}      # eval words: validation (unseen) or train (the overfit test)
SILENCE1=${SILENCE1:-}           # 1: speaker 1 (Tom) is silent after the prefix, as in the training rows (opt-in)
NARR_REPLIER=${NARR_REPLIER:-}  # build: narrative of the replier (the model's role); default names the word, e.g. "{partner} asks you to listen carefully and repeat a word. You repeat it." hides it
NARR_REQUESTER=${NARR_REQUESTER:-}  # build: narrative of the requester; same placeholders {partner} {word}
KEEP_FINAL=${KEEP_FINAL:-}      # train: 1 = no early stopping, best_model/ holds the final weights (see finetune_rows.sh)
TFNARR=${TFNARR:-}              # tf: narrative variants, default "orig none swap"; "orig none" when the narrative has no word
PROMPT=${PROMPT:-default}       # default | training: narrative written as in training ("- <narrative>"), see inference_example.prepare_prompt
WHO=${WHO:-"baseline tuned"}    # which models eval generates with
REQ=${REQ:-$WS/checks/p6_full_request_s0_ok}       # merged clip dirs of the scaled set: REQ=$WS/checks/p8_request_ok REPLY=$WS/checks/p8_reply_ok
REPLY=${REPLY:-$WS/checks/p7_reply_short_s0_ok}
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
  PYTHONPATH=${TTS_ROWS:-$WS/tts_rows}/src F_ACTOR=$PWD $PY -m tts_rows.build_dataset --request-dir "$REQ" --reply-dir "$REPLY" --exclude "$GATE" \
      --limit "$N" --val-frac "$VAL" --out-dir "$WS/$DATA" \
      ${NARR_REPLIER:+--narrative-replier "$NARR_REPLIER"} ${NARR_REQUESTER:+--narrative-requester "$NARR_REQUESTER"}
}
stage_train() { bash scripts/train/finetune_rows.sh "$NAME" "$STEPS" "$DATA" "$LR"; }
stage_eval() {
  local conf=confs/rows_test_$TAG out=outputs/rows_test_$TAG
  for who in $WHO; do
    rm -rf "$conf/$who"; mkdir -p "$out/$who"
    $PY scripts/analysis/make_rows_test_configs.py --rows-dir "$WS/$DATA" --request-dir "$REQ" --timings "$WS/$DATA.timings.json" \
        --out-dir "$conf/$who" --output-dir "$out/$who" --split "$SPLIT" --tail-s "$TAIL" --lookahead-s "${LOOKAHEAD:-0}" --prompt-format "${PROMPT:-default}" ${SILENCE1:+--silence-speaker1} --repeats "${REPEATS:-1}"
  done
  [[ " $WHO " == *" baseline "* ]] && $PY training/run_oov_batch.py --configs-glob "$conf/baseline/*.json"
  [[ " $WHO " == *" tuned "* ]] && FACTOR_MODEL_DIR=$WS/finetune/runs/$NAME/best_model $PY training/run_oov_batch.py --configs-glob "$conf/tuned/*.json"
  for who in $WHO; do
    echo "== ended_by / frames: $who"
    $PY - "$out/$who" <<'PYEOF'
import glob, json, sys
r = sorted((json.load(open(f))["generation"] for f in glob.glob(sys.argv[1] + "/*.json")), key=lambda g: g["n_generated_frames"])
print(len(r), "dialogues; ended_by:", {k: sum(g["ended_by"] == k for g in r) for k in {g["ended_by"] for g in r}},
      "| immediate stops (eos, <=2 frames):", sum(g["ended_by"] == "eos" and g["n_generated_frames"] <= 2 for g in r),
      "| frames:", [g["n_generated_frames"] for g in r])
PYEOF
  done
  for who in $WHO; do echo "== answer score: $who"; $PY scripts/analysis/score_rows_test.py --configs-glob "$conf/$who/*.json" | tail -15; done
}

stage_probe() {   # P(EOS) at the first generated frame per word, base and tuned model, with and without prefix look-ahead (needs eval's configs)
  local conf=confs/rows_test_$TAG
  [[ -d $conf/tuned ]] || { echo "error: run the eval stage first ($conf/tuned missing)" >&2; exit 2; }
  for who in baseline tuned; do
    echo "== probe: $who"
    local md=""; [[ $who == tuned ]] && md=$WS/finetune/runs/$NAME/best_model
    FACTOR_MODEL_DIR=$md $PY scripts/analysis/eos_probe.py "$conf/$who/*.json" raw:0 0.3:0 0.3:1.5 raw:1.5 2>&1 | grep -v -i "warn" | grep -E "^(word|mean|RESULT|[a-z]+ +[0-9a-z.]+:?)" | tail -20
  done
}

stage_tf() {
  local md=$WS/finetune/runs/$NAME/best_model; [[ ${TFMODEL:-tuned} == baseline ]] && md=""
  FACTOR_MODEL_DIR=$md $PY scripts/analysis/teacher_forced_check.py --rows-dir "$WS/$DATA" --split "${TFSPLIT:-validation}" \
      --narrative ${TFNARR:-orig none swap} 2>&1 | grep -v -i "warn" | grep -E "^(########|REPLY|TOTAL|[a-z]+ )" 
}

[[ $# -gt 0 ]] || { sed -n 2,14p "$0"; exit 1; }
for st in "$@"; do echo "=== $st ($(date +%T))"; "stage_$st"; done

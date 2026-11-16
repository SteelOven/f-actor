#!/usr/bin/env bash
# Fine-tune the released f-actor checkpoint on rows from tts_rows (request/reply clips), one GPU, plain `python training/finetune.py`.
# The model plays the reply speaker (speaker 1: --system_speaker 1), prompted with that speaker's narrative (--use_system_narrative).
# Loss only on the system (reply) streams (--calc_loss_on_c1_only, the same half eval always scores): the request is teacher-forced at
# inference, so training on it is wasted. Without the flag, training scores both streams while eval scores only the reply.
#
#   bash scripts/train/finetune_rows.sh <run_name> <max_steps> <data> [lr=1e-5]
#   EXTRA_FLAGS="--mask_prompt_loss --word_loss_weight 10" ...   extra finetune.py flags, word-split on purpose (unquoted)
#                        --word_loss_weight weights every word token of the reply ("okay" too); --target_word_loss_weight only the row's word
#   MODEL_ID=<dir> ...   start from another checkpoint (default hf_models/f_actor), e.g. a fine-tune's out/ dir: its best_model/ is used
#                        when present (model.py). It must hold the audio heads, audio embeds and speaker projection, else they would be
#                        initialised silently; this is checked before training.
#   FULL_CHECKPOINTS=1 ...   checkpoint-N/ also holds the optimizer state (10.9 GB each, only needed to resume training). Default: weights only.
#   KEEP_FINAL=1 ...   no early stopping, best_model/ = the last weights (eval loss picks an under-trained checkpoint here: it is dominated by the audio
#                      codes of a word the model cannot copy yet, so it rises while the model learns the task; judge checkpoints with teacher_forced_check.py)
#
# Start it as a recorded run, from the login node (personaplex cluster/run.sh --kind finetune ... -- bash -c "cd /workspace/factor-code &&
# bash scripts/train/finetune_rows.sh ..."): the model then goes to that run's out/ folder. <data> is a recorded rows run
# (runs/rows/<date>-<name>, built by tts_rows cluster/build_rows.sh) or a dir below the workspace holding train.parquet +
# validation.parquet. Started by hand inside the container (cluster_shell.sh), the output goes to <ws>/data/scratch/<date>-<run_name>/out,
# which is never synced; without the env/ + data/ workspace layout (laptop, old workspace) to <ws>/finetune/runs/<run_name>.
# The architecture flags are those of scripts/train/train_nanocodec_special_tokens.sh, which the released checkpoint was trained with.
set -euo pipefail
cd "$(dirname "$0")/../.."
NAME=${1:?usage: $0 <run_name> <max_steps> <data> [lr]}
STEPS=${2:?max_steps (20 for a smoke test)}
DATA=${3:?data: a rows run (runs/rows/<date>-<name>) or a dir below the workspace (finetune/p7_short_ok)}
LR=${4:-1e-5}
MODEL_ID=${MODEL_ID:-hf_models/f_actor}
[[ -d $MODEL_ID ]] || { echo "error: MODEL_ID $MODEL_ID is not a directory" >&2; exit 2; }
${FACTOR_PYTHON:-python3} - "$MODEL_ID" <<'PY' || exit 2
import json, os, struct, sys
d = sys.argv[1]
d = os.path.join(d, "best_model") if os.path.isdir(os.path.join(d, "best_model")) else d   # as model.py picks it
idx, st = os.path.join(d, "model.safetensors.index.json"), os.path.join(d, "model.safetensors")
if os.path.isfile(idx):
    names = json.load(open(idx))["weight_map"]
elif os.path.isfile(st):
    with open(st, "rb") as f:
        names = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
else:
    sys.exit(f"error: no model.safetensors(.index.json) in {d}")
missing = [p for p in ("dsu_head", "audio_embeds.0", "speaker_embed_proj.weight") if not any(n.startswith(p) for n in names)]
if missing:
    sys.exit(f"error: {d} lacks {missing}: training would start them from random weights")
print(f"model: {d}", file=sys.stderr)
PY
WS_DIR=${WS_DIR:-/workspace/ws}
EVAL=${EVAL_STEPS:-10}   # save_steps must be a multiple of eval_steps (load_best_model_at_end)
SPEECH=$WS_DIR/$DATA; [[ -d $WS_DIR/data/$DATA/out/rows ]] && SPEECH=$WS_DIR/data/$DATA/out/rows   # a recorded rows run
[[ -f $SPEECH/train.parquet ]] || { echo "error: no train.parquet in $SPEECH (data: a rows run or a dir below $WS_DIR)" >&2; exit 2; }
if [[ -n ${OOV_RUN_DIR:-} ]]; then OUT=$OOV_RUN_DIR/out                                  # a recorded run (cluster/run.sh)
elif [[ -d $WS_DIR/data/scratch ]]; then OUT=$WS_DIR/data/scratch/$(date +%Y%m%d)-$NAME/out
  echo "note: not a recorded run, writing to $OUT (scratch). For a result, start it through personaplex cluster/run.sh." >&2
else OUT=$WS_DIR/finetune/runs/$NAME; fi

export WANDB_MODE=${WANDB_MODE:-offline} HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
${FACTOR_PYTHON:-python3} training/finetune.py \
  --model_name "$NAME" \
  --model_id "$MODEL_ID" \
  --speech_path "$SPEECH" \
  --output_dir "$OUT" \
  --preprocessing_num_workers 4 \
  --num_dsus 4 \
  --text_stream \
  --n_delay_audio_stream 2 \
  --use_speaker_embedding \
  --use_system_narrative \
  --system_speaker 1 \
  --calc_loss_on_c1_only \
  --word_alignment \
  --max_length 2048 \
  --audio_vocab_size 4032 \
  --add_bc_token \
  --add_interrupt_token \
  --learning_rate "$LR" \
  --max_steps "$STEPS" \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --eval_steps "$EVAL" \
  --save_steps "$EVAL" \
  --save_total_limit 2 \
  ${KEEP_FINAL:+--keep_final_model --early_stopping_patience 0} \
  $([[ -n ${FULL_CHECKPOINTS:-} ]] || echo --save_only_model) \
  ${EXTRA_FLAGS:-}

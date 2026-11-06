#!/usr/bin/env bash
# Fine-tune the released f-actor checkpoint on rows from tts_rows (request/reply clips), one GPU, plain `python training/finetune.py`.
# The model plays the reply speaker (speaker 1: --system_speaker 1), prompted with that speaker's narrative (--use_system_narrative).
# Loss only on the system (reply) streams (--calc_loss_on_c1_only, the same half eval always scores): the request is teacher-forced at
# inference, so training on it is wasted. Without the flag, training scores both streams while eval scores only the reply.
#
#   bash scripts/train/finetune_rows.sh <run_name> [max_steps=20] [data=finetune/p7_short_ok] [lr=1e-5]
#
# Run it inside the container (cluster_shell.sh / cluster_run.sh): $FACTOR_PYTHON and /workspace/ws are set there. <data> is a dir below
# the workspace holding train.parquet + validation.parquet (what tts_rows.rows writes). Output: <ws>/finetune/runs/<run_name>.
# The architecture flags are those of scripts/train/train_nanocodec_special_tokens.sh, which the released checkpoint was trained with.
set -euo pipefail
cd "$(dirname "$0")/../.."
NAME=${1:?usage: $0 <run_name> [max_steps] [data] [lr]}
STEPS=${2:-20}
DATA=${3:-finetune/p7_short_ok}
LR=${4:-1e-5}
WS_DIR=${WS_DIR:-/workspace/ws}
EVAL=${EVAL_STEPS:-10}   # save_steps must be a multiple of eval_steps (load_best_model_at_end)

export WANDB_MODE=${WANDB_MODE:-offline} HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
${FACTOR_PYTHON:-python3} training/finetune.py \
  --model_name "$NAME" \
  --model_id hf_models/f_actor \
  --speech_path "$WS_DIR/$DATA" \
  --output_dir "$WS_DIR/finetune/runs/$NAME" \
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
  --save_total_limit 2

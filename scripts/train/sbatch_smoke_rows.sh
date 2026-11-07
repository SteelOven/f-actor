#!/usr/bin/env bash
#SBATCH --job-name=smoke_rows
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=64G
#SBATCH --time=00:45:00
#SBATCH --output=%x_%j.log
#
# scripts/train/smoke_rows.sh stages as a job. Usage (login node, ~/f-actor up to date):
#   sbatch scripts/train/sbatch_smoke_rows.sh probe            # P(EOS) at the first frame of the smoke40 model, needs the eval configs
#   NAME=smoke40 sbatch --export=ALL scripts/train/sbatch_smoke_rows.sh train eval probe
set -euo pipefail
~/personaplex/cluster/cluster_run.sh bash -c "cd /workspace/factor-code && NAME='${NAME:-smoke40}' N='${N:-50}' STEPS='${STEPS:-20}' LR='${LR:-1e-5}' VAL='${VAL:-0.2}' EVAL_STEPS='${EVAL_STEPS:-10}' LOOKAHEAD='${LOOKAHEAD:-0}' TAG='${TAG:-${NAME:-smoke40}}' SPLIT='${SPLIT:-validation}' PROMPT='${PROMPT:-default}' WHO='${WHO:-baseline tuned}' bash scripts/train/smoke_rows.sh $*"

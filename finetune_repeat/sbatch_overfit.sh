#!/usr/bin/env bash
#SBATCH --job-name=ft_overfit
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.log
#
# Overfit test (finetune_repeat/check_row.py): N AdamW steps on one fixed batch of 2 synthetic + 2 real rows; the loss should fall.
# The data dir comes from finetune_repeat/make_dataset.py (made on the laptop, rsynced to $WS/finetune/<name>; it is tiny).
#
# Usage (login node, ~/f-actor checked out on the finetune-repeat branch):
#   sbatch finetune_repeat/sbatch_overfit.sh                        # $WS/finetune/p5_small, 20 steps
#   sbatch finetune_repeat/sbatch_overfit.sh finetune/p5_small 40   # dir below $WS, number of steps
set -euo pipefail
DATA=${1:-finetune/p5_small}
STEPS=${2:-20}
~/personaplex/cluster/cluster_run.sh bash -c "
  set -euo pipefail
  cd /workspace/factor-code
  HF_HUB_OFFLINE=1 \$FACTOR_PYTHON finetune_repeat/check_row.py --data /workspace/ws/$DATA --steps $STEPS --n-synthetic 2 --n-real 2
"

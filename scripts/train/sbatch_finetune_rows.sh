#!/usr/bin/env bash
#SBATCH --job-name=ft_rows
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=udtmx@student.kit.edu
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=64G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.log
#
# scripts/train/finetune_rows.sh as a job. Usage (login node, ~/f-actor up to date):
#   sbatch scripts/train/sbatch_finetune_rows.sh p7_short_ok_v1 300                       # run name, max_steps
#   sbatch scripts/train/sbatch_finetune_rows.sh p7_short_ok_v1 300 finetune/p7_short_ok 1e-5
set -euo pipefail
~/personaplex/cluster/cluster_run.sh bash -c "cd /workspace/factor-code && bash scripts/train/finetune_rows.sh $*"

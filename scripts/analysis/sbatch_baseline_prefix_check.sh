#!/usr/bin/env bash
#SBATCH --job-name=pfx_check
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.log
#
# scripts/analysis/baseline_prefix_check.sh as a job; the summary table is the end of the log. Usage (login node, ~/f-actor up to date):
#   sbatch scripts/analysis/sbatch_baseline_prefix_check.sh
#   WORDS="adherer alto" TAILS="none 0.3 1.0" sbatch --export=ALL scripts/analysis/sbatch_baseline_prefix_check.sh
set -euo pipefail
WORDS=${WORDS:-adherer alto ambling anabolic archband bacchante}
TAILS=${TAILS:-none 0.3}
~/personaplex/cluster/cluster_run.sh bash -c "cd /workspace/factor-code && WORDS='$WORDS' TAILS='$TAILS' bash scripts/analysis/baseline_prefix_check.sh"

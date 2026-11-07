#!/usr/bin/env bash
#SBATCH --job-name=cont_check
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=udtmx@student.kit.edu
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=48G
#SBATCH --time=02:15:00
#SBATCH --output=%x_%j.log
#
# scripts/analysis/prefix_continuation_check.sh as a job; the summary table is the end of the log. Usage (login node,
# ~/f-actor up to date, ~/oov-shared up to date for the gate list):
#   # optional first, no GPU: write and check the configs (fails in seconds if an alignment or word file is missing)
#   ~/personaplex/cluster/cluster_run.sh bash -c "cd /workspace/factor-code && CONFIGS_ONLY=1 bash scripts/analysis/prefix_continuation_check.sh"
#   sbatch scripts/analysis/sbatch_prefix_continuation_check.sh
#   TAILS="0.2 0.3 0.5" REPEATS=2 sbatch --export=ALL scripts/analysis/sbatch_prefix_continuation_check.sh
# Time: ~10 s per full-length dialogue, ~1 s per immediate stop, one ~2.5 min model load. Defaults: (112 + 92 words) x 3
# repeats x 1 tail = 612 dialogues ~ 1.8 h (limit 2h15). Finished dialogues are skipped, so a timeout is safe to resubmit.
set -euo pipefail
SETS=${SETS:-}
TAILS=${TAILS:-0.3}
REPEATS=${REPEATS:-3}
setsenv=""; [[ -n $SETS ]] && setsenv="SETS='$SETS'"
~/personaplex/cluster/cluster_run.sh bash -c "cd /workspace/factor-code && $setsenv TAILS='$TAILS' REPEATS='$REPEATS' bash scripts/analysis/prefix_continuation_check.sh"

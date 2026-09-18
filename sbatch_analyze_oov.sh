#!/usr/bin/env bash
#SBATCH --job-name=oov_factor_analyze
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.log
#
# CPU-only grading pass for sbatch_run_oov_batch.sh's output - mirrors
# personaplex/sbatch_analyze.sh. analyze_oov.py runs via $TOOLS_PYTHON
# (PersonaPlex's ASR tools venv - see that script's docstring for why),
# never touches the GPU, and is idempotent/cached per ASR backend (re-running
# only fills in whatever's missing), so it's safe to run again if you check
# in mid-batch.
#
# --mos-backend utmos turns on naturalness scoring (opt-in, uses
# PersonaPlex's mos_backends/utmos_backend.py via $TOOLS_PYTHON - see
# personaplex/sbatch_analyze.sh's comment on its first-use checkpoint
# download).
#
# Usage (from a login node, in ~/f-actor):
#   sbatch sbatch_analyze_oov.sh
set -euo pipefail

# enroot's NVIDIA hook expects a GPU/driver by default (see
# personaplex/CLUSTER_SETUP.md Gotchas) - this partition has none.
export NVIDIA_VISIBLE_DEVICES=void

~/personaplex/cluster/cluster_run.sh bash -c '
  cd /workspace/factor-code
  $TOOLS_PYTHON scripts/analysis/analyze_oov.py --mos-backend utmos
'

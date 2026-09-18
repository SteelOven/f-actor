#!/usr/bin/env bash
#SBATCH --job-name=oov_factor_analyze_parakeet
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.log
#
# GPU variant of sbatch_analyze_oov.sh for asr_backends/parakeet_backend.py.
# Unlike PersonaPlex's analyze_runs.py, analyze_oov.py keys grading/asr by
# --asr-backend inside each run record (oov_common.enrich_run()) - this does
# NOT overwrite the existing whisper-large-v3 grading, it just adds
# parakeet-tdt alongside it. No backup needed here.
#
# Requires nemo_toolkit[asr] + the cached parakeet-tdt checkpoint (both
# already done in $TOOLS_PYTHON's venv/$HF_HOME during the PersonaPlex-side
# smoke test - shared, since both repos point PERSONAPLEX_DIR/asr_backends/
# at the same venv and $WS).
#
# Usage (from a login node, in ~/f-actor):
#   sbatch sbatch_analyze_oov_parakeet.sh
set -euo pipefail

~/personaplex/cluster_run.sh bash -c '
  cd /workspace/factor-code
  $TOOLS_PYTHON scripts/analysis/analyze_oov.py --asr-backend parakeet-tdt --mos-backend utmos
'

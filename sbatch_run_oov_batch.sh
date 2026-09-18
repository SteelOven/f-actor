#!/usr/bin/env bash
#SBATCH --job-name=oov_factor_gen
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.log
#
# F-Actor counterpart to personaplex/sbatch_run_batch.sh: regenerates
# confs/oov/*.json + manifest.tsv from the shared oov_wordlist.csv, then runs
# every config through training/run_oov_batch.py, which loads the ~10.5GB
# model once for the whole batch (not once per config - the dominant cost at
# this scale, see that script's docstring). Runs in the same shared
# container as PersonaPlex (personaplex/cluster_run.sh) since both repos
# mount into one $WS.
#
# Note: generate_oov_configs.py fully rewrites manifest.tsv every call,
# scoped to whatever --words-file it's given. Config *files* left over from
# an earlier word list (e.g. the original 9-word oov-band pilot) stay on
# disk - run_oov_batch.py globs confs/oov/*.json directly, not the manifest,
# so it'll find and skip them (already done, harmless) - but they won't
# appear in *this* run's manifest.tsv, so compare_oov_results.py
# --factor-dir will only see the current word list's words until those are
# regenerated too.
#
# No wall-time baseline documented yet for F-Actor generation (unlike
# PersonaPlex's ~44s/probe) - --time below is a padded guess. Check the
# per-config "(Xs)" timings in this run's log and tighten it next time.
#
# Usage (from a login node, in ~/f-actor):
#   sbatch sbatch_run_oov_batch.sh
#   sbatch --time=24:00:00 --cpus-per-gpu=16 --mem-per-gpu=64G sbatch_run_oov_batch.sh   # bigger batch
set -euo pipefail

~/personaplex/cluster_run.sh bash -c '
  set -euo pipefail
  cd /workspace/factor-code
  $FACTOR_PYTHON scripts/analysis/generate_oov_configs.py \
      --words-file "$PERSONAPLEX_DIR/oov_wordlist.csv" --outdir confs/oov
  $FACTOR_PYTHON training/run_oov_batch.py --configs-glob "confs/oov/*.json"
'

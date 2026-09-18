#!/usr/bin/env bash
# Wraps the generate-configs + run-batch pair from CLUSTER_SETUP.md's "OOV
# pipeline on the cluster" section into one parametrized script, so
# sbatch_run_oov_batch.sh can take a words-file argument the same way
# personaplex/cluster/run_batch.sh takes an experiments-CSV argument -
# lets a subset (e.g. just the oov_plausible words, or the frequency-band
# words) run as its own clean job instead of always processing the full
# shared wordlist.
#
# Runs inside the container (needs $PERSONAPLEX_DIR/$FACTOR_PYTHON, both set
# by cluster_env.sh - invoke via cluster_run.sh, not directly).
#
# Usage:
#   ./run_oov_batch_from_words.sh                                    # default: full shared wordlist
#   ./run_oov_batch_from_words.sh wordlists/oov_words_50_plausible.txt   # a subset - path relative to personaplex's repo root
set -euo pipefail

WORDS_FILE="$PERSONAPLEX_DIR/${1:-wordlists/oov_wordlist.csv}"

cd /workspace/factor-code
$FACTOR_PYTHON scripts/analysis/generate_oov_configs.py \
    --words-file "$WORDS_FILE" --outdir confs/oov
$FACTOR_PYTHON training/run_oov_batch.py --configs-glob "confs/oov/*.json"

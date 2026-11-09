#!/usr/bin/env bash
# Does the released model continue after a prefix? Same words, three different request texts, with and without a tail (oov-shared/docs/experiments/prefix_log.md).
#   p4_repeat      Please listen carefully and repeat the word X.
#   p1_oov150      Please listen carefully. The word is X. Once more, the word is X.
#   p3_prefix_tom  Hey Brian, I want to tell you about X. Have you ever heard of X?
# Prints, per variant, how many dialogues ended by eos (after the first generated frame = the failure) and by max_length.
#
#   bash scripts/analysis/baseline_prefix_check.sh                 # inside the container, from /workspace/factor-code
#   WORDS="adherer alto" TAILS="none 0.3 1.0" bash scripts/analysis/baseline_prefix_check.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
WS_DIR=${WS_DIR:-/workspace/ws}
WORDS=${WORDS:-adherer alto ambling anabolic archband bacchante}
TAILS=${TAILS:-none 0.3}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
PY=${FACTOR_PYTHON:-python3}

for run in p4_repeat p1_oov150 p3_prefix_tom; do
  for tail in $TAILS; do
    name=pfxcheck_${run}_t${tail}
    tailarg=(); [[ $tail != none ]] && tailarg=(--prefix-tail-s "$tail")
    $PY scripts/analysis/generate_oov_configs.py --words $WORDS --prefix-run-dir "$WS_DIR/checks/$run" "${tailarg[@]}" \
        --outdir confs/$name --audio-dir outputs/$name > /dev/null
    $PY training/run_oov_batch.py --configs-glob "confs/$name/*.json" > /dev/null
  done
done
echo; echo "variant (request text) x tail: ended_by"
for d in outputs/pfxcheck_*; do
  printf '%-40s' "$(basename $d)"; grep -h '"ended_by"' $d/*.json | sort | uniq -c | tr -s ' ' | tr '\n' ' '; echo
done

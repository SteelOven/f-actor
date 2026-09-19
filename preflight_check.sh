#!/usr/bin/env bash
# Preflight check before submitting sbatch_run_oov_batch.sh - sibling to
# personaplex/cluster/preflight_check.sh, same 2026-09-19 incident: the
# hf_models symlink (into the real cached weights in $WS) is gitignored and
# doesn't survive a fresh git clone, which sent both oov_factor_gen jobs
# straight into a LocalEntryNotFoundError (offline mode, nothing cached at
# the hardcoded path) before touching a single word.
#
# Must run INSIDE the container - checks symlinks that only resolve there.
#
# Usage (the path must be the container-internal one - cluster_run.sh's
# argument executes inside the container, where ~ doesn't resolve to your
# real home, only /workspace/code and /workspace/factor-code are mounted):
#   ~/personaplex/cluster/cluster_run.sh /workspace/factor-code/preflight_check.sh
set -uo pipefail
FAIL=0

echo "=== HF weights cache (hf_models symlink -> \$WS/factor-hf-models) ==="
if [[ -d hf_models/f_actor && -n "$(ls -A hf_models/f_actor 2>/dev/null)" ]]; then
  echo "OK"
else
  echo "MISSING/EMPTY - fix: ln -s /workspace/ws/factor-hf-models /workspace/factor-code/hf_models"
  FAIL=1
fi

echo
echo "=== outputs/ and confs/oov/ symlinks (-> \$WS/factor-outputs, \$WS/factor-confs-oov) ==="
for d in outputs confs/oov; do
  if [[ -L "$d" && -e "$d" ]]; then
    echo "$d: OK"
  else
    echo "$d: MISSING or broken symlink - see CLUSTER_SETUP.md's Directory layout table"
    FAIL=1
  fi
done

echo
if [[ $FAIL -eq 0 ]]; then
  echo "All checks passed - safe to submit."
else
  echo "FAILURES ABOVE - fix before submitting." >&2
fi
exit $FAIL

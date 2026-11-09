#!/usr/bin/env bash
# Larger version of baseline_prefix_check.sh: with the released model, does generation continue after the P4 request
# ("Please listen carefully and repeat the word X.") plus a tail? Measures only whether it keeps going, not what is said.
# The failure it looks for: ended_by eos after <= 2 generated frames ("immediate stop", oov-shared/docs/experiments/prefix_log.md). A run that
# ends by eos after hundreds of frames, or by max_length, is a continuation.
#
# Differences to baseline_prefix_check.sh: one model load for the whole job (run_oov_batch.py once; that was ~2.5 min
# per call), many words, several repeats (sampling is unseeded, temperature 0.9, so repeats are independent draws),
# all configs generated and checked before the GPU part starts, and a summary per set with an upper bound on the rate.
#
# SETS = space-separated "<prefix run under $WS_DIR/checks>=<word file>". A word file is a csv with a `word` column, a
# jsonl with a `word` key, or one word per line. Defaults: the 112 gated OOV words of p4_repeat, and the 92 words of the
# fine-tune set (checks/p6_full_request_s0, same request text, Tom).
#
#   bash scripts/analysis/prefix_continuation_check.sh               # inside the container, from /workspace/factor-code
#   CONFIGS_ONLY=1 bash scripts/analysis/prefix_continuation_check.sh   # write and check the configs, no GPU
#   TAILS="0.2 0.3 0.5" REPEATS=2 bash scripts/analysis/prefix_continuation_check.sh
#   SETS="p4_repeat=/workspace/oov-shared/wordlists/lists/prefix_gate_p4.csv" bash ...
# Output: outputs/cont_<run>_t<tail>/<word>_r<n>_pfx.{wav,json}, outputs/cont_summary.tsv. Finished dialogues are skipped,
# so a timed-out job can be resubmitted as is.
set -euo pipefail
cd "$(dirname "$0")/../.."
WS_DIR=${WS_DIR:-/workspace/ws}
CONFS_DIR=${CONFS_DIR:-confs}
OUTS_DIR=${OUTS_DIR:-outputs}
SETS=${SETS:-"p4_repeat=/workspace/oov-shared/wordlists/lists/prefix_gate_p4.csv p6_full_request_s0=$WS_DIR/checks/p6_full_request_s0_ok/manifest.jsonl"}
TAILS=${TAILS:-0.3}
REPEATS=${REPEATS:-3}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
PY=${FACTOR_PYTHON:-python3}

# word file -> one word per line
words_of() {
  python3 - "$1" <<'EOF'
import csv, json, sys
path = sys.argv[1]
seen = []
if path.endswith(".csv"):
    rows = [r["word"] for r in csv.DictReader(open(path))]
elif path.endswith(".jsonl"):
    rows = [r["word"] for r in map(json.loads, filter(str.strip, open(path))) if not r.get("is_control")]
else:
    rows = [l.strip() for l in open(path) if l.strip() and not l.startswith("#")]
for w in rows:
    w = w.strip().lower()
    if w and w not in seen:
        seen.append(w)
print("\n".join(seen))
EOF
}

echo "== phase 1: configs ($(date +%T))"
names=()
for spec in $SETS; do
  run=${spec%%=*}; wfile=${spec#*=}
  [[ -f $wfile ]] || { echo "error: word file $wfile not found" >&2; exit 2; }
  [[ -f $WS_DIR/checks/$run/alignment.jsonl ]] || { echo "error: $WS_DIR/checks/$run/alignment.jsonl not found - run the align stage first" >&2; exit 2; }
  mkdir -p "$CONFS_DIR"
  words_of "$wfile" > "$CONFS_DIR/.words_$run.txt"
  for tail in $TAILS; do
    name=cont_${run}_t${tail}
    tailarg=(); [[ $tail != none ]] && tailarg=(--prefix-tail-s "$tail")
    $PY scripts/analysis/generate_oov_configs.py --words-file "$CONFS_DIR/.words_$run.txt" --repeats "$REPEATS" \
        --prefix-run-dir "$WS_DIR/checks/$run" "${tailarg[@]}" \
        --outdir "$CONFS_DIR/$name" --audio-dir "$OUTS_DIR/$name" > "$CONFS_DIR/.gen_$name.log"
    echo "$name: $(ls "$CONFS_DIR/$name"/*.json | wc -l) configs from $(wc -l < "$CONFS_DIR/.words_$run.txt") words ($wfile)"
    names+=("$name")
  done
done
[[ ${CONFIGS_ONLY:-0} == 1 ]] && { echo "CONFIGS_ONLY: stop before the GPU part"; exit 0; }

echo; echo "== phase 2: generation, one model load ($(date +%T))"
rc=0
for name in "${names[@]}"; do mkdir -p "$OUTS_DIR/$name"; done
# run_oov_batch.py takes one glob; the names share the prefix cont_ and the configs dir only holds these runs' dirs
$PY training/run_oov_batch.py --configs-glob "$CONFS_DIR/cont_*/*.json" || rc=$?

echo; echo "== phase 3: summary ($(date +%T))"
$PY - "$OUTS_DIR" "${names[@]}" <<'EOF'
import glob, json, math, os, statistics, sys
outs, names = sys.argv[1], sys.argv[2:]

def wilson_upper(k, n, z=1.96):
    if n == 0:
        return float("nan")
    p = k / n
    d = 1 + z * z / n
    return (p + z * z / (2 * n) + z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d

head = ["variant", "n", "immediate_stop", "imm_rate", "imm_upper95", "eos_late", "max_length", "frames_min", "frames_median", "imm_words"]
rows = []
for name in names:
    recs = []
    for f in sorted(glob.glob(os.path.join(outs, name, "*.json"))):
        g = json.load(open(f))["generation"]
        recs.append((os.path.basename(f)[:-len("_pfx.json")], g["ended_by"], g["n_generated_frames"]))
    imm = [r for r in recs if r[1] == "eos" and r[2] <= 2]
    frames = [r[2] for r in recs]
    n = len(recs)
    rows.append([name, n, len(imm), f"{len(imm)/n:.3f}" if n else "nan", f"{wilson_upper(len(imm), n):.3f}",
                 sum(1 for r in recs if r[1] == "eos" and r[2] > 2), sum(1 for r in recs if r[1] == "max_length"),
                 min(frames) if frames else "", int(statistics.median(frames)) if frames else "",
                 " ".join(r[0] for r in imm)])
with open(os.path.join(outs, "cont_summary.tsv"), "w") as f:
    f.write("\t".join(head) + "\n")
    for r in rows:
        f.write("\t".join(map(str, r)) + "\n")
print("\t".join(head))
for r in rows:
    print("\t".join(map(str, r)))
EOF
exit $rc

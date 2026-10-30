"""Build the synthetic training/validation datasets (and optionally a small real sample) from round-trip TTS clips.

    .venv/bin/python finetune_repeat/make_dataset.py --out ../oov-data/finetune/p5_small --n-train 40 --n-val 8 --real-sample 4

Clips: oov-shared round-trip runs with an alignment.jsonl, one run of requests (voice tom) and one of replies (voice gweneth),
optionally restricted to words in gate lists and minus --exclude. A dialogue = k distinct words, each a request (Tom, +5.6 dB)
followed by the reply (Gweneth, 0 dB); two rows per dialogue (see rows.py). Saved with datasets.save_to_disk as
<out>/train, <out>/validation and <out>/real_sample (real validation rows tokenized like data.py does, for the one-row test).
"""
import argparse, json, os, random, sys
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
import torch
from datasets import Dataset
from transformers import AutoTokenizer
from rows import _ROOT, dialogue_rows

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--out", required=True)
ap.add_argument("--requests", default=f"{_ROOT}/../oov-data/checks/p5_demo_request")
ap.add_argument("--replies", default=f"{_ROOT}/../oov-data/checks/p5_demo_reply")
ap.add_argument("--gate", nargs="*", default=[], help="word lists (csv with a word column); only words in all of them")
ap.add_argument("--exclude", nargs="*", default=["coffee", "dinner", "window"])
ap.add_argument("--only", nargs="*", default=[], help="restrict the pool to these words (the rest stay held out for testing)")
ap.add_argument("--n-train", type=int, default=40); ap.add_argument("--n-val", type=int, default=8)
ap.add_argument("--min-ex", type=int, default=3); ap.add_argument("--max-ex", type=int, default=6)
ap.add_argument("--tom-gain", type=float, default=5.6); ap.add_argument("--gweneth-gain", type=float, default=0.0)
ap.add_argument("--tidy-replies", action=argparse.BooleanOptionalAction, default=True,
                help="tidy_clip on Gweneth's replies: sound before the first word cut to 0.1 s, word gaps at most 0.2 s, 40 ms fade-out")
ap.add_argument("--end-tail-s", type=float, default=0.1,
                help="silence after the last turn of a training dialogue (the model appends EOS right after it). 0.3 s, the tail of the test prefix, taught "
                     "the micro-run 1 model 'EOS after 0.3 s of silence'; real Behavior-SD dialogues end 0.1-0.2 s after the last word")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--real-sample", type=int, default=0, help="stream this many real validation rows (needs network)")
ap.add_argument("--real-only", action="store_true", help="only (re)build real_sample, keep train/ and validation/")
args = ap.parse_args()

def clips(run, voice):
    with open(f"{run}/alignment.jsonl") as f:
        return {r["word"]: {**r, "dir": run} for r in map(json.loads, f) if r["voice"] == voice}
req, rep = clips(args.requests, "tom"), clips(args.replies, "gweneth")
pool = set(req) & set(rep) - set(args.exclude)
for path in args.gate:
    import csv
    pool &= {r["word"] for r in csv.DictReader(open(path))}
if args.only:
    missing = set(args.only) - pool
    if missing:
        sys.exit(f"error: not in the pool (no clip or gated out): {sorted(missing)}")
    pool &= set(args.only)
pool = sorted(pool); print(f"word pool ({len(pool)}): {', '.join(pool)}")

tok = AutoTokenizer.from_pretrained(f"{_ROOT}/hf_models/f_actor"); tok.pad_token = tok.eos_token
from nano_decode import load_model as load_codec
codec = load_codec(num_codebooks=4).to("cuda" if torch.cuda.is_available() else "cpu")

def build(n, offset):
    out = []
    for i in range(n):
        rng = random.Random(args.seed * 1000003 + offset + i)
        words = rng.sample(pool, min(len(pool), rng.randint(args.min_ex, args.max_ex)))
        turns = []
        for w in words:
            turns.append({"speaker": 0, "wav": f"{req[w]['dir']}/{req[w]['wav']}", "words": req[w]["words"], "gain_db": args.tom_gain})
            turns.append({"speaker": 1, "wav": f"{rep[w]['dir']}/{rep[w]['wav']}", "words": rep[w]["words"], "gain_db": args.gweneth_gain,
                          **({"tidy": {"lead_keep_s": 0.1, "max_gap_s": 0.2, "fade_s": 0.04}} if args.tidy_replies else {})})
        rows, pf = dialogue_rows(turns, codec, tok, tail_s=args.end_tail_s)
        out += rows
        print(f"  dialogue {i}: {len(words)} exchanges, {pf.n_frames} frames ({pf.seconds:.0f} s): {' '.join(words)}", flush=True)
    return Dataset.from_list(out)

os.makedirs(args.out, exist_ok=True)
if not args.real_only:
    build(args.n_train, 0).save_to_disk(f"{args.out}/train")
    build(args.n_val, 500000).save_to_disk(f"{args.out}/validation")
if args.real_sample:
    from datasets import load_dataset
    from compare_rows import real_row
    ds = load_dataset("maikezu/f-actor-behavior-sd-nanocodec", split="validation", streaming=True)
    rs = []
    for ex in ds:
        if len(rs) >= args.real_sample:
            break
        r = real_row(ex, tok)
        if r["skip"]:            # the repo's own text-fit check rejects it (data.py would filter it out too)
            continue
        rs.append({"input_ids": r["input_ids"], "attention_mask": r["attention_mask"], "labels": [-100],
                   "dsu_ids": r["dsu_ids"].tolist(), "text_stream_ids": r["text_stream_ids"].tolist(), "spk_emb": r["spk_emb"].tolist()})
    Dataset.from_list(rs).save_to_disk(f"{args.out}/real_sample")
print("saved to", args.out)

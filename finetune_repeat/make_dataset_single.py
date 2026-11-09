"""Single-exchange training/validation datasets: per word one dialogue "Tom: Please listen carefully and repeat the word X." then "Gweneth: Okay, X.".

    .venv/bin/python finetune_repeat/make_dataset_single.py --out ../oov-data/finetune/single_v1

Clips: the full TTS run's seed-0 dirs (oov-data/checks/p6_full_{request,reply}_s0, with alignment.jsonl) and its state file (oov-data/tts-local/full_state.json,
whisper-small transcripts). A word is used if its request was accepted and its reply was accepted or is a "K."-lead-in reply (Whisper heard "K. <word>").
Both clips are tidied (lead 0.1 s, word gaps <= 0.2 s, 40 ms fade-out). Each dialogue gives two rows (Tom / Gweneth as the system speaker, rows.py), so
in validation/ the even rows are Tom's, the odd ones Gweneth's (local_train.py's word-frame metric relies on that). real_sample is copied from --real-from.
"""
import argparse, json, os, random, shutil, sys
_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
import torch
from datasets import Dataset
from transformers import AutoTokenizer
from rows import _ROOT, dialogue_rows

D = os.path.abspath(f"{_ROOT}/../oov-data")
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--out", required=True)
ap.add_argument("--requests", default=f"{D}/checks/p6_full_request_s0"); ap.add_argument("--replies", default=f"{D}/checks/p6_full_reply_s0")
ap.add_argument("--state", default=f"{D}/tts-local/full_state.json")
ap.add_argument("--n-val", type=int, default=10); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--end-tail-s", type=float, default=0.1); ap.add_argument("--tom-gain", type=float, default=5.6)
ap.add_argument("--gap-range", default="0.5,0.5", help="lo,hi: the gap between request and reply (s) is drawn uniformly per dialogue (default: fixed 0.5)")
ap.add_argument("--variants", type=int, default=1, help="dialogues per train word (different gaps); validation always has one")
ap.add_argument("--real-from", default=f"{D}/finetune/micro_v3/real_sample")
args = ap.parse_args()

state = json.load(open(args.state))
def usable(kind, w):
    v = state[kind][w]
    if v["accepted"]: return True
    t = v["tries"].get("0", {}).get("text", "").lower()
    return kind == "reply" and t.startswith(("k.", "k ", "k-")) and w in t        # Whisper hears only "K." for "okay"
words = sorted(w for w in state["request"] if usable("request", w) and usable("reply", w))
def clips(run, voice):
    with open(f"{run}/alignment.jsonl") as f:
        return {r["word"]: {**r, "dir": run} for r in map(json.loads, f) if r["voice"] == voice}
req, rep = clips(args.requests, "tom"), clips(args.replies, "gweneth")
val_words = set(random.Random(args.seed).sample(words, args.n_val))
train_words = [w for w in words if w not in val_words]
print(f"{len(words)} usable words: {len(train_words)} train, {len(val_words)} validation ({sorted(val_words)})")

tidy = {"lead_keep_s": 0.1, "max_gap_s": 0.2, "fade_s": 0.04}
tok = AutoTokenizer.from_pretrained(f"{_ROOT}/hf_models/f_actor"); tok.pad_token = tok.eos_token
from nano_decode import load_model as load_codec
codec = load_codec(num_codebooks=4).to("cuda" if torch.cuda.is_available() else "cpu")

gap_lo, gap_hi = map(float, args.gap_range.split(","))
gaps_used = {}
def build(ws, split, variants=1):
    out, names, grng = [], [], random.Random(f"{args.seed}-{split}-gaps")
    ws = [w for w in ws for _ in range(variants)]
    for i, w in enumerate(ws):
        gap = round(grng.uniform(gap_lo, gap_hi), 2); gaps_used.setdefault(split, []).append(gap); names.append(w)
        turns = [{"speaker": 0, "wav": f"{req[w]['dir']}/{req[w]['wav']}", "words": req[w]["words"], "gain_db": args.tom_gain, "tidy": tidy},
                 {"speaker": 1, "wav": f"{rep[w]['dir']}/{rep[w]['wav']}", "words": rep[w]["words"], "gain_db": 0.0, "tidy": tidy}]
        rows, pf = dialogue_rows(turns, codec, tok, gap_s=gap, tail_s=args.end_tail_s)
        out += rows
        print(f"  {i + 1}/{len(ws)} {w} gap {gap}: {pf.n_frames} frames ({pf.seconds:.1f} s), overflow words {pf.overflow_words}", flush=True)
    return Dataset.from_list(out), names

os.makedirs(args.out, exist_ok=True)
rng = random.Random(args.seed + 1); order = train_words[:]; rng.shuffle(order)
ds_train, names_train = build(order, "train", args.variants); ds_train.save_to_disk(f"{args.out}/train")
ds_val, names_val = build(sorted(val_words), "validation"); ds_val.save_to_disk(f"{args.out}/validation")
if os.path.isdir(args.real_from):
    shutil.copytree(args.real_from, f"{args.out}/real_sample", dirs_exist_ok=True)
json.dump({"train": names_train, "validation": names_val, "gap_range": [gap_lo, gap_hi], "variants": args.variants, "gaps": gaps_used, "tail_s": args.end_tail_s, "tidy": tidy}, open(f"{args.out}/words.json", "w"), indent=1)
print("saved to", args.out)

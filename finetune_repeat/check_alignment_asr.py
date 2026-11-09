"""Independent check of the training rows' word timeline: decode each dialogue's rows back to audio (the real training audio, after
aligner, tidy_clip, turn offsets, frame rounding and the codec), transcribe each speaker's channel with Parakeet (word timestamps from the
TDT durations) and compare the words' start times with the rows' text stream.

    .venv/bin/python finetune_repeat/check_alignment_asr.py --data ../oov-data/finetune/single_v1/train [--limit 20] [--out report.json]

Parakeet's own resolution is ~80 ms, so only large deviations mean something. The offsets are reported relative to their median (a constant
bias is the ASR's clock, not the alignment). Words are matched in order (difflib) on the lower-cased words; unmatched ones are listed.
"""
import argparse, difflib, json, os, re, statistics, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, soundfile as sf, librosa
from datasets import load_from_disk
from rowaudio import SR, decode_pair
from nano_decode import load_model as load_codec

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--data", required=True, help="a dataset split dir (rows alternate Tom / Gweneth)"); ap.add_argument("--limit", type=int, default=0, help="dialogues (0 = all)")
ap.add_argument("--model", default="nvidia/parakeet-tdt-0.6b-v2"); ap.add_argument("--out", default=""); ap.add_argument("--tol", type=float, default=0.25, help="flag |offset - median| above this (s)")
args = ap.parse_args()

import nemo.collections.asr as nemo_asr
asr = nemo_asr.models.ASRModel.from_pretrained(args.model).eval(); codec = load_codec(num_codebooks=4)
ds = load_from_disk(args.data); n_dlg = len(ds) // 2; n_dlg = min(n_dlg, args.limit) if args.limit else n_dlg
norm = lambda w: re.sub(r"[^a-z']", "", w.lower())
tmp = tempfile.mkdtemp(prefix="asrcheck_"); results = []
for k in range(n_dlg):
    tom, gw, tw, gww = decode_pair(codec, ds, 2 * k)
    for who, audio, words in (("Tom", tom, tw), ("Gweneth", gw, gww)):
        p = f"{tmp}/{k}_{who}.wav"; sf.write(p, librosa.resample(audio, orig_sr=SR, target_sr=16000), 16000)
        hyp = asr.transcribe([p], timestamps=True, verbose=False)[0]
        heard = [(w["word"], w["start"], w["end"]) for w in hyp.timestamp["word"]]
        sm = difflib.SequenceMatcher(None, [norm(w) for w, _, _ in words], [norm(w) for w, _, _ in heard], autojunk=False)
        pairs = [(a + o, b + o) for blk in sm.get_matching_blocks() for o in range(blk.size) for a, b in [(blk.a, blk.b)]]
        res = {"dialogue": k, "speaker": who, "n_words": len(words), "n_matched": len(pairs), "text": hyp.text,
               "pairs": [{"word": words[a][0], "row_start": words[a][1], "asr_start": round(heard[b][1], 2), "dt": round(heard[b][1] - words[a][1], 2)} for a, b in pairs],
               "missed": [words[i][0] for i in range(len(words)) if i not in {a for a, _ in pairs}]}
        results.append(res); print(f"[{k + 1}/{n_dlg}] {who}: {len(pairs)}/{len(words)} matched | {hyp.text[:90]}", flush=True)
    if args.out and (k % 5 == 4 or k == n_dlg - 1):
        json.dump(results, open(args.out, "w"), indent=1)

print("\n== summary (offset = ASR start - row start, s)")
for who in ("Tom", "Gweneth"):
    rs = [r for r in results if r["speaker"] == who]; dts = [p["dt"] for r in rs for p in r["pairs"]]
    n_words = sum(r["n_words"] for r in rs)
    if not dts:
        print(who, "no matched words"); continue
    med = statistics.median(dts); dev = [abs(d - med) for d in dts]
    bad = [(r["dialogue"], p["word"], p["dt"]) for r in rs for p in r["pairs"] if abs(p["dt"] - med) > args.tol]
    print(f"{who}: {len(dts)}/{n_words} words matched; median offset {med:+.2f} s; |dev| median {statistics.median(dev):.2f}, p90 {np.percentile(dev, 90):.2f}, max {max(dev):.2f}; "
          f"{len(bad)} words beyond {args.tol} s")
    for d, w, dt in sorted(bad, key=lambda x: -abs(x[2] - med))[:8]:
        print(f"   dialogue {d}: {w!r} {dt:+.2f}")
    missed = [(r["dialogue"], w) for r in rs for w in r["missed"]]
    print(f"   not matched ({len(missed)}):", missed[:12])
if args.out:
    json.dump(results, open(args.out, "w"), indent=1)

"""Both speakers' frame-level text of a run record, side by side with each channel's level, forced frames marked.

    python scripts/analysis/show_text_frames.py scratch/forced_local/xylophone_n2.json --around-forced
    python scripts/analysis/show_text_frames.py <record>.json --from 6.0 --to 8.0

Reads generation.text_frames (inference_example.py writes it). Frame k's text is next to the audio of k * 0.08 s .. (k+1) * 0.08 s in
the record's _c1 / _c2 files (the text leads the audio by 2 frames, and decoding drops the 2 delay frames, so the two line up). The
level columns are RMS dBFS per frame, floored at -99 (digital silence is about -90); left out when the wav files are not next to the record.
Pads are abbreviated: SIL = [SILENCE_PAD], UP = [UTTERANCE_PAD], WP = [WORD_PAD]; '>>' marks a forced token.
"""

import argparse
import json
import math
import os

SHORT = {"[SILENCE_PAD]": "SIL", "[UTTERANCE_PAD]": "UP", "[WORD_PAD]": "WP"}


def levels(path, n_frames, frame_s):
    try:
        import soundfile as sf
        audio, sr = sf.read(path)
    except Exception:
        return None
    hop = int(round(frame_s * sr))
    out = []
    for k in range(n_frames):
        x = audio[k * hop:(k + 1) * hop]
        out.append(max(-99.0, 20 * math.log10(math.sqrt(sum(v * v for v in x) / len(x)) + 1e-9)) if len(x) else None)   # floor: exact zeros
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("record")
    ap.add_argument("--from", dest="t0", type=float, help="seconds")
    ap.add_argument("--to", dest="t1", type=float, help="seconds")
    ap.add_argument("--around-forced", action="store_true", help="1 s before to 3 s after the first forced word")
    args = ap.parse_args(argv)

    rec = json.load(open(args.record))
    gen = rec["generation"]
    tf = gen.get("text_frames")
    if not tf:
        raise SystemExit("no generation.text_frames in this record (written by inference_example.py since forced text was added)")
    fs = tf["frame_s"]
    n = len(tf["speaker1"]["ids"])
    forced = {(f["speaker"], f["step"] + i) for f in gen.get("forced", []) if f.get("happened") for i in range(len(f["tokens"]))}
    t0, t1 = args.t0, args.t1
    if args.around_forced:
        steps = [f["step"] for f in gen.get("forced", []) if f.get("happened")]
        if not steps:
            raise SystemExit("no forced word happened in this run")
        t0, t1 = steps[0] * fs - 1.0, steps[0] * fs + 3.0
    k0 = max(0, int((t0 or 0) / fs))
    k1 = min(n, int(math.ceil(t1 / fs)) if t1 is not None else n)

    base = os.path.dirname(os.path.abspath(args.record))
    lv = {spk: levels(os.path.join(base, rec["audio"][c]), n, fs) if c in rec.get("audio", {}) else None
          for spk, c in (("speaker1", "c1"), ("speaker2", "c2"))}
    name = {s: gen[s]["name"] for s in ("speaker1", "speaker2")}
    print(f"{'frame':>5} {'t (s)':>6}  {name['speaker1'][:18]:<20}{'dB':>6}   {name['speaker2'][:18]:<20}{'dB':>6}")
    for k in range(k0, k1):
        row = f"{k:>5} {k * fs:>6.2f}  "
        for spk in ("speaker1", "speaker2"):
            tok = tf[spk]["tokens"][k]
            tok = SHORT.get(tok, repr(tok))
            mark = ">>" if (spk, k) in forced else "  "
            db = lv[spk][k] if lv[spk] else None
            row += f"{mark}{tok[:18]:<18}{'' if db is None else f'{db:6.1f}':>6}   "
        print(row.rstrip())


if __name__ == "__main__":
    main()

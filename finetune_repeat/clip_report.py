"""Before/after numbers for tidy_clip on a round-trip run's clips (default: Gweneth's p5 replies).

    .venv/bin/python finetune_repeat/clip_report.py [--run ../oov-data/checks/p5_demo_reply --voice gweneth]

Per clip: seconds before the first word, largest gap between words, level of the sound before the first word (dBFS, - if none), and the
level of the last sound relative to the speech (dB; -30 is a natural decay, ~0 means cut off at full level).
"""
import argparse, json, os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [f"{_ROOT}/training"]
import numpy as np
from prefix import CODEC_SR, load_audio, tidy_clip

ap = argparse.ArgumentParser()
ap.add_argument("--run", default=f"{_ROOT}/../oov-data/checks/p5_demo_reply"); ap.add_argument("--voice", default="gweneth")
ap.add_argument("--lead", type=float, default=0.1); ap.add_argument("--max-gap", type=float, default=0.2); ap.add_argument("--fade", type=float, default=0.04)
args = ap.parse_args()

def db(x): return 20 * np.log10(np.sqrt((x ** 2).mean()) + 1e-10) if len(x) else float("nan")

def metrics(a, ws):
    n = int(CODEC_SR * 0.02); m = len(a) // n
    fr = np.array([db(a[i * n:(i + 1) * n]) for i in range(m)]); act = np.nonzero(fr > -50)[0]
    ref = np.median(fr[act][fr[act] > np.percentile(fr[act], 50)])
    pre = a[: max(0, int((ws[0]["start"] - 0.02) * CODEC_SR))]
    gaps = [ws[i + 1]["start"] - (ws[i]["start"] + ws[i]["dur"]) for i in range(len(ws) - 1)]
    return ws[0]["start"], max(gaps) if gaps else 0.0, db(pre) if len(pre) > n else float("nan"), fr[act[-1]] - ref

rows = [json.loads(l) for l in open(f"{args.run}/alignment.jsonl")]
print(f"{'word':9} | before: lead  maxgap  pre-sound  end   | after: lead  maxgap  pre-sound  end")
B, A = [], []
for r in rows:
    if r["voice"] != args.voice: continue
    a, _ = load_audio(f"{args.run}/{r['wav']}", 0.0)
    b = metrics(a, r["words"])
    a2, w2 = tidy_clip(a, r["words"], lead_keep_s=args.lead, max_gap_s=args.max_gap, fade_s=args.fade)
    c = metrics(a2, w2); B.append(b); A.append(c)
    f = lambda t: f"{t[0]:5.2f} {t[1]:6.2f} {t[2]:8.1f} {t[3]:6.1f}"
    print(f"{r['word']:9} |         {f(b)}  |        {f(c)}")
med = lambda X, k: np.nanmedian([x[k] for x in X])
print(f"{'median':9} |         {med(B,0):5.2f} {med(B,1):6.2f} {med(B,2):8.1f} {med(B,3):6.1f}  |        {med(A,0):5.2f} {med(A,1):6.2f} {med(A,2):8.1f} {med(A,3):6.1f}")

"""Micro-run results: per model tag and word group, how often the partner (speaker 2) says the target word.

    .venv/bin/python finetune_repeat/summarize_eval.py [../oov-data/finetune/micro_eval/eval_out]

Reads <eval_out>/<tag>/<group>/<word>_r<n>_pfx.json. "text" = the word is in speaker 2's text stream (the model's own plan);
"audible" = speaker 2 is audible in the continuation, seconds until it is (-45 dBFS, 80 ms frames); "s1" the same for speaker 1.
If analyze_oov.py has graded the runs, audio2 shows speaker 2's ASR label.
"""
import glob, json, os, sys
import numpy as np, soundfile as sf

root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "oov-data", "finetune", "micro_eval", "eval_out")

def onset(path, thr_db=-45.0, frame=0.08):
    if not os.path.exists(path): return None
    a, sr = sf.read(path); n = int(sr * frame)
    for i in range(len(a) // n):
        if 20 * np.log10(np.sqrt((a[i * n:(i + 1) * n] ** 2).mean()) + 1e-10) > thr_db: return round(i * frame, 2)
    return None

rows = []
for f in sorted(glob.glob(f"{root}/*/*/*_r*.json")):
    tag, group = f.split(os.sep)[-3:-1]
    r = json.load(open(f)); g = r["generation"]; w = r["meta"]["word"]; b = f[:-5]
    grade = next(iter((r.get("grading") or {"-": {}}).values()))
    rows.append(dict(tag=tag, group=group, word=w, rep=r["meta"]["repeat"], ended=g["ended_by"], frames=g["n_generated_frames"],
                     text=w in g["speaker2"]["transcript"].lower().split(), s1=onset(b + "_c1_cont.wav"), s2=onset(b + "_c2_cont.wav"),
                     audio2=grade.get("audio2_label", "-"), said=g["speaker2"]["transcript"][:56]))
print(f"{'tag':6} {'group':8} {'word':16} {'r':>1} {'ended':10} {'frames':>6} {'text':>5} {'s1':>5} {'s2':>5} {'audio2':10} speaker 2's text stream")
for x in rows:
    print(f"{x['tag']:6} {x['group']:8} {x['word']:16} {x['rep']:>1} {x['ended']:10} {x['frames']:>6} {str(x['text']):>5} {str(x['s1']):>5} {str(x['s2']):>5} {x['audio2']:10} {x['said']}")
print("\nper model and group: runs | target in speaker 2's text | audio2 intact | stopped early")
for tag in sorted({x["tag"] for x in rows}):
    for group in ("train", "val", "heldout", "oov"):
        c = [x for x in rows if x["tag"] == tag and x["group"] == group]
        if c: print(f"  {tag:6} {group:8}: {len(c):2d} | {sum(x['text'] for x in c):2d} | {sum(x['audio2'] == 'intact' for x in c):2d} | {sum(x['ended'] == 'eos' for x in c):2d}")

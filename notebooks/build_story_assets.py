"""Collect the small, cached inputs of notebooks/f_actor_story.py from the local oov-data/ tree (not in git).

    .venv/bin/python notebooks/build_story_assets.py

Writes notebooks/f_actor_story_assets/: data.json (parsed logs, eval runs, OOV table, row summaries), rows.npz (two real-shaped
training rows' code arrays), clips/*.ogg (trimmed audio) and MANIFEST.json (what each clip is and how it was made). Nothing is
recomputed with the model: every number is read from a file that an earlier run wrote. Re-run only to refresh the assets.
"""
import csv, datetime, glob, json, os, re, sys
import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
D = os.path.abspath(f"{ROOT}/../oov-data"); OUT = f"{HERE}/f_actor_story_assets"; os.makedirs(f"{OUT}/clips", exist_ok=True)
rel = lambda p: os.path.relpath(p, D)
mtime = lambda p: datetime.date.fromtimestamp(os.path.getmtime(p)).isoformat()

# ---------------------------------------------------------------- rows (shapes + decoded prompt/stream)
from datasets import load_from_disk
from transformers import AutoTokenizer
sys.path.insert(0, f"{ROOT}/training")
from special_tokens import TEXT_STREAM_TOKENS
tok = AutoTokenizer.from_pretrained(f"{ROOT}/hf_models/f_actor", legacy=False)
extra = {128256 + i: t for i, t in enumerate(TEXT_STREAM_TOKENS)}          # ids our rows use (the release has the first 4)
def dec(i):
    return extra.get(i) or tok.decode([i])
def summarize(r):
    ids = np.array(r["input_ids"]); m = np.array(r["attention_mask"]).astype(bool); d = np.array(r["dsu_ids"]); t = np.array(r["text_stream_ids"])
    return {"prompt": tok.decode(ids[m]), "n_prompt_tokens": int(m.sum()), "input_ids_len": len(ids), "dsu_shape": list(d.shape),
            "text_shape": list(t.shape), "spk_emb_dim": len(r["spk_emb"]), "fields": sorted(r),
            "text_tokens": [dec(int(i)) for i in t[0]]}, d, t
real = load_from_disk(f"{D}/finetune/micro_v3/real_sample"); syn = load_from_disk(f"{D}/finetune/micro_v3/train")
rows, arr = {}, {}
for name, r in (("real", real[0]), ("syn_tom", syn[0]), ("syn_gweneth", syn[1])):
    s, d, t = summarize(r); rows[name] = s; arr[f"{name}_dsu"] = d; arr[f"{name}_text"] = t
np.savez_compressed(f"{OUT}/rows.npz", **arr)
rows["sources"] = {"real": f"{rel(f'{D}/finetune/micro_v3/real_sample')}[0] (streamed from HF maikezu/f-actor-behavior-sd-nanocodec, copied by make_dataset.py)",
                   "syn": f"{rel(f'{D}/finetune/micro_v3/train')}[0:2] (finetune_repeat/make_dataset.py: CosyVoice TTS clips, NanoCodec-encoded)"}

# ---------------------------------------------------------------- training logs
def parse_log(path, audio_text_div=1.0):
    out = {"steps": [], "evals": []}
    for ln in open(path, errors="replace"):
        m = re.match(r"step (\d+)/(\d+): loss ([\d.]+) \(audio ([\d.]+), text ([\d.]+)\) lr ([\d.e+-]+) gnorm ([\d.]+)", ln)
        if m:
            s, tot, l, a, t, lr, g = m.groups(); out["steps"].append({"step": int(s), "total": int(tot), "loss": float(l), "audio": float(a) / audio_text_div,
                                                                     "text": float(t) / audio_text_div, "lr": float(lr), "gnorm": float(g)})
        m = re.match(r"\[(before training|step \d+|after training)\] synthetic val: loss ([\d.]+) \(audio ([\d.]+), text ([\d.]+)\) \| real: loss ([\d.]+) \(audio ([\d.]+), text ([\d.]+)\)", ln)
        if m:
            g = m.groups(); out["evals"].append({"at": g[0], "val_loss": float(g[1]), "val_audio": float(g[2]) / audio_text_div, "val_text": float(g[3]) / audio_text_div,
                                                  "real_loss": float(g[4]), "real_audio": float(g[5]) / audio_text_div, "real_text": float(g[6]) / audio_text_div})
        m = re.match(r"\[(before training|step \d+|after training)\] word frames(?:, (\w+))?: (.*)", ln)
        if m:                                  # "word frames, val: Tom: ..." (older log) or "word frames: val Tom: ..., train Tom: ..." (newer)
            wf = {(f"{m.group(2)} {k}" if m.group(2) else k): {"ce": float(a), "top1": float(b), "n": int(n)}
                  for k, a, b, n in re.findall(r"((?:val |train )?\w+): CE ([\d.]+), top-1 ([\d.]+) \((\d+) tokens\)", m.group(3))}
            for e in out["evals"]:
                if e["at"] == m.group(1): e["word_frames"] = wf
    out["source"] = rel(path); out["mtime"] = mtime(path)
    return out
logs = {"micro_v2_lora": parse_log(f"{D}/finetune/micro_lora/train.log", 4.0),       # this log printed audio/text 4x too large (docs/prefix_log.md)
        "single_v1_lora": parse_log(f"{D}/finetune/single_v1_train.log"),
        "single_v1_heads": parse_log(f"{D}/finetune/single_v1_heads_train.log")}
for k, v in logs.items(): print(k, len(v["steps"]), "steps,", len(v["evals"]), "evals,", "word-frame evals:", sum("word_frames" in e for e in v["evals"]))

# ---------------------------------------------------------------- eval runs
VERSIONS = {"micro_eval": ["base", "base_la", "tuned", "tuned_la", "tuned_mask"], "single_eval": ["lora", "heads", "heads_t5"]}
runs = []
for sub, tags in VERSIONS.items():
    for tag in tags:
        for f in sorted(glob.glob(f"{D}/finetune/{sub}/eval_out/{tag}/*/*_r*_pfx.json")):
            r = json.load(open(f)); g = r["generation"]; w = r["meta"]["word"]; said = g["speaker2"]["transcript"]
            runs.append({"set": sub, "tag": tag, "group": f.split(os.sep)[-2], "word": w, "repeat": r["meta"]["repeat"], "ended_by": g["ended_by"],
                         "frames": g["n_generated_frames"], "mask_stop_frames": g.get("mask_stop_frames", 0), "tail_s": r["prefix"].get("tail_s"),
                         "lookahead_s": r["prefix"].get("lookahead_s", 0.0), "prefix_text": r["prefix"]["text"], "tom": g["speaker1"]["transcript"],
                         "gweneth": said, "hit": bool(re.search(rf"\b{re.escape(w.lower())}\b", said.lower())),
                         "max_p_eos": g.get("max_p_eos_per_step"), "json": rel(f)})
from collections import Counter
for (s, t), n in sorted(Counter((r["set"], r["tag"]) for r in runs).items()):
    sub = [r for r in runs if (r["set"], r["tag"]) == (s, t)]
    print(f"{s}/{t}: {n} runs, hit {sum(r['hit'] for r in sub)}, eos-stopped {sum(r['ended_by']=='eos' for r in sub)}, eos at frame<=2: {sum(r['ended_by']=='eos' and r['frames']<=2 for r in sub)}")

# ---------------------------------------------------------------- tail-sweep table, parsed from the log
txt = open(f"{ROOT}/docs/prefix_log.md").read()
blk = txt.split("Stage 1, normalized")[1].split("Raw tail length")[0]
lines = [l.strip() for l in blk.splitlines() if l.strip().startswith("|")]
hdr = [c.strip() for c in lines[0].strip("|").split("|")]
tail = {"words": hdr[1:], "rows": {l.strip("|").split("|")[0].strip(): [float(x) for x in l.strip("|").split("|")[1:]] for l in lines[2:]},
        "source": "docs/prefix_log.md, 2026-10-29 'Stage 1' (first-frame P(any EOS), base model, p4_repeat prefix)"}

# ---------------------------------------------------------------- OOV first batch
rows_oov = list(csv.DictReader(open(f"{D}/f-actor/outputs/oov/analysis.tsv"), delimiter="\t"))
ex = {}
for w in ("schadenfreude", "algorithm"):
    f = f"{D}/f-actor/outputs/oov/{w}_r1.json"
    if os.path.exists(f):
        r = json.load(open(f)); asr = r["asr"]; name = next(iter(asr))
        ex[w] = {"band": r["meta"]["band"], "mono": r["generation"]["speaker1"]["transcript"][:400], "asr": asr[name]["transcript"][:400], "asr_backend": name,
                 "grading": r["grading"][name] if "grading" in r and name in r["grading"] else r.get("grading"), "json": rel(f), "config": r["config"]["speaker1"]["narrative"]}
oov = {"table": rows_oov, "examples": ex, "source": rel(f"{D}/f-actor/outputs/oov/analysis.tsv"), "mtime": mtime(f"{D}/f-actor/outputs/oov/analysis.tsv")}
print("oov rows", len(rows_oov), "examples", list(ex))

def parse_probe(path):
    out, split = {}, None
    for ln in open(path, errors="replace"):
        if ln.startswith("== "): split = ln[3:].strip(); out[split] = []
        m = re.match(r"row (\d+): (.*)", ln)
        if m and split:
            toks = re.findall(r"'(.*?)'->'(.*?)' \(P ([\d.]+)\)", m.group(2))
            out[split].append({"row": int(m.group(1)), "tokens": [{"target": a, "top1": b, "p_target": float(c)} for a, b, c in toks]})
    return out
probes = {"single_v1_lora": {"splits": parse_probe(f"{D}/finetune/probe_out/single_v1_lora.txt"), "source": rel(f"{D}/finetune/probe_out/single_v1_lora.txt"), "script": "finetune_repeat/word_probe.py"},
          "single_v1_heads": {"splits": parse_probe(f"{D}/finetune/probe_out/single_v1_heads.txt"), "source": rel(f"{D}/finetune/probe_out/single_v1_heads.txt"), "script": "finetune_repeat/word_probe.py"}}
import subprocess
cfg = json.load(open(f"{ROOT}/hf_models/f_actor/config.json"))
model = {k: cfg.get(k) for k in ("hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "intermediate_size", "vocab_size", "num_dsus", "text_stream",
                                  "multi_text_stream", "audio_vocab_size", "use_speaker_embedding", "calc_loss_on_c1_only", "tie_word_embeddings")}
model["source"] = "hf_models/f_actor/config.json (released checkpoint maikezu/f-actor)"
git = subprocess.run(["git", "-C", ROOT, "log", "-1", "--format=%h %s"], capture_output=True, text=True).stdout.strip()
branch = subprocess.run(["git", "-C", ROOT, "branch", "--show-current"], capture_output=True, text=True).stdout.strip()
json.dump({"rows": rows, "logs": logs, "runs": runs, "tail": tail, "oov": oov, "model": model, "probes": probes,
           "built": {"date": datetime.date.today().isoformat(), "branch": branch, "commit": git}}, open(f"{OUT}/data.json", "w"))
print("data.json", os.path.getsize(f"{OUT}/data.json") // 1024, "KB")

# ---------------------------------------------------------------- audio clips + manifest (provenance for every clip)
sys.path[:0] = [f"{ROOT}/training/inference_audio"]
import prefix as P
def stamp(p):
    return mtime(p)
def tts_meta(run):
    m = json.loads(open(f"{D}/checks/{run}/meta.jsonl").readline())
    return {"tts": m.get("tts"), "config": m.get("config"), "date": m.get("time", "")[:10], "factor_commit": m.get("repos", {}).get("factor", {}).get("commit", "")[:7]}
clips = {}
def add(name, audio, sr, title, how, source, seconds=None, **extra):
    if audio.ndim == 2 and audio.shape[1] > 6: audio = audio.T
    if seconds: audio = audio[: int(seconds * sr)]
    sf.write(f"{OUT}/clips/{name}.ogg", audio, sr, format="OGG", subtype="VORBIS")
    clips[name] = {"title": title, "how": how, "source": source, "seconds": round(len(audio) / sr, 1), "channels": 1 if audio.ndim == 1 else audio.shape[1], **extra}
def load(p):
    a, sr = sf.read(p, dtype="float32"); return a, sr

p = f"{ROOT}/example_dialogues/example_1.wav"; a, sr = load(p)
add("released_example_1", a, sr, "Released example dialogue 1 (first 25 s)", "Generated by the authors' F-Actor release with the instructions in example_dialogues/instructions_for_example_1_2_3.txt; "
    "stereo, left = speaker 1, right = speaker 2. Not generated by us.", "f-actor/example_dialogues/example_1.wav", 25, date=stamp(p))
req_a, req_sr = load(f"{D}/checks/p5_demo_request/audio/music__tom.wav")
add("tts_request_music", req_a, req_sr, "Tom's request, TTS: 'Please listen carefully and repeat the word music.'", "Synthetic. CosyVoice zero-shot voice cloning from the Behavior-SD reference voice 'tom'. "
    "Used as the first turn of every prefix; before it enters the model it gets +5.6 dB and is encoded with NanoCodec (see section 5).", "checks/p5_demo_request/audio/music__tom.wav",
    date=tts_meta("p5_demo_request")["date"], **{"tts": tts_meta("p5_demo_request")["tts"], "run_config": tts_meta("p5_demo_request")["config"]})
rep_p = f"{D}/checks/p5_demo_reply/audio/music__gweneth.wav"; rep_a, rep_sr = load(rep_p)
words = next(json.loads(l) for l in open(f"{D}/checks/p5_demo_reply/alignment.jsonl") if json.loads(l)["word"] == "music" and json.loads(l)["voice"] == "gweneth")
tm = tts_meta("p5_demo_reply")
add("tts_reply_music_raw", rep_a, rep_sr, "Gweneth's reply 'Okay, music.', raw TTS", "Synthetic. CosyVoice zero-shot voice cloning from the reference voice 'gweneth'. Note the long lead-in before 'okay' and the pause at the comma.",
    "checks/p5_demo_reply/audio/music__gweneth.wav", date=tm["date"], tts=tm["tts"], run_config=tm["config"])
wav22, _ = P.load_audio(rep_p)                                  # the builder's own loader (mono, resampled to the codec rate)
tid, tw = P.tidy_clip(wav22, words["words"], lead_keep_s=0.1, max_gap_s=0.2, fade_s=0.04)
add("tts_reply_music_tidy", tid if isinstance(tid, np.ndarray) else tid.numpy(), P.CODEC_SR, "The same reply after tidy_clip", "Computed here, deterministically, from the raw clip above by training/prefix.py tidy_clip "
    "(lead-in cut to 0.1 s, word gaps shortened to <= 0.2 s, 40 ms fade-out), the setting make_dataset.py uses for training replies. No model involved.",
    "checks/p5_demo_reply/audio/music__gweneth.wav + alignment.jsonl", date=datetime.date.today().isoformat())
def gen(name, tag_path, title, how, seconds=None):
    p = f"{D}/finetune/{tag_path}.wav"; j = json.load(open(p[:-4] + ".json")); g = j["generation"]; a, sr = load(p)
    add(name, a, sr, title, how + f" Ended by {g['ended_by']} after {g['n_generated_frames']} generated frames (80 ms each). Stereo: left = Tom (speaker 1), right = Gweneth (speaker 2); "
        f"the first {j['prefix']['seconds']:.1f} s is the teacher-forced prefix (TTS request, tail {j['prefix'].get('tail_s')} s), the rest is generated.", f"finetune/{tag_path}.wav", seconds, date=stamp(p))
gen("base_water", "micro_eval/eval_out/base/heldout/water_r1_pfx", "Released F-Actor continuing Tom's request ('water')",
    "Released checkpoint, no fine-tuning; prefix = Tom's request only, 0.3 s tail, training prompt format, 150 frames.")
gen("tuned_water", "micro_eval/eval_out/tuned/heldout/water_r1_pfx", "micro_v2 LoRA model, same prefix: stops at once",
    "LoRA r16 (micro_v2, 5 words), merged. Same prefix and settings as the base clip.")
gen("tuned_mask_water", "micro_eval/eval_out/tuned_mask/heldout/water_r1_pfx", "micro_v2 LoRA model with the stop mask",
    "As above but audio EOS and text-pad forbidden for the first 60 generated frames (mask_stop_frames), 0.1 s tail + 1.5 s look-ahead.")
gen("heads_moon", "single_eval/eval_out/heads_t5/val/moon_r1_pfx", "single_v1 Run B (LoRA + heads) on the validation word 'moon'",
    "LoRA r16 plus trainable audio embeddings/heads, trained on 69 single-exchange dialogues; prefix with 0.5 s tail.")
p = f"{D}/f-actor/outputs/oov/schadenfreude_r1.wav"; a, sr = load(p)
add("oov_schadenfreude", a, sr, "OOV probe: 'schadenfreude' (released F-Actor, no prefix)", "Generated by the released checkpoint from the explainer/listener narratives (Tom explains the word to Brian); config in "
    "outputs/oov/schadenfreude_r1.json. Stereo: left = Tom, right = Brian. First 20 s.", "f-actor/outputs/oov/schadenfreude_r1.wav", 20, date=stamp(p))
json.dump(clips, open(f"{OUT}/MANIFEST.json", "w"), indent=1)
print({k: v["seconds"] for k, v in clips.items()}, "| clips dir KB:", sum(os.path.getsize(f) for f in glob.glob(f"{OUT}/clips/*")) // 1024)

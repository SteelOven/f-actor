"""Training-data examples for the showcase, decoded from the dataset rows themselves (no re-assembly from clips): the codec codes of a
dialogue's two rows are decoded back to audio, the frame-aligned text streams give the word timeline. Per example: the full stereo dialogue
(left = Tom, right = Gweneth) and one clip per utterance of each speaker (Tom's requests, Gweneth's answers). Writes
<oov-data>/finetune/examples/*.wav and examples.json.
    .venv/bin/python finetune_repeat/make_examples.py"""
import json, os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [f"{_ROOT}/training", f"{_ROOT}/training/inference_audio"]
import numpy as np, soundfile as sf, torch
from datasets import load_from_disk
from transformers import AutoTokenizer
from nano_decode import load_model as load_codec

F = os.path.abspath(f"{_ROOT}/../oov-data/finetune"); OUT = f"{F}/examples"; os.makedirs(OUT, exist_ok=True)
from rowaudio import SR, FRAME_S, DELAY, tok, words_of, decode_pair
codec = load_codec(num_codebooks=4)

def utterances(words, split_s=1.0):
    """Group consecutive words into utterances: a new one after a gap of more than split_s."""
    us = []
    for w in words:
        if us and w[1] - us[-1][-1][2] <= split_s:
            us[-1].append(w)
        else:
            us.append([w])
    return us

def example(name, ds, i, split, extra=None):
    """Rows i (Tom is the system speaker) and i + 1 (Gweneth) of one dialogue."""
    tom_row, gw_row = ds[i], ds[i + 1]
    tom, gw, _, _ = decode_pair(codec, ds, i); n = len(tom)
    sf.write(f"{OUT}/{name}.wav", np.stack([tom, gw], 1), SR)
    ex = {"name": name, "split": split, "row_index": i, "seconds": round(n / SR, 1), "wav": f"examples/{name}.wav", "timeline": [], "tom_clips": [], "gweneth_clips": []}
    for who, row, audio, key in (("Tom", tom_row, tom, "tom_clips"), ("Gweneth", gw_row, gw, "gweneth_clips")):
        ws = words_of(row); ex["timeline"] += [{"speaker": who, "word": w, "start": a, "end": b} for w, a, b in ws]
        for k, u in enumerate(utterances(ws)):
            a, b = max(0, int((u[0][1] - 0.15) * SR)), min(n, int((u[-1][2] + 0.25) * SR)); p = f"{name}_{who.lower()}{k}.wav"
            sf.write(f"{OUT}/{p}", audio[a:b], SR); ex[key].append({"wav": f"examples/{p}", "text": " ".join(w for w, _, _ in u)})
    ex["timeline"].sort(key=lambda w: w["start"])
    ex.update(extra or {}); return ex

def prompt_of(ds, i):
    return tok.decode([t for t in ds[i]["input_ids"] if t != tok.pad_token_id])

ex = {"micro": [], "single_v1": [], "single_v2": []}
mic = load_from_disk(f"{F}/micro_v2/train"); micv = load_from_disk(f"{F}/micro_v2/validation")
print("micro_v2 rows:", len(mic), len(micv))
for k in (0, 1):
    ex["micro"].append(example(f"micro_train{k}", mic, 2 * k, "train"))
ex["micro"].append(example("micro_val0", micv, 0, "validation"))
w1 = json.load(open(f"{F}/single_v1/words.json")); w2 = json.load(open(f"{F}/single_v2/words.json"))
d1, d1v, d2 = (load_from_disk(f"{F}/{p}") for p in ("single_v1/train", "single_v1/validation", "single_v2/train"))
for w in ("designate", "divide"):
    i = 2 * w1["train"].index(w)
    ex["single_v1"].append(example(f"v1_{w}", d1, i, "train", {"gap": "0.5 (fixed)", "word": w, "prompt_gweneth_row": prompt_of(d1, i + 1), "prompt_tom_row": prompt_of(d1, i)}))
i = 2 * w1["validation"].index("moon"); ex["single_v1"].append(example("v1_val_moon", d1v, i, "validation", {"gap": "0.5 (fixed)", "word": "moon"}))
for v, j in enumerate(k for k, n in enumerate(w2["train"]) if n == "designate"):
    ex["single_v2"].append(example(f"v2_designate_{v}", d2, 2 * j, "train", {"gap": f"{w2['gaps']['train'][j]} (drawn)", "word": "designate"}))
json.dump(ex, open(f"{OUT}/examples.json", "w"), indent=1); print({k: len(v) for k, v in ex.items()})
for k in ex:
    for e in ex[k]:
        print(e["name"], e["seconds"], "s |", " / ".join(c["text"] for c in e["tom_clips"][:3]), "||", " / ".join(c["text"] for c in e["gweneth_clips"][:5]))

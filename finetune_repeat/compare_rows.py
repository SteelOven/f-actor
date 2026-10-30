"""Stage 1: what does a real training row look like, and what would a synthetic one (from the p5 clips) look like?

Streams a few validation rows of the Behavior-SD dataset, turns them into training rows with the repo's own functions in the
flags of scripts/train/train_nanocodec_special_tokens.sh (what the released model was trained with), builds one synthetic row
from the multi-turn prefix builder (Gweneth as the "system" speaker), and prints both. Nothing is written except the report.

    cd <worktree>; .venv/bin/python finetune_repeat/compare_rows.py [--n 3]
"""
import argparse, json, os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [f"{_ROOT}/training", f"{_ROOT}/training/inference_audio"]
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from dialogue_creation.get_prompt import build_prompt
from dialogue_creation.get_text_stream import adapt_to_text_stream
from dialogue_creation.utils import make_attention_mask, prepare_dsu

# flags of scripts/train/train_nanocodec_special_tokens.sh
NUM_DSUS, DELAY_AUDIO, DELAY_TEXT, MAX_LENGTH, AUDIO_DELAY_ID = 4, 2, 0, 2048, 4032

def real_row(example, tok):
    """tokenize_speech of training/data.py (training branch), with those flags and one text stream (text_stream, not multi)."""
    dsu_s, dsu_u, _, orig_len, role_map = prepare_dsu(example, NUM_DSUS, False, MAX_LENGTH, inference=False)
    dsu_ids, ts, skip, overflow = adapt_to_text_stream(
        False, np.concatenate([dsu_s, dsu_u], axis=0), DELAY_AUDIO, DELAY_TEXT, True, orig_len, example, tok, MAX_LENGTH,
        AUDIO_DELAY_ID, role_map, add_bc_token=True, add_interrupt_token=True, add_counting_tokens=False)
    prompt = build_prompt(example, max_length=MAX_LENGTH, orig_dsu_length=orig_len, role_to_speaker_map=role_map,
                          use_system_narrative=True, speech=True)
    ids = tok(prompt, truncation=True, padding="max_length", max_length=MAX_LENGTH)["input_ids"]
    emb = np.array(example["spk_emb_c1" if role_map["system"] == 0 else "spk_emb_c2"])
    return dict(prompt=prompt, input_ids=ids, attention_mask=make_attention_mask(ids, tok.pad_token_id), dsu_ids=dsu_ids,
                text_stream_ids=ts, spk_emb=emb, skip=skip, overflow=overflow, role_map=role_map, orig_len=orig_len)

def describe(name, r, tok):
    a = np.asarray(r["dsu_ids"]); t = None if r["text_stream_ids"] is None else np.asarray(r["text_stream_ids"])
    print(f"\n=== {name}")
    print("role_map:", r["role_map"], "| skip:", r["skip"], "| overflow words:", r["overflow"])
    print("prompt:\n" + "\n".join("   " + l for l in r["prompt"].split("\n")))
    print(f"input_ids: len {len(r['input_ids'])} (non-pad {sum(r['attention_mask'])}) | dsu_ids {a.shape} {a.dtype}, "
          f"delay frames at start: {(a[:, :4] == AUDIO_DELAY_ID).all(axis=0).sum()}, max code {a[a < AUDIO_DELAY_ID].max()}"
          f" | text_stream_ids {None if t is None else t.shape} | spk_emb {np.asarray(r['spk_emb']).shape}")
    if t is not None:
        ids = [i for i in t[0].tolist()]
        txt = tok.decode([i for i in ids if i not in set(tok.convert_tokens_to_ids(["[SILENCE_PAD]", "[UTTERANCE_PAD]", "[WORD_PAD]"]))])
        print("system's text stream (pads removed):", txt[:240])
        print("share of frames: text silence-pad", round(float((t[0] == tok.convert_tokens_to_ids("[SILENCE_PAD]")).mean()), 2))
    frames = a.shape[1]; print(f"frames {frames} = {frames/12.5:.1f} s")

def synthetic_row(tok):
    from types import SimpleNamespace
    from nano_decode import load_model as load_codec
    from prefix import build_multiturn_prefix
    from inference_example import prepare_prompt
    root = os.path.abspath(f"{_ROOT}/../oov-data/checks")
    rows = lambda run, voice: {r["word"]: r for r in map(json.loads, open(f"{root}/{run}/alignment.jsonl")) if r["voice"] == voice}
    req, rep = rows("p5_demo_request", "tom"), rows("p5_demo_reply", "gweneth")
    codec = load_codec(num_codebooks=4).to("cpu")
    turns = []
    for w in ("music", "movie", "kitchen"):
        turns.append({"speaker": 0, "wav": f"{root}/p5_demo_request/{req[w]['wav']}", "words": req[w]["words"], "gain_db": 5.6})
        turns.append({"speaker": 1, "wav": f"{root}/p5_demo_reply/{rep[w]['wav']}", "words": rep[w]["words"], "gain_db": 0.0})
    pf = build_multiturn_prefix(turns, codec, tok, gap_s=0.5, tail_s=0.3)
    d, t = pf.dsu[0].numpy(), pf.text[0].numpy()               # d: heads 0-3 Tom, 4-7 Gweneth; t: [Tom, Gweneth]
    sys_idx = 1                                               # Gweneth as the system speaker
    dsu_ids = np.concatenate([d[4:], d[:4]], axis=0)           # system's heads first, as dsu_s, dsu_u in data.py
    prompt = prepare_prompt("Gweneth", "Tom", "Tom will say a word. Listen to it and say the same word back to Tom.", False, 0, 0)
    ids = tok(prompt, truncation=True, padding="max_length", max_length=MAX_LENGTH)["input_ids"]
    emb = np.array(json.load(open(f"{_ROOT}/training/example_speakers/speaker_embeddings.json"))["gweneth"])
    return dict(prompt=prompt, input_ids=ids, attention_mask=make_attention_mask(ids, tok.pad_token_id), dsu_ids=dsu_ids,
                text_stream_ids=t[sys_idx][None], spk_emb=emb, skip=False, overflow=pf.overflow_words,
                role_map={"system": 1, "user": 0}, orig_len=d.shape[1])

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=3); args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(f"{_ROOT}/hf_models/f_actor"); tok.pad_token = tok.eos_token
    ds = load_dataset("maikezu/f-actor-behavior-sd-nanocodec", split="validation", streaming=True)
    for i, ex in zip(range(args.n), ds):
        print(f"\n##### real example {i}: {ex['file_name']} | speakers {ex['speakers']} | narrative(s1): {ex['new_narrative_s1'][:150]}")
        describe(f"REAL row {i}", real_row(ex, tok), tok)
    describe("SYNTHETIC row (Gweneth = system; demos music, movie; target kitchen)", synthetic_row(tok), tok)

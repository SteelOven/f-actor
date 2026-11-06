"""Request + reply TTS clips -> a fine-tune dataset (train.parquet + validation.parquet) of raw rows (training/train_rows.py).

One row per word: speaker 0 (Tom) says the request clip as it is, speaker 1 (Gweneth, the model's role in training:
--system_speaker 1) says the reply clip as it is. The clips are not changed; --end-silence-s only appends digital
silence after the reply (default 0), so the model sees what follows the answer.

    python scripts/train/build_rows_dataset.py \\
        --request-dir $WS/checks/p6_full_request_s0_ok --reply-dir $WS/checks/p7_reply_short_s0_ok --out-dir $WS/finetune/p8_rows

Each run dir needs manifest.jsonl (word, wav) and alignment.jsonl (word, wav, words[{word,start,dur}]; oov-shared's align stage).
Words missing from either dir, controls, and words without an alignment are skipped and listed. --exclude takes word files
(csv with a `word` column, jsonl, or one word per line) that must not be trained on, e.g. the held-out gated OOV list.
Narratives are the prompts of the rows: the same text must be used at generation time (generate_oov_configs.py
--narrative-explainer/--narrative-listener), otherwise the model is prompted differently than it was trained.
"""

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
for p in (_REPO / "training", _REPO / "training" / "inference_audio"):
    sys.path.insert(0, str(p))

NARRATIVE_REQUESTER = "You ask {partner} to listen carefully and repeat the word '{word}'."
NARRATIVE_REPLIER = "{partner} asks you to listen carefully and repeat the word '{word}'. You repeat it."


def read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def word_file(path):
    path = str(path)
    if path.endswith(".csv"):
        return {r["word"].strip().lower() for r in csv.DictReader(open(path))}
    if path.endswith(".jsonl"):
        return {r["word"].strip().lower() for r in read_jsonl(path)}
    return {l.strip().lower() for l in open(path) if l.strip() and not l.startswith("#")}


def clips(run_dir):
    """word -> (wav path, aligned words) for the clips of a run dir that have a manifest row and an alignment."""
    run_dir = Path(run_dir)
    wav_of = {m["word"]: run_dir / m["wav"] for m in read_jsonl(run_dir / "manifest.jsonl")
              if "wav" in m and not m.get("is_control")}
    out = {}
    for a in read_jsonl(run_dir / "alignment.jsonl"):
        if a["word"] in wav_of and a.get("words"):
            out[a["word"]] = (wav_of[a["word"]], a["words"])
    return wav_of, out


def with_end_silence(wav, seconds, tmp):
    import soundfile as sf
    import numpy as np

    audio, sr = sf.read(str(wav))
    dst = os.path.join(tmp, wav.name)
    sf.write(dst, np.concatenate([audio, np.zeros(int(round(seconds * sr)), dtype=audio.dtype)]), sr)
    return dst


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--request-dir", required=True)
    ap.add_argument("--reply-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--speakers", nargs=2, default=["Tom", "Gweneth"], help="request speaker, reply speaker (the model's role)")
    ap.add_argument("--exclude", nargs="*", default=[], help="word files that must not be trained on")
    ap.add_argument("--end-silence-s", type=float, default=0.0, help="digital silence appended after the reply clip")
    ap.add_argument("--narrative-requester", default=NARRATIVE_REQUESTER)
    ap.add_argument("--narrative-replier", default=NARRATIVE_REPLIER)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--limit", type=int, help="first N words only (smoke test)")
    args = ap.parse_args(argv)

    from nano_decode import load_model as load_codec
    from train_rows import build_row, write_dataset
    import torch

    req_wavs, req = clips(args.request_dir)
    rep_wavs, rep = clips(args.reply_dir)
    excluded = set().union(*map(word_file, args.exclude)) if args.exclude else set()
    words = sorted(set(req) & set(rep) - excluded)
    both = set(req_wavs) & set(rep_wavs)
    skipped = {"request only": sorted(set(req_wavs) - set(rep_wavs)), "reply only": sorted(set(rep_wavs) - set(req_wavs)),
               "no alignment": sorted(both - set(words) - excluded),
               "excluded": sorted(both & excluded)}
    for why, ws in skipped.items():
        if ws:
            print(f"skipped ({why}): {len(ws)}: {' '.join(ws)}")
    if args.limit:
        words = words[:args.limit]
    print(f"{len(words)} words -> rows")

    codec = load_codec(num_codebooks=4).to("cuda" if torch.cuda.is_available() else "cpu")
    t0, t1 = args.speakers
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for w in words:
            reply_wav = rep[w][0]
            if args.end_silence_s > 0:
                reply_wav = Path(with_end_silence(reply_wav, args.end_silence_s, tmp))
            rows.append(build_row(
                [{"speaker": 0, "wav": req[w][0], "words": req[w][1]},
                 {"speaker": 1, "wav": reply_wav, "words": rep[w][1]}],
                codec, speakers=[t0, t1], narrative="",
                new_narrative_s1=args.narrative_requester.format(partner=t1, word=w),
                new_narrative_s2=args.narrative_replier.format(partner=t0, word=w),
                file_name=w))
    n_train, n_val = write_dataset(rows, args.out_dir, val_frac=args.val_frac)
    Path(args.out_dir, "run.json").write_text(json.dumps({"args": vars(args), "n_train": n_train, "n_val": n_val,
                                                          "words": words}, indent=1))
    print(f"wrote {n_train} train, {n_val} validation rows to {args.out_dir}")


if __name__ == "__main__":
    main()

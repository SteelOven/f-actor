"""Build F-Actor prefixes from round-trip TTS clips and check them, without running the model.

Input: an oov-shared round-trip run dir (audio/<word>__<voice>.wav + alignment.jsonl from its `align` stage).
Per word, training/prefix.py builds the prefix; this script checks it and writes, into <out>/<word>__<voice>/:

    prefix.wav (+ _c1/_c2)  the prefix codes decoded back to audio, delay frames dropped  -> listen: check 1
    frames.tsv              one row per frame: delay flag, both text streams (decoded), the word being spoken
                            at that time per the alignment                                   -> read: checks 2 and 3
    prefix.pt               the tensors + numbers, loadable by the generation hook later
    summary.json            N, seconds, gain, overflow_words, and the asserts that passed

Asserts (a failure stops that word, the others continue): shapes; frames 0-1 are audio_delay_id on all 8 heads and
only codec codes (< 4032) come after them; speaker 2's text stream is all [SILENCE_PAD]; speaker 1's text tokens
decode to the aligned words in order.

Usage (laptop, f-actor .venv; NanoCodec and the tokenizer come from the HF cache / hf_models):
    python scripts/analysis/check_prefix.py --run-dir ../oov-data/checks/p1_oov150 --voice tom \\
        --words quokka schadenfreude water --out outputs/prefix_check
Overview: docs/oov_experiment.md, "Prefix experiment".
"""

import argparse
import json
import os
import re
import sys
import traceback

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (os.path.join(_REPO, "training"), os.path.join(_REPO, "training", "inference_audio")):
    sys.path.insert(0, p)

import torch  # noqa: E402
from prefix import (AUDIO_DELAY_ID, CODEC_CODEBOOK_SIZE, COSYVOICE_GAIN_DB, N_DELAY_AUDIO, SAMPLES_PER_FRAME,  # noqa: E402
                    CODEC_SR, build_prefix)
from special_tokens import SILENCE_PAD, TEXT_STREAM_TOKENS  # noqa: E402

FRAME_SECONDS = SAMPLES_PER_FRAME / CODEC_SR


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def spoken_in(words, t0, t1):
    """The aligned word(s) overlapping [t0, t1)."""
    return " ".join(w["word"] for w in words if w["start"] < t1 and w["start"] + w["dur"] > t0)


def text_words(tokenizer, ids):
    """Speaker 1's text stream -> its words, special tokens removed (as inference_example.remove_special_tokens)."""
    text = tokenizer.decode(ids, skip_special_tokens=False)
    text = re.sub("|".join(re.escape(t) for t in TEXT_STREAM_TOKENS), " ", text)
    return text.lower().split()


def check(prefix, row, tokenizer):
    """The asserts; returns the list of names that passed (raises AssertionError on the first failure)."""
    dsu, text, n = prefix.dsu, prefix.text, prefix.n_frames
    passed = []
    assert dsu.shape == (1, 8, n) and text.shape == (1, 2, n), f"shapes {tuple(dsu.shape)} {tuple(text.shape)}"
    passed.append("shapes")
    assert (dsu[0, :, :N_DELAY_AUDIO] == AUDIO_DELAY_ID).all(), "first frames are not audio_delay_id"
    assert (dsu[0, :, N_DELAY_AUDIO:] < CODEC_CODEBOOK_SIZE).all(), "delay/eos id inside the audio"
    passed.append("delay_frames")
    silence = tokenizer.convert_tokens_to_ids(SILENCE_PAD)
    assert (text[0, 1] == silence).all(), "speaker 2's text stream is not all [SILENCE_PAD]"
    passed.append("speaker2_text_silent")
    got = text_words(tokenizer, text[0, 0].tolist())
    want = [w["word"].lower() for w in row["words"]]
    assert got == want, f"text stream words {got} != aligned words {want}"
    passed.append("text_words_in_order")
    return passed


def frame_table(prefix, row, tokenizer):
    lines = ["frame\tt_text\tt_audio\tdelay\tc1_code0\tspeaker1_text\tspeaker2_text\tspoken_in_frame"]
    for i in range(prefix.n_frames):
        t_text = i * FRAME_SECONDS                       # text is not delayed; audio frame i holds time (i-2) * 80 ms
        t_audio = (i - N_DELAY_AUDIO) * FRAME_SECONDS
        delay = int(prefix.dsu[0, 0, i] == AUDIO_DELAY_ID)
        s1 = tokenizer.convert_ids_to_tokens(int(prefix.text[0, 0, i]))
        s2 = tokenizer.convert_ids_to_tokens(int(prefix.text[0, 1, i]))
        lines.append(f"{i}\t{t_text:.2f}\t{t_audio:.2f}\t{delay}\t{int(prefix.dsu[0, 0, i])}\t{s1}\t{s2}\t"
                     f"{spoken_in(row['words'], t_text, t_text + FRAME_SECONDS)}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="oov-shared round-trip run dir with alignment.jsonl")
    ap.add_argument("--voice", default="tom")
    ap.add_argument("--words", nargs="*", help="default: every word aligned for this voice")
    ap.add_argument("--out", default="outputs/prefix_check")
    ap.add_argument("--gain-db", type=float, default=COSYVOICE_GAIN_DB)
    ap.add_argument("--model-dir", default="hf_models/f_actor", help="for the tokenizer")
    args = ap.parse_args()

    from nano_decode import convert_to_audio, load_model as load_codec
    from transformers import AutoTokenizer

    rows = [r for r in read_jsonl(os.path.join(args.run_dir, "alignment.jsonl")) if r["voice"] == args.voice]
    if args.words:
        missing = set(args.words) - {r["word"] for r in rows}
        if missing:
            sys.exit(f"error: not aligned for voice {args.voice!r}: {sorted(missing)} (run oov-shared's align stage)")
        rows = [r for r in rows if r["word"] in args.words]

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    codec = load_codec(num_codebooks=4)
    codec = codec.to("cuda" if torch.cuda.is_available() else "cpu")

    n_ok = 0
    for row in rows:
        name = f"{row['word']}__{row['voice']}"
        out = os.path.join(args.out, name)
        os.makedirs(out, exist_ok=True)
        try:
            prefix = build_prefix(os.path.join(args.run_dir, row["wav"]), row["words"], codec, tokenizer,
                                  gain_db=args.gain_db)
            passed = check(prefix, row, tokenizer)
        except (AssertionError, ValueError) as e:
            print(f"[FAIL] {name}: {e}")
            with open(os.path.join(out, "error.txt"), "w") as f:
                f.write(traceback.format_exc())
            continue

        convert_to_audio(codec, prefix.dsu[0, :, N_DELAY_AUDIO:].T.numpy(), os.path.join(out, "prefix.wav"))
        with open(os.path.join(out, "frames.tsv"), "w") as f:
            f.write(frame_table(prefix, row, tokenizer))
        summary = {"word": row["word"], "voice": row["voice"], "source_wav": row["wav"], "text": row["text"],
                   "aligner": row["aligner"], "target_idx": row["target_idx"], "n_frames": prefix.n_frames,
                   "seconds": round(prefix.seconds, 3), "gain_db": prefix.gain_db,
                   "requested_gain_db": args.gain_db, "overflow_words": prefix.overflow_words,
                   "checks_passed": passed}
        torch.save({"dsu": prefix.dsu, "text": prefix.text, **summary}, os.path.join(out, "prefix.pt"))
        with open(os.path.join(out, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        n_ok += 1
        print(f"[ok] {name}: N={prefix.n_frames} ({prefix.seconds:.2f}s), gain {prefix.gain_db:+.1f} dB, "
              f"overflow_words={prefix.overflow_words}")
    print(f"{n_ok}/{len(rows)} prefixes built and checked -> {args.out}")
    return 0 if n_ok == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())

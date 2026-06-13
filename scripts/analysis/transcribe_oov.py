"""Transcribe a single OOV-experiment WAV and emit transcript + metadata.

Hand it one audio file from outputs/oov/ (the mix or a _c1/_c2 channel) and it:
  1. transcribes it with openai/whisper-base.en (cached HF model, runs on CPU),
  2. looks up the target word, frequency band, and source config from
     confs/oov/manifest.tsv (falling back to parsing the filename),
  3. checks whether the target word actually appears in the transcript,
  4. writes <stem>.txt and <stem>.json into outputs/oov/transcripts/.

Usage:
    uv run python scripts/analysis/transcribe_oov.py outputs/oov/algorithm_r1.wav
    python scripts/analysis/transcribe_oov.py outputs/oov/borborygmus_r1_c1.wav --model openai/whisper-small.en
"""

import argparse
import csv
import datetime
import json
import os
import re

DEFAULT_MANIFEST = "confs/oov/manifest.tsv"
DEFAULT_OUTDIR = "outputs/oov/transcripts"
DEFAULT_MODEL = "openai/whisper-base.en"

# Trailing channel suffix produced by the dialogue model: foo_r1_c1.wav etc.
CHANNEL_RE = re.compile(r"_(c[12])$")
# Filename stem -> target word, e.g. "schadenfreude_r1" -> "schadenfreude".
WORD_FROM_STEM_RE = re.compile(r"^(.+)_r\d+$")


def parse_stem(wav_path):
    """Return (mix_stem, channel) for a wav, stripping any _c1/_c2 suffix.

    mix_stem matches the manifest's audio entry (always the mix); channel is
    'mix', 'c1', or 'c2'.
    """
    stem = os.path.splitext(os.path.basename(wav_path))[0]
    m = CHANNEL_RE.search(stem)
    if m:
        return stem[: m.start()], m.group(1)
    return stem, "mix"


def lookup_manifest(manifest_path, mix_stem):
    """Find the manifest row whose audio basename matches mix_stem."""
    if not os.path.exists(manifest_path):
        return None
    with open(manifest_path, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            audio_stem = os.path.splitext(os.path.basename(row["audio"]))[0]
            if audio_stem == mix_stem:
                return row
    return None


def resolve_metadata(wav_path, manifest_path):
    """Collect target word / band / config for a wav from the manifest."""
    mix_stem, channel = parse_stem(wav_path)
    row = lookup_manifest(manifest_path, mix_stem)
    if row:
        return {
            "channel": channel,
            "target_word": row["word"],
            "band": row["band"],
            "speech_count": int(row["speech_count"]),
            "narrative_count": int(row["narrative_count"]),
            "repeat": int(row["repeat"]),
            "config": row["config"],
            "source": "manifest",
        }
    # Manifest miss: recover what we can from the filename.
    m = WORD_FROM_STEM_RE.match(mix_stem)
    guessed_config = os.path.join("confs/oov", f"{mix_stem}.json")
    return {
        "channel": channel,
        "target_word": m.group(1) if m else None,
        "band": None,
        "speech_count": None,
        "narrative_count": None,
        "repeat": None,
        "config": guessed_config if os.path.exists(guessed_config) else None,
        "source": "filename",
    }


def count_word(text, word):
    """Case-insensitive whole-word occurrences of word in text."""
    if not word:
        return 0
    return len(re.findall(rf"\b{re.escape(word)}\b", text, flags=re.IGNORECASE))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", help="Path to one OOV wav (mix or _c1/_c2 channel).")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--chunk-length", type=int, default=30, help="Chunking for long audio."
    )
    args = parser.parse_args()

    if not os.path.exists(args.wav):
        parser.error(f"No such file: {args.wav}")

    meta = resolve_metadata(args.wav, args.manifest)

    # Import here so --help stays fast and doesn't need torch/transformers.
    from transformers import pipeline

    asr = pipeline("automatic-speech-recognition", model=args.model)
    transcript = asr(args.wav, chunk_length_s=args.chunk_length)["text"].strip()

    occurrences = count_word(transcript, meta["target_word"])
    record = {
        "wav": args.wav,
        "model": args.model,
        "transcribed_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "word_appeared": occurrences > 0,
        "occurrences": occurrences,
        **meta,
        "transcript": transcript,
    }

    os.makedirs(args.outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.wav))[0]
    # Tag outputs with the model so different models don't clobber each other,
    # e.g. "openai/whisper-large-v3-turbo" -> "large-v3-turbo".
    model_slug = args.model.rsplit("/", 1)[-1].replace("whisper-", "")
    txt_path = os.path.join(args.outdir, f"{stem}.{model_slug}.txt")
    json_path = os.path.join(args.outdir, f"{stem}.{model_slug}.json")
    with open(txt_path, "w") as f:
        f.write(transcript + "\n")
    with open(json_path, "w") as f:
        json.dump(record, f, indent=2)

    flag = "✓ found" if record["word_appeared"] else "✗ MISSING"
    print(f"word '{meta['target_word']}' ({meta['band']}): {flag} x{occurrences}")
    print(f"transcript: {txt_path}")
    print(f"metadata:   {json_path}")


if __name__ == "__main__":
    main()

"""Transcribe a single OOV-experiment WAV and merge the result into its run
record.

Hand it one audio file from outputs/oov/ (the mix or a _c1/_c2 channel) and it:
  1. resolves the mix stem back to its run record, outputs/oov/<mix_stem>.json
     (written by training/inference_example.py),
  2. transcribes the explainer (_c1) channel via PersonaPlex's asr_backends/
     registry (see oov_common.ASR; run this via $TOOLS_PYTHON, not this
     repo's own venv),
  3. looks up the target word from confs/oov/manifest.tsv (falling back to
     parsing the filename if the manifest has no matching row),
  4. merges the ASR transcript + word-fidelity grading into the run record in
     place -- via scripts/analysis/oov_common.enrich_run(), the same function
     analyze_oov.py uses, so there's exactly one JSON per run no matter which
     of the two you run.

Usage:
    $TOOLS_PYTHON scripts/analysis/transcribe_oov.py outputs/oov/algorithm_r1.wav
    $TOOLS_PYTHON scripts/analysis/transcribe_oov.py outputs/oov/borborygmus_r1_c1.wav --asr-backend whisper-large-v3
"""

import argparse
import csv
import os
import re

from oov_common import ASR, enrich_run

DEFAULT_MANIFEST = "confs/oov/manifest.tsv"
DEFAULT_OUTDIR = "outputs/oov"
DEFAULT_ASR_BACKEND = "whisper-large-v3"

# Trailing channel suffix produced by the dialogue model: foo_r1_c1.wav etc.
CHANNEL_RE = re.compile(r"_(c[12])$")
# Filename stem -> target word, e.g. "schadenfreude_r1" -> "schadenfreude".
WORD_FROM_STEM_RE = re.compile(r"^(.+)_r\d+$")


def parse_stem(wav_path):
    """Return (mix_stem, channel) for a wav, stripping any _c1/_c2 suffix."""
    stem = os.path.splitext(os.path.basename(wav_path))[0]
    m = CHANNEL_RE.search(stem)
    if m:
        return stem[: m.start()], m.group(1)
    return stem, "mix"


def lookup_manifest_word(manifest_path, mix_stem):
    """Find the target word for mix_stem from the manifest, if present."""
    if not os.path.exists(manifest_path):
        return None
    with open(manifest_path, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if os.path.splitext(os.path.basename(row["audio"]))[0] == mix_stem:
                return row["word"]
    return None


def guess_word_from_filename(mix_stem):
    m = WORD_FROM_STEM_RE.match(mix_stem)
    return m.group(1) if m else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", help="Path to one OOV wav (mix or _c1/_c2 channel).")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument(
        "--asr-backend", default=DEFAULT_ASR_BACKEND,
        help="registered backend name from PersonaPlex's asr_backends/ (default: %(default)s)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.wav):
        parser.error(f"No such file: {args.wav}")

    mix_stem, channel = parse_stem(args.wav)
    record_path = os.path.join(args.outdir, f"{mix_stem}.json")
    if not os.path.exists(record_path):
        parser.error(
            f"No run record at {record_path} -- generate it first with "
            "training/inference_example.py."
        )

    word = lookup_manifest_word(args.manifest, mix_stem) or guess_word_from_filename(mix_stem)
    if channel != "mix" and channel != "c1":
        print(f"note: {channel} given, but enrich_run always transcribes the c1 (explainer) channel")

    record = enrich_run(record_path, ASR(args.asr_backend), word=word)
    grading = record.get("grading", {}).get(args.asr_backend)
    transcript = record["asr"][args.asr_backend]["transcript"]

    print(f"transcript: {transcript}")
    if grading:
        flag = "found" if grading["audio_label"] in ("intact", "split") else "MISSING"
        print(f"word '{word}': {flag} ({grading['audio_label']}, {grading['audio_score']:.0f})")
    else:
        print(f"no target word resolved for '{mix_stem}' -- grading skipped")
    print(f"merged into: {record_path}")


if __name__ == "__main__":
    main()

"""Aggregate OOV-experiment results across the two F-Actor output streams.

For every generated run in confs/oov/manifest.tsv this joins:

  * the model's **inner monologue** -- the text stream it generated for the
    explainer (speaker1), saved in outputs/oov/<stem>.json by
    inference_example.py; and
  * the **explainer audio** -- the _c1 channel wav, transcribed here with
    Whisper (cached under outputs/oov/transcripts/).

and reports, per run and per frequency band:

  * target-word fidelity in each stream, graded intact/split/substituted/dropped
    (a single word, so character similarity -- not WER -- is the right tool);
  * WER(inner monologue -> explainer audio), i.e. how faithfully the audio
    rendered what the model planned to say (this is the WER-shaped comparison).

The point is the *gap*: a word can be planned correctly in the text stream yet
break in the audio (acoustic OOV) or break already in the text stream (lexical
OOV). Grouping the gap by band is the headline result.

Usage:
    uv run python scripts/analysis/analyze_oov.py
    uv run python scripts/analysis/analyze_oov.py --asr-model openai/whisper-base.en
"""

import argparse
import csv
import json
import os
import re
import string

from rapidfuzz import fuzz

BANDS = ["frequent", "medium", "rare", "narrative-only", "oov"]
DEFAULT_MANIFEST = "confs/oov/manifest.tsv"
DEFAULT_OUTDIR = "outputs/oov"
DEFAULT_TRANSCRIPTS = "outputs/oov/transcripts"
DEFAULT_ASR = "openai/whisper-base.en"

_PUNCT = str.maketrans("", "", string.punctuation)


def normalize(text):
    """Lowercase, drop punctuation, collapse whitespace."""
    text = text.lower().translate(_PUNCT)
    return re.sub(r"\s+", " ", text).strip()


def grade_word(text, word):
    """Grade how well `word` survives in `text`.

    Returns (label, score, evidence) where score is a 0-100 character
    similarity of the target word to the closest span in the text. We test the
    text both as-is and space-collapsed, so a split rendering ("schaden freude")
    still scores as essentially the whole word.

        intact       exact whole-word hit, or >=90 similar
        split        word is there once spaces are removed (e.g. "schaden freude")
        substituted  a near miss, 65-90 similar (e.g. "kobiola", "arum")
        dropped      nothing close (<65)
    """
    norm = normalize(text)
    w = normalize(word)
    if not w:
        return "n/a", 0.0, ""

    # exact whole-word
    if re.search(rf"\b{re.escape(w)}\b", norm):
        return "intact", 100.0, w

    tokens = norm.split()
    collapsed = norm.replace(" ", "")
    # split: the letters are contiguous once spaces are removed
    if w in collapsed and w not in tokens:
        return "split", 100.0, "(space-split)"

    # otherwise: best similarity of the word against any 1-3 token window
    best, span = 0.0, ""
    for n in (1, 2, 3):
        for i in range(len(tokens) - n + 1):
            cand = "".join(tokens[i : i + n])
            r = fuzz.ratio(w, cand)
            if r > best:
                best, span = r, " ".join(tokens[i : i + n])
    if best >= 90:
        return "intact", best, span
    if best >= 65:
        return "substituted", best, span
    return "dropped", best, span


def load_manifest(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def load_monologue(outdir, stem):
    """Inner-monologue transcripts written by inference_example.py."""
    path = os.path.join(outdir, f"{stem}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    return d["speaker1"]["transcript"], d["speaker2"]["transcript"]


class ASR:
    """Lazy, single-load Whisper wrapper that caches transcripts to disk."""

    def __init__(self, model, cache_dir):
        self.model = model
        self.slug = model.rsplit("/", 1)[-1].replace("whisper-", "")
        self.cache_dir = cache_dir
        self._pipe = None
        os.makedirs(cache_dir, exist_ok=True)

    def transcribe(self, wav):
        stem = os.path.splitext(os.path.basename(wav))[0]
        cache = os.path.join(self.cache_dir, f"{stem}.{self.slug}.json")
        if os.path.exists(cache):
            with open(cache) as f:
                return json.load(f)["transcript"]
        if self._pipe is None:
            from transformers import pipeline

            print(f"[loading {self.model} once ...]")
            self._pipe = pipeline("automatic-speech-recognition", model=self.model)
        text = self._pipe(wav, chunk_length_s=30)["text"].strip()
        with open(cache, "w") as f:
            json.dump({"wav": wav, "model": self.model, "transcript": text}, f, indent=2)
        return text


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    ap.add_argument("--transcripts", default=DEFAULT_TRANSCRIPTS)
    ap.add_argument("--asr-model", default=DEFAULT_ASR)
    ap.add_argument("--report", default="outputs/oov/analysis.tsv")
    args = ap.parse_args()

    from jiwer import wer

    asr = ASR(args.asr_model, args.transcripts)
    rows = load_manifest(args.manifest)

    results = []
    for row in rows:
        stem = os.path.splitext(os.path.basename(row["audio"]))[0]  # e.g. algorithm_r2
        mono = load_monologue(args.outdir, stem)
        if mono is None:
            continue  # not generated yet
        mono_s1, _mono_s2 = mono
        c1_wav = os.path.join(args.outdir, f"{stem}_c1.wav")
        if not os.path.exists(c1_wav):
            continue
        audio_s1 = asr.transcribe(c1_wav)

        word = row["word"]
        m_label, m_score, m_ev = grade_word(mono_s1, word)
        a_label, a_score, a_ev = grade_word(audio_s1, word)
        try:
            render_wer = wer(normalize(mono_s1), normalize(audio_s1))
        except ValueError:
            render_wer = float("nan")

        results.append(
            {
                "word": word,
                "band": row["band"],
                "repeat": row["repeat"],
                "mono_label": m_label,
                "mono_score": round(m_score, 1),
                "audio_label": a_label,
                "audio_score": round(a_score, 1),
                "audio_evidence": a_ev,
                "render_wer": round(render_wer, 3),
            }
        )

    if not results:
        print("No generated runs found (need outputs/oov/<stem>.json + _c1.wav).")
        return

    # per-run table
    hdr = ["word", "band", "repeat", "mono_label", "mono_score",
           "audio_label", "audio_score", "audio_evidence", "render_wer"]
    with open(args.report, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hdr, delimiter="\t")
        w.writeheader()
        w.writerows(results)

    print(f"\n{'word':<16}{'band':<10}{'monologue':<14}{'audio':<22}{'WER':>6}")
    print("-" * 68)
    order = {b: i for i, b in enumerate(BANDS)}
    for r in sorted(results, key=lambda r: (order.get(r["band"], 9), r["word"])):
        mono = f"{r['mono_label']}"
        audio = f"{r['audio_label']}({r['audio_score']:.0f}) {r['audio_evidence']}"
        print(f"{r['word']:<16}{r['band']:<10}{mono:<14}{audio:<22}{r['render_wer']:>6}")

    # per-band aggregate
    print(f"\n{'band':<16}{'n':>3}  {'mono intact%':>12}{'audio intact%':>14}{'mean WER':>10}")
    print("-" * 60)
    for band in BANDS:
        b = [r for r in results if r["band"] == band]
        if not b:
            continue
        mono_ok = 100 * sum(r["mono_label"] in ("intact", "split") for r in b) / len(b)
        audio_ok = 100 * sum(r["audio_label"] in ("intact", "split") for r in b) / len(b)
        wers = [r["render_wer"] for r in b if r["render_wer"] == r["render_wer"]]
        mean_wer = sum(wers) / len(wers) if wers else float("nan")
        print(f"{band:<16}{len(b):>3}  {mono_ok:>11.0f}%{audio_ok:>13.0f}%{mean_wer:>10.2f}")

    print(f"\nPer-run table written to {args.report}")


if __name__ == "__main__":
    main()

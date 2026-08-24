"""Shared helpers for the OOV experiment: word-fidelity grading, ASR, and the
single canonical per-run JSON record that generation and analysis both read
and write.

A run record (outputs/oov/<stem>.json) grows in place across two independent
stages instead of being duplicated into a second file:

  1. training/inference_example.py writes `meta`, `config`, `audio`, and
     `generation` (the model's own text-stream transcripts) when it creates
     the dialogue.
  2. enrich_run() adds `asr` (the ASR transcript of the explainer channel,
     via PersonaPlex's asr_backends/ registry -- see ASR below -- plus WER
     against the model's own transcript) and, if a target word is known,
     `grading` (intact/split/substituted/dropped fidelity). Always a
     standalone step (analyze_oov.py, transcribe_oov.py), run separately
     from generation -- see run_oov_batch.py/inference_example.py, which are
     generation-only.

Both additions are keyed by the ASR backend name, so switching backends --
or re-running with the same one -- never clobbers earlier results and is a
no-op when that backend's entry already exists.
"""

import json
import os
import re
import string
import sys

from rapidfuzz import fuzz


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

    if re.search(rf"\b{re.escape(w)}\b", norm):
        return "intact", 100.0, w

    tokens = norm.split()
    collapsed = norm.replace(" ", "")
    if w in collapsed and w not in tokens:
        return "split", 100.0, "(space-split)"

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


class ASR:
    """Lazy, single-load ASR wrapper around PersonaPlex's asr_backends
    registry, so both models' outputs are graded by the exact same ASR
    implementation instead of two different ones (this used to wrap a
    transformers whisper-base.en pipeline directly).

    Reached via PERSONAPLEX_DIR (see cluster_env.sh; defaults to a sibling
    checkout for local dev) rather than a package dependency, the same way
    training/*.py already reaches this repo's own scripts/analysis/ via
    sys.path. `model` is a registry name (see asr_backends/registry.py),
    e.g. "whisper-large-v3", not a HuggingFace model id.

    No file-level caching here -- callers cache inside the run record itself
    (see enrich_run), so loading the backend once and calling .transcribe()
    per wav is all this needs to do.
    """

    def __init__(self, model):
        self.model = model
        self._backend = None

    def transcribe(self, wav_path):
        if self._backend is None:
            personaplex_dir = os.environ.get(
                "PERSONAPLEX_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "..", "personaplex")
            )
            if personaplex_dir not in sys.path:
                sys.path.insert(0, personaplex_dir)
            from asr_backends import get_backend

            print(f"[loading asr_backends:{self.model} once ...]")
            self._backend = get_backend(self.model)
        return self._backend.transcribe(wav_path).text


def load_run(run_json_path):
    with open(run_json_path) as f:
        return json.load(f)


def save_run(run_json_path, record):
    with open(run_json_path, "w") as f:
        json.dump(record, f, indent=2)


def enrich_run(run_json_path, asr, word=None):
    """Ensure `asr[asr.model]` (and, if a target word is known,
    `grading[asr.model]`) are populated in the run record, then rewrite it in
    place. Idempotent: a model already present is not re-transcribed.

    `word` overrides `record["meta"]["word"]` if given -- analyze_oov.py
    always knows the word from the manifest; transcribe_oov.py falls back to
    whatever the config's own `meta` carries (which is empty for a non-OOV,
    plain dialogue config, so grading is simply skipped there and only the
    ASR transcript is added).
    """
    record = load_run(run_json_path)
    target_word = word if word is not None else record.get("meta", {}).get("word")

    asr_results = record.setdefault("asr", {})
    if asr.model not in asr_results:
        run_dir = os.path.dirname(run_json_path)
        c1_wav = os.path.join(run_dir, record["audio"]["c1"])
        transcript = asr.transcribe(c1_wav)
        mono_s1 = record["generation"]["speaker1"]["transcript"]

        from jiwer import wer

        try:
            render_wer = wer(normalize(mono_s1), normalize(transcript))
        except ValueError:
            render_wer = float("nan")

        asr_results[asr.model] = {
            "transcript": transcript,
            "render_wer": round(render_wer, 3),
        }

    if target_word:
        grading = record.setdefault("grading", {})
        if asr.model not in grading:
            mono_s1 = record["generation"]["speaker1"]["transcript"]
            audio_s1 = asr_results[asr.model]["transcript"]
            m_label, m_score, _m_ev = grade_word(mono_s1, target_word)
            a_label, a_score, a_ev = grade_word(audio_s1, target_word)
            grading[asr.model] = {
                "word": target_word,
                "mono_label": m_label,
                "mono_score": round(m_score, 1),
                "audio_label": a_label,
                "audio_score": round(a_score, 1),
                "audio_evidence": a_ev,
            }

    save_run(run_json_path, record)
    return record

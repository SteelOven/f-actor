"""Shared helpers for the OOV experiment: word-fidelity grading, ASR, and the
single canonical per-run JSON record that generation and analysis both read
and write.

A run record (outputs/oov/<stem>.json) grows in place across two independent
stages instead of being duplicated into a second file:

  1. training/inference_example.py writes `meta`, `config`, `audio`, and
     `generation` (the model's own text-stream transcripts) when it creates
     the dialogue.
  2. enrich_run() adds `asr` (the ASR transcript of the explainer channel,
     via oov-shared's asr_backends/ registry -- see ASR below -- plus WER
     against the model's own transcript), if a target word is known,
     `grading` (intact/split/substituted/dropped fidelity), and, if a MOS
     backend is given, `mos` (a naturalness score for that same explainer
     channel, via oov-shared's mos_backends/ registry -- see MOS below).
     Always a standalone step (analyze_oov.py, transcribe_oov.py), run
     separately from generation -- see run_oov_batch.py/inference_example.py,
     which are generation-only.

All three additions are keyed by backend name, so switching backends -- or
re-running with the same one -- never clobbers earlier results and is a
no-op when that backend's entry already exists. Unlike ASR, MOS is opt-in
(no default backend) -- it's a heavier, newer-Python-only dependency (see
oov-shared's mos_backends/utmos_backend.py) that not every tools-venv will
have installed.
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


def _add_shared_grading_to_path():
    """Put oov-shared's grading/ (asr_backends, mos_backends) on sys.path.

    Reached via OOV_SHARED_DIR (see personaplex's cluster_env.sh; defaults to
    a sibling checkout, which also resolves to /workspace/oov-shared inside
    the container) rather than a package dependency, the same way
    training/*.py already reaches this repo's own scripts/analysis/ via
    sys.path.
    """
    shared_dir = os.environ.get(
        "OOV_SHARED_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "..", "oov-shared")
    )
    grading_dir = os.path.join(shared_dir, "grading")
    if grading_dir not in sys.path:
        sys.path.insert(0, grading_dir)


class ASR:
    """Lazy, single-load ASR wrapper around oov-shared's asr_backends
    registry, so both models' outputs are graded by the exact same ASR
    implementation instead of two different ones (this used to wrap a
    transformers whisper-base.en pipeline directly).

    `model` is a registry name (see asr_backends/registry.py),
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
            _add_shared_grading_to_path()
            from asr_backends import get_backend

            print(f"[loading asr_backends:{self.model} once ...]")
            self._backend = get_backend(self.model)
        return self._backend.transcribe(wav_path).text


class MOS:
    """Lazy, single-load naturalness-scoring wrapper around oov-shared's
    mos_backends registry -- same reasoning as ASR above: one shared
    implementation instead of a second one built here, reached the same way
    (_add_shared_grading_to_path, not a package dependency).

    `model` is a registry name (see mos_backends/registry.py), e.g. "utmos".
    """

    def __init__(self, model):
        self.model = model
        self._backend = None

    def score(self, wav_path):
        if self._backend is None:
            _add_shared_grading_to_path()
            from mos_backends import get_backend

            print(f"[loading mos_backends:{self.model} once ...]")
            self._backend = get_backend(self.model)
        return self._backend.score(wav_path)


def load_run(run_json_path):
    with open(run_json_path) as f:
        return json.load(f)


def save_run(run_json_path, record):
    with open(run_json_path, "w") as f:
        json.dump(record, f, indent=2)


def enrich_run(run_json_path, asr, word=None, mos=None):
    """Ensure `asr[asr.model]` (and, if a target word is known,
    `grading[asr.model]`) are populated in the run record, then rewrite it in
    place. Idempotent: a model already present is not re-transcribed.

    `word` overrides `record["meta"]["word"]` if given -- analyze_oov.py
    always knows the word from the manifest; transcribe_oov.py falls back to
    whatever the config's own `meta` carries (which is empty for a non-OOV,
    plain dialogue config, so grading is simply skipped there and only the
    ASR transcript is added).

    Speaker 2 (the listener) is transcribed and graded too: asr[..]["transcript2"],
    grading[..]["mono2_*"/"audio2_*"], same region rule (c2_cont for prefixed runs).

    `mos`, if given (a MOS instance), also populates `mos[mos.model]` with a
    naturalness score for the same explainer (c1) channel ASR grades -- one
    number per backend, no fidelity/label split like grading has, since MOS
    scores the audio overall rather than one target word in it.

    Prefixed runs (record["prefix"], a fixed first turn by speaker 1, see
    training/inference_example.py): everything above is computed on the
    *continuation* only (audio["c1_cont"]; generation transcripts already
    exclude the prefix), since the prefix contains the word by construction.
    The prefix itself (audio["c1_prefix"]) is transcribed and graded as a
    control (prefix_label/prefix_score: did the word survive the codec in
    this run?). Every entry records its "graded_region" ("continuation" or
    "full"); an entry for a prefixed run without it is refused instead of
    reused, so a run graded by older code is never mistaken for a graded
    continuation.
    """
    record = load_run(run_json_path)
    target_word = word if word is not None else record.get("meta", {}).get("word")
    run_dir = os.path.dirname(run_json_path)
    has_prefix = "prefix" in record
    region = "continuation" if has_prefix else "full"
    c1_wav = os.path.join(run_dir, record["audio"]["c1_cont" if has_prefix else "c1"])
    c2_wav = os.path.join(run_dir, record["audio"]["c2_cont" if has_prefix else "c2"])

    for section in ("asr", "grading", "mos"):
        for backend, entry in record.get(section, {}).items():
            if entry.get("graded_region", "full") != region:
                raise ValueError(
                    f"{run_json_path}: {section}[{backend}] was computed on region "
                    f"{entry.get('graded_region', 'full')!r}, this run needs {region!r}; delete that entry and re-grade"
                )

    asr_results = record.setdefault("asr", {})
    if asr.model not in asr_results:
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
            "graded_region": region,
        }
        if has_prefix:
            asr_results[asr.model]["prefix_transcript"] = asr.transcribe(
                os.path.join(run_dir, record["audio"]["c1_prefix"])
            )

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
                "graded_region": region,
            }
            if has_prefix:
                p_label, p_score, _p_ev = grade_word(asr_results[asr.model]["prefix_transcript"], target_word)
                grading[asr.model].update(prefix_label=p_label, prefix_score=round(p_score, 1))

    # speaker 2 (the listener; in the repeat prefix variant the one asked to say the word). Added to existing
    # entries too, without re-transcribing speaker 1.
    if "transcript2" not in asr_results[asr.model]:
        asr_results[asr.model]["transcript2"] = asr.transcribe(c2_wav)
    if target_word and "audio2_label" not in record["grading"][asr.model]:
        m2_label, m2_score, _ = grade_word(record["generation"]["speaker2"]["transcript"], target_word)
        a2_label, a2_score, a2_ev = grade_word(asr_results[asr.model]["transcript2"], target_word)
        record["grading"][asr.model].update(
            mono2_label=m2_label, mono2_score=round(m2_score, 1),
            audio2_label=a2_label, audio2_score=round(a2_score, 1), audio2_evidence=a2_ev,
        )

    if mos is not None:
        mos_results = record.setdefault("mos", {})
        if mos.model not in mos_results:
            mos_results[mos.model] = {"score": round(mos.score(c1_wav), 3), "graded_region": region}

    save_run(run_json_path, record)
    return record

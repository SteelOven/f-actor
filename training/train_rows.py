"""TTS clips -> raw training rows, in the schema data.py's `tokenize_speech` reads (the Behavior-SD HF dataset).

This stops one stage before prefix.py: a row holds the codec codes, the word timings and the prompt metadata, and
data.py itself adds the audio delay, text streams, prompt and role swap at load time. Do not put prefix.py's output
(dsu/text with delay frames) in a row.

Plain variant, no TTS adaptation: no gain or peak guard, no tail trimming or appended silence, no gap. Turns are
laid back to back, each starting where the previous clip's samples end (clip lead-in and trailing silence stay as
the TTS made them). A turn's channel gets the clip, the other channel stays digital silence (encoded, as in the
prefix). The last utterance's `end_time` is the codec duration, T frames, because create_text_stream and
get_meta_data derive frames-per-second from it.

    rows = [build_row([{"speaker": 0, "wav": w, "words": ws}], codec, speakers=["Tom", "Brian"],
                      narrative="...") for ...]
    write_dataset(rows, "data/tts_rows")          # train.parquet + validation.parquet
    # then: --speech_path data/tts_rows

Selftest: scripts/analysis/train_rows_selftest.py.
"""

import json
import os

import numpy as np
import torch
from prefix import CODEC_SR, CODEC_CODEBOOK_SIZE, SAMPLES_PER_FRAME, encode

SPEAKER_EMBEDDINGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "example_speakers",
                                  "speaker_embeddings.json")
# Columns data.py removes after mapping (utils.COLUMNS_TO_REMOVE): they must exist in every row.
# Typed placeholders: parquet cannot store empty lists or empty structs.
_FILLER = {"file_name": "", "behaviors": [""], "num_turns": 0, "tts_speaker_ids": [0], "tts_genders": [""],
           "statistics": {"filler": 0}}


def speaker_embedding(
    name: str,  # speaker name, case-insensitive (Rebeka, Gweneth, Brian, Tom)
    path: str = SPEAKER_EMBEDDINGS,  # json: lowercase name -> 192 floats
) -> list[float]:  # 192-dim ECAPA vector
    """192-dim ECAPA vector of a known speaker (Rebeka, Gweneth, Brian, Tom), as inference_example uses."""
    with open(path) as f:
        return json.load(f)[name.lower()]


def _load(
    wav_path: str | os.PathLike,
) -> np.ndarray:  # float32 [n_samples], mono, resampled to CODEC_SR
    import librosa

    audio, _ = librosa.load(str(wav_path), sr=CODEC_SR, mono=True)
    return audio.astype(np.float32)


def build_row(
    turns: list[dict],  # [{"speaker": 0 | 1, "wav": path, "words": [{"word", "start", "dur"}]}], in dialogue order
    codec: torch.nn.Module,  # NanoCodec; encode() turns each channel's samples into [4, T] codes
    *,
    speakers: list[str],  # [name1, name2], appear in the prompt
    narrative: str,  # shared narrative (the two role narratives below override it per speaker)
    spk_emb: list[list[float]] | None = None,  # [emb1, emb2], 192-dim each; None: looked up by name
    new_narrative_s1: str | None = None,  # narrative as speaker 1 sees it (None: narrative)
    new_narrative_s2: str | None = None,  # narrative as speaker 2 sees it (None: narrative)
    file_name: str = "",
) -> dict:  # one raw row: the columns data.py reads (dsu_c1_path/dsu_c2_path [4, T] as lists, utterances, spk_emb_c1/c2, ...)
    """turns -> one raw row (dict).

    turns:    [{"speaker": 0 | 1, "wav": path, "words": [{"word", "start", "dur"}] (seconds, relative to the clip)}]
              in dialogue order; speaker 0 is speakers[0] (channel 1, dsu_c1_path), 1 is speakers[1].
    speakers: [name1, name2], appear in the prompt.
    spk_emb:  [emb1, emb2] (192-dim each); default looks the names up in speaker_embeddings.json.
    Raises ValueError if the words do not lie inside the audio or the codes leave the codebook."""
    if not turns:
        raise ValueError("turns is empty")
    channels = [[], []]
    utterances, cursor = [], 0
    for k, t in enumerate(turns):
        spk = t["speaker"]
        if spk not in (0, 1) or not t["words"]:
            raise ValueError(f"turn {k}: speaker must be 0 or 1 and words non-empty")
        audio = _load(t["wav"])
        offset = cursor / CODEC_SR
        words = [{"word": w["word"], "start": offset + w["start"], "dur": w["dur"]} for w in t["words"]]
        for ch in (0, 1):
            channels[ch].append(audio if ch == spk else np.zeros_like(audio))
        utterances.append({"speaker_idx": spk, "uttr_type": "normal", "start_time": words[0]["start"],
                           "end_time": words[-1]["start"] + words[-1]["dur"], "words": words})
        cursor += len(audio)

    c1, c2 = (encode(codec, np.concatenate(c)) for c in channels)
    if c1.shape != c2.shape:
        raise ValueError(f"codes of the two channels differ in shape: {c1.shape} {c2.shape}")
    if max(c1.max(), c2.max()) >= CODEC_CODEBOOK_SIZE:
        raise ValueError(f"codes must be < {CODEC_CODEBOOK_SIZE}")
    duration = c1.shape[1] * SAMPLES_PER_FRAME / CODEC_SR
    if utterances[-1]["end_time"] > duration + SAMPLES_PER_FRAME / CODEC_SR:  # codec drops a partial last frame; allow one frame of overshoot (end_time is set to duration below)
        raise ValueError(f"last word ends at {utterances[-1]['end_time']:.2f} s, audio is {duration:.2f} s")
    utterances[-1]["end_time"] = duration

    if spk_emb is None:
        spk_emb = [speaker_embedding(s) for s in speakers]
    return {**_FILLER, "file_name": file_name, "num_turns": len(turns),
            "narrative": narrative,
            "new_narrative_s1": new_narrative_s1 or narrative, "new_narrative_s2": new_narrative_s2 or narrative,
            "speakers": list(speakers), "utterances": utterances,
            "dsu_c1_path": c1.tolist(), "dsu_c2_path": c2.tolist(),
            "spk_emb_c1": list(spk_emb[0]), "spk_emb_c2": list(spk_emb[1])}


def write_dataset(
    rows: list[dict],  # rows from build_row
    out_dir: str,  # gets train.parquet and validation.parquet
    val_frac: float = 0.1,  # fraction of rows for validation (at least 1 row)
    seed: int = 0,  # seed of the train/validation shuffle
) -> tuple[int, int]:  # (n_train, n_validation)
    """rows -> out_dir/train.parquet and validation.parquet, which load_dataset(out_dir) reads as the two splits
    data.py needs. Each split gets at least one row (so at least 2 rows are required). Note data.py further trims
    each split to a multiple of batch size x world size x grad accumulation."""
    from datasets import Dataset

    if len(rows) < 2:
        raise ValueError("need at least 2 rows for a train and a validation split")
    order = np.random.default_rng(seed).permutation(len(rows))
    n_val = max(1, int(round(len(rows) * val_frac)))
    os.makedirs(out_dir, exist_ok=True)
    for name, idx in (("validation", order[:n_val]), ("train", order[n_val:])):
        Dataset.from_list([rows[i] for i in idx]).to_parquet(os.path.join(out_dir, f"{name}.parquet"))
    return len(rows) - n_val, n_val

"""Self-test for training/train_rows.py, with a stub codec (no model, runs in seconds).

    python scripts/analysis/train_rows_selftest.py

Checks: the row has every column data.py reads or removes; the plain layout (no gain, tail or gap: clips back to back,
channel 2 silent for a speaker-0 turn, end_time = codec duration); the row goes through data.py's load_data with the
training flags of train_nanocodec_special_tokens.sh and comes out with the audio delay, the words on the right frames
and the prompt; bad rows are rejected. Needs the tokenizer in hf_models/f_actor.
"""

import logging
import os
import sys
import tempfile
from types import SimpleNamespace

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "training"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from data import load_data  # noqa: E402
from dialogue_creation.utils import COLUMNS_TO_REMOVE  # noqa: E402
from prefix import CODEC_SR, SAMPLES_PER_FRAME  # noqa: E402
from prefix_selftest import ACTIVE, StubCodec, words, write_clip  # noqa: E402
from special_tokens import SILENCE_PAD  # noqa: E402
from train_rows import build_row, write_dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

FPS = CODEC_SR / SAMPLES_PER_FRAME
EMB = [0.01] * 192


def main():
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(_REPO, "hf_models", "f_actor"))
    tmp = tempfile.mkdtemp()
    a, b = os.path.join(tmp, "a.wav"), os.path.join(tmp, "b.wav")
    write_clip(a, 1.0, lead_s=0.2, trail_s=0.15)      # 1.35 s
    write_clip(b, 0.8, lead_s=0.2, trail_s=0.15)      # 1.15 s
    wa = words(0.2, [("hello", 0.4), ("there", 0.6)])
    wb = words(0.2, [("okay", 0.4), ("hello", 0.4)])
    kw = dict(speakers=["Tom", "Brian"], narrative="Tom says hello.", spk_emb=[EMB, EMB])
    codec = StubCodec()

    # 1. one turn: schema, layout
    row = build_row([{"speaker": 0, "wav": a, "words": wa}], codec, **kw)
    missing = [c for c in COLUMNS_TO_REMOVE if c not in row and c not in ("skip_example", "n_overflow_words")]
    assert not missing, missing
    c1, c2 = np.array(row["dsu_c1_path"]), np.array(row["dsu_c2_path"])
    assert c1.shape == c2.shape and c1.shape[0] == 4
    assert (c1 == ACTIVE).any() and not (c2 == ACTIVE).any(), "channel 2 must be silent"
    T = c1.shape[1]
    assert abs(row["utterances"][-1]["end_time"] - T / FPS) < 1e-9, "end_time must be the codec duration"
    assert row["utterances"][0]["words"][0]["start"] == 0.2, "no lead-in change"
    last = np.nonzero((c1 == ACTIVE).any(axis=0))[0].max()
    assert T - 1 - last <= 2, f"clip's own tail kept: {T - 1 - last} frames after speech"   # no appended silence
    print(f"ok  one turn: {T} frames, columns complete, channel 2 silent")

    # 2. two turns: back to back, no gap, each on its own channel
    row2 = build_row([{"speaker": 0, "wav": a, "words": wa}, {"speaker": 1, "wav": b, "words": wb}], codec, **kw)
    u1, u2 = row2["utterances"]
    assert abs(u2["words"][0]["start"] - (1.35 + 0.2)) < 1e-3, u2["words"][0]
    c1, c2 = np.array(row2["dsu_c1_path"]), np.array(row2["dsu_c2_path"])
    a1, a2 = (c1 == ACTIVE).any(axis=0), (c2 == ACTIVE).any(axis=0)
    assert not (a1 & a2).any() and np.nonzero(a1)[0].max() < np.nonzero(a2)[0].min()
    assert row2["utterances"][-1]["end_time"] == c1.shape[1] / FPS
    print("ok  two turns: back to back, channels disjoint")

    # 3. through data.py, as training does
    rows = [row2] * 8
    out = os.path.join(tmp, "ds")
    n_tr, n_va = write_dataset(rows, out, val_frac=0.25)
    assert (n_tr, n_va) == (6, 2)
    args = dict(
        model=SimpleNamespace(max_length=2048, num_dsus=4, text_stream=True, multi_text_stream=False),
        data=SimpleNamespace(speech_path=out, remove_start_silence=False, n_delay_audio_stream=2,
                             n_delay_text_stream=0, word_alignment=True, add_bc_token=True,
                             add_interrupt_token=True, add_counting_tokens=False, use_system_narrative=True,
                             debug=False, train_on_subset=None, preprocessing_num_workers=1),
        train=SimpleNamespace(train_batch_size=1, eval_batch_size=1, gradient_accumulation_steps=1))
    train, val, collator = load_data(args["model"], args["data"], args["train"], 4032, logging.getLogger("t"), tokenizer)
    assert len(train) == 6 and len(val) == 2
    ex = train[0]
    dsu = np.array(ex["dsu_ids"])
    assert dsu.shape == (8, c1.shape[1] + 2), dsu.shape
    assert (dsu[:, :2] == 4032).all(), "audio delay frames"
    assert np.array(ex["text_stream_ids"]).shape == (1, c1.shape[1] + 2)
    assert "Tom" in tokenizer.decode(ex["input_ids"]) and "<|SOS|>" in tokenizer.decode(ex["input_ids"])
    assert len(ex["spk_emb"]) == 192
    batch = collator([train[0], train[1]])
    assert batch["dsu_ids"].shape[:2] == (2, 8)
    silence = tokenizer.convert_tokens_to_ids(SILENCE_PAD)
    assert (np.array(ex["text_stream_ids"][0]) != silence).any(), "text stream is empty"
    print(f"ok  load_data: train {len(train)}, val {len(val)}, dsu_ids {dsu.shape}, batch {tuple(batch['dsu_ids'].shape)}")

    # 4. bad rows
    for bad, why in (([{"speaker": 0, "wav": a, "words": []}], "empty words"),
                     ([{"speaker": 0, "wav": a, "words": words(0.2, [("x", 5.0)])}], "word past the audio"),
                     ([], "no turns")):
        try:
            build_row(bad, codec, **kw)
        except ValueError as e:
            print(f"ok  rejected ({why}): {e}")
        else:
            raise AssertionError(f"accepted: {why}")
    print("PASS")


if __name__ == "__main__":
    main()

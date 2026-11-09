"""Decode dataset rows back to audio and read their word timeline (shared by make_examples.py and check_alignment_asr.py).

A dialogue is two consecutive rows (2k: Tom is the system speaker, 2k+1: Gweneth); dsu_ids heads 0-3 = the system speaker, 4-7 = the other, the
first DELAY frames are the audio delay id; text_stream_ids is frame-aligned with dsu_ids (12.5 frames/s)."""
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [f"{_ROOT}/training", f"{_ROOT}/training/inference_audio"]
import numpy as np
from transformers import AutoTokenizer

SR, FRAME_S, DELAY = 22050, 1764 / 22050, 2
tok = AutoTokenizer.from_pretrained(f"{_ROOT}/hf_models/f_actor")
PADS = set(tok.convert_tokens_to_ids(["[SILENCE_PAD]", "[UTTERANCE_PAD]", "[WORD_PAD]"])) | {tok.pad_token_id}

WORD_PAD = tok.convert_tokens_to_ids("[WORD_PAD]")

def words_of(row):
    """[(word, start_s, end_s)] from the system speaker's text stream. Text and codec frames are aligned as built (create_text_stream): a word's
    tokens sit on consecutive frames from its start frame (floor(start * 12.5)), [WORD_PAD] fills the rest of its span, [UTTERANCE_PAD] the
    gaps inside an utterance, [SILENCE_PAD] the rest. The audio delay shifts the audio, not the text: after dropping the DELAY audio frames,
    text frame f is at f * FRAME_S in the decoded audio."""
    out, prev, open_word = [], None, False
    for f, t in enumerate(row["text_stream_ids"][0]):
        if t == WORD_PAD and open_word:
            out[-1][2] = (f + 1) * FRAME_S; continue
        if t in PADS:
            prev, open_word = None, False; continue
        s = tok.decode([t])
        if prev == f - 1 and not s.startswith(" "):
            out[-1][0] += s; out[-1][2] = (f + 1) * FRAME_S
        else:
            out.append([s.strip(), f * FRAME_S, (f + 1) * FRAME_S])
        prev, open_word = f, True
    return [(w, round(a, 2), round(b, 2)) for w, a, b in out]

def decode_pair(codec, ds, i):
    """Rows i, i + 1 of one dialogue -> (tom audio, gweneth audio, tom words, gweneth words). Checks that the rows belong together."""
    from nano_decode import get_audio
    tom_row, gw_row = ds[i], ds[i + 1]
    d = np.array(tom_row["dsu_ids"])[:, DELAY:]
    assert (d < 4032).all(), "delay/EOS ids in the decoded frames"
    assert (np.array(gw_row["dsu_ids"])[:, DELAY:] == np.concatenate([d[4:], d[:4]])).all(), "the two rows are not the same dialogue"
    tom, gw = get_audio(codec, d[:4]), get_audio(codec, d[4:]); n = min(len(tom), len(gw))
    return tom[:n], gw[:n], words_of(tom_row), words_of(gw_row)

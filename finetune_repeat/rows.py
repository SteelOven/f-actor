"""A synthetic repeat-the-word dialogue -> training rows, in exactly the shape training/data.py's tokenize_speech makes them.

One dialogue (turns by Tom = speaker 0 and Gweneth = speaker 1, built by training/prefix.py build_multiturn_prefix) gives two rows,
one per "system" perspective, like prepare_dsu's random choice, but both: heads 0-3 = the system speaker, 4-7 = the other; the
text stream, narrative, "starts the dialogue" flag and speaker embedding are the system speaker's. Prompt text comes from the
repo's build_prompt (training format, "- <narrative>"). Row = {input_ids, attention_mask, labels, dsu_ids, text_stream_ids, spk_emb}.
Stage 1 (compare_rows.py) showed this matches real rows field by field.
"""
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [f"{_ROOT}/training", f"{_ROOT}/training/inference_audio"]

import numpy as np  # noqa: E402
from dialogue_creation.get_prompt import build_prompt  # noqa: E402
from dialogue_creation.utils import make_attention_mask  # noqa: E402
from prefix import build_multiturn_prefix  # noqa: E402

MAX_LENGTH = 2048
NAMES = ("Tom", "Gweneth")
# plain instruction sentences (the quick version): the same ones are used when testing
NARRATIVE_TOM = "You ask {partner} to repeat words after you, one word at a time."
NARRATIVE_PARTNER = "{partner} will say a word. Listen to it and say the same word back to {partner}."
SPEAKER_EMBEDDINGS = f"{_ROOT}/training/example_speakers/speaker_embeddings.json"


def _example(pf, names):
    """The minimal Behavior-SD example build_prompt reads (speakers, per-speaker narratives, utterance order and end times)."""
    utts = [{"speaker_idx": t["speaker"], "uttr_type": "normal", "words": [], "end_time": t["end_s"]} for t in pf.info["turns"]]
    utts[-1]["end_time"] = pf.seconds          # get_meta_data derives frames/second from the last end_time
    return {"speakers": list(names),
            "new_narrative_s1": NARRATIVE_TOM.format(partner=names[1]),
            "new_narrative_s2": NARRATIVE_PARTNER.format(partner=names[0]),
            "utterances": utts}


def dialogue_rows(turns, codec, tokenizer, names=NAMES, gap_s=0.5, tail_s=0.3, lookahead_s=0.0):
    """turns: build_multiturn_prefix's (speaker 0 / 1, wav, words, gain_db). Returns ([row_tom_system, row_gweneth_system], Prefix)."""
    pf = build_multiturn_prefix(turns, codec, tokenizer, gap_s=gap_s, tail_s=tail_s, lookahead_s=lookahead_s)
    if pf.n_frames > MAX_LENGTH:
        raise ValueError(f"dialogue has {pf.n_frames} frames, max_length is {MAX_LENGTH}")
    d, t = pf.dsu[0].numpy(), pf.text[0].numpy()        # d: heads 0-3 speaker 0, 4-7 speaker 1; t: [speaker 0, speaker 1]
    example, embeddings = _example(pf, names), json.load(open(SPEAKER_EMBEDDINGS))
    rows = []
    for system in (0, 1):
        role_map = {"system": system, "user": 1 - system}
        prompt = build_prompt(example, max_length=MAX_LENGTH, orig_dsu_length=pf.info["n_codes"],
                              role_to_speaker_map=role_map, use_system_narrative=True, speech=True)
        ids = tokenizer(prompt, truncation=True, padding="max_length", max_length=MAX_LENGTH)["input_ids"]
        dsu = d if system == 0 else np.concatenate([d[4:], d[:4]], axis=0)
        rows.append({"input_ids": ids, "attention_mask": make_attention_mask(ids, tokenizer.pad_token_id), "labels": [-100],
                     "dsu_ids": dsu.tolist(), "text_stream_ids": t[system][None].tolist(),
                     "spk_emb": embeddings[names[system].lower()]})
    return rows, pf

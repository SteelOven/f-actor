"""Build a teacher-forced dialogue prefix from a TTS clip: the first turn of a dialogue, fixed in advance.

`DSUModel.generate(..., use_speaker_sample=N, dsu_sample=..., text_sample=...)` copies the first N frames of
dsu_sample/text_sample into the sequence and generates from frame N on. This module produces those tensors from one
wav (speaker 1's turn, e.g. a CosyVoice clip of Tom) plus its word timings (oov-shared's `align` stage), in exactly
the layout training used:

    dsu   [1, 8, N]  heads 0-3 = speaker 1's NanoCodec codes, heads 4-7 = speaker 2 (encoded silence);
                     frames 0-1 = audio_delay_id on all heads (n_delay_audio_stream=2), N = 2 + T
    text  [1, 2, N]  stream 0 = speaker 1's frame-aligned text, stream 1 = speaker 2's (all [SILENCE_PAD])

The text stream and the delay are made by the training code itself (dialogue_creation/get_text_stream.py's
adapt_to_text_stream, called as data.py calls it at inference), so tokenization, word padding and the audio delay
cannot drift from what the model saw. The checkpoint's flags (word alignment, BC/interrupt tokens) are those of
scripts/inference_eval/inference_nanocodec_special_tokens.sh.

Level: CosyVoice clips are ~5.6 dB below Behavior-SD's active-speech level (oov-shared checks/LOG.md, 2026-10-16);
pass gain_db, a peak guard keeps the result below PEAK_CEILING_DBFS. Check a prefix by ear and by frame table with
scripts/analysis/check_prefix.py. Overview: docs/oov_experiment.md, "Prefix experiment".
"""

from dataclasses import dataclass, field

import numpy as np
import torch
from dialogue_creation.get_text_stream import adapt_to_text_stream

CODEC_SR = 22050
SAMPLES_PER_FRAME = 1764          # NanoCodec 22 kHz at 12.5 fps
N_DELAY_AUDIO = 2                 # n_delay_audio_stream of the released checkpoint
AUDIO_VOCAB_SIZE = 4032           # config.audio_vocab_size in inference_example.load_model
AUDIO_DELAY_ID = AUDIO_VOCAB_SIZE - 2
AUDIO_EOS_ID = AUDIO_VOCAB_SIZE - 1
PEAK_CEILING_DBFS = -0.5
COSYVOICE_GAIN_DB = 5.6           # oov-shared checks/LOG.md, loudness entry of 2026-10-16


@dataclass
class Prefix:
    dsu: torch.Tensor             # [1, 8, N] long
    text: torch.Tensor            # [1, 2, N] long
    n_frames: int                 # N, pass as use_speaker_sample
    seconds: float                # audio covered by the prefix (T frames); grading of the generated part starts here
    overflow_words: int           # words pushed later because their tokens did not fit their frames
    gain_db: float                # gain actually applied (after the peak guard)
    info: dict = field(default_factory=dict)


def load_audio(wav_path, gain_db=0.0):
    """Mono float32 at CODEC_SR with gain_db applied, scaled down if the peak would exceed PEAK_CEILING_DBFS.
    Returns (audio, applied_gain_db)."""
    import librosa

    audio, _ = librosa.load(str(wav_path), sr=CODEC_SR, mono=True)
    gain = 10 ** (gain_db / 20)
    ceiling = 10 ** (PEAK_CEILING_DBFS / 20)
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak * gain > ceiling:
        gain = ceiling / peak
    return (audio * gain).astype(np.float32), 20 * np.log10(gain) if gain > 0 else float("-inf")


def encode(codec, audio):
    """[n_samples] float -> [4, T] int64 codes."""
    device = next(codec.parameters()).device
    a = torch.from_numpy(audio).float().unsqueeze(0).to(device)
    with torch.no_grad():
        tokens, tokens_len = codec.encode(audio=a, audio_len=torch.tensor([a.shape[-1]], device=device))
    return tokens[0, :, : int(tokens_len[0])].cpu().numpy().astype(np.int64)


def synthetic_example(words, n_frames):
    """The minimal Behavior-SD example create_text_stream reads: one utterance of speaker 0.

    end_time is the duration of the T codec frames (not the end of the last word): create_text_stream derives
    frames-per-second as n_frames / utterances[-1]["end_time"], so anything else would put every word on a wrong frame.
    """
    return {"utterances": [{
        "speaker_idx": 0, "uttr_type": "normal",
        "start_time": words[0]["start"], "end_time": n_frames * SAMPLES_PER_FRAME / CODEC_SR,
        "words": [{"word": w["word"], "start": w["start"], "dur": w["dur"]} for w in words],
    }]}


def build_prefix(wav_path, words, codec, tokenizer, *, gain_db=0.0, add_bc_token=True, add_interrupt_token=True,
                 max_length=4096):
    """wav + [{"word", "start", "dur"}] (seconds) -> Prefix. Raises ValueError if the text does not fit the frames."""
    audio, applied_gain = load_audio(wav_path, gain_db)
    speaker1 = encode(codec, audio)
    speaker2 = encode(codec, np.zeros_like(audio))
    if speaker1.shape != speaker2.shape:
        raise ValueError(f"codes of speech {speaker1.shape} and silence {speaker2.shape} differ in shape")
    n_codes = speaker1.shape[1]
    # 4030/4031 are also ordinary codec codes (FSQ codebook size = audio_vocab_size = 4032): they occur a few times per
    # channel in Behavior-SD audio too, so training saw them as audio. Counted, not rejected; generate() checks for
    # EOS only in the frames it generates, not in the prefix.
    shared_ids = int(np.isin(speaker1, [AUDIO_DELAY_ID, AUDIO_EOS_ID]).sum())

    dsu, text, skip, overflow = adapt_to_text_stream(
        True,                                   # two text streams, as data.py does for inference
        np.concatenate([speaker1, speaker2], axis=0),
        N_DELAY_AUDIO, 0, True,                 # audio delay, text delay, word alignment
        n_codes, synthetic_example(words, n_codes), tokenizer, max_length, AUDIO_DELAY_ID,
        {"system": 0, "user": 1},
        add_bc_token=add_bc_token, add_interrupt_token=add_interrupt_token,
    )
    if skip:
        raise ValueError("adapt_to_text_stream skipped the example: the words' tokens do not fit their frames")

    dsu = torch.from_numpy(np.asarray(dsu, dtype=np.int64)).unsqueeze(0)
    text = torch.from_numpy(np.asarray(text, dtype=np.int64)).unsqueeze(0)
    if dsu.shape[-1] != text.shape[-1]:
        raise ValueError(f"audio has {dsu.shape[-1]} frames but text {text.shape[-1]}")
    return Prefix(dsu=dsu, text=text, n_frames=dsu.shape[-1], seconds=n_codes * SAMPLES_PER_FRAME / CODEC_SR,
                  overflow_words=int(overflow), gain_db=round(applied_gain, 2),
                  info={"wav": str(wav_path), "n_codes": n_codes, "requested_gain_db": gain_db,
                        "codes_equal_to_delay_or_eos": shared_ids})

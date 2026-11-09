"""Build a teacher-forced dialogue prefix from a TTS clip: the first turn of a dialogue, fixed in advance.

`DSUModel.generate(..., use_speaker_sample=N, dsu_sample=..., text_sample=...)` copies the first N frames of
dsu_sample/text_sample into the sequence and generates from frame N on. This module produces those tensors from one
wav (speaker 1's turn, e.g. a CosyVoice clip of Tom) plus its word timings (oov-shared's `align` stage), in exactly
the layout training used:

    dsu   [1, 8, N]  heads 0-3 = speaker 1's NanoCodec codes, heads 4-7 = speaker 2 (encoded silence);
                     frames 0-1 = audio_delay_id on all heads (n_delay_audio_stream=2), N = 2 + T
                     (codec codes are 0..4031; the model's delay/EOS ids come after them, 4032/4033)
    text  [1, 2, N]  stream 0 = speaker 1's frame-aligned text, stream 1 = speaker 2's (all [SILENCE_PAD])

The text stream and the delay are made by the training code itself (dialogue_creation/get_text_stream.py's
adapt_to_text_stream, called as data.py calls it at inference), so tokenization, word padding and the audio delay
cannot drift from what the model saw. The checkpoint's flags (word alignment, BC/interrupt tokens) are those of
scripts/inference_eval/inference_nanocodec_special_tokens.sh.

Level: CosyVoice clips are ~5.6 dB below Behavior-SD's active-speech level (oov-shared checks/LOG.md, 2026-10-16);
pass gain_db, a peak guard keeps the result below PEAK_CEILING_DBFS. Check a prefix by ear and by frame table with
scripts/analysis/check_prefix.py. Overview: oov-shared/docs/experiments/oov_experiment.md, "Prefix experiment".
"""

from dataclasses import dataclass, field

import numpy as np
import torch
from dialogue_creation.get_text_stream import adapt_to_text_stream

CODEC_SR = 22050
SAMPLES_PER_FRAME = 1764          # NanoCodec 22 kHz at 12.5 fps
N_DELAY_AUDIO = 2                 # n_delay_audio_stream of the released checkpoint
CODEC_CODEBOOK_SIZE = 4032        # = config.audio_vocab_size in inference_example.load_model: codes 0..4031
# DSUModel appends its two special ids after the codes (modeling_dsu.py: audio_vocab_size = config + 2). With a
# loaded model, pass model.audio_delay_id instead of relying on these; check_prefix.py has no model.
AUDIO_DELAY_ID = CODEC_CODEBOOK_SIZE        # 4032
AUDIO_EOS_ID = CODEC_CODEBOOK_SIZE + 1      # 4033
PEAK_CEILING_DBFS = -0.5
TAIL_FLOOR_DBFS = -50.0           # what counts as trailing silence (before gain)
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


def encode(
    codec: torch.nn.Module,  # NanoCodec (4 codebooks, 12.5 fps)
    audio: np.ndarray,  # float32 [n_samples] at CODEC_SR, mono
) -> np.ndarray:  # int64 [4, T]
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


def set_tail(audio, applied_gain_db, words, tail_s, floor_dbfs=TAIL_FLOOR_DBFS):
    """Cut audio after its last active 20 ms frame (RMS above floor_dbfs before gain, never before the last word's
    end) and append tail_s of digital silence."""
    n = int(CODEC_SR * 0.02)
    m = len(audio) // n
    rms_db = 20 * np.log10(np.sqrt((audio[: m * n].reshape(m, n) ** 2).mean(axis=1)) + 1e-10)
    active = np.nonzero(rms_db > floor_dbfs + applied_gain_db)[0]
    end = (int(active[-1]) + 1) * n if len(active) else len(audio)
    end = min(len(audio), max(end, int(round((words[-1]["start"] + words[-1]["dur"]) * CODEC_SR))))
    return np.concatenate([audio[:end], np.zeros(int(round(tail_s * CODEC_SR)), dtype=np.float32)])


def build_prefix(wav_path, words, codec, tokenizer, *, gain_db=0.0, audio_delay_id=AUDIO_DELAY_ID,
                 add_bc_token=True, add_interrupt_token=True, max_length=4096, tail_s=None, lookahead_s=0.0):
    """wav + [{"word", "start", "dur"}] (seconds) -> Prefix. Raises ValueError if the text does not fit the frames.

    tail_s: if set, the clip's trailing silence is trimmed and replaced by exactly tail_s of digital silence, so
    the turn neither ends on a hard cut nor carries the TTS's own, varying tail (0.04-0.8 s over the P4 clips).
    A cut right after the last word looks like the end of a training dialogue and the model emits audio EOS at the
    first generated frame (water: P(EOS) 0.98 at 0 s, 0.47 at +0.1 s, 0.00 from +0.2 s). None keeps the clip as is.
    The extra frames carry [SILENCE_PAD] in the text streams; word timings are unchanged."""
    audio, applied_gain = load_audio(wav_path, gain_db)
    if tail_s is not None:
        audio = set_tail(audio, applied_gain, words, tail_s)
    if lookahead_s:          # encode extra silence after the prefix and cut the codes back (see build_multiturn_prefix)
        n_real = -(-len(audio) // SAMPLES_PER_FRAME)
        padded = np.concatenate([audio, np.zeros(int(round(lookahead_s * CODEC_SR)), dtype=np.float32)])
        speaker1, speaker2 = encode(codec, padded)[:, :n_real], encode(codec, np.zeros_like(padded))[:, :n_real]
    else:
        speaker1 = encode(codec, audio)
        speaker2 = encode(codec, np.zeros_like(audio))
    if speaker1.shape != speaker2.shape:
        raise ValueError(f"codes of speech {speaker1.shape} and silence {speaker2.shape} differ in shape")
    n_codes = speaker1.shape[1]
    if max(speaker1.max(), speaker2.max()) >= CODEC_CODEBOOK_SIZE or audio_delay_id < CODEC_CODEBOOK_SIZE:
        raise ValueError(f"codes must be < {CODEC_CODEBOOK_SIZE} <= audio_delay_id ({audio_delay_id})")

    dsu, text, skip, overflow = adapt_to_text_stream(
        True,                                   # two text streams, as data.py does for inference
        np.concatenate([speaker1, speaker2], axis=0),
        N_DELAY_AUDIO, 0, True,                 # audio delay, text delay, word alignment
        n_codes, synthetic_example(words, n_codes), tokenizer, max_length, audio_delay_id,
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
                  overflow_words=int(overflow), gain_db=round(float(applied_gain), 2),
                  info={"wav": str(wav_path), "n_codes": n_codes, "requested_gain_db": gain_db,
                        "audio_delay_id": audio_delay_id, "tail_s": tail_s})


def _pack(speaker1, speaker2, example, tokenizer, audio_delay_id, add_bc_token, add_interrupt_token, max_length):
    """Codes [4, T] per speaker + the Behavior-SD style example -> (dsu [1, 8, N], text [1, 2, N], overflow_words)."""
    dsu, text, skip, overflow = adapt_to_text_stream(
        True,                                   # two text streams, as data.py does for inference
        np.concatenate([speaker1, speaker2], axis=0),
        N_DELAY_AUDIO, 0, True,                 # audio delay, text delay, word alignment
        speaker1.shape[1], example, tokenizer, max_length, audio_delay_id,
        {"system": 0, "user": 1},
        add_bc_token=add_bc_token, add_interrupt_token=add_interrupt_token,
    )
    if skip:
        raise ValueError("adapt_to_text_stream skipped the example: the words' tokens do not fit their frames")
    dsu = torch.from_numpy(np.asarray(dsu, dtype=np.int64)).unsqueeze(0)
    text = torch.from_numpy(np.asarray(text, dtype=np.int64)).unsqueeze(0)
    if dsu.shape[-1] != text.shape[-1]:
        raise ValueError(f"audio has {dsu.shape[-1]} frames but text {text.shape[-1]}")
    return dsu, text, int(overflow)


def build_multiturn_prefix(turns, codec, tokenizer, *, gain_db=0.0, gap_s=0.5, tail_s=0.3,
                           audio_delay_id=AUDIO_DELAY_ID, add_bc_token=True, add_interrupt_token=True,
                           max_length=4096, lookahead_s=0.0):
    """Several turns by both speakers -> Prefix; the last turn is the one the model answers.

    turns: [{"speaker": 0 | 1, "wav": path, "words": [{"word", "start", "dur"}], "gain_db": optional,
             "text": optional, any other keys are kept in info["turns"]}], in dialogue order. Speaker 0 is speaker 1 of
    the config (heads 0-3), speaker 1 is speaker 2 (heads 4-7). Each clip goes onto its speaker's channel, the other
    channel stays silent (encoded silence, as in build_prefix).

    Timing: the first clip starts at 0, as in build_prefix. Every later turn is placed so that its first word begins
    gap_s after the previous turn's last word (Behavior-SD median gap: 0.49 s); the clips' own lead-in and trailing
    silence do not count. Trailing silence of every clip is trimmed (set_tail), and the last turn ends in exactly
    tail_s of digital silence (see build_prefix: a cut right after speech makes the model emit EOS at once).
    Word timings are shifted by the clip's offset, so the text streams come out as the training code makes them.
    Two clips of the same speaker may not overlap in time. Raises ValueError if the text does not fit the frames.

    lookahead_s: encode lookahead_s of extra silence after the last turn, then cut the codes back to the real length. NanoCodec codes
    near the end of an encoded waveform differ from the codes of the same audio in the middle of a longer one (measured: from 0.34 s
    before the speech ends, with a 0.3 s tail), and a model fine-tuned on dialogues that all end that way reads those end-of-file
    codes as "EOS next" (oov-shared/docs/experiments/prefix_log.md). With lookahead the prefix codes are what they would be inside a longer dialogue."""
    if not turns:
        raise ValueError("turns is empty")
    placed, cursor, ch_end = [], 0.0, {0: 0, 1: 0}
    for k, t in enumerate(turns):
        spk = {0: 0, 1: 1, "speaker1": 0, "speaker2": 1}[t["speaker"]]
        audio, applied = load_audio(t["wav"], t.get("gain_db", gain_db))
        audio = set_tail(audio, applied, t["words"], tail_s if k == len(turns) - 1 else 0.0)
        offset = 0.0 if k == 0 else cursor + gap_s - t["words"][0]["start"]
        if offset < 0:
            raise ValueError(f"turn {k}: its lead-in ({t['words'][0]['start']:.2f} s) is longer than the time before it")
        off_samples = int(round(offset * CODEC_SR))
        offset = off_samples / CODEC_SR
        if off_samples < ch_end[spk]:
            raise ValueError(f"turn {k}: overlaps the previous clip of speaker {spk}")
        ch_end[spk] = off_samples + len(audio)
        last = t["words"][-1]
        cursor = offset + (last["start"] + last["dur"])
        placed.append((spk, t, audio, applied, off_samples, offset))

    total = max(p[4] + len(p[2]) for p in placed)
    extra = int(round(lookahead_s * CODEC_SR))
    channels = [np.zeros(total + extra, dtype=np.float32), np.zeros(total + extra, dtype=np.float32)]
    for spk, _, audio, _, off_samples, _ in placed:
        channels[spk][off_samples: off_samples + len(audio)] = audio
    speaker1, speaker2 = encode(codec, channels[0]), encode(codec, channels[1])
    if extra:
        n_real = -(-total // SAMPLES_PER_FRAME)                    # frames the real audio alone would give
        speaker1, speaker2 = speaker1[:, :n_real], speaker2[:, :n_real]
    if speaker1.shape != speaker2.shape:
        raise ValueError(f"codes of the two channels differ in shape: {speaker1.shape} {speaker2.shape}")
    n_codes = speaker1.shape[1]
    if max(speaker1.max(), speaker2.max()) >= CODEC_CODEBOOK_SIZE or audio_delay_id < CODEC_CODEBOOK_SIZE:
        raise ValueError(f"codes must be < {CODEC_CODEBOOK_SIZE} <= audio_delay_id ({audio_delay_id})")

    utterances = []
    for spk, t, _, _, _, offset in placed:
        words = [{"word": w["word"], "start": offset + w["start"], "dur": w["dur"]} for w in t["words"]]
        utterances.append({"speaker_idx": spk, "uttr_type": "normal", "start_time": words[0]["start"],
                           "end_time": words[-1]["start"] + words[-1]["dur"], "words": words})
    utterances[-1]["end_time"] = n_codes * SAMPLES_PER_FRAME / CODEC_SR   # fps is derived from it, as in synthetic_example

    dsu, text, overflow = _pack(speaker1, speaker2, {"utterances": utterances}, tokenizer, audio_delay_id,
                                add_bc_token, add_interrupt_token, max_length)
    info_turns = [{**{k: v for k, v in t.items() if k != "words"}, "speaker": spk, "offset_s": round(offset, 3),
                   "start_s": round(u["start_time"], 3), "end_s": round(u["words"][-1]["start"] + u["words"][-1]["dur"], 3),
                   "gain_db": round(float(applied), 2)}
                  for (spk, t, _, applied, _, offset), u in zip(placed, utterances)]
    return Prefix(dsu=dsu, text=text, n_frames=dsu.shape[-1], seconds=n_codes * SAMPLES_PER_FRAME / CODEC_SR,
                  overflow_words=overflow, gain_db=info_turns[-1]["gain_db"],
                  info={"n_codes": n_codes, "audio_delay_id": audio_delay_id, "gap_s": gap_s, "tail_s": tail_s,
                        "turns": info_turns})

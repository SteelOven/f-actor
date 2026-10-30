"""Self-test for training/prefix.py's multi-turn builder, with a stub codec (no model, runs in seconds).

    python scripts/analysis/prefix_selftest.py

Checks: one turn through build_multiturn_prefix equals build_prefix (same tensors); two turns land on the right
channels and frames, with the requested gap and tail; both text streams carry only their own speaker's words; bad
timing is rejected. Needs the tokenizer in hf_models/f_actor.
"""

import math
import os
import sys
import tempfile

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "training"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from prefix import CODEC_SR, SAMPLES_PER_FRAME, N_DELAY_AUDIO, build_multiturn_prefix, build_prefix, tidy_clip  # noqa: E402
from special_tokens import SILENCE_PAD, UTTERANCE_PAD, WORD_PAD  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

FPS = CODEC_SR / SAMPLES_PER_FRAME
ACTIVE = 5


class StubCodec(torch.nn.Module):
    """encode(): one frame per 1764 samples (ceil, as NanoCodec), code ACTIVE where the frame has sound, else 0."""

    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))

    def encode(self, audio, audio_len):
        a = audio[0].cpu().numpy()
        t = math.ceil(len(a) / SAMPLES_PER_FRAME)
        a = np.pad(a, (0, t * SAMPLES_PER_FRAME - len(a)))
        active = np.abs(a.reshape(t, SAMPLES_PER_FRAME)).max(axis=1) > 1e-4
        codes = np.where(active, ACTIVE, 0)[None, None, :].repeat(4, axis=1)
        return torch.from_numpy(codes).long(), torch.tensor([t])


def write_clip(path, speech_s, lead_s=0.2, trail_s=0.15):
    t = np.arange(int(speech_s * CODEC_SR)) / CODEC_SR
    x = np.concatenate([np.zeros(int(lead_s * CODEC_SR)), 0.3 * np.sin(2 * np.pi * 220 * t),
                        np.zeros(int(trail_s * CODEC_SR))])
    sf.write(path, x.astype(np.float32), CODEC_SR)


def words(lead, parts):
    out, t = [], lead
    for w, d in parts:
        out.append({"word": w, "start": t, "dur": d})
        t += d
    return out


def stream_words(tokenizer, ids):
    pads = {tokenizer.convert_tokens_to_ids(t) for t in (SILENCE_PAD, UTTERANCE_PAD, WORD_PAD)}
    return tokenizer.decode([i for i in ids if i not in pads]).split()


def main():
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(_REPO, "hf_models", "f_actor"))
    codec = StubCodec()
    tmp = tempfile.mkdtemp()
    a, b = os.path.join(tmp, "a.wav"), os.path.join(tmp, "b.wav")
    write_clip(a, 1.0)
    write_clip(b, 0.8)
    wa = words(0.2, [("hello", 0.4), ("there", 0.6)])      # speech 0.2 - 1.2 s
    wb = words(0.2, [("okay", 0.4), ("hello", 0.4)])       # speech 0.2 - 1.0 s

    # 1. one turn == build_prefix
    old = build_prefix(a, wa, codec, tokenizer, tail_s=0.3)
    new = build_multiturn_prefix([{"speaker": 0, "wav": a, "words": wa}], codec, tokenizer, tail_s=0.3)
    assert torch.equal(old.dsu, new.dsu) and torch.equal(old.text, new.text), "one-turn multi-turn differs from build_prefix"
    print("ok  one turn == build_prefix")

    # 2. two turns
    p = build_multiturn_prefix([{"speaker": 0, "wav": a, "words": wa}, {"speaker": 1, "wav": b, "words": wb}],
                               codec, tokenizer, gap_s=0.5, tail_s=0.3)
    t1, t2 = p.info["turns"]
    assert abs((t2["start_s"] - t1["end_s"]) - 0.5) < 1e-3, (t1, t2)
    assert abs(t2["start_s"] - (t1["end_s"] + 0.5)) < 1e-3
    dsu = p.dsu[0].numpy()
    assert (dsu[:, :N_DELAY_AUDIO] == 4032).all(), "delay frames"
    codes = dsu[:, N_DELAY_AUDIO:]
    act1, act2 = (codes[:4] == ACTIVE).any(axis=0), (codes[4:] == ACTIVE).any(axis=0)
    f = lambda s: int(s * FPS)
    assert act1[f(t1["start_s"]) + 1: f(t1["end_s"])].all() and not act1[f(t2["start_s"]) - 1:].any(), "channel 1 activity"
    assert act2[f(t2["start_s"]) + 1: f(t2["end_s"])].all() and not act2[: f(t1["end_s"])].any(), "channel 2 activity"
    last_active = max(np.nonzero(act1)[0].max(), np.nonzero(act2)[0].max())
    tail_frames = codes.shape[1] - 1 - last_active
    assert 3 <= tail_frames <= 5, f"tail of {tail_frames} frames, wanted ~0.3 s"
    assert stream_words(tokenizer, p.text[0, 0].tolist()) == ["hello", "there"], p.text[0, 0]
    assert stream_words(tokenizer, p.text[0, 1].tolist()) == ["okay", "hello"], p.text[0, 1]
    first = lambda ids, w: ids.index(tokenizer(w, add_special_tokens=False)["input_ids"][0])
    assert first(p.text[0, 1].tolist(), "okay") == f(t2["start_s"]), "speaker 2's first word on the wrong frame"
    assert p.overflow_words == 0 and p.n_frames == dsu.shape[1]
    print(f"ok  two turns: {p.n_frames} frames, turn 2 starts at {t2['start_s']} s, gap {t2['start_s'] - t1['end_s']:.2f} s, tail {tail_frames} frames")

    # 3. bad timing is rejected
    try:
        build_multiturn_prefix([{"speaker": 0, "wav": a, "words": wa}, {"speaker": 1, "wav": b, "words": wb}],
                               codec, tokenizer, gap_s=-5.0)
    except ValueError as e:
        print("ok  rejected:", e)
    else:
        raise AssertionError("negative offset accepted")
    # 3b. tidy_clip: lead cut, long gap shortened, later words shifted, fade-out
    sr = CODEC_SR
    tone = lambda d: (0.3 * np.sin(2 * np.pi * 220 * np.arange(int(d * sr)) / sr)).astype(np.float32)
    hiss = (0.01 * np.random.RandomState(0).randn(int(0.8 * sr))).astype(np.float32)
    clip = np.concatenate([hiss, tone(0.3), np.zeros(int(0.6 * sr), np.float32), tone(0.4)])
    ws = [{"word": "okay", "start": 0.8, "dur": 0.3}, {"word": "dog", "start": 1.7, "dur": 0.4}]      # gap 0.6 s
    out, tw = tidy_clip(clip, ws, lead_keep_s=0.1, max_gap_s=0.2, fade_s=0.05)
    assert abs(tw[0]["start"] - 0.1) < 2 / sr, tw
    assert abs((tw[1]["start"] - (tw[0]["start"] + tw[0]["dur"])) - 0.2) < 2 / sr, tw
    assert abs(len(out) / sr - (0.1 + 0.3 + 0.2 + 0.4)) < 3 / sr, len(out) / sr
    seg = out[int(tw[1]["start"] * sr): int((tw[1]["start"] + 0.4) * sr)]
    assert np.abs(seg[int(0.05 * sr): int(0.3 * sr)]).max() > 0.25, "second word damaged"
    assert abs(out[-1]) < 1e-3, "no fade-out"
    out2, tw2 = tidy_clip(clip, ws)                                                         # no options: unchanged
    assert np.array_equal(out2, clip) and tw2 == ws
    print("ok  tidy_clip: lead", round(tw[0]["start"], 3), "| gap", round(tw[1]["start"] - tw[0]["start"] - tw[0]["dur"], 3), "| length", round(len(out) / sr, 3))
    wav_c = os.path.join(tmp, "c.wav"); sf.write(wav_c, clip, sr)
    pf = build_multiturn_prefix([{"speaker": 1, "wav": wav_c, "words": ws, "tidy": {"lead_keep_s": 0.1, "max_gap_s": 0.2}}], codec, tokenizer, tail_s=0.3)
    t0 = pf.info["turns"][0]
    assert abs(t0["start_s"] - 0.1) < 0.01 and abs(t0["end_s"] - 1.0) < 0.01, t0
    assert stream_words(tokenizer, pf.text[0, 1].tolist()) == ["okay", "dog"]
    print("ok  tidy through the builder:", t0["start_s"], "-", t0["end_s"])

    # 4. config path: single-turn and turns configs build what the builders build; channel files per kind
    from types import SimpleNamespace

    from inference_example import config_prefix, split_prefix_channels

    model = SimpleNamespace(audio_delay_id=4032)
    os.environ["SELFTEST_DIR"] = tmp
    base_cfg = {"speaker1": {"starts": True}}
    c = config_prefix({**base_cfg, "prefix": {"wav": "$SELFTEST_DIR/a.wav", "words": wa, "tail_s": 0.3}},
                      model, codec, tokenizer)
    assert torch.equal(c.dsu, old.dsu) and torch.equal(c.text, old.text), "single-turn config differs from build_prefix"
    turns_cfg = {"turns": [{"speaker": "speaker1", "wav": "$SELFTEST_DIR/a.wav", "words": wa, "text": "hello there"},
                           {"speaker": "speaker2", "wav": "$SELFTEST_DIR/b.wav", "words": wb}],
                 "gap_s": 0.5, "tail_s": 0.3}
    c2 = config_prefix({**base_cfg, "prefix": turns_cfg}, model, codec, tokenizer)
    assert torch.equal(c2.dsu, p.dsu) and torch.equal(c2.text, p.text), "turns config differs from build_multiturn_prefix"
    for bad in ({"turns": [{**turns_cfg["turns"][1]}]}, {"turns": turns_cfg["turns"][:1], "speaker": "speaker2"}):
        try:
            config_prefix({"speaker1": {"starts": False}, "prefix": bad}, model, codec, tokenizer)
        except ValueError:
            pass
        else:
            raise AssertionError("prefix not starting with speaker1 accepted")
    try:
        config_prefix({**base_cfg, "prefix": {"turns": [{"speaker": "speaker1", "wav": "$SELFTEST_DIR/none.wav", "words": wa}]}},
                      model, codec, tokenizer)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("missing wav accepted")
    for name, pf, want in (("single", old, {"c1_prefix", "c1_cont", "c2_cont"}),
                           ("turns", p, {"c1_prefix", "c1_cont", "c2_cont", "c2_prefix"})):
        base = os.path.join(tmp, name)
        n = (pf.n_frames - N_DELAY_AUDIO + 10) * SAMPLES_PER_FRAME
        for ch in ("c1", "c2"):
            sf.write(f"{base}_{ch}.wav", np.zeros(n, dtype=np.float32), CODEC_SR)
        files = split_prefix_channels(base, ".wav", pf)
        assert set(files) == want, (name, files)
        assert all(os.path.exists(os.path.join(tmp, f)) for f in files.values()), files
        assert sf.info(os.path.join(tmp, files["c1_cont"])).frames == 10 * SAMPLES_PER_FRAME, "cut at the wrong frame"
    print("ok  config path and channel files")
    print("PASS")


if __name__ == "__main__":
    main()

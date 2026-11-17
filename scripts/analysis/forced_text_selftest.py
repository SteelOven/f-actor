"""Self-test for training/forced_text.py and generate()'s text_override hook (tiny random model, CPU, runs in seconds).

    python scripts/analysis/forced_text_selftest.py

Checks:
  1. word_tokens writes a word as create_text_stream does (utterance-initial: no leading space; inside an utterance: leading space).
  2b. Over a create_text_stream stream with a normal turn, a backchannel and an interrupt, n = k gives exactly the stream training
      makes with word k replaced (backchannel words not counted, the plain-piece [INTER] marker not a word).
  2. The hook on scripted text streams: the n-th generated word is replaced, its remaining tokens follow on the next frames, the
     sampled tokens under them are not counted; word starts after a pad and after a word token with a leading space; prefix words
     do not count; utterance-initial is read from the frame before; each speaker's rule touches only its own row; bad specs raise.
  3. generate() with a tiny random DSULlama: text_override=None gives the same output as no hook (same seed); a forced run has the
     forced tokens in its text at the recorded frames; and a fresh, cache-free forward over the forced run's frames predicts the same
     audio and text as the cached generation did (the forced tokens entered the model's context, the KV cache did not go stale).
Needs the tokenizer in hf_models/f_actor.
"""

import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "training"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from dialogue_creation.get_text_stream import create_text_stream  # noqa: E402
from forced_text import ForcedText, parse_rules, word_tokens  # noqa: E402
from special_tokens import SILENCE_PAD, UTTERANCE_PAD, WORD_PAD  # noqa: E402
from transformers import AutoTokenizer, DynamicCache, LlamaConfig  # noqa: E402

N_DELAY = 2


def check_word_tokens(tok):
    # 25 frames at 12.5 fps; utterance 1 "okay xylophone", silence, utterance 2 "water" (opens an utterance)
    ex = {"utterances": [
        {"speaker_idx": 0, "uttr_type": "normal", "start_time": 0.4, "end_time": 1.3,
         "words": [{"word": "Okay", "start": 0.4, "dur": 0.3}, {"word": "xylophone", "start": 0.8, "dur": 0.5}]},
        {"speaker_idx": 0, "uttr_type": "normal", "start_time": 1.6, "end_time": 2.0,
         "words": [{"word": "water", "start": 1.6, "dur": 0.3}]},
    ]}
    ids, skip, _ = create_text_stream(25, ex, tok, speaker_to_use=0, delay_frames=0, word_alignment=True)
    assert not skip
    for frame, word, initial in ((5, "Okay", True), (10, "xylophone", False), (20, "water", True)):
        t = word_tokens(tok, word, initial)
        assert list(ids[frame: frame + len(t)]) == t, (word, list(ids[frame: frame + len(t)]), t)
    assert word_tokens(tok, "water", True) != word_tokens(tok, "water", False)
    print("1. word_tokens matches create_text_stream (initial and inside an utterance)")


def bc_inter_example(tok, replace=None, word="xylophone"):
    """Speaker 0 (the stream's speaker): a normal turn "okay water", a backchannel "yeah yeah" inside speaker 1's turn, an interrupting
    turn "wait now"; 75 frames at 12.5 fps (6 s). replace=k puts `word` in place of speaker 0's k-th counted word (backchannels excluded).
    Every word lasts 0.4 s = 5 frames, so a 3-token word fits its frames as the original did."""
    w = lambda t, s: {"word": t, "start": s, "dur": 0.4}
    ex = {"utterances": [
        {"speaker_idx": 0, "uttr_type": "normal", "start_time": 0.4, "end_time": 1.3, "words": [w("okay", 0.4), w("water", 0.9)]},
        {"speaker_idx": 1, "uttr_type": "normal", "start_time": 1.6, "end_time": 3.2, "words": [w("so", 1.6), w("then", 2.4)],
         "backchannels": [{"start_time": 2.0, "end_time": 2.8, "words": [w("yeah", 2.0), w("yeah", 2.4)]}]},
        {"speaker_idx": 0, "uttr_type": "interrupt", "start_time": 3.6, "end_time": 6.0, "words": [w("wait", 3.6), w("now", 4.2)]},
    ]}
    if replace:
        own = [x for u in ex["utterances"] if u["speaker_idx"] == 0 for x in u["words"]]
        own[replace - 1]["word"] = word
    ids, skip, _ = create_text_stream(75, ex, tok, speaker_to_use=0, delay_frames=0, word_alignment=True,
                                      add_bc_token=True, add_interrupt_token=True)
    assert not skip
    return list(ids)


def check_against_training(tok):
    """Driving the hook over a training-made stream with n = k must give exactly the stream training makes when word k is the target."""
    orig = bc_inter_example(tok)
    as_tokens = tok.convert_ids_to_tokens(orig)
    assert "[BC]" in as_tokens and ["[", "INTER", "]"] == as_tokens[45:48], as_tokens[40:52]   # the formats this test is about
    for k in (1, 2, 3, 4):
        h = ForcedText(parse_rules({"speaker": "speaker1", "n": k, "word": "xylophone"}), tok)
        got = run_hook(h, torch.tensor([orig]), 0)[0].tolist()
        want = bc_inter_example(tok, replace=k)
        assert got == want, (k, tok.convert_ids_to_tokens(got), tok.convert_ids_to_tokens(want))
        assert h._row[0]["bc_skipped"] == 2                         # "yeah yeah", seen whether before or after the forced word
    h = ForcedText(parse_rules({"speaker": "speaker1", "n": 5, "word": "xylophone"}), tok)
    assert run_hook(h, torch.tensor([orig]), 0)[0].tolist() == orig and not h.rules[0].happened   # only 4 counted words
    print("2b. hook over create_text_stream output (normal turn, backchannel, interrupt): n = 1..4 equals training's stream "
          "with that word replaced; backchannel words skipped, [INTER] marker not a word")


def run_hook(hook, text, start):
    """Drive the hook as generate() does: text [rows, L] holds what the model "samples"; overrides are written back per frame."""
    text = text.clone()
    for step in range(start, text.shape[1]):
        for row, t in hook(step, text[:, : step + 1].clone()).items():
            text[row, step] = t
    return text


def check_hook(tok):
    ids = lambda *ws: [i for w in ws for i in (tok.convert_tokens_to_ids([w]) if w.startswith("[") else tok(w, add_special_tokens=False)["input_ids"])]
    S, W, U = SILENCE_PAD, WORD_PAD, UTTERANCE_PAD
    xy_in, xy_init = word_tokens(tok, "xylophone", False), word_tokens(tok, "xylophone", True)
    assert len(xy_in) == 3

    # A: inside an utterance ("okay [WP] [WP] water ..."): word 2 replaced, with a leading space; the model's own tokens under the
    #    forced continuation are dropped and not counted
    row = ids(S, S, "okay", W, W, " water", W, W, " again", W, S)
    text = torch.tensor([ids(*[S] * len(row)), row])
    h = ForcedText(parse_rules({"n": 2, "word": "Xylophone"}), tok)
    out = run_hook(h, text, 0)
    assert out[1, 5:8].tolist() == xy_in and out[1, 8:].tolist() == text[1, 8:].tolist() and out[0].tolist() == text[0].tolist()
    r = h.rules[0]
    assert (r.happened, r.step, r.utterance_initial, r.replaced) == (True, 5, False, " water")
    assert h._row[1]["count"] == 3                                   # okay, (water -> xylophone), again; WP under the forced tokens not counted

    # B: the word opens an utterance (frame before is [SILENCE_PAD]): no leading space
    row = ids(S, "okay", W, S, S, "water", W, S)
    text = torch.tensor([ids(*[S] * len(row)), row])
    h = ForcedText(parse_rules({"n": 2, "word": "xylophone"}), tok)
    out = run_hook(h, text, 0)
    assert out[1, 5:8].tolist() == xy_init and h.rules[0].utterance_initial is True

    # C: boundaries without a pad: "okay Ġwater" is two words, "xy lo" (no leading space) is one
    row = ids("okay", " water", "xy", "lo", " again", S)
    h = ForcedText(parse_rules({"n": 3, "word": "test"}), tok)
    run_hook(h, torch.tensor([row, row]), 0)
    assert h.rules[0].step == 4, h.rules[0].step                     # okay=1, water=2, xylo=-, again=3

    # D: words in the prefix (frames before start) do not count; utterance-initial is read from the last prefix frame
    row = ids("okay", W, " water", W, " again", W, S)
    h = ForcedText(parse_rules({"n": 1, "word": "test"}), tok)
    run_hook(h, torch.tensor([row, row]), 3)                          # generation starts at frame 3: first generated word is "again"
    assert (h.rules[0].step, h.rules[0].utterance_initial) == (4, False)

    # E: one rule per speaker; each touches only its row; a rule that never fires is recorded as not happened
    row = ids(S, "okay", W, " water", W, S)
    h = ForcedText(parse_rules([{"speaker": "speaker1", "n": 1, "word": "alpha"}, {"speaker": "speaker2", "n": 2, "word": "beta"},
                                {"speaker": "speaker2", "n": 9, "word": "never"}]), tok)
    out = run_hook(h, torch.tensor([row, row]), 0)
    alpha = word_tokens(tok, "alpha", True)
    assert out[0, 1: 1 + len(alpha)].tolist() == alpha and out[0, 1 + len(alpha):].tolist() == row[1 + len(alpha):]
    assert out[1, 3] == word_tokens(tok, "beta", False)[0] and out[1, 1] == row[1]
    rec = h.record(n_total=len(row), n_prefix=0, n_delay=N_DELAY, frame_s=0.08)
    assert [d["happened"] for d in rec] == [True, True, False] and rec[1]["audio_s"] == 0.24 and "cont_s" not in rec[1]
    rec = h.record(n_total=len(row), n_prefix=2, n_delay=N_DELAY, frame_s=0.08)
    assert rec[1]["cont_s"] == round((3 - 2 + N_DELAY) * 0.08, 3)

    # F: bad specs
    for bad in ([], {"n": 0, "word": "x"}, {"n": 1}, {"n": 1, "word": "two words"}, {"n": 1, "word": "x", "speaker": "tom"},
                {"n": 1, "word": "x", "when": 3}, [{"n": 1, "word": "x"}, {"n": 1, "word": "y"}], {"n": True, "word": "x"}):
        try:
            parse_rules(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted bad spec {bad!r}")
    assert parse_rules({"n": 2}, default_word="water")[0].word == "water"
    print("2. hook: replacement, continuation, word boundaries, prefix, utterance-initial, per-speaker rows, record, bad specs")


def tiny_model(tok):
    from modeling_dsu import DSULlama
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, tie_word_embeddings=True)
    cfg.num_dsus, cfg.text_stream, cfg.multi_text_stream = 4, True, False
    cfg.audio_vocab_size, cfg.use_speaker_embedding, cfg.calc_loss_on_c1_only = 4032, True, False
    torch.manual_seed(0)
    model = DSULlama(cfg)
    model.pad_token_id, model.grad_acc_steps = tok.pad_token_id, 1
    model.init_dsu_heads()
    model.init_audio_embeds()
    model.init_speaker_embed_proj(192)
    return model.eval()


def check_generate(tok):
    model = tiny_model(tok)
    enc = tok(["You are Tom. Talk.<|SOS|>", "You are Gweneth, you answer Tom.<|SOS|>"], return_tensors="pt", padding=True)
    rng = np.random.default_rng(0)
    spk = [rng.standard_normal(192), rng.standard_normal(192)]
    # a small prefix: 2 delay frames + 4 frames of random codes for speaker 1, silence pads in both text streams
    n_prefix, sil = 6, tok.convert_tokens_to_ids(SILENCE_PAD)
    dsu = torch.randint(0, 4032, (1, 8, n_prefix))
    dsu[:, :, :N_DELAY] = model.audio_delay_id
    text = torch.full((1, 2, n_prefix), sil)

    def gen(seed, do_sample, hook="absent"):
        torch.manual_seed(seed)
        kw = dict(input_ids=enc.input_ids, attention_mask=enc.attention_mask, max_length=12, do_sample=do_sample, temperature=0.9,
                  top_k=40, top_p=1.0, tokenizer=tok, stop_strings=["<|EOS|>"], use_speaker_sample=n_prefix, dsu_sample=dsu,
                  text_sample=text, n_delay_text_stream=0, n_delay_audio_stream=N_DELAY, talk_to_itself=True, spk_emb=spk)
        if hook != "absent":
            kw["text_override"] = hook
        return model.generate(**kw)

    assert gen(1, True) == gen(1, True, hook=None)
    print("3a. text_override=None: same sampled output as without the hook")

    # forced run, greedy: the random model's argmax text is a word token at nearly every frame, so word 2 comes early
    h = ForcedText(parse_rules([{"speaker": "speaker2", "n": 2, "word": "xylophone"}, {"speaker": "speaker1", "n": 1, "word": "water"}]), tok)
    out_dsu, out_text = gen(0, False, hook=h)
    free_dsu, free_text = gen(0, False)
    for r in h.rules:
        row = {"speaker1": 0, "speaker2": 1}[r.speaker]
        assert r.happened, r
        assert out_text[row][r.step: r.step + len(r.tokens)] == r.tokens, (r, out_text[row])
    assert out_text != free_text and out_dsu != free_dsu, "forcing changed nothing downstream"
    print(f"3b. forced tokens in the output at the recorded frames ({[(r.speaker, r.step, r.replaced) for r in h.rules]})")

    # cache-free recompute: rebuild both rows from row 0's output (row 1 = row 0 rotated by 4 heads, see interleave_halves), then for
    # every generated frame s a fresh forward over frames [:s] must predict (greedy) what generate wrote at s: the own audio heads of
    # each row, and the text unless that frame was forced
    g0 = torch.tensor(out_dsu)
    grid = torch.stack([g0, torch.roll(g0, -4, dims=0)])                      # [2, 8, L]
    txt = torch.tensor(out_text).unsqueeze(1)                                 # [2, 1, L]
    forced = {(({"speaker1": 0, "speaker2": 1}[r.speaker]), r.step + k) for r in h.rules for k in range(len(r.tokens))}
    for s in range(n_prefix, grid.shape[-1]):
        with torch.no_grad():
            o = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, dsu_ids=grid[:, :, :s], text_stream_ids=txt[:, :, :s],
                      spk_emb=spk, inference=True, need_loss=False, past_key_values=DynamicCache(), use_cache=True)
        a = o["logits"][:, -1].argmax(-1)                                     # [2, 8]
        t = o["ts_logits"][:, -1, 0].argmax(-1)                               # [2]
        assert torch.equal(a[:, :4], grid[:, :4, s]), f"audio differs at frame {s}"
        for row in (0, 1):
            if (row, s) not in forced:
                assert int(t[row]) == int(txt[row, 0, s]), f"text differs at frame {s}, row {row}"
    print(f"3c. cache-free recompute agrees with the cached forced generation on all {grid.shape[-1] - n_prefix} generated frames")


def main():
    tok = AutoTokenizer.from_pretrained(os.path.join(_REPO, "hf_models", "f_actor"))
    tok.pad_token = tok.eos_token
    check_word_tokens(tok)
    check_hook(tok)
    check_against_training(tok)
    check_generate(tok)
    print("OK")


if __name__ == "__main__":
    main()

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # 2. The inference step: two instances, left-padded prompts, KV cache, `interleave_halves`

    Follows `docs/model_walkthrough.md` section 4 (`DSUModel.generate`, `modeling_dsu.py:651-827`). Same style as notebook 01: a tiny **random** `DSULlama`
    (so shapes and bookkeeping are exact, the generated tokens are meaningless), the **real NanoCodec** and a **real request clip** for the prefix.
    `generate` is too long to copy, so: one step is unrolled cell by cell, then a short loop mirrors `generate`'s loop and is checked against the **real
    `generate(do_sample=False)`** (argmax, so it is deterministic).

    Run from the repo root: `.venv/bin/marimo edit notebooks/02_inference_step.py`. Debugger: `TRACE_BREAK=1 .venv/bin/python notebooks/02_inference_step.py` (last cells).
    """)
    return


@app.cell
def _():
    import json, os, sys
    from pathlib import Path

    REPO = Path(__file__).resolve().parent.parent
    sys.path[:0] = [str(REPO / "training"), str(REPO / "training" / "inference_audio")]

    import numpy as np
    import pandas as pd
    import torch
    import matplotlib.pyplot as plt
    from transformers import AutoTokenizer, DynamicCache, LlamaConfig

    # repo code under test
    from inference_example import prepare_prompt
    from modeling_dsu import DSULlama
    from nano_decode import load_model as load_codec
    from prefix import CODEC_SR, build_prefix, encode as codec_encode
    from train_rows import speaker_embedding

    return (
        AutoTokenizer,
        CODEC_SR,
        DSULlama,
        DynamicCache,
        LlamaConfig,
        REPO,
        build_prefix,
        codec_encode,
        json,
        load_codec,
        np,
        os,
        pd,
        plt,
        prepare_prompt,
        speaker_embedding,
        torch,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## A. Setup: codec, request clip, tiny model, prefix, prompts
    """)
    return


@app.cell
def _(load_codec):
    codec = load_codec(num_codebooks=4)  # first call may take ~10-30 s
    return (codec,)


@app.cell
def _(CODEC_SR, codec, codec_encode, np):
    # digital silence encodes to the same code in every frame (checked in notebook 01), per codebook
    silence_codes = codec_encode(codec, np.zeros(3 * CODEC_SR, dtype=np.float32))[:, 0]
    return (silence_codes,)


@app.cell
def _(REPO, json, mo, os):
    # Tom's real request clips + forced alignments (oov-data is a sibling repo; override with $OOV_CHECKS)
    OOV = os.environ.get("OOV_CHECKS", os.path.join(os.path.dirname(REPO), "oov-data", "checks"))
    _p = os.path.join(OOV, "p6_full_request_s0", "alignment.jsonl")
    mo.stop(not os.path.exists(_p), mo.md(f"Missing `{_p}`: set `$OOV_CHECKS` to the oov-data checks directory."))
    req_clips = {
        a["word"]: (os.path.join(OOV, "p6_full_request_s0", a["wav"]), a["words"])
        for a in map(json.loads, open(_p))
        if a.get("words")
    }
    word = mo.ui.dropdown(options=sorted(req_clips), value="amazing" if "amazing" in req_clips else sorted(req_clips)[0], label="word")
    word
    return req_clips, word


@app.cell
def _(AutoTokenizer, DSULlama, LlamaConfig, REPO, torch):
    tok = AutoTokenizer.from_pretrained(str(REPO / "hf_models" / "f_actor"))
    # tiny random model, as in notebook 01 (eval(): inference mode)
    _cfg = LlamaConfig(
        vocab_size=len(tok),
        hidden_size=64,  # the real model: 2048
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        tie_word_embeddings=True,
    )
    _cfg.num_dsus = 4  # 4 codebooks per speaker -> 8 audio heads
    _cfg.text_stream = True
    _cfg.multi_text_stream = False
    _cfg.audio_vocab_size = 4032
    _cfg.use_speaker_embedding = True
    _cfg.calc_loss_on_c1_only = False  # as in inference_example.load_model
    torch.manual_seed(0)
    model = DSULlama(_cfg)
    model.pad_token_id, model.grad_acc_steps = tok.pad_token_id, 1
    model.init_dsu_heads()
    model.init_audio_embeds()
    model.init_speaker_embed_proj(192)
    model.eval()
    return model, tok


@app.cell
def _(build_prefix, codec, model, req_clips, tok, word):
    # the fixed first turn (Tom's request) as the fine-tune evaluation builds it: 0.3 s tail, 1.5 s codec look-ahead
    _wav, _words = req_clips[word.value]
    prefix = build_prefix(
        wav_path=_wav,
        words=_words,  # list[dict]: word, start, dur (seconds)
        codec=codec,
        tokenizer=tok,
        audio_delay_id=model.audio_delay_id,  # 4032
        tail_s=0.3,  # trim trailing silence, append exactly 0.3 s of digital silence
        lookahead_s=1.5,  # encode 1.5 s more silence after the clip, then cut the codes back
    )
    N = prefix.n_frames  # frames copied in before generation starts (2 delay frames + the clip)
    return N, prefix


@app.cell
def _(mo, prefix, word):
    mo.md(f"""
    Prefix for `{word.value}`: **dsu** `{tuple(prefix.dsu.shape)}` (heads 0-3 = speaker 1 = Tom's request, heads 4-7 = speaker 2 = encoded silence),
    **text** `{tuple(prefix.text.shape)}` (stream 0 = Tom's words, stream 1 = speaker 2's, all silence pad); N = {prefix.n_frames} frames = {prefix.seconds:.2f} s.
    Generation starts at frame N.
    """)
    return


@app.cell
def _(model, np, prepare_prompt, speaker_embedding, tok, torch, word):
    # the two instances' prompts, written as in training (narrative_prefix "- "); the notebook-01 narratives, 0 backchannels / interruptions
    _w = word.value
    prompts = [
        prepare_prompt(
            system_speaker="Tom",
            user_speaker="Gweneth",
            narrative=f"You ask Gweneth to listen carefully and repeat the word '{_w}'.",
            speaks_first=True,
            backchannels=0,
            interruptions=0,
            narrative_prefix="- ",
        ),
        prepare_prompt(
            system_speaker="Gweneth",
            user_speaker="Tom",
            narrative=f"Tom asks you to listen carefully and repeat the word '{_w}'. You repeat it.",
            speaks_first=False,
            backchannels=0,
            interruptions=0,
            narrative_prefix="- ",
        ),
    ]
    _enc = tok(prompts, return_tensors="pt", max_length=4096, truncation=True, padding=True)
    ids = _enc.input_ids  # [2, P_max]
    mask = _enc.attention_mask  # [2, P_max]
    spk = [np.array(speaker_embedding("tom")), np.array(speaker_embedding("gweneth"))]  # one 192-d vector per instance
    prompt_len = mask.sum(1).tolist()
    print("real prompt tokens per instance [Tom, Gweneth]:", prompt_len, "| tokenizer padded both to", ids.shape[1])
    assert torch.all(model.pad_token_id != ids[mask.bool()])
    return ids, mask, prompt_len, spk


@app.cell
def _(mo):
    mo.md(r"""
    ## B. Two instances of the same weights

    `generate` (l.723-762) builds one `[2, 8, L]` tensor: row 0 is Tom's view (own = heads 0-3), row 1 is **the same tensor rotated by 4 heads**
    (own = heads 4-7 of the original = Gweneth). Both rows run through the same model, each with its own prompt, speaker vector and text stream.
    """)
    return


@app.cell
def _(N, prefix, torch):
    K = 6  # frames to generate in this notebook
    _pad = 128009  # model.pad_token_id: unfilled slots

    # l.709-716,723-725,747-752: audio = prefix + K empty frames, then [row0 | row0 rotated by 4 heads]
    _dsu = torch.cat([prefix.dsu, torch.full((1, 8, K), _pad, dtype=torch.long)], dim=-1)  # [1, 8, N+K]
    gen_dsu0 = torch.cat([_dsu, torch.roll(_dsu, -_dsu.shape[1] // 2, dims=1)], dim=0)  # [2, 8, N+K]

    # l.754-762: each row has ONE text stream, its own (row 0 = stream 0 = Tom's words, row 1 = stream 1 = speaker 2's)
    _ctx = prefix.text[:, :2, :N].permute(1, 0, 2)  # [2, 1, N]
    gen_txt0 = torch.cat([_ctx, torch.full((2, 1, K), _pad, dtype=torch.long)], dim=-1)  # [2, 1, N+K]
    print("audio", tuple(gen_dsu0.shape), "| text", tuple(gen_txt0.shape))
    return K, gen_dsu0, gen_txt0


@app.cell
def _(N, gen_dsu0, np, plt, silence_codes):
    _cat = lambda x: np.where(x == 4032, 2, np.where(x == np.tile(silence_codes, 2)[:, None], 1, 0))
    _fig, _ax = plt.subplots(2, 1, figsize=(11, 5.2), layout="constrained", sharex=True)
    for _r, (_a, _who) in enumerate(zip(_ax, ["row 0: Tom's view", "row 1: Gweneth's view (same tensor rotated by 4 heads)"])):
        _g = gen_dsu0[_r, :, :N].numpy()
        _a.imshow(_cat(_g), aspect="auto", cmap=plt.matplotlib.colors.ListedColormap(["#2a6fbb", "#f0f0f0", "#e08a1e"]), vmin=0, vmax=2, interpolation="nearest")
        _a.axhline(3.5, color="k", lw=1)
        _a.set_yticks(range(8))
        _a.set_yticklabels([f"h{h} {'OWN' if h < 4 else 'other'}" for h in range(8)], fontsize=8)
        _a.set_title(_who, fontsize=9, loc="left")
    _ax[1].set_xlabel("frame of the prefix (blue = speech code, light = silence code, orange = delay id)", fontsize=8)
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    Tom's request is in Tom's **own** heads (row 0, top) and in Gweneth's **other** heads (row 1, bottom): the two rows are mirror images. Each instance reads the other speaker's audio
    through its "other" heads, and writes its own speech through its "own" heads.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## C. One step, unrolled (`generate` l.764-818)

    Step `N`: build the inputs of both rows, run the model, take the **last** frame's logits, pick tokens, write them into the tensors, swap the halves.
    """)
    return


@app.cell
def _(N, gen_dsu0, gen_txt0, ids, mask, model, plt, prompt_len, spk):
    # step 1 (l.770-781): the real embedding builder with inference=True: prompts LEFT-padded, frames right-padded
    emb, att, _, _ = model.get_input_embeds_and_labels_padded(
        prompt_ids=ids,  # LongTensor [2, P_max]
        prompt_att_mask=mask,  # LongTensor [2, P_max]
        dsu_ids=gen_dsu0[:, :, :N],  # LongTensor [2, 8, N]: frames so far
        text_stream_ids=gen_txt0[:, :, :N],  # LongTensor [2, 1, N]
        spk_emb=spk,  # 2 vectors of 192 floats
        inference=True,  # no EOS column
    )
    _front = ids.shape[1] + 1  # left block width: P_max + 1 spk position
    pads_per_row = (att[:, :_front] == 0).sum(1).tolist()
    print("embeds", tuple(emb.shape), "= 2 rows x (1 + P_max =", _front, "| left block) + N =", N, "frames")
    print("left padding per row [Tom, Gweneth]:", pads_per_row)

    _fig, _ax = plt.subplots(figsize=(11, 1.8), layout="constrained")
    for _r, _who in enumerate(["Tom", "Gweneth"]):
        _p, _P = pads_per_row[_r], prompt_len[_r]
        for _x0, _w, _c, _l in [(0, _p, "#cccccc", "pad"), (_p, 1, "#e08a1e", "spk"), (_p + 1, _P, "#2a6fbb", "prompt"), (_front, N, "#1f8a4c", "frames")]:
            _ax.barh(1 - _r, _w, left=_x0, color=_c, edgecolor="white", height=0.7)
            if _w > 2:
                _ax.text(_x0 + _w / 2, 1 - _r, f"{_l} {_w}", ha="center", va="center", fontsize=7, color="white" if _c != "#cccccc" else "k")
    _ax.set_yticks([1, 0])
    _ax.set_yticklabels(["Tom", "Gweneth"])
    _ax.set_xlabel("position in the padded sequence (the shorter prompt is padded on the LEFT, so the frames line up on the right)", fontsize=8)
    _fig
    return att, emb, pads_per_row


@app.cell
def _(DynamicCache, N, emb, gen_dsu0, gen_txt0, ids, mask, model, spk, torch):
    # step 2 (l.770-781): the real forward, cache empty -> it processes ALL positions and fills the KV cache
    cache = DynamicCache()
    with torch.no_grad():
        out_N = model(
            input_ids=ids,
            attention_mask=mask,
            dsu_ids=gen_dsu0[:, :, :N],
            text_stream_ids=gen_txt0[:, :, :N],
            spk_emb=spk,
            inference=True,
            need_loss=False,
            past_key_values=cache,
            use_cache=True,
        )
    print("cache holds", cache.get_seq_length(), "positions per row (= the embedding width", emb.shape[1], ")")
    print("logits", tuple(out_N["logits"].shape), "| ts_logits", tuple(out_N["ts_logits"].shape))
    return cache, out_N


@app.cell
def _(mo, pads_per_row):
    mo.md(f"""
    ### A check on the first step: left padding and the audio rows

    Both prompts are left-padded to the same width (pads per row: {pads_per_row}), so the frames are the **last N positions of every row**.
    `forward` has two ways to pick the audio rows out of the hidden states (`modeling_dsu.py:124-137`):

    - **cache branch** (`use_cache=True`, what `generate` uses): take the last position. The test `len(past_key_values) > 0` is evaluated *after* the Llama has run,
      so it is already true on the **first** call too. Note the first call's logits have a single frame, not N.
    - **no-cache branch** (`use_cache` unset): slice `[prompt_len : total_len]` with the **unpadded** lengths. For a left-padded row that window is shifted
      left by its pad count.

    Below, both are compared with the same computation on the true frame positions.
    """)
    return


@app.cell
def _(
    N,
    att,
    emb,
    gen_dsu0,
    gen_txt0,
    ids,
    mask,
    mo,
    model,
    out_N,
    pads_per_row,
    pd,
    spk,
    torch,
):
    with torch.no_grad():
        _h = model.model(inputs_embeds=emb, attention_mask=att).last_hidden_state  # [2, width, H]
        true_logits = model.dsu_head(_h[:, -N:, :]).view(2, N, 8, -1)  # the N frame positions, right-aligned in both rows
        out_nocache = model(  # same inputs, but use_cache is not set -> the "slice with unpadded lengths" branch
            input_ids=ids, attention_mask=mask, dsu_ids=gen_dsu0[:, :, :N], text_stream_ids=gen_txt0[:, :, :N],
            spk_emb=spk, inference=True, need_loss=False,
        )
    _d = lambda a, b: f"{(a - b).abs().max().item():.1e}"
    _rows = [
        {
            "instance": _who,
            "left pads": pads_per_row[_r],
            "cache branch (generate), last frame": _d(out_N["logits"][_r, -1], true_logits[_r, -1]),
            "no-cache branch, last frame": _d(out_nocache["logits"][_r, -1], true_logits[_r, -1]),
        }
        for _r, _who in enumerate(["Tom (row 0)", "Gweneth (row 1)"])
    ]
    mo.vstack([
        mo.ui.table(pd.DataFrame(_rows), selection=None, show_download=False),
        mo.md(
            "`max |diff|` against the true frame positions. `generate` uses the cache branch, which is exact for both rows. The no-cache branch is off for the padded row "
            "(here Tom's), so a batch of 2 prompts of different length through `forward(inference=True)` **without a cache** would read stale rows. "
            "The repo does not do that: `generate` always has the cache, and `inference.py`'s perplexity loop uses batch size 1 (no padding)."
        ),
    ])
    return


@app.cell
def _(N, gen_dsu0, gen_txt0, mo, out_N, pd, torch):
    # step 3 (l.784-798): take the LAST frame's logits and pick tokens (argmax here; generate samples with T=0.9, top-k 40 in inference_example)
    a_next = out_N["logits"][:, -1, :, :].argmax(dim=-1)  # [2, 8]: one code per head, per instance
    t_next = out_N["ts_logits"][:, -1, :, :].argmax(dim=-1)  # [2, 1]: one text token per instance

    # step 4 (l.787-798): write them into frame N of both rows
    dsu_w, txt_w = gen_dsu0.clone(), gen_txt0.clone()
    dsu_w[:, :, N] = a_next
    txt_w[:, :, N] = t_next

    # step 5 (l.727-732,802-805): interleave_halves: each instance's OTHER heads := the other instance's OWN heads
    _half = dsu_w.shape[1] // 2
    dsu_i = dsu_w.clone()
    dsu_i[0::2, _half:] = dsu_w[1::2, :_half].clone()
    dsu_i[1::2, _half:] = dsu_w[0::2, :_half].clone()

    _tab = lambda t, name: pd.DataFrame(t[:, :, N].numpy(), index=["row 0 (Tom)", "row 1 (Gweneth)"], columns=[f"h{h}" for h in range(8)]).assign(when=name)
    assert torch.equal(dsu_i[0, 4:, N], dsu_w[1, :4, N]) and torch.equal(dsu_i[1, 4:, N], dsu_w[0, :4, N])
    mo.vstack([
        mo.md("**Frame N after sampling** (each row sampled all 8 heads; h0-h3 = OWN, h4-h7 = other, which is a guess about the other speaker):"),
        _tab(dsu_w, "sampled"),
        mo.md("**After `interleave_halves`**: Tom's h4-h7 are now Gweneth's sampled h0-h3 and vice versa. The other-head guesses are thrown away; each instance now hears what the other one actually produced:"),
        _tab(dsu_i, "interleaved"),
    ])
    return dsu_i, txt_w


@app.cell
def _(mo):
    mo.md(r"""
    ### The KV cache: step N+1 feeds one frame, and gives the same logits as a full pass

    `forward` still builds the embeddings of **all** frames so far, but when the cache is filled it keeps only the last position (l.108-109) and the
    Llama only computes that one row; the earlier rows' keys and values come from the cache.
    """)
    return


@app.cell
def _(DynamicCache, N, cache, dsu_i, ids, mask, model, spk, torch, txt_w):
    _step = N + 1
    with torch.no_grad():
        out_cached = model(  # cache filled at step N: only the new frame goes through the Llama
            input_ids=ids, attention_mask=mask, dsu_ids=dsu_i[:, :, :_step], text_stream_ids=txt_w[:, :, :_step],
            spk_emb=spk, inference=True, need_loss=False, past_key_values=cache, use_cache=True,
        )
        out_full = model(  # same inputs, no cache: the whole sequence again
            input_ids=ids, attention_mask=mask, dsu_ids=dsu_i[:, :, :_step], text_stream_ids=txt_w[:, :, :_step],
            spk_emb=spk, inference=True, need_loss=False, past_key_values=DynamicCache(), use_cache=True,
        )
    print("cache now holds", cache.get_seq_length(), "positions (one more than before)")
    print("cached last-frame logits == full-pass last-frame logits:",
          torch.allclose(out_cached["logits"][:, -1], out_full["logits"][:, -1], atol=1e-4),
          "| max diff", (out_cached["logits"][:, -1] - out_full["logits"][:, -1]).abs().max().item())
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## D. A short loop, checked against the real `generate`

    The same five steps repeated `K` times, with `generate`'s stop test (any sampled head equal to the audio EOS id). The result is compared to
    `model.generate(..., do_sample=False)` called exactly as `inference_example.generate_dialogue` does.
    """)
    return


@app.cell
def _(DynamicCache, K, N, gen_dsu0, gen_txt0, ids, mask, model, spk, torch):
    def unrolled_generate(
        n_steps: int,  # frames to generate
    ) -> tuple[torch.Tensor, torch.Tensor, int]:  # (audio [2, 8, N+K], text [2, 1, N+K], last_step as generate() sets it)
        dsu, txt = gen_dsu0.clone(), gen_txt0.clone()
        cache, last_step = DynamicCache(), -1
        half = dsu.shape[1] // 2
        for step in range(N, N + n_steps):
            with torch.no_grad():
                o = model(
                    input_ids=ids, attention_mask=mask, dsu_ids=dsu[:, :, :step], text_stream_ids=txt[:, :, :step],
                    spk_emb=spk, inference=True, need_loss=False, past_key_values=cache, use_cache=True,
                )
            cache = o["past_key_values"]
            dsu[:, :, step] = o["logits"][:, -1].argmax(-1)  # [2, 8]
            txt[:, :, step] = o["ts_logits"][:, -1].argmax(-1)  # [2, 1]
            x = dsu.clone()  # interleave_halves
            x[0::2, half:] = dsu[1::2, :half].clone()
            x[1::2, half:] = dsu[0::2, :half].clone()
            dsu = x
            if torch.isin(dsu[:, :, step], torch.tensor([model.audio_eos_id])).any():  # EOS on any head of any row
                last_step = step + 1
                break
        return dsu, txt, last_step

    mine_dsu, mine_txt, mine_last = unrolled_generate(K)
    return mine_dsu, mine_last


@app.cell
def _(
    K,
    N,
    ids,
    mask,
    mine_dsu,
    mine_last,
    mo,
    model,
    prefix,
    spk,
    tok,
    torch,
):
    with torch.no_grad():
        real_dsu, real_txt = model.generate(  # the REAL generate, called as inference_example.generate_dialogue does (argmax instead of sampling)
            input_ids=ids,
            attention_mask=mask,
            max_length=K,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
            tokenizer=tok,
            stop_strings=["<|EOS|>"],
            use_speaker_sample=N,  # number of prefix frames
            dsu_sample=prefix.dsu,
            text_sample=prefix.text,
            n_delay_text_stream=0,
            n_delay_audio_stream=2,
            talk_to_itself=True,
            spk_emb=spk,
        )
    _mine = mine_dsu[0, :, :mine_last]  # generate() returns row 0, cut at last_step (-1 drops the final column when no EOS stopped it)
    _same = torch.equal(torch.tensor(real_dsu), _mine)
    assert _same, "unrolled loop != model.generate"
    mo.md(f"""
    **OK: the unrolled loop == the real `generate`.** Real output: audio `{len(real_dsu)} heads x {len(real_dsu[0])} frames` = {N} prefix + {len(real_dsu[0]) - N} generated.
    `generate` returned {len(real_dsu[0]) - N} new frames for `max_length={K}`: with no EOS, `last_step` stays `-1`, so `[:, :, :-1]` drops the last frame it computed
    (`modeling_dsu.py:820-821`).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## E. Normal mode: no prefix, generation starts at frame 0

    Everything above started after a fixed first turn. Without a prefix (`inference_example.generate_dialogue`, `prefix` absent) the caller passes
    **placeholder** tensors and `use_speaker_sample=0`: `generate` ignores them (`dsu_sample[:, :, :0]` is empty) and starts at step 0 with **no frames at all**.
    Differences: the first two frames are not sampled, `generate` writes the delay id itself (`n_delay_audio_stream=2`), and the prompt alone decides who speaks
    first ("starts the dialogue").
    """)
    return


@app.cell
def _(K, torch):
    # as inference_example.py:222-224 without a prefix: placeholders, never read
    nm_dsu_sample = torch.zeros(1, 8, 1, dtype=torch.int64)
    nm_text_sample = torch.zeros(1, 2, 1, dtype=torch.int64)
    _pad = 128009  # model.pad_token_id

    # generate l.690-697, 704-706, 731-733, 735-743 with len_context = 0: nothing from the placeholders, K empty frames, row 1 = row 0 rotated by 4 heads
    _dsu = torch.full((1, 8, K), _pad, dtype=torch.long)
    nm_dsu0 = torch.cat([_dsu, torch.roll(_dsu, -4, dims=1)], dim=0)  # [2, 8, K]
    nm_txt0 = torch.full((2, 1, K), _pad, dtype=torch.long)  # [2, 1, K]
    print("audio", tuple(nm_dsu0.shape), "| text", tuple(nm_txt0.shape), "| frames before step 0: 0")
    return nm_dsu0, nm_dsu_sample, nm_text_sample, nm_txt0


@app.cell
def _(DynamicCache, ids, mask, model, nm_dsu0, nm_txt0, spk, tok, torch):
    # step 0: the frame slices are EMPTY ([2, 8, 0] and [2, 1, 0]); the Llama only sees [left pads | spk | prompt]
    nm_cache = DynamicCache()
    with torch.no_grad():
        nm_out0 = model(
            input_ids=ids, attention_mask=mask,
            dsu_ids=nm_dsu0[:, :, :0],  # LongTensor [2, 8, 0]
            text_stream_ids=nm_txt0[:, :, :0],  # LongTensor [2, 1, 0]
            spk_emb=spk, inference=True, need_loss=False, past_key_values=nm_cache, use_cache=True,
        )
    print("cache holds", nm_cache.get_seq_length(), "positions per row = spk + P_max + 0 frames =", 1 + ids.shape[1])
    print("audio logits", tuple(nm_out0["logits"].shape), "(one position)  | text logits", tuple(nm_out0["ts_logits"].shape), "(all positions)")
    # `ids` is right-padded by the tokenizer (Tom's row ends in pad tokens); the model drops those and re-pads on the LEFT, so the last
    # position of each row in the Llama is that row's last REAL prompt token
    _last_real = [ids[r][mask[r].bool()][-1:] for r in range(2)]
    print("the position those logits come from = the last real prompt token of each row:", [repr(tok.decode(t)) for t in _last_real],
          "(the end of `<|SOS|>`; the tokenizer splits it into several pieces)")
    return


@app.cell
def _(mo):
    mo.md(r"""
    The hidden state at the last prompt token (`<|SOS|>` ends the prompt) is what predicts frame 0 (text) and would predict frame 0's audio. `generate` computes
    the audio logits but **discards** them for steps 0 and 1 (`if step >= n_delay_audio_stream: sample, else: delay id`, `modeling_dsu.py:786-790`); the text
    stream is sampled from step 0.
    """)
    return


@app.cell
def _(DynamicCache, K, ids, mask, model, nm_dsu0, nm_txt0, spk, torch):
    def unrolled_generate_noprefix(
        n_steps: int,  # frames to generate, starting at frame 0
        n_delay_audio_stream: int = 2,  # steps below this get the delay id instead of a sampled code
    ) -> tuple[torch.Tensor, torch.Tensor, int]:  # (audio [2, 8, K], text [2, 1, K], last_step as generate() sets it)
        dsu, txt = nm_dsu0.clone(), nm_txt0.clone()
        cache, last_step, half = DynamicCache(), -1, dsu.shape[1] // 2
        for step in range(0, n_steps):
            with torch.no_grad():
                o = model(
                    input_ids=ids, attention_mask=mask, dsu_ids=dsu[:, :, :step], text_stream_ids=txt[:, :, :step],
                    spk_emb=spk, inference=True, need_loss=False, past_key_values=cache, use_cache=True,
                )
            cache = o["past_key_values"]
            if step >= n_delay_audio_stream:
                dsu[:, :, step] = o["logits"][:, -1].argmax(-1)  # [2, 8]
            else:
                dsu[:, :, step] = model.audio_delay_id  # forced: the logits are not used
            txt[:, :, step] = o["ts_logits"][:, -1].argmax(-1)  # [2, 1]: text IS sampled from step 0
            x = dsu.clone()  # interleave_halves
            x[0::2, half:] = dsu[1::2, :half].clone()
            x[1::2, half:] = dsu[0::2, :half].clone()
            dsu = x
            if torch.isin(dsu[:, :, step], torch.tensor([model.audio_eos_id])).any():
                last_step = step + 1
                break
        return dsu, txt, last_step

    nm_dsu, nm_txt, nm_last = unrolled_generate_noprefix(K)
    return nm_dsu, nm_last, nm_txt


@app.cell
def _(
    K,
    ids,
    mask,
    mo,
    model,
    nm_dsu,
    nm_dsu_sample,
    nm_last,
    nm_text_sample,
    nm_txt,
    pd,
    spk,
    tok,
    torch,
):
    with torch.no_grad():
        nm_real_dsu, nm_real_txt = model.generate(  # the REAL generate, called as inference_example.generate_dialogue does without a prefix (argmax)
            input_ids=ids,
            attention_mask=mask,
            max_length=K,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
            tokenizer=tok,
            stop_strings=["<|EOS|>"],
            use_speaker_sample=0,  # no prefix
            dsu_sample=nm_dsu_sample,
            text_sample=nm_text_sample,
            n_delay_text_stream=0,
            n_delay_audio_stream=2,
            talk_to_itself=True,
            spk_emb=spk,
        )
    assert torch.equal(torch.tensor(nm_real_dsu), nm_dsu[0, :, :nm_last]), "unrolled (no prefix) != model.generate"
    assert torch.equal(torch.tensor(nm_real_txt), nm_txt[:, 0, :nm_last]), "text differs"
    _n = len(nm_real_dsu[0])
    _tab = pd.DataFrame(
        {f"frame {f}": [nm_real_dsu[h][f] for h in range(8)] + [tok.convert_ids_to_tokens(nm_real_txt[0][f])] for f in range(_n)},
        index=[f"h{h} {'own' if h < 4 else 'other'}" for h in range(8)] + ["text (Tom)"],
    )
    mo.vstack([
        mo.md(f"**OK: the unrolled no-prefix loop == the real `generate`** (audio and text). It returns {_n} frames for `max_length={K}` (last frame dropped, as before). "
              "Row 0 (Tom): frames 0-1 are the forced delay id 4032 on every head, audio is sampled from frame 2, text from frame 0. "
              "Heads h4-h7 equal Gweneth's own heads (interleave); the random model's token ids are arbitrary."),
        mo.ui.table(_tab.reset_index(names="head"), selection=None, show_download=False),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## F. Debugger entry

    ```
    TRACE_BREAK=1 .venv/bin/python notebooks/02_inference_step.py
    ```
    stops before the call below. Useful breakpoints: `b modeling_dsu.py:764` (top of the generate loop), `b modeling_dsu.py:784` (logits of the step),
    `b modeling_dsu.py:802` (interleave), `b modeling_dsu.py:124` (the audio-row slice from the check above). `c` continues, `p dsu_ids_step.shape` prints.
    """)
    return


@app.cell
def _(N, ids, mask, model, os, prefix, spk, tok, torch):
    if os.environ.get("TRACE_BREAK"):
        import pdb

        pdb.set_trace()
        with torch.no_grad():
            _ = model.generate(
                input_ids=ids, attention_mask=mask, max_length=3, do_sample=False, pad_token_id=tok.eos_token_id,
                eos_token_id=tok.eos_token_id, tokenizer=tok, stop_strings=["<|EOS|>"], use_speaker_sample=N,
                dsu_sample=prefix.dsu, text_sample=prefix.text, n_delay_text_stream=0, n_delay_audio_stream=2,
                talk_to_itself=True, spk_emb=spk,
            )
    return


if __name__ == "__main__":
    app.run()

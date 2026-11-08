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
    # 1. One row -> one training sample -> what is scored

    Self-contained trace of `docs/model_walkthrough.md` sections 2-3. Everything is built in this file: a toy request -> reply row,
    the repo's real `data.py` pipeline, a **tiny random** `DSULlama` (shapes/layout/masks are exact, loss values are meaningless),
    and then the repo's tensor-building code **unrolled into one cell per step** (`get_input_embeds_and_labels`, the head/loss part of `forward`).
    Each unrolled block ends in an `assert` against the real function, so a wrong copy or a changed repo fails loudly.

    - Notebook: `.venv/bin/marimo edit notebooks/01_row_to_sample.py` (repo root).
    - Debugger: `TRACE_BREAK=1 .venv/bin/python notebooks/01_row_to_sample.py` runs it as a script and drops into `pdb` right before the real `model(**inp)` (last cells).
    """)
    return


@app.cell
def _():
    import json, logging, os, sys, tempfile
    from pathlib import Path
    from types import SimpleNamespace

    REPO = Path(__file__).resolve().parent.parent
    sys.path[:0] = [str(REPO / "training"), str(REPO / "training" / "inference_audio")]

    import numpy as np
    import pandas as pd
    import soundfile as sf
    import torch
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    from transformers import AutoTokenizer, LlamaConfig

    # repo code under test
    from data import load_data
    from modeling_dsu import DSULlama
    from nano_decode import load_model as load_codec  # NeMo NanoCodec 22 kHz, 12.5 fps, 4 codebooks (as build_rows_dataset.py)
    from prefix import CODEC_SR, encode as codec_encode
    from train_rows import build_row, write_dataset

    return (
        AutoTokenizer,
        CODEC_SR,
        DSULlama,
        F,
        LlamaConfig,
        REPO,
        SimpleNamespace,
        build_row,
        codec_encode,
        json,
        load_codec,
        load_data,
        logging,
        np,
        os,
        pd,
        plt,
        sf,
        tempfile,
        torch,
        write_dataset,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## A. The row

    The codec is the **real NanoCodec** (`nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps`, 4 codebooks, 12.5 fps, loaded from the local HF cache).
    The audio is **real TTS**: Tom's request clip ("Please listen carefully and repeat the word X.", `p6_full_request_s0`) followed by Gweneth's reply
    clip ("Okay, X.", `p6_full_reply_s0`), with their forced word alignments, from the sibling repo `oov-data/checks` (override with `$OOV_CHECKS`).
    The dropdown can switch to a synthetic tone if those files are missing. The two channels are encoded exactly as `train_rows.build_row` does: the
    turn's clip on its own channel, **digital silence on the other**, which the codec also encodes to real (non-zero) codes. Everything else is repo code.
    """)
    return


@app.cell
def _(load_codec):
    codec = load_codec(num_codebooks=4)  # first call may take ~10-30 s (NeMo import + weights)
    return (codec,)


@app.cell
def _(CODEC_SR, codec, codec_encode, np, sf):
    # Why: the channel that is not speaking is digital silence (all zeros). The codec still turns it into tokens, and they are NOT 0.
    # We need to know which tokens, to tell "silent" from "speech" frames when reading the grid and in the plot below.
    # (Only for reading the notebook; the model and the loss never use this.)

    # step 1: encode 3 seconds of pure zeros, exactly like the silent channel of a row
    zeros = np.zeros(3 * CODEC_SR, dtype=np.float32)
    silence_tokens = codec_encode(codec, zeros)  # [4 codebooks, 38 frames]

    # step 2: every frame comes out identical (checked: all 38 frames, per codebook), so frame 0 is the answer
    silence_codes = silence_tokens[:, 0]  # [4], one code per codebook

    def write_clip(
        path: str,  # where to write the .wav
        speech_s: float,  # seconds of tone
        lead_s: float = 0.2,  # seconds of digital silence before the tone
        trail_s: float = 0.15,  # seconds of digital silence after the tone
    ) -> None:
        """Synthetic stand-in for a TTS clip: silence, a 220 Hz tone, silence (mono, CODEC_SR)."""
        t = np.arange(int(speech_s * CODEC_SR)) / CODEC_SR
        x = np.concatenate(
            [
                np.zeros(int(lead_s * CODEC_SR)),
                0.3 * np.sin(2 * np.pi * 220 * t),
                np.zeros(int(trail_s * CODEC_SR)),
            ]
        )
        sf.write(path, x.astype(np.float32), CODEC_SR)

    def word_timings(
        lead: float,  # seconds before the first word
        parts: list[tuple[str, float]],  # [(word, duration_s), ...], back to back
    ) -> list[dict[str, str | float]]:
        """-> [{"word", "start", "dur"}, ...] in seconds, the format build_row() expects."""
        out, t = [], lead
        for w, d in parts:
            out.append({"word": w, "start": t, "dur": d})
            t += d
        return out

    return silence_codes, word_timings, write_clip


@app.cell
def _(REPO, json, os):
    # real TTS clips + forced alignments of the fine-tune data (oov-data is a sibling repo; override with $OOV_CHECKS)
    OOV = os.environ.get("OOV_CHECKS", os.path.join(os.path.dirname(REPO), "oov-data", "checks"))

    def _load(
        run: str,  # a directory under OOV, e.g. "p6_full_request_s0"
    ) -> dict[str, tuple[str, list[dict]]]:  # word -> (wav path, forced-aligned words)
        p = os.path.join(OOV, run, "alignment.jsonl")
        if not os.path.exists(p):
            return {}
        return {
            a["word"]: (os.path.join(OOV, run, a["wav"]), a["words"])
            for a in map(json.loads, open(p))
            if a.get("words") and os.path.exists(os.path.join(OOV, run, a["wav"]))
        }

    req_clips = _load("p6_full_request_s0")  # Tom: "Please listen carefully and repeat the word X."
    rep_clips = _load("p6_full_reply_s0")  # Gweneth: "Okay, X."
    common = sorted(set(req_clips) & set(rep_clips))
    return common, rep_clips, req_clips


@app.cell
def _(common, rep_clips):
    for k,v in rep_clips.items():
        print(f"{k},{v} \n")
    print(common)
    return


@app.cell
def _(common, mo):
    _real = "real TTS clips"
    source = mo.ui.dropdown(
        options=[_real, "synthetic tone"] if common else ["synthetic tone"],
        value=_real if common else "synthetic tone",
        label="audio",
    )
    word = mo.ui.dropdown(
        options=common or ["water"],
        value="amazing" if "amazing" in common else (common[0] if common else "water"),
        label="word",
    )
    mo.hstack([source, word], justify="start")
    return source, word


@app.cell
def _(
    build_row,
    codec,
    os,
    rep_clips,
    req_clips,
    source,
    tempfile,
    word,
    word_timings,
    write_clip,
):
    # Tom (speaker 0) asks, Gweneth (speaker 1, the model's role) answers "Okay, <word>."; clips back to back (train_rows.build_row)
    _w = word.value
    tmp = tempfile.mkdtemp()
    if source.value == "real TTS clips":
        (_req, _req_words), (_rep, _rep_words) = req_clips[_w], rep_clips[_w]
    else:
        _req, _rep = os.path.join(tmp, "req.wav"), os.path.join(tmp, "rep.wav")
        write_clip(_req, 2.0)
        write_clip(_rep, 1.0)
        _req_words = word_timings(
            lead=0.2,
            parts=[("please", 0.4), ("repeat", 0.4), ("the", 0.2), ("word", 0.4), (_w, 0.6)],
        )
        _rep_words = word_timings(
            lead=0.2,
            parts=[("okay", 0.5), (_w, 0.5)],
        )
    # build_row(turns, codec, *, speakers, narrative, spk_emb, new_narrative_s1, new_narrative_s2, file_name) -> dict
    row = build_row(
        turns=[  # list[dict], in dialogue order
            {"speaker": 0, "wav": _req, "words": _req_words},  # speaker: int (0 = channel 1 = Tom), wav: str, words: list[dict]
            {"speaker": 1, "wav": _rep, "words": _rep_words},  # 1 = channel 2 = Gweneth, the model's role
        ],
        codec=codec,  # NanoCodec model (nn.Module); encodes each channel
        speakers=["Tom", "Gweneth"],  # list[str]: names that appear in the prompt
        narrative="",  # str: unused here, the two role narratives below are what the prompt gets
        new_narrative_s1=f"You ask Gweneth to listen carefully and repeat the word '{_w}'.",  # str: Tom's
        new_narrative_s2=f"Tom asks you to listen carefully and repeat the word '{_w}'. You repeat it.",  # str: Gweneth's
        spk_emb=[[0.01] * 192, [0.02] * 192],  # list[list[float]]: one 192-d vector per speaker (dummy values)
        file_name=_w,  # str
    )
    return row, tmp


@app.cell
def _(row):
    row
    return


@app.cell
def _(
    AutoTokenizer,
    REPO,
    SimpleNamespace,
    load_data,
    logging,
    os,
    row,
    tmp,
    write_dataset,
):
    # the real data.py (tokenize_speech) with the flags of scripts/train/finetune_rows.sh; system_speaker=1 -> Gweneth is "system"
    tok = AutoTokenizer.from_pretrained(str(REPO / "hf_models" / "f_actor"))
    _out = os.path.join(tmp, "ds")
    write_dataset([row] * 4, _out, val_frac=0.25)
    # the three argument groups of the repo (arguments/arguments.py: ModelArgs, DataArgs, TrainingArgs); only the fields data.py reads
    model_args = SimpleNamespace(
        max_length=2048,  # int: frames kept per sample
        num_dsus=4,  # int: codebooks per speaker
        text_stream=True,  # bool: one text stream (the system speaker's)
        multi_text_stream=False,  # bool
    )
    data_args = SimpleNamespace(
        speech_path=_out,  # str: dataset dir with train.parquet + validation.parquet
        system_speaker=1,  # int | None: which channel is the model (1 = Gweneth); None = random per sample
        n_delay_audio_stream=2,  # int: audio delayed by 2 frames vs text
        n_delay_text_stream=0,  # int
        word_alignment=True,  # bool: put each word on the frame where it starts
        add_bc_token=True,  # bool
        add_interrupt_token=True,  # bool
        add_counting_tokens=False,  # bool
        use_system_narrative=True,  # bool: prompt uses new_narrative_s{system+1}
        remove_start_silence=False,  # bool
        debug=False,  # bool
        train_on_subset=None,  # float | None
        preprocessing_num_workers=1,  # int
    )
    training_args = SimpleNamespace(
        train_batch_size=1,  # int
        eval_batch_size=1,  # int
        gradient_accumulation_steps=1,  # int
    )
    # load_data(model_args, data_args, training_args, audio_delay_id, logger, tokenizer, inference=False)
    #   -> (train_dataset, validation_dataset, data_collator)
    _train, _, _collator = load_data(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        audio_delay_id=4032,  # int: id of the delay token (= audio_vocab_size)
        logger=logging.getLogger("trace"),  # logging.Logger
        tokenizer=tok,  # transformers tokenizer (the released f_actor one)
    )
    print(_train)
    batch = _collator([_train[0]])  # batch of 1, as the Trainer feeds the model
    inp = {k: batch[k] for k in ["input_ids", "attention_mask", "dsu_ids", "text_stream_ids", "spk_emb"]}
    return batch, inp, tok


@app.cell
def _(DSULlama, LlamaConfig, tok, torch):
    # tiny random model with the repo's own heads/embeds/speaker projection (cf. model.py:load_model); eval() -> loss on the system half, as eval does
    _cfg = LlamaConfig(
        vocab_size=len(tok),  # int: text vocab incl. the added special tokens
        hidden_size=64,  # int: H (the real model: 2048)
        intermediate_size=128,  # int
        num_hidden_layers=2,  # int (real: 16)
        num_attention_heads=4,  # int
        num_key_value_heads=2,  # int
        tie_word_embeddings=True,  # bool: LM head shares the token embedding, as in the real model
    )
    # the extra fields DSUModel.__init__ reads from the config
    _cfg.num_dsus = 4  # int: codebooks per speaker (-> 8 audio heads)
    _cfg.text_stream = True  # bool
    _cfg.multi_text_stream = False  # bool
    _cfg.audio_vocab_size = 4032  # int: codes 0..4031 (+2 special ids inside the model)
    _cfg.use_speaker_embedding = True  # bool
    _cfg.calc_loss_on_c1_only = True  # bool: loss on the system half only
    torch.manual_seed(0)
    model = DSULlama(_cfg)
    model.pad_token_id, model.grad_acc_steps = tok.pad_token_id, 1
    model.init_dsu_heads()
    model.init_audio_embeds()
    model.init_speaker_embed_proj(192)
    model.eval()
    return (model,)


@app.cell
def _(batch, mo, model, tok):
    # the REAL function; used below only as the thing the unrolled cells must match
    #   -> (embeds [B, N, H], attention_mask [B, N], audio labels [B, 8, L], text labels [B, 1, N])
    ref_emb, ref_att, ref_dl, ref_tl = model.get_input_embeds_and_labels(
        prompt_ids=batch["input_ids"],  # LongTensor [B, max_length]: prompt, padded
        prompt_att_mask=batch["attention_mask"],  # LongTensor [B, max_length]: 1 = real prompt token
        dsu_ids=batch["dsu_ids"],  # LongTensor [B, 8, T+2]: audio codes, 8 heads
        text_stream_ids=batch["text_stream_ids"],  # LongTensor [B, 1, T+2]: system text stream
        spk_emb=batch["spk_emb"],  # list[B] of 192-d vectors
    )
    _ids = batch["input_ids"][0][batch["attention_mask"][0].bool()]
    mo.md(f"""
    ## B. The sample coming out of `data.py`

    | tensor | shape | meaning |
    |---|---|---|
    | `input_ids` | {tuple(batch['input_ids'].shape)} | prompt padded to max_length; only {int(batch['attention_mask'].sum())} real |
    | `dsu_ids` | {tuple(batch['dsu_ids'].shape)} | 8 audio heads x (T + 2 delay frames); heads 0-3 system (Gweneth), 4-7 user (Tom) |
    | `text_stream_ids` | {tuple(batch['text_stream_ids'].shape)} | system speaker's frame-aligned text (1 stream) |
    | `spk_emb` | {len(batch['spk_emb'][0])} | system speaker vector -> 1 position |

    Prompt (`build_prompt`, note `- ` before the narrative):

    ```
    {tok.decode(_ids)}
    ```
    """)
    return ref_att, ref_dl, ref_emb, ref_tl


@app.cell
def _(batch, mo, np, pd, plt, silence_codes, tok, torch):
    # --- 1. what is in the batch: one row per key ---
    _role = {
        "input_ids": "prompt token ids, padded to max_length",
        "attention_mask": "1 = real prompt token, 0 = padding",
        "labels": "collator placeholder, ignored: the model rebuilds its labels from dsu_ids / text_stream_ids",
        "dsu_ids": "audio codes: 8 heads (0-3 system, 4-7 user) x (T + 2 delay frames)",
        "text_stream_ids": "system speaker's text, one token id per frame (same length as dsu_ids)",
        "spk_emb": "system speaker's 192-d vector, becomes 1 position in front of the prompt",
        "spk_emb_c1": "dataset column, not used by the model (speaker 1's vector)",
        "spk_emb_c2": "dataset column, not used by the model (speaker 2's vector)",
    }
    _rows = []
    for _k, _v in batch.items():
        _is_t = torch.is_tensor(_v)
        _rows.append(
            {
                "key": _k,
                "type": "Tensor" if _is_t else type(_v).__name__,
                "dtype": str(_v.dtype).replace("torch.", "") if _is_t else "",
                "shape": str(tuple(_v.shape)) if _is_t else f"len {len(_v)} x {len(_v[0])}",
                "role": _role.get(_k, ""),
            }
        )
    _table = mo.ui.table(pd.DataFrame(_rows), selection=None, show_download=False, page_size=20)

    # --- 2. the same, drawn ---
    _ids = batch["dsu_ids"][0].numpy()  # [8, L]
    _txt = batch["text_stream_ids"][0, 0].numpy()  # [L]
    _mask = batch["attention_mask"][0].numpy()  # [max_length]
    _spk = np.asarray(batch["spk_emb"][0], dtype=float)  # [192]
    _L = _ids.shape[1]

    # audio: 0 speech code, 1 silence code, 2 delay id (4032)
    _sil = np.tile(silence_codes, 2)[:, None]  # silence code per head
    _audio_cat = np.where(_ids == 4032, 2, np.where(_ids == _sil, 1, 0))

    # text: 0 silence pad, 1 utterance pad, 2 word pad, 3 word token
    _pad_ids = {tok.convert_tokens_to_ids(t): i for i, t in enumerate(["[SILENCE_PAD]", "[UTTERANCE_PAD]", "[WORD_PAD]"])}
    _text_cat = np.array([_pad_ids.get(int(i), 3) for i in _txt])

    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    _fig = plt.figure(figsize=(11, 6.2), layout="constrained")
    _ax_p, _ax_a, _ax_t, _ax_s = _fig.subplots(4, 1, height_ratios=[0.45, 3.2, 1.1, 0.45])

    _ax_p.imshow(_mask[None, :], aspect="auto", cmap=ListedColormap(["#d9d9d9", "#2a6fbb"]), interpolation="nearest")
    _ax_p.set_yticks([])
    _ax_p.set_title(
        f"input_ids / attention_mask: {int(_mask.sum())} real prompt tokens (blue) of {len(_mask)} positions (grey = padding, dropped by the model)",
        fontsize=9,
        loc="left",
    )

    _ax_a.imshow(_audio_cat, aspect="auto", cmap=ListedColormap(["#2a6fbb", "#f0f0f0", "#e08a1e"]), vmin=0, vmax=2, interpolation="nearest")
    _ax_a.set_yticks(range(8))
    _ax_a.set_yticklabels([f"h{h}  {'system' if h < 4 else 'user'}" for h in range(8)], fontsize=8)
    _ax_a.axhline(3.5, color="k", lw=1)
    _ax_a.set_xticks([])
    _ax_a.set_title(f"dsu_ids {tuple(batch['dsu_ids'].shape)}: 8 codebook heads x {_L} frames (12.5 per second)", fontsize=9, loc="left")
    _ax_a.legend(
        handles=[
            Patch(color="#2a6fbb", label="speech code"),
            Patch(color="#f0f0f0", ec="#bbbbbb", label="silence code"),
            Patch(color="#e08a1e", label="delay id (4032)"),
        ],
        loc="upper right",
        fontsize=7,
        ncol=3,
        framealpha=0.9,
    )

    _ax_t.imshow(
        _text_cat[None, :],
        aspect="auto",
        cmap=ListedColormap(["#f0f0f0", "#bfe3c8", "#d7efdd", "#1f8a4c"]),
        vmin=0,
        vmax=3,
        interpolation="nearest",
        extent=(-0.5, _L - 0.5, 0, 1),  # strip occupies y in [0, 1]; the word labels go above it
    )
    _ax_t.set_ylim(0, 2.2)
    for _f in np.nonzero(_text_cat == 3)[0]:
        _w = tok.convert_ids_to_tokens(int(_txt[_f])).replace("Ġ", "")
        _ax_t.text(_f, 1.1, _w, ha="center", va="bottom", fontsize=8, color="#1f8a4c")
    _ax_t.set_yticks([])
    _ax_t.set_xlabel("frame (same axis as the audio grid above; text sits 2 frames ahead of its audio)", fontsize=8)
    _ax_t.set_title(f"text_stream_ids {tuple(batch['text_stream_ids'].shape)}: system speaker's words (silence pad / utterance pad / word pad / word token)", fontsize=9, loc="left")
    _ax_t.set_xlim(_ax_a.get_xlim())

    _ax_s.imshow(_spk[None, :], aspect="auto", cmap="RdBu_r", vmin=-np.abs(_spk).max(), vmax=np.abs(_spk).max(), interpolation="nearest")
    _ax_s.set_yticks([])
    _flat = " (all values equal: the dummy vector this notebook passes to build_row, so the strip is flat)" if np.ptp(_spk) == 0 else ""
    _ax_s.set_title(f"spk_emb: {len(_spk)}-d speaker vector, one value per cell{_flat}", fontsize=9, loc="left")

    mo.vstack([_table, _fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## C. Unrolled: `get_input_embeds_and_labels` (`modeling_dsu.py:226-384`, batch element 0, training mode)

    One cell per step. Goal: the sequence `[spk | prompt | frames]` of embeddings, plus the label grids.
    """)
    return


@app.cell
def _(batch, model):
    # step 1 (l.274-275): prompt = non-padded ids, embedded with the Llama token embedding
    embed_layer = model.get_input_embeddings()
    prompt_ids = batch["input_ids"][0, batch["attention_mask"][0].bool()]
    prompt_embeds = embed_layer(prompt_ids)
    P = len(prompt_ids)
    print("prompt ids", tuple(prompt_ids.shape), "-> embeds", tuple(prompt_embeds.shape))
    return P, embed_layer, prompt_embeds, prompt_ids


@app.cell
def _(batch, model, torch):
    # step 2 (l.278-283): audio ids of this sample, trimmed of batch padding, + one audio-EOS column (4033 on all 8 heads)
    dsu_pad_mask = batch["dsu_ids"][0] != model.pad_token_id
    non_pad_len = dsu_pad_mask.sum(dim=1)[0]
    audio_eos = torch.full((8, 1), model.audio_eos_id, dtype=torch.long)
    audio_ids = torch.cat([batch["dsu_ids"][0, :, :non_pad_len], audio_eos], dim=1)
    L = audio_ids.shape[1]
    print("audio ids", tuple(audio_ids.shape), " last column:", audio_ids[:, -1].tolist())
    return L, audio_ids, non_pad_len


@app.cell
def _(audio_ids, model, torch):
    # step 3 (l.285-293): one embedding TABLE PER HEAD, then SUM over the 8 heads -> one vector per frame
    per_head = torch.stack([model.audio_embeds[h](audio_ids[h]) for h in range(8)], dim=0)
    summed_audio = per_head.sum(dim=0)
    print("per head", tuple(per_head.shape), "-> summed", tuple(summed_audio.shape))
    return (summed_audio,)


@app.cell
def _(L, batch, embed_layer, model, non_pad_len, summed_audio, torch):
    # step 4 (l.322-327): the system text stream (+ a pad-id EOS column) is embedded and ADDED to the same frame vectors
    text_eos = torch.full((1, 1), model.pad_token_id, dtype=torch.long)
    text_ids = torch.cat([batch["text_stream_ids"][0, :, :non_pad_len], text_eos], dim=1)  # [1, L]
    frame_embeds = summed_audio + embed_layer(text_ids).sum(dim=0)
    assert text_ids.shape[1] == L
    print("text ids", tuple(text_ids.shape), "-> frame embeds", tuple(frame_embeds.shape))
    return frame_embeds, text_eos, text_ids


@app.cell
def _(batch, frame_embeds, model, prompt_embeds, torch):
    # step 5 (l.296-305,346-348): speaker vector -> 1 position through a Linear; sequence = [spk | prompt | frames]
    spk_pos = model.speaker_embed_proj(torch.tensor(batch["spk_emb"][0], dtype=model.dtype)).unsqueeze(0)
    seq = torch.cat([spk_pos, prompt_embeds, frame_embeds], dim=0)
    print("spk", tuple(spk_pos.shape), "| sequence", tuple(seq.shape))
    return (seq,)


@app.cell
def _(P, audio_ids, prompt_ids, text_eos, text_ids, torch):
    # step 6 (l.329-339,349-375): label grids. audio labels = the audio id grid itself; text labels cover spk + prompt + frames
    dl = audio_ids.unsqueeze(0)  # [1, 8, L]
    tl = torch.cat([text_eos, prompt_ids.unsqueeze(0), text_ids], dim=1).unsqueeze(0)  # [1, 1, 1 + P + L]
    off = 1 + P  # position of grid frame 0 in the sequence
    return dl, off, tl


@app.cell
def _(dl, ref_att, ref_dl, ref_emb, ref_tl, seq, tl, torch):
    assert torch.allclose(seq, ref_emb[0], atol=1e-6), "unrolled sequence != get_input_embeds_and_labels"
    assert torch.equal(dl, ref_dl) and torch.equal(tl, ref_tl)
    assert int(ref_att.sum()) == seq.shape[0]
    print(f"OK: unrolled == real function. sequence {tuple(seq.shape)}, audio labels {tuple(dl.shape)}, text labels {tuple(tl.shape)}")
    return


@app.cell
def _(L, dl, off, pd, tl, tok):
    SYM = {"[SILENCE_PAD]": "·", "[UTTERANCE_PAD]": "_", "[WORD_PAD]": "~"}

    def _tt(
        i: int,  # a text token id
    ) -> str:  # short printable form: pads as one symbol, word tokens as text
        t = tok.convert_ids_to_tokens(int(i))
        return SYM.get(t, t.replace("Ġ", " "))

    text_by_frame = tl[0, 0, off : off + L].numpy()  # [L]: the text token of each grid frame
    grid = pd.DataFrame(
        [
            {
                "frame": f,
                "text": _tt(text_by_frame[f]) if f < L - 1 else "EOS(pad id)",
                **{f"h{h}": int(dl[0, h, f]) for h in range(8)},  # the 8 audio heads
            }
            for f in range(L)
        ]
    ).set_index("frame")
    grid
    return SYM, text_by_frame


@app.cell
def _(L, SYM, dl, mo, silence_codes, source, text_by_frame, tok):
    _pad_ids = {tok.convert_tokens_to_ids(k) for k in SYM}
    _w0 = next(f for f in range(L - 1) if int(text_by_frame[f]) not in _pad_ids)
    _a0 = next(f for f in range(2, L - 1) if int(dl[0, 0, f]) != int(silence_codes[0]))
    _src = (
        "Real TTS clips, so this is the real size of the effect for this clip (it varies by word and clip)."
        if source.value == "real TTS clips"
        else "Caveat: a tone has an abrupt onset; speech onsets are softer, so switch to the real clips."
    )
    mo.md(f"""
    ### Reading the grid

    - `h0-h3` Gweneth (system, trained), `h4-h7` Tom (user). Codes are real NanoCodec codes (0..4031); `4032` delay id, `4033` EOS.
      **Silence is a code, not 0:** digital silence encodes to {silence_codes.tolist()} on codebooks 0-3 (middle frames of 3 s of zeros).
    - `text`: `·` silence pad, `_` utterance pad, `~` word pad, else a word token.
    - **Text leads audio by design:** the first reply word token is in row {_w0}; the audio of that same original frame is in row {_w0 + 2}
      (audio delay 2, `data.py`).
    - **Codes vs words:** `h0` first leaves the digital-silence code in row {_a0}, {_w0 + 2 - _a0} rows ({(_w0 + 2 - _a0) * 80} ms) before the word's audio row {_w0 + 2}.
      That gap is not one thing: (1) the reply clip begins when the request clip ends (the request's length in 12.5 fps frames, +2 for the delay), and
      NanoCodec is not causal, so a few rows *before* that the codes already drift; (2) a TTS clip starts with a lead-in of low-level room tone, which is
      not digital silence and encodes to other codes, until the first word. Check both by eye in the grid (Tom's heads show the mirror image: they settle
      back to the silence code a few rows after his clip ends). {_src}
    - Tom's text never appears (only the system stream exists); his heads carry the request, then settle to the silence code.
    - The last rows before EOS differ from the silence code: the codec's end-of-file codes (see `docs/prefix_log.md`).
    """)
    return


@app.cell
def _(L, SYM, dl, np, plt, silence_codes, text_by_frame, tok):
    _fig, _ax = plt.subplots(figsize=(11, 3.4))
    _a = dl[0].numpy()
    _sil = np.tile(silence_codes, 2)[:, None]  # silence code per head (heads 4-7 use codebooks 0-3 again)
    _ax.imshow((_a != _sil) & (_a < 4032), aspect="auto", cmap="Blues", vmin=0, vmax=1.4, interpolation="nearest")
    _pads = {tok.convert_tokens_to_ids(k) for k in SYM}
    for _f in range(L - 1):
        if int(text_by_frame[_f]) not in _pads:
            _ax.plot(_f, -1, "rs", ms=5)
    _ax.set_yticks(range(8))
    _ax.set_yticklabels([f"h{h} {'system' if h < 4 else 'user'}" for h in range(8)])
    _ax.set_ylim(7.5, -1.6)
    _ax.set_xlim(-0.5, L - 0.5)
    _ax.axvline(1.5, color="k", lw=0.6, ls=":")
    _ax.set_xlabel("grid frame")
    _ax.set_title("audio frames that are not the silence code (blue); word tokens of the system text stream (red, above h0)")
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## D. Unrolled: the heads in `forward` (`modeling_dsu.py:111-159`)

    The Llama runs on the sequence; its last hidden state feeds two heads. Audio uses only the frame positions.
    """)
    return


@app.cell
def _(model, ref_att, seq, torch):
    # l.111-121: Llama on inputs_embeds -> last hidden state (after the final norm)
    with torch.no_grad():
        hidden = model.model(
            inputs_embeds=seq.unsqueeze(0),  # FloatTensor [1, N, H]: the sequence built above
            attention_mask=ref_att,  # LongTensor [1, N]: all ones (no padding at batch size 1)
        ).last_hidden_state  # [1, N, H]
    print("hidden", tuple(hidden.shape))
    return (hidden,)


@app.cell
def _(L, hidden, model, off, torch):
    # l.124-145: audio hidden = the frame positions only (skip spk + prompt) -> ONE Linear gives all 8 heads at once
    audio_hidden = hidden[:, off : off + L, :]  # [1, L, H]
    with torch.no_grad():
        dsu_logits = model.dsu_head(audio_hidden).view(1, L, 8, -1)  # [1, L, 8 heads, 4034]
        text_logits = model.lm_head(hidden).unsqueeze(2)  # l.157-159: tied Llama LM head over ALL positions -> [1, N, 1, V]
    print("audio logits", tuple(dsu_logits.shape), "| text logits", tuple(text_logits.shape))
    return dsu_logits, text_logits


@app.cell
def _(dsu_logits, inp, model, text_logits, torch):
    with torch.no_grad():
        out = model(**inp)  # the REAL forward
    assert torch.allclose(dsu_logits, out["logits"], atol=1e-5) and torch.allclose(text_logits, out["ts_logits"], atol=1e-5)
    print("OK: unrolled heads == real forward logits")
    return (out,)


@app.cell
def _(L, SYM, mo, text_by_frame, tok):
    # default frame: the one just before the reply's first word token, so the text prediction slot is the interesting one
    _pad_ids = {tok.convert_tokens_to_ids(k) for k in SYM}
    _w0 = next(f for f in range(L - 1) if int(text_by_frame[f]) not in _pad_ids)
    f_slider = mo.ui.slider(start=0, stop=L - 2, value=max(_w0 - 1, 0), label="frame f", show_value=True, full_width=True)
    h_slider = mo.ui.slider(start=0, stop=7, value=0, label="audio head h", show_value=True)
    mo.hstack([f_slider, h_slider], widths=[3, 1], justify="start")
    return f_slider, h_slider


@app.cell
def _(
    P,
    dl,
    dsu_logits,
    f_slider,
    h_slider,
    mo,
    np,
    off,
    plt,
    text_logits,
    tl,
    tok,
    torch,
):
    # How to read the two logits tensors: every (frame, head) cell of dsu_logits and every position of text_logits is a
    # VECTOR of scores over classes. Pick a cell; its vector is drawn as probabilities (softmax), with the true next token marked.
    _f, _h, _TOPK = f_slider.value, h_slider.value, 12
    _n = text_logits.shape[1]  # 182 positions
    _pos = off + _f  # sequence position of grid frame f

    def _label_audio(i: int) -> str:
        return {4032: "delay", 4033: "EOS"}.get(i, str(i))

    def _label_text(i: int) -> str:
        return tok.convert_ids_to_tokens(int(i)).replace("Ġ", "·")

    def _top(
        logits_vec: torch.Tensor,  # [V] one cell's scores
        target: int,  # the true next token id
        label,  # id -> short string
    ):
        p = torch.softmax(logits_vec.float(), dim=-1)
        top_p, top_i = p.topk(_TOPK)
        rank = int((p > p[target]).sum()) + 1
        return p, top_p.numpy(), top_i.numpy(), float(p[target]), rank

    _a_target = int(dl[0, _h, _f + 1])  # true code of head h at frame f+1
    _t_target = int(tl[0, 0, _pos + 1])  # true text token at frame f+1
    _pa, _ta_p, _ta_i, _pa_t, _ra = _top(dsu_logits[0, _f, _h], _a_target, _label_audio)
    _pt, _tt_p, _tt_i, _pt_t, _rt = _top(text_logits[0, _pos, 0], _t_target, _label_text)

    _fig, _ax = plt.subplots(2, 2, figsize=(11, 6.6), width_ratios=[1.35, 1], layout="constrained")

    # --- audio: the [frame x head] grid, selected cell, then that cell's vector ---
    _g = np.where(np.arange(8)[:, None] < 4, 0.35, 0.12) * np.ones((8, dsu_logits.shape[1]))
    _ax[0, 0].imshow(_g, aspect="auto", cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    _ax[0, 0].add_patch(plt.Rectangle((_f - 0.5, _h - 0.5), 1, 1, fill=False, ec="#d62728", lw=2.5))
    _ax[0, 0].set_yticks(range(8))
    _ax[0, 0].set_yticklabels([f"h{k}" for k in range(8)], fontsize=8)
    _ax[0, 0].set_xlabel("frame (89 rows of dsu_logits)", fontsize=8)
    _ax[0, 0].set_title(f"audio logits (1, {dsu_logits.shape[1]}, 8, {dsu_logits.shape[-1]}): each cell is a vector of {dsu_logits.shape[-1]} scores", fontsize=9, loc="left")
    _ax[0, 1].barh(range(_TOPK), _ta_p[::-1], color="#2a6fbb")
    _ax[0, 1].set_yticks(range(_TOPK))
    _ax[0, 1].set_yticklabels([_label_audio(int(i)) + ("  <- target" if int(i) == _a_target else "") for i in _ta_i[::-1]], fontsize=8)
    _ax[0, 1].axvline(1 / dsu_logits.shape[-1], color="k", ls=":", lw=1)
    _ax[0, 1].set_title(
        f"cell (frame {_f}, head {_h}): top {_TOPK} of {dsu_logits.shape[-1]} codes\ntrue next code {_label_audio(_a_target)}: p={_pa_t:.5f}, rank {_ra} (dotted = uniform 1/{dsu_logits.shape[-1]})",
        fontsize=8,
        loc="left",
    )
    _ax[0, 1].set_xlabel("probability (softmax of the scores)", fontsize=8)

    # --- text: the sequence of 182 positions, selected position, then that position's vector ---
    _reg = np.zeros((1, _n))
    _reg[0, 1 : 1 + P] = 0.5  # prompt
    _reg[0, off:] = 1.0  # frames
    _ax[1, 0].imshow(_reg, aspect="auto", cmap="Greens", vmin=0, vmax=1.4, interpolation="nearest", extent=(-0.5, _n - 0.5, 0, 1))
    _ax[1, 0].add_patch(plt.Rectangle((_pos - 0.5, 0), 1, 1, fill=False, ec="#d62728", lw=2.5))
    _ax[1, 0].set_yticks([])
    _ax[1, 0].set_xlabel(f"sequence position (all {_n} rows of text_logits): spk | prompt (92) | frames (89)", fontsize=8)
    _ax[1, 0].set_title(f"text logits (1, {_n}, 1, {text_logits.shape[-1]}): each position is a vector of {text_logits.shape[-1]} scores", fontsize=9, loc="left")
    _ax[1, 1].barh(range(_TOPK), _tt_p[::-1], color="#1f8a4c")
    _ax[1, 1].set_yticks(range(_TOPK))
    _ax[1, 1].set_yticklabels([_label_text(int(i)) + ("  <- target" if int(i) == _t_target else "") for i in _tt_i[::-1]], fontsize=8)
    _ax[1, 1].axvline(1 / text_logits.shape[-1], color="k", ls=":", lw=1)
    _ax[1, 1].set_title(
        f"position {_pos} (= frame {_f}): top {_TOPK} of {text_logits.shape[-1]} text tokens\ntrue next token {_label_text(_t_target)}: p={_pt_t:.6f}, rank {_rt}",
        fontsize=8,
        loc="left",
    )
    _ax[1, 1].set_xlabel("probability (softmax of the scores)", fontsize=8)

    mo.vstack(
        [
            _fig,
            mo.md(
                "Random tiny model, so the distributions are nearly flat (the dotted line is a uniform guess) and the true token's rank is arbitrary. "
                "A trained model puts most of the mass on one or a few classes; the loss is `-log p(target)` for every cell the mask keeps."
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## E. Unrolled: the loss (`modeling_dsu.py:165-204`), audio first

    Position `i` is scored on the token at `i+1`; only the system half (heads 0-3) counts (`calc_loss_on_c1_only`, and always in eval).
    """)
    return


@app.cell
def _(F, dl, model):
    # l.170-171: shift labels by one: target of position f = label of frame f+1; the last position gets pad (= ignored)
    _padded = F.pad(dl, (0, 1), value=model.pad_token_id)
    a_labels = _padded[..., 1:]  # [1, 8, L]
    # l.173-179: keep the system half
    a_labels = a_labels[:, :4, :]
    a_mask = a_labels != model.pad_token_id  # l.181
    print("scored audio targets:", int(a_mask.sum()), "= 4 heads x", int(a_mask.sum()) // 4, "frames")
    return a_labels, a_mask


@app.cell
def _(F, a_labels, a_mask, dsu_logits, torch):
    # l.182-193: cross-entropy per token, masked mean (per conversation), then mean over conversations
    _lg = dsu_logits[:, :, :4, :].permute(0, 2, 1, 3)  # [1, 4, L, V]
    _target = torch.where(a_mask, a_labels, torch.zeros_like(a_labels))
    a_ce = F.cross_entropy(
        input=_lg.reshape(-1, _lg.shape[-1]),  # [4*L, V]: logits, one row per (head, frame)
        target=_target.reshape(-1),  # [4*L]: class ids (pad positions replaced by 0, masked out below)
        reduction="none",  # keep per-token losses so the mask can drop pads
    ).view_as(_target)  # [1, 4, L]
    audio_loss = (a_ce * a_mask).sum() / a_mask.sum().clamp(min=1)
    return (audio_loss,)


@app.cell
def _(mo):
    mo.md(r"""
    ### ... and text

    Same recipe, but the text labels span the **whole** sequence (spk, prompt, frames), and there is one text row so nothing is sliced.
    """)
    return


@app.cell
def _(F, model, text_logits, tl, torch):
    _padded = F.pad(tl, (0, 1), value=model.pad_token_id)
    t_labels = _padded[..., 1:]  # [1, 1, N]
    t_mask = t_labels != model.pad_token_id
    _target = torch.where(t_mask, t_labels, torch.zeros_like(t_labels))
    _lg = text_logits.permute(0, 2, 1, 3)  # [1, 1, N, V]
    t_ce = F.cross_entropy(
        input=_lg.reshape(-1, _lg.shape[-1]),  # [N, V]: logits, one row per sequence position
        target=_target.reshape(-1),  # [N]: class ids (pad positions replaced by 0, masked out below)
        reduction="none",
    ).view_as(_target)  # [1, 1, N]
    text_loss = (t_ce * t_mask).sum() / t_mask.sum().clamp(min=1)
    return t_labels, t_mask, text_loss


@app.cell
def _(audio_loss, mo, out, text_loss):
    assert abs(float(audio_loss) - float(out["c1_dsu_loss"])) < 1e-4
    assert abs(float(text_loss) - float(out["c1_text_loss"])) < 1e-4
    mo.md(f"""
    OK: unrolled losses == the model's. audio = {float(audio_loss):.4f}, text = {float(text_loss):.4f}, **total = audio + text = {float(audio_loss + text_loss):.4f}**
    (`/ grad_acc_steps` happens after, l.203-204). The two terms are separate masked means, so each counts equally whatever its token count.
    """)
    return


@app.cell
def _(L, P, mo, t_labels, t_mask, tok):
    _pads = {tok.convert_tokens_to_ids(k): k for k in ("[SILENCE_PAD]", "[UTTERANCE_PAD]", "[WORD_PAD]")}
    _kind = {}
    for _j in range(1 + P + L):
        if t_mask[0, 0, _j]:
            _k = "prompt tokens" if _j < P else _pads.get(int(t_labels[0, 0, _j]), "word tokens")
            _kind[_k] = _kind.get(_k, 0) + 1
    _tot = sum(_kind.values())
    _rows = "\n".join(f"| {k} | {v} | {100 * v / _tot:.0f} % |" for k, v in sorted(_kind.items(), key=lambda kv: -kv[1]))
    mo.md(f"""
    ### What the text mean is made of ({_tot} scored targets)

    | text target | count | share |
    |---|---|---|
    {_rows}

    The word tokens (what you want the model to say) are the small slice; most of the text term is the prompt and pad classes.
    Toy row: check this on a real row from `p8_rows`.
    """)
    return


@app.cell
def _(L, P, a_mask, off, plt, t_mask, torch):
    _n = 1 + P + L
    _fig, _ax = plt.subplots(figsize=(11, 1.9))
    _img = torch.zeros(2, _n)
    _img[0, :] = t_mask[0, 0].float()
    _img[1, off : off + L] = a_mask[0].any(dim=0).float()
    _ax.imshow(_img, aspect="auto", cmap="Greens", vmin=0, vmax=1.3, interpolation="nearest")
    _ax.set_yticks([0, 1])
    _ax.set_yticklabels(["text scored", "audio scored"])
    _ax.axvline(0.5, color="k", lw=0.8)
    _ax.axvline(off - 0.5, color="k", lw=0.8)
    _ax.text(0, 2.0, "spk", fontsize=8)
    _ax.text(P / 2, 2.0, "prompt", fontsize=8)
    _ax.text(off + L / 2, 2.0, "frames", fontsize=8)
    _ax.set_xlabel("sequence position (position i is scored on the token at i+1)")
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## F. Debugger entry

    The repo functions above are called for real in two places (`get_input_embeds_and_labels` in section B, `model(**inp)` in D). To step *inside* them:

    ```
    TRACE_BREAK=1 .venv/bin/python notebooks/01_row_to_sample.py
    ```
    stops in `pdb` before the call below. `s` first lands in torch's call wrapper, so set breakpoints in the repo instead, then `c`:
    `b modeling_dsu.py:63` (`forward`), `b modeling_dsu.py:226` (`get_input_embeds_and_labels`), `b modeling_dsu.py:168` (loss loop).
    `n` next line, `s` into a call, `p seq.shape` print, `l` list source. Cells run in dependency order in script mode, so everything above is already built.
    """)
    return


@app.cell
def _(inp, model, os):
    if os.environ.get("TRACE_BREAK"):
        import pdb

        pdb.set_trace()  # step with `s` into the next line
        _ = model(**inp)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## G. Summary: what gets multiplied with what

    Four pictures, in the order the data flows. Every block is a **real tensor from this notebook** (random weights, so the colours are noise;
    what matters are the shapes and which block is the *same* tensor in two places: same border colour = same tensor).
    Block sizes are schematic, the shapes in the captions are exact. `H` = hidden size (64 here, 2048 in the real model).
    """)
    return


@app.cell
def _(np, plt, torch):
    def draw_matrix(
        fig: plt.Figure,  # figure to draw into
        rect: tuple[float, float, float, float],  # (left, bottom, width, height) as fractions of the figure
        data: torch.Tensor,  # 2-D tensor to show (very wide/tall ones are strided down for display)
        caption: str,  # name + real shape, printed under the block
        cmap: str = "RdBu_r",  # diverging for weights/activations, "Greys" for one-hot
        edge: str = "0.3",  # border colour: the same colour means the same tensor
    ) -> plt.Axes:
        d = data.detach().float().numpy()
        _rows, _cols = d.shape  # true shape: axes keep these units even if the image is strided down
        d = d[:: max(1, d.shape[0] // 400), :: max(1, d.shape[1] // 400)]  # strided (not averaged: averaging noise turns it white)
        finite = d[np.isfinite(d)]
        vmax = float(np.abs(finite).max()) if finite.size else 1.0
        ax = fig.add_axes(rect)
        if cmap == "RdBu_r":
            ax.imshow(np.ma.masked_invalid(d), aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax, interpolation="nearest", extent=(-0.5, _cols - 0.5, _rows - 0.5, -0.5))
        else:
            ax.imshow(d, aspect="auto", cmap=cmap, vmax=float(np.percentile(finite, 99)) or 1.0, interpolation="nearest", extent=(-0.5, _cols - 0.5, _rows - 0.5, -0.5))
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(edge)
            s.set_linewidth(2.5)
        ax.set_xlabel(caption, fontsize=8)
        return ax

    def op(
        fig: plt.Figure,  # figure to draw into
        x: float,  # figure fraction
        y: float,  # figure fraction
        s: str,  # the symbol: "×", "=", "+", ...
        size: int = 20,
    ) -> None:
        fig.text(x, y, s, ha="center", va="center", fontsize=size)

    def onehot_img(
        ids: torch.Tensor,  # [n] token ids
        vocab: int,  # real vocabulary size
        width: int = 300,  # display width (the real one-hot is `vocab` wide; ids are binned and drawn 3 px wide)
    ) -> torch.Tensor:  # [n, width]
        img = torch.zeros(len(ids), width)
        col = (ids * width // vocab).clamp(max=width - 1)
        for dx in (-1, 0, 1):
            img[torch.arange(len(ids)), (col + dx).clamp(0, width - 1)] = 1.0
        return img

    return draw_matrix, onehot_img, op


@app.cell
def _(L, audio_ids, draw_matrix, embed_layer, frame_embeds, mo, model, onehot_img, op, plt, text_ids):
    # picture 1: one frame vector = sum over the 8 audio heads (one-hot x own table) + text (one-hot x Llama token table)
    _Va = model.audio_embeds[0].num_embeddings
    _Vt = embed_layer.num_embeddings
    _H = frame_embeds.shape[1]
    _fig = plt.figure(figsize=(13, 4.4))
    _fig.suptitle("1. input: a frame vector = Σ over 8 audio heads (one-hot × that head's table) + text token (one-hot × Llama token table)", fontsize=10)
    draw_matrix(_fig, (0.02, 0.2, 0.08, 0.6), onehot_img(audio_ids[0], _Va), f"one-hot of head 0's ids\n[L={L}, {_Va}]", "Greys")
    op(_fig, 0.12, 0.5, "×")
    draw_matrix(_fig, (0.14, 0.2, 0.035, 0.6), model.audio_embeds[0].weight, f"E_audio[0]\n[{_Va}, {_H}]", edge="C0")
    _fig.text(0.255, 0.5, "+  ⋯ 7 more heads ⋯  +\neach with its OWN table\nE_audio[1] … E_audio[7]\n+", ha="center", va="center", fontsize=9)
    draw_matrix(_fig, (0.33, 0.2, 0.08, 0.6), onehot_img(text_ids[0], _Vt), f"one-hot of text ids\n[L={L}, {_Vt}]", "Greys")
    op(_fig, 0.43, 0.5, "×")
    draw_matrix(_fig, (0.45, 0.2, 0.035, 0.6), embed_layer.weight, f"E_text\n[{_Vt}, {_H}]", edge="C3")
    op(_fig, 0.52, 0.5, "=")
    draw_matrix(_fig, (0.54, 0.2, 0.05, 0.6), frame_embeds, f"frame_embeds\n[L={L}, H={_H}]", edge="C2")
    _fig.text(
        0.62,
        0.5,
        "In code this is a table lookup\n`E[ids]` (same result, no 4034-wide\none-hot is ever built).\n\n"
        "9 lookups, 1 sum: one vector per\n80 ms frame. The 8 audio heads and\nthe text token all land in the same\nH-dim space and are simply added.",
        va="center",
        fontsize=9,
    )
    mo.as_html(_fig)
    return


@app.cell
def _(L, P, batch, draw_matrix, embed_layer, frame_embeds, model, mo, onehot_img, op, plt, prompt_embeds, prompt_ids, seq, torch):
    # picture 2: assembling the sequence X = [spk ; prompt ; frames] (rows are stacked, not multiplied)
    _N, _H = seq.shape
    _Vt = embed_layer.num_embeddings
    _fig = plt.figure(figsize=(13, 7))
    _fig.suptitle("2. building the sequence X: three sources, stacked as rows (row = one position Llama attends over)", fontsize=10)
    _spk = torch.tensor(batch["spk_emb"][0]).float().unsqueeze(0)
    draw_matrix(_fig, (0.02, 0.83, 0.12, 0.03), _spk, "speaker vector [1, 192]\n(dummy constant in this notebook)")
    op(_fig, 0.16, 0.845, "×")
    draw_matrix(_fig, (0.18, 0.74, 0.03, 0.2), model.speaker_embed_proj.weight.T, f"W_spk (+bias)\n[192, {_H}]", edge="C4")
    op(_fig, 0.24, 0.845, "=")
    draw_matrix(_fig, (0.26, 0.83, 0.03, 0.03), seq[0:1], f"[1, {_H}]", edge="C0")
    draw_matrix(_fig, (0.02, 0.4, 0.12, 0.22), onehot_img(prompt_ids, _Vt), f"one-hot of the prompt ids\n[P={P}, {_Vt}]", "Greys")
    op(_fig, 0.16, 0.51, "×")
    draw_matrix(_fig, (0.18, 0.4, 0.03, 0.22), embed_layer.weight, f"E_text (the SAME table\nas in picture 1)\n[{_Vt}, {_H}]", edge="C3")
    op(_fig, 0.24, 0.51, "=")
    draw_matrix(_fig, (0.26, 0.4, 0.03, 0.22), prompt_embeds, f"[P={P}, {_H}]", edge="C1")
    draw_matrix(_fig, (0.26, 0.06, 0.03, 0.22), frame_embeds, f"frame_embeds from picture 1\n[L={L}, {_H}]", edge="C2")
    _fig.text(0.02, 0.17, "frames: nothing new to multiply here,\nthey come out of picture 1", fontsize=9, va="center")
    _top, _hgt = 0.92, 0.86
    _ax = draw_matrix(_fig, (0.66, 0.06, 0.08, _hgt), seq, f"X = [spk ; prompt ; frames]\n[N={_N}, H={_H}]", edge="0.3")
    for _start, _n, _col, _name in [(0, 1, "C0", "spk (1 row)"), (1, P, "C1", f"prompt ({P} rows)"), (1 + P, L, "C2", f"frames ({L} rows)")]:
        _ax.add_patch(plt.Rectangle((-0.5, _start - 0.5), _H, _n, fill=False, ec=_col, lw=3))
        _yc = _top - (_start + _n / 2) / _N * _hgt
        _fig.text(0.755, _yc, f"◀ {_name}", color=_col, va="center", fontsize=10)
        _src_y = {"C0": 0.845, "C1": 0.51, "C2": 0.17}[_col]
        _fig.add_artist(plt.Line2D([0.295, 0.66], [_src_y, _yc], transform=_fig.transFigure, color=_col, lw=1.2, ls="--"))
    mo.as_html(_fig)
    return


@app.cell
def _(mo, model):
    attn_head = mo.ui.slider(
        start=0,
        stop=model.config.num_attention_heads - 1,
        value=0,
        label="picture 3: attention head of layer 0",
        show_value=True,
    )
    attn_head
    return (attn_head,)


@app.cell
def _(attn_head, draw_matrix, mo, model, op, plt, seq, torch):
    # picture 3: Llama layer 0, one head. Recomputed from the layer's own weights (RoPE left out, so scores differ slightly from the real layer)
    _layer = model.model.layers[0]
    _at = _layer.self_attn
    _cfg = model.config
    _nh, _nkv, _d, _h = _cfg.num_attention_heads, _cfg.num_key_value_heads, _at.head_dim, attn_head.value
    _kvh = _h // (_nh // _nkv)
    _N, _H = seq.shape
    with torch.no_grad():
        _x = _layer.input_layernorm(seq.unsqueeze(0))[0]
        _wq = _at.q_proj.weight[_h * _d : (_h + 1) * _d].T
        _wk = _at.k_proj.weight[_kvh * _d : (_kvh + 1) * _d].T
        _wv = _at.v_proj.weight[_kvh * _d : (_kvh + 1) * _d].T
        _q, _k, _v = _x @ _wq, _x @ _wk, _x @ _wv
        _s = _q @ _k.T / _d**0.5
        _s_masked = _s.masked_fill(torch.triu(torch.ones(_N, _N, dtype=torch.bool), 1), float("nan"))
        _a = torch.softmax(_s.masked_fill(torch.triu(torch.ones(_N, _N, dtype=torch.bool), 1), float("-inf")), dim=-1)
        _o = _a @ _v
    _fig = plt.figure(figsize=(14, 7))
    _fig.suptitle(f"3. inside Llama, layer 0, head {_h}: every position looks only at itself and EARLIER positions (causal)", fontsize=10)
    draw_matrix(_fig, (0.02, 0.58, 0.05, 0.34), _x, f"x̂ = RMSNorm(X)\n[{_N}, {_H}]", edge="0.3")
    for _x0, _w, _r, _nm, _src in [(0.09, _wq, _q, "Q", "q_proj"), (0.24, _wk, _k, "K", "k_proj"), (0.39, _wv, _v, "V", "v_proj")]:
        op(_fig, _x0, 0.75, "×")
        draw_matrix(_fig, (_x0 + 0.015, 0.64, 0.03, 0.22), _w, f"W_{_nm.lower()} (this head's\n{_d} cols of {_src})\n[{_H}, {_d}]", edge="C4")
        op(_fig, _x0 + 0.07, 0.75, "=")
        draw_matrix(_fig, (_x0 + 0.09, 0.58, 0.04, 0.34), _r, f"{_nm}\n[{_N}, {_d}]", edge="C1")
    draw_matrix(_fig, (0.02, 0.08, 0.04, 0.34), _q, f"Q [{_N}, {_d}]", edge="C1")
    op(_fig, 0.08, 0.25, "×")
    draw_matrix(_fig, (0.095, 0.17, 0.1, 0.16), _k.T, f"Kᵀ [{_d}, {_N}]", edge="C1")
    op(_fig, 0.215, 0.25, "=")
    draw_matrix(_fig, (0.235, 0.08, 0.13, 0.34), _s_masked, f"scores QKᵀ/√d  [{_N}, {_N}]\nblank = future, masked", edge="C5")
    _fig.text(0.395, 0.25, "softmax\nper row\n▶", ha="center", va="center", fontsize=9)
    draw_matrix(_fig, (0.425, 0.08, 0.13, 0.34), _a, f"attention A  [{_N}, {_N}]\nrows sum to 1", "viridis", edge="C5")
    op(_fig, 0.575, 0.25, "×")
    draw_matrix(_fig, (0.595, 0.08, 0.04, 0.34), _v, f"V [{_N}, {_d}]", edge="C1")
    op(_fig, 0.655, 0.25, "=")
    draw_matrix(_fig, (0.675, 0.08, 0.04, 0.34), _o, f"O [{_N}, {_d}]", edge="C6")
    _fig.text(
        0.74,
        0.25,
        f"{_nh} heads in parallel ({_nkv} K/V\nheads shared, GQA), O's concatenated\n"
        f"[{_N}, {_nh * _d}] then:\n  × W_o [{_nh * _d}, {_H}], + x̂ (residual)\n"
        f"  × W_gate, W_up [{_H}, {_cfg.intermediate_size}]\n  SiLU(gate)·up\n  × W_down [{_cfg.intermediate_size}, {_H}], + residual\n"
        f"= the layer's output [{_N}, {_H}]\nrepeated for {_cfg.num_hidden_layers} layers (real: 16)",
        va="center",
        fontsize=8.5,
    )
    mo.as_html(_fig)
    return


@app.cell
def _(L, dsu_logits, embed_layer, hidden, mo, model, off, plt, text_logits):
    # picture 4: the output side. Two matmuls with the final hidden states; the text one re-uses the input table.
    _H = hidden.shape[-1]
    _N = hidden.shape[1]
    _Va = dsu_logits.shape[-1]
    _Vt = text_logits.shape[-1]
    _fig = plt.figure(figsize=(14, 7.5))
    _fig.suptitle("4. output: hidden states × heads. ONE matmul gives all 8 audio heads; the text head is the input table turned around (tied)", fontsize=10)
    # audio
    draw_matrix(_fig, (0.02, 0.58, 0.04, 0.32), hidden[0, off : off + L], f"hidden at the frame\npositions only\n[L={L}, {_H}]", edge="C2")
    op(_fig, 0.085, 0.74, "×")
    _ax = draw_matrix(_fig, (0.1, 0.64, 0.4, 0.2), model.dsu_head.weight.T, f"W_dsu (+bias)  [{_H}, 8·{_Va} = {8 * _Va}]", edge="C4")
    op(_fig, 0.525, 0.74, "=")
    _ax2 = draw_matrix(_fig, (0.55, 0.58, 0.43, 0.32), dsu_logits[0].reshape(L, -1), f"audio logits  [L={L}, 8·{_Va}]  →  .view(L, 8, {_Va})", edge="0.3")
    for _a in (_ax, _ax2):
        for _hd in range(1, 8):
            _a.axvline(_hd * _Va - 0.5, color="k", lw=1)
    for _i in range(8):
        _ax2.text((_i + 0.5) * _Va, -0.04 * L, f"h{_i}", ha="center", va="bottom", fontsize=8, color="C0" if _i < 4 else "0.5")
    _fig.text(0.55, 0.935, "heads 0-3 = system speaker (scored)", color="C0", fontsize=8)
    _fig.text(0.78, 0.935, "heads 4-7 = other speaker (not scored)", color="0.5", fontsize=8)
    # text
    draw_matrix(_fig, (0.02, 0.1, 0.04, 0.4), hidden[0], f"hidden at ALL\npositions\n[N={_N}, {_H}]", edge="C2")
    op(_fig, 0.085, 0.3, "×")
    draw_matrix(_fig, (0.1, 0.2, 0.4, 0.2), embed_layer.weight.T, f"E_textᵀ  [{_H}, {_Vt}]   (the INPUT table of picture 1, transposed: tied weights)", edge="C3")
    op(_fig, 0.525, 0.3, "=")
    draw_matrix(_fig, (0.55, 0.1, 0.43, 0.4), text_logits[0, :, 0], f"text logits  [N={_N}, {_Vt}]", edge="0.3")
    mo.as_html(_fig)
    return


@app.cell
def _(embed_layer, mo, model, pd):
    # parameter count: where the weights live, this tiny model vs the real one (Llama-3.2-1B dims; text vocab assumed = len(tokenizer) here)
    def _llama_params(H: int, I: int, layers: int, nh: int, nkv: int, d: int) -> int:
        attn = H * nh * d + 2 * H * nkv * d + nh * d * H
        return layers * (attn + 3 * H * I + 2 * H)

    _Va = model.audio_embeds[0].num_embeddings
    _Vt, _H = embed_layer.weight.shape
    _cfg = model.config
    _real = dict(H=2048, I=8192, layers=16, nh=32, nkv=8, d=64)
    _rows = {
        "text embedding = LM head (tied)": (embed_layer.weight.numel(), _Vt * _real["H"]),
        "8 audio embedding tables": (sum(p.numel() for p in model.audio_embeds.parameters()), 8 * _Va * _real["H"]),
        "audio head W_dsu (H → 8·4034)": (sum(p.numel() for p in model.dsu_head.parameters()), _real["H"] * 8 * _Va + 8 * _Va),
        "speaker projection (192 → H)": (sum(p.numel() for p in model.speaker_embed_proj.parameters()), 192 * _real["H"] + _real["H"]),
        "Llama transformer layers": (
            sum(p.numel() for p in model.model.layers.parameters()),
            _llama_params(_real["H"], _real["I"], _real["layers"], _real["nh"], _real["nkv"], _real["d"]),
        ),
    }
    _df = pd.DataFrame(
        [{"component": k, "this tiny model": f"{a / 1e6:.2f} M", "real model (est.)": f"{b / 1e6:.0f} M"} for k, (a, b) in _rows.items()]
    )
    mo.vstack([mo.md("**Where the weights live** (the audio parts are new and trained from scratch/fine-tuned, the rest is Llama):"), _df])
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            r"""
            **One line per stage**

            1. `X = [ W_spk·spk ; E_text[prompt] ; Σ₈ E_audio[h][ids_h] + E_text[text_ids] ]`  → `[N, H]`
            2. `hidden = Llama(X)`: per layer `softmax(QKᵀ/√d + causal mask)·V`, `×W_o`, MLP → `[N, H]`
            3. `audio logits = hidden[frames] · W_dsu`  → `[L, 8, 4034]` (all heads in parallel, no depth transformer)
            4. `text logits  = hidden · E_textᵀ`  → `[N, V]` (same table as in 1)
            5. loss: position `i` is scored on the token at `i+1`; audio heads 0-3 only; audio and text are each a masked mean, then summed.
            """
        ),
        kind="info",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Next sections (not built yet)

    3. training vs inference side by side on this same row (teacher forcing vs sampling)
    4. the same on a real `p8_rows` row
    """)
    return


if __name__ == "__main__":
    app.run()

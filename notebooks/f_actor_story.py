"""F-Actor: the architecture, and the ideas behind the "repeat the word" fine-tune.

A read-only story notebook. It trains nothing and loads no model: every number and clip comes from files written by earlier runs,
collected by `build_story_assets.py` into `f_actor_story_assets/` (see its MANIFEST.json for the audio).

    uv run --extra analysis marimo edit notebooks/f_actor_story.py        # or: .venv/bin/marimo edit ...
    .venv/bin/marimo run notebooks/f_actor_story.py                        # read-only app view
"""
import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium", app_title="F-Actor: architecture and the repeat-the-word fine-tune")


@app.cell
def _():
    import json
    import re
    from collections import Counter
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
    return Counter, Path, json, mo, np, pd, plt, re


@app.cell
def _(Path, json, mo, np):
    ROOT = Path(mo.notebook_dir()).parent
    ASSETS = Path(mo.notebook_dir()) / "f_actor_story_assets"
    DATA = json.load(open(ASSETS / "data.json"))
    MAN = json.load(open(ASSETS / "MANIFEST.json"))
    Z = np.load(ASSETS / "rows.npz")
    BLUE, ORANGE, GREY, RED, GREEN = "#3b6ea8", "#d9822b", "#8a8f98", "#c0392b", "#3f8f5b"

    def clip(name):
        """Audio player + provenance caption from MANIFEST.json (what it is, how it was made, where the file is)."""
        m = MAN[name]
        extra = f" · TTS `{m['tts']}`" if m.get("tts") else ""
        return mo.vstack([
            mo.md(f"**{m['title']}**"),
            mo.audio(ASSETS / "clips" / f"{name}.ogg"),
            mo.md(f"<small>{m['how']}<br>source: `oov-data/{m['source']}` · file date {m['date']} · {m['seconds']} s, {m['channels']} ch{extra}</small>"),
        ], gap=0.3)

    def source_lines(relpath, needle, before=0, after=10):
        """Verbatim excerpt of a repo file, located at run time by a search string (so it cannot drift from the code)."""
        lines = (ROOT / relpath).read_text().splitlines()
        i = next(k for k, l in enumerate(lines) if needle in l)
        a, b = max(0, i - before), min(len(lines), i + after)
        body = "\n".join(lines[a:b])
        return mo.md(f"`{relpath}` lines {a + 1}-{b}\n```python\n{body}\n```")

    def mdt(df):
        """DataFrame -> markdown table (long text wraps, unlike mo.ui.table)."""
        esc = lambda v: str(v).replace("|", "\\|").replace("\n", " ")
        head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
        return mo.md(head + "\n".join("| " + " | ".join(esc(v) for v in row) + " |" for row in df.itertuples(index=False)))

    def cite(text):
        return mo.md(f"<small style='color:#666'>{text}</small>")

    return ASSETS, BLUE, DATA, GREEN, GREY, MAN, ORANGE, RED, ROOT, Z, cite, clip, mdt, source_lines


@app.cell
def _(DATA, mo):
    b = DATA["built"]
    mo.md(f"""
# F-Actor: how it works, and what we tried to teach it

*Josef · working notebook for discussion with the advisor · state of the repo and logs on **{b['date']}***

This notebook tells one story in four parts:

1. **The model** (§1-3): what F-Actor actually consumes and predicts, read from the code in `training/`, not from the paper.
2. **The problem** (§4): out-of-vocabulary (OOV) words fail in the *audio*, not in the planned text.
3. **Two ideas to act on it** (§5-6): put the target word into the context as a spoken prefix; fine-tune with LoRA on synthetic
   "repeat the word" dialogues.
4. **What went wrong and what is still open** (§7-8): a stop-token detective story, and a result that is still negative.

**Conventions.** Nothing here is recomputed with the model. Numbers come from logs, run records and probe outputs written earlier; each
figure or table says where. Where I quote a number from `docs/prefix_log.md` instead of loading data, it says "quoted". Every audio clip
carries a caption saying whether it is released-model output, our model output, or TTS, and with which settings. **Stereo clips: left channel = speaker 1,
right = speaker 2.** Section 8 lists corrections to my own earlier statements.

<small style='color:#666'>Built from branch `{b['branch']}`, last commit {b['commit'][:60]}…; assets collected {b['date']} by
`notebooks/build_story_assets.py` from the local `oov-data/` tree (not in git). Uncommitted work in the same worktree
(stop mask in `modeling_dsu.py`, single-exchange dataset scripts) is what produced the `single_v1` results.</small>
""")
    return


# ----------------------------------------------------------------------------------------------------------------------------------
@app.cell
def _(mo):
    mo.md(r"""
---
## 1 · The model: two audio streams and one text stream in a single Llama

F-Actor is a Llama-3.2-1B-style decoder (16 layers, hidden 2048) that does **not** produce one token per step. At every **80 ms frame** it
reads and writes *both* speakers' audio at once, plus a time-aligned text stream. So "full duplex" is literal: one forward step covers
"what I say now" and "what I hear now".
""")
    return


@app.cell
def _(DATA, mo, pd):
    m = DATA["model"]
    _V = m["audio_vocab_size"] + 2
    _tbl = pd.DataFrame([
        ("backbone", f"Llama, {m['num_hidden_layers']} layers, hidden {m['hidden_size']}, {m['num_attention_heads']} heads / {m['num_key_value_heads']} KV heads, MLP {m['intermediate_size']}", "config.json"),
        ("text vocabulary", f"{m['vocab_size']} in the release (+61 special tokens added at load time → 128321; see §2)", "config.json; docs/prefix_log.md 2026-10-31"),
        ("audio codec", "NanoCodec, 22 kHz, 0.6 kbps, 12.5 frames/s → **80 ms/frame**, 4 codebooks per speaker", "`nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps`, training/model.py `num_dsus=4`"),
        ("audio heads", f"4 codebooks × 2 speakers = **8 heads**; each has {m['audio_vocab_size']} codes + delay id ({m['audio_vocab_size']}) + EOS id ({m['audio_vocab_size'] + 1}) = {_V} classes", "modeling_dsu.py `audio_vocab_size + 2`, `audio_delay_id`, `audio_eos_id`"),
        ("text stream", f"`text_stream={m['text_stream']}`, `multi_text_stream={m['multi_text_stream']}` → **one** text stream (the *system* speaker's) read and predicted every frame", "config.json; modeling_dsu.py `num_text_streams = 1`"),
        ("speaker conditioning", f"192-d speaker embedding → linear → one extra vector at the front of the sequence (`use_speaker_embedding={m['use_speaker_embedding']}`)", "modeling_dsu.py `speaker_embed_proj`"),
        ("parameters", "≈ 1.38 B in total (11.3 M of them are the LoRA adapters of §6)", "train logs: `trainable 11.3 M of 1380 M`"),
    ], columns=["component", "what", "source"])
    mo.vstack([mdt(_tbl), mo.md("**Input at frame *t*** (`get_input_embeds_and_labels`): the model never sees an audio *token*, it sees a **sum of embeddings**")])
    return


@app.cell
def _(mo):
    mo.md(r"""
$$
x_t \;=\; \sum_{h=0}^{7} E_h\big[c_{h,t}\big] \;+\; W_{\text{text}}\big[w_t\big],
\qquad
\text{sequence} = \big[\,P\,s\,\big] \;\Vert\; \text{prompt tokens} \;\Vert\; x_1, \dots, x_T
$$

* $c_{h,t}$: code of head $h$ at frame $t$. Heads 0-3 are the four codebooks of the **system** speaker, heads 4-7 those of the **user** (the other speaker).
  Each head has its own embedding table $E_h$.
* $w_t$: the text-stream token at frame $t$, embedded with the ordinary Llama token embedding $W_{\text{text}}$ (tied to the LM head).
* $s$: speaker embedding, $P$ its projection. The **prompt** is natural-language text (narrative + behaviour counts, §2) placed before the audio frames.

**Output at frame *t*:** the last hidden state goes through (a) one linear `dsu_head`, reshaped to **8 × 4034 logits** — the next code for every head, i.e. *both*
speakers' next audio frame — and (b) the Llama LM head, giving the next **text-stream** token. The loss at position *t* is computed against frame *t+1*.
""")
    return


@app.cell
def _(Z, mo, np):
    _d = Z["real_dsu"]; _t = Z["real_text"]
    _T = _d.shape[1]

    def modal_tuple(d4):
        u, c = np.unique(d4.T, axis=0, return_counts=True)
        return u[c.argmax()]

    SIL_SYS, SIL_USR = modal_tuple(_d[:4]), modal_tuple(_d[4:])
    frame = mo.ui.slider(0, 149, value=9, label="frame t (80 ms each)", full_width=True)
    mo.vstack([
        mo.md("### Look at one real training row\nThe row is a Behavior-SD dialogue from the released training data (§2 gives its source). Drag the slider: the table lists the eight codes that are embedded and summed at that frame, plus the text-stream token."),
        frame,
    ])
    return SIL_SYS, SIL_USR, frame, modal_tuple


@app.cell
def _(DATA, SIL_SYS, SIL_USR, Z, cite, frame, mo, pd):
    _t = frame.value
    _d = Z["real_dsu"]
    _words = DATA["rows"]["real"]["text_tokens"]
    _col = _d[:, _t]
    _rows = [(f"head {h}", "system" if h < 4 else "user", f"codebook {h % 4}", int(_col[h]),
              "delay id" if _col[h] == 4032 else ("silence code" if int(_col[h]) == int((SIL_SYS if h < 4 else SIL_USR)[h % 4]) else "")) for h in range(8)]
    _df = pd.DataFrame(_rows, columns=["head", "speaker", "codebook", "code id", "note"])
    _act_s = not (_col[:4] == SIL_SYS).all() and _col[0] != 4032
    _act_u = not (_col[4:] == SIL_USR).all() and _col[4] != 4032
    mo.vstack([
        mo.hstack([mdt(_df), mo.md(
            f"**text-stream token $w_t$:** `{_words[_t]}`<br>**system speaker audio:** {'active' if _act_s else 'silent / delay'}<br>"
            f"**user speaker audio:** {'active' if _act_u else 'silent / delay'}<br><br>"
            f"Frames 0-1 are `delay id` (4032) on all eight heads: the 2 delay frames in front of each row.")], gap=2, align="start"),
        cite("Real row = first row of the Behavior-SD sample copied by make_dataset.py (`oov-data/finetune/micro_v3/real_sample[0]`, 983 frames = 78.6 s). "
             "'silence code' = the most frequent code tuple of that speaker in this row; I assume it is the codec's silence pattern. That is an inference from counts, not something checked against the codec."),
    ])
    return


@app.cell
def _(BLUE, DATA, GREY, ORANGE, SIL_SYS, SIL_USR, Z, cite, mo, np, plt):
    _d = Z["real_dsu"]; _w = DATA["rows"]["real"]["text_tokens"]; _N = 160
    _act = lambda blk, sil: ~(blk == sil[:, None]).all(0) & (blk[0] != 4032)
    _as, _au = _act(_d[:4], SIL_SYS)[:_N], _act(_d[4:], SIL_USR)[:_N]
    _fig, _ax = plt.subplots(figsize=(9, 2.4))
    for _y, _a, _c in ((1, _as, BLUE), (0, _au, ORANGE)):
        _on = np.flatnonzero(_a)
        _ax.broken_barh([(i, 1) for i in _on], (_y - 0.35, 0.7), facecolors=_c, linewidth=0)
    for _i, _tok in enumerate(_w[:_N]):
        if not _tok.startswith("["):
            _ax.annotate(_tok.strip(), (_i + 0.5, 1.45), rotation=60, fontsize=6.5, ha="left", va="bottom", color=BLUE)
    _ax.set_yticks([0, 1], ["user audio", "system audio"]); _ax.set_ylim(-0.6, 2.4); _ax.set_xlim(0, _N)
    _ax.set_xlabel(f"frame (×80 ms; {_N} frames = {_N * 0.08:.1f} s)")
    _ax.set_title("Non-silent audio frames of both speakers and the system speaker's words (real row)", fontsize=9, loc="left")
    _ax.spines["left"].set_visible(False); _ax.tick_params(length=0)
    mo.vstack([_fig, cite("Same row as above. Blue = system speaker (heads 0-3) not in its modal code tuple; orange = user (heads 4-7). Words = non-special tokens of the single text stream, "
                          "printed at the frame where they occur: the text stream is **aligned in time** with the audio and only ever carries the *system* speaker's words.")])
    return


@app.cell
def _(mo):
    mo.md(r"""
**Take-aways for the experiments later on**

* The text stream is an *inner monologue* of the system speaker, in lock-step with its audio. The user speaker is **audio only**: the model hears the other side but never reads its words.
  That matters for "repeat the word": the partner has to get the word from Tom's *audio codes*, there is no text to copy from.
  (I earlier wrote that both text streams were embedded; for this released checkpoint that is wrong, `num_text_streams = 1`.)
* Generation is one model run on a **batch of two rows**: the speakers' views of the same conversation, each with its own prompt, speaker embedding and text stream (§3).
""")
    return


# ----------------------------------------------------------------------------------------------------------------------------------
@app.cell
def _(mo):
    mo.md(r"""
---
## 2 · What a training row is, and what the loss covers
""")
    return


@app.cell
def _(DATA, Z, cite, mo, np, pd):
    _r, _s = DATA["rows"]["real"], DATA["rows"]["syn_tom"]
    _fields = pd.DataFrame([
        ("input_ids / attention_mask", f"prompt padded to {_r['input_ids_len']}; real row uses {_r['n_prompt_tokens']} tokens, synthetic row {_s['n_prompt_tokens']}", "text"),
        ("dsu_ids", f"int64 {_r['dsu_shape']} (real) / {_s['dsu_shape']} (synthetic): 8 heads × frames, 2 delay frames in front; heads 0-3 = system, 4-7 = user", "audio codes"),
        ("text_stream_ids", f"int64 {_r['text_shape']}: the system speaker's words, one token per frame", "text stream"),
        ("spk_emb", f"float, {_r['spk_emb_dim']}-d: the *system* speaker's embedding", "speaker"),
        ("labels", "[-100] placeholder; the real labels are built in the model from the sequence itself", "-"),
    ], columns=["field", "content", "kind"])
    mo.vstack([
        mo.md("A row, field by field (identical for released data and for our synthetic dialogues; `finetune_repeat/compare_rows.py` checked this):"),
        mdt(_fields),
        cite(f"Source: `training/data.py tokenize_speech` and `finetune_repeat/rows.py`. Real rows in the released data are {527}-{983} frames (42-79 s); our synthetic dialogues are 17-33 s. "
             "Row shapes read from the stored sample rows (micro_v3)."),
    ])
    return


@app.cell
def _(DATA, mo):
    _r, _s = DATA["rows"]["real"]["prompt"], DATA["rows"]["syn_tom"]["prompt"]
    mo.vstack([mo.md("### The prompt: narrative and behaviour counts, per speaker"), mo.hstack([
        mo.vstack([mo.md("**Real row** (Behavior-SD)"), mo.md(f"```text\n{_r}\n```")]),
        mo.vstack([mo.md("**Our synthetic row** (Tom as system speaker)"), mo.md(f"```text\n{_s}\n```")]),
    ], widths="equal", align="start")])
    return


@app.cell
def _(cite, mo):
    mo.vstack([
        mo.md("""
Two things worth knowing: **each speaker only sees their own narrative**, and `backchannels` / `interruptions` are *counts* the model is trained to hit (the text stream carries
`[BC]` / `[INTER]` tokens for them). In our synthetic dialogues both counts are 0 and the narratives are plain instruction sentences (§6).
There is also a small train/inference difference: training writes the narrative as `- <narrative>` (`build_prompt`), inference
(`prepare_prompt`) writes it without the dash. Experiments before 2026-10-31 used the inference form; `prompt_format: "training"` in a config selects the other one."""),
        cite("Real prompt: Behavior-SD sample; synthetic prompt: `finetune_repeat/rows.py` NARRATIVE_TOM. Prompt-format difference: docs/prefix_log.md, 2026-10-31."),
    ])
    return


@app.cell
def _(DATA, Z, cite, mo, np):
    _w = DATA["rows"]["real"]["text_tokens"]
    _cls = lambda t: ("word" if not t.startswith("[") else t.strip("[]"))
    _n = len(_w)
    from collections import Counter as _C
    _c = _C(_cls(t) for t in _w)
    _lbl = lambda k: f"{k}: {_c[k]} ({_c[k] / _n:.0%})"
    mo.vstack([
        mo.md(f"""
### What the loss is (from `DSUModel.forward`)

* **Audio:** cross-entropy for **all 8 heads** (both speakers' next frame), averaged over non-pad labels. A frame has to be right on 8 codebook streams at once.
* **Text:** cross-entropy on the **one** text stream. The label sequence is `[speaker-slot, prompt tokens, text-stream tokens]`, so the *prompt* is a text target too (code comment: "use prompt in label").
* The two means are **added** (`total_loss += loss`), then divided by `grad_acc_steps`. Per conversation the mean is over non-pad labels, then over the batch.
* `calc_loss_on_c1_only: false` in the released config; the "first half of the heads only" branch applies in eval mode (`not self.model.training`).

**The text stream is mostly padding.** In the real row above ({_n} frames): {_lbl('SILENCE_PAD')}, {_lbl('UTTERANCE_PAD')}, {_lbl('WORD_PAD')}, {_lbl('word')}.
`[SILENCE_PAD]`, `[UTTERANCE_PAD]` and `[WORD_PAD]` are real tokens, **not** the ignored pad id, so predicting them counts, while a word is about one frame in {_n // max(_c['word'], 1)}.
A decreasing text loss therefore says little about *words*; that is why the later runs add a separate **word-frame cross-entropy** (§6).
"""),
        cite("Counts computed from the stored real row (rows.npz / data.json). Loss structure: training/modeling_dsu.py forward(), lines ~100-200, and get_input_embeds_and_labels()."),
    ])
    return


@app.cell
def _(DATA, Z, cite, mo, np):
    _a, _b = Z["syn_tom_dsu"], Z["syn_gweneth_dsu"]
    _swap = np.array_equal(_a[:4], _b[4:]) and np.array_equal(_a[4:], _b[:4])
    _wd = lambda k: " ".join(t.strip() for t in DATA["rows"][k]["text_tokens"] if not t.startswith("["))
    mo.vstack([
        mo.md(f"""
### Perspective: every dialogue is two rows

In training the "system" speaker is picked at random (50/50) per row (`prepare_dsu`). A synthetic dialogue is therefore turned into **two rows**, one per perspective
(`finetune_repeat/rows.py`). Check on the stored pair: heads 0-3 of Gweneth's row equal heads 4-7 of Tom's row and vice versa: **{_swap}**.

| | text stream (words of the system speaker) |
|---|---|
| Tom is system | *{_wd('syn_tom')}* |
| Gweneth is system | *{_wd('syn_gweneth')}* |

For "Gweneth repeats the word" only the **second** row teaches the behaviour: it is the one whose text stream contains the reply. (`word_metrics` in local_train.py relies on rows alternating Tom, Gweneth.)
"""),
        cite("Rows: `oov-data/finetune/micro_v3/train[0:2]`, the first synthetic dialogue; the first stream words come from CosyVoice TTS alignments (MMS forced aligner)."),
    ])
    return


# ----------------------------------------------------------------------------------------------------------------------------------
@app.cell
def _(mo):
    mo.md(r"""
---
## 3 · Generating a dialogue, and what ends it

`generate()` runs **one model on a batch of two rows** (`talk_to_itself=True`): row 0 is speaker 1's view, row 1 is a copy with the two speakers' head groups rotated
(`duplicate_with_head_rotation`), so what A says is what B hears. Each row has its own prompt, speaker embedding and text stream. Every frame samples all 8 heads and the text token of both rows.
The loop ends at `max_length` frames or when the audio **EOS id (4033)** is sampled by *any* head of *either* row.
""")
    return


@app.cell
def _(mo, source_lines):
    mo.vstack([
        source_lines("training/modeling_dsu.py", "# EOS in dsus", before=0, after=9),
        mo.md("The check is `.any()` over the batch (both rows) **and** all eight heads. With independent heads, the chance of stopping at a frame is $1-\\prod_h(1-p_h)$: one head leaking a little EOS probability is enough, and all later probes use exactly this \"P(any head = EOS)\"."),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
### A surprise from the base model: it sometimes stops at the first frame

We continue a **prefix** (a TTS clip teleported into the audio streams, §5) and ask the *released* model to carry on. Whether it carries on depends on the **silence we append after the clip** (`tail_s`).
First-frame P(any head = EOS), six words, base model:
""")
    return


@app.cell
def _(DATA, GREY, RED, cite, mo, np, plt):
    _t = DATA["tail"]; _rows = list(_t["rows"].items()); _w = _t["words"]
    _M = np.array([r[1] for r in _rows])
    _fig, _ax = plt.subplots(figsize=(7.2, 2.6))
    _im = _ax.imshow(_M, cmap="Reds", vmin=0, vmax=1, aspect="auto")
    _ax.set_xticks(range(len(_w)), _w, rotation=20, ha="right"); _ax.set_yticks(range(len(_rows)), [f"tail {r[0]}" + ("" if r[0] == "raw" else " s") for r in _rows])
    for (_i, _j), _v in np.ndenumerate(_M):
        _ax.text(_j, _i, f"{_v:.2f}", ha="center", va="center", fontsize=8, color="white" if _v > 0.55 else "black")
    _ax.spines[:].set_visible(False); _ax.tick_params(length=0)
    mo.vstack([_fig, cite(f"{_t['source']}. Quoted from the log, not recomputed. 'raw' = the clip as TTS produced it (silence after speech 0.04-0.8 s, median 0.18 s); other rows trim and append exactly that much silence. "
                          "The tail is quantised to 80 ms frames, and the raw length does *not* predict the stop (hello: 0.1 s tail 0.998 vs raw 0.003). The log's first interpretation, 'the tail looks like the end of a training sample', is revised in §7.")])
    return


# ----------------------------------------------------------------------------------------------------------------------------------
@app.cell
def _(mo):
    mo.md(r"""
---
## 4 · The motivating problem: OOV words are lost in the audio

The thesis question is how F-Actor handles rare words. The model has two parallel renderings of what it "says": the **text stream** (planning, the inner monologue) and the **audio**.
We compare the target word's fate in each: transcribe the audio with ASR, grade the word in both by character similarity (not WER, the target is one word) into
**intact / split / substituted / dropped** (`scripts/analysis/analyze_oov.py`).
""")
    return


@app.cell
def _(Counter, DATA, GREY, cite, mo, np, plt):
    _tab = DATA["oov"]["table"]; _L = ["intact", "split", "substituted", "dropped"]
    _c = Counter((r["mono_label"], r["audio_label"]) for r in _tab)
    _M = np.array([[_c[(a, b)] for b in _L] for a in _L])
    _fig, _ax = plt.subplots(figsize=(4.8, 3.4))
    _ax.imshow(_M, cmap="Blues", vmin=0, vmax=_M.max())
    _ax.set_xticks(range(4), _L, rotation=25, ha="right"); _ax.set_yticks(range(4), _L)
    _ax.set_xlabel("target word in the AUDIO (Parakeet-TDT ASR)"); _ax.set_ylabel("target word in the TEXT stream"); _fig.tight_layout()
    for (_i, _j), _v in np.ndenumerate(_M):
        _ax.text(_j, _i, str(_v), ha="center", va="center", color="white" if _v > _M.max() / 2 else "black")
    _n = len(_tab); _intact_text = _M[0].sum(); _lost_audio = _M[0, 2:].sum()
    mo.hstack([
        mo.vstack([_fig, cite(f"{DATA['oov']['source']} (file date {DATA['oov']['mtime']}). n = {_n} runs, all in the OOV band; ASR = Parakeet-TDT for this table; the ASR confound is not calibrated.")]),
        mo.md(f"""
**Reading it.** Of {_n} runs, the word is *intact in the text stream* in {_intact_text}; in {_lost_audio} of those the audio then has it **substituted or dropped** (ASR hears another word).
That is the acoustic-failure signature: the Llama backbone planned the word, the audio heads plus codec did not render it.

**What this table does not show.** The largest cell ({_M[3, 3]}/{_n}) is *dropped in both*: the model avoids the word altogether (topic drift). Only {_M[1:3].sum()} runs show a
planning-side mangle. And with one ASR system and no round-trip control, "dropped in audio" can also mean "the ASR missed it".
The cheap controls that gate any claim are listed in §8 (codec round-trip, ASR calibration, a frequent-word topline).
"""),
    ], widths=[1, 1], align="start")
    return


@app.cell
def _(DATA, clip, cite, mo):
    _e = DATA["oov"]["examples"]["schadenfreude"]
    mo.hstack([
        mo.vstack([
            mo.md(f"""**Example: `schadenfreude`** (graded here with Whisper large-v3, a different backend from the table above). Text stream (inner monologue): *"…have you heard about **schaden freude** it's a small world…"* → graded **{_e['grading']['mono_label']}** (planned as two tokens, the word is recognisable).
Audio, Whisper large-v3: graded **{_e['grading']['audio_label']}**, closest ASR evidence `{_e['grading']['audio_evidence']}`."""),
            cite(f"Run record `oov-data/{_e['json']}`; narrative of the explainer: \"{_e['config']}\""),
        ]),
        clip("oov_schadenfreude"),
    ], widths=[1, 1], align="start")
    return


# ----------------------------------------------------------------------------------------------------------------------------------
@app.cell
def _(mo):
    mo.md(r"""
---
## 5 · Idea 1: get the word into the context with a spoken **prefix**

If the model cannot say `schadenfreude`, can we at least see whether it can *repeat* it after hearing it? The test needs the model to hear a request, so we do not run the dialogue from the start.
Instead we build the first turn ourselves and let the model continue.

**Pipeline** (`training/prefix.py`, `build_prefix` / `build_multiturn_prefix`):

1. **TTS**: CosyVoice (`Fun-CosyVoice3-0.5B-2512`, zero-shot cloning of the Behavior-SD voices Tom / Gweneth): *"Please listen carefully and repeat the word X."*, gated by ASR so the word really is in the clip.
2. **Forced alignment** (torchaudio MMS) gives word times, which become the *text stream* of the prefix.
3. **Gain**: +5.6 dB on CosyVoice clips (they are 5.6 dB below Behavior-SD level, measured in `oov-shared/checks/LOG.md`) with a peak guard.
4. **Optional `tidy_clip`**: cut TTS's long lead-in, shorten word gaps, fade out.
5. **Tail** of `tail_s` seconds of silence, optionally **look-ahead** (encode extra silence, cut the codes back).
6. **NanoCodec encode** → `dsu_ids` for the speaker's four heads, partner heads filled with the silence code; hand to `generate()` with the delay frames and let it run.
""")
    return


@app.cell
def _(clip, mo):
    mo.vstack([
        mo.md("What the clips sound like (all synthetic). The raw reply has a long hiss-like lead-in and a pause at the comma; `tidy_clip` removes both:"),
        mo.hstack([clip("tts_request_music"), clip("tts_reply_music_raw"), clip("tts_reply_music_tidy")], widths="equal", align="start"),
    ])
    return


@app.cell
def _(BLUE, GREY, ORANGE, cite, mo, plt):
    # schematic of multi-turn prefixes: durations are typical values from the logs, drawn to scale but not one specific dialogue
    REQ, REP, GAP, TAIL = 3.1, 1.1, 0.5, 0.3     # 'water' request 3.12 s (prefix json), tidy 'music' reply 1.1 s, gap 0.5 s and tail 0.3 s (docs/prefix_log.md decisions)
    _fig, _axs = plt.subplots(3, 1, figsize=(9, 2.9), sharex=True)
    for _ax, _k in zip(_axs, (0, 1, 3)):
        _t = 0.0
        for _i in range(_k + 1):
            _tag = "target request" if _i == _k else "demo request"
            _ax.barh(1, REQ, left=_t, color=BLUE, height=0.6); _ax.text(_t + REQ / 2, 1, _tag, ha="center", va="center", color="white", fontsize=7)
            _t += REQ + GAP
            if _i < _k:
                _ax.barh(0, REP, left=_t, color=ORANGE, height=0.6); _ax.text(_t + REP / 2, 0, "Okay, w.", ha="center", va="center", fontsize=6.5)
                _t += REP + GAP
        _ax.barh(1, TAIL, left=_t - GAP, color=GREY, height=0.6)
        _ax.set_yticks([0, 1], ["Gweneth", "Tom"]); _ax.set_ylabel(f"{_k} demos", rotation=0, ha="right", labelpad=40); _ax.tick_params(length=0)
        _ax.axvline(_t - GAP + TAIL, color="k", lw=0.8, ls=":")
    _axs[-1].set_xlabel("seconds · the model generates from the dotted line on")
    mo.vstack([_fig, cite("Schematic, to scale for typical durations: request 3.1 s ('water' prefix record), tidied reply 1.1 s, gap 0.5 s and tail 0.3 s (decisions in docs/prefix_log.md, plan section). Demonstration words are frequent words, never the target.")])
    return


@app.cell
def _(mo, pd):
    _t = pd.DataFrame([
        ("2026-10-16/18", "P3 opener ('…tell you about X. Have you ever heard of X?'), 3 words", "prefix intact; Tom continues with 'it'/'they' and never says the word again. Measures reuse, not production → shelved", "log"),
        ("2026-10-19/25", "P4 request ('…repeat the word X'), frequent words hello / water, default narratives, 6 dialogues", "partner said hello/water in **0/6**; `water` ended after 1 generated frame in 3/3", "log"),
        ("2026-10-19/25", "same, partner narrative told to say the word back", "partner's audio graded `dropped` **6/6**; `water` again stopped at frame 1 (3/3); `hello` drifted to generic meeting talk", "log"),
        ("2026-10-29", "tail sweeps (§3 table), 24 runs with prefix + 0.3/0.5 s tail", "21/24 ran to max_length; Tom audible 0.64-1.28 s after the prefix in all 24; partner never said the word", "log"),
        ("2026-10-31 (log)", "multi-turn pilot: 0, 1, 3 demos × 4 words × 2 repeats, 150 frames", "**0 / 24** dialogues with the partner saying the target", "log (pilot)"),
    ], columns=["date", "setup", "result", "source"])
    mo.vstack([mo.md("### What the prefix experiments found"), mdt(_t),
               mo.md("<small style='color:#666'>Quoted from `docs/prefix_log.md`; raw outputs are in `oov-data/` (`scratch_eos/`, `exps/`), not re-aggregated here. P4 gate: TTS ASR check kept 112 of 157 words.</small>")])
    return


@app.cell
def _(mo):
    mo.md(r"""
**Reading the pilot.** Even with demonstrations of the exact behaviour in context ("Okay, coffee." after "…repeat the word coffee."), the released model does not echo.
That is a statement about *in-context learning of this skill in this model*, with a TTS-built prefix that differs from Behavior-SD in voice, level and silence statistics. It does not show that the skill is impossible. It is what motivates changing weights.
""")
    return


# ----------------------------------------------------------------------------------------------------------------------------------
@app.cell
def _(mo):
    mo.md(r"""
---
## 6 · Idea 2: teach the repeat with LoRA on synthetic dialogues

Same builder, different use: instead of a prefix to *prompt* with, assemble full **training rows** from the TTS clips: *Tom asks, Gweneth replies "Okay, X."*, with plain instruction narratives
(Tom: "You ask Gweneth to repeat words after you, one word at a time." · Gweneth: "Tom will say a word. Listen to it and say the same word back to Tom."). Each dialogue becomes two rows (§2).
The base model's weights stay frozen except for low-rank adapters.
""")
    return


@app.cell
def _(DATA, mo, pd):
    _lm, _ls, _lh = DATA["logs"]["micro_v2_lora"], DATA["logs"]["single_v1_lora"], DATA["logs"]["single_v1_heads"]
    _t = pd.DataFrame([
        ("micro_v2 (micro-run 1)", "LoRA r16 on q,k,v,o,gate,up,down of all layers; audio heads frozen", "11.3 M of 1380 M", "40 rows (20 dialogues × 2), 5 words (doctor, garden, morning, movie, music), 3-5 exchanges, 16-28 s",
         f"{len(_lm['steps'])} steps × 4 rows (6 epochs), lr 2e-4, ~85 min"),
        ("single_v1 Run A", "same LoRA, heads frozen", "11.3 M", "138 rows = 69 words × 1 exchange × 2 perspectives (+10 validation words), request, 0.5 s gap, reply, 0.1 s tail",
         f"{len(_ls['steps'])} steps (4 epochs)"),
        ("single_v1 Run B", "LoRA + trainable `audio_embeds`, `dsu_head`, `speaker_embed_proj`", "143.9 M", "same data", f"{len(_lh['steps'])} steps (8 epochs)"),
    ], columns=["run", "trainable", "parameters", "data", "schedule"])
    mo.vstack([mdt(_t),
               mo.md("<small style='color:#666'>All on the CPU of the laptop (fp32, 1.38 B parameters: ~10 GB RSS, 25-50 s per row-step); cluster access was lost, per the 2026-10-31 log entries. AdamW, 5 warm-up steps then cosine to 10 % of lr, grad-norm clip 1.0, 4 rows per optimizer step. Sources: docs/prefix_log.md, train logs under `oov-data/finetune/`. The merged adapters are saved as full models.</small>")])
    return


@app.cell
def _(BLUE, DATA, GREY, ORANGE, cite, mo, plt):
    _fig, _axs = plt.subplots(1, 3, figsize=(10, 2.9))
    for _ax, (_k, _ttl) in zip(_axs, (("micro_v2_lora", "micro_v2 (5 words)"), ("single_v1_lora", "single_v1 Run A (LoRA)"), ("single_v1_heads", "single_v1 Run B (LoRA + heads)"))):
        _L = DATA["logs"][_k]
        _ax.plot([s["step"] for s in _L["steps"]], [s["loss"] for s in _L["steps"]], color=GREY, lw=0.8, label="train batch (mean of 4 rows)")
        _ev = _L["evals"]
        _x = [0 if e["at"] == "before training" else (_L["steps"][-1]["step"] if e["at"] == "after training" else int(e["at"].split()[1])) for e in _ev]
        _ax.plot(_x, [e["val_loss"] for e in _ev], "o-", color=BLUE, ms=3, label="synthetic validation")
        _ax.plot(_x, [e["real_loss"] for e in _ev], "s-", color=ORANGE, ms=3, label="real Behavior-SD rows")
        _ax.set_title(_ttl, fontsize=9, loc="left"); _ax.set_xlabel("optimizer step"); _ax.set_ylim(0, 5.2)
    _axs[0].set_ylabel("loss (audio + text)"); _axs[0].legend(fontsize=6.5, frameon=False)
    mo.vstack([_fig, cite("Parsed from `oov-data/finetune/micro_lora/train.log`, `single_v1_train.log`, `single_v1_heads_train.log`. The real rows are 2 Behavior-SD rows, a sanity check against forgetting. LoRA-only runs: real-row loss drifts up slightly (micro 4.43→4.50; Run A 4.43→4.57) while synthetic validation falls. "
                          "**Run B (heads trained) damages the model:** real-row loss climbs 4.43→4.98 and synthetic *validation* loss turns around after step 34 (3.35→4.31) while the train-batch loss goes to ~0. "
                          "micro_v2's validation uses the *same five words* as training, so its drop shows the format was learned, not that it generalises.")])
    return


@app.cell
def _(DATA, cite, mo, pd):
    _rows = []
    for _k, _nm in (("single_v1_lora", "Run A (LoRA)"), ("single_v1_heads", "Run B (LoRA + heads)")):
        _ev = {e["at"]: e for e in DATA["logs"][_k]["evals"]}
        for _at in ("before training", "after training"):
            _wf = _ev[_at].get("word_frames", {})
            for _who in ("val Tom", "val Gweneth", "train Tom", "train Gweneth"):
                _w = _wf.get(_who) or _wf.get(_who.split()[1]) if _wf else None
                if _w: _rows.append((_nm, _at, _who, _w["ce"], _w["top1"], _w["n"]))
    _df = pd.DataFrame(_rows, columns=["run", "when", "split / speaker", "word-frame CE", "top-1", "tokens"])
    mo.vstack([
        mo.md("""### Did it learn the *words*? The word-frame metric
The model's own text loss is dominated by pad tokens (§2), so `local_train.py` computes CE and top-1 on the **system speaker's real word tokens only**, teacher-forced.
Tom's words are the request (*"please listen carefully and repeat the word X"*, the instruction template is easy); **Gweneth's** are "okay" and the **target word**, the part that has to be copied."""),
        mo.ui.table(_df.round(3), selection=None, page_size=16),
        cite("Parsed from the train logs (`--n-eval 4`: 2 Tom rows + 2 Gweneth rows per split, ~20 sub-word tokens per speaker and split; a very small sample). The `Run A` log prints only the validation split."),
    ])
    return


@app.cell
def _(DATA, GREEN, RED, cite, mo, plt):
    # teacher-forced probe: P(target token) at Gweneth's word frames, for 'okay' and the target word
    _fig, _axs = plt.subplots(1, 2, figsize=(8.6, 3.0), sharey=True)
    _names = {"single_v1_lora": "Run A (LoRA only)", "single_v1_heads": "Run B (LoRA + heads)"}
    for _ax, _k in zip(_axs, _names):
        _P = DATA["probes"][_k]["splits"]
        for _x0, _sp in ((0, "validation"), (1, "train")):
            for _r in _P[_sp]:
                _tok = _r["tokens"]
                _ok = _tok[0]["p_target"] if _tok[0]["target"] == "okay" else None
                _wd = _tok[1]["p_target"] if len(_tok) > 1 else None
                _ax.scatter([_x0 * 3 + 0 + 0.0], [_ok], color=GREEN, alpha=0.5, s=14) if _ok is not None else None
                _ax.scatter([_x0 * 3 + 1], [_wd], color=RED, alpha=0.5, s=14) if _wd is not None else None
        _ax.set_xticks([0, 1, 3, 4], ["'okay'\nval", "word\nval", "'okay'\ntrain", "word\ntrain"]); _ax.set_title(_names[_k], fontsize=9, loc="left")
    _axs[0].set_ylabel("P(target token) at its frame")
    _n = {k: sum(len(v["splits"][s]) for s in v["splits"]) for k, v in DATA["probes"].items()}
    mo.vstack([_fig, cite(f"Teacher-forced `word_probe.py` output (`oov-data/finetune/probe_out/*.txt`), Gweneth's rows only: {_n['single_v1_lora']} rows for Run A, {_n['single_v1_heads']} for Run B (8 validation + 8 train rows each; dots at the same value overlap). "
                          "Run A gets 'okay' right on most rows but the word on none, **even on training words**. Run B reproduces its 8 training words (P = 0.97-1.0) and gets validation words ~0: memorisation, not copying.")])
    return


@app.cell
def _(DATA, cite, mo, np):
    _g = {k: np.array([s["gnorm"] for s in v["steps"]]) for k, v in DATA["logs"].items()}
    mo.vstack([
        mo.md("**Optimisation note (unexpected).** `clip_grad_norm_(…, 1.0)` is applied to the *summed* gradient of the 4 rows. The logged pre-clip norms: " +
              "; ".join(f"{k}: {int((v > 1).sum())}/{len(v)} steps above 1.0 (median {np.median(v):.1f}, max {v.max():.0f})" for k, v in _g.items()) +
              ". So clipping was active at essentially every step, the update is a normalised-gradient step and lr alone sets the step size; I have not checked whether that helped or hurt."),
        cite("Parsed `gnorm` from the three train logs. In local_train.py the model's loss is already divided by `--accum`, then multiplied back by it before `backward()` (grads summed over rows, not averaged)."),
    ])
    return


# ----------------------------------------------------------------------------------------------------------------------------------
@app.cell
def _(mo):
    mo.md(r"""
---
## 7 · Evaluating the fine-tune, and the stop-token detective story

**Test.** Give the model Tom's request clip as a prefix (training prompt format), generate up to 150 frames, count a **hit** only if the target word appears in **Gweneth's text stream**
(a run-record field written by `generate()`; my regex is a whole-word match). Groups: `train` words, `heldout` frequent words (water, kitchen, hello), `oov` words, 2 repeats each.
""")
    return


@app.cell
def _(DATA, mdt, mo, pd):
    _runs = pd.DataFrame(DATA["runs"])
    _desc = {
        "base": "released model, prefix tail 0.3 s", "base_la": "released model, tail 0.1 s + 1.5 s look-ahead",
        "tuned": "micro_v2 LoRA, tail 0.3 s", "tuned_la": "micro_v2 LoRA, tail 0.1 s + look-ahead",
        "tuned_mask": "micro_v2 LoRA, look-ahead + stop forbidden in the first 60 frames",
        "lora": "single_v1 Run A, look-ahead + mask 60", "heads": "single_v1 Run B, look-ahead + mask 60", "heads_t5": "single_v1 Run B, 0.5 s tail, no look-ahead",
    }
    _g = _runs.groupby(["set", "tag"], sort=False).agg(runs=("hit", "size"), hits=("hit", "sum"), stopped_by_eos=("ended_by", lambda s: int((s == "eos").sum())),
                                                         median_frames=("frames", "median")).reset_index()
    _g["frame<=2 stops"] = [int(((_runs.set == r.set) & (_runs.tag == r.tag) & (_runs.ended_by == "eos") & (_runs.frames <= 2)).sum()) for r in _g.itertuples()]
    _g["configuration"] = _g.tag.map(_desc)
    mo.vstack([
        mdt(_g[["tag", "configuration", "runs", "hits", "stopped_by_eos", "frame<=2 stops", "median_frames"]]),
        mo.md("""**Outcome: 0 hits in every fine-tuned configuration** (base_la has 1/18, but it is `music` appearing in an unrelated sentence — *"oh i love them speaking of music have you heard about the new event"* — topic drift, not a repeat; so effectively 0 everywhere). The first thing the table shows, though, is not the skill:
the micro-run model **stops at the very first generated frame in 18 of 18 runs**, so nothing could be tested."""),
        mo.md("<small style='color:#666'>Counts recomputed from every run record `*_pfx.json` under `oov-data/finetune/{micro_eval,single_eval}/eval_out/` (so they can differ from hand-written summaries if those had a different hit rule); "
              "they agree with `showcase.html` (0/18, 1/18, 0/18, 0/18, 0/18; 0/10, 0/20).</small>"),
    ])
    return


@app.cell
def _(DATA, mo, pd):
    _runs = pd.DataFrame(DATA["runs"])
    pick = mo.ui.dropdown(sorted(_runs.tag.unique(), key=list(_runs.tag.unique()).index), value="tuned_mask", label="configuration")
    pick
    return (pick,)


@app.cell
def _(DATA, mdt, mo, pd, pick):
    _r = pd.DataFrame(DATA["runs"]); _r = _r[_r.tag == pick.value]
    mdt(_r[["group", "word", "repeat", "ended_by", "frames", "gweneth"]].rename(columns={"gweneth": "Gweneth's text stream (what she 'said')"}))
    return


@app.cell
def _(mo):
    mo.md(r"""
### Why did the tuned model stop at once?

*First hypothesis (tail/silence).* Every training dialogue ended with exactly `tail_s = 0.3` s of silence, then the EOS that training appends. The test prefix also ended in 0.3 s of silence. "The model learned: ~0.3 s of silence after a speaker is done ⇒ EOS."
That is a real design flaw (a synthetic dialogue teaches an end-of-dialogue cue, which should look like the real data's), fixed in `micro_v3` by ending dialogues after 0.1 s.

*But the probes said it was not (only) that.* Teacher-forced P(EOS) from the tuned model, on a training dialogue and on an unseen validation dialogue, is ≈1.0 exactly after the **last** frame and < 0.05 elsewhere, including the 0.5 s gaps between turns.
It learned *the end of these dialogues*, not silence. And the same request + 0.3 s tail has P(EOS)=0.000 *inside* a longer dialogue but 1.000 as a standalone prefix, with identical history. The audio codes of that identical history already differ from ~0.34 s before the end of speech:
**NanoCodec's encoder behaves differently near the end of a file**, so *every* real dialogue and every standalone prefix carries an "end-of-file code signature". The tuned model learned "that signature ⇒ EOS" much more sharply than the base model had (40 rows, each with exactly one such end).

*Two cues, then:* (1) the codec's end-of-file signature, seen by the base and the tuned model, removed by **look-ahead** (encode extra silence after the prefix, cut the codes back); (2) the tuned model's own "0.3 s silence after the last speech ⇒ EOS".
""")
    return


@app.cell
def _(cite, mo, pd):
    # quoted from docs/prefix_log.md, 2026-10-31 'why the micro-run 1 model stops'; P(EOS) at the last prefix frame, water / kitchen / hello
    _t = pd.DataFrame([
        ("base", "0.1 s", ".22 / .18 / .76", ".00 / .00 / .00"), ("base", "0.3 s", "0 / 0 / 0", "0 / 0 / 0"),
        ("tuned", "0.0 s", ".00 / .07 / .00", ".00 / .00 / .00"), ("tuned", "0.1 s", ".61 / .95 / 1.0", ".00 / .00 / .00"), ("tuned", "0.3 s", "1.0 / 1.0 / 1.0", "1.00 / .81 / .17"),
    ], columns=["model", "tail", "prefix encoded alone", "with 1.5 s look-ahead"])
    mo.vstack([mo.md("**The look-ahead remedy,** P(any head = EOS) at the last prefix frame for `water / kitchen / hello`:"), mdt(_t),
               cite("Quoted from docs/prefix_log.md (2026-10-31), measured with `oov-data/finetune/micro_eval/eos_lookahead.py`, `eos_where*.py`; those probes print to the terminal and were **not** re-run for this notebook. "
                    "Read: look-ahead removes the codec cue (all zeros for the base model, tuned at 0.0 and 0.1 s tail), while the tuned model's 0.3 s-silence cue survives look-ahead (1.00/.81/.17).")])
    return


@app.cell
def _(BLUE, GREY, ORANGE, RED, cite, DATA, mo, np, plt):
    _fig, _axs = plt.subplots(1, 3, figsize=(10, 2.8), sharey=True)
    for _ax, (_set, _tag) in zip(_axs, (("micro_eval", "tuned_mask"), ("single_eval", "lora"), ("single_eval", "heads"))):
        _rs = [r for r in DATA["runs"] if r["set"] == _set and r["tag"] == _tag and r["max_p_eos"]]
        for _r in _rs:
            _p = np.array(_r["max_p_eos"]); _x = np.arange(len(_p))
            _ax.plot(_x, _p, color=BLUE if _r["ended_by"] != "eos" else RED, lw=0.6, alpha=0.55)
            if _r["ended_by"] == "eos": _ax.scatter([_x[-1]], [_p[-1]], color=RED, s=8, zorder=3)
        _m = max((r["mask_stop_frames"] for r in _rs), default=0)
        if _m: _ax.axvline(_m, color="k", lw=0.8, ls=":"); _ax.text(_m + 1, 0.55, "mask lifts", fontsize=7)
        _ax.set_title(f"{_tag}  ({len(_rs)} runs)", fontsize=9, loc="left"); _ax.set_xlabel("generated step")
    _axs[0].set_ylabel("max over heads of P(EOS)");
    mo.vstack([_fig, cite("`max_p_eos_per_step` stored in each run record: per generated step, the largest softmax P(EOS) over the heads, before temperature/top-k (the stop-mask switch added to `generate()` logs it). "
                          "Red = the run ended by EOS (dot at the stop). Only runs made with the stop-mask build record this trace (the earlier base / tuned / tuned_la runs do not). "
                          "`tuned_mask`: with EOS forbidden for the first 60 steps the model speaks; 4 of 18 runs then end by EOS at steps 70-119. In `single_v1` every run ends by EOS, most soon after the mask lifts (next cell). "
                          "Whether the P(EOS) spikes inside the masked window are the same end-of-dialogue cue is not established.")])
    return


@app.cell
def _(DATA, cite, mo):
    _f = {t: sorted(r["frames"] for r in DATA["runs"] if r["set"] == "single_eval" and r["tag"] == t) for t in ("lora", "heads", "heads_t5")}
    _near = {t: sum(1 for x in v if x <= 66) for t, v in _f.items()}
    mo.vstack([
        mo.md(f"""**The single_v1 models also want to stop.** Generated frames at the stop (mask = first 60 steps; the prefix is not counted):
Run A `{_f['lora']}`, Run B `{_f['heads']}`, Run B with 0.5 s tail and no look-ahead `{_f['heads_t5']}`. {_near['lora']}/10, {_near['heads']}/10 and {_near['heads_t5']}/10 end within 6 steps of the mask lifting.
Their *training* rows were only 51-74 frames long in total (request + 0.5 s + reply + 0.1 s tail, median 62), so by the time the mask lifts, the model is long past the point where its training dialogues ended.
The traces above show P(EOS) jumping to ≈1 repeatedly *inside* the masked window in nearly every `single_v1` run, so the model wants to stop from the first steps on (the `tuned_mask` runs spike much less). If that is the learned end-of-dialogue cue, the masked window is a regime the model never saw: generated speech there is produced under "should be over" pressure, which is a caveat on reading '0 hits' as 'cannot repeat'. This is a hypothesis; I have not run an unmasked comparison with longer training dialogues."""),
        cite("Frames from the run records; training-row lengths from `oov-data/finetune/single_v1/train` (138 rows)."),
    ])
    return


@app.cell
def _(clip, mo):
    mo.vstack([
        mo.md("**Listen.** Tom's request \"…repeat the word water\" through the released model, the micro_v2 model and the micro_v2 model with the stop mask; the fourth clip is Run B on the validation word \"moon\". The first ~3-5 s of each clip are the teacher-forced prefix, the rest is generated:"),
        mo.hstack([clip("base_water"), clip("tuned_water")], widths="equal", align="start"),
        mo.hstack([clip("tuned_mask_water"), clip("heads_moon")], widths="equal", align="start"),
    ])
    return


# ----------------------------------------------------------------------------------------------------------------------------------
@app.cell
def _(mo):
    mo.md(r"""
---
## 8 · Where this stands, and what is not known
""")
    return


@app.cell
def _(mo, pd):
    _t = pd.DataFrame([
        ("Where does OOV fail?", "mostly in the audio, with a large 'avoidance' class", "first batch, n=50, one ASR", "codec round-trip control; ASR calibration; frequent-word topline"),
        ("Can the released model repeat in context?", "no, 0/24 with 0-3 demos, 0/6 single request", "TTS-built prefixes, ~4 words", "different demo wording/voices; Behavior-SD-style audio levels"),
        ("Why did micro_v2 stop at frame 1?", "codec end-of-file signature + learned end-of-dialogue cue", "teacher-forced probes + P(EOS) logs", "re-run probes after micro_v3 (tail 0.1 s); not done"),
        ("Can LoRA teach the format?", "yes: validation loss falls, 'okay' is predicted", "5-69 words", "-"),
        ("Can LoRA teach the *copy*?", "**no so far**: 0 hits; Run A never gets the word right, even for training words", "138 rows, 4 epochs, CPU", "see hypotheses"),
        ("Run B: memorises (teacher-forced P≈0.99) but generates 0/20", "**unexplained**; heads load correctly; tail/look-ahead timing does not explain it (heads_t5). Also: real-row loss 4.43→4.98, validation loss up after step 34: it overfits and damages the model", "teacher-forced probes + train logs; generation n=20", "unmasked free-running comparison; early stopping / lower lr for the heads"),
    ], columns=["question", "answer so far", "evidence strength", "next check"])
    mo.vstack([mdt(_t)])
    return


@app.cell
def _(mo):
    mo.md(r"""
### Hypotheses I have **not** tested

1. **Too few distinct words for a copy mechanism.** Copying means attending from Gweneth's text-stream position to the *audio codes of Tom's word* (4 codebooks, earlier in time). With 69 words the cheapest solution is to memorise word→text (Run B does exactly that). Test: hundreds of words, with a validation set of words that never occur in training; the word-frame top-1 on held-out words is the metric to watch.
2. **The target never gets weight in the loss.** Word frames are ~1 % of text-stream labels (§2). Test: loss-weight or mask the loss to the reply's word frames.
3. **The training signal is a template** ("Okay, X.") whose first token is free to learn. Test: vary the reply form ("X.", "I heard X", spelled-out repetitions), keeping the word the only unpredictable part.
4. **Free-running mismatch for Run B.** Training words are right under teacher forcing but absent in generation. Candidates: the prefix I build differs from the training row in a way the probes do not see (gain 4.51 vs 5.6 dB peak-guarded, tail, look-ahead), or sampling (temperature 0.9, top-k 40) rather than argmax.
5. **Alternative to building audio:** mine Behavior-SD for existing "repeat/echo" exchanges and fine-tune on real audio only (no TTS domain shift).
""")
    return


@app.cell
def _(mo):
    mo.md(r"""
### Corrections to my own earlier statements (so they do not get copied into the thesis)

* *Both text streams are embedded* — wrong for the released checkpoint: one text stream, the system speaker's (§1).
* *"Trailing silence was ruled out as the cause of the immediate stop"* (`oov-shared/checks/p4_controls.yaml`) — the tail sweeps showed otherwise for the prefix; and §7 shows the codec end-of-file signature is a second cause. The 0.3 s-tail fix of the earlier experiments was partly an artefact fix.
* *micro-run 1 train.log shows audio/text losses 4× too large* — a logging scale bug (fixed in `local_train.py`); this notebook divides that log's audio/text values by 4. The totals were right.
* *First-order OOV numbers rely on a single ASR system* — treat as indicative until the ASR-calibration control is done.

### Reproducing this notebook

`python notebooks/build_story_assets.py` regenerates `notebooks/f_actor_story_assets/` from the local `oov-data/` tree (needs the repo `.venv`: datasets, soundfile, transformers). The notebook itself needs only `marimo`, `numpy`, `pandas`, `matplotlib`:
`uv run --extra analysis marimo edit notebooks/f_actor_story.py`.
""")
    return


if __name__ == "__main__":
    app.run()

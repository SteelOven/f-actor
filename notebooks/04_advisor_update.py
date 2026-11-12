import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    # Paths, constants and every loader. Nothing below types a result by hand: the numbers come from the files named here.
    #   .venv/bin/marimo edit notebooks/04_advisor_update.py     (repo root; slides: the layout switch in the top right)
    #   .venv/bin/marimo run  notebooks/04_advisor_update.py     (presentation mode)
    import io
    import json
    import re
    import sys
    from dataclasses import fields
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    import soundfile as sf
    from matplotlib.patches import Rectangle

    REPO = Path(__file__).resolve().parent.parent
    THESIS = REPO.parent
    DATA = THESIS / "oov-data"
    FT = DATA / "finetune"
    LOGS = FT / "cluster_logs"  # copied from the cluster workspace on 2026-11-10
    OUT = DATA / "f-actor" / "outputs"
    FPS = 12.5  # NanoCodec frames per second

    # categorical slots 1 and 2 of the dataviz reference palette (validated for the light surface), inks and surface
    C_REQ, C_REP = "#2a78d6", "#eb6834"
    INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    return (
        C_REP,
        C_REQ,
        DATA,
        FPS,
        FT,
        GRID,
        INK,
        INK2,
        LOGS,
        OUT,
        REPO,
        Rectangle,
        SURFACE,
        THESIS,
        fields,
        io,
        json,
        np,
        pd,
        plt,
        pq,
        re,
        sf,
        sys,
    )


@app.cell
def _(FPS, np, pd, re):
    from pathlib import Path as _Path

    def parse_tf(path):
        """Output of scripts/analysis/teacher_forced_check.py -> {(narrative, split): summary + per-word rows}."""
        out, key = {}, None
        head = re.compile(r"^(\S+)\s+(\d+)/(\d+)\s{2}")
        tail = re.compile(r"\s+(\d+)/(\d+)\s+(\d\.\d+)\s+(\d+)/(\d+)\s+(\d+)\s*$")
        for line in _Path(path).read_text(errors="replace").splitlines():
            m = re.match(r"^######## narrative: (\w+)\s+split: (\w+)", line)
            if m:
                key = (m[1], m[2])
                out[key] = {"rows": []}
                continue
            if key is None:
                continue
            if line.startswith("TOTAL"):
                n = [int(x) for x in re.findall(r"\d+", line)]
                out[key].update(tokens=(n[0], n[1]), speak=(n[2], n[3]), silent=(n[4], n[5]), early_eos=n[6])
            elif line.startswith("REPLY"):
                n = [int(x) for x in re.findall(r"\d+", line)]
                out[key].update(exact=(n[0], n[1]), word=(n[2], n[3]))
            else:
                h, t = head.match(line), tail.search(line)
                if h and t and t.start() > h.end():
                    mid = line[h.end() : t.start()]
                    out[key]["rows"].append(
                        {
                            "word": h[1],
                            "target": mid[:24].strip(),
                            "predicted": mid[24:].strip(),
                            "word tokens right": f"{h[2]}/{h[3]}",
                            "speaking frames right": f"{t[1]}/{t[2]}",
                        }
                    )
        for v in out.values():
            v["rows"] = pd.DataFrame(v["rows"])
        return out

    def frac(pair):
        return f"{pair[0]}/{pair[1]}"

    def md_table(df):
        """DataFrame -> markdown table (no tabulate dependency)."""
        cols = [str(c) for c in df.columns]
        lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
        for _, r in df.iterrows():
            lines.append("| " + " | ".join(str(x).replace("|", "\\|") for x in r.tolist()) + " |")
        return "\n".join(lines)

    def word_end(u):
        return u["words"][-1]["start"] + u["words"][-1]["dur"]

    def row_stats(r):
        """Timing of one raw row (the schema data.py reads), in seconds unless named otherwise."""
        U = r["utterances"]
        c = [np.array(r["dsu_c1_path"]), np.array(r["dsu_c2_path"])]
        T = c[0].shape[1]
        pauses = [U[i + 1]["words"][0]["start"] - word_end(U[i]) for i in range(len(U) - 1)]
        return {
            "frames": T,
            "utterances": len(U),
            "lead_in": min(u["words"][0]["start"] for u in U),
            "after_last_word": T / FPS - max(word_end(u) for u in U),
            "pauses": pauses,
        }

    def silence_code(r):
        """Most common 4-codebook tuple on frames where a channel does not speak (+-3 frames margin), and its share."""
        from collections import Counter

        cnt = Counter()
        for ch, key in ((0, "dsu_c1_path"), (1, "dsu_c2_path")):
            c = np.array(r[key])
            T = c.shape[1]
            speak = np.zeros(T, bool)
            for u in r["utterances"]:
                segs = [u] if u["speaker_idx"] == ch else (u.get("backchannels") or [])
                for s in segs:
                    if s["words"]:
                        a, b = int(s["words"][0]["start"] * FPS), int(word_end(s) * FPS) + 1
                        speak[max(0, a - 3) : b + 3] = True
            cnt.update(map(tuple, c[:, ~speak].T.tolist()))
        (code, n), total = cnt.most_common(1)[0], sum(cnt.values())
        return code, n / total

    def med_range(xs, unit=" s", nd=2):
        xs = np.asarray(xs, float)
        return f"{np.median(xs):.{nd}f}{unit} ({xs.min():.{nd}f} to {xs.max():.{nd}f})"

    return (
        frac,
        md_table,
        med_range,
        parse_tf,
        row_stats,
        silence_code,
        word_end,
    )


@app.cell
def _():
    return


@app.cell
def _(DATA, FT, LOGS, OUT, THESIS, json, mo, parse_tf, pd, pq, re):
    # Every file the slides read. Loading and the "Source" line under a slide both go through SRC, so they cannot drift apart.
    SRC = {
        # teacher_forced_check.py output (released model, and three fine-tunes)
        "tf_base": LOGS / "tf_base_val.txt",
        "tf_I_train": LOGS / "tf_run170i_ck120_train_local.txt",
        "tf_I_val": LOGS / "tf_run170i_ck120_validation_local.txt",
        "tf_N_train": LOGS / "tf_run170n_ck120_train_local.txt",
        "tf_N_val": LOGS / "tf_run170n_ck120_validation_local.txt",
        "tf_A_train": LOGS / "tf_runA100_train.txt",
        "tf_A_val": LOGS / "tf_runA100_validation.txt",
        "tf_runA": LOGS / "tf_runA_train.txt",
        "tf_runC": LOGS / "tf_runC_train.txt",
        "tf_runC100": LOGS / "tf_runC100_train.txt",
        # rows
        "rows_repeat": FT / "run170n" / "train.parquet",  # the 70 repeat-task training rows
        "row_bsd": THESIS / "tts_rows" / "scratch" / "reference_row.json",  # one Behavior-SD row, dumped from the HF dataset
        "bsd_words": DATA / "corpus" / "aligned_words.parquet",  # Behavior-SD extract: word timings
        "bsd_text": DATA / "corpus" / "text.parquet",  # Behavior-SD extract: speaker per utterance
        # clips and word lists
        "request_manifest": DATA / "checks" / "p6_full_request_s0_ok" / "manifest.jsonl",
        "reply_manifest": DATA / "checks" / "p7_reply_short_s0_ok" / "manifest.jsonl",
        "rare_words": THESIS / "oov-shared" / "wordlists" / "lists" / "prefix_gate_p4.csv",
        # runs
        "config_run170i": FT / "runs" / "run170i" / "config.json",
        "config_run170n": FT / "runs" / "run170n" / "config.json",
        "state_run170i": FT / "runs" / "run170i" / "checkpoint-120" / "trainer_state.json",
        "log_run170n": FT / "runs" / "run170n" / "training.log",
        "timings": FT / "run170n.timings.json",  # forced-alignment word timings per clip, written next to the dataset
        "scale1": FT / "scale1" / "run.json",
        "breakdown_base": OUT / "breakdown_base.json",
        "breakdown_run100": OUT / "breakdown_run100.json",
        "generation": OUT / "rows_test_run100_la" / "tuned",
    }

    def src(*names):
        """The 'Source' line of a slide: the files (relative to the thesis folder) its numbers were read from."""
        return mo.md("<small>Source: " + " · ".join(f"`{SRC[n].relative_to(THESIS)}`" for n in names) + "</small>")

    tf_base = parse_tf(SRC["tf_base"])  # released checkpoint, 18 unseen words
    # run170i and run170n have the same config.json apart from the dataset: only the narrative differs (word named / word hidden).
    # Both are checkpoint-120, scored on the laptop (CPU, fp32).
    tf_I_train, tf_I_val = parse_tf(SRC["tf_I_train"]), parse_tf(SRC["tf_I_val"])
    tf_N_train, tf_N_val = parse_tf(SRC["tf_N_train"]), parse_tf(SRC["tf_N_val"])
    tf_A_train, tf_A_val = parse_tf(SRC["tf_A_train"]), parse_tf(SRC["tf_A_val"])  # backup: lower learning rate, 100 steps
    tf_weighting = {
        "runA": parse_tf(SRC["tf_runA"]),
        "runC": parse_tf(SRC["tf_runC"]),
        "runA100": tf_A_train,
        "runC100": parse_tf(SRC["tf_runC100"]),
    }

    rows_mine = pq.read_table(SRC["rows_repeat"]).to_pylist()
    row_ref = json.loads(SRC["row_bsd"].read_text())

    def run_config(name):
        return json.loads((FT / "runs" / name / "config.json").read_text())

    def run_log(name):
        return (FT / "runs" / name / "training.log").read_text(errors="replace")

    def eval_curve(name):
        """[(step, eval_loss)] from training.log; evals happen every eval_steps."""
        every = run_config(name)["eval_steps"]
        vals = [float(x) for x in re.findall(r"\] eval_loss: ([0-9.]+)", run_log(name))]
        return [((i + 1) * every, v) for i, v in enumerate(vals)]

    def logged_losses(name):
        """Training and validation loss from a retained Hugging Face trainer state."""
        state = json.loads(SRC[f"state_{name}"].read_text())
        history = state["log_history"]
        train = [(x["step"], x["loss"]) for x in history if "loss" in x]
        validation = [(x["step"], x["eval_loss"]) for x in history if "eval_loss" in x]
        return {"train": train, "validation": validation}

    def dataset_info(name):
        return json.loads((FT / name / "run.json").read_text())

    def clip_text(kind, word):
        """The sentence the TTS was given for one clip, from the clip run's manifest. kind: 'request' | 'reply'."""
        for line in SRC[f"{kind}_manifest"].read_text().splitlines():
            m = json.loads(line)
            if m.get("word") == word:
                return m["text"]
        return None

    def clip_record(kind, word):
        """One clip's manifest record plus the local path of its wav. kind: 'request' | 'reply'."""
        for line in SRC[f"{kind}_manifest"].read_text().splitlines():
            m = json.loads(line)
            if m.get("word") == word:
                return {**m, "path": SRC[f"{kind}_manifest"].parent / m["wav"]}
        return None

    # {wav file name: [{"word", "start", "dur", "score"}]}; the file keys are cluster paths, the file name is what both machines share
    timings = {k.rsplit("/", 1)[-1]: v for k, v in json.loads(SRC["timings"].read_text()).items()}

    rare_words = pd.read_csv(SRC["rare_words"])["word"].tolist()  # held-out rare-word test list (never trained on)

    def bsd_turn_pauses(split="validation"):
        """Behavior-SD, every change of speaker: next utterance's first word start minus this utterance's last word end (s).
        Same definition as the pause in a repeat-task row; negative = overlap."""
        w = pq.read_table(SRC["bsd_words"], filters=[("split", "=", split), ("source", "=", "utterance")]).to_pandas()
        w["end"] = w["start"] + w["dur"]
        u = w.groupby(["dialogue_id", "uttr_idx"]).agg(first=("start", "min"), last=("end", "max")).reset_index()
        t = pq.read_table(
            SRC["bsd_text"], filters=[("split", "=", split), ("source", "=", "utterance")], columns=["dialogue_id", "uttr_idx", "speaker"]
        ).to_pandas()
        u = u.merge(t, on=["dialogue_id", "uttr_idx"]).sort_values(["dialogue_id", "uttr_idx"]).reset_index(drop=True)
        nxt = u.groupby("dialogue_id")[["first", "speaker"]].shift(-1)
        change = nxt["speaker"].notna() & (nxt["speaker"] != u["speaker"])
        return (nxt["first"] - u["last"])[change].to_numpy()

    breakdown = {k: json.loads(SRC[f"breakdown_{k}"].read_text()) for k in ("base", "run100")}
    return (
        breakdown,
        bsd_turn_pauses,
        clip_record,
        clip_text,
        dataset_info,
        eval_curve,
        logged_losses,
        rare_words,
        row_ref,
        rows_mine,
        run_config,
        run_log,
        src,
        tf_A_train,
        tf_A_val,
        tf_I_train,
        tf_I_val,
        tf_N_train,
        tf_N_val,
        tf_base,
        tf_weighting,
        timings,
    )


@app.cell
def _(clip_text, mo, rare_words, row_audio, rows_mine, src):
    # ===== slide 1 =====
    _w = rows_mine[0]["file_name"]  # the example word: the first training row
    _buf, _ = row_audio(_w)
    mo.vstack(
        [
            mo.md("# Goal: the model repeats a word it has just heard"),
            mo.md(
                """
    F-Actor writes rare words correctly in its text stream, but does not render them in audio.
    **If it has just heard the word, can it say it?**
    """
            ),
            mo.md(
                f"""
    | speaker | example with the word "{_w}" |
    |---|---|
    | **Tom** (request, played to the model) | {clip_text("request", _w)} |
    | **Gweneth** (the model's reply) | {clip_text("reply", _w)} |
    """
            ),
            mo.audio(src=_buf) if _buf is not None else mo.md("_(example clips are not on this machine)_"),
            mo.md(f"The probe is the same request with a rare word, for example **{rare_words[0]}** ({len(rare_words)} rare words are held out for it)."),
            mo.md("**Today:** 1 · the training rows  2 · what the model learned from them  3 · the next run, and three questions"),
            src("request_manifest", "reply_manifest", "rare_words"),
        ]
    )
    return


@app.cell
def _(json):
    def row_json(r, n_utt, width=110):
        """One raw row as text, shortened for reading: long lists are cut and say how long they are."""

        def short(xs, k, unit):
            return list(xs[:k]) + ([f"... {len(xs)} {unit}"] if len(xs) > k else [])

        def val(k, v):
            if k in ("dsu_c1_path", "dsu_c2_path"):
                return [short(cb, 4, "frames") for cb in v]
            if k.startswith("spk_emb"):
                return short([round(x, 3) for x in v], 4, "values")
            if k == "words":
                return short([{a: (round(b, 3) if isinstance(b, float) else b) for a, b in w.items()} for w in v], 3, "words")
            return round(v, 3) if isinstance(v, float) else v

        def line(indent, k, v):
            t = " " * indent + f'"{k}": ' + json.dumps(val(k, v), ensure_ascii=False)
            return (t[: width - 2] + " …" if len(t) > width else t) + ","

        order = ["file_name", "speakers", "new_narrative_s2", "num_turns", "utterances", "dsu_c1_path", "dsu_c2_path", "spk_emb_c1", "spk_emb_c2"]
        read_first = ["speaker_idx", "uttr_type", "start_time", "end_time", "words", "backchannels"]
        out = ["{"]
        for k in order + [k for k in r if k not in order]:
            if k != "utterances":
                out.append(line(1, k, r[k]))
                continue
            out.append(' "utterances": [')
            for u in r[k][:n_utt]:
                out.append("  {")
                for kk in [x for x in read_first if x in u] + [x for x in u if x not in read_first]:
                    if kk != "words":
                        out.append(line(3, kk, u[kk]))
                        continue
                    out.append('   "words": [')  # one word per line, so the count stays visible
                    out += ["    " + json.dumps(w, ensure_ascii=False) + "," for w in val("words", u[kk])]
                    out.append("   ],")
                out.append("  },")
            if len(r[k]) > n_utt:
                out.append(f"  ... {len(r[k]) - n_utt} more utterances")
            out.append(" ],")
        return "\n".join(out + ["}"])

    return (row_json,)


@app.cell
def _(FPS, md_table, mo, pd, row_json, row_ref, rows_mine, src, word_end):
    # ===== slide 2: the two rows as data =====
    _mine = rows_mine[0]

    def _facts(r):
        U, T = r["utterances"], len(r["dsu_c1_path"][0])
        u0 = U[0]
        cut = lambda t: (t[:80] + " …") if len(t) > 80 else (t or "(empty)")
        return {
            "speakers": ", ".join(r["speakers"]),
            "frames (seconds)": f"{T} ({T / FPS:.1f})",
            "utterances": len(U),
            "backchannels": sum(len(u.get("backchannels") or []) for u in U),
            "`uttr_type` values": ", ".join(sorted({"null" if u["uttr_type"] is None else str(u["uttr_type"]) for u in U})),
            "first utterance: `start_time` to `end_time`": f"{u0['start_time']:.2f} to {u0['end_time']:.2f} s",
            "first utterance: first word start to last word end": f"{u0['words'][0]['start']:.2f} to {word_end(u0):.2f} s",
            "shared `narrative`": cut(r["narrative"]),
            "`new_narrative_s2`": cut(r["new_narrative_s2"]),
            "fields per utterance": ", ".join(u0),
        }

    _a, _b = _facts(row_ref), _facts(_mine)
    _table = pd.DataFrame({"field": list(_a), "Behavior-SD row": list(_a.values()), f"repeat-task row ('{_mine['file_name']}')": list(_b.values())})
    _code = lambda r, n: mo.md("```json\n" + row_json(r, n) + "\n```")
    mo.vstack(
        [
            mo.md("# Training rows as data: Behavior-SD and the repeat task"),
            mo.ui.tabs(
                {
                    "Behavior-SD row": _code(row_ref, 1),
                    f"Repeat-task row ('{_mine['file_name']}')": _code(_mine, 2),
                    "Fields compared": mo.md(md_table(_table)),
                }
            ),
            mo.md("Both rows are read from file; long lists are cut and state their length."),
            src("row_bsd", "rows_repeat"),
        ]
    )
    return


@app.cell
def _(mo, rows_mine):
    # the word shown on the next slide (its own cell, so the slide re-runs when the choice changes)
    word_pick = mo.ui.dropdown(options=[r["file_name"] for r in rows_mine], value=rows_mine[0]["file_name"], label="word")
    return (word_pick,)


@app.cell
def _(
    C_REP,
    C_REQ,
    FPS,
    GRID,
    INK,
    INK2,
    REPO,
    SURFACE,
    clip_record,
    md_table,
    mo,
    np,
    pd,
    plt,
    row_json,
    rows_mine,
    sf,
    silence_code,
    src,
    sys,
    timings,
    word_end,
    word_pick,
):
    # ===== slide 3: how a repeat-task row is built, one step per tab =====
    if str(REPO / "training") not in sys.path:
        sys.path.insert(0, str(REPO / "training"))
    from dialogue_creation.get_prompt import build_prompt as _build_prompt  # the repo's own prompt builder

    _w = word_pick.value
    _row = next(r for r in rows_mine if r["file_name"] == _w)
    _T = len(_row["dsu_c1_path"][0])
    _req, _rep = clip_record("request", _w), clip_record("reply", _w)
    _have_audio = bool(_req and _rep and _req["path"].exists() and _rep["path"].exists())

    def _mono(path):
        a, sr = sf.read(path)
        return (a.mean(axis=1) if a.ndim > 1 else a), sr

    def _lane_axes(ax, labels, x_max):
        ax.set_facecolor(SURFACE)
        ax.set_xlim(0, x_max)
        ax.set_ylim(-0.6, 1.6)
        ax.set_yticks([1.0, 0.0], labels, fontsize=9, color=INK)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", colors=INK2, labelsize=8.5)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_visible(False)

    # ---- step 1: the two clips
    if _have_audio:
        _clips = pd.DataFrame(
            [
                {"clip": "request", "voice": r["voice"], "text given to the TTS": r["text"], "seconds": r["audio_seconds"], "TTS": r["tts"], "file": r["wav"]}
                for r in (_req, _rep)
            ]
        )
        _step1 = mo.vstack(
            [
                mo.md(md_table(_clips.assign(clip=["request", "reply"]))),
                mo.hstack(
                    [mo.vstack([mo.md("request"), mo.audio(src=str(_req["path"]))]), mo.vstack([mo.md("reply"), mo.audio(src=str(_rep["path"]))])],
                    justify="start",
                ),
            ]
        )
    else:
        _step1 = mo.md("_(the clips for this word are not on this machine)_")

    # ---- step 2: word timings from forced alignment
    def _timing_table(rec):
        ws = timings.get(rec["wav"].rsplit("/", 1)[-1], []) if rec else []
        return mo.md(md_table(pd.DataFrame([{"word": w["word"], "start (s)": w["start"], "dur (s)": w["dur"], "score": round(w.get("score", float("nan")), 2)} for w in ws])))

    _step2 = mo.hstack(
        [mo.vstack([mo.md("**request clip**"), _timing_table(_req)]), mo.vstack([mo.md("**reply clip** (times are inside the clip)"), _timing_table(_rep)])],
        justify="start",
        gap=3,
    )

    # ---- step 3: the clips laid back to back on two channels
    if _have_audio:
        (_a, _sr), (_b, _) = _mono(_req["path"]), _mono(_rep["path"])
        _off, _total = len(_a) / _sr, (len(_a) + len(_b)) / _sr
        _fig3, _ax = plt.subplots(figsize=(10.5, 2.6), facecolor=SURFACE)
        _lane_axes(_ax, ["ch 1  request", "ch 2  reply"], _total * 1.02)
        for _sig, _t0, _lane, _col in ((_a, 0.0, 1.0, C_REQ), (_b, _off, 0.0, C_REP)):
            _n = max(1, len(_sig) // 900)
            _env = np.abs(_sig[: len(_sig) // _n * _n]).reshape(-1, _n).max(axis=1)
            _env = 0.42 * _env / max(_env.max(), 1e-9)
            _t = _t0 + np.arange(len(_env)) * _n / _sr
            _ax.fill_between(_t, _lane - _env, _lane + _env, color=_col, lw=0)
        _ax.axvline(_off, color=INK, lw=1, ls=(0, (4, 3)))
        _ax.text(_off - 0.06, 1.48, f"reply clip starts at {_off:.2f} s, where the request clip ends", fontsize=8.5, color=INK, va="center", ha="right")
        _ax.axvline(_total, color=INK, lw=1)
        _ax.set_xlabel("seconds", fontsize=9, color=INK2)
        _fig3.tight_layout()
        _step3 = mo.vstack([_fig3, mo.md(f"Nothing is trimmed or added. The other channel is zeros. Total: {_total:.2f} s.")])
    else:
        _step3 = mo.md("_(the clips for this word are not on this machine)_")

    # ---- step 4: codec frames
    _sil, _ = silence_code(_row)
    _fig4, _ax4 = plt.subplots(figsize=(10.5, 1.9), facecolor=SURFACE)
    _lane_axes(_ax4, ["ch 1  request", "ch 2  reply"], _T)
    for _key, _lane, _col in (("dsu_c1_path", 1.0, C_REQ), ("dsu_c2_path", 0.0, C_REP)):
        _codes = np.array(_row[_key]).T  # [T, 4]
        for _f in np.flatnonzero((_codes != np.array(_sil)).any(axis=1)):
            _ax4.add_patch(plt.Rectangle((_f + 0.06, _lane - 0.3), 0.88, 0.6, color=_col, lw=0))
    _ax4.set_xlabel("codec frame (80 ms each)", fontsize=9, color=INK2)
    _fig4.tight_layout()
    _first = pd.DataFrame(
        {
            "frame": list(range(4)),
            "ch 1 codes (4 codebooks)": [str(tuple(np.array(_row["dsu_c1_path"])[:, i].tolist())) for i in range(4)],
            "ch 2 codes (4 codebooks)": [str(tuple(np.array(_row["dsu_c2_path"])[:, i].tolist())) for i in range(4)],
        }
    )
    _step4 = mo.vstack(
        [
            mo.md(f"NanoCodec encodes each channel on its own: **{_T} frames** ({_T / FPS:.2f} s), 4 codebooks per channel. A filled frame differs from the silence code `{_sil}`."),
            _fig4,
            mo.md(md_table(_first)),
        ]
    )

    # ---- step 5: the stored row
    _step5 = mo.md("```json\n" + row_json(_row, 2) + "\n```")

    # ---- step 6: what the unchanged loader adds
    _prompt = _build_prompt(_row, max_length=2048, orig_dsu_length=_T, role_to_speaker_map={"system": 1, "user": 0}, use_system_narrative=True, speech=True)
    _reply = _row["utterances"][1]
    _stream = pd.DataFrame(
        [{"reply word": w["word"], "start (s)": round(w["start"], 3), "text-stream frame": int(w["start"] * FPS), "end (s)": round(w["start"] + w["dur"], 3)} for w in _reply["words"]]
    )
    _step6 = mo.vstack(
        [
            mo.md("Prompt, from `build_prompt` in `get_prompt.py` with `--system_speaker 1 --use_system_narrative`:"),
            mo.md("```text\n" + _prompt + "\n```"),
            mo.md(f"Text stream: the reply's words at their start frame (start x {FPS}), padding elsewhere. Audio is delayed by 2 frames. Reply ends at {word_end(_reply):.2f} s, row at {_T / FPS:.2f} s."),
            mo.md(md_table(_stream)),
        ]
    )

    mo.vstack(
        [
            mo.hstack([mo.md("# How a repeat-task row is built"), word_pick], justify="space-between", align="center"),
            mo.ui.tabs(
                {
                    "1 · two clips": _step1,
                    "2 · word timings": _step2,
                    "3 · back to back": _step3,
                    "4 · codec frames": _step4,
                    "5 · the stored row": _step5,
                    "6 · added at load time": _step6,
                }
            ),
            src("request_manifest", "reply_manifest", "timings", "rows_repeat"),
        ]
    )
    return


@app.cell
def _(C_REP, C_REQ, GRID, INK, INK2, Rectangle, SURFACE, plt, word_end):
    def draw_row(ax, row, lane_labels, t_max, end_s=None):
        """One row as two lanes (channel 1 on top). Each block is one aligned word."""
        color, lane = {0: C_REQ, 1: C_REP}, {0: 1.0, 1: 0.0}
        ax.set_facecolor(SURFACE)
        shown = []
        for u in row["utterances"]:
            segs = [(u["speaker_idx"], u["words"])] + [(1 - u["speaker_idx"], b["words"]) for b in (u.get("backchannels") or [])]
            for spk, words in segs:
                words = [w for w in words if w["start"] < t_max]
                for w in words:
                    x0, x1 = w["start"], min(w["start"] + w["dur"], t_max)
                    ax.add_patch(Rectangle((x0, lane[spk] - 0.2), max(x1 - x0 - 0.015, 0.01), 0.4, color=color[spk], lw=0))
                    ax.text((x0 + x1) / 2, lane[spk] + 0.25, w["word"].lower(), rotation=50, ha="left", va="bottom", fontsize=7.5, color=INK2)
            if u["words"][0]["start"] < t_max:
                shown.append(u)
        for a, b in zip(shown, shown[1:]):  # pause where the speaker changes
            x0, x1 = word_end(a), b["words"][0]["start"]
            if a["speaker_idx"] != b["speaker_idx"] and x1 - x0 > 0.05 and x1 < t_max:
                ax.annotate("", xy=(x0, 0.5), xytext=(x1, 0.5), arrowprops=dict(arrowstyle="<->", color=INK, lw=1))
                ax.text((x0 + x1) / 2, 0.57, f"{x1 - x0:.2f} s", ha="center", va="bottom", fontsize=9, color=INK)
        if end_s is not None:
            ax.axvline(end_s, color=INK, lw=1, ls=(0, (4, 3)))
            ax.text(end_s + 0.06, 0.5, "row ends,\nEOS on all heads", ha="left", va="center", fontsize=8.5, color=INK)
        ax.set_xlim(0, t_max)
        ax.set_ylim(-0.45, 1.95)
        ax.set_yticks([1.0, 0.0], lane_labels, fontsize=9, color=INK)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", colors=INK2, labelsize=8.5)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_visible(False)

    def fig_two_rows(row_ref, row_mine, t_max=8.0):
        fig, axes = plt.subplots(2, 1, figsize=(11, 5.2), sharex=True, facecolor=SURFACE)
        n_ref = len(row_ref["dsu_c1_path"][0]) / 12.5
        draw_row(axes[0], row_ref, [f"ch 1  {row_ref['speakers'][0]}", f"ch 2  {row_ref['speakers'][1]}"], t_max)
        axes[0].set_title(f"Behavior-SD row (first {t_max:.0f} s of {n_ref:.0f} s)", loc="left", fontsize=10.5, color=INK, pad=6)
        n_mine = len(row_mine["dsu_c1_path"][0]) / 12.5
        draw_row(axes[1], row_mine, ["ch 1  Tom\n(request)", "ch 2  Gweneth\n(reply, model's role)"], t_max, end_s=n_mine)
        axes[1].set_title(f"Repeat-task row for '{row_mine['file_name']}' (complete)", loc="left", fontsize=10.5, color=INK, pad=6)
        axes[1].set_xlabel("seconds", fontsize=9, color=INK2)
        fig.tight_layout(h_pad=1.6)
        return fig

    return (fig_two_rows,)


@app.cell
def _(DATA, io, np, sf):
    def row_audio(word):
        """The two source clips laid back to back as build_row does, as stereo (left = request, right = reply).
        These are the TTS clips before the codec, not decoded training codes. -> (wav bytes, info) or (None, None)."""
        req = DATA / "checks" / "p6_full_request_s0_ok" / "audio" / f"{word}__tom.wav"
        rep = DATA / "checks" / "p7_reply_short_s0_ok" / "audio" / f"{word}__gweneth.wav"
        if not (req.exists() and rep.exists()):
            return None, None
        a, sr = sf.read(req)
        b, sr_b = sf.read(rep)
        if sr != sr_b:
            return None, None
        a, b = (x.mean(axis=1) if x.ndim > 1 else x for x in (a, b))
        st = np.zeros((len(a) + len(b), 2), dtype=np.float32)
        st[: len(a), 0], st[len(a) :, 1] = a, b
        buf = io.BytesIO()
        sf.write(buf, st, sr, format="WAV")
        buf.seek(0)
        return buf, {"request_s": len(a) / sr, "reply_s": len(b) / sr, "sr": sr, "request": req.name, "reply": rep.name}

    return (row_audio,)


@app.cell
def _(
    FPS,
    bsd_turn_pauses,
    fig_two_rows,
    md_table,
    med_range,
    mo,
    np,
    pd,
    row_audio,
    row_ref,
    row_stats,
    rows_mine,
    silence_code,
    src,
):
    # ===== slide 4: timing =====
    _mine = rows_mine[0]
    fig_rows = fig_two_rows(row_ref, _mine)

    _s_ref = row_stats(row_ref)
    _s_mine = [row_stats(r) for r in rows_mine]
    _code_ref, _share_ref = silence_code(row_ref)
    _code_mine, _share_mine = silence_code(_mine)
    _pauses_mine = [p for s in _s_mine for p in s["pauses"]]
    _bsd = bsd_turn_pauses("validation")  # every change of speaker in the Behavior-SD validation split

    _cmp = pd.DataFrame(
        [
            ["Pause between turns", f"**{np.median(_bsd):.2f} s** (median over {len(_bsd):,} speaker changes, validation split)", f"**{med_range(_pauses_mine)}**"],
            ["Audio after the last word", f"{_s_ref['after_last_word']:.2f} s", med_range([s["after_last_word"] for s in _s_mine])],
            ["Lead-in before the first word", f"{_s_ref['lead_in']:.2f} s", med_range([s["lead_in"] for s in _s_mine])],
            ["Length", f"{_s_ref['frames']} frames, {_s_ref['utterances']} utterances", med_range([s["frames"] for s in _s_mine], " frames", 0) + ", 2 utterances"],
            ["Silence code (4 codebooks)", f"`{_code_ref}`", f"`{_code_mine}`" + (" (identical)" if _code_mine == _code_ref else " (**differs**)")],
        ],
        columns=["", "Behavior-SD (the reference row)", f"Repeat task ({len(rows_mine)} rows: median, range)"],
    )

    _buf, _info = row_audio(_mine["file_name"])
    _audio = (
        mo.vstack([mo.md(f"**Listen** (`{_mine['file_name']}`: source clips back to back, left = Tom, right = Gweneth; before the codec)"), mo.audio(src=_buf)])
        if _buf is not None
        else mo.md("_(source clips for this word are not on this machine)_")
    )

    mo.vstack(
        [
            mo.md("# Row timing: Behavior-SD and the repeat task"),
            mo.md("Two TTS clips laid back to back; the channel that is not speaking is digital silence."),
            fig_rows,
            mo.md(md_table(_cmp)),
            src("row_bsd", "rows_repeat", "bsd_words", "bsd_text"),
            _audio,
            mo.accordion(
                {
                    "Two more repeat-task rows (shortest and longest pause)": mo.md(
                        md_table(
                            pd.DataFrame(
                                [
                                    {
                                        "word": r["file_name"],
                                        "pause (s)": f"{s['pauses'][0]:.2f}",
                                        "frames": s["frames"],
                                        "seconds": f"{s['frames'] / FPS:.2f}",
                                        "lead-in (s)": f"{s['lead_in']:.2f}",
                                        "after last word (s)": f"{s['after_last_word']:.2f}",
                                    }
                                    for r, s in sorted(zip(rows_mine, _s_mine), key=lambda rs: rs[1]["pauses"][0])[:: len(rows_mine) - 1]
                                ]
                            )
                        )
                    )
                }
            ),
        ]
    )
    return


@app.cell
def _(frac, md_table, mo, src, tf_base):
    # ===== slide 3 =====
    _b = tf_base[("orig", "validation")]
    EXAMPLE_UNSEEN = _b["rows"]["word"].tolist()[:3]  # rule: the first three words of the validation split; reused on later slides
    mo.vstack(
        [
            mo.md("# Released model on the repeat task"),
            mo.md(
                f"""
    Released checkpoint, 18 held-out words, teacher-forced (whole row fed in, greedy argmax).

    - Reply word right: **{frac(_b["word"])}**
    - Reply word tokens right: {frac(_b["tokens"])} (mostly the "okay")
    - Speaking frames with all 4 codebooks right: {frac(_b["speak"])}; silent frames: {frac(_b["silent"])}
    """
            ),
            mo.md(md_table(_b["rows"][_b["rows"]["word"].isin(EXAMPLE_UNSEEN)])),
            src("tf_base"),
            mo.accordion({"All 18 words": mo.md(md_table(_b["rows"]))}),
        ]
    )
    return (EXAMPLE_UNSEEN,)


@app.cell
def _(dataset_info, pd, re, run_config, run_log):
    def hyper_table(evaluated):
        """One column per run, from each run's config.json, training.log and its dataset's run.json."""
        cols = {}
        for name, ckpt in evaluated.items():
            c = run_config(name)
            d = dataset_info(c["speech_path"].rstrip("/").split("/")[-1])
            narr = d["args"]["narrative_replier"]
            n_eval = len(re.findall(r"\] eval_loss:", run_log(name)))
            cols[name] = {
                "Training / held-out words": f"{d['n_train']} / {d['n_val']}",
                "Narrative of the model's role": f"*{narr.format(partner='Tom', word='X')}*",
                "Steps run (max_steps)": f"{n_eval * c['eval_steps']} ({c['max_steps']})",
                "Learning rate": c["learning_rate"],
                "Batch x accumulation": f"{c['train_batch_size']} x {c['gradient_accumulation_steps']}",
                "Early stopping patience": c["early_stopping_patience"] or "off",
                "Checkpoint evaluated": ckpt,
                "Loss": "reply streams only" if c["calc_loss_on_c1_only"] else "both speakers",
            }
        df = pd.DataFrame(cols)
        df.insert(0, "", df.index)
        return df

    return (hyper_table,)


@app.cell
def _(
    fig_train_val_loss,
    frac,
    hyper_table,
    logged_losses,
    md_table,
    mo,
    pd,
    src,
    tf_I_train,
    tf_I_val,
):
    # ===== slide 4 =====
    _o, _n = tf_I_train[("orig", "train")], tf_I_train[("none", "train")]
    _vo, _vn = tf_I_val[("orig", "validation")], tf_I_val[("none", "validation")]
    _words = _o["rows"]["word"].tolist()[:3]  # rule: the first three training words
    _pair = _o["rows"].merge(_n["rows"], on=["word", "target"], suffixes=(" (word in narrative)", " (word removed)"))
    _cols = ["word", "target", "predicted (word in narrative)", "predicted (word removed)"]

    _summary = pd.DataFrame(
        [
            ["70 trained words", f"**{frac(_o['word'])}**", f"**{frac(_n['word'])}**"],
            ["18 unseen words", frac(_vo["word"]), frac(_vn["word"])],
        ],
        columns=["run170i, reply word right", "narrative names the word", "word replaced by 'a word'"],
    )
    _loss = logged_losses("run170i")
    fig_loss = fig_train_val_loss(_loss)
    _best_val_step, _ = min(_loss["validation"], key=lambda point: point[1])
    mo.vstack(
        [
            mo.md("# Fine-tune with the word in the prompt (run170i)"),
            mo.md(
                "Same model, same rows, same audio. Only the narrative changes: "
                "*'Tom asks you to listen carefully and repeat the word **X**'* against *'... repeat **a word**'*."
            ),
            mo.md(md_table(_summary)),
            mo.md("## Training and held-out loss"),
            fig_loss,
            mo.md(
                f"Train loss is logged each step; held-out loss is measured every 10 steps on 18 unseen words. "
                f"Held-out loss is lowest at step {_best_val_step}, then rises while train loss continues down. "
                "The per-step trainer history is retained for run170i; the local run170n log retains validation evaluations only."
            ),
            src("tf_I_train", "tf_I_val", "config_run170i", "state_run170i"),
            mo.md(md_table(_pair[_pair["word"].isin(_words)][_cols])),
            mo.md(
                f"Without the word in the prompt, {_n['word'][0]} of {_n['word'][1]} trained words are still right: "
                f"those can only come from the request audio. Unseen words depend on the prompt entirely ({frac(_vo['word'])} against {frac(_vn['word'])})."
            ),
            mo.accordion(
                {
                    "All 70 trained words": mo.md(md_table(_pair[_cols])),
                    "Run settings": mo.md(md_table(hyper_table({"run170i": "checkpoint-120 (last saved)", "run170n": "checkpoint-120 (last saved)"}))),
                }
            ),
        ]
    )
    return


@app.cell
def _(C_REP, C_REQ, GRID, INK, INK2, SURFACE, plt):
    def fig_eval_loss(curve, marks):
        """Eval loss over training steps for one run; marks = {step: label}."""
        fig, ax = plt.subplots(figsize=(6.2, 3.4), facecolor=SURFACE)
        ax.set_facecolor(SURFACE)
        xs, ys = zip(*curve)
        ax.plot(xs, ys, color=C_REQ, lw=2, marker="o", ms=4)
        by_step = dict(curve)
        for step, (label, dx, dy) in marks.items():
            ax.plot([step], [by_step[step]], marker="o", ms=9, mfc=C_REQ, mec=SURFACE, mew=2)
            ax.annotate(label, xy=(step, by_step[step]), xytext=(step + dx, by_step[step] + dy), fontsize=8.5, color=INK, va="center",
                        arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
        ax.set_xlabel("training step", fontsize=9, color=INK2)
        ax.set_ylabel("eval loss (18 unseen words)", fontsize=9, color=INK2)
        ax.set_ylim(min(ys) - 0.03, max(ys) + 0.03)
        ax.set_xlim(0, max(xs) + 8)
        ax.tick_params(colors=INK2, labelsize=8.5)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        fig.tight_layout()
        return fig

    def fig_train_val_loss(history):
        """Training and held-out loss in separate panels so both scales stay readable."""
        fig, (train_ax, val_ax) = plt.subplots(1, 2, figsize=(9.2, 3.4), sharex=True, facecolor=SURFACE)
        train_steps, train_values = zip(*history["train"])
        val_steps, val_values = zip(*history["validation"])
        for ax in (train_ax, val_ax):
            ax.set_facecolor(SURFACE)
            ax.tick_params(colors=INK2, labelsize=8.5)
            ax.grid(axis="y", color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            ax.set_xlabel("training step", fontsize=9, color=INK2)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(GRID)

        train_ax.plot(train_steps, train_values, color=C_REQ, lw=1.4)
        train_ax.set_title("Training rows", loc="left", fontsize=10, color=INK)
        train_ax.set_ylabel("training loss", fontsize=9, color=INK2)

        val_ax.plot(val_steps, val_values, color=C_REP, lw=1.8, marker="o", ms=4)
        best_i = min(range(len(val_values)), key=val_values.__getitem__)
        best_step, best_loss = val_steps[best_i], val_values[best_i]
        val_ax.scatter([best_step], [best_loss], color=C_REP, edgecolor=SURFACE, linewidth=1.5, s=75, zorder=3)
        val_ax.annotate(
            f"lowest at step {best_step}",
            xy=(best_step, best_loss),
            xytext=(best_step + 10, best_loss + 0.025),
            fontsize=8.5,
            color=INK,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8),
        )
        val_ax.set_title("Held-out words (18)", loc="left", fontsize=10, color=INK)
        val_ax.set_ylabel("validation loss", fontsize=9, color=INK2)
        val_ax.set_xlim(0, max(train_steps) + 5)
        fig.tight_layout(w_pad=2)
        return fig

    return fig_eval_loss, fig_train_val_loss


@app.cell
def _(
    EXAMPLE_UNSEEN,
    eval_curve,
    fig_eval_loss,
    frac,
    md_table,
    mo,
    pd,
    src,
    tf_I_train,
    tf_I_val,
    tf_N_train,
    tf_N_val,
    tf_base,
):
    # ===== slide 5 =====
    _nt, _nv = tf_N_train[("orig", "train")], tf_N_val[("orig", "validation")]
    _grid = pd.DataFrame(
        [
            [
                "evaluated **with** the word in the prompt",
                f"{frac(tf_I_train[('orig', 'train')]['word'])} trained, {frac(tf_I_val[('orig', 'validation')]['word'])} unseen",
                "not applicable",
            ],
            [
                "evaluated **without** the word in the prompt",
                f"{frac(tf_I_train[('none', 'train')]['word'])} trained, {frac(tf_I_val[('none', 'validation')]['word'])} unseen",
                f"**{frac(_nt['word'])} trained, {frac(_nv['word'])} unseen**",
            ],
        ],
        columns=["reply word right", "trained with the word in the prompt (run170i)", "trained without it (run170n)"],
    )

    def _pred(tf, key):
        return tf[key]["rows"].set_index("word")["predicted"]

    _unseen = pd.DataFrame(
        {
            "target": tf_base[("orig", "validation")]["rows"].set_index("word")["target"],
            "released": _pred(tf_base, ("orig", "validation")),
            "run170i (word in prompt)": _pred(tf_I_val, ("orig", "validation")),
            "run170n (word hidden)": _pred(tf_N_val, ("orig", "validation")),
        }
    )
    _unseen.insert(0, "unseen word", _unseen.index)
    _trained3 = _nt["rows"].head(3)[["word", "target", "predicted", "speaking frames right"]]

    _curve = eval_curve("run170n")
    fig_eval = fig_eval_loss(
        _curve,
        {
            20: ("lowest eval loss:\nwhat early stopping keeps", 8, -0.012),
            _curve[-1][0]: (f"evaluated here:\n{frac(_nt['word'])} trained, {frac(_nv['word'])} unseen", -58, -0.05),
        },
    )

    mo.vstack(
        [
            mo.md("# Fine-tune with the word hidden from the prompt (run170n)"),
            mo.md("The two runs differ only in the narrative of the training rows (same words, steps, learning rate, seed)."),
            mo.md(md_table(_grid)),
            src("tf_I_train", "tf_I_val", "tf_N_train", "tf_N_val", "log_run170n"),
            mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.md(
                                f"""
    - Trained words without the word in the prompt: {frac(tf_I_train[('none', 'train')]['word'])} → **{frac(_nt['word'])}**. Only the request audio can tell them apart.
    - Unseen words: zero in both runs ("okay", then no word).
    - Eval loss is on unseen words, so it rises. Early stopping kept step 20.
    """
                            ),
                            mo.md(md_table(_trained3)),
                        ]
                    ),
                    fig_eval,
                ],
                widths=[1.1, 1],
                align="start",
            ),
            mo.md(md_table(_unseen[_unseen["unseen word"].isin(EXAMPLE_UNSEEN)])),
            mo.accordion(
                {
                    "All 18 unseen words, three models": mo.md(md_table(_unseen)),
                    "All 70 trained words, run170n": mo.md(md_table(_nt["rows"])),
                }
            ),
        ]
    )
    return


@app.cell
def _(dataset_info, mo, rare_words, src):
    # ===== slide 6 =====
    _d = dataset_info("scale1")
    _text = mo.md(
        f"""
    # Next run: {_d['n_train']:,} words. 

    | | |
    |---|---|
    | Training / held-out words | **{_d['n_train']:,}** / **{_d['n_val']}** (the {len(rare_words)} rare test words are excluded) |
    | Narrative | word hidden |
    | Checkpoints | fixed steps, no early stopping, each scored on the held-out words |
    | Held-out words go up | copying is learnable: the probe can be built |
    | Held-out words stay at zero | it memorises at this size too: the probe needs another design |

    """
    )
    mo.vstack([_text, src("scale1", "rare_words")])
    return


@app.cell
def _(mo):
    mo.md("""
    # Backup

    Training flags · text loss by token type · three fine-tunes compared · word loss weight · free generation and early EOS
    """)
    return


@app.cell
def _(REPO, fields, md_table, mo, pd, re, run_config, sys):
    # ===== backup: flags against the released training script =====
    def _sh_flags(path):
        txt = path.read_text().replace("\\\n", " ")
        var = dict(re.findall(r"^(\w+)=\"?([^\"\n]*)\"?$", txt, flags=re.M))
        toks = txt.split("training/finetune.py", 1)[1].split()
        out, i = {}, 0
        while i < len(toks):
            if toks[i].startswith("--"):
                k = toks[i][2:]
                if i + 1 < len(toks) and not toks[i + 1].startswith("--"):
                    v = toks[i + 1].strip('"')
                    out[k] = re.sub(r"\$\{?(\w+)\}?", lambda m: var.get(m[1], m[0]), v)
                    i += 2
                    continue
                out[k] = True
            i += 1
        return out

    sys.path.insert(0, str(REPO / "training"))
    from arguments.arguments import DataArgs, ModelArgs, TrainingArgs

    _defaults = {f.name: f.default for cls in (ModelArgs, DataArgs, TrainingArgs) for f in fields(cls)}
    _hers = {**_defaults, **_sh_flags(REPO / "scripts" / "train" / "train_nanocodec_special_tokens.sh")}
    _mine = run_config("run170n")
    _skip = {"model_name", "output_dir", "preprocessing_num_workers", "save_steps", "save_total_limit", "logging_steps", "debug"}
    _rows = []
    for _k in _defaults:
        if _k in _skip or _k not in _mine:
            continue
        _a, _b = _hers[_k], _mine[_k]
        if str(_a) != str(_b) and not (_a in (None, False) and _b in (None, False)):
            _rows.append({"flag": f"`--{_k}`", "released model (train_nanocodec_special_tokens.sh)": _a, "my fine-tune (run170n)": _b})
    mo.vstack(
        [
            mo.md("# Backup: training flags, released script and run170n"),
            mo.md(
                "Left: the script plus `arguments.py` defaults for what it does not set (4 GPUs via accelerate). "
                "Right: the resolved `config.json` of run170n (1 GPU). Flags that are equal are not listed."
            ),
            mo.md(md_table(pd.DataFrame(_rows))),
        ]
    )
    return


@app.cell
def _(breakdown, md_table, mo, pd):
    # ===== backup: text-loss composition =====
    _names = {"prompt": "prompt tokens", "silence_pad": "[SILENCE_PAD]", "utterance_pad": "[UTTERANCE_PAD]", "word_pad": "[WORD_PAD]", "word": "real word tokens"}
    _b, _t = breakdown["base"]["groups"], breakdown["run100"]["groups"]
    _df = pd.DataFrame(
        [
            {
                "text targets": _names[k],
                "share of targets": f"{_b[k]['pct_targets']:.1f}%",
                "share of text loss (released)": f"{_b[k]['pct_loss']:.1f}%",
                "mean CE, released": f"{_b[k]['mean_ce']:.2f}",
                "mean CE, after run100": f"{_t[k]['mean_ce']:.2f}",
            }
            for k in _names
        ]
    )
    mo.vstack(
        [
            mo.md("# Backup: text loss by token type"),
            mo.md("The 18 validation rows (narrative names the word). Text loss only; audio loss is added on top in full."),
            mo.md(md_table(_df)),
            mo.md(
                "The prompt is more than half of the text targets and most of the text loss; the reply's word tokens are about 1% of targets. "
                "`--mask_prompt_loss` and `--word_loss_weight` address this (both default off)."
            ),
        ]
    )
    return


@app.cell
def _(
    frac,
    md_table,
    mo,
    pd,
    run_config,
    tf_A_train,
    tf_A_val,
    tf_I_train,
    tf_I_val,
    tf_N_train,
    tf_N_val,
):
    # ===== backup: three fine-tunes side by side =====
    def _row(name, names_word, tr, va, ckpt):
        c = run_config(name)
        with_word = lambda tf, split: frac(tf[("orig", split)]["word"]) if names_word else "n/a"
        without = lambda tf, split: frac(tf[("none" if names_word else "orig", split)]["word"])
        return {
            "run": name,
            "narrative in training": "names the word" if names_word else "hides the word",
            "learning rate": c["learning_rate"],
            "checkpoint": ckpt,
            "trained, word in prompt": with_word(tr, "train"),
            "trained, word removed": without(tr, "train"),
            "unseen, word in prompt": with_word(va, "validation"),
            "unseen, word removed": without(va, "validation"),
            "trained speaking frames right": frac(tr[("orig", "train")]["speak"]),
        }

    _df = pd.DataFrame(
        [
            _row("runA100", True, tf_A_train, tf_A_val, "final (step 100)"),
            _row("run170i", True, tf_I_train, tf_I_val, "checkpoint-120"),
            _row("run170n", False, tf_N_train, tf_N_val, "checkpoint-120"),
        ]
    )
    _a, _i = tf_A_train[("none", "train")]["word"], tf_I_train[("none", "train")]["word"]
    mo.vstack(
        [
            mo.md("# Backup: three fine-tunes compared"),
            mo.md(
                "Reply word right, teacher-forced. runA100 has a 5x lower learning rate and 100 steps; "
                "the other two are the matched pair from the main slides."
            ),
            mo.md(md_table(_df)),
            mo.md(
                f"With less training the model takes the word only from the prompt ({_a[0]} of {_a[1]} without it). With more, it also picks up "
                f"some of the audio ({_i[0]} of {_i[1]}), and its audio frames on trained words go up. "
                "None of the three gets an unseen word right without the prompt."
            ),
        ]
    )
    return


@app.cell
def _(frac, md_table, mo, pd, run_config, tf_weighting):
    # ===== backup: loss weighting =====
    _rows = []
    for _name, _tf in tf_weighting.items():
        _c, _s = run_config(_name), _tf[("orig", "train")]
        _rows.append(
            {
                "run": _name,
                "steps": _c["max_steps"],
                "word_loss_weight": _c["word_loss_weight"],
                "trained reply words right": frac(_s["word"]),
                "speaking frames right": frac(_s["speak"]),
            }
        )
    mo.vstack(
        [
            mo.md("# Backup: word loss weight"),
            mo.md("70 trained words, narrative names the word, teacher-forced."),
            mo.md(md_table(pd.DataFrame(_rows).sort_values(["steps", "word_loss_weight"]))),
            mo.md("The two 100-step runs use different weights (1 and 5), and the weighted one has fewer audio frames right."),
        ]
    )
    return


@app.cell
def _(OUT, json, mo):
    # ===== backup: generation =====
    _dir = OUT / "rows_test_run100_la" / "tuned"
    _recs = sorted(_dir.glob("*.json")) if _dir.exists() else []
    _ex = []
    for _p in _recs[:2]:  # rule: the first two words alphabetically
        _j = json.loads(_p.read_text())
        _g = _j["generation"]
        _stem = _p.with_suffix("")
        _ex.append(
            mo.vstack(
                [
                    mo.md(
                        f"**{_p.stem}**: ended by `{_g['ended_by']}` after {_g['n_generated_frames']} frames. "
                        f"Reply text stream: *{_g['speaker2']['transcript'] or '(empty)'}*"
                    ),
                    mo.hstack(
                        [
                            mo.vstack([mo.md("request (prefix)"), mo.audio(src=str(_stem) + "_c1_prefix.wav")]),
                            mo.vstack([mo.md("reply channel after the prefix"), mo.audio(src=str(_stem) + "_c2_cont.wav")]),
                        ],
                        justify="start",
                    ),
                ]
            )
        )
    mo.vstack(
        [
            mo.md("# Backup: free generation and early EOS"),
            mo.md(
                """
    - Free generation: the request is given as a prefix and the model continues. The main slides are teacher-forced.
    - Runs ended right after the request; cause and fix are in a separate notebook: `notebooks/05_early_eos.py`.
    - Below: two dialogues from run100 (word in the prompt), tail 0.3 s plus look-ahead, read from their run records.
    """
            ),
            *_ex,
        ]
    )
    return


if __name__ == "__main__":
    app.run()

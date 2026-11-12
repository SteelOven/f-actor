import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    # Side story to 04_advisor_update.py: why free generation ended right after the request, and what fixed it.
    #   .venv/bin/marimo edit notebooks/05_early_eos.py     (repo root)
    #   .venv/bin/marimo run  notebooks/05_early_eos.py     (presentation mode)
    import json
    import re
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    import soundfile as sf
    from matplotlib.colors import LinearSegmentedColormap

    REPO = Path(__file__).resolve().parent.parent
    THESIS = REPO.parent
    DATA = THESIS / "oov-data"
    FT = DATA / "finetune"
    OUT = DATA / "f-actor" / "outputs"
    EOS = DATA / "scratch_eos"
    FPS = 12.5  # NanoCodec frames per second (one frame = 80 ms)

    # dataviz reference palette: categorical slots 1 and 2, a one-hue blue ramp for magnitude, inks and surface
    C1, C2 = "#2a78d6", "#eb6834"
    BLUES = LinearSegmentedColormap.from_list("blues", ["#eef4fd", "#cde2fb", "#86b6ef", "#2a78d6", "#0d366b"])
    INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    return (
        BLUES,
        C1,
        C2,
        EOS,
        FPS,
        FT,
        GRID,
        INK,
        INK2,
        OUT,
        SURFACE,
        THESIS,
        json,
        np,
        pd,
        plt,
        pq,
        re,
        sf,
    )


@app.cell
def _(EOS, FPS, FT, OUT, THESIS, json, mo, pd, pq, re):
    # Every file the slides read. Loading and the "Source" line under a slide both go through SRC.
    SRC = {
        "tail_sweep": EOS / "tail_out" / "results.tsv",  # scratch_eos/tail_sweep.py: one line per run
        "tail_runs": EOS / "tail_out",  # the run records and audio of that sweep
        "probe": FT / "cluster_logs" / "jobs" / "smoke_rows_8394616.log",  # eos_probe.py, released and fine-tuned model
        "smoke40_config": FT / "runs" / "smoke40" / "config.json",
        "smoke40_data": FT / "smoke40" / "run.json",
        "cont": OUT / "cont_summary.tsv",  # prefix_continuation_check.sh, job 8371735
        "pfx_p4_none": OUT / "pfxcheck_p4_repeat_tnone",
        "pfx_p4_03": OUT / "pfxcheck_p4_repeat_t0.3",
        "pfx_p1_none": OUT / "pfxcheck_p1_oov150_tnone",
        "pfx_p1_03": OUT / "pfxcheck_p1_oov150_t0.3",
        "pfx_p3_none": OUT / "pfxcheck_p3_prefix_tom_tnone",
        "pfx_p3_03": OUT / "pfxcheck_p3_prefix_tom_t0.3",
        "raw_runs": OUT / "rows_test" / "baseline",  # released model, request clip as made, partner Gweneth
        "gen_released": OUT / "rows_test_run100" / "baseline",
        "gen_tuned": OUT / "rows_test_run100" / "tuned",
        "gen_tuned_la": OUT / "rows_test_run100_la" / "tuned",
        "run100_rows": FT / "run100" / "validation.parquet",  # the 18 rows whose requests are the prefixes of the run100 evals
    }

    def src(*names):
        """The 'Source' line of a slide: the files (relative to the thesis folder) its numbers were read from."""
        return mo.md("<small>Source: " + " · ".join(f"`{SRC[n].relative_to(THESIS)}`" for n in names) + "</small>")

    def records(folder):
        """All run records (one json per generated dialogue) of a folder -> DataFrame."""
        rows = []
        for f in sorted(folder.glob("*.json")):
            j = json.loads(f.read_text())
            if "generation" not in j:
                continue
            g, pre = j["generation"], j.get("prefix") or {}
            rows.append(
                {
                    "run": f.stem,
                    "word": (j.get("meta") or {}).get("word", f.stem),
                    "ended_by": g["ended_by"],
                    "frames": g["n_generated_frames"],
                    "request": pre.get("text"),
                    "prefix_s": pre.get("seconds"),
                    "tail_s": pre.get("tail_s"),
                    "path": f,
                }
            )
        return pd.DataFrame(rows)

    # ---- tail sweep (released model)
    sweep = pd.read_csv(SRC["tail_sweep"], sep="\t", na_values=["None"], dtype={"tail": str}).rename(columns={"ended": "ended_by"})
    sweep_first = sweep[sweep["frames"] == 1]  # stage 1: one generated frame per run, P(EOS) at that frame

    # ---- EOS probe (released model and one fine-tune), parsed from the job log
    def parse_probe(path):
        rows, model, cols = [], None, None
        for line in path.read_text(errors="replace").splitlines():
            m = re.match(r"== probe: (\w+)", line)
            if m:
                model, cols = m[1], None
                continue
            if line.startswith("word ") and "raw:0" in line:
                cols = line.split()[1:]
                continue
            parts = line.split()
            if cols and len(parts) == len(cols) + 1 and parts[0] != "mean":
                try:
                    vals = [float(x) for x in parts[1:]]
                except ValueError:
                    cols = None
                    continue
                rows += [{"model": model, "word": parts[0], "condition": c, "p_eos": v} for c, v in zip(cols, vals)]
            elif cols and (not parts or parts[0] == "mean"):
                cols = None
        return pd.DataFrame(rows)

    probe = parse_probe(SRC["probe"])

    # ---- prefix checks and the continuation check (released model)
    pfx = pd.concat(
        [records(SRC[k]).assign(request_set=k.split("_")[1], tail="none" if k.endswith("none") else "0.3 s") for k in SRC if k.startswith("pfx_")],
        ignore_index=True,
    )
    cont = pd.read_csv(SRC["cont"], sep="\t")

    # ---- free generation with a fine-tune (run100)
    gen = {
        "released model, tail 0.3 s": records(SRC["gen_released"]),
        "fine-tuned, tail 0.3 s": records(SRC["gen_tuned"]),
        "fine-tuned, tail 0.3 s + look-ahead 1.5 s": records(SRC["gen_tuned_la"]),
    }

    def reply_frames():
        """Per training-format row: frames from the end of the request's last word to the end of the row."""
        out = []
        for r in pq.read_table(SRC["run100_rows"]).to_pylist():
            req = r["utterances"][0]["words"][-1]
            out.append(len(r["dsu_c1_path"][0]) - int((req["start"] + req["dur"]) * FPS))
        return out

    def run_info(name):
        cfg = json.loads((FT / "runs" / name / "config.json").read_text())
        data = json.loads((FT / name / "run.json").read_text())
        return {"rows": data["n_train"], "steps": cfg["max_steps"], "lr": cfg["learning_rate"]}

    return SRC, cont, gen, pfx, probe, reply_frames, run_info, src, sweep_first


@app.cell
def _(BLUES, C1, C2, GRID, INK, INK2, SURFACE, np, plt, sf):
    def _style(ax, grid_axis="x"):
        ax.set_facecolor(SURFACE)
        ax.tick_params(colors=INK2, labelsize=8.5)
        ax.grid(axis=grid_axis, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_visible(False)

    def _envelope(path, bins=1100):
        a, sr = sf.read(path)
        a = a.mean(axis=1) if a.ndim > 1 else a
        n = max(1, len(a) // bins)
        env = np.abs(a[: len(a) // n * n]).reshape(-1, n).max(axis=1)
        return np.arange(len(env)) * n / sr, env, len(a) / sr

    def fig_two_runs(runs, names):
        """runs: [(title, c1 wav, c2 wav, prefix seconds, end label)]. Two channels per run on one time axis."""
        envs = [(_envelope(c1), _envelope(c2)) for _, c1, c2, _, _ in runs]
        x_max = max(e[0][2] for e in envs) * 1.02
        peak = max(max(e[0][1].max(), e[1][1].max()) for e in envs)
        fig, axes = plt.subplots(len(runs), 1, figsize=(11, 2.3 * len(runs)), sharex=True, facecolor=SURFACE)
        for ax, (title, _, _, prefix_s, end_label), ((t1, e1, dur), (t2, e2, _)) in zip(axes, runs, envs):
            _style(ax)
            for t, e, lane, col in ((t1, e1, 1.0, C1), (t2, e2, 0.0, C2)):
                e = 0.42 * e / peak
                ax.fill_between(t, lane - e, lane + e, color=col, lw=0)
            ax.axvspan(0, prefix_s, color=GRID, alpha=0.45, lw=0)
            ax.text(prefix_s / 2, 1.62, "prefix (given)", ha="center", va="center", fontsize=8.5, color=INK2)
            ax.axvline(dur, color=INK, lw=1.2)
            ax.text(dur + x_max * 0.008, 0.5, end_label, ha="left", va="center", fontsize=8.5, color=INK)
            ax.set_ylim(-0.6, 1.85)
            ax.set_yticks([1.0, 0.0], [f"ch 1  {names[0]}", f"ch 2  {names[1]}"], fontsize=9, color=INK)
            ax.tick_params(axis="y", length=0)
            ax.set_xlim(0, x_max)
            ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=5)
        axes[-1].set_xlabel("seconds", fontsize=9, color=INK2)
        fig.tight_layout(h_pad=1.4)
        return fig

    def fig_heatmap(table, x_label):
        """table: DataFrame of probabilities in [0, 1], rows = words, columns = conditions."""
        fig, ax = plt.subplots(figsize=(1.25 * table.shape[1] + 2.2, 0.5 * table.shape[0] + 1.1), facecolor=SURFACE)
        ax.set_facecolor(SURFACE)
        ax.imshow(table.to_numpy(), cmap=BLUES, vmin=0, vmax=1, aspect="auto")
        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                v = table.iat[i, j]
                ax.text(j, i, f"{v:.2f}" if v >= 0.005 else "0", ha="center", va="center", fontsize=9.5, color="#ffffff" if v > 0.55 else INK)
        ax.set_xticks(range(table.shape[1]), table.columns, fontsize=9, color=INK)
        ax.set_yticks(range(table.shape[0]), table.index, fontsize=9, color=INK)
        ax.set_xticks(np.arange(-0.5, table.shape[1]), minor=True)
        ax.set_yticks(np.arange(-0.5, table.shape[0]), minor=True)
        ax.grid(which="minor", color=SURFACE, lw=2)
        ax.tick_params(which="both", length=0)
        ax.xaxis.set_label_position("top")
        ax.xaxis.tick_top()
        ax.set_xlabel(x_label, fontsize=9, color=INK2, labelpad=8)
        for sp in ax.spines.values():
            sp.set_visible(False)
        fig.tight_layout()
        return fig

    def fig_frames(groups, x_max, band=None, band_label=None, label_eos=False):
        """groups: {row label: DataFrame with 'frames', 'ended_by', 'word'}. One dot per run: how many frames were generated."""
        labels = list(groups)
        fig, ax = plt.subplots(figsize=(10.5, 0.75 * len(labels) + 1.5), facecolor=SURFACE)
        _style(ax)
        if band is not None:
            ax.axvspan(band[0], band[1], color=C2, alpha=0.13, lw=0)
            ax.text((band[0] + band[1]) / 2, len(labels) - 0.42, band_label, ha="center", va="bottom", fontsize=8.5, color=INK2)
        for i, lab in enumerate(labels):
            y0 = len(labels) - 1 - i
            for kind, col, mk in (("max_length", C1, "o"), ("eos", C2, "D")):
                d = groups[lab][groups[lab]["ended_by"] == kind].sort_values("frames")
                if d.empty:
                    continue
                jit = (np.arange(len(d)) % 5 - 2) * 0.09
                ax.scatter(d["frames"], y0 + jit, s=42, color=col, marker=mk, edgecolor=SURFACE, linewidth=1.2, zorder=3)
                if kind == "max_length":
                    ax.text(d["frames"].max() - x_max * 0.02, y0, f"{len(d)} run" + ("s" if len(d) != 1 else ""), ha="right", va="center", fontsize=8.5, color=INK2)
                if label_eos and kind == "eos":
                    for k, ((_, r), dy) in enumerate(zip(d.iterrows(), jit)):
                        ax.text(r["frames"], y0 + dy + (0.17 if k % 2 == 0 else -0.2), r["word"], ha="center", va="bottom" if k % 2 == 0 else "top", fontsize=7.5, color=INK2)
        ax.scatter([], [], s=42, color=C1, marker="o", label="ran to max_length")
        ax.scatter([], [], s=42, color=C2, marker="D", label="ended by EOS")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32 / (0.75 * len(labels) + 1.5) * 3), ncol=2, fontsize=8.5, frameon=False, labelcolor=INK2)
        ax.set_yticks(range(len(labels)), labels[::-1], fontsize=9, color=INK)
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(-0.6, len(labels) - 0.1)
        ax.set_xlim(-x_max * 0.02, x_max * 1.03)
        ax.set_xlabel("frames generated after the prefix (80 ms each)", fontsize=9, color=INK2)
        fig.tight_layout()
        return fig

    def fig_probe(df, conditions, labels):
        """One dot per word, released and fine-tuned side by side per prefix condition; the bar is the mean."""
        fig, ax = plt.subplots(figsize=(10.5, 3.8), facecolor=SURFACE)
        _style(ax, grid_axis="y")
        for model, col, dx in (("baseline", C1, -0.17), ("tuned", C2, 0.17)):
            for i, c in enumerate(conditions):
                v = df[(df["model"] == model) & (df["condition"] == c)]["p_eos"].to_numpy()
                jit = np.linspace(-0.09, 0.09, len(v))
                ax.scatter(i + dx + jit, v, s=30, color=col, alpha=0.85, edgecolor=SURFACE, linewidth=0.8, zorder=3)
                ax.plot([i + dx - 0.13, i + dx + 0.13], [v.mean(), v.mean()], color=col, lw=2.5, zorder=4)
                ax.text(i + dx + (0.16 if dx > 0 else -0.16), v.mean(), f"{v.mean():.2f}", ha="left" if dx > 0 else "right", va="center", fontsize=9, color=INK)
        ax.scatter([], [], s=30, color=C1, label="released model")
        ax.scatter([], [], s=30, color=C2, label="fine-tuned model")
        ax.legend(loc="upper right", fontsize=8.5, frameon=False, labelcolor=INK2)
        ax.set_xticks(range(len(conditions)), labels, fontsize=9, color=INK)
        ax.tick_params(axis="x", length=0)
        ax.set_ylim(-0.04, 1.16)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_ylabel("chance of an immediate stop", fontsize=9, color=INK2)
        fig.tight_layout()
        return fig

    return fig_frames, fig_heatmap, fig_probe, fig_two_runs


@app.cell
def _(pd):
    def md_table(df):
        """DataFrame -> markdown table (no tabulate dependency)."""
        cols = [str(c) for c in df.columns]
        lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
        for _, r in df.iterrows():
            lines.append("| " + " | ".join("" if pd.isna(x) else str(x).replace("|", "\\|") for x in r.tolist()) + " |")
        return "\n".join(lines)

    return (md_table,)


@app.cell
def _(SRC, fig_two_runs, json, md_table, mo, pd, src):
    # ===== slide 1 =====
    # the first word (alphabetical) whose run with the clip as made ended within 2 frames
    _word = next(f.stem for f in sorted(SRC["raw_runs"].glob("*.json")) if json.loads(f.read_text())["generation"]["n_generated_frames"] <= 2)
    _runs, _rows = [], []
    for _tail, _dir in (("raw", SRC["raw_runs"]), ("0.3", SRC["gen_released"])):
        _j = json.loads((_dir / f"{_word}.json").read_text())
        _g, _p = _j["generation"], _j["prefix"]
        _label = f"ends after {_g['n_generated_frames']} frame" if _g["n_generated_frames"] == 1 else f"{_g['n_generated_frames']} frames generated"
        _title = "Clip as the TTS made it" if _tail == "raw" else f"Same clip, trailing silence replaced by {_tail} s"
        _runs.append((_title, _dir / _j["audio"]["c1"], _dir / _j["audio"]["c2"], _p["seconds"], _label))
        _rows.append(
            {
                "prefix": _title,
                "request (played as Tom)": _p["text"],
                "prefix length": f"{_p['seconds']} s",
                "run ended by": _g["ended_by"],
                "frames generated": _g["n_generated_frames"],
                "audio": str(_dir / _j["audio"]["wav"]),
            }
        )
    _names = (_j["generation"]["speaker1"]["name"], _j["generation"]["speaker2"]["name"])
    fig_symptom = fig_two_runs(_runs, _names)
    mo.vstack(
        [
            mo.md("# Free generation ends right after the request"),
            mo.md(
                f"""
    - Setup: the request clip is given as a prefix, then the released model generates both speakers.
    - Example word: **{_word}**. The two runs differ only in the silence at the end of the prefix.
    """
            ),
            fig_symptom,
            mo.md(md_table(pd.DataFrame(_rows).drop(columns="audio"))),
            mo.hstack(
                [mo.vstack([mo.md(r["prefix"]), mo.audio(src=r["audio"])]) for r in _rows],
                justify="start",
                gap=3,
            ),
            src("raw_runs", "gen_released"),
        ]
    )
    return


@app.cell
def _(fig_heatmap, mo, src, sweep_first):
    # ===== slide 2 =====
    _order = ["raw", "0.1", "0.2", "0.3", "0.5"]
    _table = sweep_first.pivot(index="word", columns="tail", values="p_eos_first")[_order]
    _table = _table.loc[sweep_first["word"].drop_duplicates()]  # keep the sweep's word order
    _table.columns = ["as made"] + [f"{t} s" for t in _order[1:]]
    fig_tail = fig_heatmap(_table, "silence after the last word of the prefix")
    mo.vstack(
        [
            mo.md("# Silence at the end of the prefix and P(EOS)"),
            mo.md(
                f"""
    - Released model, {_table.shape[0]} words, one request clip each. The clip's trailing silence is trimmed and replaced by a fixed length.
    - EOS is the model's stop token. P(EOS) here is the estimated chance generation stops on the first new audio frame; higher values mean an immediate stop is more likely.
    """
            ),
            fig_tail,
            src("tail_sweep"),
        ]
    )
    return


@app.cell
def _(fig_probe, md_table, mo, probe, run_info, src):
    # ===== slide 3 =====
    _conds = ["raw:0", "raw:1.5", "0.3:0", "0.3:1.5"]
    _labels = ["clip as made", "clip as made\n+ look-ahead 1.5 s", "tail 0.3 s", "tail 0.3 s\n+ look-ahead 1.5 s"]
    fig_probe_plot = fig_probe(probe, _conds, _labels)
    _means = probe.pivot_table(index="model", columns="condition", values="p_eos", aggfunc="mean")[_conds].round(3)
    _means.columns = [x.replace("\n", " ") for x in _labels]
    _means.insert(0, "mean over words", ["released model", "fine-tuned model"])
    _ft = run_info("smoke40")
    mo.vstack(
        [
            mo.md("# Fine-tuned model: tail and codec look-ahead"),
            mo.md(
                f"""
    - Fine-tuned model: smoke40 ({_ft['rows']} repeat-task rows, {_ft['steps']} steps). {probe['word'].nunique()} held-out words.
    - **P(EOS):** the chance of the model signalling its stop token on the first new audio frame. Here it is measured after temperature, top-k and top-p are applied.
    - **Tail 0.3 s:** replace the request's trailing silence with 0.3 s. **Look-ahead 1.5 s:** encode 1.5 s of extra silence after the request, then discard those extra codes. The model receives only the original prefix length, with its final codec codes formed as if the audio continued.
    """
            ),
            fig_probe_plot,
            mo.md(md_table(_means)),
            src("probe", "smoke40_config", "smoke40_data"),
        ]
    )
    return


@app.cell
def _(
    cont,
    fig_frames,
    gen,
    md_table,
    mo,
    np,
    pd,
    pfx,
    reply_frames,
    run_info,
    src,
):
    # ===== slide 4 =====
    _imm = pfx.assign(immediate=(pfx["ended_by"] == "eos") & (pfx["frames"] <= 2))
    _req = _imm.groupby("request_set")["request"].first()
    _pfx_table = _imm.pivot_table(index="request_set", columns="tail", values="immediate", aggfunc=lambda s: f"{int(s.sum())} of {len(s)}")[["none", "0.3 s"]]
    _pfx_table.columns = ["immediate stops, clip as made", "immediate stops, tail 0.3 s"]
    _pfx_table.insert(0, "request text (first word of the set)", [_req[i] for i in _pfx_table.index])

    _cont = pd.DataFrame(
        {
            "prefix set, tail 0.3 s": cont["variant"],
            "dialogues": cont["n"],
            "immediate stops": cont["immediate_stop"],
            "upper 95% bound": cont["imm_upper95"],
            "ended by EOS later": cont["eos_late"],
            "ran to max_length": cont["max_length"],
            "median frames": cont["frames_median"],
        }
    )
    _rf = reply_frames()
    _band = (float(np.percentile(_rf, 25)), float(np.percentile(_rf, 75)))
    fig_gen = fig_frames(gen, 256, band=_band, band_label=f"reply length in the training rows\n(middle half: {_band[0]:.0f} to {_band[1]:.0f} frames)")
    _ft = run_info("run100")
    mo.vstack(
        [
            mo.md("# Status with tail 0.3 s"),
            mo.md("**Released model.** Immediate stop = EOS within 2 generated frames."),
            mo.md(md_table(_pfx_table)),
            mo.md(md_table(_cont)),
            mo.md(f"**Fine-tuned model** (run100: {_ft['rows']} rows, {_ft['steps']} steps), 18 held-out words, one dialogue each."),
            fig_gen,
            mo.md(
                """
    - Chosen: tail 0.3 s for every prefix; look-ahead 1.5 s for fine-tuned models.
    - Open: fine-tuned runs end near the reply length of the training rows, where each row has its EOS.
    """
            ),
            src("pfx_p4_none", "pfx_p1_none", "pfx_p3_none", "cont", "gen_released", "gen_tuned", "gen_tuned_la", "run100_rows"),
        ]
    )
    return


if __name__ == "__main__":
    app.run()

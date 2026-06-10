"""Marimo notebook to explore the Behavior-SD vocabulary table.

Run with:
    uv run marimo edit scripts/analysis/explore_vocab.py
"""

import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import altair as alt
    import marimo as mo
    import pandas as pd

    return alt, mo, pd


@app.cell
def _(mo, pd):
    vocab = pd.read_csv(
        mo.notebook_dir() / "behavior_sd_vocab.tsv", sep="\t", keep_default_na=False
    )
    vocab["rank"] = range(1, len(vocab) + 1)
    return (vocab,)


@app.cell
def _(mo, vocab):
    mo.md(
        f"""
    # Behavior-SD vocabulary (F-Actor training data)

    **{len(vocab):,} unique words**,
    **{int(vocab.speech_count.sum()):,} spoken word tokens**
    (train split of `maikezu/f-actor-behavior-sd-nanocodec`).

    - `speech_count`: occurrences in spoken utterances and backchannels —
      what the audio heads were trained on.
    - `narrative_count`: occurrences in instruction narratives —
      text-side exposure only.
    """
    )
    return


@app.cell
def _(mo):
    search = mo.ui.text(
        placeholder="e.g. ^photo or quantum", label="Search (regex)", debounce=300
    )
    search
    return (search,)


@app.cell
def _(mo, search, vocab):
    filtered = (
        vocab[vocab.word.str.contains(search.value, regex=True, na=False)]
        if search.value
        else vocab
    )
    mo.ui.table(filtered, page_size=15)
    return


@app.cell
def _(alt, mo, vocab):
    spoken = vocab[vocab.speech_count > 0]
    zipf_sample = spoken.iloc[:: max(1, len(spoken) // 2000)]
    zipf = mo.ui.altair_chart(
        alt.Chart(zipf_sample)
        .mark_line(point=True)
        .encode(
            x=alt.X("rank:Q", scale=alt.Scale(type="log"), title="frequency rank"),
            y=alt.Y(
                "speech_count:Q", scale=alt.Scale(type="log"), title="speech count"
            ),
            tooltip=["word", "speech_count", "narrative_count", "rank"],
        )
        .properties(title="Zipf curve: spoken word frequencies", height=300)
    )
    zipf
    return


@app.cell
def _(mo, vocab):
    bands = {
        "frequent (>100)": (vocab.speech_count > 100).sum(),
        "medium (11-100)": vocab.speech_count.between(11, 100).sum(),
        "rare (1-10)": vocab.speech_count.between(1, 10).sum(),
        "narrative-only (0 speech)": (
            (vocab.speech_count == 0) & (vocab.narrative_count > 0)
        ).sum(),
    }
    mo.md(
        "## Frequency bands\n\n"
        + "\n".join(f"- **{k}**: {v:,} words" for k, v in bands.items())
        + "\n\nWords not in the table at all are fully OOV: never spoken, "
        "never in a narrative."
    )
    return


if __name__ == "__main__":
    app.run()

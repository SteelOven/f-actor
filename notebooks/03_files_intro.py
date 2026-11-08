import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import json
    import wave
    from pathlib import Path

    import pandas as pd

    REPO = Path(__file__).resolve().parent.parent
    return Path, REPO, json, mo, pd, wave


@app.cell
def _(mo):
    mo.md(r"""
    # 3. Working with files: list -> pick -> load -> look

    Part 1 lists files (pathlib + pandas). Part 2 loads one, dispatching on the file format.
    Change the widgets and everything downstream re-runs.
    """)
    return


@app.cell
def _(mo):
    folder = mo.ui.dropdown(
        options=["scratch", "eval", "example_dialogues", "confs", "hf_models/f_actor"],
        value="scratch",
        label="folder",
    )
    pattern = mo.ui.text(value="*", label="glob (e.g. *.json, **/*.wav)")
    mo.hstack([folder, pattern], justify="start")
    return folder, pattern


@app.cell
def _(REPO, folder, pattern, pd):
    # Part 1: list files into a DataFrame (one row per file).
    root = REPO / folder.value
    files = pd.DataFrame(
        [
            {
                "path": str(p.relative_to(root)),
                "suffix": p.suffix or "(none)",
                "size_kb": round(p.stat().st_size / 1024, 1),
            }
            for p in sorted(root.glob(pattern.value))
            if p.is_file()
        ]
    )
    files
    return files, root


@app.cell
def _(files, mo):
    # Table that lets you click a row to select a file.
    table = mo.ui.table(files, selection="single", page_size=8)
    table
    return (table,)


@app.cell
def _(table):
    table.value["path"]
    return


@app.cell
def _(files):
    # Quick first chart-able fact: what formats and how big?
    files.groupby("suffix")["size_kb"].agg(["count", "sum"]).sort_values("sum", ascending=False)
    return


@app.function
def read_tail(path, n_lines=40):
    from collections import deque
    with open(path, errors="replace") as f:
        return "".join(deque(f, maxlen=n_lines))   # streams, keeps only the last n


@app.cell
def _(Path, json, mo, pd, ssh, wave):
    def load_any(path: Path):
        """Load a file according to its extension and return something displayable.

        The Part 2 exercise: choose how each format should be loaded.
        """
        suffix = path.suffix.lower()
    
        # TODO(human): implement the branches for ".json", ".csv" and ".wav".
        # Everything else falls through to the raw-text preview below.
        if suffix == ".json":
            return json.loads(path.read_text())
        elif suffix == ".csv":
            return pd.read_csv(path)
        elif suffix == ".wav":
            with wave.open(str(path), "r") as wav:
                return wav.getnframes()/wav.getframerate()
        if suffix in {".log", ".txt", ".md", ".yaml", ".sh"}:
            return mo.plain_text(ssh(path))   # monospace, no markdown mangling
        # unknown / binary:
        return {"name": path.name, "size_kb": round(path.stat().st_size / 1024, 1), "suffix": suffix}

    return (load_any,)


@app.cell
def _(load_any, mo, root, table):
    mo.stop(not table.value.shape[0], mo.md("*Select a row in the table above.*"))
    chosen = root / table.value.iloc[0]["path"]
    mo.vstack([mo.md(f"**{chosen.name}**"), load_any(chosen)])
    return (chosen,)


@app.cell
def _(chosen):
    lines = chosen.read_text(errors="replace").splitlines()
    len(lines), max(map(len,lines))
    return (lines,)


@app.cell
def _(lines):
    import random
    # 1. Join your lists into strings first
    head_text = "\n".join(lines[:15])
    tail_text = "\n".join(lines[-15:])

    randoms = "\n".join(random.sample(lines,15))

    # 2. Then build your clean multi-line f-string template
    template = (
        f"--- HEAD ---\n"
        f"{head_text}\n"
        f"\n--- TAIL ---\n"
        f"{tail_text}"
        f"\n---Ranom ---\n"
        f"{randoms}"
    
    )

    print(template)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

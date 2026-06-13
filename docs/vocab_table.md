# Building a training-vocabulary table

This documents how `scripts/analysis/build_behavior_sd_vocab.py` produced
`scripts/analysis/behavior_sd_vocab.tsv`, and — more importantly — the method,
so you can rebuild an equivalent table for **another model/dataset**.

## Goal

We want to know, for any word, **how often the model was trained to speak it**.
That single number is what lets the OOV experiment (see
[`oov_experiment.md`](oov_experiment.md)) sort target words into frequency bands
(`frequent`/`medium`/`rare`/`oov`). The table is just `word -> counts`.

The crucial distinction is *spoken* vs *read*:

- **speech_count** — words in the text the model was trained to **produce**
  (the TTS targets). This is the number that predicts whether it can say a word.
- **narrative_count** — words in the **instruction/conditioning** text. The
  model reads these but is never trained to pronounce them. Tracked separately
  because a word can be common in prompts yet never spoken (band:
  `narrative-only`).

Splitting the counts this way is the part most worth copying to another model:
**count the output/target field, not the whole record.**

## Method (and why)

### 1. Read columns straight from HuggingFace over HTTP — no download

The training set (`maikezu/f-actor-behavior-sd-nanocodec`) is many GB once the
audio/codec columns are included. We only need the text columns, so we never
download the dataset. Two pieces make that work:

- The HF datasets-server exposes the underlying parquet shards. We list them via
  the API:

  ```
  https://huggingface.co/api/datasets/<DATASET>/parquet/default/<split>
  ```

- **DuckDB with the `httpfs` extension** reads those parquet URLs directly and,
  because parquet is columnar, fetches **only the columns the query touches**
  (column projection). Selecting just `utterances`/`narrative` skips the heavy
  audio columns entirely — you pay for text, not for the codec tokens.

```python
con = duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs;")
rows = con.execute(query, {"urls": urls}).fetchall()
```

This is the key transferable trick: **any** parquet-backed HF dataset can be
word-counted this way without a local copy, as long as you know which columns
hold text.

### 2. Unnest the structured text, count per field

Behavior-SD stores a list of `utterances`, each with `tts_text` and a list of
`backchannels` (also with `tts_text`), plus a top-level `narrative`. The SQL
unnests those nested lists, then counts words in two pools:

- **speech** = `utterances[].tts_text` **UNION ALL** `backchannels[].tts_text`
- **narrative** = the `narrative` field

```sql
WITH utts AS (SELECT unnest(utterances) AS u FROM read_parquet($urls)),
speech_texts AS (
    SELECT u.tts_text AS t FROM utts
    UNION ALL
    SELECT b.tts_text FROM (SELECT unnest(u.backchannels) AS b FROM utts)
),
...
FULL OUTER JOIN narrative_counts n ON s.word = n.word
ORDER BY speech_count DESC, narrative_count DESC
```

A `FULL OUTER JOIN` keeps words that appear in only one pool (speech-only or
narrative-only), with `coalesce(..., 0)` filling the missing side.

### 3. Tokenize consistently

Words are extracted with one regex, lower-cased, applied identically wherever we
count or look up — so the table and any later lookup agree:

```python
WORD_PATTERN = "[a-z]+(?:'[a-z]+)?"   # letters, allowing one internal apostrophe
regexp_extract_all(lower(t), WORD_PATTERN)
```

Consequences to keep in mind: counting is case-folded; contractions like
`don't` stay whole; numbers and punctuation are dropped; hyphenated or
multi-word targets are counted as separate sub-words. Keep the *same* pattern in
the lookup path or "OOV" verdicts won't match the table.

## Running it

```bash
# build (default split=train, out=scripts/analysis/behavior_sd_vocab.tsv)
python scripts/analysis/build_behavior_sd_vocab.py [--split train] [--out PATH]

# look words up in an existing table (no rebuild)
python scripts/analysis/build_behavior_sd_vocab.py --lookup schadenfreude recipe
```

Output is a TSV sorted by frequency: `word \t speech_count \t narrative_count`.
It's committed, so rebuilding is only needed if the dataset or the counting
changes.

## Adapting to another model

The script is specific to Behavior-SD's schema, but the recipe generalizes.
To replicate for a different model:

1. **Find the training dataset's parquet URLs.** If it's on the Hub, use the
   `/api/datasets/<id>/parquet/default/<split>` endpoint. If it's local parquet,
   point `read_parquet` at the path/glob and drop the `httpfs` install.
2. **Identify which column holds the model's output/target text** (here:
   `utterances[].tts_text` + backchannels). This is the one that becomes
   `speech_count`. Decide what, if anything, is the conditioning/instruction
   text for a second count.
3. **Adjust the unnest/SELECT** to that schema. Flat text columns need no
   `unnest`; deeply nested ones need more. Inspect the schema first with
   `DESCRIBE SELECT * FROM read_parquet($urls)`.
4. **Reuse the same `WORD_PATTERN`** (or a deliberate variant) for both counting
   and lookup so they stay consistent.

Everything else — the band thresholds, the config generator, the inference
loop — is downstream of this table and works unchanged once `word -> counts`
exists for the new model.

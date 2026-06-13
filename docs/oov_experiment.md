# OOV experiment: how it works

**Question.** When we force F-Actor to talk about a word it rarely or never saw
during training, can it still *say* the word and use it sensibly? We expect a
gradient: frequent words come out clean, and out-of-vocabulary (OOV) words get
mangled, split (`schaden freude`), substituted, or quietly dropped.

The experiment ties each target word to its **training-speech frequency** so
results can be read off by frequency band instead of one word at a time.

## The pipeline

Three stages, each a separate script so you can re-run the cheap parts without
the expensive ones:

```
build_behavior_sd_vocab.py   ->  behavior_sd_vocab.tsv   (word -> training counts)
generate_oov_configs.py      ->  confs/oov/*.json        (one dialogue per word)
training/inference_example.py->  outputs/oov/*.{wav,json}(audio + transcript)
```

### 1. Vocabulary table — what the model actually heard

`scripts/analysis/build_behavior_sd_vocab.py` counts every word in the
Behavior-SD training set (`maikezu/f-actor-behavior-sd-nanocodec`) and writes
`scripts/analysis/behavior_sd_vocab.tsv` with two counts per word:

- **speech_count** — tokens in `utterances[].tts_text` and the backchannels,
  i.e. text the model was trained to *speak*. This is the number that matters
  for "can it say the word".
- **narrative_count** — tokens in the instruction-side `narrative` field. The
  model reads these as conditioning but is never trained to pronounce them, so a
  word can be narrative-only.

It reads only the text columns straight from the HuggingFace parquet files over
HTTP (DuckDB column projection), so there's no multi-GB download. The table is
checked in, so you normally don't need to rebuild it. See
[`vocab_table.md`](vocab_table.md) for how the table is built and how to
reproduce it for a different model.

Look a word up without regenerating anything:

```bash
python scripts/analysis/build_behavior_sd_vocab.py --lookup schadenfreude quokka recipe
```

### 2. Config generator — one dialogue per word

`scripts/analysis/generate_oov_configs.py` turns a list of target words into
dialogue configs. For each word it builds a two-speaker scene:

- **speaker1 (explainer)** is told to excitedly explain the word and *use it
  several times* — this is what forces the model to produce the token.
- **speaker2 (listener)** is curious and asks questions about it.

Each word is tagged with the band it falls into, from its `speech_count`:

| band            | rule                              | meaning                         |
| --------------- | --------------------------------- | ------------------------------- |
| `frequent`      | speech > 100                      | control; model says it easily   |
| `medium`        | 10 < speech ≤ 100                 | seen, not common                |
| `rare`          | 0 < speech ≤ 10                   | a handful of examples           |
| `narrative-only`| speech = 0, narrative > 0         | read but never spoken           |
| `oov`           | speech = 0, narrative = 0         | never seen at all               |

It writes the configs plus a `manifest.tsv` (word, counts, band, repeat, config
path, audio path) that is the index for analysis — every generated dialogue is
one row, annotated with the band, so you can group transcript results by band.

### 3. Inference — generate and transcribe

`training/inference_example.py --config <file>` loads F-Actor, generates the
dialogue audio, and writes a sibling transcript `.json` next to the `.wav`
(speaker1 text, speaker2 text, and the config used). Reading the transcript
tells you whether the target word survived; the audio tells you how it was
pronounced.

## Running the experiment

Generate configs for the curated word list (spans every band) and run them:

```bash
# 1. (optional) refresh the vocab table — usually skip, it's committed
python scripts/analysis/build_behavior_sd_vocab.py

# 2. make one config per word, 2 repeats each (sampling is stochastic)
python scripts/analysis/generate_oov_configs.py \
    --words-file scripts/analysis/oov_words.txt --repeats 2

# 3. run every config
for f in confs/oov/*.json; do
    python training/inference_example.py --config "$f"
done
```

Add a few ad-hoc words without touching the file:

```bash
python scripts/analysis/generate_oov_configs.py --words antidisestablishmentarianism gnocchi
```

Useful knobs on the generator: `--repeats` (samples per word — OOV behaviour is
noisy, so >1 helps), `--speakers Tom,Brian` (must be valid speakers),
`--max-length`, `--backchannels`, `--interruptions`, and `--outdir` /
`--audio-dir` if you want to keep a run separate from a previous one.

## Reading the results

Open `confs/oov/manifest.tsv` and, for each row, the matching transcript in
`outputs/oov/<word>_r<n>.json`. For each target word ask:

1. **Did the word appear at all** in speaker1's transcript? (it was instructed
   to use it several times)
2. **Was it intact, split, substituted, or dropped?** e.g. the first OOV run
   produced `schaden freude` and then drifted off-topic.
3. **Does it track the band?** The hypothesis is fidelity degrades from
   `frequent` → `oov`. The manifest's band column is what you group by.

Because generation is sampled (`temperature 0.9`, `top_k 40`), run a few repeats
per word before drawing conclusions.

## Tips / gotchas

- Speaker names are validated against a fixed list in `inference_example.py`
  (`Rebeka, Gweneth, Brian, Tom`); the generator defaults to `Tom,Brian`.
- Counts are case-folded and apostrophe-aware (`WORD_PATTERN` in the builder);
  hyphenated or multi-word targets are counted per sub-word.
- The vocab table is sorted by frequency, so `head`/`tail` of the `.tsv` gives
  you the most common and the rarest seen words to sample new targets from.

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
count_vocab.py --write-contract   ->  behavior_sd_vocab.tsv   (word -> training counts; in oov-shared)
generate_oov_configs.py      ->  confs/oov/*.json        (one dialogue per word)
training/inference_example.py->  outputs/oov/*.{wav,json}(audio + transcript)
```

### 1. Vocabulary table — what the model actually heard

oov-shared's `wordlists/count_vocab.py --write-contract` counts every word in the
Behavior-SD training set (`maikezu/f-actor-behavior-sd-nanocodec`) and writes
`wordlists/behavior_sd_vocab.tsv` (also in oov-shared, since it defines the
bands for both models) with two counts per word:

- **speech_count** — tokens in `utterances[].tts_text` and the backchannels,
  i.e. text the model was trained to *speak*. This is the number that matters
  for "can it say the word".
- **narrative_count** — tokens in the instruction-side `narrative` field. The
  model reads these as conditioning but is never trained to pronounce them, so a
  word can be narrative-only.

It reads only the text columns straight from the HuggingFace parquet files over
HTTP (DuckDB column projection), so there's no multi-GB download. The table is
checked in, so you normally don't need to rebuild it. See
oov-shared's `docs/vocab_table.md` for how the table is built and how to
reproduce it for a different model.

Look a word up without regenerating anything (from the oov-shared checkout):

```bash
python3 wordlists/explain.py schadenfreude quokka recipe
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

### 3. Inference — generate (and, optionally, transcribe)

`training/inference_example.py --config <file>` loads F-Actor, generates the
dialogue audio, and writes one run record `.json` next to the `.wav` (`meta`
from the config, `config`, `audio` paths, and `generation` — speaker1/speaker2
text). That's everything needed to check whether the target word survived in
the model's own plan; the audio tells you how it was actually pronounced.

ASR + word-fidelity grading merge into that *same* file rather than a second
one, via `scripts/analysis/oov_common.enrich_run()`. Two ways to run it:

- **Separate step (default, good for small local runs)**: generate first,
  transcribe later with `scripts/analysis/analyze_oov.py` (whole manifest) or
  `scripts/analysis/transcribe_oov.py <wav>` (one file at a time).
- **One pass (useful when batching, e.g. on a cluster)**: pass `--transcribe`
  (and optionally `--asr-model`) to `inference_example.py` so Whisper runs
  immediately after generation, in the same process, before the wav ever
  leaves local disk.

Either path is idempotent per ASR model — re-running doesn't re-transcribe or
clobber a different model's results, it just fills in whatever's missing.

## Running the experiment

Generate configs for the curated word list (spans every band) and run them:

```bash
# 1. (optional) refresh the vocab table — usually skip, it's committed
#    (in oov-shared: python3 wordlists/count_vocab.py --write-contract)

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

Open `confs/oov/manifest.tsv` and, for each row, the matching run record in
`outputs/oov/<word>_r<n>.json` (run `analyze_oov.py` first if you haven't
transcribed yet — it fills in `asr`/`grading` for every row and also writes
the aggregate `outputs/oov/analysis.tsv`). For each target word ask:

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

## Prefix experiment (in progress, branch `prefix`)

**Question.** Once the model has *heard the correct pronunciation* of a rare word in its own channel, does it
reproduce the word correctly later in the dialogue? The first seconds of the dialogue are fixed: a TTS turn by Tom
(the explainer, who starts) that contains the word, fed to the model teacher-forced via `generate(...,
use_speaker_sample=N)`. If that helps, the failure is "doesn't know how the word sounds"; if not, it is the audio
head's production. The codec is ruled out separately by `scripts/analysis/codec_roundtrip.py`.

**Where things are.** Model-independent parts live in oov-shared (TTS, ASR, alignment, levels); everything that
knows F-Actor's token layout lives here.

| Step | File | Runs in | Status |
|---|---|---|---|
| TTS clips of the opener + ASR gate | oov-shared `checks/configs/p3_prefix_tom.yaml` (`checks/asr_roundtrip.py`) | cluster: cosy-venv / `$TOOLS_PYTHON` | submitted 2026-10-16 |
| Level of TTS vs Behavior-SD | oov-shared `checks/loudness.py`, result in `checks/LOG.md` | laptop | done: +5.6 dB |
| Word timings from the known text | oov-shared `checks/align.py`, stage `align` -> `<run dir>/alignment.jsonl` | laptop (`uv sync --group align`) or `$TOOLS_PYTHON` | done, tested on P1 |
| wav + timings -> `dsu`/`text` tensors | `training/prefix.py` (`build_prefix`) | factor-venv / `.venv` | done |
| Sanity checks without the model | `scripts/analysis/check_prefix.py` -> `outputs/prefix_check/<word>__<voice>/` | laptop | done, 4/4 P1 clips pass |
| Generation hook (config key `prefix`) | `training/inference_example.py`, `training/run_oov_batch.py` | cluster | open |
| Grading only after the prefix | `scripts/analysis/oov_common.py` (`enrich_run`) | `$TOOLS_PYTHON` | open, **required**: c1 contains the prefix |

**Data flow.**

```
oov-shared run dir (e.g. $WS/checks/p3_prefix_tom, copied to oov-data/checks/)
  audio/<word>__tom.wav (CosyVoice, 24 kHz)   manifest.jsonl (the text)   results.tsv (ASR gate)
      │  asr_roundtrip.py align <config>
      ▼
  alignment.jsonl  (words with start/dur, target_idx)
      │  training/prefix.py build_prefix(wav, words, codec, tokenizer, gain_db=5.6)
      ▼
  Prefix: dsu [1, 8, N], text [1, 2, N], N = 2 + T, seconds = T * 80 ms
      ├─ scripts/analysis/check_prefix.py -> prefix.wav, frames.tsv, prefix.pt, summary.json
      └─ (open) generate_dialogue(..., use_speaker_sample=N, dsu_sample=dsu, text_sample=text)
```

**Check prefixes (laptop).**

```bash
python scripts/analysis/check_prefix.py --run-dir ../oov-data/checks/p1_oov150 --voice tom \
    --words adherer appalachia --out outputs/prefix_check
column -t -s$'\t' outputs/prefix_check/adherer__tom/frames.tsv | less    # text tokens vs the word spoken per frame
```

**Layout details the builder takes care of** (see `training/prefix.py`):
- frames 0-1 are `audio_delay_id` on all 8 heads; `generate` does not insert them once `N > 2`;
- speaker 2's channel is *encoded* silence, not pad ids; its text stream is all `[SILENCE_PAD]`;
- the text stream is made by the training code (`adapt_to_text_stream`), with `end_time` = the codec frames'
  duration, because `create_text_stream` derives frames-per-second from it;
- text is not delayed, audio is by 2 frames (`frames.tsv`: `t_text` vs `t_audio`), as in training;
- gain +5.6 dB with a peak guard at -0.5 dBFS (a few peaky clips get less, e.g. `appalachia` +0.8 dB).

**Gotchas.**
- The decoded output still starts with the prefix, and the existing grading transcribes the whole Tom channel:
  without the grading fix every prefixed run "contains" the word.
- `alignment.jsonl`'s `score` measures how well the spelling fits the sound, not correctness; gate on the ASR
  (`results.tsv`), not on it.

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
| TTS clips of the opener + ASR gate | oov-shared `checks/configs/p4_repeat.yaml` (P3: `p3_prefix_tom.yaml`) | cluster: cosy-venv / `$TOOLS_PYTHON` | P3 done; P4 submitted 2026-10-18 |
| Level of TTS vs Behavior-SD | oov-shared `checks/loudness.py`, result in `checks/LOG.md` | laptop | done: +5.6 dB |
| Word timings from the known text | oov-shared `checks/align.py`, stage `align` -> `<run dir>/alignment.jsonl` | cluster: `$WS/align-venv` (torchaudio 2.8); laptop: `uv sync --group align` | done, tested on P1 |
| wav + timings -> `dsu`/`text` tensors | `training/prefix.py` (`build_prefix`) | factor-venv / `.venv` | done |
| Sanity checks without the model | `scripts/analysis/check_prefix.py` -> `outputs/prefix_check/<word>__<voice>/` | laptop (optional; the smoke test covers it) | done: all 157 P1 Tom clips pass (2026-10-16) |
| Gate: words with a usable clip | oov-shared `checks/prefix_gate.py` -> `wordlists/lists/prefix_gate_p4.csv` | cluster: any `python3` (commit on the login node) | P3 list done; P4 waits for the run |
| Configs with a `prefix` block | `scripts/analysis/generate_oov_configs.py --prefix-run-dir ...` | factor-venv | done |
| Generation hook | `training/inference_example.py` (`config_prefix`, `split_prefix_audio`) | cluster | done; P3 smoke test ran (3 words, 2026-10-18) |
| Grading only after the prefix, both speakers | `scripts/analysis/oov_common.py` (`enrich_run`), `analyze_oov.py` | `$TOOLS_PYTHON` | done, tested with a fake ASR |
| Paired experiments | oov-shared `experiments/factor_prefix_repeat.json`, `factor_repeat_baseline.json` | cluster | ready once the P4 gate list exists |

**Data flow.**

```
oov-shared run dir on the cluster ($WS/checks/p4_repeat)
  audio/<word>__tom.wav (CosyVoice, 24 kHz)   manifest.jsonl (the text)   results.tsv (ASR gate)
      │  asr_roundtrip.py align <config>
      ▼
  alignment.jsonl  (words with start/dur, target_idx)
      │  training/prefix.py build_prefix(wav, words, codec, tokenizer, gain_db=5.6)
      ▼
  Prefix: dsu [1, 8, N], text [1, 2, N], N = 2 + T, seconds = T * 80 ms
      ├─ scripts/analysis/check_prefix.py -> prefix.wav, frames.tsv, prefix.pt, summary.json
      └─ generate_dialogue(..., use_speaker_sample=N, dsu_sample=dsu, text_sample=text)
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
- gain +5.6 dB with a peak guard at -0.5 dBFS. The guard lowers the gain on about half the P1 clips (isolated
  peaks, ~0.003% of samples), e.g. `quokka` +2.0 dB; the 157 prefixes still average -17.2 dBFS active speech vs
  Behavior-SD's -16.1, inside F-Actor's own output range. A soft limiter would recover the full gain if needed.

**Variants and findings.**
- P3 opener, Tom: "Hey Brian, I want to tell you about X. Have you ever heard of X?" Smoke test (3 words,
  2026-10-18): prefix fine (`prefix_label` intact), but Tom continues with "it"/"they" and never says the word
  again, in text or audio; without a prefix he at least attempts it. Measures reuse, not production: shelved
  (`experiments/factor_prefix_tom.json`, `factor_gated_baseline.json`).
- **P4 repeat (current):** Tom says "Please listen carefully and repeat the word X." Main measure is Brian's
  channel (`audio2_label`, `mono2_label`): does the partner produce the word after hearing it?
  Experiments `factor_prefix_repeat.json` / `factor_repeat_baseline.json`.

**Running it (P4), cluster only.** The laptop is needed only for listening. Steps, with what to check after each:

*A. Login node: update, submit the TTS run, wait.*

```bash
cd ~/oov-shared && git pull && cd ~/f-actor && git pull
cd ~/oov-shared && sbatch --time=01:30:00 checks/sbatch_roundtrip.sh checks/configs/p4_repeat.yaml
sacct -X --starttime=today --format=JobID,JobName%20,State,Elapsed | tail -5
export WS=$(ws_find personaplex)
wc -l $WS/checks/p4_repeat/{manifest.jsonl,transcripts.jsonl,results.tsv}
tail -25 ~/oov-shared/oov_roundtrip_<jobid>.log
```
Check: `COMPLETED`, 157 / 471 / 472 lines (fewer: timed out, resubmit the same command, it resumes), controls
`hello`/`water` intact, no `SUSPECT` lines.

*B. One interactive GPU session: align, gate, smoke test.* (`salloc` gives 30 min; if it runs out after the gate,
start a new one for B4.)

```bash
salloc -p dev_gpu_h100 --gres=gpu:1 -t 00:30:00
~/personaplex/cluster/cluster_shell.sh

# B1. one time only: the alignment venv (oov-shared checks/README.md, "Align")
python3 -m venv $WS/align-venv
$WS/align-venv/bin/pip install -q --upgrade pip uv
$WS/align-venv/bin/uv pip install --python $WS/align-venv/bin/python3 \
    --index-url https://download.pytorch.org/whl/cpu "torch==2.8.*" "torchaudio==2.8.*"
$WS/align-venv/bin/uv pip install --python $WS/align-venv/bin/python3 numpy soundfile pyyaml
$WS/align-venv/bin/python3 -c "import torch, torchaudio, soundfile, yaml; from torchaudio.functional import forced_align; print(torch.__version__, torchaudio.__version__, 'forced_align ok')"

# B2. align (~10 min on CPU; first run downloads the 1.2 GB MMS_FA weights)
cd /workspace/oov-shared
export TORCH_HOME=$WS/torch-hub
$WS/align-venv/bin/python3 checks/asr_roundtrip.py align checks/configs/p4_repeat.yaml 2>&1 | grep -v Warning | tail -3
python3 -c "import json,os; r=[json.loads(l) for l in open(os.path.expandvars('\$WS/checks/p4_repeat/alignment.jsonl'))]; print(len(r), 'rows; bad:', [x['word'] for x in r if len(x['target_idx'])!=1])"

# B3. gate, written into the oov-shared checkout the experiments read
python3 checks/prefix_gate.py $WS/checks/p4_repeat --voice tom \
    --asr parakeet-tdt:nvidia/parakeet-tdt-1.1b --out wordlists/lists/prefix_gate_p4.csv | head -15

# B4. smoke test, 5 gated words (old smoke outputs are deleted: finished configs would be skipped)
cd /workspace/factor-code
rm -rf confs/oov_prefix outputs/oov_prefix && mkdir -p outputs/oov_prefix
W=$(sed -n '2,6p' /workspace/oov-shared/wordlists/lists/prefix_gate_p4.csv | cut -d, -f1 | tr '\n' ' '); echo $W
$FACTOR_PYTHON scripts/analysis/generate_oov_configs.py --words $W \
    --prefix-run-dir '$WS/checks/p4_repeat' --outdir confs/oov_prefix --audio-dir outputs/oov_prefix
$FACTOR_PYTHON training/run_oov_batch.py --configs-glob 'confs/oov_prefix/*.json' 2>&1 | tee outputs/oov_prefix/batch.log
$TOOLS_PYTHON scripts/analysis/analyze_oov.py --manifest confs/oov_prefix/manifest.tsv \
    --outdir outputs/oov_prefix --asr-backend parakeet-tdt --report outputs/oov_prefix/analysis.tsv
grep -E "\([0-9]+s\)" outputs/oov_prefix/batch.log
for f in outputs/oov_prefix/*_pfx.json; do python3 -c "
import json,sys; r=json.load(open(sys.argv[1])); g=r['generation']; gr=list(r['grading'].values())[0]
print(sys.argv[1].split('/')[-1], g['ended_by'], 'prefix:', gr['prefix_label'], '| brian:', gr['audio2_label'], repr(gr['audio2_evidence']))
print('   brian says:', g['speaker2']['transcript'][:120])" $f; done
```
Check: B1 prints `2.8.0+cpu 2.8.0+cpu forced_align ok`; B2 gives 157 rows and an empty `bad` list; B3 "kept X of 157"
(about 100+ is good, under ~60 rethink before going on; dropped reasons should be misrecognitions, not lead-ins); B4:
`prefix: intact` everywhere, and **does Brian say or attempt the word** (`brian:` label and what he says) in most of the
five; note the seconds per dialogue for the experiments' `--time`.

*C. Login node: commit the gate list (the container has no git).*

```bash
cd ~/oov-shared && git add wordlists/lists/prefix_gate_p4.* && git commit -m "prefix gate: P4" && git push origin roundtrip
```

*D. Laptop: listen to the smoke test.*

```bash
rsync -av unicluster:/pfs/work9/workspace/scratch/ka_udtmx-personaplex/factor-outputs/oov_prefix/ \
    ~/Projects/GitHub_Repos/oov-data/f-actor/outputs/oov_prefix/
cd ~/Projects/GitHub_Repos/oov-data/f-actor/outputs/oov_prefix && mpv *_pfx.wav *_pfx_c2_cont.wav
```
Check: Tom's request flows into the dialogue (no click or gap), voices stay the same, Brian's attempt at the word is
audible near the start of `_c2_cont`. Go on only if Brian attempts the word in most of the five.

*E. Login node: the paired experiments.*

```bash
cd ~/oov-shared
~/personaplex/cluster/cluster_run.sh /workspace/oov-shared/run_experiment.sh experiments/factor_prefix_repeat.json prepare
~/personaplex/cluster/cluster_run.sh /workspace/oov-shared/run_experiment.sh experiments/factor_repeat_baseline.json prepare
# --time = kept words x 3 reps x seconds per dialogue (B4) + ~30 min
sbatch --time=05:00:00 sbatch_experiment.sh experiments/factor_prefix_repeat.json
sbatch --time=05:00:00 sbatch_experiment.sh experiments/factor_repeat_baseline.json
ls $WS/exps/factor_prefix_repeat/gen/factor/outputs/*.json | wc -l     # progress: towards kept x 3
```
Check: both `prepare` runs report the same word count; a timed-out job is resubmitted unchanged (finished dialogues
are skipped). Results: `$WS/exps/<name>/grade-parakeet-tdt/factor_analysis.tsv` per experiment, band table with
`s2 audio%` at the end of each job's log; pair the two by word and repeat.

**Telling prefixed runs from regular ones.**

| Level | Prefixed | Regular |
|---|---|---|
| directory | `$WS/exps/factor_prefix_tom/`, `outputs/oov_prefix/` | `$WS/exps/factor_gated_baseline/`, `outputs/oov/` |
| config / record | `"prefix"` block; `meta.prefix` = source run | neither |
| file names | `<word>_r<n>_pfx*.wav`, plus `_c1_prefix.wav`, `_c1_cont.wav`, `_c2_cont.wav` | `<word>_r<n>*.wav` |
| transcripts | `generation.*.transcript` = generated part, `prefix_transcript` = forced part | `transcript` only |
| grading entries | `graded_region: continuation`, `prefix_label`/`prefix_score` (control); speaker 2 in `*2_label` | `graded_region: full` (missing in old records = full); speaker 2 in `*2_label` |
| `manifest.tsv`, `analysis.tsv` | column `prefix` = source wav | column empty |

Every run also records `generation.n_generated_frames` and `generation.ended_by` (`eos` or `max_length`).

**Gotchas.**
- Grading a prefixed run with code older than this section would transcribe the prefix too; `enrich_run` refuses
  to reuse an entry whose `graded_region` does not match the run.
- `prefix_label` should be intact for every gated word; a run where it is not had its prefix damaged in decoding
  and is worth a listen before counting it.
- Special audio ids: codec codes are 0..4031, the model appends delay = 4032 and EOS = 4033
  (`modeling_dsu.py`: `audio_vocab_size = config.audio_vocab_size + 2`). Generation passes `model.audio_delay_id`
  to the builder. (Until 2026-10-18 the builder used 4030 as delay id, i.e. a real code: prefixed runs from before
  that fix, incl. the first 3-word smoke test, are invalid and must be regenerated.)
- `alignment.jsonl`'s `score` measures how well the spelling fits the sound, not correctness; gate on the ASR
  (`results.tsv`), not on it.

# Prefix experiment: run log

Chronological notebook, one entry per run or decision, newest last. Entry = question, what was run, result, what it changed.
Status table and how-to: [`oov_experiment.md`](oov_experiment.md), "Prefix experiment". Raw outputs are not in git: `oov-data/`
on the laptop (`checks/`, `exps/`, `scratch_eos/`), `$WS` on the cluster. The TTS side has its own log:
`oov-shared/checks/LOG.md`.

Tags: `prefix-tail-baseline` = `faeb917` (single-turn builder with `tail_s`). Single-turn tensors are pinned by
`scripts/analysis/prefix_golden.py` (`prefix_golden.json`, recorded on cpu).

## Plan (multi-turn prefix, branch `prefix-multiturn`)

Why: with the P4 request alone, the partner repeats neither frequent nor OOV words (below), so no OOV result is
interpretable. Few-shot demonstrations in the prefix (Tom asks for a frequent word, partner says it, twice or thrice, then the
target) test whether repetition can be elicited in context; if not, the same builder generates fine-tuning data.

Decisions: speakers Tom + Gweneth (Brian's reference gives lead-ins); demo words frequent only and never the target; reply
text "Okay, {word}."; gap 0.5 s (Behavior-SD median); tail 0.3 s; gain +5.6 dB Tom, Gweneth to be measured.

- [ ] 1. cluster: oov-shared `p5_demo` (Tom requests + Gweneth replies, ~12 frequent words), ASR gate, align, Gweneth loudness
- [ ] 2. multi-turn prefix in code:
  - [x] `training/prefix.py` `build_multiturn_prefix` (stub-codec selftest `scripts/analysis/prefix_selftest.py`; golden check still PASS)
  - [x] `config_prefix` (`prefix.turns`, `gap_s`, `tail_s`) and `split_prefix_channels` (`_c2_prefix` written when there are turns);
    selftest covers the config path; 2-frame smoke run with the real model OK (`oov-data/scratch_eos/multiturn_smoke/`)
  - [ ] `check_prefix` multi-turn mode
  - [x] generator flags (`generate_oov_configs.py --prefix-demos N --demo-request-run-dir --demo-reply-run-dir --demo-pool-file --demo-exclude --demo-seed --demo-reply-gain-db --prefix-gap-s`); without `--prefix-demos` the configs are byte-identical to before (checked on the 6 tail-sweep configs)
  - [ ] grading of a non-silent channel 2 / `prefix_label` on the last request only
- [ ] 3. golden check still PASS; `check_prefix` on a few multi-turn prefixes, listen
- [ ] 4. local pilot: 4 frequent targets x {0,1,3} demos x 2 seeds, 150 frames, tail 0.3; Brian-channel ASR + `onset_s2`
- [ ] 5. gate: repetition clearly above the 0-demo rate -> OOV paired run on the cluster; else 5 demos / other reply wording once, else fine-tune
- [ ] housekeeping: `oov-shared/checks/configs/p4_controls.yaml` says trailing silence was ruled out as the cause of the immediate stop; the tail sweep below shows it is the cause for the prefix tokens

## 2026-10-16/18: builder and first smoke tests
- Builder `training/prefix.py`, check script, +5.6 dB gain (CosyVoice is 5.6 dB below Behavior-SD, `oov-shared/checks/LOG.md`); all 157 P1 Tom
  clips pass `check_prefix.py`.
- Delay/EOS ids fixed to 4032/4033 (`69badb2`); runs before that are invalid.
- P3 opener ("...tell you about X. Have you ever heard of X?"), 3 words: prefix intact, Tom continues with "it"/"they" and never says the word again.
  Measures reuse, not production: shelved.
- P4 opener ("Please listen carefully and repeat the word X."), TTS gate keeps 112 of 157 words (Parakeet 1.1b).

## 2026-10-19..25: controls with frequent words (hello, water)
- `oov_ctrl_pfx` (default narratives, 6 dialogues): the partner said hello/water 0/6; `water` ended after 1 generated frame in 3/3 (`ended_by: eos`).
- `ctrl_repeat_back` (`oov-data/exps/ctr_prefix`, Brian told to say the word back): audio2 `dropped` 6/6; `water` stopped at frame 1 again 3/3;
  hello ran long but drifted to generic meeting talk ("project scope").
- No-prefix baselines also stop before `max_length` sometimes (`oov_ctrl_base`: frames 115 and 426 of 511).

## 2026-10-29: why generation stopped, tail sweeps (`oov-data/scratch_eos/`)
Question: why does generation end at the first frame after some prefixes?
- Mechanism (`modeling_dsu.py` generate): the loop breaks when any of the 2B rows x 8 audio heads samples the audio EOS id (or a text stream samples the pad / `<|EOS|>` id).
  First-frame P(any EOS) logged per run.
- Clips end 0.04-0.8 s after speech (median 0.18 s; 55% under 0.2 s). The decoded Behavior-SD dialogues end with only 0.08-0.20 s of silence in the mix (3 dialogues),
  so a short tail looks like the end of a training sample.
- Sweep 1 (water, silence appended to the raw clip, 3 seeds): 0 s P(EOS)=0.98 and 3/3 stopped at frame 1; +0.5 s and +1 s P=0.000, 0/6 stopped.
  Continuations: Tom drifts ("the meaning, it's a phrase"), the partner is silent or unrelated, nobody says water.
- Sweep 2 (water, appended 0.1/0.2/0.3/0.5): P = 0.47 / 0 / 0 / 0.
- Stage 1, normalized `tail_s` (trim trailing silence, append exactly `tail_s`), first-frame P(any EOS), 6 words:

  | tail | water | hello | archband | adherer | appalachia | anthropophagous |
  |---|---|---|---|---|---|---|
  | raw | 0.98 | 0.003 | 1.00 | 0.47 | 0.001 | 0.000 |
  | 0.1 | 0.88 | 0.998 | 0.25 | 0.02 | 0.09 | 0.08 |
  | 0.2 | 0.11 | 0.001 | 0.002 | 0.000 | 0.007 | 0.002 |
  | 0.3 | 0 | 0 | 0 | 0 | 0 | 0 |
  | 0.5 | 0 | 0 | 0 | 0 | 0 | 0 |

  Raw tail length does not predict it (hello 0.1 s has the same frame count as raw yet 0.998 vs 0.003): the content of the last frames matters; the tail is quantized to 80 ms codec frames.
- Stage 2 (0.3 and 0.5 s, 6 words, 2 seeds, 150 frames, 24 runs): 21 ran to `max_length`; 3 stopped late (hello 0.3 s1 frame 79, hello 0.5 s1 frame 88, archband 0.3 s0 frame 78).
  Tom is audible 0.64-1.28 s after the prefix in all 24 runs (no difference between 0.3 and 0.5 s). Partner: under 1 s in 2 runs, 2.2-10.6 s in 17, never in 5.
- No-prefix stop rate (239 runs, `outputs/oov`, max_length 512, from wav length): 59 (25%) ended early, median frame 429; 5 (2%) before frame 150, versus 3/24 with prefix+tail.
  The late stops show no shared cue (one mid-speech, one after ~1 s of silence).
- Decision: `tail_s = 0.3` for prefix runs (`faeb917`). Late stops are handled in generation, not in the builder: mask the stop for the experiment window and log per-step max P(EOS) (not done yet).
- Open: the partner does not repeat the word. That is what the multi-turn plan addresses.

Turn statistics from Behavior-SD validation (13.6k turn boundaries): gap between turns median 0.49 s (IQR 0.27-0.67), 12% overlap;
turn median 9 words / 3.1 s.

## 2026-10-31: fine-tune prototype, stage 1 (branch `finetune-repeat`, worktree `../f-actor-ft`)
Pilot result that led here: 0 / 24 dialogues with the partner saying the target (0, 1, 3 demos; 4 words x 2 repeats; `oov-data/scratch_eos/pilot`).
`finetune_repeat/compare_rows.py` builds real training rows with the repo's own functions (flags of `train_nanocodec_special_tokens.sh`)
and one synthetic row from the p5 clips. Findings:
- Row = `input_ids` (prompt padded to 2048), `attention_mask`, `dsu_ids` [8, T] int64 with 2 delay frames in front, `text_stream_ids` [1, T], `spk_emb` [192].
  The synthetic row has exactly this shape. Real rows are 527-983 frames (42-79 s), synthetic ~215 (17 s).
- **One text stream per row** (released config: `text_stream: true`, `multi_text_stream: false`, so `num_text_streams = 1`): the model never sees the
  partner's text, only the partner's audio. (Earlier I said both text streams are embedded: wrong for this checkpoint.) Consequences: "hearing" in the
  prefix experiments is audio-only, and the text-side stop in `generate()` (`num_text_streams > 1`) never fires; only audio EOS can end a run.
- Perspective: `prepare_dsu` picks the "system" speaker at random (50/50) per row; heads 0-3 = system, 4-7 = user; prompt, narrative (`new_narrative_s<k>`),
  speaker embedding and the text stream are the system speaker's. Loss: audio on all 8 heads (both speakers), text on the system's stream only
  (`calc_loss_on_c1_only: false`). A synthetic dialogue therefore wants rows from both perspectives (Gweneth = system carries the words in her text stream).
- Prompt: training (`build_prompt`) writes the narrative as `- <narrative>`, inference (`prepare_prompt`, used by all runs so far) writes it without `- `.
  Real narratives are second-person situation stories ("You felt helpful when ..."), backchannel/interruption counts come from the data (here 0/0).
- Real text streams contain `[BC]`/`[INTER]` tokens; the synthetic ones have none.

## 2026-10-31: fine-tune prototype, stage 2 (plain instruction sentences)
Quick version, plain instruction sentences: Tom "You ask Gweneth to repeat words after you, one word at a time.", Gweneth
"Tom will say a word. Listen to it and say the same word back to Tom." (the same two at test time).
- `finetune_repeat/rows.py` (dialogue -> two training rows, one per system perspective, prompt from the repo's `build_prompt`),
  `make_dataset.py` (p5 clips -> `train/`, `validation/`, optional `real_sample/` streamed from the HF dataset), `train.py` (wrapper that
  patches `finetune.load_data` to append the synthetic rows; `training/` is not edited), `check_row.py` (one-row test + overfit),
  `sbatch_overfit.sh`.
- Smoke data `oov-data/finetune/p5_small`: 6 + 2 dialogues (3-6 exchanges, 17-33 s) from the 9 p5 words without `--gate` (includes `weekend`,
  whose request clip failed the gate: use `--gate` for real data), 12 + 4 rows, 2 real rows.
- Forward pass through the repo's collator and model (laptop, CPU, released checkpoint, `model.train()`): finite. Loss before any training:
  synthetic row 4.89 (audio 3.87, text 1.01), real row 4.07 (audio 3.52, text 0.56). The synthetic text loss is about twice the real one.
- Tokenizer: `training/model.py load_model` adds 61 special tokens on top of the released tokenizer (128260 -> 128321; the release lacks
  `[INTER]` and the counting tokens). Ids used by our rows (`[SILENCE_PAD]` 128256 ... `[BC]` 128259) do not change; the embedding matrix grows.
- Cluster checkout for this: `git checkout finetune-repeat` in `~/f-actor` (the container mounts `~/f-actor` only), switch back to `prefix` afterwards.

## 2026-10-31: fine-tune prototype runs on the laptop (no cluster access)
Laptop: i7-1165G7 (4 cores), 15 GB RAM, no NVIDIA GPU, CPU-only torch 2.8. Full AdamW on the 1.37 B model needs ~22 GB: does not fit.
`check_row.py --train top:1` trains only the audio heads/embeddings, the speaker projection, the final norm and the last transformer layer
(193 M of 1369 M parameters). One synthetic row (412 frames), lr 5e-5, fp32, memory capped with `systemd-run --user --scope -p MemoryMax=9G`:
loss 4.89 -> 4.49 -> 4.09 over 3 steps (text 1.01 -> 0.58, audio 3.87 -> 3.52), 44-57 s per step, peak RSS 10.3 GB. So the data
path, collator, model, optimizer and loss work on real model code, and the loss falls on a fixed row. This does not tell whether the
behaviour is learned: one row, 3 steps. At ~50 s per row-step, 1000 row-steps are ~14 h.

## 2026-10-31: micro-run prepared (LoRA, laptop)
- `finetune_repeat/local_train.py`: LoRA (`--mode lora:R`, all layers, q/k/v/o/gate/up/down, peft without task type so the custom forward runs
  as is) or `top:N`; audio heads/embeddings frozen by default (`--heads train` to train them); one row per forward, `--accum` rows per step;
  evaluates synthetic-validation and real rows before/during/after; merges the adapters and saves `<out>/final` (+ tokenizer).
  Smoke (p5_small, 2 steps): 11.3 M trainable of 1380 M, ~25-30 s per row-step, peak RSS 10.1 GB under a 11 GB cgroup cap.
- Chain train -> merge -> save (5.4 GB) -> reload through `training/inference_example.py` (new `FACTOR_MODEL_DIR` env var) -> generate: works
  (1 training step, 2-frame generation, no size mismatches although the vocabulary grew to 128321).
- `inference_example.py`: `prompt_format: "training"` in a config writes the narrative as `- <narrative>` (training's `build_prompt`);
  default unchanged (the single-turn configs of the tail sweep regenerate byte-identical, golden check PASS).
  `generate_oov_configs.py`: `--prompt-format training`; `{word}` is no longer required in the explainer narrative when there is a prefix.
- Data: `oov-data/finetune/micro` = 20 train + 4 validation dialogues (3-5 exchanges, 16-28 s) from 5 words (doctor, garden, morning,
  movie, music; gate lists applied) + 4 real validation rows. Held out: water, kitchen, hello. Narratives: Tom "You ask Gweneth to repeat
  words after you, one word at a time.", Gweneth "Tom will say a word. Listen to it and say the same word back to Tom."
- Test: `make_eval_configs.sh <tag>` (groups train: doctor, music; heldout: water, kitchen, hello; oov: archband, adherer, appalachia,
  anthropophagous; Tom's request only, training prompt format, backchannels/interruptions 0, 2 repeats, 150 frames),
  `run_eval.sh <tag> [model dir]`, `summarize_eval.py`.
- Known weakness carried into training: Gweneth's TTS replies still have the pause after "okay" and the pre-sound (see the p5 clip analysis);
  the model will imitate them.

## 2026-10-31: fixing the Gweneth replies without the cluster
Cluster access lost, so TTS cannot be rerun. Clip-level fix instead: `training/prefix.py tidy_clip` (optional `"tidy"` per turn in `build_multiturn_prefix`;
clips without it are unchanged, golden PASS): cut the sound before the first word to 0.1 s, shorten word gaps to <= 0.2 s (cut the middle of the gap,
8 ms fades at each cut), 40 ms fade-out. `finetune_repeat/clip_report.py` (12 p5 replies, medians before -> after): lead 0.85 -> 0.10 s, largest word gap
0.53 -> 0.20 s, level of the last sound relative to speech -32.7 -> -34.1 dB (`coffee` 2.6 -> -7.4, still too loud: exclude it). `make_dataset.py` tidies the replies
by default (`--no-tidy-replies` to turn it off). Listening set `oov-data/scratch_eos/listen_3` (A raw, B lead cut only, C tidy). Not checked: that the cut at
first word - 0.1 s never clips the onset of "okay" (the aligner's start can be imprecise) - listen, or ASR the tidied clips when Parakeet is available.
TTS variants for when the cluster is back: oov-shared `p5_demo_reply_b/c/d` (no comma / bare word / seed 1).

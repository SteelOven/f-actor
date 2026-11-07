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

## 2026-11-06: baseline continuation check, released model, three request texts (job 8366726)

Question: does the released model continue after the prefix, and does it depend on the request text? `baseline_prefix_check.sh`
(6 words, 1 dialogue each, tail none / 0.3, request texts p4_repeat, p1_oov150, p3_prefix_tom). Raw outputs and per-dialogue
table: `oov-data/f-actor/outputs/pfxcheck_*`, `pfxcheck_job8366726/run.json`. (Job 8327977 died first: p1_oov150 had no
`alignment.jsonl` on the cluster; copied from the laptop after checking wavs and manifest are byte-identical.)
- Immediate stop = `ended_by: eos` after <= 2 generated frames. The log's summary table counts every `eos` together, so read the
  frame counts, not that table.
- No tail: p4_repeat 5/6 immediate stops, p1_oov150 1/6 (archband), p3_prefix_tom 0/6. Tail 0.3: 0/18 in all three.
- Late `eos` (394-505 frames) and `max_length` are continuations. 0/6 per cell is only an upper bound of ~40% on the stop rate.
- Not graded: whether Tom or the partner said the target word.
- Decision: request text p4_repeat (matches the fine-tune request) with `tail_s = 0.3`; never p4 without the tail.
- Next: `prefix_continuation_check.sh`, 112 gated OOV words (p4_repeat) + the 92 fine-tune words (p6_full_request_s0), 3 repeats,
  tail 0.3, one model load. Pass if the immediate-stop rate is < 2% (upper 95% bound reported). Repeat on the fine-tuned model.

## 2026-11-06: continuation check result (job 8371735, released model, tail 0.3)
- p4_repeat (112 gated OOV words) 336 dialogues: 0 immediate stops (upper 95% bound 1.1%), 114 late eos, 222 max_length, min 18 / median 511 frames.
- p6_full_request_s0 (92 fine-tune words) 276 dialogues: 0 immediate stops (upper 1.4%), 95 late eos, 181 max_length, min 23 / median 511 frames.
- Content after the prefix is gibberish; only the continuation was measured. Decision: request p6 + tail 0.3 is the prefix for the fine-tune.

## 2026-11-07: request -> reply fine-tune on `answer-finetune`: what was tried, what it showed
Setup: rows from unchanged clips (`training/train_rows.py`, `scripts/train/build_rows_dataset.py`): Tom's request (p6_full_request_s0_ok) then Gweneth's reply "Okay, {word}."
(p7_reply_short_s0_ok), reply speaker = system role (`--system_speaker 1`), loss on the reply stream only (`--calc_loss_on_c1_only`; eval loss scores the same half).
The 112 gated OOV words are excluded (held out). Pipeline: `scripts/train/smoke_rows.sh` (align build train eval probe), `sbatch_smoke_rows.sh`.
Earlier attempts: micro-run 1 (laptop, `finetune-repeat`, archived as tag `archive/finetune-repeat-tidy`): tuned stopped at frame 1 in 18/18; `p7_short_ok_v1`
(~76 train rows, 130 steps, loss on both streams, raw-prefix eval): tuned 9/9 stops, but the baseline stopped 6/9 under that raw prefix, so it showed nothing.

Results (all on the cluster; "immediate stop" = eos after <= 2 generated frames):
- smoke40 (40 train rows, 20 steps, LR 1e-5), 10 unseen words, tail 0.3: baseline 10/10 max_length; tuned 8/10 eos (3 within 3 frames), okay 0/10.
- run100 (70 train / 18 val, 80 steps, LR 1e-5): eval loss 0.6226 -> 0.5923 (epoch 1.2) -> 0.6179, so it barely trained. Tail 0.3 without look-ahead: baseline 0/18 immediate
  stops, tuned 5/18, okay 0/18. With look-ahead 1.5 s: tuned 1/18 immediate stops, okay 4/18, word 2/18 (baseline 2/18, 2/18).
- EOS probe (P(any audio head = EOS) at the first generated frame, mean over held-out words; job 8394616 and the run100 job):
  baseline raw 0.41-0.43, tail 0.3 0.000, tail 0.3 + look-ahead 1.5 0.000;
  tuned smoke40 raw 0.914 / 0.3 s 0.204 / 0.3 s + look-ahead 0.021 / raw + look-ahead 0.495;
  tuned run100 raw 0.965 / 0.3 s 0.235 / 0.3 s + look-ahead 0.074 / raw + look-ahead 0.671.
  The codec's end-of-file codes explain most of the tuned model's first-frame stops; the tail is still needed.
- Rows end with an audio EOS after the reply (training appends it), so the tuned model stops at about reply length (median ~39 frames in the run100 look-ahead eval): trained behaviour.
- overfit8 (8 train rows, LR 5e-5, 100 steps ~ 50 epochs, `finetune/overfit8`): train loss 3.56 -> 0.0126, eval loss on 4 unseen rows 0.826.
  Teacher-forced check (`scripts/analysis/teacher_forced_check.py`): 17/17 reply word tokens, 315/315 speaking frames, 287/287 silent frames right, 0 early EOS.
  Free generation on the same 8 words (tail 0.3 + look-ahead 1.5 unless noted), okay / word: default prompt 2/8, 1/8; training prompt ("- <narrative>") 1/8, 1/8;
  training prompt and tail 1.0 s 0/8, 0/8; training prompt and speaker 1 silenced after the prefix 3/8, 3/8 (n = 8, one sample each: noise-level differences).
- Ruled out: tokenizer (best_model vs hf_models/f_actor give identical ids; the "incorrect regex" warning is cosmetic), the "weights not used" warning (heads and embeds are
  loaded afterwards by `init_or_load_*`), the prompt dash, the prefix gap.

Reading:
- Training works mechanically (the reply is memorised under teacher forcing), so the gap is in generation. At generation the request speaker is also generated by the model and often
  talks after the request (tuned runs, `*_c1_cont.wav`: up to 8-10 s in several), while in the rows he is silent. `silence_speaker1` (opt-in) moved answers to 3/8: suggestive, not proven.
- The model picks "Okay, <a trained word>" rather than copying the requested word ("OKAY GIFTED" for represent). Copying an unseen word is not shown yet.
- Open: the remaining gap (teacher forcing 8/8 vs generation 3/8); generalisation to unseen words (job run170: 70 train / 18 unseen words x 3 samples, LR 5e-5, 170 steps,
  silence_speaker1, training prompt, tail 0.3 + look-ahead 1.5, baseline vs tuned); the gate should be immediate stops < 2% plus an answer rate (okay / word), not stops alone.
- Opt-in switches added (all default off, existing behaviour unchanged): `lookahead_s` (prefix), `prompt_format: training`, `silence_speaker1`, `make_rows_test_configs --repeats`, `smoke_rows.sh` TAG/SPLIT/WHO/PROMPT/SILENCE1/LOOKAHEAD/REPEATS.
- Housekeeping: oov-shared `asr_roundtrip.py align` skips clips it cannot align (a reply shorter than its text crashed the stage); the `personaplex` workspace expires 2026-12-19.

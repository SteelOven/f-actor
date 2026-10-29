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
  - [ ] generator flags (`--prefix-demos` ...)
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

# What this fork changes (relative to MaikeZuefle/f-actor)

The model-code changes are meant to be opt-in (default flags leave training/generation as upstream); the exceptions are the local-inference
changes in `main` (device selection, fp16 load on CUDA). Not verified by a numerical comparison against upstream. The fork's base is upstream
commit `2efa25d`; upstream has moved on, see "Upstream sync". Branches are stacked: each contains the one before.

| Branch | Adds | Commits |
|---|---|---|
| `main` | Local inference (CUDA/MPS/CPU, JSON-config driven `inference_example.py`, fp16), OOV analysis scripts, cluster sbatch wrappers, `run_oov_batch.py` | 35 over upstream |
| `layer/1-train-options` | Training flags that change the loss/data: `--system_speaker`, `--mask_prompt_loss`, `--word_loss_weight` | 3 |
| `layer/2-prefix` | Prefix generation: a fixed first turn (TTS clip) before the model speaks; tail, look-ahead, multi-turn prefixes; checks and golden file | 17 |
| `layer/3-rows-finetune` | Request to reply fine-tune and its evals: `finetune_rows.sh`, `smoke_rows.sh`, teacher-forced check, EOS probe, inference options | 30 |
| `dev` | `layer/3` minus the row builders (moved to the `tts_rows` repo) and the experiment docs (moved to `oov-shared/docs/experiments`) | +4 |

`answer-finetune`, `prefix`, `prefix-multiturn`, `finetune-repeat` are the pre-restructure branches (same content, different history); tags
`backup/2026-11-09*` point at them. The `layer/*` branches were built by cherry-picking those commits (`git log` shows `(cherry picked from commit ...)`
with the old hash, which is what the experiment logs cite).

## Flags and keys (defaults reproduce upstream behaviour)

Training (`training/arguments/arguments.py`):

| Flag | Default | Effect |
|---|---|---|
| `--system_speaker 0\|1` | none | Fix which speaker is the system speaker (otherwise random role swap) |
| `--mask_prompt_loss` | off | Prompt tokens are inputs only, not text-loss targets |
| `--word_loss_weight W` | 1.0 | Up-weight real word tokens in the text loss (normalised by the summed weights) |
| `--keep_final_model`, `--early_stopping_patience 0` | off, 10 | Keep the last weights instead of the best-eval checkpoint (`finetune_rows.sh`: `KEEP_FINAL=1`) |

Generation (`confs/*.json`, read by `training/inference_example.py`):

| Key / env | Effect |
|---|---|
| `prefix` (`wav`, `words`, `tail_s`, `lookahead_s`, `gain_db`, or `turns[]` for several turns) | Fixed first turn(s) by speaker 1; built by `training/prefix.py` |
| `silence_speaker1` | After the prefix the request speaker's audio heads are set to silence |
| `prompt_format: "training"` | Narrative written as in training (`- <narrative>`) |
| `FACTOR_MODEL_DIR` (env) | Load a local model dir instead of the released checkpoint |

## Where the rest went

| Moved to | What |
|---|---|
| `tts_rows` | Row builders (`train_rows.py` is now `tts_rows.training_rows`, `build_rows_dataset.py` is `tts_rows.build_dataset`); `smoke_rows.sh` calls them with `PYTHONPATH=$WS/tts_rows/src` |
| `oov-shared/docs/experiments` | `prefix_log.md`, `oov_experiment.md`, `variants.md`, `advisor_update.md` |
| stays | `docs/model_walkthrough.md`, `notebooks/01-03` (they walk through this repo's code) |

## Upstream sync

Upstream added a depth-decoder head and backchannel control (22 commits, 15 files, 13 of them in `training/`). A trial merge into the fork conflicts in 7 files
(13 hunks, mostly `modeling_dsu.py`). It changes the code under every result so far, so it is done on its own branch and not merged into `dev`
until you decide to move the experiments onto it.

## Adding an experiment

Branch from `dev` as `exp/<name>`, add the new behaviour behind a default-off flag or config key, log the run in `oov-shared/docs/experiments`, tag what
produced a result (`exp/<name>-v1`). Merge it into `dev` only if it should stay.

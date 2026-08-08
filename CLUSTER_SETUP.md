# F-Actor on bwUniCluster 3.0 — Setup Notes

F-Actor's cluster setup rides on top of PersonaPlex's — same OOV/rare-word
research, same enroot container, same `$WS` workspace allocation. See
`personaplex/CLUSTER_SETUP.md` for the cluster basics (login, `salloc`,
what `$WS`/`ws_find`/the container mounts actually are); this file only
covers what's specific to F-Actor.

This currently covers **getting F-Actor running standalone on the cluster** —
confirmed working end to end (2026-08-08): `training/inference_example.py`
generated `outputs/choose_your_name.wav` + `.json` with a full two-speaker
transcript. Mirroring the local OOV
probing setup (`scripts/analysis/generate_oov_configs.py`,
`training/run_oov_batch.py`, `scripts/analysis/analyze_oov.py`, see
`docs/oov_experiment.md`) onto the cluster is deliberately **not** part of
this yet — that comes after bring-up is confirmed working.

Cluster: bwUniCluster 3.0 (KIT), login `uc3.scc.kit.edu`, account `ka_udtmx`.

---

## Quick start (once one-time setup below exists)

```bash
# 1. Local machine: connect
ssh unicluster

# 2. Get a GPU node
salloc -p dev_gpu_h100 --gres=gpu:1 -t 00:30:00

# 3. Same entry point as PersonaPlex - now also mounts ~/f-actor at
#    /workspace/factor-code and exports $FACTOR_PYTHON
~/personaplex/cluster_shell.sh

# 4. Generate the example dialogue
cd /workspace/factor-code
$FACTOR_PYTHON training/inference_example.py
```

Output lands in `outputs/` (stereo wav, per-speaker mono wavs, a `.json` with
transcripts + the config that produced them) — same directory the local setup
uses, per `LOCAL_SETUP.md`.

---

## Directory layout

| Path | What | Persistence |
|---|---|---|
| `$HOME/f-actor` (mounted at `/workspace/factor-code`) | This repo's code (rsynced from local, not git-cloned) | Permanent (`$HOME`, Lustre) |
| `$WS/factor-venv` | Python 3.10 venv — `torch` (default index, CUDA build), `transformers<5`, `nemo-toolkit[tts]`, this package (`pip install -e .`). Kept separate from moshi's `$WS/venv` (`torch<2.5` pinned) and `$WS/tools-venv`, for the same isolation reason both exist. Never activated — invoke via `$FACTOR_PYTHON` | In workspace |
| `$WS/factor-hf-models` | F-Actor's HF weights (~10.5 GB, `maikezu/f-actor`) — see the symlink gotcha below, this isn't where the code thinks it's writing | In workspace |
| `~/f-actor/hf_models` | A **symlink** into `$WS/factor-hf-models`, not a real directory — see gotcha below | Permanent path, workspace-backed content |

---

## One-time setup

1. **Code transfer** — separate rsync from PersonaPlex's, own repo/history.
   `.git` included for the same traceability reasons as PersonaPlex's:
   ```bash
   rsync -avz \
     --exclude='.venv' --exclude='hf_models' --exclude='outputs' --exclude='UniCluster' \
     /home/joesteel/Projects/GitHub_Repos/f-actor/ unicluster:~/f-actor/
   ```
2. **Venv + install** (inside the container, `dev_cpu` node is fine — no GPU
   needed for installing packages, same reasoning as PersonaPlex's own
   one-time setup):
   ```bash
   export NVIDIA_VISIBLE_DEVICES=void
   ~/personaplex/cluster_shell.sh

   uv venv $WS/factor-venv --python 3.10
   # torch first, from the default index (CUDA build on Linux) - this repo's
   # pyproject.toml pins CPU-only torch via `uv sync`, which we don't want on
   # a GPU cluster. See f-actor's own LOCAL_SETUP.md ("Linux + NVIDIA GPU is
   # the exception"), same recipe, just retargeted with --python instead of
   # `source .venv/bin/activate` - $WS/venv is already the active venv this
   # whole session (cluster_env.sh), and uv prefers that over whichever venv's
   # own binary you invoke unless you force it with --python explicitly.
   uv pip install --python $WS/factor-venv/bin/python3 torch
   cd /workspace/factor-code
   uv pip install --python $WS/factor-venv/bin/python3 -e .
   ```
   If `nemo-toolkit`'s install complains about a missing system library
   (`libsndfile`, `ffmpeg`, etc.), install it the same way the original
   `apt-get` step did for PersonaPlex (needs `--root --rw`, see that file's
   Gotchas section). One already hit and confirmed fixed this way:
   `python3-dev` — needed by `cdifflib` (a transitive dep pulled in via
   `full-duplex`/nemo), which fails to build with `fatal error: Python.h: No
   such file or directory` otherwise:
   ```bash
   export NVIDIA_VISIBLE_DEVICES=void
   enroot start --root --rw \
     -m $HOME/personaplex:/workspace/code -m $HOME/f-actor:/workspace/factor-code -m $WS:/workspace/ws \
     personaplex
   apt-get update && apt-get install -y --no-install-recommends python3-dev
   exit
   ```
   Same as the original `libopus-dev`/`build-essential` install — a one-time,
   image-persisted fix, not something to repeat per session.
3. **Weights — redirect before the first run.** `training/inference_example.py`
   downloads the model to a **hardcoded relative path**, `./hf_models/f_actor`
   (see `snapshot_download(..., local_dir="./hf_models/f_actor")` in that
   file) — no flag or env var overrides it. Left alone, ~10.5 GB lands in
   `$HOME` (small/permanent territory, wrong place). Fix: symlink the
   hardcoded path into the workspace *before* the first run:
   ```bash
   mkdir -p $WS/factor-hf-models
   ln -s $WS/factor-hf-models /workspace/factor-code/hf_models
   ```
4. **First run downloads weights — needs real network access.**
   `cluster_env.sh` sets `HF_HUB_OFFLINE=1` for the whole session (so
   PersonaPlex's GPU-timed runs never try to hit the network) — that would
   block F-Actor's *first* download the same way it would for a fresh
   PersonaPlex weight cache. Unset it just for this one run:
   ```bash
   salloc -p dev_gpu_h100 --gres=gpu:1 -t 00:30:00
   ~/personaplex/cluster_shell.sh
   cd /workspace/factor-code
   HF_HUB_OFFLINE=0 $FACTOR_PYTHON training/inference_example.py
   ```
   This both downloads the weights and generates the example dialogue — the
   bring-up smoke test and the weight-prefetch step are the same command.
   Confirmed: the NanoCodec checkpoint downloads fine into `$HF_HOME`
   (`$WS/huggingface`, already redirected by `cluster_env.sh`) — no extra
   NeMo-specific cache path to worry about.

---

## Gotchas encountered (and why the fixes are what they are)

- **`uv pip install` silently installs into the wrong venv** if you don't pass
  `--python` explicitly — `cluster_env.sh` keeps moshi's `$WS/venv` active
  (`$VIRTUAL_ENV` set) for the whole session, and `uv` prefers that over the
  path of whichever venv's own `uv`/`pip` binary you actually invoke. Bit
  PersonaPlex's `tools-venv` setup once already; same fix here (always
  `--python $WS/factor-venv/bin/python3`).
- **Hardcoded `./hf_models/f_actor` download path** — not configurable, must
  be pre-empted with a symlink into `$WS` (step 3 above) or it silently fills
  up `$HOME` instead.
- **`HF_HUB_OFFLINE=1` blocks the first download** — inherited from
  `cluster_env.sh`, which assumes weights are already cached (true for
  PersonaPlex after its own one-time setup, not true for F-Actor's first run).
- **`cdifflib` fails to build: `fatal error: Python.h: No such file or
  directory`** — the container had `build-essential` (gcc etc.) from
  PersonaPlex's own setup but never `python3-dev` (the Python headers), since
  nothing PersonaPlex installs needed them. Fix in step 2 above.
- **`OSError: Read-only file system: '.../.triton'`** during generation (not
  install) — same read-only-`$HOME` cause as PersonaPlex's own Triton/uv
  gotchas, but a different trigger: F-Actor's `transformers` version uses
  native fused Triton kernels directly (e.g. in the rope implementation), not
  just `torch.compile`, so `NO_TORCH_COMPILE=1` doesn't prevent it. Fixed by
  `export TRITON_CACHE_DIR=$WS/.triton-cache`, now baked into
  `cluster_env.sh` — nothing to do manually anymore.
- **`mkdir -p .../.config/matplotlib` fails, read-only** — cosmetic (falls
  back to `/tmp` automatically) but silenced anyway via `MPLCONFIGDIR`, also
  now in `cluster_env.sh`.
- **`enroot start --root --rw` fails with `No such file or directory:
  .../.local/share/enroot/personaplex`** if you forget `export
  ENROOT_DATA_PATH=$WS/enroot-data` first — the container image doesn't live
  under the default enroot data path, only under the workspace one.
  `cluster_shell.sh` sets this for you; a manual `--root --rw` invocation
  (for one-off `apt-get install`s) has to set it explicitly too.
- **`AssertionError: CUDA or MPS is required`** if `NVIDIA_VISIBLE_DEVICES=void`
  is still exported in your shell from an earlier non-GPU (`--root --rw`)
  step — it silently hides the GPU from the container even on a real GPU
  node. `unset` it (or start a fresh shell) before GPU work.

---

## Open items / next steps

1. ~~Confirm the steps above actually work end to end on the real cluster~~ —
   done 2026-08-08, see top of file.
2. Mirror the local OOV probing pipeline
   (`generate_oov_configs.py` → `confs/oov/*.json` →
   `training/inference_example.py` → `analyze_oov.py`) onto the cluster,
   analogous to how PersonaPlex's `experiments.csv`/`run_batch.sh` work. Given
   `inference_example.py` already supports `--transcribe`/`--asr-model` for
   one-pass ASR grading, worth deciding then whether to keep that self-contained
   path or point it at the shared `asr_backends/` package from `personaplex/`
   for a single ASR implementation used by both models — not decided yet.

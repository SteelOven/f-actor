# Running F-Actor locally (macOS / CPU / CUDA)

This fork makes inference device-agnostic: it picks **CUDA → MPS (Apple
Silicon) → CPU** automatically (see `training/device_utils.py`). The codec
decoders (NanoCodec / Mimi) run on CUDA or CPU only — decoding takes seconds,
and NeMo on MPS is untested.

What to expect:

| Machine | Generation of one ~80 s dialogue |
|---|---|
| NVIDIA GPU (≥6 GB) | well under a minute |
| M1 Pro (MPS) | a few minutes |
| x86 laptop CPU | 5–15+ minutes |

On CUDA the model loads in **fp16** (the fp32 checkpoint alone is ~5.2 GB),
so it fits 6 GB cards like a GTX 1660 Ti. CPU and MPS keep fp32.

This is **offline** dialogue generation (two model instances talking to each
other) — not a real-time voice interface.

## Setup with uv

Requires [uv](https://docs.astral.sh/uv/). A `uv.lock` is committed
(Python 3.10, pinned via `.python-version`), so on **macOS (Apple Silicon)**
and **Linux (CPU)** setup is one command — uv fetches Python, creates
`.venv`, and installs everything pinned:

```bash
cd f-actor
uv sync                  # inference only
uv sync --extra eval     # + paper metrics (eval/ scripts)
uv sync --extra train    # + training deps (deepspeed, wandb, ...)
```

Then either `source .venv/bin/activate` or prefix commands with `uv run`.

Note: the lock pins **CPU-only torch on Linux** (the default PyPI wheel
bundles ~3 GB of CUDA libraries). **Linux + NVIDIA GPU** is the exception —
skip the lock and install the CUDA build directly:

```bash
uv venv --python 3.10 && source .venv/bin/activate
uv pip install torch
uv pip install -e .
```

### Windows + NVIDIA (WSL2)

Use the **Linux + NVIDIA** path above inside a WSL2 distro (e.g. Ubuntu).
The only driver needed is the regular NVIDIA driver **on Windows** — do not
install a Linux driver or CUDA toolkit inside WSL; the default torch wheel
bundles the CUDA runtime. Verify the GPU is visible with `nvidia-smi` inside
WSL before installing. `nemo-toolkit[tts]` installs normally (WSL counts as
Linux). Budget ~25 GB of disk in the WSL VM: 10.5 GB weights + ~3 GB torch
+ NeMo's dependencies.

Notes:

- `nemo-toolkit[tts]` is the heavyweight dependency (needed only to decode
  audio tokens to WAV). It is pip-installable on macOS, but it is officially
  Linux-only — if it fails to build, install the core deps without it and run
  the token→audio step (`training/convert_dsu_to_audio.py`) on a Linux
  machine or container instead. Generation itself does not need NeMo.
- The `eval/` scripts still assume CUDA in places (`eval/calc_speaker_sim.py`,
  `eval/utmos_scores.py`); they are for reproducing paper metrics on a GPU
  machine.

## On Apple Silicon, set the MPS fallback

Some ops may not be implemented on MPS depending on your torch version. This
makes them fall back to CPU instead of crashing:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

## Generate a dialogue

```bash
uv run python training/inference_example.py
```

First run downloads ~10.5 GB of model weights from HuggingFace into
`./hf_models/f_actor` (the repo stores the weights twice — both copies are
used: `transformers` loads `model.safetensors`, the custom audio heads load
from the sharded files) plus the NanoCodec checkpoint via NeMo.

The dialogue is configured in `confs/example_dialogue.json` (or pass your own
file with `--config path/to/my_dialogue.json`):

- per speaker: `name` (`Tom`, `Brian`, `Gweneth`, `Rebeka` — each has a
  precomputed voice embedding in `training/example_speakers/`), `narrative`
  (each side only sees its own instructions), `starts`, and the behavior
  controls `backchannels` / `interruptions`,
- plus `output_file`, `max_length` (1024 ≈ 80 s; 12.5 frames/s), and sampling
  parameters (`temperature`, `top_k`, `top_p`).

Output (in `output_dir`, default `outputs/`): a stereo WAV (channel 1 =
speaker 1, channel 2 = speaker 2), separate `_c1`/`_c2` mono files, and a
`<name>.json` with both transcripts plus the config that produced them. The
transcripts are also printed to stdout.

## Batch inference / paper reproduction

`scripts/inference_eval/inference_nanocodec_special_tokens.sh` generates
dialogues from the Behavior-SD test set (downloads the full dataset) and
`eval.sh` computes the paper metrics — GPU strongly recommended.

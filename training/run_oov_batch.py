"""Run many dialogue configs through F-Actor without reloading the model
per config.

`training/inference_example.py --config X` reloads the ~10.5 GB model +
NanoCodec on every invocation. That's fine for one-off runs, but looping it
over many configs (e.g. `for f in confs/oov/*.json; do ...; done`) pays that
full load cost once per config instead of once per job -- the dominant cost
once you're generating more than a handful of dialogues. This script loads
the model once and reuses it for every config in its shard.

Usage:
    # everything matching a glob
    python training/run_oov_batch.py --configs-glob 'confs/oov/*.json'

    # this job's 1/8 slice, e.g. one SLURM array task of 8
    python training/run_oov_batch.py --configs-glob 'confs/oov/*.json' \\
        --shard-index "$SLURM_ARRAY_TASK_ID" --shard-count "$SLURM_ARRAY_TASK_COUNT"

Generation only -- no ASR/grading here. Once configs are generated, run
scripts/analysis/analyze_oov.py separately (via $TOOLS_PYTHON, PersonaPlex's
ASR/TTS tools venv) to merge transcripts + word-fidelity grading into each
run record. Keeping ASR out of this process means this venv (factor-venv)
never needs faster-whisper/asr_backends, matching how PersonaPlex itself
keeps run_probe.sh and analyze_runs.py in separate venvs.
"""

import argparse
import glob
import json
import os
import sys
import time

from inference_example import generate_dialogue, load_model
from inference_audio.nano_decode import load_model as load_nanocodecs

SPEAKERS = ["Rebeka", "Gweneth", "Brian", "Tom"]


def is_already_done(config):
    stem = os.path.splitext(config["output_file"])[0]
    record_path = os.path.join(config["output_dir"], f"{stem}.json")
    return os.path.exists(record_path)


def select_shard(paths, shard_index, shard_count):
    paths = sorted(paths)
    if shard_count <= 1:
        return paths
    return [p for i, p in enumerate(paths) if i % shard_count == shard_index]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs-glob", required=True,
        help="Glob for config JSONs, e.g. 'confs/oov/*.json'.",
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()

    if args.shard_count < 1 or not (0 <= args.shard_index < args.shard_count):
        parser.error("--shard-index must be in [0, shard_count) and shard_count >= 1")

    all_paths = glob.glob(args.configs_glob)
    if not all_paths:
        parser.error(f"No configs matched {args.configs_glob!r}")
    paths = select_shard(all_paths, args.shard_index, args.shard_count)
    print(f"Shard {args.shard_index}/{args.shard_count}: {len(paths)}/{len(all_paths)} configs")
    if not paths:
        return

    print("Loading model and tokenizer (once for the whole batch)...")
    model, tokenizer = load_model()
    nanocodec_model = load_nanocodecs(num_codebooks=4)

    failures = []
    for i, path in enumerate(paths, 1):
        t0 = time.time()
        try:
            with open(path) as f:
                config = json.load(f)

            if is_already_done(config):
                print(f"[{i}/{len(paths)}] {path}: skip (already completed)")
                continue

            for spk in (config["speaker1"], config["speaker2"]):
                if spk["name"] not in SPEAKERS:
                    raise KeyError(f"Invalid speaker '{spk['name']}', choose from {SPEAKERS}")

            _, _, record_file = generate_dialogue(model, tokenizer, nanocodec_model, config)

            print(f"[{i}/{len(paths)}] {path} -> {record_file} ({time.time() - t0:.0f}s)")
        except Exception as e:
            print(f"[{i}/{len(paths)}] FAILED {path}: {e}", file=sys.stderr)
            failures.append((path, str(e)))

    ok = len(paths) - len(failures)
    print(f"\nDone: {ok}/{len(paths)} succeeded")
    if failures:
        print(f"{len(failures)} failed:")
        for path, err in failures:
            print(f"  {path}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()

"""Did the model answer "Okay, <word>"? Transcribes speaker 2's continuation of each rows-test config with parakeet.

Superseded 2026-11-12: evaluations of fine-tuned models go through oov-shared run_experiment.sh (factor.model + factor.rows),
graded by analyze_oov.py with the numbers in metrics.json (oov-shared README.md, "Evaluating a fine-tuned model"). This script
is kept to reproduce old numbers; its rule is stricter (exact word match) than analyze_oov's intact/split labels.

    python scripts/analysis/score_rows_test.py --configs-glob 'confs/rows_test/finetuned/*.json'

The continuation is <output_dir>/<word>_c2_cont.wav (written by inference_example's split_prefix_channels). A reply counts as
"okay" if the first transcribed word is OKAY and as "word" if the target word is among the words (exact match after
upper-casing, punctuation removed). Run on a GPU node in the container ($FACTOR_PYTHON); prints a table and the two rates.
"""

import argparse
import glob
import json
import re
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[2] / "eval/bc_interrupt")]


def norm(w: str) -> str:
    return re.sub(r"[^A-Z']", "", w.upper())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--configs-glob", required=True)
    args = ap.parse_args(argv)

    import nemo.collections.asr as nemo_asr
    from get_word_timestamps import get_parakeet_timestps

    asr = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2").eval()
    n = n_okay = n_word = n_missing = 0
    print(f"{'word':<14}{'okay':<6}{'word':<6}heard")
    for path in sorted(glob.glob(args.configs_glob)):
        cfg = json.load(open(path))
        word = norm(cfg["meta"]["word"])
        wav = Path(cfg["output_dir"]) / f"{Path(cfg['output_file']).stem}_c2_cont.wav"
        if not wav.exists():
            n_missing += 1
            print(f"{word.lower():<14}missing {wav}")
            continue
        heard = [norm(w["word"]) for w in get_parakeet_timestps(str(wav), asr, return_words_tmsp=True)]
        heard = [w for w in heard if w]
        okay, hit = bool(heard) and heard[0] == "OKAY", word in heard
        n, n_okay, n_word = n + 1, n_okay + okay, n_word + hit
        print(f"{word.lower():<14}{'yes' if okay else 'no':<6}{'yes' if hit else 'no':<6}{' '.join(heard)}")
    print(f"\n{n} scored ({n_missing} missing): okay {n_okay}/{n}, word {n_word}/{n}")


if __name__ == "__main__":
    main()

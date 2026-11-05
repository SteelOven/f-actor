"""Generation configs for a fine-tune on tts_rows data: the request clip as the teacher-forced prefix, the model answers as speaker 2.

One config per row of a split (default: validation = words the fine-tune never saw). Narratives, speaker names and the
prefix words are taken from the rows themselves, so the model is prompted exactly as in training: speaker 1 (the request
speaker) starts, speaker 2 (the reply speaker, system_speaker=1 in training) does not, no backchannels or interruptions.
The prefix is the request clip as it is (tail_s unset), as in the rows: the reply follows the clip's own trailing silence.

    python scripts/analysis/make_rows_test_configs.py --rows-dir $WS/finetune/p7_short_ok \\
        --request-dir $WS/checks/p6_full_request_s0_ok --timings $WS/finetune/p7_short_ok.timings.json \\
        --out-dir confs/rows_test/finetuned --output-dir outputs/rows_test/finetuned
    FACTOR_MODEL_DIR=<run>/best_model python training/run_oov_batch.py --configs-glob 'confs/rows_test/finetuned/*.json'
    python scripts/analysis/score_rows_test.py --configs-glob 'confs/rows_test/finetuned/*.json'

Run it twice with another --out-dir/--output-dir (and no FACTOR_MODEL_DIR) for the released model as the baseline.
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def request_clips(request_dir: Path) -> dict[str, Path]:
    return {m["word"]: request_dir / m["wav"]
            for m in map(json.loads, (request_dir / "manifest.jsonl").read_text().splitlines()) if "wav" in m}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rows-dir", type=Path, required=True, help="dir with train.parquet / validation.parquet")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--request-dir", type=Path, required=True, help="the request run dir the rows were built from")
    ap.add_argument("--timings", type=Path, required=True, help="the rows script's --timing-cache json (word timings per wav)")
    ap.add_argument("--out-dir", type=Path, required=True, help="where the config JSONs go")
    ap.add_argument("--output-dir", required=True, help="output_dir written into the configs (generated audio)")
    ap.add_argument("--max-length", type=int, default=256, help="generation limit in frames (12.5 per second)")
    ap.add_argument("--temperature", type=float, default=0.9)
    args = ap.parse_args(argv)

    rows = load_dataset(str(args.rows_dir), split=args.split)
    clips = request_clips(args.request_dir)
    timings = json.loads(args.timings.read_text())
    by_name = {Path(k).name: v for k, v in timings.items()}  # robust to the cluster vs laptop path prefix

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for r in rows:
        word = r["file_name"]
        wav = clips[word]
        words = timings.get(str(wav)) or by_name[wav.name]
        spk1, spk2 = r["speakers"]
        cfg = {
            "meta": {"word": word, "split": args.split},
            "speaker1": {"name": spk1, "narrative": r["new_narrative_s1"], "starts": True, "backchannels": 0, "interruptions": 0},
            "speaker2": {"name": spk2, "narrative": r["new_narrative_s2"], "starts": False, "backchannels": 0, "interruptions": 0},
            "prefix": {"wav": str(wav), "words": words},
            "output_dir": args.output_dir,
            "output_file": f"{word}.wav",
            "max_length": args.max_length,
            "temperature": args.temperature,
            "top_k": 40,
            "top_p": 1.0,
        }
        (args.out_dir / f"{word}.json").write_text(json.dumps(cfg, indent=2))
    print(f"wrote {len(rows)} configs ({args.split}) to {args.out_dir}")


if __name__ == "__main__":
    main()

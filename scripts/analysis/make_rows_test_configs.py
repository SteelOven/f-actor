"""Generation configs for a fine-tune on tts_rows data: the request clip as the teacher-forced prefix, the model answers as speaker 2.

One config per row of a split (default: validation = words the fine-tune never saw). Narratives, speaker names and the
prefix words are taken from the rows themselves, so the model is prompted exactly as in training: speaker 1 (the request
speaker) starts, speaker 2 (the reply speaker, system_speaker=1 in training) does not, no backchannels or interruptions.
By default the prefix is the request clip as it is; --tail-s N replaces its trailing silence by N seconds (a prefix that ends right
after the last word makes the model emit EOS at the first generated frame, see docs/prefix_log.md).

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
    ap.add_argument("--tail-s", type=float, help="trim the request clip's trailing silence and add exactly this much digital silence "
                    "(prefix.py build_prefix tail_s); default: keep the clip as it is")
    ap.add_argument("--lookahead-s", type=float, default=0.0, help="prefix look-ahead (prefix.py): encode this much extra silence and cut the "
                    "codes back, which removes the codec's end-of-file codes from the prefix end")
    ap.add_argument("--prompt-format", choices=["default", "training"], default="default",
                    help="'training': write the narrative as '- <narrative>' like training's build_prompt (inference_example's default omits the dash)")
    ap.add_argument("--silence-speaker1", action="store_true", help="after the prefix, speaker 1 (the request speaker) stays silent "
                    "(its audio heads are set to silence, not sampled), as in the training rows")
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
            "prefix": {"wav": str(wav), "words": words, **({"tail_s": args.tail_s} if args.tail_s is not None else {}),
                       **({"lookahead_s": args.lookahead_s} if args.lookahead_s else {})},
            **({"prompt_format": args.prompt_format} if args.prompt_format != "default" else {}),
            **({"silence_speaker1": True} if args.silence_speaker1 else {}),
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

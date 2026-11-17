"""Forced-text configs: groups of words x repeats from one base config, for a control run (do common words come out clear where a
rare one does not?).

    python scripts/analysis/make_forced_text_configs.py --out-dir confs/forced_control --output-dir scratch/forced_control
    python scripts/analysis/make_forced_text_configs.py --group common=water,really,okay --group rare=xylophone,kombucha \
        --repeats 5 --n 2 --free --out-dir confs/forced_control --output-dir scratch/forced_control
    python training/run_oov_batch.py --configs-glob 'confs/forced_control/*.json'
    python scripts/analysis/show_text_frames.py scratch/forced_control/xylophone_n2_r1.json --around-forced

Every config is the base config (default confs/example_dialogue.json; no prefix, the generic setup) plus
"force_text": [{"speaker": <speaker>, "n": <n>}] and meta.word = the word. --free also writes the same configs without force_text
(files <word>_free_r<k>): the model's own behaviour at the same settings, to compare against. Sampling is unseeded, so repeats are
independent draws. Writes <out-dir>/manifest.tsv in the format of generate_oov_configs.py (band = the group, "<group>_free" for --free),
so analyze_oov.py can grade the runs.
"""

import argparse
import json
from pathlib import Path

DEFAULT_GROUPS = ["common=water,really,okay", "rare=xylophone"]


def parse_group(s: str) -> tuple[str, list[str]]:
    name, sep, words = s.partition("=")
    words = [w.strip().lower() for w in words.split(",") if w.strip()]
    if not sep or not name.strip() or not words or any(" " in w for w in words):
        raise argparse.ArgumentTypeError(f"--group NAME=word1,word2 (single words), got {s!r}")
    return name.strip(), words


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", type=Path, default=Path("confs/example_dialogue.json"), help="config every run starts from")
    ap.add_argument("--group", type=parse_group, action="append", help=f"NAME=word1,word2 (repeatable); default: {' '.join(DEFAULT_GROUPS)}")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--n", type=int, default=2, help="force the speaker's n-th generated word (1-based)")
    ap.add_argument("--speaker", choices=["speaker1", "speaker2"], default="speaker2")
    ap.add_argument("--free", action="store_true", help="also write each config without force_text (the unforced comparison)")
    ap.add_argument("--max-length", type=int, default=200, help="generation limit in frames (12.5 per second)")
    ap.add_argument("--out-dir", type=Path, required=True, help="where the config JSONs go")
    ap.add_argument("--output-dir", required=True, help="output_dir written into the configs (generated audio and run records)")
    args = ap.parse_args(argv)
    groups = args.group or [parse_group(g) for g in DEFAULT_GROUPS]
    seen = [w for _, ws in groups for w in ws]
    if len(set(seen)) != len(seen):
        ap.error(f"a word is in two groups: {seen}")

    base = json.loads(args.base.read_text())
    base.pop("force_text", None)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = ["word\tspeech_count\tnarrative_count\tband\trepeat\tconfig\taudio\tprefix"]
    variants = [("forced", True)] + ([("free", False)] if args.free else [])
    for group, words in groups:
        for word in words:
            for variant, forced in variants:
                for k in range(1, args.repeats + 1):
                    stem = f"{word}_n{args.n}_r{k}" if forced else f"{word}_free_r{k}"
                    cfg = json.loads(json.dumps(base))
                    cfg["meta"] = {**cfg.get("meta", {}), "word": word, "group": group, "variant": variant, "repeat": k}
                    if forced:
                        cfg["force_text"] = [{"speaker": args.speaker, "n": args.n}]
                    cfg.update(output_dir=args.output_dir, output_file=f"{stem}.wav", max_length=args.max_length)
                    path = args.out_dir / f"{stem}.json"
                    path.write_text(json.dumps(cfg, indent=2))
                    band = group if forced else f"{group}_free"
                    manifest.append(f"{word}\t\t\t{band}\t{k}\t{path}\t{Path(args.output_dir) / (stem + '.wav')}\t")
    (args.out_dir / "manifest.tsv").write_text("\n".join(manifest) + "\n")
    n = len(manifest) - 1
    print(f"wrote {n} configs ({len(seen)} words x {args.repeats} repeats x {len(variants)} variant(s)) to {args.out_dir}")


if __name__ == "__main__":
    main()

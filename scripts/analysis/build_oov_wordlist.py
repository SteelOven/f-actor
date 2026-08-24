"""Build the canonical OOV word list shared with PersonaPlex.

Looks up each target word's training frequency (scripts/analysis/oov_vocab.py,
same table generate_oov_configs.py uses) and writes one CSV that both
pipelines build their test cases from: this repo's generate_oov_configs.py
(word -> two-speaker dialogue config) and PersonaPlex's
build_stimuli_from_wordlist.py (word -> prompt file + experiments_oov.csv row).

Usage:
    python scripts/analysis/build_oov_wordlist.py --words hello quantum schadenfreude \\
        --out ../personaplex/oov_wordlist.csv
    python scripts/analysis/build_oov_wordlist.py --words-file mywords.txt \\
        --out "$PERSONAPLEX_DIR/oov_wordlist.csv"
"""

import argparse
import csv

from oov_vocab import get_band, load_vocab

VOCAB_TABLE = "scripts/analysis/behavior_sd_vocab.tsv"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--words", nargs="+", default=[], help="Target words.")
    parser.add_argument(
        "--words-file", help="File with one target word per line (# comments ok)."
    )
    parser.add_argument("--vocab", default=VOCAB_TABLE)
    parser.add_argument("--out", required=True, help="Output CSV path.")
    args = parser.parse_args()

    words = [w.lower() for w in args.words]
    if args.words_file:
        with open(args.words_file) as f:
            words += [
                line.strip().lower()
                for line in f
                if line.strip() and not line.startswith("#")
            ]
    if not words:
        parser.error("No words given (use --words or --words-file).")

    vocab = load_vocab(args.vocab)

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["word", "speech_count", "narrative_count", "band"])
        for word in words:
            speech_count, narrative_count = vocab.get(word, (0, 0))
            band = get_band(speech_count, narrative_count)
            writer.writerow([word, speech_count, narrative_count, band])
            print(f"{word:<25}{band:<16}speech={speech_count}")

    print(f"\n{len(words)} word(s) -> {args.out}")


if __name__ == "__main__":
    main()

"""Test H2 (acoustic coverage) at the sub-word level for the OOV experiment.

H2 says F-Actor's audio head fails on rare words because it never learned to
*voice* their sub-units -- not because the whole word is rare. The distinguishing
prediction: audio fidelity should track the training frequency of a word's
**sub-units**, even among words whose whole-word `speech_count` is identical
(e.g. all the band=oov words have speech_count 0, yet "cryptocurrency" is built
from common chunks and "borborygmus" is not).

This script measures sub-unit coverage with **no model and no extra deps**: it
derives spoken character-n-gram frequencies straight from
`behavior_sd_vocab.tsv` by weighting each word's n-grams by its `speech_count`
(how often that chunk was actually spoken in training). Character n-grams are a
dependency-free proxy for phonemes; for true phonemes, swap `ngrams()` for a G2P
(e.g. g2p_en) -- the rest is unchanged.

For each target word it computes:
  * frac_unseen   -- fraction of its n-grams never spoken in training (0 freq)
  * min_logfreq   -- log10 freq of its rarest n-gram (the bottleneck unit)
  * mean_logfreq  -- mean log10 n-gram freq

then joins audio fidelity from outputs/oov/analysis.tsv (if present) and reports
Spearman correlations. H2 predicts: audio_score rises with min/mean_logfreq and
falls with frac_unseen -- and that those sub-unit features predict fidelity
*better than* whole-word speech_count.

Usage:
    uv run python scripts/analysis/subword_coverage.py
    uv run python scripts/analysis/subword_coverage.py --n 2 --words-file scripts/analysis/oov_words.txt
"""

import argparse
import csv
import math
from collections import Counter, defaultdict

from scipy.stats import spearmanr

DEFAULT_VOCAB = "scripts/analysis/behavior_sd_vocab.tsv"
DEFAULT_WORDS = "scripts/analysis/oov_words.txt"
DEFAULT_ANALYSIS = "outputs/oov/analysis.tsv"
PAD = "^"  # word-boundary marker so onsets/codas count as their own units


def ngrams(word, n):
    s = PAD + word.lower() + PAD
    return [s[i : i + n] for i in range(len(s) - n + 1)]


def build_ngram_freq(vocab_path, n):
    """Spoken n-gram frequency = sum over words of speech_count * occurrences."""
    freq = Counter()
    word_speech = {}
    with open(vocab_path) as f:
        next(f)  # header
        for line in f:
            word, speech, _narr = line.rstrip("\n").split("\t")
            speech = int(speech)
            word_speech[word] = speech
            if speech == 0:
                continue
            for g in ngrams(word, n):
                freq[g] += speech
    return freq, word_speech


def word_features(word, n, freq):
    grams = ngrams(word, n)
    counts = [freq.get(g, 0) for g in grams]
    logs = [math.log10(c + 1) for c in counts]
    unseen = [g for g, c in zip(grams, counts) if c == 0]
    return {
        "n_grams": len(grams),
        "frac_unseen": round(sum(c == 0 for c in counts) / len(counts), 3),
        "min_logfreq": round(min(logs), 2),
        "mean_logfreq": round(sum(logs) / len(logs), 2),
        "unseen_grams": " ".join(g.replace(PAD, "#") for g in unseen),
    }


def load_words(path):
    with open(path) as f:
        return [l.strip().lower() for l in f if l.strip() and not l.startswith("#")]


def load_audio_scores(path):
    """Mean audio_score per word from analyze_oov.py output (may be absent)."""
    scores = defaultdict(list)
    try:
        with open(path, newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                scores[row["word"]].append(float(row["audio_score"]))
    except FileNotFoundError:
        return {}
    return {w: sum(v) / len(v) for w, v in scores.items()}


def corr(xs, ys):
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None  # undefined when an input is constant
    rho, p = spearmanr(xs, ys)
    return round(rho, 3), round(p, 3)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vocab", default=DEFAULT_VOCAB)
    ap.add_argument("--words-file", default=DEFAULT_WORDS)
    ap.add_argument("--analysis", default=DEFAULT_ANALYSIS)
    ap.add_argument("--n", type=int, default=3, help="character n-gram size")
    ap.add_argument("--report", default="outputs/oov/subword_coverage.tsv")
    args = ap.parse_args()

    freq, word_speech = build_ngram_freq(args.vocab, args.n)
    words = load_words(args.words_file)
    audio = load_audio_scores(args.analysis)

    rows = []
    for w in words:
        feats = word_features(w, args.n, freq)
        rows.append({
            "word": w,
            "speech_count": word_speech.get(w, 0),
            **feats,
            "audio_score": round(audio[w], 1) if w in audio else "",
        })

    hdr = ["word", "speech_count", "n_grams", "frac_unseen", "min_logfreq",
           "mean_logfreq", "audio_score", "unseen_grams"]
    with open(args.report, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=hdr, delimiter="\t")
        wtr.writeheader()
        wtr.writerows(rows)

    # --- predictor side: ready even with zero runs ---
    print(f"Sub-unit coverage (char {args.n}-grams), sorted by coverage:\n")
    print(f"{'word':<18}{'speech':>7}{'frac_unseen':>12}{'min_lf':>8}{'mean_lf':>8}"
          f"{'audio':>7}  unseen n-grams")
    print("-" * 86)
    for r in sorted(rows, key=lambda r: r["mean_logfreq"]):
        a = f"{r['audio_score']}" if r["audio_score"] != "" else "-"
        print(f"{r['word']:<18}{r['speech_count']:>7}{r['frac_unseen']:>12}"
              f"{r['min_logfreq']:>8}{r['mean_logfreq']:>8}{a:>7}  {r['unseen_grams']}")

    # --- correlation side: needs runs ---
    scored = [r for r in rows if r["audio_score"] != ""]
    print(f"\nRuns with audio fidelity: {len(scored)}")
    if len(scored) < 3:
        print("Too few runs for correlation -- generate more (balanced bands) and re-run.")
        print(f"\nPer-word table: {args.report}")
        return

    ys = [r["audio_score"] for r in scored]
    print("\nSpearman vs audio_score (H2: positive for *_logfreq, negative for frac_unseen):")
    for feat, sign in [("speech_count", "+"), ("frac_unseen", "-"),
                       ("min_logfreq", "+"), ("mean_logfreq", "+")]:
        c = corr([r[feat] for r in scored], ys)
        if c:
            print(f"  {feat:<14} rho={c[0]:>6}  p={c[1]:<6}  (H2 expects {sign})")
    print("\nKey H2 test: among speech_count==0 words, do sub-unit features still")
    print("predict fidelity? (whole-word frequency can't -- it's 0 for all of them)")
    print(f"\nPer-word table: {args.report}")


if __name__ == "__main__":
    main()

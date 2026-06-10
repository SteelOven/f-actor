"""Generate dialogue configs that force target words, for OOV experiments.

Each target word becomes one config (x --repeats) for
training/inference_example.py: speaker 1 is instructed to explain the word,
speaker 2 to ask questions about it. Words are annotated with their training
frequency from the Behavior-SD vocab table (see build_behavior_sd_vocab.py)
so results can be analyzed by frequency band.

Usage:
    python scripts/analysis/generate_oov_configs.py --words hello quantum schadenfreude
    python scripts/analysis/generate_oov_configs.py --words-file mywords.txt --repeats 3

Then run the generated configs:
    for f in confs/oov/*.json; do
        python training/inference_example.py --config "$f"
    done
"""

import argparse
import json
import os

VOCAB_TABLE = "scripts/analysis/behavior_sd_vocab.tsv"

NARRATIVE_EXPLAINER = (
    "You are excited to tell {partner} about '{word}'. You explain to "
    "{partner} what '{word}' means and you use the word '{word}' several times."
)
NARRATIVE_LISTENER = (
    "{partner} tells you about something called '{word}'. You are curious "
    "and ask {partner} questions about it."
)


def get_band(speech_count, narrative_count):
    if speech_count > 100:
        return "frequent"
    if speech_count > 10:
        return "medium"
    if speech_count > 0:
        return "rare"
    if narrative_count > 0:
        return "narrative-only"
    return "oov"


def load_vocab(path):
    table = {}
    with open(path) as f:
        next(f)
        for line in f:
            word, speech_count, narrative_count = line.rstrip("\n").split("\t")
            table[word] = (int(speech_count), int(narrative_count))
    return table


def make_config(word, speaker1, speaker2, rep, args):
    return {
        "speaker1": {
            "name": speaker1,
            "narrative": NARRATIVE_EXPLAINER.format(partner=speaker2, word=word),
            "starts": True,
            "backchannels": args.backchannels,
            "interruptions": args.interruptions,
        },
        "speaker2": {
            "name": speaker2,
            "narrative": NARRATIVE_LISTENER.format(partner=speaker1, word=word),
            "starts": False,
            "backchannels": args.backchannels,
            "interruptions": args.interruptions,
        },
        "output_dir": args.audio_dir,
        "output_file": f"{word}_r{rep}.wav",
        "max_length": args.max_length,
        "temperature": 0.9,
        "top_k": 40,
        "top_p": 1.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--words", nargs="+", default=[], help="Target words.")
    parser.add_argument(
        "--words-file", help="File with one target word per line (# comments ok)."
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512, dest="max_length")
    parser.add_argument("--backchannels", type=int, default=2)
    parser.add_argument("--interruptions", type=int, default=2)
    parser.add_argument("--speakers", default="Tom,Brian")
    parser.add_argument("--outdir", default="confs/oov")
    parser.add_argument("--audio-dir", default="outputs/oov", dest="audio_dir")
    parser.add_argument("--vocab", default=VOCAB_TABLE)
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

    speaker1, speaker2 = args.speakers.split(",")
    vocab = load_vocab(args.vocab)
    os.makedirs(args.outdir, exist_ok=True)

    manifest_path = os.path.join(args.outdir, "manifest.tsv")
    with open(manifest_path, "w") as manifest:
        manifest.write(
            "word\tspeech_count\tnarrative_count\tband\trepeat\tconfig\taudio\n"
        )
        for word in words:
            speech_count, narrative_count = vocab.get(word, (0, 0))
            band = get_band(speech_count, narrative_count)
            for rep in range(1, args.repeats + 1):
                config = make_config(word, speaker1, speaker2, rep, args)
                config_path = os.path.join(args.outdir, f"{word}_r{rep}.json")
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=2)
                audio_path = os.path.join(args.audio_dir, config["output_file"])
                manifest.write(
                    f"{word}\t{speech_count}\t{narrative_count}\t{band}"
                    f"\t{rep}\t{config_path}\t{audio_path}\n"
                )
            print(f"{word:<25}{band:<16}speech={speech_count}")

    print(f"\n{len(words) * args.repeats} configs in {args.outdir}/")
    print(f"Manifest: {manifest_path}")
    print(
        "\nRun them with:\n"
        f'  for f in {args.outdir}/*.json; do '
        f'python training/inference_example.py --config "$f"; done'
    )


if __name__ == "__main__":
    main()

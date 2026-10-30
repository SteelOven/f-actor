"""Generate dialogue configs that force target words, for OOV experiments.

Each target word becomes one config (x --repeats) for
training/inference_example.py: speaker 1 is instructed to explain the word,
speaker 2 to ask questions about it. Words are annotated with their training
frequency from the Behavior-SD vocab table (see oov-shared's wordlists/count_vocab.py --write-contract)
so results can be analyzed by frequency band.

Usage:
    python scripts/analysis/generate_oov_configs.py --words hello quantum schadenfreude
    python scripts/analysis/generate_oov_configs.py --words-file mywords.txt --repeats 3

    # --words-file also accepts the shared oov_wordlist.csv (see
    # oov-shared/wordlists/build_oov_wordlist.py) - band/frequency come from
    # --vocab either way, the CSV's own speech_count/narrative_count/band
    # columns are ignored so there's one source of truth for those numbers,
    # not two.
    python scripts/analysis/generate_oov_configs.py --words-file ../oov-shared/wordlists/oov_wordlist.csv

    # with a fixed first turn by speaker 1 (TTS clip of the word from an oov-shared
    # round-trip run that has alignment.jsonl; see docs/oov_experiment.md "Prefix
    # experiment"). Output files get a _pfx suffix.
    python scripts/analysis/generate_oov_configs.py --words quokka water \
        --prefix-run-dir '$WS/checks/p3_prefix_tom' --outdir confs/oov_prefix --audio-dir outputs/oov_prefix

    # multi-turn prefix: N demonstration exchanges (Tom asks for a frequent word, speaker 2 says it) before the target
    # request; training/prefix.py build_multiturn_prefix, plan in docs/prefix_log.md. The demo words come from the
    # pool of words that have both a request clip and a reply clip, never the target, seeded per (word, repeat).
    python scripts/analysis/generate_oov_configs.py --words music kitchen --speakers Tom,Gweneth \\
        --prefix-run-dir ../oov-data/checks/p5_demo_request --prefix-tail-s 0.3 --prefix-demos 2 \\
        --demo-request-run-dir ../oov-data/checks/p5_demo_request --demo-reply-run-dir ../oov-data/checks/p5_demo_reply \\
        --demo-exclude coffee dinner --outdir confs/pilot_d2 --audio-dir outputs/pilot_d2

Then run the generated configs:
    for f in confs/oov/*.json; do
        python training/inference_example.py --config "$f"
    done
"""

import argparse
import csv
import json
import os
import random
import sys

# oov_vocab (the shared band definition) and the Behavior-SD frequency table
# live in oov-shared's wordlists/ - $OOV_SHARED_DIR on the cluster (see
# personaplex's cluster_env.sh), otherwise a sibling checkout (also
# /workspace/oov-shared in the container).
OOV_WORDLISTS = os.path.join(
    os.environ.get("OOV_SHARED_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "..", "oov-shared")),
    "wordlists",
)
sys.path.insert(0, OOV_WORDLISTS)

from oov_vocab import get_band, load_vocab  # noqa: E402

VOCAB_TABLE = os.path.join(OOV_WORDLISTS, "behavior_sd_vocab.tsv")

# Defaults only - template variants come in via --narrative-explainer/
# --narrative-listener (oov-shared's run_experiment.sh passes the text from its
# templates/*.json). {word} and {partner} are substituted.
NARRATIVE_EXPLAINER = (
    "You are excited to tell {partner} about '{word}'. You explain to "
    "{partner} what '{word}' means and you use the word '{word}' several times."
)
NARRATIVE_LISTENER = (
    "{partner} tells you about something called '{word}'. You are curious "
    "and ask {partner} questions about it."
)


# CosyVoice clips vs Behavior-SD's level (oov-shared checks/LOG.md, 2026-10-16; = training/prefix.py's
# COSYVOICE_GAIN_DB, not imported so this script stays free of torch)
DEFAULT_PREFIX_GAIN_DB = 5.6
DEFAULT_MULTITURN_TAIL_S = 0.3   # the tail that stops audio EOS at the first frame (docs/prefix_log.md)


def load_prefixes(run_dir, voice):
    """{word: alignment row} for one voice of an oov-shared round-trip run (its `align` stage output)."""
    path = os.path.join(os.path.expandvars(run_dir), "alignment.jsonl")
    if not os.path.exists(path):
        sys.exit(f"error: {path} not found - run oov-shared's `asr_roundtrip.py align` on that run first")
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return {r["word"]: r for r in rows if r["voice"] == voice}


def clip_path(run_dir, row):
    """Path of a clip of a round-trip run: "$WS/..." is kept as given (expanded at generation, so the config works
    in and outside the container); anything else is made absolute, so generation does not depend on the cwd."""
    return os.path.join(run_dir if "$" in run_dir else os.path.abspath(run_dir), row["wav"])


def make_prefix(row, args):
    """The config's "prefix" block (read by training/inference_example.py)."""
    return {
        "speaker": "speaker1",
        "wav": clip_path(args.prefix_run_dir, row),
        "text": row["text"],
        "words": [{"word": w["word"], "start": w["start"], "dur": w["dur"]} for w in row["words"]],
        "target_idx": row["target_idx"],
        "gain_db": args.prefix_gain_db,
        **({"tail_s": args.prefix_tail_s} if args.prefix_tail_s is not None else {}),
        "source": {"run": os.path.basename(os.path.normpath(args.prefix_run_dir)), "voice": row["voice"],
                   "aligner": row["aligner"]},
    }


def make_turn(row, run_dir, speaker, gain_db, role):
    return {
        "speaker": speaker, "role": role, "word": row["word"],
        "wav": clip_path(run_dir, row), "text": row["text"],
        "words": [{"word": w["word"], "start": w["start"], "dur": w["dur"]} for w in row["words"]],
        "gain_db": gain_db,
    }


def make_multiturn_prefix(row, demos, args):
    """Prefix block with turns: each demo = speaker1's request + speaker2's reply, then the target's request."""
    turns = []
    for req, rep in demos:
        turns.append(make_turn(req, args.demo_request_run_dir, "speaker1", args.prefix_gain_db, "demo_request"))
        turns.append(make_turn(rep, args.demo_reply_run_dir, "speaker2", args.demo_reply_gain_db, "demo_reply"))
    last = make_turn(row, args.prefix_run_dir, "speaker1", args.prefix_gain_db, "request")
    last["target_idx"] = row["target_idx"]
    turns.append(last)
    return {
        "turns": turns, "gap_s": args.prefix_gap_s,
        "tail_s": args.prefix_tail_s if args.prefix_tail_s is not None else DEFAULT_MULTITURN_TAIL_S,
        "source": {"run": os.path.basename(os.path.normpath(args.prefix_run_dir)), "voice": row["voice"],
                   "aligner": row["aligner"], "demo_requests": os.path.basename(os.path.normpath(args.demo_request_run_dir)),
                   "demo_replies": os.path.basename(os.path.normpath(args.demo_reply_run_dir))},
    }


def pick_demos(word, rep, pool, n, seed):
    """n distinct demo words from pool (sorted, without the target), deterministic per (seed, word, rep)."""
    candidates = sorted(w for w in pool if w != word)
    if len(candidates) < n:
        raise SystemExit(f"error: only {len(candidates)} demo words available for {word!r}, --prefix-demos is {n}")
    return random.Random(f"{seed}|{word}|{rep}").sample(candidates, n)


def make_config(word, speaker1, speaker2, rep, args, speech_count, narrative_count, band, prefix_row=None,
                demos=None):
    config = {
        "meta": {
            "word": word,
            "band": band,
            "repeat": rep,
            "speech_count": speech_count,
            "narrative_count": narrative_count,
        },
        "speaker1": {
            "name": speaker1,
            "narrative": args.narrative_explainer.format(partner=speaker2, word=word),
            "starts": True,
            "backchannels": args.backchannels,
            "interruptions": args.interruptions,
        },
        "speaker2": {
            "name": speaker2,
            "narrative": args.narrative_listener.format(partner=speaker1, word=word),
            "starts": False,
            "backchannels": args.backchannels,
            "interruptions": args.interruptions,
        },
        "output_dir": args.audio_dir,
        "output_file": f"{word}_r{rep}.wav",
        "max_length": args.max_length,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        **({"prompt_format": "training"} if args.prompt_format == "training" else {}),
    }
    if prefix_row is not None:
        if demos:
            config["prefix"] = make_multiturn_prefix(prefix_row, demos, args)
            config["meta"]["demos"] = [req["word"] for req, _ in demos]
        else:
            config["prefix"] = make_prefix(prefix_row, args)
        config["meta"]["prefix"] = config["prefix"]["source"]["run"]
        config["output_file"] = f"{word}_r{rep}_pfx.wav"
    return config


def load_words_file(path: str) -> list[str]:
    """One word per line (# comments ok), or the shared oov_wordlist.csv -
    detected by extension so the same --words-file flag accepts either."""
    if path.endswith(".csv"):
        with open(path, newline="") as f:
            return [row["word"].strip().lower() for row in csv.DictReader(f) if row["word"].strip()]
    with open(path) as f:
        return [
            line.strip().lower()
            for line in f
            if line.strip() and not line.startswith("#")
        ]


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
    parser.add_argument("--narrative-explainer", default=NARRATIVE_EXPLAINER, dest="narrative_explainer")
    parser.add_argument("--narrative-listener", default=NARRATIVE_LISTENER, dest="narrative_listener")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=40, dest="top_k")
    parser.add_argument("--top-p", type=float, default=1.0, dest="top_p")
    parser.add_argument("--prefix-run-dir", dest="prefix_run_dir",
                        help="oov-shared round-trip run dir with alignment.jsonl: give speaker 1 a fixed first turn "
                             "(that run's clip of the word); quote '$WS/...' to keep it unexpanded in the configs")
    parser.add_argument("--prefix-voice", default="tom", dest="prefix_voice")
    parser.add_argument("--prefix-gain-db", type=float, default=DEFAULT_PREFIX_GAIN_DB, dest="prefix_gain_db")
    parser.add_argument("--prefix-tail-s", type=float, default=None, dest="prefix_tail_s",
                        help="trim the clip's trailing silence and end it with exactly this much digital silence "
                             "(training/prefix.py set_tail); default: the clip as it is")
    parser.add_argument("--prefix-demos", type=int, default=0, dest="prefix_demos",
                        help="number of demonstration exchanges (speaker1's request for a pool word, speaker 2 says "
                             "it) before the target request; 0 = the single-turn prefix. Needs the --demo-* run dirs")
    parser.add_argument("--demo-request-run-dir", dest="demo_request_run_dir",
                        help="round-trip run (alignment.jsonl) with speaker1's requests for the demo words")
    parser.add_argument("--demo-reply-run-dir", dest="demo_reply_run_dir",
                        help="round-trip run (alignment.jsonl) with speaker 2's replies for the demo words")
    parser.add_argument("--demo-reply-voice", default="gweneth", dest="demo_reply_voice",
                        help="voice of the replies; must be --speakers' second name (the model's voice for speaker 2)")
    parser.add_argument("--demo-pool-file", nargs="+", dest="demo_pool_file",
                        help="word lists (csv with a word column, or one word per line), e.g. oov-shared's "
                             "prefix_gate_p5_demo_request.csv and ..._reply.csv: only words in all of them")
    parser.add_argument("--demo-exclude", nargs="+", default=[], dest="demo_exclude", help="words never used as demos")
    parser.add_argument("--demo-seed", type=int, default=0, dest="demo_seed")
    parser.add_argument("--demo-reply-gain-db", type=float, default=0.0, dest="demo_reply_gain_db",
                        help="gain of the replies (Gweneth's CosyVoice clips are already at Behavior-SD's level)")
    parser.add_argument("--prompt-format", choices=["inference", "training"], default="inference", dest="prompt_format",
                        help="'training': the narrative is written as '- <narrative>' like training's build_prompt (for fine-tuned models)")
    parser.add_argument("--prefix-gap-s", type=float, default=0.5, dest="prefix_gap_s",
                        help="between a turn's last word and the next turn's first word (Behavior-SD median 0.49 s)")
    args = parser.parse_args()
    # the explainer must be told the word; the listener may not be (e.g. the prefix "listen" variant, where
    # speaker 2 only hears the word in speaker 1's prefix: docs/oov_experiment.md "Prefix experiment")
    if "{word}" not in args.narrative_explainer and not args.prefix_run_dir:
        parser.error("--narrative-explainer must contain {word} (unless there is a prefix: the word is then in the fixed first turn)")

    words = [w.lower() for w in args.words]
    if args.words_file:
        words += load_words_file(args.words_file)
    if not words:
        parser.error("No words given (use --words or --words-file).")

    speaker1, speaker2 = args.speakers.split(",")
    prefixes = None
    if args.prefix_run_dir:
        prefixes = load_prefixes(args.prefix_run_dir, args.prefix_voice)
        missing = [w for w in words if w not in prefixes]
        if missing:
            parser.error(f"no {args.prefix_voice!r} clip aligned in {args.prefix_run_dir} for: {', '.join(missing)} "
                         "(gate the word list first: oov-shared checks/prefix_gate.py)")
    demo_pool = None
    if args.prefix_demos:
        if not (prefixes and args.demo_request_run_dir and args.demo_reply_run_dir):
            parser.error("--prefix-demos needs --prefix-run-dir, --demo-request-run-dir and --demo-reply-run-dir")
        if speaker2.lower() != args.demo_reply_voice.lower():
            parser.error(f"the replies are in the voice {args.demo_reply_voice!r}, so speaker 2 must be named "
                         f"{args.demo_reply_voice.capitalize()!r} (--speakers), not {speaker2!r}")
        requests = load_prefixes(args.demo_request_run_dir, args.prefix_voice)
        replies = load_prefixes(args.demo_reply_run_dir, args.demo_reply_voice)
        demo_pool = set(requests) & set(replies)
        for path in args.demo_pool_file or []:
            demo_pool &= set(load_words_file(path))
        demo_pool -= {w.lower() for w in args.demo_exclude}
        print(f"demo pool ({len(demo_pool)}): {', '.join(sorted(demo_pool))}\n")
    vocab = load_vocab(args.vocab)
    os.makedirs(args.outdir, exist_ok=True)

    manifest_path = os.path.join(args.outdir, "manifest.tsv")
    with open(manifest_path, "w") as manifest:
        manifest.write(
            "word\tspeech_count\tnarrative_count\tband\trepeat\tconfig\taudio\tprefix\n"
        )
        for word in words:
            speech_count, narrative_count = vocab.get(word, (0, 0))
            band = get_band(speech_count, narrative_count)
            for rep in range(1, args.repeats + 1):
                demos = None
                if demo_pool is not None:
                    demos = [(requests[d], replies[d]) for d in pick_demos(word, rep, demo_pool, args.prefix_demos, args.demo_seed)]
                config = make_config(
                    word, speaker1, speaker2, rep, args,
                    speech_count, narrative_count, band,
                    prefix_row=prefixes[word] if prefixes else None,
                    demos=demos,
                )
                config_path = os.path.join(args.outdir, f"{word}_r{rep}.json")
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=2)
                audio_path = os.path.join(args.audio_dir, config["output_file"])
                manifest.write(
                    f"{word}\t{speech_count}\t{narrative_count}\t{band}"
                    f"\t{rep}\t{config_path}\t{audio_path}"
                    f"\t{(config['prefix'].get('wav') or config['prefix']['turns'][-1]['wav']) if prefixes else ''}\n"
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

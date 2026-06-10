"""Build a word-frequency table of the Behavior-SD data F-Actor was trained on.

Reads only the text columns of maikezu/f-actor-behavior-sd-nanocodec over HTTP
(DuckDB column projection; no full dataset download) and counts words in

- speech: utterances[].tts_text and utterances[].backchannels[].tts_text
  (what the model heard / was trained to say), and
- narrative: the instruction-side narrative field.

Usage:
    python scripts/analysis/build_behavior_sd_vocab.py [--split train] \
        [--out outputs/behavior_sd_vocab.tsv]
    python scripts/analysis/build_behavior_sd_vocab.py --lookup photolithography mitosis
"""

import argparse
import json
import os
import sys
import urllib.request

DATASET = "maikezu/f-actor-behavior-sd-nanocodec"
WORD_PATTERN = "[a-z]+(?:'[a-z]+)?"
DEFAULT_OUT = "scripts/analysis/behavior_sd_vocab.tsv"


def get_parquet_urls(split):
    url = f"https://huggingface.co/api/datasets/{DATASET}/parquet/default/{split}"
    with urllib.request.urlopen(url) as r:
        return json.load(r)


def build_table(split, out_file):
    import duckdb

    urls = get_parquet_urls(split)
    print(f"Counting words in {len(urls)} parquet files of {DATASET} [{split}] ...")

    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")

    sql_pattern = WORD_PATTERN.replace("'", "''")
    query = f"""
    WITH utts AS (
        SELECT unnest(utterances) AS u FROM read_parquet($urls)
    ),
    speech_texts AS (
        SELECT u.tts_text AS t FROM utts
        UNION ALL
        SELECT b.tts_text FROM (SELECT unnest(u.backchannels) AS b FROM utts)
    ),
    speech_words AS (
        SELECT unnest(regexp_extract_all(lower(t), '{sql_pattern}')) AS word
        FROM speech_texts
    ),
    narrative_words AS (
        SELECT unnest(regexp_extract_all(lower(narrative), '{sql_pattern}')) AS word
        FROM read_parquet($urls)
    ),
    speech_counts AS (
        SELECT word, count(*) AS speech_count FROM speech_words GROUP BY word
    ),
    narrative_counts AS (
        SELECT word, count(*) AS narrative_count FROM narrative_words GROUP BY word
    )
    SELECT
        coalesce(s.word, n.word) AS word,
        coalesce(s.speech_count, 0) AS speech_count,
        coalesce(n.narrative_count, 0) AS narrative_count
    FROM speech_counts s
    FULL OUTER JOIN narrative_counts n ON s.word = n.word
    ORDER BY speech_count DESC, narrative_count DESC
    """
    rows = con.execute(query, {"urls": urls}).fetchall()

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        f.write("word\tspeech_count\tnarrative_count\n")
        for word, speech_count, narrative_count in rows:
            f.write(f"{word}\t{speech_count}\t{narrative_count}\n")

    total_speech = sum(r[1] for r in rows)
    print(f"{len(rows)} unique words, {total_speech} spoken word tokens")
    print(f"Saved to {out_file}")


def lookup(words, out_file):
    if not os.path.isfile(out_file):
        sys.exit(f"No vocab table at {out_file}; build it first (run without --lookup).")

    table = {}
    with open(out_file) as f:
        next(f)
        for line in f:
            word, speech_count, narrative_count = line.rstrip("\n").split("\t")
            table[word] = (int(speech_count), int(narrative_count))

    print(f"{'word':<25}{'speech':>10}{'narrative':>12}")
    for word in words:
        speech_count, narrative_count = table.get(word.lower(), (0, 0))
        marker = "  <-- OOV" if speech_count == 0 else ""
        print(f"{word.lower():<25}{speech_count:>10}{narrative_count:>12}{marker}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument(
        "--lookup", nargs="+", help="Look up words in an existing table."
    )
    args = parser.parse_args()

    if args.lookup:
        lookup(args.lookup, args.out)
    else:
        build_table(args.split, args.out)


if __name__ == "__main__":
    main()

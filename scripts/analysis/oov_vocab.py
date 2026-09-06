"""Shared word-frequency lookup for the OOV experiment.

Extracted out of generate_oov_configs.py so both it and
build_oov_wordlist.py agree on one band-threshold definition instead of
maintaining two copies.
"""


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

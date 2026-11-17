"""Forced text decoding: put a chosen word into one speaker's text stream while `DSUModel.generate` runs, and let the audio follow.

The text stream leads the audio by n_delay_audio_stream = 2 frames, so the tokens of a word that starts at text frame s are inputs
before the audio of that word is sampled (at frame s + 2): forcing text never asks the audio to look ahead. Only the word's tokens are
forced; the pads after it ([WORD_PAD] = how long the word lasts, [UTTERANCE_PAD] / [SILENCE_PAD] = pause or end of the utterance) and
everything else stay sampled, so the model decides the word's length. Stopping is unchanged: an audio EOS can end the run before or
right after the injection; the record says which (`ended_by`, `eos_by`, `happened`, `frames_after_last_token`).

Rule ("force_text" in a generation config, one dict or a list of dicts):

    {"speaker": "speaker2", "n": 2, "word": "water"}

    speaker  "speaker1" | "speaker2" (default "speaker2"): whose text stream; speaker1 = row 0, speaker2 = row 1 of talk_to_itself
    n        the n-th word (1-based) that speaker starts in the *generated* frames (a prefix's words do not count): when the model
             starts its n-th word, that word is replaced by `word`. For a reply "Okay, X." the target is n = 2.
    word     the word to force (default: the config's meta.word)

A word starts at a sampled token that is not a text-stream pad / special token and that follows a pad, or that has a leading space
(create_text_stream lets a word's tokens follow the previous word's directly when they did not fit its frames). The forced tokens are
written as create_text_stream writes a word: lowercased, with a leading space unless the word opens an utterance (the frame before it
is not [WORD_PAD], [UTTERANCE_PAD] or a word token). Tokens are written only into the frame being generated: earlier frames are already
in the KV cache, so rewriting them would silently desync the model's context.

Backchannels and interrupts (add_bc_token / add_interrupt_token, on in the released checkpoint), as create_text_stream writes them:
  - a backchannel is `[BC]` (one added token) + " yeah" ...: its words are not counted and never replaced (a forced word in a
    "yeah" is not what the experiment asks); the backchannel lasts until the next [SILENCE_PAD]. Counted in the record as
    `backchannel_words_skipped`.
  - an interrupting turn starts with "[INTER]", which is NOT a token of the vocabulary: it is the plain pieces "[", "INTER", "]",
    followed by " wait" ... . The marker is not a word; the turn's words count as normal words (with a leading space after the marker).
  Any "[...]" written out of plain pieces is treated as such a marker; one whose text starts with "[BC" (the counting tokens
  [BC1] ... of add_counting_tokens) opens a backchannel.

Self-test: scripts/analysis/forced_text_selftest.py.
"""

from dataclasses import dataclass, field

from special_tokens import BC_TOKEN, SILENCE_PAD, TEXT_STREAM_TOKENS, UTTERANCE_PAD, WORD_PAD

SPEAKER_ROW = {"speaker1": 0, "speaker2": 1}
RULE_KEYS = {"speaker", "n", "word"}


def word_tokens(tokenizer, word: str, utterance_initial: bool) -> list[int]:
    """The token ids create_text_stream writes for a normal (not backchannel / interrupt) word, word_alignment on."""
    w = word.lower() if utterance_initial else " " + word.lower()
    return tokenizer(w, add_special_tokens=False)["input_ids"]


@dataclass
class Rule:
    speaker: str
    n: int
    word: str
    # filled in during generation
    happened: bool = False
    step: int | None = None
    tokens: list[int] = field(default_factory=list)
    utterance_initial: bool | None = None
    replaced: str | None = None          # the token the model had sampled where the forced word starts (what it was about to say)


def parse_rules(spec, default_word=None) -> list[Rule]:
    """The config's "force_text" (dict or list of dicts) -> rules. Raises ValueError on a bad spec."""
    specs = [spec] if isinstance(spec, dict) else list(spec)
    if not specs:
        raise ValueError("force_text: empty")
    rules = []
    for s in specs:
        if not isinstance(s, dict):
            raise ValueError(f"force_text: each rule is a dict, got {s!r}")
        unknown = set(s) - RULE_KEYS
        if unknown:
            raise ValueError(f"force_text: unknown key(s) {sorted(unknown)} (known: {sorted(RULE_KEYS)})")
        speaker = s.get("speaker", "speaker2")
        if speaker not in SPEAKER_ROW:
            raise ValueError(f"force_text: speaker must be one of {sorted(SPEAKER_ROW)}, got {speaker!r}")
        n = s.get("n")
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise ValueError(f"force_text: n must be an int >= 1 (1 = the first generated word), got {n!r}")
        word = s.get("word") or default_word
        if not word or not str(word).strip() or " " in str(word).strip():
            raise ValueError(f"force_text: need one word (rule 'word' or the config's meta.word), got {word!r}")
        rules.append(Rule(speaker=speaker, n=n, word=str(word).strip()))
    seen = [(r.speaker, r.n) for r in rules]
    if len(set(seen)) != len(seen):
        raise ValueError(f"force_text: two rules for the same speaker and n: {seen}")
    return rules


class ForcedText:
    """The `text_override` hook of DSUModel.generate: called once per generated frame with the text stream 0 of every row up to and
    including the frame just sampled, returns {row: token id} to write into that frame instead."""

    def __init__(self, rules: list[Rule], tokenizer):
        self.rules = rules
        self.tokenizer = tokenizer
        non_word = set(tokenizer.convert_tokens_to_ids(TEXT_STREAM_TOKENS)) | set(tokenizer.all_special_ids)
        non_word.add(tokenizer.pad_token_id)
        self.non_word = {i for i in non_word if i is not None}
        self.in_utterance = set(tokenizer.convert_tokens_to_ids([WORD_PAD, UTTERANCE_PAD]))
        self.silence_id = tokenizer.convert_tokens_to_ids(SILENCE_PAD)
        self.bc_id = tokenizer.convert_tokens_to_ids(BC_TOKEN)
        # per row: words started, tokens still to force, rules by n, inside a backchannel, text of a "[...]" marker being read (or None)
        self._row = {}
        for r in rules:
            st = self._row.setdefault(SPEAKER_ROW[r.speaker], {"count": 0, "pending": [], "rules": {}, "in_bc": False,
                                                               "marker": None, "bc_skipped": 0})
            st["rules"][r.n] = r

    def is_word(self, tok: int) -> bool:
        return tok not in self.non_word

    def starts_word(self, tok: int, prev: int | None) -> bool:
        if not self.is_word(tok):
            return False
        if prev is None or not self.is_word(prev):
            return True
        return self.tokenizer.convert_ids_to_tokens(tok).startswith("Ġ")   # byte-level BPE: Ġ = leading space

    def utterance_initial(self, prev: int | None) -> bool:
        return prev is None or not (prev in self.in_utterance or self.is_word(prev))

    def __call__(self, step: int, text) -> dict[int, int]:
        """text: LongTensor [rows, step + 1], text stream 0; column `step` holds the tokens just sampled."""
        out = {}
        for row, st in self._row.items():
            if row >= text.shape[0]:
                raise ValueError(f"force_text: row {row} does not exist (speaker2 needs talk_to_itself)")
            if st["pending"]:                       # the rest of a forced word; the sampled token is dropped, not counted
                out[row] = st["pending"].pop(0)
                continue
            tok = int(text[row, step])
            prev = int(text[row, step - 1]) if step > 0 else None
            if tok == self.bc_id:                   # [BC]: a backchannel until the next silence
                st["in_bc"], st["marker"] = True, None
                continue
            if tok == self.silence_id:
                st["in_bc"], st["marker"] = False, None
                continue
            if st["marker"] is not None:            # inside "[...]" written as plain pieces ("[INTER]", "[BC1]" ...)
                st["marker"] += self.tokenizer.decode([tok])
                if "]" in st["marker"]:
                    st["in_bc"], st["marker"] = st["marker"].lstrip().startswith("[BC"), None
                continue
            if not self.starts_word(tok, prev):
                continue
            if self.tokenizer.decode([tok]).strip() == "[":
                st["marker"] = "["
                continue
            if st["in_bc"]:
                st["bc_skipped"] += 1
                continue
            st["count"] += 1
            rule = st["rules"].get(st["count"])
            if rule is None:
                continue
            initial = self.utterance_initial(prev)
            tokens = word_tokens(self.tokenizer, rule.word, initial)
            rule.happened, rule.step, rule.tokens, rule.utterance_initial = True, step, tokens, initial
            rule.replaced = self.tokenizer.decode([tok])
            out[row] = tokens[0]
            st["pending"] = list(tokens[1:])
        return out

    def record(self, n_total: int, n_prefix: int, n_delay: int, frame_s: float) -> list[dict]:
        """One dict per rule for the run record.

        Timing: the word's first token is at grid frame `step`, its audio is expected from grid frame step + n_delay. The decoded channel
        files drop the n_delay delay frames, so that audio is at decoded frame `step`: audio_s = step * frame_s. A prefixed run's _cont
        files start at decoded frame n_prefix - n_delay: cont_s = (step - n_prefix + n_delay) * frame_s.
        frames_after_last_token: grid frames after the last forced token (the word's audio needs n_delay of them plus its length)."""
        out = []
        for r in self.rules:
            d = {"speaker": r.speaker, "n": r.n, "word": r.word, "happened": r.happened,
                 "backchannel_words_skipped": self._row[SPEAKER_ROW[r.speaker]]["bc_skipped"]}   # by the end of the run
            if r.happened:
                d.update(step=r.step, tokens=r.tokens, utterance_initial=r.utterance_initial, replaced=r.replaced,
                         audio_s=round(r.step * frame_s, 3), frames_after_last_token=n_total - (r.step + len(r.tokens)))
                if n_prefix:
                    d["cont_s"] = round((r.step - n_prefix + n_delay) * frame_s, 3)
            out.append(d)
        return out

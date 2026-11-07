"""Does the fine-tuned model reproduce the training rows when it is fed the whole row (teacher forcing, greedy argmax)?

The training loss is an average over every frame, and most frames are silence, so a loss of ~0.01 does not show that the reply was
learned. This feeds each row through the same data pipeline as training (data.py load_data, finetune_rows.sh's flags) and the model's own
forward, captures the logits and labels the loss sees (system streams only, as eval mode scores them), and reports, per row:

  text   the word tokens of the system text stream (the reply's words, without the prompt): how many are predicted right, and the
         predicted vs target words ("okay doubled" ...)
  audio  top-1 accuracy of the 4 system audio heads on the frames where the reply speaks (full frame = all 4 heads right) and on
         silent frames; "early EOS" = frames where head 0 predicts the audio EOS but the label is not the end of the row

    FACTOR_MODEL_DIR=<run>/best_model $FACTOR_PYTHON scripts/analysis/teacher_forced_check.py --rows-dir $WS/finetune/overfit8 --split train
Inside the container, from the repo root. One model load; fast (one forward per row).

Unseen words (does the model copy at all, or only recall trained words?):

    FACTOR_MODEL_DIR=<run>/best_model $FACTOR_PYTHON scripts/analysis/teacher_forced_check.py --rows-dir $WS/finetune/<name> \
        --split validation --narrative orig none swap

--narrative: the replier's narrative contains the word ("... repeat the word 'X'"), so a right word under `orig` may be read from
the prompt, not heard in the request. `none` replaces the word by "a word" (the audio request is the only source); `swap` puts the next
row's word in the narrative (audio and prompt disagree: which one does the reply follow?). Per run: exact reply match, reply word
(last word) accuracy, and for the misses: how often the predicted word is a training word, the narrative's word, or neither.
"""
import argparse
import logging
import os
import re
import sys
import tempfile
from types import SimpleNamespace

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (os.path.join(_REPO, "training"), os.path.join(_REPO, "training", "inference_audio")):
    sys.path.insert(0, p)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from datasets import load_dataset  # noqa: E402
from special_tokens import TEXT_STREAM_TOKENS  # noqa: E402


def with_narrative(rows_dir, mode, tmp):
    """A copy of rows_dir whose narratives mention no word (none) or the next validation/train row's word (swap); returns (dir, narrative word per row)."""
    out = os.path.join(tmp, mode)
    os.makedirs(out, exist_ok=True)
    words = {}
    for split in ("train", "validation"):
        if not os.path.exists(os.path.join(rows_dir, f"{split}.parquet")):
            continue
        ds = load_dataset(rows_dir, split=split)
        names = ds["file_name"]
        other = {w: names[(i + 1) % len(names)] for i, w in enumerate(names)}
        def edit(ex):
            repl = "the word 'WORD'" if mode == "swap" else "a word"
            for col in ("new_narrative_s1", "new_narrative_s2"):
                ex[col] = re.sub(r"the word '[^']*'", repl.replace("WORD", other[ex["file_name"]]), ex[col])
            return ex
        ds.map(edit).to_parquet(os.path.join(out, f"{split}.parquet"))
        words[split] = [other[w] if mode == "swap" else None for w in names]
    return out, words


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rows-dir", required=True)
    ap.add_argument("--split", default="train", choices=["train", "validation"])
    ap.add_argument("--system-speaker", type=int, default=1, help="as in finetune_rows.sh")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--narrative", nargs="+", default=["orig"], choices=["orig", "none", "swap"], help="narrative variants to run (one model load)")
    args = ap.parse_args()

    from inference_example import load_model

    model, tokenizer = load_model()   # FACTOR_MODEL_DIR or the released checkpoint; eval mode, so the loss scores the system streams only
    train_words = set(load_dataset(args.rows_dir, split="train")["file_name"])
    with tempfile.TemporaryDirectory() as tmp:
        for mode in args.narrative:
            rows_dir, narr = (args.rows_dir, {}) if mode == "orig" else with_narrative(args.rows_dir, mode, tmp)
            print(f"\n######## narrative: {mode}  split: {args.split}")
            _run(model, tokenizer, rows_dir, args, narr.get(args.split), train_words)


def _run(model, tokenizer, rows_dir, args, narr_words, train_words):
    from data import load_data
    dargs = SimpleNamespace(
        model=SimpleNamespace(max_length=2048, num_dsus=4, text_stream=True, multi_text_stream=False),
        data=SimpleNamespace(speech_path=rows_dir, remove_start_silence=False, n_delay_audio_stream=2, n_delay_text_stream=0,
                             word_alignment=True, add_bc_token=True, add_interrupt_token=True, add_counting_tokens=False,
                             use_system_narrative=True, system_speaker=args.system_speaker, debug=False, train_on_subset=None,
                             preprocessing_num_workers=1),
        train=SimpleNamespace(train_batch_size=1, eval_batch_size=1, gradient_accumulation_steps=1))
    train, val, collator = load_data(dargs.model, dargs.data, dargs.train, model.audio_delay_id, logging.getLogger("tf"), tokenizer)
    data = train if args.split == "train" else val
    names = load_dataset(rows_dir, split=args.split)["file_name"]
    if len(names) != len(data):
        print(f"warning: {len(names)} rows but {len(data)} examples (some skipped): names may not line up")
    pad, eos = model.pad_token_id, model.audio_eos_id
    special = {i for i in tokenizer.convert_tokens_to_ids(TEXT_STREAM_TOKENS) if i is not None}
    print(f"pad id model {pad} tokenizer {tokenizer.pad_token_id}; audio eos {eos}; {len(data)} {args.split} examples\n")

    cap = {}
    orig_get = model.get_input_embeds_and_labels
    def get_wrapped(*a, **k):
        out = orig_get(*a, **k)
        cap["dsu_labels"], cap["ts_labels"] = out[2], out[3]
        return out
    model.get_input_embeds_and_labels = get_wrapped   # restored at the end of _run (one model, several runs)
    orig_ce = F.cross_entropy
    def ce_wrapped(logits, target, *a, **k):
        cap.setdefault("logits", []).append(logits.detach())
        return orig_ce(logits, target, *a, **k)

    print(f"{'word':<12}{'text ok':>8}  {'target words':<24}{'predicted words':<30}{'audio full':>11}{'head acc':>10}{'silent acc':>11}{'early EOS':>10}")
    tot = dict(tw=0, tc=0, af=0, an=0, sf=0, sn=0, ee=0)
    n_rows = exact = word_ok = 0
    miss = dict(train_word=0, narrative_word=0, other=0)
    for i in range(len(data) if not args.limit else min(args.limit, len(data))):
        batch = collator([dict(data[i])])
        batch = {k: (v.to(model.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        cap.clear()
        F.cross_entropy = ce_wrapped
        try:
            with torch.no_grad():
                model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], dsu_ids=batch["dsu_ids"],
                      text_stream_ids=batch["text_stream_ids"], spk_emb=batch["spk_emb"])
        finally:
            F.cross_entropy = orig_ce
        a_log, t_log = cap["logits"][0], cap["logits"][1]
        shift = lambda lab: F.pad(lab, (0, 1), value=pad)[..., 1:]
        lab_a = shift(cap["dsu_labels"])[0, : cap["dsu_labels"].shape[1] // 2]            # [4, La], system heads
        lab_t = shift(cap["ts_labels"])[0, 0]                                                # [Lf]
        pred_a = a_log.reshape(lab_a.shape[0], lab_a.shape[1], -1).argmax(-1)                # [4, La]
        pred_t = t_log.reshape(lab_t.shape[0], -1).argmax(-1)                                # [Lf]

        valid = lab_a != pad
        not_end = valid & (lab_a != eos)
        silence = torch.stack([torch.mode(lab_a[h][not_end[h]]).values for h in range(lab_a.shape[0])])
        active = ((lab_a != silence[:, None]) & not_end).any(0)                              # frames where the system speaks
        quiet = not_end.all(0) & ~active
        right = pred_a == lab_a
        af, an = int((right.all(0) & active).sum()), int(active.sum())
        sf, sn = int((right.all(0) & quiet).sum()), int(quiet.sum())
        head = float(right[:, active].float().mean()) if an else float("nan")
        early = int(((pred_a[0] == eos) & (lab_a[0] != eos) & valid[0]).sum())

        off = lab_t.shape[0] - cap["dsu_labels"].shape[2]                                    # skip speaker slot + prompt
        pos = [j for j in range(max(off - 1, 0), lab_t.shape[0]) if int(lab_t[j]) != pad and int(lab_t[j]) not in special]
        tc = sum(int(pred_t[j] == lab_t[j]) for j in pos)
        tgt = tokenizer.decode([int(lab_t[j]) for j in pos]).strip()
        prd = tokenizer.decode([int(pred_t[j]) for j in pos]).strip()
        name = names[i] if i < len(names) else str(i)
        print(f"{name:<12}{f'{tc}/{len(pos)}':>8}  {tgt:<24.24}{prd:<30.30}{f'{af}/{an}':>11}{head:>10.2f}{f'{sf}/{sn}':>11}{early:>10}")
        norm = lambda t: re.sub(r"[^a-z' ]", "", t.lower()).split()
        tw, pw = norm(tgt), norm(prd)
        n_rows += 1
        exact += tw == pw
        if tw and pw and tw[-1] == pw[-1]:
            word_ok += 1
        elif pw:
            if narr_words and narr_words[i] == pw[-1]:
                miss["narrative_word"] += 1
            elif pw[-1] in train_words:
                miss["train_word"] += 1
            else:
                miss["other"] += 1
        for k, v in dict(tw=len(pos), tc=tc, af=af, an=an, sf=sf, sn=sn, ee=early).items():
            tot[k] += v
    print(f"\nTOTAL word tokens {tot['tc']}/{tot['tw']}; speaking frames fully right {tot['af']}/{tot['an']}; "
          f"silent frames fully right {tot['sf']}/{tot['sn']}; early EOS predictions {tot['ee']}")
    model.get_input_embeds_and_labels = orig_get
    print(f"REPLY exact {exact}/{n_rows}; reply word (last word) right {word_ok}/{n_rows}; "
          f"wrong word was: a training word {miss['train_word']}, the narrative's word {miss['narrative_word']}, other {miss['other']}")


if __name__ == "__main__":
    main()

"""Where does the text loss come from? Forward-only breakdown of the text-head cross-entropy by label group.

Groups (labels of the text row of `[spk | prompt | frames]`, as modeling_dsu builds them):
  prompt          template + narrative tokens (trained as an LM, no audio involved)
  silence_pad     [SILENCE_PAD] frames        utterance_pad  [UTTERANCE_PAD] frames      word_pad  [WORD_PAD] frames
  other_special   [BC], [INTER], [BCn], ...   word           real word tokens
Per group: number of targets, share of targets, mean CE, share of the summed loss. With --grad N the first N rows also get
the L2 norm of d(group_sum / n_targets_total)/d(params), i.e. each group's contribution to the actual gradient.

Same CLI flags as training/finetune.py (data + model flags; --model_id is the checkpoint dir), plus the ones below.
Run it once per checkpoint (base model, tuned model) with the data flags of the fine-tune:

    python scripts/analysis/text_loss_breakdown.py <finetune.py flags, e.g. from scripts/train/finetune_rows.sh> \\
        --model_id hf_models/f_actor --split validation --grad 8 --out outputs/breakdown_base.json
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
for p in (_REPO / "training", _REPO / "training" / "inference_audio"):
    sys.path.insert(0, str(p))

GROUPS = ["prompt", "silence_pad", "utterance_pad", "word_pad", "other_special", "word"]


def label_groups(
    labels: torch.Tensor,  # [N] unshifted text labels of one row: spk pad | prompt | frames
    n_prompt: int,  # real prompt tokens P
    pad_id: int,
    special: dict[str, int | set[int]],  # silence_pad / utterance_pad / word_pad -> id, other_special -> set of ids
) -> dict[str, torch.Tensor]:  # group -> bool [N] over label positions (a position j is predicted from logits j-1)
    pos = torch.arange(labels.numel(), device=labels.device)
    real = labels != pad_id
    in_prompt = (pos >= 1) & (pos < 1 + n_prompt)
    frame = real & ~in_prompt
    out = {"prompt": real & in_prompt}
    for name in ("silence_pad", "utterance_pad", "word_pad"):
        out[name] = frame & (labels == special[name])
    other = torch.isin(labels, torch.tensor(sorted(special["other_special"]), device=labels.device))
    out["other_special"] = frame & other
    out["word"] = frame & ~sum(out[g] for g in ("silence_pad", "utterance_pad", "word_pad", "other_special")).bool()
    return out


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--split", default="validation", choices=["train", "validation"])
    ap.add_argument("--max_rows", type=int, default=0, help="0: all rows of the split")
    ap.add_argument("--grad", type=int, default=0, help="also compute per-group gradient norms on the first N rows")
    ap.add_argument("--out", default=None, help="write the summary as json")
    mine, rest = ap.parse_known_args()
    sys.argv = [sys.argv[0]] + rest

    from arguments.parse_arguments import parse_args
    from model import load_model
    from data import load_data
    from special_tokens import SILENCE_PAD, UTTERANCE_PAD, WORD_PAD, TEXT_STREAM_TOKENS
    from torch.utils.data import DataLoader

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("breakdown")
    model_args, data_args, training_args = parse_args()
    model_args.mask_prompt_loss = False  # we want the unmasked labels
    if not model_args.text_stream:
        sys.exit("error: needs --text_stream (single text row with prompt labels)")

    model, tokenizer = load_model(model_args, grad_acc_steps=1, logger=logger)
    model.eval()
    train_ds, eval_ds, collator = load_data(
        model_args=model_args, data_args=data_args, training_args=training_args,
        audio_delay_id=model.audio_delay_id, logger=logger, tokenizer=tokenizer,
    )
    ds = eval_ds if mine.split == "validation" else train_ds
    if mine.max_rows:
        ds = ds.select(range(min(mine.max_rows, len(ds))))
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collator)

    ids = tokenizer.convert_tokens_to_ids
    special = {"silence_pad": ids(SILENCE_PAD), "utterance_pad": ids(UTTERANCE_PAD), "word_pad": ids(WORD_PAD)}
    special["other_special"] = {ids(t) for t in TEXT_STREAM_TOKENS} - {special["silence_pad"], special["utterance_pad"], special["word_pad"]}
    pad_id = model.pad_token_id
    params = [p for p in model.parameters() if p.requires_grad]

    tot = {g: {"n": 0, "sum": 0.0} for g in GROUPS}
    gnorm = defaultdict(list)
    check = []  # (our text loss, model's c1_text_loss, model's c1_dsu_loss) per row
    dev = model.device

    for i, batch in enumerate(loader):
        batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.no_grad():
            _, _, _, ts_labels = model.get_input_embeds_and_labels(
                prompt_ids=batch["input_ids"], prompt_att_mask=batch["attention_mask"],
                dsu_ids=batch["dsu_ids"], text_stream_ids=batch["text_stream_ids"], spk_emb=batch["spk_emb"],
            )
        labels = ts_labels[0, 0]  # [N]
        n_prompt = int(batch["attention_mask"][0].sum())

        want_grad = i < mine.grad
        with torch.set_grad_enabled(want_grad):
            out = model(**batch, need_loss=False)
            logits = out["ts_logits"][0, :, 0, :].float()  # [N, V]; logits j-1 predicts label j
        target = labels[1:]
        ce = torch.nn.functional.cross_entropy(logits[:-1], torch.where(target != pad_id, target, 0), reduction="none")
        grp = {g: m[1:] for g, m in label_groups(labels, n_prompt, pad_id, special).items()}
        n_total = int(sum(m.sum() for m in grp.values()))
        row_loss = float(sum((ce * m).sum() for m in grp.values())) / max(n_total, 1)

        for g, m in grp.items():
            tot[g]["n"] += int(m.sum())
            tot[g]["sum"] += float((ce.detach() * m).sum())
        if want_grad:
            for g, m in grp.items():
                if int(m.sum()) == 0:
                    gnorm[g].append(0.0)
                    continue
                gs = torch.autograd.grad((ce * m).sum() / n_total, params, retain_graph=True, allow_unused=True)
                gnorm[g].append(float(torch.sqrt(sum((x.float() ** 2).sum() for x in gs if x is not None))))
            del out, logits, ce

        if i < 3:  # sanity: our group sums reproduce the model's own text loss
            with torch.no_grad():
                ref = model(**batch)
            check.append((row_loss, float(ref["c1_text_loss"]), float(ref["c1_dsu_loss"])))
        logger.info(f"row {i}: P={n_prompt} targets={n_total} text_loss={row_loss:.4f}")

    n_all = sum(v["n"] for v in tot.values())
    s_all = sum(v["sum"] for v in tot.values())
    rows = {}
    print(f"\n{len(loader)} rows ({mine.split}), model_id={model_args.model_id}; pooled over rows (mean of per-row losses differs slightly)")
    print(f"{'group':14s} {'targets':>8s} {'%targets':>9s} {'mean CE':>8s} {'%loss':>7s}" + ("  mean |grad|" if gnorm else ""))
    for g in GROUPS:
        n, s = tot[g]["n"], tot[g]["sum"]
        rows[g] = {"n": n, "pct_targets": 100 * n / max(n_all, 1), "mean_ce": s / max(n, 1), "pct_loss": 100 * s / max(s_all, 1e-12)}
        line = f"{g:14s} {n:8d} {rows[g]['pct_targets']:8.1f}% {rows[g]['mean_ce']:8.3f} {rows[g]['pct_loss']:6.1f}%"
        if gnorm:
            rows[g]["mean_grad_norm"] = sum(gnorm[g]) / len(gnorm[g])
            line += f"  {rows[g]['mean_grad_norm']:.4f}"
        print(line)
    print("\nsanity (ours vs model c1_text_loss; model c1_dsu_loss for scale):")
    for ours, ref, dsu in check:
        print(f"  {ours:.5f} vs {ref:.5f}   audio {dsu:.4f}")
    if mine.out:
        Path(mine.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"model_id": model_args.model_id, "split": mine.split, "groups": rows, "check": check}, open(mine.out, "w"), indent=1)


if __name__ == "__main__":
    main()

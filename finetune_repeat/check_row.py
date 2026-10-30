"""One-row test and tiny overfit test: do the synthetic rows go through the repo's collator and model, and does the loss fall?

    # laptop, forward only (CPU): finite loss for a synthetic and a real row
    .venv/bin/python finetune_repeat/check_row.py --data ../oov-data/finetune/p5_small --steps 0 --n-synthetic 1 --n-real 1
    # cluster, 20 optimizer steps on a fixed batch of 2 synthetic + 2 real rows
    python finetune_repeat/check_row.py --data <dir> --steps 20

Loads the model exactly like training/finetune.py (training/model.py load_model, from --model_id), model.train(), fp32, the repo's
DSUDataCollator. With --steps N it runs N AdamW steps on the SAME batch (no dropout in this model): the loss should fall clearly.
Prints loss, audio loss (c1_dsu_loss) and text loss per step.
"""
import argparse, logging, os, sys, time
from types import SimpleNamespace
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [f"{_ROOT}/training", f"{_ROOT}/training/inference_audio"]
import torch
from datasets import load_from_disk

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--data", required=True, help="dir from make_dataset.py (train/, real_sample/)")
ap.add_argument("--model_id", default=f"{_ROOT}/hf_models/f_actor")
ap.add_argument("--steps", type=int, default=0)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--train", default="all", help="'all' or 'top:N': freeze everything except the audio heads/embeddings, the speaker "
                "projection, the final norm and the last N transformer layers (the laptop has no GPU and 15 GB: full AdamW does not fit)")
ap.add_argument("--n-synthetic", type=int, default=2); ap.add_argument("--n-real", type=int, default=2)
args = ap.parse_args()
logging.basicConfig(level=logging.INFO, format="%(message)s"); logger = logging.getLogger("check_row")

from data_collator_dsu import DSUDataCollator
from model import load_model
margs = SimpleNamespace(model_id=args.model_id, num_dsus=4, text_stream=True, multi_text_stream=False, audio_vocab_size=4032,
                        use_speaker_embedding=True, calc_loss_on_c1_only=False)
model, tok = load_model(margs, grad_acc_steps=1, logger=logger)
model.train()
if args.train != "all":
    import re
    n_top = int(args.train.split(":")[1]); n_layers = model.config.num_hidden_layers
    for name, p in model.named_parameters():
        m = re.match(r"model\.layers\.(\d+)\.", name)
        p.requires_grad = bool(
            name.startswith(("audio_embeds", "dsu_head", "speaker_embed_proj", "model.norm")) or (m and int(m.group(1)) >= n_layers - n_top))
dev = next(model.parameters()).device
print("device", dev, "| trainable params %.0f M of %.0f M" % (sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6,
      sum(p.numel() for p in model.parameters()) / 1e6))

syn = load_from_disk(f"{args.data}/train"); real = load_from_disk(f"{args.data}/real_sample")
feats = [syn[i] for i in range(args.n_synthetic)] + [real[i] for i in range(args.n_real)]
kinds = ["synthetic"] * args.n_synthetic + ["real"] * args.n_real
for k, f in zip(kinds, feats):
    print(f"{k:9s} frames {len(f['dsu_ids'][0])} heads {len(f['dsu_ids'])} text streams {len(f['text_stream_ids'])} "
          f"prompt tokens {sum(f['attention_mask'])} spk_emb {len(f['spk_emb'])}")
batch = DSUDataCollator(tokenizer=tok, mlm=False)([dict(f) for f in feats])
batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
print("batch:", {k: (tuple(v.shape) if torch.is_tensor(v) else f"list[{len(v)}]") for k, v in batch.items()})

def step(train):
    t = time.time()
    out = model(**batch)
    loss = out["loss"]
    if train:
        loss.backward(); opt.step(); opt.zero_grad()
    return float(loss), float(out["c1_dsu_loss"]), float(out["c1_text_loss"]), time.time() - t

if args.steps == 0:
    with torch.no_grad():
        l, a, tx, s = step(False)
    print(f"forward only: loss {l:.4f} (audio {a:.4f}, text {tx:.4f}) in {s:.0f}s | finite: {all(map(torch.isfinite, [torch.tensor(l)]))}")
else:
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    for i in range(args.steps):
        l, a, tx, s = step(True)
        import resource
        print(f"step {i:2d}: loss {l:.4f} (audio {a:.4f}, text {tx:.4f})  {s:.1f}s  peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.1f} GB", flush=True)

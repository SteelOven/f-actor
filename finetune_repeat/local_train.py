"""Laptop-sized fine-tune on the synthetic repeat dialogues: LoRA on every transformer layer (default) or the last N layers only.

    systemd-run --user --scope -p MemoryMax=11G -p MemorySwapMax=0 \\
      .venv/bin/python finetune_repeat/local_train.py --data ../oov-data/finetune/micro --out ../oov-data/finetune/micro_lora \\
        --mode lora:16 --epochs 5 --accum 4

Starts from the released checkpoint (training/model.py load_model, as finetune.py does), model.train(), fp32, one row per forward pass
and --accum rows per optimizer step. --mode lora:R adds rank-R adapters to q/k/v/o/gate/up/down of all layers (peft; get_peft_model
without a task type, so the custom forward is called as is); --mode top:N trains the last N layers fully. --heads train also trains the
audio heads/embeddings and the speaker projection (default freeze: the question is whether the behaviour is learned). The loss is the
repo's (audio on all 8 heads, text on the system's stream). At the end the adapters are merged and the full model + tokenizer saved to
<out>/final (load it with FACTOR_MODEL_DIR=<out>/final, see training/inference_example.py); every --save-every steps the trainable
tensors go to <out>/trainable_step<N>.pt.
"""
import argparse, logging, math, os, random, resource, sys, time
from types import SimpleNamespace
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [f"{_ROOT}/training", f"{_ROOT}/training/inference_audio"]
import torch
from datasets import load_from_disk

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--data", required=True, help="make_dataset.py output (train/, validation/, real_sample/)")
ap.add_argument("--out", required=True)
ap.add_argument("--model_id", default=f"{_ROOT}/hf_models/f_actor")
ap.add_argument("--mode", default="lora:16", help="lora:R or top:N")
ap.add_argument("--heads", default="freeze", choices=["freeze", "train"])
ap.add_argument("--epochs", type=int, default=5)
ap.add_argument("--max-steps", type=int, default=0, help="stop after this many optimizer steps (0 = all epochs)")
ap.add_argument("--accum", type=int, default=4)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--warmup", type=int, default=5)
ap.add_argument("--eval-every", type=int, default=10)
ap.add_argument("--n-eval", type=int, default=4, help="validation rows per evaluation")
ap.add_argument("--save-every", type=int, default=0)
ap.add_argument("--no-save", action="store_true")
ap.add_argument("--grad-ckpt", action="store_true", help="gradient checkpointing: less memory, ~30%% slower")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[logging.FileHandler(f"{args.out}/train.log"), logging.StreamHandler()])
log = logging.getLogger("local_train").info
random.seed(args.seed); torch.manual_seed(args.seed)

from data_collator_dsu import DSUDataCollator
from model import load_model
margs = SimpleNamespace(model_id=args.model_id, num_dsus=4, text_stream=True, multi_text_stream=False, audio_vocab_size=4032,
                        use_speaker_embedding=True, calc_loss_on_c1_only=False)
model, tok = load_model(margs, grad_acc_steps=args.accum, logger=logging.getLogger("model"))
model.train()
for p in model.parameters():
    p.requires_grad = False
HEADS = ("audio_embeds", "dsu_head", "speaker_embed_proj")
if args.mode.startswith("lora"):
    from peft import LoraConfig, get_peft_model
    r = int(args.mode.split(":")[1])
    model = get_peft_model(model, LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.0, bias="none",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
else:
    n_top = int(args.mode.split(":")[1]); n_layers = model.config.num_hidden_layers
    import re
    for name, p in model.named_parameters():
        m = re.match(r"model\.layers\.(\d+)\.", name)
        p.requires_grad = bool(m and int(m.group(1)) >= n_layers - n_top)
if args.heads == "train":
    for n, p in model.named_parameters():
        if any(h in n for h in HEADS):
            p.requires_grad = True
if args.grad_ckpt:
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    base.enable_input_require_grads()
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
log(f"mode {args.mode}, heads {args.heads}: trainable {sum(p.numel() for _, p in trainable) / 1e6:.1f} M of "
    f"{sum(p.numel() for p in model.parameters()) / 1e6:.0f} M")
opt = torch.optim.AdamW([p for _, p in trainable], lr=args.lr, weight_decay=0.0)

syn = load_from_disk(f"{args.data}/train"); val = load_from_disk(f"{args.data}/validation"); real = load_from_disk(f"{args.data}/real_sample")
collate = DSUDataCollator(tokenizer=tok, mlm=False)
dev = next(model.parameters()).device

def batch_of(rows):
    b = collate([dict(r) for r in rows])
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}

def fwd(rows):
    out = model(**batch_of(rows))
    # the model divides only the total loss by grad_acc_steps (= --accum); the audio/text parts are returned undivided
    return out["loss"] * args.accum, float(out["c1_dsu_loss"]), float(out["c1_text_loss"])

PADS = set(tok.convert_tokens_to_ids(["[SILENCE_PAD]", "[UTTERANCE_PAD]", "[WORD_PAD]"])) | {tok.pad_token_id}

def word_metrics(row):
    """Text loss on the word frames only. The model's own text loss is a mean over [speaker slot, prompt tokens, text stream] (the prompt is
    part of the labels) and the stream is mostly [SILENCE_PAD], so it says little about the words. Here: cross-entropy and top-1 accuracy of the
    system speaker's text stream on its real word tokens, teacher-forced. The stream is the last N+1 positions of the labels (N frames + the pad
    EOS the model appends); the logit one position before a label predicts it. Returns (sum of cross-entropy, n correct, n word tokens)."""
    b = batch_of([row]); out = model(**b)
    lg = out["ts_logits"][0, :, 0, :].float(); ids = b["text_stream_ids"][0, 0]
    stream = torch.cat([ids, ids.new_full((1,), tok.pad_token_id)])           # labels of the stream part, length N+1
    start = len(lg) - len(stream)                                               # first stream position in the full sequence
    assert start >= 1, (len(lg), len(stream))
    pred = lg[start - 1: len(lg) - 1]                                           # pred[k] predicts stream[k]
    mask = torch.tensor([int(t) not in PADS for t in stream.tolist()], device=stream.device)
    assert int(mask.sum()) == sum(int(t) not in PADS for t in ids.tolist()), "word-frame mask does not match the stream"
    if not mask.any():
        return 0.0, 0, 0
    ce = torch.nn.functional.cross_entropy(pred[mask], stream[mask], reduction="sum")
    return float(ce), int((pred[mask].argmax(-1) == stream[mask]).sum()), int(mask.sum())

def evaluate(tag):
    with torch.no_grad():
        res = {}
        for name, ds, n in (("synthetic val", val, args.n_eval), ("real", real, 2)):
            ls = [fwd([ds[i]]) for i in range(min(n, len(ds)))]
            res[name] = tuple(sum(float(l[k]) for l in ls) / len(ls) for k in range(3))
        wm = {}                                       # rows alternate: even = Tom is the system speaker, odd = Gweneth
        for split, ds in (("val", val), ("train", syn)):
            for who, par in (("Tom", 0), ("Gweneth", 1)):
                ms = [word_metrics(ds[i]) for i in range(par, min(args.n_eval, len(ds)), 2)]
                n = sum(m[2] for m in ms); wm[f"{split} {who}"] = (sum(m[0] for m in ms) / max(n, 1), sum(m[1] for m in ms) / max(n, 1), n)
    log(f"[{tag}] " + " | ".join(f"{k}: loss {v[0]:.3f} (audio {v[1]:.3f}, text {v[2]:.3f})" for k, v in res.items()))
    log(f"[{tag}] word frames: " + " | ".join(f"{k}: CE {v[0]:.3f}, top-1 {v[1]:.2f} ({v[2]} tokens)" for k, v in wm.items()))

n_rows = len(syn); steps_per_epoch = n_rows // args.accum
total = args.max_steps or steps_per_epoch * args.epochs
log(f"{n_rows} synthetic rows, {steps_per_epoch} steps per epoch, {total} optimizer steps in total (accum {args.accum}, lr {args.lr})")
evaluate("before training")
step, t0 = 0, time.time()
order = []
while step < total:
    if len(order) < args.accum:
        order += random.sample(range(n_rows), n_rows)
    rows, order = order[:args.accum], order[args.accum:]
    lr = args.lr * min(1.0, (step + 1) / max(1, args.warmup)) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / total)))
    for g in opt.param_groups:
        g["lr"] = lr
    tot = [0.0, 0.0, 0.0]
    for i in rows:
        loss, a, tx = fwd([syn[i]])
        loss.backward()
        tot = [tot[0] + float(loss) / args.accum, tot[1] + a / args.accum, tot[2] + tx / args.accum]
    gn = torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
    opt.step(); opt.zero_grad(); step += 1
    log(f"step {step}/{total}: loss {tot[0]:.3f} (audio {tot[1]:.3f}, text {tot[2]:.3f}) lr {lr:.2e} gnorm {float(gn):.2f} | "
        f"{(time.time() - t0) / 60:.1f} min, peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB")
    if args.eval_every and step % args.eval_every == 0 and step < total:
        evaluate(f"step {step}")
    if args.save_every and step % args.save_every == 0:
        torch.save({n: p.detach().cpu() for n, p in trainable}, f"{args.out}/trainable_step{step}.pt")
evaluate("after training")
if not args.no_save:
    merged = model.merge_and_unload() if hasattr(model, "merge_and_unload") else model
    merged.save_pretrained(f"{args.out}/final"); tok.save_pretrained(f"{args.out}/final")
    log(f"saved merged model to {args.out}/final")

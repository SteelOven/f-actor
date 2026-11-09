"""What does a model predict at Gweneth's word frames? Teacher-forced, per token: target, top-1 prediction, cross-entropy, P(target).

    .venv/bin/python finetune_repeat/word_probe.py --model /mnt/data/tts-local/finetune/single_v1_lora/final --data ../oov-data/finetune/single_v1 [--n 6]
Rows alternate Tom / Gweneth as system speaker; only Gweneth's (odd) rows are shown, for validation and for training.
"""
import argparse, logging, os, sys
from types import SimpleNamespace
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [f"{_ROOT}/training", f"{_ROOT}/training/inference_audio"]
import torch
from datasets import load_from_disk
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--data", required=True); ap.add_argument("--n", type=int, default=6)
args = ap.parse_args()
from data_collator_dsu import DSUDataCollator
from model import load_model
margs = SimpleNamespace(model_id=args.model, num_dsus=4, text_stream=True, multi_text_stream=False, audio_vocab_size=4032,
                        use_speaker_embedding=True, calc_loss_on_c1_only=False)
model, tok = load_model(margs, grad_acc_steps=1, logger=logging.getLogger("model")); model.eval()
collate = DSUDataCollator(tokenizer=tok, mlm=False); dev = next(model.parameters()).device
PADS = set(tok.convert_tokens_to_ids(["[SILENCE_PAD]", "[UTTERANCE_PAD]", "[WORD_PAD]"])) | {tok.pad_token_id}
for split in ("validation", "train"):
    ds = load_from_disk(f"{args.data}/{split}"); print(f"\n== {split}")
    for i in range(1, min(len(ds), 2 * args.n), 2):
        b = collate([dict(ds[i])]); b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
        with torch.no_grad():
            out = model(**b)
        lg = out["ts_logits"][0, :, 0, :].float(); ids = b["text_stream_ids"][0, 0]
        stream = torch.cat([ids, ids.new_full((1,), tok.pad_token_id)]); start = len(lg) - len(stream); pred = lg[start - 1: len(lg) - 1]
        row = []
        for k, t in enumerate(stream.tolist()):
            if t in PADS: continue
            p = torch.softmax(pred[k], -1)
            row.append(f"{tok.decode([t]).strip()!r}->{tok.decode([int(p.argmax())]).strip()!r} (P {float(p[t]):.3f})")
        print(f"row {i}: " + "  |  ".join(row))

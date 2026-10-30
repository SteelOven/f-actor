"""training/finetune.py with synthetic rows added to the real data. The training code is not edited: this wrapper patches load_data.

    python finetune_repeat/train.py --synthetic_dir <dir with train/ and validation/> [--synthetic_repeat N] <finetune.py arguments>

e.g. the arguments of scripts/train/train_nanocodec_special_tokens.sh, with --model_id hf_models/f_actor (start from the released
checkpoint), --train_on_subset 0.02 (a small share of the real dialogues; also applies to the real validation split) and a low
--learning_rate. The synthetic rows get the real split's column types, are appended (train: repeated --synthetic_repeat times,
then shuffled) and each split is trimmed to a multiple of the effective batch size again, as data.py does.
"""
import argparse, os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [f"{_ROOT}/training", f"{_ROOT}/training/inference_audio"]

pre = argparse.ArgumentParser(add_help=False)
pre.add_argument("--synthetic_dir", required=True)
pre.add_argument("--synthetic_repeat", type=int, default=1)
known, rest = pre.parse_known_args()
sys.argv = [sys.argv[0]] + rest

from datasets import concatenate_datasets, load_from_disk   # noqa: E402
import finetune                                              # noqa: E402
from data import get_world_size                              # noqa: E402

_load_data = finetune.load_data


def load_data_with_synthetic(**kwargs):
    train, val, collator = _load_data(**kwargs)
    targs = kwargs["training_args"]
    logger = kwargs["logger"]

    def add(real, name, repeat):
        syn = load_from_disk(f"{known.synthetic_dir}/{name}").cast(real.features)
        parts = [real] + [syn] * repeat
        merged = concatenate_datasets(parts)
        eff = (targs.train_batch_size if name == "train" else targs.eval_batch_size) * get_world_size()
        if name == "train":
            eff *= targs.gradient_accumulation_steps
        merged = merged.shuffle(seed=42) if name == "train" else merged
        merged = merged.take(len(merged) - len(merged) % eff)
        logger.info(f"{name}: {len(real)} real + {len(syn)} x {repeat} synthetic -> {len(merged)} rows (batch multiple {eff})")
        return merged

    return add(train, "train", known.synthetic_repeat), add(val, "validation", 1), collator


finetune.load_data = load_data_with_synthetic

if __name__ == "__main__":
    finetune.main()

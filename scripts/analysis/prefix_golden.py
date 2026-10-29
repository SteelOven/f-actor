"""Golden check for the single-turn prefix builder: the tensors training/prefix.py builds must not change.

Builds the prefix of three fixed P4 clips (tail off and tail_s=0.3) and compares sha256 hashes of the dsu/text
tensors with prefix_golden.json. Run it before and after touching training/prefix.py or the config path: the
single-turn path has to stay byte-identical, so earlier experiments remain valid.

    python scripts/analysis/prefix_golden.py              # check (exit 1 on a difference)
    python scripts/analysis/prefix_golden.py --update     # record the current output as the golden file

Needs the clips and alignment of oov-shared's p4_repeat run (default ../oov-data/checks/p4_repeat) and runs the
real NanoCodec, so the hashes are only comparable on the same device (cpu vs cuda can differ in rare code ids);
the device is stored in the golden file.
"""

import argparse
import hashlib
import json
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (os.path.join(_REPO, "training"), os.path.join(_REPO, "training", "inference_audio")):
    sys.path.insert(0, p)

import torch  # noqa: E402
from prefix import COSYVOICE_GAIN_DB, build_prefix  # noqa: E402

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prefix_golden.json")
WORDS = ["water", "hello", "archband"]
TAILS = [None, 0.3]


def digest(t):
    return hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest()[:16]


def build_all(run_dir, device):
    from nano_decode import load_model as load_codec
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(_REPO, "hf_models", "f_actor"))
    codec = load_codec(num_codebooks=4).to(device)
    rows = {}
    with open(os.path.join(run_dir, "alignment.jsonl")) as f:
        aligned = {r["word"]: r for r in map(json.loads, f) if r["voice"] == "tom"}
    for word in WORDS:
        for tail in TAILS:
            kw = {} if tail is None else {"tail_s": tail}
            pf = build_prefix(os.path.join(run_dir, aligned[word]["wav"]), aligned[word]["words"], codec, tokenizer,
                              gain_db=COSYVOICE_GAIN_DB, **kw)
            rows[f"{word}|tail={tail}"] = {"n_frames": pf.n_frames, "gain_db": pf.gain_db,
                                           "overflow_words": pf.overflow_words,
                                           "dsu": digest(pf.dsu), "text": digest(pf.text)}
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default=os.path.join(_REPO, "..", "oov-data", "checks", "p4_repeat"))
    ap.add_argument("--update", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    got = {"device": device, "prefixes": build_all(args.run_dir, device)}
    if args.update:
        with open(GOLDEN, "w") as f:
            json.dump(got, f, indent=2)
        print(f"recorded {len(got['prefixes'])} prefixes ({device}) in {GOLDEN}")
        return
    want = json.load(open(GOLDEN))
    if want["device"] != device:
        print(f"warning: golden file was recorded on {want['device']}, this is {device}")
    bad = [k for k in want["prefixes"] if got["prefixes"].get(k) != want["prefixes"][k]]
    for k in bad:
        print(f"DIFF {k}\n  want {want['prefixes'][k]}\n  got  {got['prefixes'].get(k)}")
    print("PASS: single-turn prefixes unchanged" if not bad else f"FAIL: {len(bad)} of {len(want['prefixes'])} differ")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

"""P(any audio head = EOS) at the first generated frame, per word and prefix setting. Model from FACTOR_MODEL_DIR (default: released).

A setting is TAIL:LOOKAHEAD, TAIL = raw (the clip as it is) or seconds of trailing silence (prefix.py set_tail), LOOKAHEAD = seconds of
extra silence encoded after the prefix and cut back (0 = none; removes the codec's end-of-file codes). Only the first half of the heads of
every row counts (interleave_halves overwrites the second half with the other row's samples before the stop check).

    FACTOR_MODEL_DIR=<run>/best_model python scripts/analysis/eos_probe.py 'confs/rows_test_x/tuned/*.json' raw:0 0.3:0 0.3:1.5 raw:1.5
Run from the repo root (inside the container); one model load, one forward pass per word and setting.
"""
import glob
import json
import os
import sys
import tempfile

sys.path[:0] = ["training", "training/inference_audio", "."]
import torch  # noqa: E402
from inference_example import generate_dialogue, load_model  # noqa: E402
from nano_decode import load_model as load_codec  # noqa: E402


def main():
    configs = sorted(glob.glob(sys.argv[1]))
    settings = [s.split(":") for s in sys.argv[2:]]
    if not configs or not settings:
        sys.exit(__doc__)
    model, tok = load_model()
    codec = load_codec(num_codebooks=4)
    first, orig = [], model.get_next_tokens

    def probe(B, logits, do_sample, temperature, top_k, top_p):
        if logits.shape[-1] == model.audio_vocab_size and not first:
            pr = model.sample_next_token(logits, temperature=temperature, top_k=top_k, top_p=top_p)
            first.append(pr[:, : pr.shape[1] // 2, model.audio_eos_id].cpu())
        return orig(B, logits, do_sample, temperature, top_k, top_p)

    model.get_next_tokens = probe
    out = tempfile.mkdtemp()
    table = {}
    for path in configs:
        cfg0 = json.load(open(path))
        word = cfg0.get("meta", {}).get("word", os.path.basename(path))
        for tail, la in settings:
            c = json.loads(json.dumps(cfg0))
            c.update(max_length=2, output_dir=out, output_file="p.wav")
            c["prefix"].pop("tail_s", None)
            c["prefix"].pop("lookahead_s", None)
            if tail != "raw":
                c["prefix"]["tail_s"] = float(tail)
            if float(la):
                c["prefix"]["lookahead_s"] = float(la)
            first.clear()
            generate_dialogue(model, tok, codec, c)
            p = first[0]
            table[(word, f"{tail}:{la}")] = float(1 - (1 - p).prod())
            print(f"{word:<12} tail {tail:>4} lookahead {la:>4}: P(any EOS at first frame) = {table[(word, f'{tail}:{la}')]:.3f}", flush=True)
    print("\nRESULT P(EOS at first frame), word x TAIL:LOOKAHEAD")
    cols = [f"{t}:{l}" for t, l in settings]
    print(f"{'word':<12}" + "".join(f"{c:>10}" for c in cols))
    words = sorted({w for w, _ in table})
    for w in words:
        print(f"{w:<12}" + "".join(f"{table[(w, c)]:>10.3f}" for c in cols))
    print(f"{'mean':<12}" + "".join(f"{sum(table[(w, c)] for w in words) / len(words):>10.3f}" for c in cols))


if __name__ == "__main__":
    main()

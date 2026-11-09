"""Free-running generation from an exact dataset row: the row's own history up to just before Gweneth's first text token is the prefix
(no prefix builder, no config wav), the model generates from there. Compares with the eval pipeline's prefix.
    FACTOR_MODEL_DIR=<model> .venv/bin/python finetune_repeat/first_turn_check.py --data ../oov-data/finetune/single_v1 --words designate divide [--greedy] [--repeats 2]"""
import argparse, json, os, sys, copy
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [f"{_ROOT}/training", f"{_ROOT}/training/inference_audio"]
import torch
from datasets import load_from_disk
import inference_example as ie
from prefix import Prefix
ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--split", default="train"); ap.add_argument("--words", nargs="+", required=True)
ap.add_argument("--repeats", type=int, default=2); ap.add_argument("--out", required=True); ap.add_argument("--mask", type=int, default=60); ap.add_argument("--cut-shift", type=int, default=0)
ap.add_argument("--blank-reply", action="store_true", help="with --cut-shift > 0: replace Gweneth's audio codes and text from her first word frame on by the preceding silence")
ap.add_argument("--builder", action="store_true", help="build the prefix with build_multiturn_prefix from the request clip instead of cutting the row")
ap.add_argument("--tidy", action="store_true"); ap.add_argument("--tail", type=float, default=0.5); ap.add_argument("--lookahead", type=float, default=1.5); ap.add_argument("--gain", type=float, default=5.6)
args = ap.parse_args()
model, tok = ie.load_model(); codec = ie.load_nanocodecs(num_codebooks=4)
ds = load_from_disk(f"{args.data}/{args.split}"); words = json.load(open(f"{args.data}/words.json"))[args.split]
PADS = set(tok.convert_tokens_to_ids(["[SILENCE_PAD]", "[UTTERANCE_PAD]", "[WORD_PAD]"])) | {tok.pad_token_id}
base_cfg = json.load(open(f"{_ROOT}/../oov-data/finetune/single_eval/eval_out/heads_t5/train/designate_r1_pfx.json"))["config"]
for w in args.words:
    j = words.index(w); even, odd = ds[2 * j], ds[2 * j + 1]
    dsu = torch.tensor(even["dsu_ids"]); tom_t = torch.tensor(even["text_stream_ids"])[0]; gw_t = torch.tensor(odd["text_stream_ids"])[0]
    f0 = next(i for i, t in enumerate(gw_t.tolist()) if t not in PADS)           # Gweneth's first word token ("okay")
    cut = f0 + args.cut_shift
    print(f"{w}: row {2*j}/{2*j+1}, {dsu.shape[1]} frames, Gweneth's first text token at frame {f0} ({tok.decode([int(gw_t[f0])]).strip()!r}); cut at {cut}; "
          f"Tom text: {tok.decode([t for t in tom_t.tolist() if t not in PADS])!r}; Gweneth text: {tok.decode([t for t in gw_t.tolist() if t not in PADS])!r}")
    d_cut, t_cut = dsu[:, :cut].clone(), torch.stack([tom_t, gw_t])[:, :cut].clone()
    if args.blank_reply:
        d_cut[4:, f0:] = d_cut[4:, f0 - 1:f0]; t_cut[1, f0:] = gw_t[f0 - 1]
    pf = Prefix(dsu=d_cut[None], text=t_cut[None], n_frames=cut, seconds=(cut - 2) * 0.08, overflow_words=0, gain_db=0.0,
                info={"turns": [{"speaker": 0}]})
    if args.builder:
        from prefix import build_multiturn_prefix
        D = f"{_ROOT}/../oov-data"
        rq = next(r for r in map(json.loads, open(f"{D}/checks/p6_full_request_s0/alignment.jsonl")) if r["word"] == w and r["voice"] == "tom")
        turn = {"speaker": 0, "wav": f"{D}/checks/p6_full_request_s0/{rq['wav']}", "words": rq["words"], "gain_db": args.gain}
        if args.tidy: turn["tidy"] = {"lead_keep_s": 0.1, "max_gap_s": 0.2, "fade_s": 0.04}
        built = build_multiturn_prefix([turn], codec, tok, tail_s=args.tail, lookahead_s=args.lookahead)
        n = min(built.n_frames, cut); same = (built.dsu[0, :, :n] == dsu[:, :n]).float().mean().item(); same_t = (built.text[0, :, :n] == pf.text[0, :, :n]).float().mean().item()
        print(f"  builder prefix: {built.n_frames} frames (row cut {cut}); audio codes equal to the row's over the first {n} frames: {same:.3f}; text equal: {same_t:.3f}")
        pf = built
    ie.config_prefix = lambda config, m, c, t, pf=pf: pf
    for r in range(1, args.repeats + 1):
        cfg = copy.deepcopy(base_cfg); cfg.pop("prefix", None); cfg["prefix"] = {"turns": [{"wav": f"row:{args.split}:{2*j}"}]}
        cfg.update(output_dir=args.out, output_file=f"{w}_r{r}_row.wav", mask_stop_frames=args.mask)
        _, _, record_file = ie.generate_dialogue(model, tok, codec, cfg)
        g = json.load(open(record_file))["generation"]; print(f"  r{r}: ended {g['ended_by']} @ {g['n_generated_frames']}; Gweneth: {g['speaker2']['transcript']!r}")

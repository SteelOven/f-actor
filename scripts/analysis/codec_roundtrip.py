"""Codec round-trip control for the OOV experiment.

Question this answers: when F-Actor fails to *say* a rare word, is the ceiling
the **nanocodec** (it physically can't represent those sounds) or the **model**
(its audio head generates the wrong tokens)? This isolates the codec by taking
*clean* speech of the word and running only:

    clean wav -> nanocodec.encode -> tokens -> nanocodec.decode -> wav -> ASR

If the word survives that, the codec is fine and any F-Actor failure is the
model. If it doesn't, the codec is the ceiling and no amount of training fixes
it.

Confound control: we ASR the *original* clean wav too and report the codec's
*delta* (original -> round-trip). That way a TTS mispronunciation in the demo
input, or a flaky ASR, doesn't get blamed on the codec -- we measure only the
degradation the encode/decode introduces.

Demo input: with --synth, clean wavs are generated with a TTS model
(facebook/mms-tts-eng) in a carrier sentence. For the real thesis, drop human
recordings (or a vetted TTS) into --clean-dir instead; the original-ASR baseline
keeps the comparison honest either way.

Usage:
    uv run python scripts/analysis/codec_roundtrip.py --synth \
        --words recipe galaxy schadenfreude borborygmus
    uv run python scripts/analysis/codec_roundtrip.py --clean-dir my_recordings/
"""

import argparse
import os
import sys

# nano_decode.py lives under training/inference_audio/ (the inference scripts run
# with that on the path). Make it importable from here.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "training", "inference_audio"))

import librosa
import numpy as np
import soundfile as sf
import torch

# Same-dir helpers: grading + cached Whisper wrapper used by analyze_oov.py.
from analyze_oov import ASR, grade_word, normalize

CODEC_SR = 22050  # nemo-nano-codec-22khz
CARRIER = "The word is {word}."
DEFAULT_CLEAN = "outputs/oov/roundtrip/clean"
DEFAULT_OUTDIR = "outputs/oov/roundtrip"
DEFAULT_ASR = "openai/whisper-base.en"


def synth_clean(words, clean_dir, tts_model="facebook/mms-tts-eng"):
    """Synthesize a clean carrier-sentence wav per word (demo input only)."""
    os.makedirs(clean_dir, exist_ok=True)
    missing = [w for w in words if not os.path.exists(os.path.join(clean_dir, f"{w}.wav"))]
    if not missing:
        return
    from transformers import pipeline

    print(f"[synth] loading {tts_model} for {len(missing)} word(s) ...")
    tts = pipeline("text-to-speech", model=tts_model)
    for w in missing:
        out = tts(CARRIER.format(word=w))
        audio = np.asarray(out["audio"]).squeeze()
        sf.write(os.path.join(clean_dir, f"{w}.wav"), audio, out["sampling_rate"])
        print(f"[synth] {w}.wav")


def roundtrip(codec, wav_in, wav_out):
    """Encode then decode a wav through the nanocodec; write the result."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio, _ = librosa.load(wav_in, sr=CODEC_SR, mono=True)
    a = torch.from_numpy(audio).float().unsqueeze(0).to(device)  # [1, T]
    a_len = torch.tensor([a.shape[-1]], device=device)
    codec = codec.to(device)
    with torch.no_grad():
        tokens, tok_len = codec.encode(audio=a, audio_len=a_len, sample_rate=CODEC_SR)
        recon, _ = codec.decode(tokens=tokens, tokens_len=tok_len)
    recon = recon.squeeze().cpu().numpy()
    sf.write(wav_out, recon, CODEC_SR)
    return tokens.shape[-1]  # number of codec frames (for reference)


def load_words(args):
    words = [w.lower() for w in args.words]
    if args.words_file:
        with open(args.words_file) as f:
            words += [l.strip().lower() for l in f if l.strip() and not l.startswith("#")]
    return list(dict.fromkeys(words))  # dedupe, keep order


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--words", nargs="+", default=[])
    ap.add_argument("--words-file")
    ap.add_argument("--clean-dir", default=DEFAULT_CLEAN)
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    ap.add_argument("--synth", action="store_true", help="TTS any missing clean wavs.")
    ap.add_argument("--asr-model", default=DEFAULT_ASR)
    ap.add_argument("--report", default="outputs/oov/roundtrip/roundtrip.tsv")
    args = ap.parse_args()

    from jiwer import wer

    from nano_decode import load_model

    words = load_words(args)
    if not words:
        ap.error("No words (use --words or --words-file).")
    if args.synth:
        synth_clean(words, args.clean_dir)

    os.makedirs(args.outdir, exist_ok=True)
    rt_dir = os.path.join(args.outdir, "rt")
    os.makedirs(rt_dir, exist_ok=True)
    asr = ASR(args.asr_model, os.path.join(args.outdir, "transcripts"))

    print("[codec] loading nemo-nano-codec (4 codebooks) ...")
    codec = load_model(num_codebooks=4)

    results = []
    for w in words:
        clean = os.path.join(args.clean_dir, f"{w}.wav")
        if not os.path.exists(clean):
            print(f"[skip] no clean wav for '{w}' (use --synth or add {clean})")
            continue
        rt = os.path.join(rt_dir, f"{w}.wav")
        roundtrip(codec, clean, rt)

        orig_txt = asr.transcribe(clean)
        rt_txt = asr.transcribe(rt)
        o_label, o_score, _ = grade_word(orig_txt, w)
        r_label, r_score, r_ev = grade_word(rt_txt, w)
        try:
            distortion = round(wer(normalize(orig_txt), normalize(rt_txt)), 3)
        except ValueError:
            distortion = float("nan")

        results.append({
            "word": w,
            "orig_label": o_label, "orig_score": round(o_score, 1),
            "rt_label": r_label, "rt_score": round(r_score, 1), "rt_evidence": r_ev,
            "delta": round(o_score - r_score, 1),
            "codec_wer": distortion,
        })

    if not results:
        print("Nothing to report.")
        return

    import csv
    hdr = ["word", "orig_label", "orig_score", "rt_label", "rt_score",
           "rt_evidence", "delta", "codec_wer"]
    with open(args.report, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=hdr, delimiter="\t")
        wtr.writeheader()
        wtr.writerows(results)

    print(f"\n{'word':<16}{'clean ASR':<16}{'round-trip ASR':<24}{'codec WER':>10}")
    print("-" * 66)
    for r in results:
        o = f"{r['orig_label']}({r['orig_score']:.0f})"
        rt_ = f"{r['rt_label']}({r['rt_score']:.0f}) {r['rt_evidence']}"
        print(f"{r['word']:<16}{o:<16}{rt_:<24}{r['codec_wer']:>10}")
    print("\nReading: if clean ASR is intact but round-trip is not, the codec is")
    print("the ceiling for that word. If clean ASR already fails, that's a TTS/ASR")
    print(f"issue, not the codec. Per-word table: {args.report}")


if __name__ == "__main__":
    main()

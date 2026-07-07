"""Aggregate OOV-experiment results across the two F-Actor output streams.

For every generated run in confs/oov/manifest.tsv this ensures the run's own
record (outputs/oov/<stem>.json, written by training/inference_example.py)
has an ASR transcript and word-fidelity grading merged in -- via
scripts/analysis/oov_common.enrich_run(), the same function
training/inference_example.py's --transcribe flag calls inline -- then reads
back every record to report, per run and per frequency band:

  * target-word fidelity in each stream, graded intact/split/substituted/dropped
    (a single word, so character similarity -- not WER -- is the right tool);
  * WER(inner monologue -> explainer audio), i.e. how faithfully the audio
    rendered what the model planned to say (this is the WER-shaped comparison).

The point is the *gap*: a word can be planned correctly in the text stream yet
break in the audio (acoustic OOV) or break already in the text stream (lexical
OOV). Grouping the gap by band is the headline result.

If a run was already transcribed inline (--transcribe at generation time),
this is a no-op for it and just reads the cached result back.

Usage:
    uv run python scripts/analysis/analyze_oov.py
    uv run python scripts/analysis/analyze_oov.py --asr-model openai/whisper-base.en
"""

import argparse
import csv
import os

from oov_common import ASR, enrich_run, load_run

BANDS = ["frequent", "medium", "rare", "narrative-only", "oov"]
DEFAULT_MANIFEST = "confs/oov/manifest.tsv"
DEFAULT_OUTDIR = "outputs/oov"
DEFAULT_ASR = "openai/whisper-base.en"


def load_manifest(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    ap.add_argument("--asr-model", default=DEFAULT_ASR)
    ap.add_argument("--report", default="outputs/oov/analysis.tsv")
    args = ap.parse_args()

    asr = ASR(args.asr_model)
    rows = load_manifest(args.manifest)

    results = []
    for row in rows:
        stem = os.path.splitext(os.path.basename(row["audio"]))[0]  # e.g. algorithm_r2
        record_path = os.path.join(args.outdir, f"{stem}.json")
        if not os.path.exists(record_path):
            continue  # not generated yet
        record = load_run(record_path)
        if "audio" not in record or not os.path.exists(
            os.path.join(args.outdir, record["audio"]["c1"])
        ):
            continue

        record = enrich_run(record_path, asr, word=row["word"])
        grading = record["grading"][asr.model]
        asr_result = record["asr"][asr.model]

        results.append(
            {
                "word": row["word"],
                "band": row["band"],
                "repeat": row["repeat"],
                "mono_label": grading["mono_label"],
                "mono_score": grading["mono_score"],
                "audio_label": grading["audio_label"],
                "audio_score": grading["audio_score"],
                "audio_evidence": grading["audio_evidence"],
                "render_wer": asr_result["render_wer"],
            }
        )

    if not results:
        print("No generated runs found (need outputs/oov/<stem>.json + _c1.wav).")
        return

    # per-run table
    hdr = ["word", "band", "repeat", "mono_label", "mono_score",
           "audio_label", "audio_score", "audio_evidence", "render_wer"]
    with open(args.report, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hdr, delimiter="\t")
        w.writeheader()
        w.writerows(results)

    print(f"\n{'word':<16}{'band':<10}{'monologue':<14}{'audio':<22}{'WER':>6}")
    print("-" * 68)
    order = {b: i for i, b in enumerate(BANDS)}
    for r in sorted(results, key=lambda r: (order.get(r["band"], 9), r["word"])):
        mono = f"{r['mono_label']}"
        audio = f"{r['audio_label']}({r['audio_score']:.0f}) {r['audio_evidence']}"
        print(f"{r['word']:<16}{r['band']:<10}{mono:<14}{audio:<22}{r['render_wer']:>6}")

    # per-band aggregate
    print(f"\n{'band':<16}{'n':>3}  {'mono intact%':>12}{'audio intact%':>14}{'mean WER':>10}")
    print("-" * 60)
    for band in BANDS:
        b = [r for r in results if r["band"] == band]
        if not b:
            continue
        mono_ok = 100 * sum(r["mono_label"] in ("intact", "split") for r in b) / len(b)
        audio_ok = 100 * sum(r["audio_label"] in ("intact", "split") for r in b) / len(b)
        wers = [r["render_wer"] for r in b if r["render_wer"] == r["render_wer"]]
        mean_wer = sum(wers) / len(wers) if wers else float("nan")
        print(f"{band:<16}{len(b):>3}  {mono_ok:>11.0f}%{audio_ok:>13.0f}%{mean_wer:>10.2f}")

    print(f"\nPer-run table written to {args.report}")


if __name__ == "__main__":
    main()

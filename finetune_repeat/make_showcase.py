"""Local HTML overview of every model version tried so far, with players for the generated clips (nothing is uploaded).
    .venv/bin/python finetune_repeat/make_showcase.py [out.html]"""
import glob, html, json, os, sys
D = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "oov-data", "finetune"))
out = sys.argv[1] if len(sys.argv) > 1 else f"{D}/showcase.html"
# (title, eval dir, tags, what changed, training data, numbers)
VERSIONS = [
 ("0 · Released F-Actor (base)", "micro_eval", ["base", "base_la"], "Nothing changed. Prefix = Tom's request only; base_la adds 0.1 s tail + 1.5 s look-ahead.",
  "Original training (not touched).", "Target word in Gweneth's text: 0 of 18 (base), 1 of 18 (base_la). Rarely stops early."),
 ("1 · micro_v2 LoRA", "micro_eval", ["tuned", "tuned_la", "tuned_mask"], "LoRA r16 on all linear layers, heads frozen, 60 steps. tuned = 0.3 s tail; tuned_la = 0.1 s tail + look-ahead; tuned_mask = + stop mask (EOS blocked in first frames).",
  "20 multi-turn dialogues, 5 words (doctor, garden, morning, movie, music), 3-5 exchanges each.",
  "Stops at once in 18 of 18 (tuned), 15 of 18 (tuned_la), 4 of 18 (tuned_mask). With the mask Gweneth speaks, but only training-vocabulary words; target word said: 0 of 18."),
 ("2 · single_v1, Run A (LoRA only)", "single_eval", ["lora"], "LoRA r16, heads frozen, 136 steps. Stop mask 60 frames, 0.1 s tail + 1.5 s look-ahead.",
  "69 train words (+10 validation), one exchange each: request, 0.5 s gap, reply. CosyVoice TTS, 138 rows.",
  "Teacher-forced probe: 'okay' right, word token wrong in 16 of 16 rows (P ≤ 0.018). Generation: target word 0 of 10; all stopped by EOS at 61-98 frames."),
 ("3 · single_v1, Run B (LoRA + trainable heads)", "single_eval", ["heads", "heads_t5"], "Same data; audio embeds, dsu_head and speaker projection also trained (143.9 M params), 272 steps. heads_t5 = prefix with 0.5 s tail (training gap), no look-ahead.",
  "Same as Run A.",
  "Training words memorised (teacher-forced train CE 0.009, top-1 1.00) but validation unchanged (CE 6.4, top-1 0.40). Generation: target word 0 of 20 across both prefix settings."),

 ("4 · single_v2 (LoRA + heads, 3 gap variants)", "single_eval", ["v2_shift"], "Same recipe as Run B, trained on 3x the rows. v2_shift = eval by cutting the prefix out of a dataset row, shifted by 0-4 frames (s0-s4), train words and validation words (val_s*).",
  "Same 69 train words, each as 3 dialogues with the gap drawn from 0.3-0.9 s; 207 dialogues, 414 rows. Same 10 validation words.",
  "Training words again memorised (train Gweneth CE 0.125, top-1 1.00); validation Gweneth CE 6.5, top-1 0.45, no better than v1. Generation: she opens with 'okay' and then names a different word (substring hit 1 of 32). More gap variants did not add word diversity."),
]
DATA = [  # (dataset, source, words, rows, structure, gap, notes)
 ("micro (micro_v2)", "p5 round-trip TTS demo clips", "5 words: doctor, garden, morning, movie, music", "40 train rows (20 dialogues) + 8 val", "Multi-turn: 3-5 exchanges, each Tom's request (+5.6 dB) then Gweneth's reply; 2 rows per dialogue (Tom / Gweneth as system speaker)", "natural, from the multi-turn builder", "Four times the same words; teaches the format, not the copy skill. Held-out words: coffee, dinner, window."),
 ("single_v1", "p6 full TTS run, seed 0 (Whisper-small accepted)", "79 usable words: 69 train, 10 val (breath, game, instantly, moon, officer, paddle, peculiar, robust, sanctuary, somewhere)", "138 train rows (69 dialogues) + 20 val", "One exchange: 'Please listen carefully and repeat the word X.' then 'Okay, X.' (a reply Whisper heard as 'K. X' is accepted)", "fixed 0.5 s", "Clips tidied: lead 0.1 s, word gaps <= 0.2 s, 40 ms fade-out; 0.1 s end tail. Words span frequency bands."),
 ("single_v2", "same clips as single_v1", "same 69 train / 10 val", "414 train rows (207 dialogues) + 20 val", "Same single exchange, 3 dialogues per train word", "uniform 0.3-0.9 s per dialogue", "Only the gap varies; the clips and words repeat, so the model sees each word 6x per epoch (3 dialogues x 2 rows)."),
]
SKETCH = """<section><h2>Training loop sketch (finetune_repeat/local_train.py)</h2>
<svg viewBox="0 0 1040 330" width="100%" font-family="system-ui" font-size="13">
<defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#555"/></marker></defs>
<g fill="#f4f7fb" stroke="#678" rx="6">
<rect x="10" y="20" width="150" height="70" rx="6"/><rect x="190" y="20" width="170" height="70" rx="6"/><rect x="390" y="20" width="190" height="70" rx="6"/>
<rect x="610" y="20" width="190" height="70" rx="6"/><rect x="830" y="20" width="190" height="70" rx="6"/>
<rect x="390" y="150" width="190" height="70" rx="6" fill="#eef7ee"/><rect x="610" y="150" width="190" height="70" rx="6" fill="#eef7ee"/><rect x="830" y="150" width="190" height="70" rx="6" fill="#eef7ee"/>
<rect x="10" y="150" width="150" height="70" rx="6" fill="#fff6e8"/><rect x="190" y="150" width="170" height="70" rx="6" fill="#fff6e8"/></g>
<g fill="#222" text-anchor="middle">
<text x="85" y="48">Dialogue</text><text x="85" y="66" fill="#555">request + 0.5 s gap</text><text x="85" y="80" fill="#555">+ reply + 0.1 s tail</text>
<text x="275" y="48">Row builder</text><text x="275" y="66" fill="#555">dialogue_rows: codec tokens,</text><text x="275" y="80" fill="#555">text stream, 2 rows (Tom / Gweneth)</text>
<text x="485" y="48">Sample 4 rows</text><text x="485" y="66" fill="#555">(grad. accumulation 4)</text><text x="485" y="80" fill="#555">one forward each</text>
<text x="705" y="48">Loss</text><text x="705" y="66" fill="#555">audio CE (8 heads) +</text><text x="705" y="80" fill="#555">text CE (prompt + stream)</text>
<text x="925" y="48">backward</text><text x="925" y="66" fill="#555">grads only on LoRA</text><text x="925" y="80" fill="#555">(+ heads in Run B)</text>
<text x="925" y="178">Clip grad norm 1.0</text><text x="925" y="196" fill="#555">AdamW step, lr 2e-4</text><text x="925" y="210" fill="#555">warmup 5, cosine to 10%</text>
<text x="705" y="178">Every 34 steps: evaluate</text><text x="705" y="196" fill="#555">val loss + word-frame CE /</text><text x="705" y="210" fill="#555">top-1 (Tom, Gweneth; val, train)</text>
<text x="485" y="178">After the last step</text><text x="485" y="196" fill="#555">merge_and_unload,</text><text x="485" y="210" fill="#555">save_pretrained(final)</text>
<text x="85" y="178">Frozen base</text><text x="85" y="196" fill="#555">F-Actor 1.38 B, fp32 CPU</text>
<text x="275" y="178">LoRA adapters</text><text x="275" y="196" fill="#555">r16, alpha 32, dropout 0, all</text><text x="275" y="210" fill="#555">linear projections: 11.3 M params</text></g>
<g stroke="#555" fill="none" marker-end="url(#a)"><path d="M160,55 H188"/><path d="M360,55 H388"/><path d="M580,55 H608"/><path d="M800,55 H828"/>
<path d="M925,90 V148"/><path d="M830,185 H802"/><path d="M610,185 H582"/><path d="M85,150 V92" stroke-dasharray="4"/><path d="M275,150 V92" stroke-dasharray="4"/></g>
<text x="10" y="270" text-anchor="start" fill="#555">Run A: only the LoRA adapters are trainable (136 steps, 4 epochs). Run B: also audio_embeds, dsu_head, speaker_embed_proj (143.9 M params, 272 steps).</text>
<text x="10" y="290" text-anchor="start" fill="#555">Rows alternate: even = Tom is the system speaker, odd = Gweneth. Word-frame metric = CE over the non-pad tokens of the system speaker's text stream.</text></svg>
<pre style="background:#f7f7f7;padding:8px;font-size:12px">for step in range(total):                       # total = steps_per_epoch * epochs, steps_per_epoch = rows // 4
    lr = base_lr * min(1, (step+1)/5) * (0.1 + 0.9*0.5*(1+cos(pi*step/total)))
    for row in 4 sampled rows:  loss, audio, text = fwd([row]);  loss.backward()   # grads summed over the 4 rows (not averaged); only the logged loss is the mean
    clip_grad_norm_(trainable, 1.0);  opt.step();  opt.zero_grad()
    if step % 34 == 0: evaluate()                  # synthetic val, real rows, word-frame metric
merged = model.merge_and_unload();  merged.save_pretrained(out/final)</pre></section>"""
def clip(p):
    return f'<audio controls preload="none" src="file://{html.escape(p)}"></audio>' if os.path.exists(p) else "<i>missing</i>"
h = ["""<!doctype html><meta charset=utf-8><title>F-Actor repeat fine-tune: versions</title>
<style>body{font:15px/1.5 system-ui;max-width:1100px;margin:2em auto;padding:0 1em;color:#222}h1{margin-bottom:0}.sub{color:#666}
section{border:1px solid #ddd;border-radius:8px;padding:1em 1.2em;margin:1.5em 0}h2{margin-top:0}table{border-collapse:collapse;width:100%;margin:.5em 0 1em}
td,th{border-bottom:1px solid #eee;padding:4px 8px;text-align:left;vertical-align:middle;font-size:13px}th{background:#fafafa}audio{height:30px;width:220px}h3{margin:1em 0 .2em}td div{margin:2px 0}.tl{line-height:1.8;max-width:1000px}
.k{color:#555}.said{font-family:ui-monospace,monospace;font-size:12px}.eos{color:#b00}.ok{color:#080}details{margin:.4em 0}</style>
<h1>F-Actor “repeat the word” fine-tune: what we tried</h1>
<p class=sub>Goal: after Tom says “Please listen carefully and repeat the word X.”, Gweneth says “Okay, X.” (held-out and OOV words later). Local file; audio plays from disk.</p>
<section><h2>Pipeline in one paragraph</h2><p>Prefix = Tom's request clip teacher-forced through the codec; the model then generates Gweneth's side (and Tom's) until EOS or 150 frames (80 ms each). A run counts as a hit only if the target word appears in Gweneth's text stream. Stop mask = audio EOS and text pad forbidden in the first N frames.</p></section>"""]
def curves():
    """val/train Gweneth word CE per eval step from the three logs with word-frame metrics."""
    import re
    runs = [("v1 LoRA only", "single_v1_train.log", "#888", 136), ("v1 +heads", "single_v1_heads_train.log", "#d62", 272), ("v2 +heads", "single_v2_heads_train.log", "#26b", 309)]
    W, H, L, B = 480, 230, 40, 25
    def panel(key, title, ymax):
        g = [f'<text x="{L}" y="14" font-weight="bold">{title}</text><rect x="{L}" y="20" width="{W - L - 10}" height="{H - B - 20}" fill="none" stroke="#ccc"/>']
        for tick in range(0, int(ymax) + 1, 2):
            y = 20 + (H - B - 20) * (1 - tick / ymax); g.append(f'<text x="{L - 5}" y="{y + 4:.0f}" text-anchor="end" fill="#666">{tick}</text><line x1="{L}" x2="{W - 10}" y1="{y:.0f}" y2="{y:.0f}" stroke="#eee"/>')
        for i, (name, f, col, total) in enumerate(runs):
            pts = []
            for l in open(f"{D}/{f}"):
                m = re.match(r"\[(before training|step (\d+)|after training)\] word frames", l)
                if not m: continue
                g_ = re.findall(r"(?:(val|train) )?Gweneth: CE ([\d.]+)", l)       # v1 LoRA-only log has val only, without the prefix
                v = [float(c) for pre, c in g_ if (pre or "val") == key]
                if v: pts.append((0 if m[1] == "before training" else int(m[2]) if m[2] else None, v[0]))
            if not pts: continue
            pts = [(total if x is None else x, y) for x, y in pts]
            xs = lambda x: L + (W - L - 10) * x / 310; ys = lambda y: 20 + (H - B - 20) * (1 - min(y, ymax) / ymax)
            g.append(f'<polyline points="{" ".join(f"{xs(x):.0f},{ys(y):.0f}" for x, y in pts)}" fill="none" stroke="{col}" stroke-width="2"/>')
            g.append(f'<text x="{W - 15}" y="{34 + 14 * i}" text-anchor="end" fill="{col}">{name}</text>')
        g.append(f'<text x="{(L + W) / 2}" y="{H - 6}" text-anchor="middle" fill="#666">optimizer step</text>')
        return f'<svg viewBox="0 0 {W} {H}" width="49%" font-family="system-ui" font-size="11">{"".join(g)}</svg>'
    return panel("val", "Validation words: Gweneth's word CE", 12) + panel("train", "Training words: Gweneth's word CE", 12)
def data_table():
    return "<table><tr><th>dataset<th>clips<th>words<th>size<th>dialogue<th>gap<th>notes</tr>" + "".join("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in r) + "</tr>" for r in DATA) + "</table>"
LOOP = """<section><h2>Training loop: how it changed between runs</h2>
<table><tr><th>run<th>trainable<th>rows<th>steps<th>what changed in the loop<th>what it showed</tr>
<tr><td>micro_v2<td>LoRA 11.3 M<td>40<td>60<td>First loop. Loss parts were scaled by accum when logged (fixed in e2d35b8).<td>Text loss 3.1 to 0.09 on val but only the 5 words; real-dialogue loss rose slightly (4.43 to 4.50).</tr>
<tr><td>single_v1 A<td>LoRA 11.3 M<td>138<td>136<td>Added the word-frame metric (CE / top-1 on Gweneth's and Tom's word tokens only): the model's own text loss averages in prompt and silence pads, so it hid the problem.<td>Val Gweneth CE 11.7 to 6.4, top-1 0.40: 'okay' learned, the word not.</tr>
<tr><td>single_v1 B<td>LoRA + heads 143.9 M<td>138<td>272<td>Also train audio_embeds, dsu_head, speaker_embed_proj; gradient norm 10-50 in the first steps (vs 3 for LoRA only), clip 1.0 bites every step.<td>Train Gweneth CE 0.009, val unchanged at 6.4; synthetic val audio loss rises 3.5 to 4.2 (overfit) and real-dialogue loss 4.43 to 4.98.</tr>
<tr><td>single_v2<td>LoRA + heads 143.9 M<td>414<td>309<td>Same loop, 3x rows (gap variants); 291 min on CPU, peak RSS 10.2 GB.<td>Memorisation slower (train Gweneth CE 0.125 at the end, v1 0.009) but the same val plateau (6.5).</tr></table>
<p>In the plots below, the heads runs drive the training-word CE towards zero while the validation-word CE stays above 5.7 from the first eval on: the model learns a per-word lookup of the 69 words and not 'say what Tom said'. The signal that changed this reading was the word-frame metric, not the total loss.</p>
<div>__CURVES__</div>
<p><b>Common settings:</b> AdamW, lr 2e-4, no weight decay, 5 warmup steps then cosine to 10 %, accum 4 (gradients summed over 4 rows and clipped at 1.0), teacher forcing, fp32 on CPU, one row per forward pass, evaluation every 34 steps (4 val rows, 2 real rows, word frames on 4 val and 4 train rows). Seed 0.</p>
<p><b>Ideas the data suggests</b> (not run yet): many more distinct words instead of more variants of the same 69; a loss that up-weights Gweneth's word frames; lower rank / fewer steps with early stopping on val Gweneth CE (it is already best at step 34-68); mixed batches with real dialogues to keep the real-dialogue loss from rising.</p></section>"""
h_data = "<section><h2>Training data per version</h2>" + data_table() + "<p>Row construction (rows.py dialogue_rows): the dialogue is encoded with the codec as two parallel streams; each dialogue yields one row with Tom and one with Gweneth as the system speaker. Audio labels cover all 8 heads, text labels the system speaker's text stream. In single_v1/single_v2 the validation words are never in training; micro_v2's validation dialogues reuse the same 5 words (different dialogues).</p></section>"
DATAGEN = """<section><h2>Data generation: from TTS to training row</h2>
<svg viewBox="0 0 1060 400" width="100%" font-family="system-ui" font-size="13"><defs><marker id="b" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#555"/></marker></defs>
<rect x="10" y="20" width="230" height="125" rx="6" fill="#f4f7fb" stroke="#678"/>
<text x="125" y="44" text-anchor="middle" font-weight="bold">1 · TTS clips</text>
<text x="125" y="70" text-anchor="middle" fill="#333">CosyVoice, one clip per word</text>
<text x="125" y="88" text-anchor="middle" fill="#333">Tom: request 'repeat the word X'</text>
<text x="125" y="106" text-anchor="middle" fill="#333">Gweneth: 'Okay, X.'</text>
<text x="125" y="133" text-anchor="middle" fill="#777" font-size="11">oov-shared round-trip runner</text>
<rect x="270" y="20" width="230" height="125" rx="6" fill="#f4f7fb" stroke="#678"/>
<text x="385" y="44" text-anchor="middle" font-weight="bold">2 · Quality gate</text>
<text x="385" y="70" text-anchor="middle" fill="#333">Whisper-small round trip</text>
<text x="385" y="88" text-anchor="middle" fill="#333">keep if the target word is heard</text>
<text x="385" y="106" text-anchor="middle" fill="#333">('K. X' replies accepted)</text>
<text x="385" y="133" text-anchor="middle" fill="#777" font-size="11">-> 79 usable words</text>
<rect x="530" y="20" width="230" height="125" rx="6" fill="#f4f7fb" stroke="#678"/>
<text x="645" y="44" text-anchor="middle" font-weight="bold">3 · Forced alignment</text>
<text x="645" y="70" text-anchor="middle" fill="#333">MMS_FA: audio + exact text</text>
<text x="645" y="88" text-anchor="middle" fill="#333">word start / duration (20 ms)</text>
<text x="645" y="106" text-anchor="middle" fill="#333">no ASR, no misspelled OOV</text>
<text x="645" y="133" text-anchor="middle" fill="#777" font-size="11">checks/align.py</text>
<rect x="790" y="20" width="230" height="125" rx="6" fill="#fff6e8" stroke="#678"/>
<text x="905" y="44" text-anchor="middle" font-weight="bold">4 · Tidy + assemble</text>
<text x="905" y="70" text-anchor="middle" fill="#333">lead cut to 0.1 s, gaps &lt;= 0.2 s</text>
<text x="905" y="88" text-anchor="middle" fill="#333">Tom +5.6 dB; gap 0.5 s (v1),</text>
<text x="905" y="106" text-anchor="middle" fill="#333">0.3-0.9 s (v2); 0.1 s end tail</text>
<text x="905" y="133" text-anchor="middle" fill="#777" font-size="11">prefix.py tidy_clip, build_multiturn_prefix</text>
<rect x="10" y="200" width="230" height="125" rx="6" fill="#fff6e8" stroke="#678"/>
<text x="125" y="224" text-anchor="middle" font-weight="bold">5 · Codec</text>
<text x="125" y="250" text-anchor="middle" fill="#333">NanoCodec, 4 codebooks</text>
<text x="125" y="268" text-anchor="middle" fill="#333">12.5 frames/s per channel</text>
<text x="125" y="286" text-anchor="middle" fill="#333">8 heads: Tom 4 + Gweneth 4</text>
<text x="125" y="313" text-anchor="middle" fill="#777" font-size="11">prefix.py encode</text>
<rect x="270" y="200" width="230" height="125" rx="6" fill="#fff6e8" stroke="#678"/>
<text x="385" y="224" text-anchor="middle" font-weight="bold">6 · Text stream</text>
<text x="385" y="250" text-anchor="middle" fill="#333">word token on its start frame</text>
<text x="385" y="268" text-anchor="middle" fill="#333">[WORD_PAD] to the word's end</text>
<text x="385" y="286" text-anchor="middle" fill="#333">[UTTERANCE_PAD] in the gaps</text>
<text x="385" y="313" text-anchor="middle" fill="#777" font-size="11">adapt_to_text_stream (audio +2 frames)</text>
<rect x="530" y="200" width="230" height="125" rx="6" fill="#eef7ee" stroke="#678"/>
<text x="645" y="224" text-anchor="middle" font-weight="bold">7 · Two rows per dialogue</text>
<text x="645" y="250" text-anchor="middle" fill="#333">Tom / Gweneth as system speaker</text>
<text x="645" y="268" text-anchor="middle" fill="#333">heads swapped, own narrative</text>
<text x="645" y="286" text-anchor="middle" fill="#333">prompt, codes, text, speaker emb.</text>
<text x="645" y="313" text-anchor="middle" fill="#777" font-size="11">rows.py dialogue_rows</text>
<rect x="790" y="200" width="230" height="125" rx="6" fill="#eef7ee" stroke="#678"/>
<text x="905" y="224" text-anchor="middle" font-weight="bold">8 · Dataset</text>
<text x="905" y="250" text-anchor="middle" fill="#333">split by word: 69 train / 10 val</text>
<text x="905" y="268" text-anchor="middle" fill="#333">v1: 138 rows, v2: 414 rows</text>
<text x="905" y="286" text-anchor="middle" fill="#333">+ real_sample (reference loss)</text>
<text x="905" y="313" text-anchor="middle" fill="#777" font-size="11">make_dataset_single.py</text>
<g stroke="#555" fill="none" marker-end="url(#b)"><path d="M240,82 H268"/><path d="M500,82 H528"/><path d="M760,82 H788"/><path d="M1280,262 H1308"/><path d="M1540,262 H1568"/><path d="M1800,262 H1828"/><path d="M925,145 V172 H125 V198"/></g>
<text x="10" y="350" fill="#555">Checks on the result: rows decoded back to audio (examples below); word starts compared with Parakeet-TDT on the decoded audio:</text>
<text x="10" y="368" fill="#555">1,676 matched words, 1,654 within 80 ms, all within 160 ms (median offset 0.0 s for Tom, 0 to -0.08 s for Gweneth).</text>
<text x="10" y="386" fill="#555">Colours: blue = clip preparation, orange = building the row, green = what training sees.</text></svg></section>"""
def examples():
    f = f"{D}/examples/examples.json"
    if not os.path.exists(f):
        return ""
    ex = json.load(open(f)); o = ["<section><h2>Training data: examples</h2><p>Decoded from the dataset rows themselves (make_examples.py): the codec codes of a dialogue's rows go back through the codec decoder, the word timeline comes from the frame-aligned text stream (start time in seconds). Full dialogue = stereo, left Tom, right Gweneth; next to it Tom's requests and Gweneth's answers as separate clips.</p>"]
    def clips(cs):
        return "".join(f"<div>{clip(D + '/' + c['wav'])} <span class=said>{html.escape(c['text'])}</span></div>" for c in cs)
    for ds, title in (("micro", "micro_v2: multi-turn, same 5 words"), ("single_v1", "single_v1: one exchange per word, gap 0.5 s"), ("single_v2", "single_v2: the same dialogue three times, different gaps")):
        o.append(f"<h3>{title}</h3><table><tr><th>example<th>full dialogue<th>Tom's requests<th>Gweneth's answers</tr>")
        for e in ex[ds]:
            tl = " ".join(f"<span style='color:{'#26b' if w['speaker'] == 'Tom' else '#c52'}'>{html.escape(w['word'])}<sub>{w['start']}</sub></span>" for w in e["timeline"])
            label = html.escape(e.get("word", e["name"])) + f"<br><span class=k>{html.escape(e['split'])}, {e['seconds']} s" + (f", gap {html.escape(e['gap'])}" if "gap" in e else "") + "</span>"
            o.append(f"<tr><td>{label}<td>{clip(D + '/' + e['wav'])}<td>{clips(e['tom_clips'])}<td>{clips(e['gweneth_clips'])}</tr><tr><td colspan=4><details><summary>word timeline</summary><div class='said tl'>{tl}</div></details></tr>")
        o.append("</table>")
    row = next((e for e in ex["single_v1"] if "prompt_gweneth_row" in e), None)
    if row:
        o.append(f"<details><summary>The text part of a dataset row (single_v1, '{row['word']}', Gweneth as system speaker; the Tom row differs in 'starts the dialogue: True' and the narrative)</summary><pre style='background:#f7f7f7;padding:8px;font-size:12px;white-space:pre-wrap'>{html.escape(row['prompt_gweneth_row'])}</pre><p>The rest of the row: dsu_ids (8 codec heads x frames, 80 ms per frame; heads 0-3 = system speaker), text_stream_ids (one token per frame: a word token on the frame where it starts, [SILENCE_PAD] elsewhere), spk_emb (the speaker's embedding).</p></details>")
    return "".join(o) + "</section>"
h += [DATAGEN, h_data, examples(), SKETCH, LOOP.replace('__CURVES__', curves())]
for title, sub, tags, change, data, nums in VERSIONS:
    h.append(f"<section><h2>{html.escape(title)}</h2><p><b>What changed:</b> {html.escape(change)}</p><p><b>Training data:</b> {html.escape(data)}</p><p><b>Result:</b> {html.escape(nums)}</p>")
    for tag in tags:
        rows = []
        for f in sorted(glob.glob(f"{D}/{sub}/eval_out/{tag}/*/*_r*_pfx.json")):
            r = json.load(open(f)); g = r["generation"]; b = f[:-5]; grp = f.split(os.sep)[-2]
            ended = g["ended_by"]
            rows.append(f"<tr><td>{grp}</td><td>{html.escape(r['meta']['word'])} r{r['meta']['repeat']}</td><td class='{'eos' if ended=='eos' else 'ok'}'>{ended} @ {g['n_generated_frames']}</td>"
                        f"<td>{clip(b + '_c2_cont.wav')}</td><td>{clip(b + '.wav')}</td><td class=said>{html.escape(g['speaker2']['transcript'][:90]) or '<i>empty</i>'}</td></tr>")
        h.append(f"<details {'open' if tag in ('heads_t5','lora','tuned_mask','base') else ''}><summary><b>{tag}</b> ({len(rows)} runs)</summary><table><tr><th>group<th>word<th>ended<th>Gweneth only<th>full dialogue<th>Gweneth's text stream</tr>{''.join(rows)}</table></details>")
    h.append("</section>")
h.append("<section><h2>Take-aways so far</h2><ul><li>Abrupt stops came from a learned end-of-dialogue cue plus a codec end-of-file artefact; the stop mask removes them.</li><li>Neither the 5-word micro run nor the 69-word single-exchange run copies the word; the micro run learned a 5-word format, the larger one memorises training words under teacher forcing.</li><li>Not explained yet: Run B says training words when teacher-forced (P≈0.99) but not in free generation. Heads load correctly; prefix tail/look-ahead timing does not explain it.</li></ul></section>")
open(out, "w").write("\n".join(h)); print(out)

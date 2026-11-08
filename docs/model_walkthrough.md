# F-Actor: model, training step, inference loop (code walkthrough, draft)

Status: written from reading the code (nothing run). Every claim carries `file:line`; the "Verify" items are what the tensor-trace
notebook should confirm. Config assumed: the released checkpoint / `scripts/train/finetune_rows.sh` flags
(`num_dsus 4`, `text_stream`, `use_speaker_embedding`, `n_delay_audio_stream 2`, `word_alignment`, Llama-3.2-1B).

## 1. The model in one paragraph

A plain Llama decoder (hidden 2048) run on `inputs_embeds`, one position per 80 ms codec frame (12.5 fps). Per frame the input is
the **sum** of 8 audio embeddings (4 NanoCodec codebooks of the system speaker + 4 of the user, one `nn.Embedding(4034, H)` per head)
and 1 text-token embedding (the system speaker's frame-aligned text). Output: the last hidden state goes into one
`Linear(H, 8*4034)` (all 8 codebook heads in parallel, **no depth transformer**, unlike Moshi) and into the tied Llama LM head for
the text token. Before the frames sit `[speaker-embedding vector][prompt tokens]` as a prefix of the sequence.
`load_dsu_model.py:79-129`, `modeling_dsu.py:29-30,54-55,143-145`.

Special ids: codes 0..4031, `audio_delay_id` 4032, `audio_eos_id` 4033 (`modeling_dsu.py:30,54-55`). Text pad = Llama `<|eot_id|>`
(128009), used as ignore-id for every label (`model.py:73-84`, `modeling_dsu.py:181`).

## 2. Data -> one training sample (`data.py:46-127`)

```
row (HF dataset)              dsu_c1_path, dsu_c2_path [4,T] codes, utterances+word timings, narratives, spk_emb_c1/c2
 prepare_dsu                  system_speaker (flag) picks which channel is "system" (heads 0-3); other = "user" (heads 4-7)   utils.py:51-92
 dsu_ids_list [8,T]           concat(system, user)                                                                    data.py:58
 adapt_to_text_stream         text stream of the SYSTEM speaker only (1 stream in training), frame-aligned            get_text_stream.py:235-293
 build_prompt                 "Generate a dialogue between you (Gweneth) and ... Narrative: - <new_narrative_s{sys+1}> ... <|SOS|>"   get_prompt.py:41-56
 spk_emb                      192-d ECAPA of the SYSTEM speaker                                                       data.py:100-104
```

Text stream (`create_text_stream`, `get_text_stream.py:35-210`): array of T frames, default `[SILENCE_PAD]`; per speech
segment `[UTTERANCE_PAD]` over the utterance span; each word's tokens start at the frame of the word's start, rest of its span
`[WORD_PAD]` (l.89-163). Backchannel / interrupt segments get `[BC]`/`[INTER]`. Words whose tokens don't fit -> sample skipped.

**Audio delay**: `n_delay_audio_stream=2`, text delay 0 -> two `audio_delay_id` frames are prepended to all 8 audio heads and two
`[SILENCE_PAD]` are appended to the text (l.213-232, 252-258). Net effect: **text leads audio by 2 frames (160 ms)**; there is no
delay between codebooks.

```
index         0     1     2     3     4     5  ...
text (own)    SIL   SIL   SIL   Okay  ,pad  WORD ...        text token of original frame k sits at index k
audio (8 hd)  DLY   DLY   a0    a1    a2    a3  ...        audio of original frame k sits at index k+2
```

## 3. Training step (`modeling_dsu.py:63-213`, `get_input_embeds_and_labels` 226-384)

Per sample (batch size 1, grad-acc 4 in `finetune_rows.sh`):

```
audio ids  [8, T+2]  + one EOS column  -> [8, T+3]           l.282-283   (EOS = 4033 on all 8 heads)
text ids   [1, T+2]  + one pad column  -> [1, T+3]           l.324-325
frame embed  = sum_h audio_embeds[h](ids[h])  +  embed_tokens(text ids)                 l.285-293,327
sequence     = [ proj(spk_emb) (1) | prompt embeds (P) | frame embeds (T+3) ]           l.346-348   (no padding at bs 1)
Llama(inputs_embeds)  -> hidden[-1]  (after final norm)                                 l.111-121
audio hidden = hidden[:, 1+P : ]  -> dsu_head -> logits [B, T+3, 8, 4034]               l.124-145
text logits  = Llama lm_head over ALL positions [B, 1+P+T+3, 1, V]                      l.157-159
```

Labels and shift (l.168-171): `labels_padded = pad(labels, (0,1), value=pad_id)`, `labels_shifted = labels_padded[..., 1:]`, so the
hidden at position i is scored on the token at i+1 (next-frame prediction). Label of the last position = pad -> masked.

- audio labels = the `[8, T+3]` id grid: codes, the delay id and the EOS id are all targets. After the shift, grid frame 0 (a delay id) is never a target (nothing precedes it), frame 1 (delay id) is predicted from frame 0, and the EOS column is predicted from the last real frame.
- **text labels = `[pad(spk) | prompt tokens | text frames]`** (l.329-339): the text head is also trained as an LM on the **prompt
  tokens** (template + narrative), not only on the stream.
- `calc_loss_on_c1_only` (and eval always): keep only head rows `[:H//2]` = system codebooks 0-3 (l.173-179). The text stream has 1 row, so
  `max(1, 1//2)=1` keeps it. So with the flag, heads 4-7 get no gradient.
- Loss = per conversation `sum(ce*mask)/sum(mask)` over (heads x frames), **separately** for audio and text, then `total = audio + text`
  (l.181-195), divided by grad_acc_steps (l.203-204). So audio and text are weighted equally regardless of token counts, and the text term
  is averaged over prompt tokens + SILENCE_PAD/UTTERANCE_PAD/WORD_PAD frames + the few word tokens.
- Trainable: everything in `DSULlama.parameters()` (Llama, audio_embeds, dsu_head, speaker_embed_proj) — codec frozen, outside the model.
  Heads/embeds are created or loaded after `from_pretrained` (`model.py:105-117`, `load_dsu_model.py:10-35`), before the Trainer builds the optimizer.
- Teacher forcing: the input at frame t contains the *true* user audio and own audio at t.

## 4. Inference loop (`generate`, `modeling_dsu.py:651-827`; driver `inference_example.py:187-314`)

Two *instances of the same weights* talk to each other ("talk_to_itself"):

```
rows:   row0 = speaker1 (Tom)      prompt0, spk_emb0, own heads 0-3, other heads 4-7
        row1 = speaker2 (Gweneth)  prompt1, spk_emb1, heads rolled by 4 so own = original 4-7          l.723-725,750-752
prefix: first N frames of dsu [1,8,N] (+delay frames) and text [1,2,N] copied in, generation starts at step N        l.734-762
loop step = N .. max_length-1:
  1. build embeds for ALL frames so far, but only the last position goes through Llama (KV cache)                    l.108-109,767-781
  2. logits[:, -1]  -> 8 heads + text, sample (T 0.9, top-k 40)  -> generated_dsu[:, :, step], generated_text[:, :, step]   l.784-798
  3. interleave_halves: row0's "other" heads := row1's sampled "own" heads and vice versa                              l.727-732,802-805
     -> the other-heads predictions of each row are thrown away; each instance hears the other's sampled audio
  4. stop if any head of any row == audio EOS (4033)                                                                    l.807-810
decode: drop first 2 (delay) frames, NanoCodec decode per channel (c1 = heads 0-3 of row0, c2 = heads 4-7)             inference_example.py:253-260
```

Text at inference: each row has its **own** 1-stream text (row1's text context = prefix stream 1 = silence pads), never interleaved
(l.804 needs >1 stream). The user's words are never given to the model as text, only as audio. Text stop-ids are dead code with 1 text stream (l.813-818).

`silence_speaker1` (l.735-748,800-801): after the prefix, row0's own heads are overwritten with the encoded-silence code of the prefix's silent channel.

## 5. Training vs inference

| aspect | training (fine-tune rows) | inference (`generate`) | possible mismatch |
|---|---|---|---|
| who is "system" | always speaker 2 (Gweneth, `--system_speaker 1`), narrative `new_narrative_s2` | **both** rows generate: Gweneth *and* Tom (requester narrative, never seen as system) | Tom instance untrained; his audio feeds Gweneth's input |
| user (heads 4-7) after request | digital silence, encoded | sampled by Tom instance (or silenced by `silence_speaker1`) | out-of-distribution input; matches the `silence_speaker1` finding |
| loss | heads 0-3 + text; heads 4-7 untrained | heads 4-7 predictions discarded | fine |
| input at frame t | true audio + true text | sampled audio + sampled text | exposure bias; rows are short (~5 s) so little context to recover |
| end of sequence | EOS(4033) appended right after reply on all 8 heads | stop when any sampled head is EOS | model learns "stop after ~reply length"; codec end-of-file codes also read as EOS (see prefix_log) |
| prompt | `build_prompt`: "- <narrative>", bc/interrupt counts from the row (0/0), "starts: False" | `prepare_prompt`; dash only with `prompt_format: training`; bc/int from config (0/0 in rows configs) | dash handled; check each config |
| text loss on prompt | yes (prompt tokens are labels) | n/a | narrative incl. the target word is LM-trained at train time |
| delay | 2 delay frames prepended, text leads | prefix carries them; `[2:]` stripped at decode | check `n_frames` vs cut in `split_prefix_audio` |
| padding | bs 1: none | 2 rows, prompts left-padded | RoPE is relative, should be harmless |
| speaker emb | system speaker's vector, position 0 | row-specific vector, position 0 (after left pad) | none expected |

## 6. Verify with the notebook (trace one real row)

1. Print the sequence layout for a row: `1 + P + (T+3)`, offsets of the three blocks, `dsu_labels`/`ts_labels` shapes, the shift, the loss mask.
2. Mask check: which positions score the reply audio, the request audio (should be none with c1-only), the prompt tokens, the word tokens.
3. Per-term loss shares on that row: audio vs text; within text: prompt / pads / word tokens.
4. Text-leads-audio-by-2: plot text ids and per-head code grids against frame index for the reply word.
5. Inference: step through 3 frames after the prefix, show what each row sees, what is sampled, and what the interleave overwrites.
6. Row 0 vs row 1 at the same step: which prompt/speaker embedding, whose audio is "own".
7. Confirm no-prefix start (`step 0` with empty dsu, l.764-781) works or not; I did not trace it.

## 7. Confirmed so far (`notebooks/01_row_to_sample.py`, toy row: 2.35 s request + reply "Okay, water.")

- Sequence = 1 spk + 91 prompt + 50 frames (49 grid frames + EOS column); a recomputation of shift/mask from the labels reproduces
  the model's audio and text loss exactly (asserted in the notebook), so the section 3 reading of the loss is right.
- Text leads audio by 2 frames: "okay" in grid row 31, the reply's first audio code in row 33.
- Only the system (Gweneth) text stream exists; Tom's request gives `[SILENCE_PAD]` text. Heads 4-7 carry Tom's request, then silence, and are not scored.
- The text EOS column is never a target (its label is the pad id, masked); only the audio EOS is learned.
- Text-loss targets (real clips, word `amazing`, 182-position sequence): 92 prompt tokens (51 %), 68 `[SILENCE_PAD]` (38 %), 10 `[WORD_PAD]`, 8 `[UTTERANCE_PAD]`
  and **2 word tokens ("okay", "amazing") = 1 %** of the 180 text targets. The text term is added in full to the audio term (352 audio targets), so the
  word the model should learn to say is 1 % of the text loss. (One row, one word; check a few more, and the toy-row numbers earlier in this section were 65 %/36/11.)
- Real NanoCodec + real TTS clips (notebook default): silence is a code (`[1818, 1584, 2870, 2151]` per codebook), not 0. The reply channel's codes leave
  that code at row 44, ~10 rows (0.8 s) *before* the reply clip starts (row 54, the request is 4.16 s): the codec is non-causal. The reply clip then has ~0.8 s of
  room-tone lead-in (not digital silence, so other codes) before "okay" (text row 62, audio row 64). The 2-frame text lead holds in the ids, but audio and
  words are far less tightly aligned than the ids suggest. One word only; the size varies per clip.

## 8. Confirmed by `notebooks/02_inference_step.py` (tiny random model, real codec and request clip)

- An unrolled copy of `generate`'s loop (step: build embeddings -> forward with KV cache -> last-frame logits -> pick tokens -> `interleave_halves` -> EOS test)
  reproduces `model.generate(do_sample=False)` exactly (asserted for K = 6 frames after a real prefix).
- Two instances: row 1 is row 0 rotated by 4 heads; after the prefix the two rows are mirror images (Tom's request is in Tom's own heads and in Gweneth's other heads).
  After each sampling step the other-head guesses are overwritten by the other instance's own heads (asserted).
- Prompts of different length are left-padded (here Tom 3 pad positions, Gweneth 0), so the frames are the last N positions of both rows.
- KV cache: `forward` rebuilds the embeddings of all frames each step but feeds only the last one once the cache is filled; cached and full-pass last-frame logits agree (4e-7).
- `forward`'s test `len(past_key_values) > 0` is evaluated twice, before and after the Llama. After it, the cache is already filled even on the first call, so with `use_cache=True`
  the audio logits come from the last position on every step, including the first (first-call logits have one frame, not N). The no-cache branch slices the audio rows with
  unpadded lengths: wrong by the pad count for a left-padded row (diff 2.2 vs 0.0 on this setup), but unused: `generate` always has the cache and `inference.py` runs batch size 1.
- With no EOS, `generate` returns `max_length - 1` new frames: `last_step` stays -1, so `[:, :, :last_step]` drops the last computed frame (`modeling_dsu.py:820-821`).
- **Normal mode (no prefix)** (`notebooks/02_inference_step.py`, section E): `generate` is called with placeholder tensors and `use_speaker_sample=0`, so it starts at step 0 with
  zero frames. The Llama sees only `[left pads | spk | prompt]`; the audio/text logits come from the last real prompt token (the end of `<|SOS|>`). Steps 0 and 1 write the delay id 4032
  on all heads instead of sampling audio (`n_delay_audio_stream=2`; the audio logits are computed and discarded); the text stream is sampled from step 0. An unrolled no-prefix loop
  equals `model.generate(do_sample=False)` for audio and text (asserted). It also returns `max_length - 1` frames.

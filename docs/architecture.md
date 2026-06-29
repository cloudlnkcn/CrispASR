# Architecture

CrispASR is structured around three layers on top of whisper.cpp.
The split between `src/` (library) and `examples/cli/` (presentation)
is deliberate: **every algorithm** — VAD, diarization, LID, CTC
alignment, HF download/cache, model registry — lives in `src/` behind
a stable C-ABI (`src/crispasr_c_api.cpp`), and every consumer (CLI,
Dart, Python, Rust, Go, Java, Ruby) reaches it through the same
symbols. The CLI keeps only presentation + UX policy.

```
┌───────────────────────────────────────────────────────────────────┐
│ examples/cli/cli.cpp (the crispasr binary)                        │
│   Parses CLI args, dispatches to backend when --backend           │
│   is set or GGUF arch is non-whisper; otherwise runs whisper_full │
│   unchanged                                                        │
├───────────────────────────────────────────────────────────────────┤
│ examples/cli/crispasr_*_cli.{h,cpp}                               │
│   Thin CLI shims for policy only — auto-download, TTY prompts,    │
│   sherpa-ONNX subprocess fallbacks. Delegate the algorithmic      │
│   work to the shared library below.                                │
├───────────────────────────────────────────────────────────────────┤
│ src/crispasr_c_api.cpp — C-ABI (shared with Dart / Python / Rust) │
│   crispasr_vad.{h,cpp}           Silero VAD + whisper-style       │
│                                  stitching, timestamp remap       │
│   crispasr_diarize.{h,cpp}       energy / xcorr / vad-turns /     │
│                                  native pyannote diarization      │
│   crispasr_speaker_embedder.{h,cpp}                                │
│                                  pluggable speaker-embedding      │
│                                  interface; TitaNet + IndexTTS    │
│                                  adapters                          │
│   crispasr_speaker_cluster.{h,cpp}                                 │
│                                  agglomerative cosine clustering  │
│                                  for globally stable speaker IDs  │
│   crispasr_lid.{h,cpp}           whisper-tiny + silero-native LID │
│   crispasr_aligner.{h,cpp}       canary-CTC + qwen3-forced-aligner│
│   crispasr_cache.{h,cpp}         HF download + ~/.cache/crispasr  │
│   crispasr_model_registry.{h,cpp} backend → canonical GGUF URL    │
├───────────────────────────────────────────────────────────────────┤
│ src/{whisper,parakeet,canary,canary_ctc,cohere,qwen3_asr,         │
│      voxtral,voxtral4b,granite_speech,silero_lid,pyannote_seg,    │
│      lid_fasttext,lid_cld3,text_lid_dispatch}.cpp                 │
│   Per-model runtimes (public C APIs)                              │
├───────────────────────────────────────────────────────────────────┤
│ src/core/      — shared model primitives (crispasr-core)          │
│   mel.{h,cpp}          log-mel spectrogram (NeMo + HF clusters)   │
│   ffn.h                SwiGLU + SiLU FFN helpers                  │
│   attention.h          Llama-style self-attention + flash-attn    │
│   gguf_loader.{h,cpp}  Unified GGUF open / weight mmap / lookup   │
├───────────────────────────────────────────────────────────────────┤
│ ggml                                                               │
└───────────────────────────────────────────────────────────────────┘
```

## `src/` — shared library surface

Every algorithm listed below is exposed as `extern "C"` functions
with a `crispasr_` prefix. The CLI, Python, Rust, and Dart bindings
all consume the same symbols.

| File | Role |
|---|---|
| `crispasr_c_api.cpp` | The C-ABI. Exports session open/close/transcribe, VAD, diarize, LID, alignment, cache, registry — everything a wrapper needs. |
| `crispasr_vad.{h,cpp}` | Silero VAD slicing + whisper-style stitching with timestamp remapping. Used by `crispasr_session_transcribe_vad`. |
| `crispasr_diarize.{h,cpp}` | Four diarizers: energy (stereo), xcorr (stereo, TDOA), vad-turns (mono, timing), pyannote (mono, GGUF; #107 added cross-slice cache + segment splitting + overlap-aware scoring). Both pyannote and sherpa/ecapa now run once globally on the full audio (#110), producing consistent speaker IDs across VAD slices. |
| `crispasr_speaker_embedder.{h,cpp}` | Pluggable speaker-embedding interface (`CrispasrSpeakerEmbedder` base class + factory). Concrete adapters: TitaNet-Large (192-d, 16 kHz) and IndexTTS-BigVGAN ECAPA-TDNN (512-d, internally resamples 16→24 kHz). Add a third by subclassing and extending the factory dispatch. |
| `crispasr_speaker_cluster.{h,cpp}` | Agglomerative single-linkage cosine clustering on speaker embeddings, with both a similarity-threshold stop and a hard `max_speakers` cap. Drives `--diarize-embedder`'s remap of pyannote-local track IDs into globally stable speaker IDs. |
| `crispasr_lid.{h,cpp}` | whisper-tiny + silero-native **audio**-LID with process-wide whisper-context cache. |
| `lid_fasttext.{h,cpp}` | Text-LID runtime for fastText supervised models — GlotLID-V3 (flat softmax, 2102 ISO 639-3 + script labels) and Facebook LID-176 (hierarchical softmax, 176 ISO 639-1 codes). Pure manual F32/F16 + on-the-fly dequant; no ggml graph. |
| `lid_cld3.{h,cpp}` | Text-LID runtime for Google CLD3 — six feature extractors (4× cbog, RelevantScript, ScriptFeature) → 80-d concat → FC + ReLU → 208-d hidden → FC → softmax over 109 ISO 639-1 labels. Pure manual F32 forward. |
| `text_lid_dispatch.{h,cpp}` | Backend-agnostic façade over `lid_fasttext` and `lid_cld3`. Peeks `general.architecture` at load time and dispatches to the matching backend; one C ABI for any text-LID GGUF. Powers `crispasr-lid` and `--lid-on-transcript`. |
| `crispasr_aligner.{h,cpp}` | canary-CTC + Qwen3-ForcedAligner forced alignment behind one entry point; filename-based dispatch. |
| `crispasr_cache.{h,cpp}` | WinHTTP / curl / wget download into `~/.cache/crispasr/`; zombie-file detection. |
| `crispasr_model_registry.{h,cpp}` | Backend → canonical GGUF URL table; fuzzy filename lookup for "did you mean …?" hints. |
| `whisper_params.h` | Shared params struct (extracted from cli.cpp, extended). |

## `examples/cli/` — presentation + policy

| File | Role |
|---|---|
| `cli.cpp` | crispasr entry point, extended with `--backend` dispatch branch. |
| `crispasr_backend.{h,cpp}` | `CrispasrBackend` abstract class, capability bitmask, factory, GGUF auto-detect. |
| `crispasr_backend_{parakeet,canary,cohere,granite,granite_nle,voxtral,voxtral4b,qwen3,fastconformer_ctc,wav2vec2,glm_asr,kyutai_stt,firered_asr,moonshine,moonshine_streaming,omniasr,gemma4_e2b,mimo_asr,vibevoice,qwen3_tts,orpheus,kokoro,chatterbox,paraformer,sensevoice,funasr,m2m100,t5}.cpp` | Per-backend thin wrapper over each model's C API. ASR backends emit `crispasr_segment`s; TTS backends (`vibevoice`, `qwen3_tts`, `orpheus`, `kokoro`, `chatterbox`) implement `synthesize(text)` instead and write 24 kHz mono WAV via `--tts-output`; the translation backends (`m2m100` for facebook m2m100 + WMT21, `t5` for MADLAD-400 / future T5 translation) implement `translate_text(text, src, tgt)` and write UTF-8 to stdout. |
| `crispasr_output.{h,cpp}` | TXT / SRT / VTT / CSV / JSON / LRC writers on `crispasr_segment`. |
| `crispasr_vad_cli.{h,cpp}` | Delegates to `src/crispasr_vad`; adds auto-download for the Silero GGUF. |
| `crispasr_lid_cli.{h,cpp}` | Delegates to `src/crispasr_lid`; adds auto-download + sherpa-ONNX subprocess fallback. |
| `crispasr_diarize_cli.{h,cpp}` | Delegates to `src/crispasr_diarize`; adds sherpa subprocess fallback + pyannote GGUF auto-download. `CrispasrSherpaCache` (#110) pre-computes the global sherpa timeline; `assign_speakers_from_global_sherpa()` assigns + splits segments at speaker turns. |
| `crispasr_model_mgr_cli.{h,cpp}` | Delegates to `src/crispasr_model_registry`; adds "Download now? [Y/n]" prompt on TTY. |
| `crispasr_aligner_cli.{h,cpp}` | Adapter converting `CrispasrAlignedWord` → the CLI's `crispasr_word` shape. |
| `crispasr_server.cpp` | HTTP server for the persistent-model mode + OpenAI-compatible endpoints. |
| `crispasr_llm_pipeline.h` | Templated audio-LLM pipeline (mel → encoder → prompt → KV decode). |
| `crispasr_run.cpp` | Top-level pipeline dispatch: resolve → detect → load → slice → transcribe → write. |

## `src/core/` — the shared model primitives

Duplicated scaffolding is bundled in a single static library,
`crispasr-core`, linked into every non-whisper model target.

| Header | Replaces | Consumers |
|---|---|---|
| `core/mel.{h,cpp}` | 7× copy-pasted STFT + mel filterbank + log + norm | parakeet, canary, canary_ctc, cohere, voxtral, voxtral4b, qwen3 |
| `core/ffn.h` | 4× inline SwiGLU blocks | qwen3, voxtral, voxtral4b, granite |
| `core/attention.h` | Llama-style self-attention with NEOX RoPE + GQA + flash-attn | voxtral, granite (via `core_granite_llm`) |
| `core/gguf_loader.{h,cpp}` | 8× identical two-pass GGUF load + mmap + tensor-map build | all non-whisper models |
| `core/fft.h` | Radix-2 Cooley-Tukey FFT (4× duplicated) | granite_speech, granite_nle (kokoro/mimo can adopt) |
| `core/cpu_ops.h` | CPU LayerNorm + matmul fallbacks (when no GPU sched is available) | granite_speech, granite_nle |
| `core/ctc.h` | `posterior_weighted_pool` + `greedy_decode_with_blank` | granite_nle (any aux-head/CTC variant can adopt) |
| `core/fastconformer.h` | NeMo-style FastConformer block (conv subsampling + MHA RPE) | parakeet, canary, canary_ctc |
| `core/conformer_ibm.h` | IBM Macaron Conformer block (FFN + Shaw RPE attn + conv module + FFN + Shaw lookup) — **sibling of `fastconformer.h`, intentionally not merged** | granite_speech, granite_nle |
| `core/granite_llm.h` | Granite-1B 40-block backbone (RMSNorm + GQA(16/4) flash-attn + RoPE + SwiGLU + µP residual scale); `is_causal` flag picks KV-cached prefill+decode (`core_attn::kv_self_attn`) vs non-causal flash (whole-sequence editing) | granite_speech, granite_nle |
| `core/qformer.h` | Windowed simplified Q-Former: pass A (LayerNorm + concat + linear + GELU) and per-window cross-attn + MLP cgraph builder | granite_nle (NAR-only — granite_speech uses a different full BLIP-2 Q-Former) |
| `core/bpe.h` | GPT-2 byte-level BPE encode + decode | granite_speech, granite_nle, voxtral, qwen3, glm-asr |
| `core/greedy_decode.h` | Autoregressive greedy decode loop with EOS handling | qwen3, voxtral, voxtral4b, granite, glm-asr |
| `core/sanm.h` | FunASR SANM encoder block (MHA + FSMN depthwise conv) | funasr, sensevoice, paraformer |
| `core/asr_context_bias.h` | Aho-Corasick CTC-WS phrase-boost trie for `--hotwords` (#98). Per-beam state in TDT/RNNT beam search | parakeet (CTC + TDT greedy + TDT beam); extensible to any CTC/TDT backend |

`core_mel::Params` spans both algorithm clusters: the NeMo family
(`ln` + per-mel z-score + `(T, n_mels)` layout) and the HF/Whisper
family (`log10` + global clip normalization + `(n_mels, T)` layout),
with knobs for `LogGuard` (add-epsilon vs max-clip), `MatmulPrecision`
(`Float` vs `Double`), `FbLayout` (`MelsFreqs` vs `FreqsMels`),
`drop_last_frame` / `drop_first_frame_if_odd`, and `pad_to_T`.

`core_gguf::WeightLoad` owns the `ggml_context`, the
`ggml_backend_buffer_t`, and the `std::map<std::string, ggml_tensor*>`
in one struct that models `std::move()` into their own state. The
mmap path has a `pread` / `fseek` fallback for filesystems that don't
support mmap.

## Whisper is the reference implementation

`src/crispasr` is **intentionally not migrated** to `src/core/` (yet)
— it's (for the time being) the battle-tested reference and the
`crispasr -m ggml-base.en.bin …` code path is byte-identical to
upstream `whisper.cpp`. This guarantee is a test gate: every
CrispASR commit that touches the CLI is checked against it.

## Regression discipline

Every `src/core/` migration commit includes a `md5sum`-level
regression test against `samples/jfk.wav`:

- **mel extraction**: bit-identical transcript + SRT on parakeet,
  canary, canary_ctc, voxtral, voxtral4b, qwen3. Cohere transcript
  is bit-identical but a single SRT boundary shifts by 80 ms due to
  the CBLAS → manual-loop matmul accumulator reorder.
- **ffn extraction**: bit-identical on qwen3, voxtral, voxtral4b,
  granite.
- **gguf_loader extraction**: bit-identical on all 8 non-whisper
  models.
- **attention extraction**: bit-identical on voxtral (only consumer
  so far).

## Backend internals

> **Note:** the snapshot below was last hand-edited in early 2026 and
> is not regenerated from the registry — treat it as a sketch, not
> ground truth. The authoritative source for what's compiled in,
> per-backend GPU support, and current capability bits is
> `src/crispasr_model_registry.cpp` and the `capabilities()` returned
> by each adapter in `examples/cli/crispasr_backend_*.cpp`.

| Backend | Arch pattern | ggml graph | Flash attn | KV cache | GPU | Shared core modules |
|---|---|:-:|:-:|:-:|---|---|
| whisper | Enc-dec transformer | ✔ | ✔ | ✔ | CUDA / Metal / Vulkan | (upstream) |
| parakeet | FastConformer + TDT | ✔ | ✔ | partial | CPU | mel, fastconformer |
| canary | FastConformer + Transformer dec | ✔ | ✔ | ✔ | CUDA / Metal | mel, fastconformer |
| cohere | Conformer + Transformer dec | ✔ | ✔ | ✔ | CUDA / Metal | mel |
| granite | Conformer + Q-Former + LLM | ✔ | ✔ | ✔ | CPU | mel, kv_self_attn, swiglu, greedy_decode, bpe |
| voxtral | Whisper enc + Mistral LLM | ✔ | ✔ | ✔ | CUDA / Metal | mel, kv_self_attn, encoder_self_attn, swiglu, greedy_decode, bpe |
| voxtral4b | RoPE enc + 3.4 B LLM | ✔ | ✔ | ✔ | CUDA / Metal | mel, kv_self_attn, encoder_self_attn, swiglu, greedy_decode, bpe |
| qwen3 | Whisper enc + Qwen3 LLM | ✔ | ✔ | ✔ | CUDA / Metal | mel, kv_self_attn, swiglu, greedy_decode, bpe |
| fc-ctc | FastConformer + CTC | ✔ | ✔ | — | CPU | mel, fastconformer |
| wav2vec2 | CNN + Transformer + CTC | ✔ | — | — | CUDA / Metal | gguf_loader |
| glm-asr | Whisper enc + Llama LLM | ✔ | ✔ | ✔ | CPU | mel, kv_self_attn, swiglu, greedy_decode, bpe |
| kyutai-stt | Mimi codec + causal LM | ✔ | ✔ | ✔ | CPU | gguf_loader |
| firered-asr | Conformer + CTC + beam dec | ✔ | ✔ | ✔ | CPU | mel, gguf_loader |
| moonshine | Conv + 6L enc-dec | ✔ | ✔ | ✔ | CPU | (vendored) |
| moonshine-streaming | Sliding-window enc + dec | ✔ | ✔ | ✔ | CPU | (vendored) |
| omniasr | wav2vec2 enc + CTC / LLM | ✔ | ✔ | CTC: — / LLM: ✔ | CPU | gguf_loader, kv_self_attn, swiglu |
| gemma4-e2b | Conformer enc + Gemma4 LLM | ✔ | ✔ | ✔ | CUDA / Metal | gguf_loader, kv_self_attn, swiglu |
| mimo-asr | wav2vec2 enc + Qwen2 LM | ✔ | ✔ | ✔ | CUDA / Metal | gguf_loader, kv_self_attn, swiglu |
| ark-asr ⚠️*exp/WIP* | Whisper enc (partial RoPE) + Qwen2.5-3B LM | ✔ | — | CPU only | CPU (GPU emits no tokens) | mel, ffn, gguf_loader, kv_self_attn, swiglu, bpe |
| vibevoice | σ-VAE + Qwen2 (ASR) / TTS LM (synth) | ✔ | ✔ | ✔ | CUDA / Metal | gguf_loader |
| kokoro | StyleTTS2 BERT + ProsodyPredictor + iSTFTNet | ✔ | — | — | CPU | gguf_loader, fft, ffn |
| qwen3-tts | Qwen3 talker + 12 Hz codec + code-predictor | ✔ | ✔ | ✔ | CUDA / Metal | gguf_loader, kv_self_attn, swiglu |
| orpheus | Llama-3.2 talker + SNAC RVQ codec | ✔ | ✔ | ✔ | CUDA / Metal | gguf_loader, kv_self_attn, swiglu |
| chatterbox | T3 (Llama / GPT-2) + S3Gen (Conformer + UNet1D CFM + HiFTGen) | ✔ | ✔ | ✔ | CUDA / Metal | gguf_loader, kv_self_attn, swiglu, fft |
| zonos-tts | 26L GQA transformer + 9-codebook DAC @ 44.1 kHz; CFG; voice cloning from WAV | ✔ | ✔ | ✔ | CUDA / Metal | gguf_loader, kv_self_attn, gated_mlp |
| m2m100 | facebook/m2m100 12L+12L transformer (text-to-text translation; WMT21 4.7B variant via `--backend m2m100-wmt21`) | ✔ | — | ✔ (cross-attn) | CUDA / Metal | gguf_loader, kv_self_attn |
| madlad / t5 | T5 encoder-decoder (MADLAD-400 12L+12L, gated-GELU, RMSNorm, bucketed rel-pos bias). Tokens match Python SP bit-by-bit; translation outputs match the HF reference. | ✔ | — | ✔ (cross-attn) | CUDA / Metal | gguf_loader, ffn |

### Architecture families

- **Feedforward CTC** (wav2vec2, omniasr-CTC, fc-ctc, firered-asr):
  No decoder, no KV cache. Fastest. No native punctuation.
- **Encoder-decoder** (whisper, canary, cohere, moonshine,
  moonshine-streaming): cross-attention KV cache, autoregressive
  text decoder.
- **Audio-LLM** (granite, voxtral, voxtral4b, qwen3, glm-asr,
  omniasr-LLM, gemma4-e2b, mimo-asr, vibevoice): audio features
  injected into LLM embedding space, KV-cached autoregressive
  decoding.
- **Transducer** (parakeet, reazonspeech): LSTM predictor + joint
  network, frame-synchronous TDT/RNNT decoding. Supports greedy
  (default), label-looping beam search (`-bs N`), and MAES (Modified
  Adaptive Expansion Search — `CRISPASR_PARAKEET_MAES=1 -bs N`), with
  per-beam LSTM state snapshots and per-beam hotword trie tracking.
  Models with `n_tdt_durations=0` auto-select the pure RNNT decode
  path; local attention (`att_context_size`) is supported for long-form
  models like ReazonSpeech.
- **Codec + LM** (kyutai-stt): neural audio codec (RVQ) →
  token-based LM.
- **TTS — codec / vocoder pipeline**:
  - **Discrete-token codec + vocoder** (qwen3-tts, orpheus): talker
    LM emits codec tokens; a separate decoder GGUF (12 Hz codec /
    SNAC RVQ) renders the audio. Two-GGUF runtime.
  - **Flow-matching mel + iSTFT vocoder** (chatterbox / chatterbox-
    turbo / kartoffelbox-turbo / lahgtna-chatterbox): T3 emits speech
    tokens; S3Gen runs an UpsampleConformerEncoder + UNet1D CFM
    (10-step Euler for base / 2-step meanflow for turbo) producing a
    mel-spectrogram, then HiFTGenerator (conv chains + Snake +
    iSTFT) renders 24 kHz audio. Two-GGUF runtime.
  - **Realtime σ-VAE** (vibevoice in TTS mode): 4L base LM + 20L TTS
    LM + DPM-Solver++ + σ-VAE decoder.
  - **StyleTTS2 / iSTFTNet** (kokoro): BERT + ProsodyPredictor
    + iSTFTNet decoder, single-shot (no AR).
- **Text-to-text translation**:
  - **m2m100** (also runs WMT21 dense-24-wide-en-x via the same
    runtime — see `--backend m2m100-wmt21`): SentencePiece BPE
    + transformer encoder + transformer decoder (with cross-attn
    KV cache) + greedy decode. Source/target language codes prefix
    the encoder/decoder input streams.
  - **t5_translate / madlad** (MADLAD-400 3B-mt and any future
    T5-family translation model): T5 encoder-decoder with gated-GELU
    FFN, RMSNorm, bucketed relative-position bias, SentencePiece
    256K Viterbi-unigram tokenizer. Target language as `<2xx>` input
    prefix on MADLAD; encoder is otherwise language-agnostic. Tokens
    match Python SP bit-by-bit; translation outputs match the HF
    reference (validated end-to-end on flan-t5-small + MADLAD-3b).

  Both are driven by `--text "..." -sl <src> -tl <tgt>`.

### Optimization opportunities

- **Beam search** for all encoder-decoder and Audio-LLM backends —
  PLAN #63 added it for several LLM backends; whisper + firered-asr
  always had it.
- **Fused QKV** (single matmul for Q / K / V projections) — used in
  CrispEmbed, applicable to all attention layers; landed for
  qwen3-tts talker (Q8_0/Q4_K-skipped) under env flag
  `QWEN3_TTS_FUSED_QKV`.
- **Temperature sampling** for the few backends that don't have it
  (glm-asr, kyutai-stt, firered-asr, moonshine, omniasr-LLM) via
  `core_greedy_decode`.
- **GPU offload** for the still-CPU-only backends — needs
  `ggml_backend_sched` with GPU primary.

---

## Per-backend architecture details

Detailed architecture notes for backends whose design warrants more than
a one-line summary. The [README backend table](../README.md#asr-backends)
links here for each entry.

### granite / granite-4.1 / granite-4.1-plus / granite-4.1-nar

**granite** (`granite-speech-{3.2-8b, 3.3-2b, 3.3-8b}`, `granite-4.0-1b-speech`):
Conformer encoder + BLIP-2 Q-Former + Granite LLM (μP scaling).

**granite-4.1** (`granite-speech-4.1-2b`): Same architecture as 4.0
(16-layer Conformer + Q-Former + Granite LLM); "2B" = full system.
Encoder runs as a single ggml graph by default with per-layer Shaw RPE
in attention (PLAN #16) — bit-near-identical to the per-op CPU loop,
~2.1× faster end-to-end on M1+Q4_K. `GRANITE_DISABLE_ENCODER_GRAPH=1`
falls back to the CPU loop.

**granite-4.1-plus** (`granite-speech-4.1-2b-plus`): 4.1 + 2-layer
encoder hidden-state concatenation (1024+1024=2048 projector input);
emits punctuated / capitalised transcripts by default. `cat_hidden_layers`
post-norm tensors are captured inline in the graph and `ggml_concat`-ed
with the final encoder output, so PLUS rides the GPU path too (~2.5×
end-to-end on M1+Q4_K).

**granite-4.1-nar** (`granite-speech-4.1-2b-nar`): 4.1 with
non-autoregressive decoder — single LLM forward over [audio, text+slots]
+ slot argmax decode (`is_causal=False` everywhere); 4-layer encoder
hidden-state concatenation + posterior-pooled BPE auxiliary CTC head;
bit-exact end-to-end on JFK via `crispasr-diff granite-nle`. Wired into
the main CLI as `--backend granite-4.1-nar` (alias `granite-nar`).
Encoder also runs as a single ggml graph (sibling builder with self-cond
residual + snapshot concat + final CTC logits), ~3× faster end-to-end on
M1+Q4_K.

### kokoro

StyleTTS2 / iSTFTNet (BERT + ProsodyPredictor + iSTFTNet decoder, 82M
params); per-voice GGUF; in-process libespeak-ng phonemizer with LRU
cache; auto-routing for `-l de` swaps in the German-trained backbone +
cascading voice fallback.

Models: [`hexgrad/Kokoro-82M`](https://huggingface.co/hexgrad/Kokoro-82M)
+ [`dida-80b/kokoro-german-hui-multispeaker-base`](https://huggingface.co/dida-80b/kokoro-german-hui-multispeaker-base)
(German backbone) + [`kikiri-tts/kikiri-german-{victoria,martin}`](https://huggingface.co/kikiri-tts)
(German voicepacks).

### orpheus

Llama-3.2-3B-Instruct talker (28L, 3072 d) + SNAC RVQ codec (3
codebooks × 4096 @ 24 kHz); 8 baked English speakers
(`tara`/`leah`/`leo`/...). Pick the speaker with `--voice <name>` and
pass `--temperature 0.6` (engine_class.py default — greedy loops).

Drop-in DE checkpoint variants:
- `--backend kartoffel-orpheus-de-natural` — [`cstr/kartoffel-orpheus-3b-german-natural-GGUF`](https://huggingface.co/cstr/kartoffel-orpheus-3b-german-natural-GGUF), 19 speakers, ASR-roundtrip word-exact via parakeet-v3 -l de
- `--backend kartoffel-orpheus-de-synthetic` — [`cstr/kartoffel-orpheus-3b-german-synthetic-GGUF`](https://huggingface.co/cstr/kartoffel-orpheus-3b-german-synthetic-GGUF), 4 speakers + 12 emotions + 5 outbursts via `{Speaker} - {Emotion}: {text}` syntax
- `--backend lex-au-orpheus-de` — `lex-au/Orpheus-3b-German-FT-Q8_0.gguf`

### chatterbox / chatterbox-turbo / kartoffelbox-turbo / lahgtna-chatterbox

Two-GGUF runtime: T3 AR text→speech-tokens + S3Gen flow-matching
speech-tokens→24 kHz waveform.

**T3 (Text-to-Tokens)**: Llama-30L for base/lahgtna, GPT-2-24L for
turbo/kartoffelbox-turbo.

**S3Gen (Tokens-to-Speech)**: UpsampleConformerEncoder + UNet1D CFM +
HiFTGenerator vocoder. Turbo uses 2-step meanflow CFM (vs 10-step cosine
for base). Default voice baked into T3 (`conds.*`); voice cloning
goes through `models/bake-chatterbox-voice-from-wav.py`, which runs
upstream `prepare_conditionals(wav)` (VoiceEncoder LSTM →
256-d speaker emb, CAMPPlus TDNN → 192-d x-vector, S3Tokenizer →
prompt tokens, 24 kHz mel extractor → prompt mel) and writes a small
voice GGUF (~150-200 KB) using the same tensor names the runtime
already accepts for the built-in voice. `--voice <voice.gguf>` then
loads it via `chatterbox_load_voice_gguf` into a separate
`voice_ctx_w` / `voice_buf_w` and rebinds `ctx->conds.*` pointers,
leaving the original baked-in default-voice tensors allocated but
unreferenced. In-process WAV → cond extraction is fully ported across four
modules — VE (`src/chatterbox_ve.cpp`), S3Tokenizer V2
(`src/chatterbox_s3tok.cpp`), CAMPPlus + Kaldi fbank
(`src/chatterbox_campplus.cpp` + `src/core/kaldi_fbank.{h,cpp}`),
and 24 kHz Matcha mel (in `chatterbox_campplus.cpp`) — all verified
bit- or fp32-rounding-tight against PyTorch via `crispasr-diff
chatterbox`. A polyphase Kaiser-windowed sinc resampler
(`src/core/audio_resample.{h,cpp}`) handles the 16 ↔ 24 kHz
conversion. The runtime forks on the input rate when `--voice` is a
`.wav` path: 24 kHz input triggers atomic cloning (all five conds
derived from one source — actually clones the speaker on simple
prompts, verified by ASR roundtrip on the JFK clip with Q4_K +
`--no-gpu`); 16 kHz input keeps a T3-side-only partial path that
**does NOT actually clone** (S3Gen renders with the default voice's
`gen.*` triple to avoid the inconsistent-conditioning silence trap;
output sounds like the default voice, not the reference speaker).
Two known issues affect end-to-end output: F16 T3 + GPU produces
broken audio (pre-existing bug; use Q4_K + `--no-gpu` until fixed),
and T3 sampling can drift on long prompts. The python baker workflow
remains the recommended path for production-quality cloning.
S3Gen GGUF is auto-discovered next to T3 or passed via `--codec-model`.
See [`docs/tts.md`](tts.md#voice-cloning) for the workflow.

For multilingual base Chatterbox, `-l <code>` sets
`chatterbox_set_language()` in the CLI adapter. The runtime validates the
`[code]` token against the embedded tokenizer and prepends it to the text
token sequence before the `[STOP]`-wrapped T3 prompt is built. The
2026-06-18 rebuilt `cstr/chatterbox-GGUF` artifacts use the paired
`grapheme_mtl_merged_expanded_v1.json` tokenizer (`2454` T3 text vocab
entries, `2454` tokenizer tokens, `265` merges); older mismatched GGUFs
are rejected at load time. A Q4_K smoke check confirmed `-l fr` inserts
`[fr]` (id 634) and changes the generated speech tokens and waveform.

Variants:
- [`cstr/chatterbox-GGUF`](https://huggingface.co/cstr/chatterbox-GGUF) — base multilingual v3
- [`cstr/chatterbox-turbo-GGUF`](https://huggingface.co/cstr/chatterbox-turbo-GGUF) — 350M distilled, meanflow
- [`cstr/kartoffelbox-turbo-GGUF`](https://huggingface.co/cstr/kartoffelbox-turbo-GGUF) — German fine-tune of turbo
- [`cstr/lahgtna-chatterbox-v1-GGUF`](https://huggingface.co/cstr/lahgtna-chatterbox-v1-GGUF) — Arabic fine-tune of base

Conformer rel-pos parity gap closed in §80 — encoder_out now bit-exact
to Python reference.

### zonos-tts

Zyphra Zonos-v0.1-transformer: 26-layer GQA transformer (d=2048,
n_heads=16, n_kv=4, head_dim=128) conditioned on speaker embedding +
phoneme tokens → 9-codebook DAC codes @ 86 Hz → 44.1 kHz PCM via the
DAC decoder GGUF.

**Conditioning prefix**: speaker embedding (512-d float32 from a reference
WAV, encoded externally or via `--voice <ref.wav>`) is projected through
an MLP and prepended as prefix tokens before the phonemised text. CFG
(classifier-free guidance) runs a conditioned path and an unconditioned
path in parallel, blending with `cfg_scale=2.0`: `uncond + 2*(cond − uncond)`.

**Backbone**: standard pre-norm transformer with RMSNorm,
GatedMLP (`fc2(y * silu(gate))`, first chunk = y, second = gate),
and consecutive-pair RoPE (`GGML_ROPE_TYPE_NORMAL`) matching
x_transformers' `apply_rotary_emb` (reshape to (…, d/2, 2), rotate
paired elements). GQA with 4 KV heads shared across 16 query heads.

**AR decode**: 9-codebook delay pattern (codebook k shifted by k+1
positions). Each step samples one token per codebook via min-p=0.1 +
temperature=1.0 + repetition penalty (factor=3.0, window=2). EOS is
only predicted on codebook 0; other codebooks have EOS masked to −∞.

**Quantisation**: selective quantization is required. Uniform Q4_K
inflates the EOS logit at AR step 0 by ~0.9 units (−1.125 → +0.21),
making P(EOS) > 60 % and causing synthesis to fail on every seed.
The `crispasr-quantize` tool keeps `heads.*`, `embeddings.*`, and
`prefix_conditioner.*` at F16 while quantizing the 210 backbone
projections — reducing the model to 931 MB (vs 872 MB for full-Q4_K).
A 3-retry guard in the runtime handles residual step-0 failures
(~25 % of seeds, 100 % resolved within 2 retries). Default via
`-m auto` is **Q8_0** (1.6 GB); Q4_K (931 MB) is safe with the
above caveats; F16 (3.0 GB) is the reference.

**DAC codec**: the companion `dac-44khz-f16.gguf` (104 MB) is a purely
convolutional architecture — all weight tensors have kernel-size ≤ 16 as
ne[0], making block quantization impossible. F16 is the only usable quant.

**Voice cloning**: pass `--voice <ref.wav>` at the CLI or set
`ZONOS_SPEAKER_EMB_PATH=/path/to/jfk_speaker_emb.bin` (raw float32,
512 floats). The runtime calls `zonos_tts_set_voice(ctx, path)` which
decodes the WAV via `src/core/audio_resample` and runs the VoiceEncoder
MLP to produce the speaker embedding.

GGUFs: [`cstr/zonos-v0.1-transformer-GGUF`](https://huggingface.co/cstr/zonos-v0.1-transformer-GGUF)
(AR transformer) + [`cstr/dac-44khz-GGUF`](https://huggingface.co/cstr/dac-44khz-GGUF)
(DAC decoder, auto-discovered as a sibling or via `--codec-model`).

### omniasr (CTC + LLM + Unlimited)

wav2vec2-style CNN frontend (7 layers, stride 5+2×6=320) + 24–48L
transformer encoder + either CTC head or 12L LLaMA decoder (SwiGLU,
RoPE, d=4096, 8 heads).

**CTC variant** (`omniasr`, `omniasr-300m`): greedy argmax with CTC blank
collapse (blank = token 0 for both v1/fairseq2 and v2/HF formats). Only
v2 (HF transformers) GGUFs work — the v1 fairseq2 `.pt` format produces
empty output because fairseq2's model loader applies weight transforms
we cannot replicate. The 300M model's positional encoding degrades beyond
~7 s; the runtime auto-chunks long audio into 5 s segments. The 1B model
handles all lengths.

**LLM variant** (`omniasr-llm-300m-v2`): Encoder projection (1024→4096)
+ language conditioning (1694 FLORES-200 codes) + autoregressive decode.
Best quality for the 1600+ language family.

**Unlimited variant** (`omniasr-llm-unlimited-300m-v2`): Same architecture
but trained with a streaming segment-token protocol. Audio is split into
15-second segments; each segment is decoded independently with a segment
marker token that signals whether more audio follows. Three special tokens
above vocab_size in tok_emb: `streaming_lang` (lid marker),
`last_segment`, `regular_segment`. Auto-detected at load time from
tok_emb shape (vocab_size + 3). Supports arbitrarily long audio input.

### vibevoice

σ-VAE ConvNeXt encoders + Qwen2.5-7B decoder. Dual-mode: ASR (with
timestamps, diarization, hotwords) and TTS (DPM-Solver++ flow matching).

### mimo-asr

6L input_local_transformer (1024d) + 36L Qwen2 LM (4096d, 32Q/8KV);
8-channel RVQ codes from separate MiMo-Audio-Tokenizer GGUF. Mandarin
(Wu/Cantonese/Hokkien/Sichuanese dialects) + English + code-switching.

**Tokenizer is a separate file.** `--auto-download` fetches both the LM
(`cstr/mimo-asr-GGUF`) and the tokenizer (`cstr/mimo-tokenizer-GGUF`)
into `~/.cache/crispasr/`; the runtime auto-discovers
`mimo-tokenizer-q4_k.gguf` next to the LM. Override with `--codec-model
PATH/mimo-tokenizer-q4_k.gguf` if you keep the tokenizer elsewhere.

### ark-asr

> ⚠️ **Experimental / WIP.** Validated verbatim on English + German (Q8_0, CPU),
> but rough edges remain (see below). Single self-contained GGUF.

[`AutoArk-AI/ARK-ASR-3B`](https://huggingface.co/AutoArk-AI/ARK-ASR-3B) — 19-language
ASR = **Whisper-large-v3 encoder with partial interleaved RoPE** (rot_dim 32 of
head_dim 64, θ=10000, applied to Q+K only; `k_proj` has no bias; the encoder's own
final LayerNorm is dropped) + **MLP adapter** (LayerNorm → merge 4 consecutive
frames → Linear 5120→4096 → GELU → Linear 4096→2048) + stock **Qwen2.5-3B decoder**
(2048d, 36L, GQA 16Q/2KV, SwiGLU, RMSNorm, θ=1e6, tied embeddings). The
`<|audio|>` (151663) placeholder embeddings are overwritten by the first
N = `((mel_frames+1)//2)//4` adapter frames; mel is the stock WhisperFeatureExtractor
recipe (128-bin, n_fft 400, hop 160). Convert with
`models/convert-arkasr-to-gguf.py` (`--outtype f16|q8_0`).

**Known limitations (WIP):**
- **CPU only.** The GPU/`ggml_backend_sched` path emits no tokens (the
  cross-backend audio-injection / KV graph hits the same weight-less-first-op
  copy class as mimo-asr PLAN #115). Defaults to CPU; opt into the unvalidated
  GPU path with `CRISPASR_ARKASR_GPU=1`.
- **Language steering is experimental.** ARK is promptless, so language can drift
  per 30 s chunk on long audio. Passing `-l <lang>` injects a best-effort
  "Transcribe the audio in <Language>." instruction (the model was not trained on
  instructions); the default (no `-l`) is the validated promptless path.
- **Decode is slow** (per-step graph rebuild; no step-graph cache yet).

### moss-audio

32-layer Whisper-style audio encoder (1280d, 20 heads, 128-mel,
sliding-window attention w=100) with **DeepStack** 3-tap cross-layer
injection + 36-layer Qwen3 LM (2560d, 32Q/8KV, SwiGLU, RoPE θ=1M).
Apache-2.0. First audio-understanding backend — supports transcription,
audio QA, scene description, and time-aware ASR via prompt.

**DeepStack architecture:** the encoder captures intermediate outputs at
layers 8, 16, and 24. Each tap is projected through an independent
GatedMLP (1280→8192→2560) into the LM embedding space. These projections
are injected as residual adds at LM blocks 0, 1, and 2, preserving
multi-resolution audio features (low-level prosody/transients alongside
high-level semantics) through the LM's early layers.

**Audio front-end:** 128-bin log-mel → 3×Conv2d (stride 2 each, 8×
downsample total) → stem_proj Linear(7680, 1280) → sinusoidal position
embeddings → 32 encoder blocks. Slaney mel filterbank normalization.
Encoder output padded to 3000 frames (Whisper 30s convention).

**Prompt format:** Qwen3 chat template with time-marker tokens inserted
at fixed intervals between audio frame embeddings. Supports custom prompts
via `--prompt` / `set_ask()` for audio understanding tasks beyond ASR.

### qwen3-tts

Qwen3 talker LM + 12 Hz RVQ speech tokenizer. Three variants:
- `qwen3-tts-0.6b-base` — 0.6B talker, baked voice pack or WAV + `--ref-text`
- `qwen3-tts-1.7b-base` — 1.7B talker, higher quality
- `qwen3-tts-1.7b-voicedesign` — natural-language voice description via `--instruct`

### csm

Sesame CSM-1B (`sesame/csm-1b`, Apache-2.0), one GGUF, 24 kHz. Three
stages run per the original two-transformer + codec design:

- **Backbone** — Llama-3.2 1B (16L, 2048d, 32 heads / 8 KV, SwiGLU,
  RMSNorm, RoPE θ=500k). Text is Llama-3.2 BPE; each frame sums 32 audio
  codebook embeddings + 1 text embedding (masked per role). Autoregressive
  over frames; its head samples codebook 0.
- **Depth decoder** — Llama-3.2 100M (4L, 1024d), KV cache reset per frame.
  Given the backbone hidden state + codebook-0 embedding (projected
  2048→1024), it fills codebooks 1–31 with position-specific heads.
- **Mimi codec** — Kyutai Mimi: 32-codebook RVQ dequant → depthwise
  upsample → 8L transformer → SEANet decoder → 24 kHz PCM. The RVQ
  codebooks are `embed_sum / cluster_usage.clamp(min=1e-5)` — the converter
  must apply that normalization (a wrong `max(cu,1.0)` clamp left ~96 % of
  codes un-normalized and produced buzzing; §135). EOS when all 32
  codebooks of a frame are 0.

GGUF built from the HF-transformers checkpoint (rotate_half layout → NEOX
rope). Diff via `crispasr-diff csm` (backbone per-layer + depth + RVQ vs
the manual PyTorch reference).

### dia

Nari Labs Dia 1.6B (`nari-labs/Dia-1.6B`, Apache-2.0, ~1.6B params),
single GGUF, 44.1 kHz mono. Dialogue TTS with inline `[S1]`/`[S2]`
speaker tags. Architecture:

- **Text encoder** — 12L Llama-style transformer (byte-level tokenizer,
  1024-d, SwiGLU, RMSNorm, RoPE). Input is raw UTF-8 bytes + special tokens.
- **AR decoder** — 18L transformer with cross-attention (2048-d, GQA
  16q/4kv). Generates 9-codebook DAC tokens using a delay pattern
  `[0, 8, 9, 10, 11, 12, 13, 14, 15]` — channel k is delayed by
  `delay[k]` steps; after EOS on channel 0, generation continues for
  max\_delay (15) more steps to flush.
- **DAC codec** — 9-codebook RVQ → 44.1 kHz PCM. Shared with the Zonos
  port (#130); a separate DAC GGUF (`--codec-model`) or embedded weights
  both work.

Classifier-Free Guidance (CFG) runs batch=2 (conditional + unconditional);
`logits = uncond + cfg_scale * (cond - uncond)`, default `cfg_scale=3.0`.
Sampling: temperature (default 1.2), top-p (0.95), top-k (45), seedable.

### speecht5

Microsoft SpeechT5 (`microsoft/speecht5_tts`, MIT, ~80M params), single
GGUF (~300 MB F16), 16 kHz mono. Architecture:

- **Text encoder** — Embedding(81, 768) + ScaledPositionalEncoding +
  LayerNorm + 12L transformer with SpeechT5RelativePositionalEncoding
  (embedding-based, max_rel=160). Post-LN, GELU FFN.
- **Speech decoder** — AR over continuous mel frames (no codebook tokens).
  Prenet: 2× Linear(80→256)+ReLU + Linear(256→768) + ScaledPosEnc +
  speaker projection Linear(1280→768)+ReLU (512-d x-vector concatenated).
  6L decoder (self-attn KV-cached + cross-attn + FFN, post-LN).
  feat_out Linear(768→160) → reshape (reduction_factor=2, 80 mel bins).
  prob_out Linear(768→2) → sigmoid → stop token.
- **Postnet** — 5-layer Conv1d(k=5) + BatchNorm + Tanh stack, residual
  add to feat_out mel.
- **HiFi-GAN vocoder** — `microsoft/speecht5_hifigan`, 4× upsample
  (rates [4,4,4,4]) with MRF resblocks (kernels [3,7,11],
  dilations [[1,3,5]×3]) → 16 kHz PCM. Weight-norm fused at conversion.

Speaker conditioning requires a 512-d x-vector (e.g. from
`Matthijs/cmu-arctic-xvectors`), passed as raw float32 `.bin` via
`--voice`. The prenet uses "consistent dropout" at inference in the
original; the C++ port omits it (deterministic).

### fastpitch

NVIDIA FastPitch (`nvidia/tts_en_fastpitch`, CC-BY-4.0, ~60M params),
single GGUF (~230 MB including HiFi-GAN vocoder), 22 kHz mono.
**Non-autoregressive** — the entire mel spectrogram is generated in a
single parallel forward pass (no AR loop, no sampling, no KV cache).

- **Text encoder** — Embedding(115, 384) + sinusoidal PE (cat [sin, cos])
  + 6L FFTransformer (1-head, d_head=64, d_inner=1536, Conv1d(k=3) FFN,
  post-LN). Bidirectional (no causal mask).
- **Duration predictor** — 2-layer Conv1d(k=3, 256 filters) + LayerNorm
  + ReLU → Linear(256→1). Output: log-durations per token, converted via
  `round(exp(x) - 1)`.
- **Pitch predictor** — same architecture as duration predictor. Output:
  normalized pitch per token.
- **Length regulator** — repeat_interleave encoder features by rounded
  durations. Pitch expanded similarly, then embedded via Conv1d(1→384, k=3)
  and added to the expanded features.
- **Mel decoder** — 6L FFTransformer (same architecture as encoder), then
  Linear(384→80) → mel spectrogram.
- **HiFi-GAN vocoder** — `nvidia/tts_hifigan`, conv_pre(80→512) + 4×
  upsample (rates [8,8,2,2], kernels [16,16,4,4]) with MRF resblocks
  (kernels [3,7,11], dilations [[1,3,5]×3]) + conv_post → 22 kHz PCM.
  Weight-norm fused at conversion.

Deterministic output — same input always produces the same audio.
Tokenizer: ARPABET vocabulary (115 tokens: space + 24 consonants +
45 stressed vowels + 26 lowercase chars + apostrophe + 15 punct +
pad/blank/oov). Currently character-level; G2P not yet implemented.

### pocket-tts

Kyutai Pocket TTS (100M, MIT / CC-BY-4.0). Continuous-latent AR TTS —
architecturally unique: no codebook, no RVQ, no softmax.

Pipeline: SentencePiece (4000 vocab) → 4001×1024 embedding LUT →
6-layer causal transformer (1024D, 16H, RoPE, pre-norm LN, GELU FFN)
→ consistency head (SimpleMLPAdaLN: 6 ResBlocks + FinalLayer, 512D,
AdaLN conditioning from timestep embeddings + backbone output) →
one-step Lagrangian Self Distillation (LSD) decode → continuous 32-dim
float vectors at 12.5 Hz → Mimi VAE decoder (depthwise ConvTranspose1d
upsample ×16 + 2L causal transformer with RoPE and LayerScale +
SEANet CNN decoder with causal convolutions, ratios [6,5,4]) → 24 kHz
mono PCM. Voice cloning: ref audio → Mimi VAE encoder → linear project
→ prepend to transformer KV cache. Model:
`kyutai/pocket-tts-without-voice-cloning` (no encoder weights) or
`kyutai/pocket-tts` (full, with encoder for voice cloning).

### parler-tts

Prompt-conditioned TTS from
[parler-tts/parler-tts-mini-v1.1](https://huggingface.co/parler-tts/parler-tts-mini-v1.1)
(Apache 2.0, ~900M params), single GGUF (~1.8 GB F16), 44.1 kHz mono.

| Component | Architecture | Details |
|---|---|---|
| T5 encoder | flan-t5-large encoder | d=1024, 16 heads, 24 layers, gated-GELU FFN, RMSNorm, relative position bias |
| MusicGen decoder | Causal transformer | d=1024, 16 heads, 24 layers, 9 codebooks, sinusoidal PE, LayerNorm, delay pattern |
| DAC codec | Descript Audio Codec 44 kHz | 9 codebooks × 1024, Snake activations, 4 upsample blocks (8×8×4×2 = 512×) |

Voice characteristics are controlled via `--instruct` (natural language
description). Temperature=1.0 required (greedy produces degenerate output;
the model is trained with stochastic sampling). `--seed` wired for
reproducibility (note: C++ `std::mt19937` differs from PyTorch RNG).

Tokenizer: LLaMA-2 sentencepiece BPE (90714 tokens). The GGUF stores the
original sentencepiece scores and a `parler.tokenizer.is_bpe=true` flag so
the runtime auto-selects `core_spm::tokenize_bpe` (iterative best-merge)
instead of the Viterbi unigram path. Quantized GGUFs preserve DAC codec
weights at F16 (audio codecs are precision-sensitive).

### m2m100 / wmt21

12L encoder + 12L decoder transformer (d=1024, 16 heads, FFN=4096, ReLU,
pre-norm) + SentencePiece BPE (128K vocab, 100 language codes) +
sinusoidal positional embeddings + cross-attention KV cache + greedy
decode. en→de exact match to the Python reference; Q8_0 (~502 MB)
preserves quality.

**WMT21** (`wmt21-dense-24-wide-en-x` + `wmt21-dense-24-wide-x-en`):
Same architecture scaled to 4.7B parameters (24L encoder, wider FFN).
Won the WMT21 News competition. Routes through the same m2m100
runtime. **Two separate checkpoints**: `en-x` for English-source
translation, `x-en` for English-target. Pick whichever matches your
direction (`-sl`/`-tl`) — the auto-download path picks `en-x` by
default; load `x-en` explicitly with `-m <path>` for X→English.

### paraformer

FunASR Paraformer-zh — non-autoregressive (single-pass decode). 220M
params, character-level tokenizer (8404 vocab), primarily Mandarin
Chinese + English.

```
Audio → Kaldi fbank (80 mel) → LFR(7,6) → CMVN → 50 SANM encoder blocks
      → CIF predictor (Conv1d + sigmoid → fire-when-alpha≥1.0)
      → 16 NAR decoder blocks (FFN → FSMN → cross-attn)
      → decoders3 post block → after_norm → output_layer → argmax
```

Key architectural points:
- Encoder reuses `core_sanm::build_block()` (shared with funasr + sensevoice)
- Decoder block order is **FFN → FSMN → cross-attn** (not the typical self-attn → cross-attn → FFN)
- FSMN = depthwise conv (no Q/K/V self-attention in the decoder)
- CIF predictor is CPU-only (sequential accumulation loop)
- Output: character sequence with `@@` BPE continuation markers; space insertion between consecutive Latin-script word tokens

F16 (421 MB), Q4_K (123 MB), Q8_0 (227 MB) at `cstr/paraformer-zh-GGUF`.
All three produce byte-identical transcripts on Chinese + English test clips.

### funasr / fun-asr-mlt-nano

FunAudioLLM Fun-ASR-Nano-2512 — 70 SANM encoder blocks + 2-block
Transformer adaptor + Qwen3-0.6B LLM AR decoder. Uses the same SANM
encoder primitive as paraformer + sensevoice (`core_sanm::build_block`).
ChatML prompt template; audio embedded at `<|startofspeech|>` slot.

### sensevoice

FunAudioLLM SenseVoice-Small — encoder-only multi-task ASR. Same 70-block
SANM encoder as funasr, but paired with a CTC head (25K SentencePiece
vocab) instead of an LLM. One forward pass emits transcript + language
ID + emotion + audio-event tags. Non-autoregressive, 15× faster than
Whisper-Large.

### madlad

T5 encoder-decoder (12L+12L, d=2048, gated-GELU FFN, RMSNorm, bucketed
relative-position bias) + SentencePiece (256K vocab). Target language
specified as `<2xx>` input prefix. Tokens match Python SentencePiece
bit-by-bit; output matches HF reference.

### melotts

[myshell-ai/MeloTTS](https://github.com/myshell-ai/MeloTTS) VITS2
architecture (~52M params, MIT). Text encoder (6-layer relative-position
transformer with speaker conditioning at layer 2) → dual duration predictor
(SDP spline flows + deterministic DP, blended via sdp_ratio) →
TransformerCouplingBlock flow (4 blocks × 3 transformer layers) → HiFi-GAN
vocoder (5 upsample stages) @ 44.1 kHz. Built-in English G2P via embedded
CMU dictionary (129k entries) + rule-based LTS fallback for OOV words.
4 English speakers (US, BR, India, AU).

Optional BERT conditioning: loads a companion bert-base-uncased GGUF (238 MB)
via `melotts_load_bert()`. Runs 10-layer BERT forward pass → hidden_states[-3]
→ word2ph expansion → `ja_bert_proj` (768→192) → added to text encoder
embeddings. Improves contextual phoneme disambiguation (4/6 → 4/6 ASR
roundtrip but fixes previously-broken sentences like "I enjoy reading").

### piper

[rhasspy/piper](https://github.com/rhasspy/piper) VITS architecture (~60M
params, Apache-2.0). Text encoder (TextConv + Transformer) → flow-based
decoder → HiFi-GAN vocoder @ 22 kHz. Uses espeak-ng for phonemization.
Single speaker per model. Lightweight, fast inference.

### indextts

IndexTTS-1.5: GPT-2 AR (24L, 1280-d) mel-code generator + BigVGAN
vocoder. Designed for Chinese + English. Zero-shot voice cloning from
any reference WAV. Two GGUFs: GPT AR model + BigVGAN vocoder.

### outetts

OuteTTS: OLMo-0.5B LM backbone AR codec-token generator +
WavTokenizer codec (CC BY 4.0). Generates speech tokens
autoregressively, decoded to PCM by the WavTokenizer.

### voxcpm2-tts

VoxCPM2: Qwen2-2B backbone + flow matching + BigVGAN @ 48 kHz. Two-stage:
AR text-to-semantic-tokens via the Qwen2 LM, then flow matching
continuous diffusion + BigVGAN vocoder. Zero-shot voice cloning from
reference WAV. Output is 48 kHz, decimated to 24 kHz for the standard
CrispASR TTS pipeline.

### cosyvoice3-tts

CosyVoice3 0.5B (FunAudioLLM, Apache-2.0): three-stage pipeline — Qwen2-0.5B
AR speech-token LM → DiT-based conditional flow matching (10-step Euler ODE) →
HiFT vocoder (NSF + iSTFT) @ 24 kHz. Supports 9 languages + 18 Chinese
dialects. Zero-shot voice cloning via baked voice packs. Three separate GGUFs:
LLM, flow, HiFT.

### kugelaudio

KugelAudio-0-Open (MIT, 23 languages): hybrid AR + diffusion TTS based on
VibeVoice. Qwen2.5-7B language model (28L, 3584d, GQA 28/4) generates
constrained tokens; on `speech_diffusion` token, a 4-layer DiT prediction
head (AdaLN + SwiGLU, v-prediction) runs 20-step SDE-DPMSolver++ with
cosine beta schedule to produce 64-dim acoustic latents. An acoustic VAE
decoder (6-stage ConvNeXt with depthwise conv mixer, transposed-conv
upsample ratios [8,5,5,4,2,2] = 3200×) converts latents to 24 kHz mono
PCM. Pre-encoded voice embeddings (acoustic connector: FC1→RMSNorm→FC2)
inject speaker identity into the LM input sequence.

### pocket-tts

Kyutai Pocket TTS: Llama-1B backbone (causal LM) generating Mimi RVQ codec
tokens + Mimi decoder (SEANet with causal convolutions) @ 24 kHz.
Streaming-capable architecture. Uses raw tensor operations on CPU (no ggml
graph), KV-cached AR decode for the Llama backbone, per-frame Mimi decoding.

### f5-tts

F5-TTS: DiT (Diffusion Transformer) for flow-matching text-to-speech.
Converts text + reference audio to mel spectrograms via ODE-based diffusion
(typically 32 Euler steps), then vocodes with a shared vocoder. Zero-shot
voice cloning.

### lfm2-audio

LiquidAI LFM2.5-Audio (LFM Open v1.0, 1.5B): end-to-end multimodal
ASR + TTS + speech-to-speech in a single model.

**ASR path:** 16 kHz mono PCM → 128-mel NeMo-style spectrogram (slaney
filterbank, ln + per-feature-z normalization) → 17L FastConformer encoder
(512-dim, 8 heads, rel-pos attention, dw-striding 8× subsampling) → audio
adapter MLP (LayerNorm → Linear(512→2048) → GELU → Linear(2048→2048)) →
16L LFM2 hybrid backbone (2048-dim, 10 ShortConv + 6 GQA attention layers,
SwiGLU FFN, RoPE θ=1M, QK layernorm) → greedy text decode via tied
embed_tokens weight.

The LFM2 backbone interleaves two layer types:
- **ShortConv** (10 of 16 layers): depthwise causal conv1d (kernel=3) with
  gated in/out projections. `BCx = in_proj(h)`, `Bx = B * x`, conv(Bx),
  `y = C * conv_out`, `out = out_proj(y)`. Conv state cache stores last
  K-1=2 Bx columns per layer for incremental decode.
- **GQA attention** (6 of 16 layers): 32 query heads, 8 KV heads, head_dim=64.
  Per-head QK RMSNorm before RoPE. Flash attention with explicit causal mask.
  Standard F16 KV cache for incremental decode.

Layer types follow the pattern `ccaccaccacacacac` (c=conv, a=attn).

**TTS path:** text tokenized via GPT-2 BPE → interleaved generation
(6 text tokens + 9 audio frames alternating) → depthformer (6L transformer,
1024-dim, fused QKV, 8-codebook Mimi code generation) → ISTFT detokenizer
(separate 8L LFM2 512-dim + Linear(512→1282) + ISTFT → 24 kHz PCM).
The detokenizer loads from a companion `*-detokenizer.gguf`.

**Speech-to-speech:** combines the ASR conformer encoder (audio input) with
interleaved generation (text + audio output). System prompt:
"Respond with interleaved text and audio."

GGUF: single file for the main model (encoder + backbone + depthformer + Mimi
codec + audio embedding) + companion detokenizer. Quantization: Q5_K
recommended for EN, Q4_K for JP. crispasr-quantize keeps encoder, adapter,
embeddings, and Mimi codec at F16; only backbone + depthformer layers quantized.

Variants: English (`LiquidAI/LFM2.5-Audio-1.5B`) and Japanese
(`LiquidAI/LFM2.5-Audio-1.5B-JP`).

### nemotron

NVIDIA Nemotron-3.5-ASR-Streaming (NVOML, 600M): cache-aware streaming
FastConformer + RNN-T. First truly streaming-native ASR backend — processes
audio in fixed-size chunks with per-layer state caching.

**Architecture:** 16 kHz mono PCM → 80-mel NeMo spectrogram (no normalization)
→ causal 4× conv subsampling (pre-encode, asymmetric padding) → 24L
Cache-Aware FastConformer (1024-dim, 8 heads, chunked_limited attention
with rel-pos bias, depthwise conv kernel=9) → language prompt kernel
(39 langs, one-hot conditioning) → 1L LSTM predictor → joint network →
RNN-T decode with greedy (default), beam search (`--beam-size N`), or
MAES (`CRISPASR_NEMOTRON_MAES=1 --beam-size N`).

**Streaming:** gated by `CRISPASR_NEMOTRON_STREAMING=1`. Per-layer caches:
- `cache_last_channel`: post-FFN1 output (up to L=56 frames), used as K/V
  context for asymmetric attention (Q from new frames only)
- `cache_last_time`: last K-1=8 frames of post-GLU signal before depthwise
  conv, prepended instead of zero-padding (NeMo's `CausalConv1D.update_cache`)

Four attention context presets (`CRISPASR_NEMOTRON_CONTEXT_PRESET=0..3`)
trade latency for accuracy: 160ms/7.67% WER to 1120ms/6.93% WER.

GGUF: `cstr/nemotron-3.5-asr-streaming-0.6b-GGUF`. F16 + Q4_K produce
identical text. Pre-encode weights kept at F32 in the GGUF (F16 causes
1.56 max accumulation error across the 4352-dim projection).
Auto-download: `-m auto --backend nemotron`.

### bark

Suno Bark (MIT, ~400M): three-stage GPT-2 pipeline — text → semantic tokens
(12L, 1024-d) → coarse acoustic tokens (12L, 1024-d) → fine acoustic tokens
(12L, 1024-d) → EnCodec decoder @ 24 kHz. All sub-models packed into one
GGUF with selective Q4_K quantization. Speaker conditioning via `.npz`
voice prompts.

### tada

HumeAI TADA-3B-ML (`HumeAI/tada-3b-ml`). Two GGUFs: backbone talker + codec.

- **Backbone**: Llama-3.2-3B (28L, 3072-d, 24 heads, 8 KV heads, RoPE, SwiGLU,
  RMSNorm). Token embeddings extended with `acoustic_proj`, `mask`, `time_start`,
  and `time_end` step-embedding tensors for per-token conditioning.
- **FM head**: 4-layer SwiGLU + AdaLN transformer with sinusoidal time embedding.
  Uses an Euler ODE solver (flow-matching) to diffuse a noise vector into a
  per-token acoustic vector conditioned on the backbone hidden state.
- **Codec**: 6-layer local-attention transformer + DAC-style upsampler → 24 kHz
  mono PCM.
- **Special**: 1:1 text-to-acoustic alignment — every text token maps to exactly
  one acoustic vector (no duration predictor, no length regulator). BPE tokenizer
  (tiktoken GPT-2 byte-level vocab).
- **Voice cloning**: reference audio is encoded via the codec and prepended as a
  prompt to the backbone KV cache.

Models: `HumeAI/tada-3b-ml` (backbone Q4_K ~2.2 GB) + companion codec GGUF
(~1 GB). Pass `--codec-model <codec.gguf>`.

#### tada-encoder (voice reference creation)

The encoder pipeline converts audio + transcript → aligned acoustic features
for voice cloning. Ported to C++ in `src/tada_encoder.{h,cpp}`:

- **Aligner**: wav2vec2-large (24L, 1024-d, 16 heads) fine-tuned with 128K-class
  CTC head (Llama-3.2 tokenizer vocab). Resamples 24kHz→16kHz internally. Outputs
  frame-level logits → DP alignment algorithm → `token_positions` + `token_masks`.
- **WavEncoder**: DAC-style strided conv encoder. Conv1d(1→64, k=7) → 4×
  EncoderBlock (strides [6,5,4,4], Snake1d activations, weight-normed convs,
  3× ResidualUnit per block with dilations 1,3,9) → Snake1d → Conv1d(1024→1024).
  Total 480× downsample: 24kHz → 50Hz.
- **LocalAttentionEncoder**: 6-layer transformer, 1024-d, 8 heads, head_dim=128,
  RoPE (θ=10000), GELU FFN (4096), v2 segment attention mask, post-norm. Input
  augmented with `pos_emb(token_masks)` (Embedding(2, 1024)).
- **hidden_linear**: Linear(1024→512).
- **Post-processing**: zero non-token frames, add Gaussian noise (std=0.5),
  gather at token positions, normalize by (mean=0, std=1.5).

GGUFs at [cstr/tada-encoder-GGUF](https://huggingface.co/cstr/tada-encoder-GGUF):
`tada-encoder-f16.gguf` (178 MB, shared encoder) + `tada-aligner-en.gguf`
(1.1 GB, wav2vec2-large + CTC head, loaded by `wav2vec2_load()`).

### mini-omni2

gpt-omni/mini-omni2 (`gpt-omni/mini-omni2`). Multimodal speech model
supporting ASR, TTS, and speech-to-speech.

- **Audio encoder**: Whisper-small (80 mel, 12 layers, 768-d, 12 heads,
  sinusoidal positional embedding, LayerNorm with bias, GELU FFN).
- **Adapter**: whisperMLP (SwiGLU gate: `fc_1(768,4864)`, `fc_2(768,4864)`
  → `silu(fc_1) * fc_2` → `proj(4864,896)`). No bias (config.bias=false).
- **LLM**: Qwen2-0.5B (896-d, 24 layers, 14 heads, 2 KV heads, GQA 7:1,
  RoPE theta=1M, SwiGLU, RMSNorm eps=1e-6, QKV bias, no O/FFN bias).
- **8-stream architecture**: 7 audio streams (SNAC codebooks, layershifted)
  + 1 text stream, all embedded and averaged. Audio features replace pad
  positions in streams 0-6 only (not text stream 7).
- **Modes**: ASR uses `_asr` token (151940) for pure transcription. S2S
  uses `_answer_a/_answer_t` for conversational audio response. TTS uses
  text-only input with `_answer_a/_answer_t`.
- **Audio output**: 7-stream SNAC tokens deinterleaved to 3 codebooks
  (c0: stream 0, c1: streams 1+4, c2: streams 2+3+5+6) → SNAC 24kHz
  decoder (separate GGUF, `hubertsiuzdak/snac_24khz`).
- **Vocab**: text 152000 + 7x audio 4160 = 181120 padded. Tied word
  embeddings (lm_head = token_embd).

Models: single GGUF (F16 ~1.6 GB) converted from `lit_model.pth` + `small.pt`.
For TTS/S2S, also needs SNAC codec GGUF (`--codec-model snac-24khz.gguf`).

### reazonspeech

ReazonSpeech NeMo v2 (reazon-research, Apache-2.0): 619M-param Japanese
FastConformer-RNNT ASR model trained on the ReazonSpeech v2.0 corpus.

```
Audio → 80 log-mel (n_fft=512, Hann, per-feature z-norm)
      → dw_striding 8× subsampling (5× Conv2d + Linear)
      → 24 FastConformer blocks (rel_pos_local_attn, window=[128,128], 1 global token)
      → RNNT joint network (enc→640, pred→640, ReLU, →3001)
      → 2-layer LSTM predictor (embed(3001,640) + LSTM(640,640)×2)
      → RNNT greedy decode (n_tdt_durations=0)
```

Key architectural points:
- Reuses the `parakeet` runtime entirely — same GGUF arch tag, same C runtime
- **Local attention** (`self_attention_model: rel_pos_local_attn`): each position
  attends only to [q-128, q+128] plus 1 global token at position 0. Implemented
  via an additive (T,T) mask in `core_conformer::build_block()`. For audio shorter
  than ~20s (T_enc < 257), the window covers the full sequence = full attention.
- **Pure RNNT** (no TDT duration head): `n_tdt_durations=0` triggers the dedicated
  `parakeet_rnnt_decode()` / `parakeet_rnnt_beam_decode()` paths.
- **Quantization**: RNNT joint network is structurally sensitive to quantization
  noise (blank/non-blank argmax can flip). `crispasr-quantize` keeps
  `joint.*` and `decoder.embed.*` at source precision for RNNT models by default
  (override with `CRISPASR_PARAKEET_QUANT_ALL=1`). Q8_0 (704 MB) is the recommended
  deployment quant; Q4_K (455 MB) works with the RNNT protection rule.
- 3000-token SentencePiece unigram vocabulary, Japanese only.
- **Parity**: transcript-identical to upstream NeMo Python / `reazonspeech`
  package on gTTS Japanese test audio (verified on Kaggle, 2026-06-28).

Models at `cstr/reazonspeech-nemo-v2-GGUF`: F16 (1240 MB), Q8_0 (704 MB), Q4_K (455 MB).

### parakeet-ctc-1.1b-ja

Community Japanese fine-tune of nvidia/parakeet-ctc-1.1b (grider-transwithai,
Apache-2.0). 42-layer FastConformer-CTC, 1.1B params.

```
Audio → 80 log-mel (n_fft=512, Hann, per-feature z-norm)
      → dw_striding 8× subsampling (5× Conv2d + Linear)
      → 42 FastConformer blocks (rel_pos, full attention, xscaling)
      → CTC head (Conv1d(1024 → 4001, k=1))
      → CTC greedy decode
```

Key points:
- Same architecture and runtime as the English parakeet-ctc-{0.6b,1.1b} models.
  Uses the `canary-ctc` GGUF arch tag + `canary_ctc.cpp` runtime.
- **Checkpoint selection**: the HF repo has two `.nemo` files. `parakeet-ja.nemo`
  has **corrupt F32 weights** in layers 26-28 (NaN + values >1e38 in
  `attn.v.weight` and `ff1.linear1.weight`). Only `parakeet-ja-gal.nemo` works.
- **Quantization**: Q8_0 is the recommended deployment quant. Q4_K degrades on
  the 42-layer CTC encoder (same finding as English parakeet-ctc-1.1b and
  OmniASR-CTC — CTC argmax is structurally sensitive to accumulated
  quantization noise across many layers).
- 4000-token SentencePiece unigram vocabulary, Japanese only.
- **Parity**: transcript-identical to upstream NeMo Python on gTTS Japanese
  test audio (verified on Kaggle, 2026-06-28).

Models at `cstr/parakeet-ctc-1.1b-ja-GGUF`: F16 (2.0 GB), Q8_0 (1.2 GB), Q4_K (631 MB).

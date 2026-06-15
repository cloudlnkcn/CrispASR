# Nemotron: GPU scheduler migration + streaming path

## What's done

The nemotron-3.5-asr-streaming-0.6b backend is **working** on main (`eaeea3be`+).
Both F16 (1.3 GB) and Q4_K (458 MB) produce correct English transcription.
Fixed GGUFs are on `cstr/nemotron-3.5-asr-streaming-GGUF`.

Key files:
- `src/nemotron.cpp` — runtime (~1800 lines)
- `src/nemotron.h` — public C API
- `models/convert-nemotron-to-gguf.py` — converter (keeps pre-encode F32)
- `examples/cli/crispasr_backend_nemotron.cpp` — CLI integration

Root cause of the 3-day debugging: `ggml_siglu` vs `ggml_siglu_swapped` —
the GLU gate/value halves were reversed in every conformer conv module.
See LEARNINGS.md for the full rule.

## What's next (two tasks)

### 1. GPU scheduler migration (§168 in PLAN.md)

The nemotron backend uses `ggml_gallocr` + `ggml_backend_graph_compute`
instead of `ggml_backend_sched`. It calls `ggml_backend_init_best()` so
it CAN use GPU, but gallocr doesn't support multi-backend scheduling
(e.g. routing large matmuls to GPU and small ops to CPU).

Migration pattern (from the 60+ backends that already use sched):
```cpp
// Replace gallocr with sched throughout nemotron_run_encoder and
// nemotron_rnnt_decode:
ggml_backend_sched_reset(ctx->sched);
ggml_backend_sched_alloc_graph(ctx->sched, gf);
// set inputs...
ggml_backend_sched_graph_compute(ctx->sched, gf);
```

See `src/parakeet.cpp` as the reference — same architecture family,
already uses sched. The `nemotron_context` already has a `sched` field
declared (line ~190) but it's never initialized or used.

Also migrate: `paraformer.cpp` (CPU-only, 0 GPU refs — highest priority),
`dia_tts.cpp`, `outetts_wavtok.cpp`.

### 2. Streaming (chunked) encoder path

The default encoder runs the full audio through all 24 layers with a
`chunked_limited` attention mask. This works but requires the full audio
upfront (batch mode). For real-time/streaming inference, the model needs
cache-aware chunked processing:

- Process audio in chunks of R+1=4 frames
- Per-layer: Q from new frames only, K/V from [cached + new]
- Cache stores post-FFN1 output per layer (NeMo's `cache_last_channel`)
- Conv module uses K-1=8 frame left-context cache

The `nemotron_run_encoder_chunked` function exists but uses the wrong
approach (full-window attention per chunk). It needs rewriting to match
NeMo's `ConformerEncoder.forward_internal` streaming path. Key insight
from the NeMo source (`nemo/collections/asr/modules/conformer_encoder.py`
line ~665): the cache is prepended to the input, and an attention mask
restricts visibility. The per-layer `update_cache` in
`RelPositionMultiHeadAttention` (line 204) concatenates cache to K/V.

Gate this behind `NEMOTRON_CHUNKED=1` env var (already wired).

## Key learnings for the agent

- `ggml_siglu` = `sigmoid(first) * second`. PyTorch `glu` = `first * sigmoid(second)`. Use `ggml_siglu_swapped` for PyTorch compatibility.
- NeMo's prompt_dictionary uses a CUSTOM language ordering (en-US=0, de-DE=9), NOT alphabetical. The mapping is in `nemotron.cpp` line ~334.
- Pre-encode weights MUST be F32 in the GGUF — F16 accumulates 1.56 max error across the 4352-dim projection.
- The mel computation matches NeMo exactly (verified per-frame at frame 0 and frame 500). No normalization (`normalize="NA"`).
- The `CausalConv2D` padding uses `ggml_pad_ext(left=2, right=1)` for true asymmetric padding.
- The `chunked_limited` attention mask is chunk-based block-diagonal, not a sliding window. Each frame sees its chunk + 14 previous chunks.

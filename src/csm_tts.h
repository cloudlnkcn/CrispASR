// csm_tts.h -- C API for Sesame CSM-1B TTS (Conversational Speech Model).
//
// Architecture:
//   Backbone:       Llama-3.2 1B (16L, 32H/8KVH, 2048d, RoPE, SwiGLU, RMSNorm)
//                   Autoregressively generates first-codebook Mimi tokens.
//   Depth decoder:  Llama-3.2 100M (4L, 8H/2KVH, 1024d)
//                   Fills remaining 31 codebooks from backbone hidden state.
//   Mimi codec:     Kyutai Mimi decoder (SEANet + 8L transformer + upsample)
//                   32-codebook RVQ -> upsampling conv -> 24 kHz PCM.
//
// Text flow:
//   Llama-3.2 BPE tokenize -> backbone AR loop (one frame = 32 codebooks)
//   -> depth decoder (7 iterations per frame) -> Mimi decode -> 24 kHz PCM
//
// Speaker conditioning: encode reference audio with Mimi encoder, prepend
//   the encoded tokens as context to the backbone prompt.
//
// Reference: github.com/SesameAILabs/csm (Apache 2.0)
//            HuggingFace: sesame/csm-1b

#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct csm_tts_context;

struct csm_tts_context_params {
    int n_threads;
    int verbosity; // 0=silent 1=normal 2=verbose
    bool use_gpu;
    float temperature;    // 0 = greedy, >0 = top-k sampling (default 0.9)
    int topk;             // top-k parameter for sampling (default 50)
    uint64_t seed;        // RNG seed (0 = non-deterministic)
    int max_audio_tokens; // max backbone AR steps (0 = default 2048)
};

struct csm_tts_context_params csm_tts_context_default_params(void);

// Load CSM-1B from a single GGUF file (backbone + depth decoder + Mimi codec).
struct csm_tts_context* csm_tts_init_from_file(const char* path_model, struct csm_tts_context_params params);

void csm_tts_free(struct csm_tts_context* ctx);

// Synthesize text to 24 kHz mono float32 PCM.
// Caller frees with csm_tts_pcm_free(). *out_n_samples is set on success.
// Returns nullptr on failure.
float* csm_tts_synthesize(struct csm_tts_context* ctx, const char* text, int* out_n_samples);

// Synthesize with speaker conditioning from reference audio.
// ref_pcm: 24 kHz mono float32 PCM of reference speaker.
// ref_text: transcript of reference audio (for text-audio alignment).
float* csm_tts_synthesize_with_reference(struct csm_tts_context* ctx, const char* text, const float* ref_pcm,
                                         int ref_n_samples, const char* ref_text, int* out_n_samples);

void csm_tts_pcm_free(float* pcm);

// Runtime parameter setters.
void csm_tts_set_temperature(struct csm_tts_context* ctx, float temperature);
void csm_tts_set_topk(struct csm_tts_context* ctx, int topk);
void csm_tts_set_seed(struct csm_tts_context* ctx, uint64_t seed);
void csm_tts_set_n_threads(struct csm_tts_context* ctx, int n_threads);

#ifdef __cplusplus
}
#endif

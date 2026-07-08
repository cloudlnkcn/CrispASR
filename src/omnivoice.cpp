// omnivoice.cpp — runtime for k2-fsa/OmniVoice TTS.
//
// Architecture: Qwen3-0.6B backbone with audio_embeddings + audio_heads
// for masked iterative multi-codebook TTS (SoundStorm-style).
//
// Status (July 2026):
//   ✓ LLM forward (28L Qwen3 with Q/K-norm, standard RoPE)
//   ✓ Masked iterative generation loop
//   ✗ Audio tokenizer encode (reference audio → codes for voice cloning)
//   ✗ Audio tokenizer decode (codes → waveform)
//
// Env knobs:
//   OMNIVOICE_DEBUG=1      — verbose per-step trace
//   OMNIVOICE_BENCH=1      — per-stage wall-clock timings
//   OMNIVOICE_DUMP_DIR=/d  — dump intermediate tensors

#include "omnivoice.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "core/activation.h"
#include "core/attention.h"
#include "core/bpe.h"
#include "core/ffn.h"
#include "core/gguf_loader.h"
#include "core/gpu_backend_pref.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Env helpers
// ---------------------------------------------------------------------------

bool env_bool(const char* k) {
    const char* v = std::getenv(k);
    return v && *v && std::strcmp(v, "0") != 0;
}
const char* env_str(const char* k) {
    const char* v = std::getenv(k);
    return (v && *v) ? v : nullptr;
}

// ---------------------------------------------------------------------------
// Bench instrumentation
// ---------------------------------------------------------------------------

static bool ov_bench_enabled() {
    static int v = -1;
    if (v < 0) {
        const char* e = std::getenv("OMNIVOICE_BENCH");
        v = (e && *e && *e != '0') ? 1 : 0;
    }
    return v != 0;
}

struct ov_bench_stage {
    const char* name;
    std::chrono::steady_clock::time_point t0;
    explicit ov_bench_stage(const char* n) : name(n), t0(std::chrono::steady_clock::now()) {}
    ~ov_bench_stage() {
        if (!ov_bench_enabled())
            return;
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::fprintf(stderr, "  omnivoice_bench: %-22s %.2f ms\n", name, ms);
    }
};

// ---------------------------------------------------------------------------
// Hyperparameters
// ---------------------------------------------------------------------------

struct ov_hp {
    // LLM backbone (Qwen3)
    uint32_t n_layers = 28;
    uint32_t d_model = 1024;
    uint32_t n_heads = 16;
    uint32_t n_kv_heads = 8;
    uint32_t head_dim = 128;
    uint32_t ff_dim = 3072;
    uint32_t vocab_size = 151676;
    uint32_t max_pos = 40960;
    float rope_theta = 1000000.0f;
    float rms_norm_eps = 1e-6f;
    bool tie_word_embeddings = true;

    // Audio
    uint32_t audio_vocab_size = 1025;
    uint32_t audio_mask_id = 1024;
    uint32_t n_codebooks = 8;
    std::vector<float> codebook_weights; // [8, 8, 6, 6, 4, 4, 2, 2]

    // Token sentinels
    uint32_t eos_token_id = 151645;
    uint32_t pad_token_id = 151643;
};

// ---------------------------------------------------------------------------
// Model weights
// ---------------------------------------------------------------------------

struct ov_layer {
    ggml_tensor* attn_norm_w = nullptr;
    ggml_tensor* attn_q_w = nullptr;
    ggml_tensor* attn_k_w = nullptr;
    ggml_tensor* attn_v_w = nullptr;
    ggml_tensor* attn_output_w = nullptr;
    ggml_tensor* attn_q_norm_w = nullptr;
    ggml_tensor* attn_k_norm_w = nullptr;
    ggml_tensor* ffn_norm_w = nullptr;
    ggml_tensor* ffn_gate_w = nullptr;
    ggml_tensor* ffn_up_w = nullptr;
    ggml_tensor* ffn_down_w = nullptr;
};

struct ov_model {
    // LLM
    ggml_tensor* token_embd_w = nullptr; // (vocab_size, d_model)
    std::vector<ov_layer> blocks;
    ggml_tensor* output_norm_w = nullptr;

    // Audio
    ggml_tensor* audio_embd_w = nullptr;   // (n_codebooks * audio_vocab_size, d_model)
    ggml_tensor* audio_output_w = nullptr; // (n_codebooks * audio_vocab_size, d_model)
};

struct ov_vocab {
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int32_t> token_to_id;
    std::unordered_map<std::string, int32_t> merge_rank;
};

// ---------------------------------------------------------------------------
// Generation config
// ---------------------------------------------------------------------------

struct ov_gen_config {
    int num_steps = 32;
    float guidance_scale = 1.0f;
    float class_temperature = 0.7f;
    float position_temperature = 4.5f;
    float layer_penalty_factor = 0.5f;
    float t_shift = 1.0f;
    uint64_t seed = 42;
};

} // namespace

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

struct omnivoice_context {
    ov_hp hp;
    ov_model model;
    ov_vocab vocab;
    ov_gen_config gen;

    int n_threads = 4;
    int verbosity = 1;
    bool use_gpu = false;
    bool flash_attn = false;

    // ggml
    ggml_context* ctx_w = nullptr; // weight context
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;

    // Voice cloning state
    std::vector<int32_t> ref_audio_codes; // (n_codebooks, T_ref) row-major
    int ref_T = 0;
    std::string ref_text;

    // Language / instruct
    std::string language;
    std::string instruct;

    // Audio tokenizer path (separate GGUF)
    std::string tokenizer_path;
};

// ---------------------------------------------------------------------------
// Default params
// ---------------------------------------------------------------------------

struct omnivoice_context_params omnivoice_context_default_params(void) {
    struct omnivoice_context_params p;
    std::memset(&p, 0, sizeof(p));
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = false;
    p.num_steps = 0;
    p.guidance_scale = 0.0f;
    p.class_temperature = 0.0f;
    p.position_temperature = 0.0f;
    p.layer_penalty_factor = 0.0f;
    p.t_shift = 0.0f;
    p.seed = 0;
    p.flash_attn = false;
    return p;
}

namespace {

// ---------------------------------------------------------------------------
// GGUF loading
// ---------------------------------------------------------------------------

static bool load_model(omnivoice_context* ctx, const char* path) {
    auto& hp = ctx->hp;
    auto& m = ctx->model;

    struct gguf_init_params gp = {/*.no_alloc=*/true, /*.ctx=*/&ctx->ctx_w};
    struct gguf_context* gf = gguf_init_from_file(path, gp);
    if (!gf) {
        fprintf(stderr, "omnivoice: failed to open %s\n", path);
        return false;
    }

    auto read_u32 = [&](const char* k, uint32_t dflt) -> uint32_t {
        int idx = gguf_find_key(gf, k);
        return idx >= 0 ? (uint32_t)gguf_get_val_u32(gf, idx) : dflt;
    };
    auto read_f32 = [&](const char* k, float dflt) -> float {
        int idx = gguf_find_key(gf, k);
        return idx >= 0 ? gguf_get_val_f32(gf, idx) : dflt;
    };
    auto read_bool = [&](const char* k, bool dflt) -> bool {
        int idx = gguf_find_key(gf, k);
        return idx >= 0 ? gguf_get_val_bool(gf, idx) : dflt;
    };

    // LLM
    hp.n_layers = read_u32("omnivoice.llm.n_layers", 28);
    hp.d_model = read_u32("omnivoice.llm.d_model", 1024);
    hp.n_heads = read_u32("omnivoice.llm.n_heads", 16);
    hp.n_kv_heads = read_u32("omnivoice.llm.n_kv_heads", 8);
    hp.head_dim = read_u32("omnivoice.llm.head_dim", 128);
    hp.ff_dim = read_u32("omnivoice.llm.ff_dim", 3072);
    hp.vocab_size = read_u32("omnivoice.llm.vocab_size", 151676);
    hp.max_pos = read_u32("omnivoice.llm.max_pos", 40960);
    hp.rope_theta = read_f32("omnivoice.llm.rope_theta", 1000000.0f);
    hp.rms_norm_eps = read_f32("omnivoice.llm.rms_norm_eps", 1e-6f);
    hp.tie_word_embeddings = read_bool("omnivoice.llm.tie_word_embeddings", true);

    // Audio
    hp.audio_vocab_size = read_u32("omnivoice.audio.vocab_size", 1025);
    hp.audio_mask_id = read_u32("omnivoice.audio.mask_id", 1024);
    hp.n_codebooks = read_u32("omnivoice.audio.n_codebooks", 8);

    // Codebook weights
    {
        int idx = gguf_find_key(gf, "omnivoice.audio.codebook_weights");
        if (idx >= 0) {
            int n = (int)gguf_get_arr_n(gf, idx);
            hp.codebook_weights.resize(n);
            const int32_t* arr = (const int32_t*)gguf_get_arr_data(gf, idx);
            for (int i = 0; i < n; i++) {
                hp.codebook_weights[i] = (float)arr[i];
            }
        } else {
            hp.codebook_weights = {8, 8, 6, 6, 4, 4, 2, 2};
        }
    }

    // Tokens
    hp.eos_token_id = read_u32("omnivoice.eos_token_id", 151645);
    hp.pad_token_id = read_u32("omnivoice.pad_token_id", 151643);

    if (ctx->verbosity >= 1) {
        fprintf(stderr, "omnivoice: LLM %uL d=%u heads=%u/%u hd=%u ff=%u vocab=%u\n", hp.n_layers, hp.d_model,
                hp.n_heads, hp.n_kv_heads, hp.head_dim, hp.ff_dim, hp.vocab_size);
        fprintf(stderr, "omnivoice: Audio %u codebooks vocab=%u mask=%u\n", hp.n_codebooks, hp.audio_vocab_size,
                hp.audio_mask_id);
    }

    // Allocate tensors
    auto find = [&](const char* name) -> ggml_tensor* { return ggml_get_tensor(ctx->ctx_w, name); };

    m.token_embd_w = find("llm.token_embd.weight");
    m.output_norm_w = find("llm.output_norm.weight");
    m.audio_embd_w = find("audio_embd.weight");
    m.audio_output_w = find("audio_output.weight");

    m.blocks.resize(hp.n_layers);
    for (uint32_t i = 0; i < hp.n_layers; i++) {
        char buf[128];
        auto L = [&](const char* sfx) -> ggml_tensor* {
            snprintf(buf, sizeof(buf), "llm.blk.%u.%s", i, sfx);
            return find(buf);
        };
        auto& b = m.blocks[i];
        b.attn_norm_w = L("attn_norm.weight");
        b.attn_q_w = L("attn_q.weight");
        b.attn_k_w = L("attn_k.weight");
        b.attn_v_w = L("attn_v.weight");
        b.attn_output_w = L("attn_output.weight");
        b.attn_q_norm_w = L("attn_q_norm.weight");
        b.attn_k_norm_w = L("attn_k_norm.weight");
        b.ffn_norm_w = L("ffn_norm.weight");
        b.ffn_gate_w = L("ffn_gate.weight");
        b.ffn_up_w = L("ffn_up.weight");
        b.ffn_down_w = L("ffn_down.weight");
    }

    // Verify critical weights
    const char* required[] = {
        "llm.token_embd.weight",
        "llm.output_norm.weight",
        "audio_embd.weight",
        "audio_output.weight",
    };
    bool ok = true;
    for (const char* name : required) {
        if (!find(name)) {
            fprintf(stderr, "omnivoice: required tensor missing: %s\n", name);
            ok = false;
        }
    }
    for (uint32_t i = 0; i < hp.n_layers && ok; i++) {
        char buf[128];
        snprintf(buf, sizeof(buf), "llm.blk.%u.attn_q.weight", i);
        if (!find(buf)) {
            fprintf(stderr, "omnivoice: required tensor missing: %s\n", buf);
            ok = false;
        }
    }
    if (!ok) {
        gguf_free(gf);
        return false;
    }

    // Create backend + buffer
    ctx->backend = ggml_backend_cpu_init();
    if (!ctx->backend) {
        fprintf(stderr, "omnivoice: failed to init CPU backend\n");
        gguf_free(gf);
        return false;
    }
    ggml_backend_cpu_set_n_threads(ctx->backend, ctx->n_threads);

    ctx->buf_w = ggml_backend_alloc_ctx_tensors(ctx->ctx_w, ctx->backend);
    if (!ctx->buf_w) {
        fprintf(stderr, "omnivoice: failed to allocate weight buffer\n");
        gguf_free(gf);
        return false;
    }

    // Load tensor data from file
    {
        FILE* fp = fopen(path, "rb");
        if (!fp) {
            fprintf(stderr, "omnivoice: cannot reopen %s\n", path);
            gguf_free(gf);
            return false;
        }
        int n_tensors = gguf_get_n_tensors(gf);
        for (int i = 0; i < n_tensors; i++) {
            const char* name = gguf_get_tensor_name(gf, i);
            ggml_tensor* t = ggml_get_tensor(ctx->ctx_w, name);
            if (!t)
                continue;
            size_t offset = gguf_get_data_offset(gf) + gguf_get_tensor_offset(gf, i);
            fseek(fp, (long)offset, SEEK_SET);
            size_t nbytes = ggml_nbytes(t);
            std::vector<uint8_t> tmp(nbytes);
            if (fread(tmp.data(), 1, nbytes, fp) != nbytes) {
                fprintf(stderr, "omnivoice: short read for %s\n", name);
                fclose(fp);
                gguf_free(gf);
                return false;
            }
            ggml_backend_tensor_set(t, tmp.data(), 0, nbytes);
        }
        fclose(fp);
    }

    // Load vocab
    {
        int idx = gguf_find_key(gf, "tokenizer.ggml.tokens");
        if (idx >= 0) {
            int n = gguf_get_arr_n(gf, idx);
            ctx->vocab.id_to_token.resize(n);
            for (int i = 0; i < n; i++) {
                ctx->vocab.id_to_token[i] = gguf_get_arr_str(gf, idx, i);
                ctx->vocab.token_to_id[ctx->vocab.id_to_token[i]] = i;
            }
            if (ctx->verbosity >= 1) {
                fprintf(stderr, "omnivoice: loaded %d tokens\n", n);
            }
        }

        int midx = gguf_find_key(gf, "tokenizer.ggml.merges");
        if (midx >= 0) {
            int n = gguf_get_arr_n(gf, midx);
            for (int i = 0; i < n; i++) {
                ctx->vocab.merge_rank[gguf_get_arr_str(gf, midx, i)] = i;
            }
            if (ctx->verbosity >= 1) {
                fprintf(stderr, "omnivoice: loaded %d merges\n", n);
            }
        }
    }

    gguf_free(gf);

    if (ctx->verbosity >= 1) {
        size_t total = ggml_backend_buffer_get_size(ctx->buf_w);
        fprintf(stderr, "omnivoice: loaded %s (%.2f GB)\n", path, total / 1e9);
    }

    return true;
}

// ---------------------------------------------------------------------------
// BPE tokenizer
// ---------------------------------------------------------------------------

static std::vector<int32_t> tokenize(const ov_vocab& vocab, const std::string& text) {
    return core_bpe::tokenize_simple(vocab.token_to_id, vocab.merge_rank, text);
}

// ---------------------------------------------------------------------------
// Estimate target audio length (frames) from text length
// ---------------------------------------------------------------------------

static int estimate_target_tokens(const std::string& text, float speed = 1.0f) {
    // Rough heuristic: ~6 audio frames per text character at 75 Hz frame rate.
    // OmniVoice's frame rate = 24000/320 = 75 Hz.
    int n_chars = 0;
    for (size_t i = 0; i < text.size();) {
        unsigned char c = text[i];
        if (c < 0x80)
            i += 1;
        else if (c < 0xE0)
            i += 2;
        else if (c < 0xF0)
            i += 3;
        else
            i += 4;
        n_chars++;
    }
    int est = (int)(n_chars * 6.0f / speed);
    return std::max(est, 10);
}

// ---------------------------------------------------------------------------
// Time steps for masked iterative schedule
// ---------------------------------------------------------------------------

static std::vector<float> get_time_steps(float t_start, float t_end, int num_step, float t_shift) {
    std::vector<float> steps(num_step + 1);
    for (int i = 0; i <= num_step; i++) {
        float t = t_start + (t_end - t_start) * i / num_step;
        // Apply shift: t' = t * shift / (1 + (shift - 1) * t)
        if (t_shift != 1.0f) {
            t = t * t_shift / (1.0f + (t_shift - 1.0f) * t);
        }
        steps[i] = t;
    }
    return steps;
}

// ---------------------------------------------------------------------------
// Build the Qwen3 LLM graph for a forward pass
// ---------------------------------------------------------------------------

static ggml_cgraph* build_llm_graph(omnivoice_context* ctx, ggml_context* ctx0, ggml_tensor* input_embeds, int T) {
    auto& hp = ctx->hp;
    auto& m = ctx->model;

    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 8192, false);

    ggml_tensor* cur = input_embeds; // (d_model, T)

    const float attn_scale = 1.0f / sqrtf((float)hp.head_dim);

    for (uint32_t il = 0; il < hp.n_layers; il++) {
        auto& b = m.blocks[il];

        // Pre-attention RMSNorm
        ggml_tensor* attn_in = ggml_rms_norm(ctx0, cur, hp.rms_norm_eps);
        attn_in = ggml_mul(ctx0, attn_in, b.attn_norm_w);

        // Q, K, V projections
        ggml_tensor* Q = ggml_mul_mat(ctx0, b.attn_q_w, attn_in);
        ggml_tensor* K = ggml_mul_mat(ctx0, b.attn_k_w, attn_in);
        ggml_tensor* V = ggml_mul_mat(ctx0, b.attn_v_w, attn_in);

        // Reshape to (head_dim, n_heads, T) / (head_dim, n_kv_heads, T)
        Q = ggml_reshape_3d(ctx0, Q, hp.head_dim, hp.n_heads, T);
        K = ggml_reshape_3d(ctx0, K, hp.head_dim, hp.n_kv_heads, T);
        V = ggml_reshape_3d(ctx0, V, hp.head_dim, hp.n_kv_heads, T);

        // Q/K norms (Qwen3 style)
        if (b.attn_q_norm_w) {
            Q = ggml_rms_norm(ctx0, Q, hp.rms_norm_eps);
            Q = ggml_mul(ctx0, Q, b.attn_q_norm_w);
        }
        if (b.attn_k_norm_w) {
            K = ggml_rms_norm(ctx0, K, hp.rms_norm_eps);
            K = ggml_mul(ctx0, K, b.attn_k_norm_w);
        }

        // RoPE (standard, not mRoPE — OmniVoice uses default rope_type)
        Q = ggml_rope_ext(ctx0, Q, nullptr, nullptr, hp.head_dim, GGML_ROPE_TYPE_NEOX, 0, hp.rope_theta, 1.0f, 0.0f,
                          1.0f, 0.0f, 0.0f);
        K = ggml_rope_ext(ctx0, K, nullptr, nullptr, hp.head_dim, GGML_ROPE_TYPE_NEOX, 0, hp.rope_theta, 1.0f, 0.0f,
                          1.0f, 0.0f, 0.0f);

        // GQA: expand KV heads
        uint32_t kv_repeat = hp.n_heads / hp.n_kv_heads;
        if (kv_repeat > 1) {
            K = ggml_reshape_4d(ctx0, K, hp.head_dim, 1, hp.n_kv_heads, T);
            K = ggml_repeat(ctx0, K, ggml_new_tensor_4d(ctx0, K->type, hp.head_dim, kv_repeat, hp.n_kv_heads, T));
            K = ggml_reshape_3d(ctx0, K, hp.head_dim, hp.n_heads, T);

            V = ggml_reshape_4d(ctx0, V, hp.head_dim, 1, hp.n_kv_heads, T);
            V = ggml_repeat(ctx0, V, ggml_new_tensor_4d(ctx0, V->type, hp.head_dim, kv_repeat, hp.n_kv_heads, T));
            V = ggml_reshape_3d(ctx0, V, hp.head_dim, hp.n_heads, T);
        }

        // Attention: Q,K,V are (head_dim, n_heads, T)
        // Permute to (head_dim, T, n_heads) for mul_mat
        Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));
        K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));

        ggml_tensor* scores = ggml_mul_mat(ctx0, K, Q); // (T, T, n_heads)
        scores = ggml_scale(ctx0, scores, attn_scale);

        // No causal mask for OmniVoice — full bidirectional attention
        // (the model sees all positions including masked tokens)
        scores = ggml_soft_max(ctx0, scores);

        V = ggml_cont(ctx0, ggml_permute(ctx0, V, 1, 2, 0, 3)); // (T, head_dim, n_heads)
        // Note: V needs to be (T, head_dim, n_heads) for mul_mat with scores (T, T, n_heads)
        // Actually let's do this correctly:
        // V = (head_dim, n_heads, T) → permute(1, 0, 2, 3) → (n_heads, head_dim, T)
        // No, let me follow the standard pattern from the dev doc:
        // V permute(1, 0, 2, 3) → swap dim0 and dim1
        V = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, V, hp.head_dim, hp.n_heads, T), 1, 0, 2, 3));
        ggml_tensor* attn_out = ggml_mul_mat(ctx0, V, scores); // (head_dim, T, n_heads)

        // Permute back and reshape to (d_model, T)
        attn_out = ggml_cont(ctx0, ggml_permute(ctx0, attn_out, 0, 2, 1, 3));
        attn_out = ggml_reshape_2d(ctx0, attn_out, hp.d_model, T);

        // Output projection
        attn_out = ggml_mul_mat(ctx0, b.attn_output_w, attn_out);

        // Residual
        cur = ggml_add(ctx0, cur, attn_out);

        // FFN: pre-norm + SwiGLU
        ggml_tensor* ffn_in = ggml_rms_norm(ctx0, cur, hp.rms_norm_eps);
        ffn_in = ggml_mul(ctx0, ffn_in, b.ffn_norm_w);

        ggml_tensor* gate = ggml_mul_mat(ctx0, b.ffn_gate_w, ffn_in);
        ggml_tensor* up = ggml_mul_mat(ctx0, b.ffn_up_w, ffn_in);
        gate = ggml_silu(ctx0, gate);
        ggml_tensor* ffn_out = ggml_mul(ctx0, gate, up);
        ffn_out = ggml_mul_mat(ctx0, b.ffn_down_w, ffn_out);

        cur = ggml_add(ctx0, cur, ffn_out);
    }

    // Final RMSNorm
    cur = ggml_rms_norm(ctx0, cur, hp.rms_norm_eps);
    cur = ggml_mul(ctx0, cur, m.output_norm_w);

    ggml_set_name(cur, "llm_out");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);

    return gf;
}

// ---------------------------------------------------------------------------
// Prepare embeddings: text + audio tokens → mixed embedding
// ---------------------------------------------------------------------------
//
// Python: _prepare_embed_inputs()
//   text_embeds = llm.embed_tokens(input_ids[:, 0, :])
//   shifted_ids = (input_ids * audio_mask) + codebook_layer_offsets
//   audio_embeds = audio_embeddings(shifted_ids).sum(dim=1)
//   return where(audio_mask, audio_embeds, text_embeds)
//
// For the ggml graph we do this entirely in CPU since it's a one-time
// embedding lookup before the main transformer forward.

static std::vector<float> prepare_embeddings(
    omnivoice_context* ctx, const std::vector<int32_t>& text_ids,
    const std::vector<int32_t>& audio_tokens, // (n_codebooks * T_audio) row-major
    const std::vector<bool>& audio_mask,      // length = total_seq_len
    int total_T) {
    auto& hp = ctx->hp;
    auto& m = ctx->model;
    const int d = (int)hp.d_model;

    // Read weights to CPU
    std::vector<float> text_embd_data(hp.vocab_size * d);
    ggml_backend_tensor_get(m.token_embd_w, text_embd_data.data(), 0, text_embd_data.size() * sizeof(float));

    std::vector<float> audio_embd_data(hp.n_codebooks * hp.audio_vocab_size * d);
    ggml_backend_tensor_get(m.audio_embd_w, audio_embd_data.data(), 0, audio_embd_data.size() * sizeof(float));

    std::vector<float> embeds(total_T * d, 0.0f);

    int audio_pos = 0; // position within audio_tokens
    for (int t = 0; t < total_T; t++) {
        if (!audio_mask[t]) {
            // Text position: look up from text_ids
            int tid = text_ids[t];
            if (tid >= 0 && tid < (int)hp.vocab_size) {
                const float* src = text_embd_data.data() + (size_t)tid * d;
                std::memcpy(embeds.data() + (size_t)t * d, src, d * sizeof(float));
            }
        } else {
            // Audio position: sum embeddings across all codebooks with offsets
            float* dst = embeds.data() + (size_t)t * d;
            for (uint32_t cb = 0; cb < hp.n_codebooks; cb++) {
                int code = audio_tokens[cb * (total_T - /* adjust for text prefix */ 0) + audio_pos];
                int shifted = code + (int)(cb * hp.audio_vocab_size);
                if (shifted >= 0 && shifted < (int)(hp.n_codebooks * hp.audio_vocab_size)) {
                    const float* src = audio_embd_data.data() + (size_t)shifted * d;
                    for (int j = 0; j < d; j++) {
                        dst[j] += src[j];
                    }
                }
            }
            audio_pos++;
        }
    }

    return embeds;
}

// ---------------------------------------------------------------------------
// Masked iterative generation loop
// ---------------------------------------------------------------------------

struct ov_gen_result {
    std::vector<int32_t> codes; // (n_codebooks * T) row-major
    int T = 0;
    int n_codebooks = 0;
};

static ov_gen_result generate_iterative(omnivoice_context* ctx, const std::string& text) {
    auto& hp = ctx->hp;
    auto& gen = ctx->gen;
    bool debug = env_bool("OMNIVOICE_DEBUG");

    ov_gen_result result;
    result.n_codebooks = (int)hp.n_codebooks;

    // 1. Tokenize text
    std::string wrapped = "<|text_start|>" + text + "<|text_end|>";
    std::vector<int32_t> text_token_ids = tokenize(ctx->vocab, wrapped);
    int T_text = (int)text_token_ids.size();

    // 2. Estimate target audio length
    int T_target = estimate_target_tokens(text);
    result.T = T_target;

    if (debug) {
        fprintf(stderr, "omnivoice: text tokens=%d, target audio frames=%d\n", T_text, T_target);
    }

    // 3. Build style prefix (simplified — language + instruct)
    std::string style_text;
    std::string lang_str = ctx->language.empty() ? "None" : ctx->language;
    std::string instruct_str = ctx->instruct.empty() ? "None" : ctx->instruct;
    style_text += "<|lang_start|>" + lang_str + "<|lang_end|>";
    style_text += "<|instruct_start|>" + instruct_str + "<|instruct_end|>";
    std::vector<int32_t> style_ids = tokenize(ctx->vocab, style_text);
    int T_style = (int)style_ids.size();

    // 4. Build the full input sequence
    // Layout: [style_tokens | text_tokens | ref_audio_tokens? | target_mask_tokens]
    int T_ref = ctx->ref_T;
    int T_total = T_style + T_text + T_ref + T_target;

    // Build text_ids (the first codebook layer for text positions)
    std::vector<int32_t> full_text_ids(T_total, 0);
    for (int i = 0; i < T_style; i++)
        full_text_ids[i] = style_ids[i];
    for (int i = 0; i < T_text; i++)
        full_text_ids[T_style + i] = text_token_ids[i];
    // ref audio and target positions get pad/mask (handled by audio path)

    // Audio mask: true for audio positions (ref + target)
    std::vector<bool> audio_mask(T_total, false);
    int audio_start = T_style + T_text;
    for (int i = audio_start; i < T_total; i++) {
        audio_mask[i] = true;
    }

    // Audio tokens: (n_codebooks, T_audio) where T_audio = T_ref + T_target
    int T_audio = T_ref + T_target;
    std::vector<int32_t> audio_tokens(hp.n_codebooks * T_audio, (int)hp.audio_mask_id);

    // Fill in reference codes if available
    if (T_ref > 0 && !ctx->ref_audio_codes.empty()) {
        for (uint32_t cb = 0; cb < hp.n_codebooks; cb++) {
            for (int t = 0; t < T_ref; t++) {
                audio_tokens[cb * T_audio + t] = ctx->ref_audio_codes[cb * T_ref + t];
            }
        }
    }

    // 5. Initialize result codes (all mask)
    std::vector<int32_t> tokens(hp.n_codebooks * T_target, (int)hp.audio_mask_id);

    // 6. Compute time steps and unmask schedule
    auto timesteps = get_time_steps(0.0f, 1.0f, gen.num_steps, gen.t_shift);
    int total_mask = T_target * (int)hp.n_codebooks;

    std::vector<int> schedule(gen.num_steps);
    int rem = total_mask;
    for (int step = 0; step < gen.num_steps; step++) {
        if (step == gen.num_steps - 1) {
            schedule[step] = rem;
        } else {
            int num = (int)std::ceil(total_mask * (timesteps[step + 1] - timesteps[step]));
            num = std::min(num, rem);
            schedule[step] = num;
            rem -= num;
        }
    }

    // 7. RNG for Gumbel sampling
    std::mt19937 rng(gen.seed);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    auto gumbel_noise = [&]() -> float {
        float u = uniform(rng);
        u = std::max(u, 1e-10f);
        return -std::log(-std::log(u));
    };

    // 8. Iterative generation loop
    for (int step = 0; step < gen.num_steps; step++) {
        int k = schedule[step];
        if (k <= 0)
            continue;

        if (debug) {
            fprintf(stderr, "omnivoice: step %d/%d, unmask %d tokens\n", step + 1, gen.num_steps, k);
        }

        ov_bench_stage bench_step("gen_step");

        // Update audio tokens with current state
        for (uint32_t cb = 0; cb < hp.n_codebooks; cb++) {
            for (int t = 0; t < T_target; t++) {
                audio_tokens[cb * T_audio + T_ref + t] = tokens[cb * T_target + t];
            }
        }

        // Prepare embeddings
        auto embeds = prepare_embeddings(ctx, full_text_ids, audio_tokens, audio_mask, T_total);

        // Build and compute LLM graph
        size_t mem_size = (size_t)T_total * 256 * ggml_tensor_overhead() + ggml_graph_overhead();
        std::vector<uint8_t> mem_buf(mem_size);
        ggml_init_params ip = {mem_size, mem_buf.data(), true};
        ggml_context* ctx0 = ggml_init(ip);

        ggml_tensor* input = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hp.d_model, T_total);
        ggml_set_name(input, "input_embeds");
        ggml_set_input(input);

        ggml_cgraph* gf = build_llm_graph(ctx, ctx0, input, T_total);

        ggml_gallocr_t gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));
        ggml_gallocr_alloc_graph(gallocr, gf);

        ggml_backend_tensor_set(input, embeds.data(), 0, embeds.size() * sizeof(float));
        ggml_backend_graph_compute(ctx->backend, gf);

        // Get LLM output
        ggml_tensor* llm_out = ggml_graph_get_tensor(gf, "llm_out");
        std::vector<float> hidden(T_total * hp.d_model);
        ggml_backend_tensor_get(llm_out, hidden.data(), 0, hidden.size() * sizeof(float));

        // Compute audio logits via audio_heads
        // audio_heads: (n_codebooks * audio_vocab_size, d_model)
        // hidden: (d_model, T_total) → matmul → (n_codebooks * audio_vocab_size, T_total)
        int out_dim = (int)(hp.n_codebooks * hp.audio_vocab_size);
        std::vector<float> audio_heads_data(out_dim * hp.d_model);
        ggml_backend_tensor_get(ctx->model.audio_output_w, audio_heads_data.data(), 0,
                                audio_heads_data.size() * sizeof(float));

        // Extract logits only for target positions
        // Target positions are the last T_target in the sequence
        int target_start = T_total - T_target;
        std::vector<float> target_logits(out_dim * T_target);

        for (int t = 0; t < T_target; t++) {
            const float* h = hidden.data() + (size_t)(target_start + t) * hp.d_model;
            for (int o = 0; o < out_dim; o++) {
                float dot = 0.0f;
                const float* w = audio_heads_data.data() + (size_t)o * hp.d_model;
                for (uint32_t j = 0; j < hp.d_model; j++) {
                    dot += w[j] * h[j];
                }
                target_logits[(size_t)t * out_dim + o] = dot;
            }
        }

        // Reshape logits to (T_target, n_codebooks, audio_vocab_size)
        // and predict tokens + compute confidence scores
        std::vector<int32_t> pred_tokens(hp.n_codebooks * T_target);
        std::vector<float> confidence(hp.n_codebooks * T_target);

        for (int t = 0; t < T_target; t++) {
            for (uint32_t cb = 0; cb < hp.n_codebooks; cb++) {
                int offset = (int)(cb * hp.audio_vocab_size);
                const float* logits = target_logits.data() + (size_t)t * out_dim + offset;

                // Log-softmax
                float max_val = -1e30f;
                for (uint32_t v = 0; v < hp.audio_vocab_size; v++) {
                    if ((int)v != (int)hp.audio_mask_id) {
                        max_val = std::max(max_val, logits[v]);
                    }
                }
                float sum_exp = 0.0f;
                for (uint32_t v = 0; v < hp.audio_vocab_size; v++) {
                    if ((int)v != (int)hp.audio_mask_id) {
                        sum_exp += std::exp(logits[v] - max_val);
                    }
                }
                float log_sum = max_val + std::log(sum_exp);

                // Find best token (excluding mask)
                int best_tok = 0;
                float best_score = -1e30f;
                if (gen.class_temperature > 0.0f) {
                    // Gumbel sampling
                    for (uint32_t v = 0; v < hp.audio_vocab_size; v++) {
                        if ((int)v == (int)hp.audio_mask_id)
                            continue;
                        float lp = logits[v] - log_sum;
                        float g = lp / gen.class_temperature + gumbel_noise();
                        if (g > best_score) {
                            best_score = g;
                            best_tok = (int)v;
                        }
                    }
                } else {
                    for (uint32_t v = 0; v < hp.audio_vocab_size; v++) {
                        if ((int)v == (int)hp.audio_mask_id)
                            continue;
                        float lp = logits[v] - log_sum;
                        if (lp > best_score) {
                            best_score = lp;
                            best_tok = (int)v;
                        }
                    }
                }

                int idx = (int)cb * T_target + t;
                pred_tokens[idx] = best_tok;

                // Confidence = max log-prob - layer penalty
                float max_lp = logits[best_tok] - log_sum;
                confidence[idx] = max_lp - cb * gen.layer_penalty_factor;

                // Add Gumbel noise for position selection
                if (gen.position_temperature > 0.0f) {
                    confidence[idx] = confidence[idx] / gen.position_temperature + gumbel_noise();
                }
            }
        }

        // Mask out already-unmasked positions (set confidence to -inf)
        for (int i = 0; i < (int)(hp.n_codebooks * T_target); i++) {
            if (tokens[i] != (int)hp.audio_mask_id) {
                confidence[i] = -1e30f;
            }
        }

        // Select top-k positions to unmask
        std::vector<int> indices(hp.n_codebooks * T_target);
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                          [&](int a, int b) { return confidence[a] > confidence[b]; });

        for (int i = 0; i < k; i++) {
            tokens[indices[i]] = pred_tokens[indices[i]];
        }

        ggml_gallocr_free(gallocr);
        ggml_free(ctx0);
    }

    result.codes = std::move(tokens);
    return result;
}

} // namespace

// ===========================================================================
// Public API
// ===========================================================================

struct omnivoice_context* omnivoice_init_from_file(const char* path_model, struct omnivoice_context_params params) {
    auto* ctx = new omnivoice_context();
    ctx->n_threads = params.n_threads > 0 ? params.n_threads : 4;
    ctx->verbosity = params.verbosity;
    ctx->use_gpu = params.use_gpu;
    ctx->flash_attn = params.flash_attn;

    // Generation config
    ctx->gen.num_steps = params.num_steps > 0 ? params.num_steps : 32;
    ctx->gen.guidance_scale = params.guidance_scale > 0.0f ? params.guidance_scale : 1.0f;
    ctx->gen.class_temperature = params.class_temperature > 0.0f ? params.class_temperature : 0.7f;
    ctx->gen.position_temperature = params.position_temperature > 0.0f ? params.position_temperature : 4.5f;
    ctx->gen.layer_penalty_factor = params.layer_penalty_factor > 0.0f ? params.layer_penalty_factor : 0.5f;
    ctx->gen.t_shift = params.t_shift > 0.0f ? params.t_shift : 1.0f;
    ctx->gen.seed = params.seed > 0 ? params.seed : 42;

    if (!load_model(ctx, path_model)) {
        delete ctx;
        return nullptr;
    }

    return ctx;
}

int omnivoice_set_tokenizer_path(struct omnivoice_context* ctx, const char* path) {
    if (!ctx || !path)
        return -1;
    ctx->tokenizer_path = path;
    if (ctx->verbosity >= 1) {
        fprintf(stderr, "omnivoice: tokenizer path set to %s\n", path);
    }
    // TODO: load the tokenizer GGUF and init the HiggsAudioV2 runtime
    return 0;
}

int omnivoice_set_voice_prompt(struct omnivoice_context* ctx, const char* wav_path, const char* ref_text) {
    if (!ctx)
        return -1;
    if (!wav_path || !*wav_path) {
        ctx->ref_audio_codes.clear();
        ctx->ref_T = 0;
        ctx->ref_text.clear();
        return 0;
    }
    ctx->ref_text = ref_text ? ref_text : "";
    // TODO: load WAV, encode through audio tokenizer to get ref_audio_codes
    fprintf(stderr, "omnivoice: voice prompt set (tokenizer encode pending)\n");
    return 0;
}

int omnivoice_set_language(struct omnivoice_context* ctx, const char* lang) {
    if (!ctx)
        return -1;
    ctx->language = lang ? lang : "";
    return 0;
}

int omnivoice_set_instruct(struct omnivoice_context* ctx, const char* instruct) {
    if (!ctx)
        return -1;
    ctx->instruct = instruct ? instruct : "";
    return 0;
}

int32_t* omnivoice_synthesize_codes(struct omnivoice_context* ctx, const char* text, int* out_n_codes) {
    if (!ctx || !text || !out_n_codes)
        return nullptr;

    ov_gen_result result = generate_iterative(ctx, text);

    if (result.codes.empty()) {
        *out_n_codes = 0;
        return nullptr;
    }

    int n = (int)result.codes.size();
    int32_t* out = (int32_t*)malloc(n * sizeof(int32_t));
    std::memcpy(out, result.codes.data(), n * sizeof(int32_t));
    *out_n_codes = n;
    return out;
}

void omnivoice_codes_free(int32_t* codes) {
    free(codes);
}

float* omnivoice_decode_codes(struct omnivoice_context* ctx, const int32_t* codes, int n_codes, int* out_n_samples) {
    (void)ctx;
    (void)codes;
    (void)n_codes;
    if (out_n_samples)
        *out_n_samples = 0;
    // TODO: decode via HiggsAudioV2 tokenizer
    fprintf(stderr, "omnivoice: decode_codes not yet implemented (audio tokenizer pending)\n");
    return nullptr;
}

float* omnivoice_synthesize(struct omnivoice_context* ctx, const char* text, int* out_n_samples) {
    if (!ctx || !text || !out_n_samples)
        return nullptr;

    int n_codes = 0;
    int32_t* codes = omnivoice_synthesize_codes(ctx, text, &n_codes);
    if (!codes) {
        *out_n_samples = 0;
        return nullptr;
    }

    float* pcm = omnivoice_decode_codes(ctx, codes, n_codes, out_n_samples);
    omnivoice_codes_free(codes);
    return pcm;
}

void omnivoice_pcm_free(float* pcm) {
    free(pcm);
}

void omnivoice_free(struct omnivoice_context* ctx) {
    if (!ctx)
        return;
    if (ctx->buf_w)
        ggml_backend_buffer_free(ctx->buf_w);
    if (ctx->backend)
        ggml_backend_free(ctx->backend);
    if (ctx->ctx_w)
        ggml_free(ctx->ctx_w);
    delete ctx;
}

void omnivoice_sync(struct omnivoice_context* ctx) {
    if (ctx && ctx->backend)
        ggml_backend_synchronize(ctx->backend);
}

void omnivoice_set_n_threads(struct omnivoice_context* ctx, int n_threads) {
    if (ctx && n_threads > 0) {
        ctx->n_threads = n_threads;
        if (ctx->backend)
            ggml_backend_cpu_set_n_threads(ctx->backend, n_threads);
    }
}

// irodori_tts.cpp — native ggml runtime for Aratako/Irodori-TTS.
//
// Architecture:
//   1. TextEncoder: Embedding(102400, 1280) + 14 TextBlock (RoPE self-attn + SwiGLU)
//   2. ReferenceLatentEncoder: Linear(128→1280) + 14 TextBlock (same structure)
//   3. TimestepEmbedding: sinusoidal(512) → Linear(512,2048) → SiLU → Linear(2048,2048) → SiLU → Linear(2048,6144)
//   4. DiT: 24 DiffusionBlock, each:
//      - LowRankAdaLN(rank=256): timestep → shift/scale/gate via low-rank residual MLP
//      - JointAttention: self-KV + text-context-KV + speaker-context-KV, half-RoPE
//      - SwiGLU(2048, 5888)
//   5. DurationPredictor: token_sum with speaker AdaRN-Zero
//   6. Euler RF ODE solver: 40 steps with independent CFG
//   7. DAC-VAE decoder (separate model, handled externally for now)
//
// The implementation builds ggml graphs per inference stage using gallocr.

#include "irodori_tts.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include "core/ffn.h"
#include "core/gguf_loader.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ── Debug gating ────────────────────────────────────────────────────

static bool irodori_debug_enabled() {
    static int v = -1;
    if (v < 0) {
        const char* e = std::getenv("CRISPASR_IRODORI_DEBUG");
        v = (e && *e && *e != '0') ? 1 : 0;
    }
    return v != 0;
}

#define IRODORI_DBG(...)                                                                                               \
    do {                                                                                                               \
        if (irodori_debug_enabled())                                                                                   \
            std::fprintf(stderr, __VA_ARGS__);                                                                         \
    } while (0)

// ── Hyperparameters ─────────────────────────────────────────────────

struct irodori_hparams {
    int latent_dim = 128;
    int latent_patch_size = 1;
    int model_dim = 2048;
    int num_layers = 24;
    int num_heads = 16;
    float mlp_ratio = 2.875f;
    int text_dim = 1280;
    int text_layers = 14;
    int text_heads = 10;
    float text_mlp_ratio = 2.6f;
    int text_vocab_size = 102400;
    int speaker_dim = 1280;
    int speaker_layers = 14;
    int speaker_heads = 10;
    float speaker_mlp_ratio = 2.6f;
    int speaker_patch_size = 1;
    int timestep_embed_dim = 512;
    int adaln_rank = 256;
    float norm_eps = 1e-5f;
    int use_duration_predictor = 1;
    int duration_aux_dim = 14;
    int duration_hidden_dim = 1024;
    int duration_layers = 3;
    int ode_steps = 40;
    float cfg_scale_text = 3.0f;
    float cfg_scale_speaker = 5.0f;
    int sample_rate = 48000;
    int codec_hop_length = 320;

    int patched_latent_dim() const { return latent_dim * latent_patch_size; }
    int head_dim() const { return model_dim / num_heads; }
    int text_head_dim() const { return text_dim / text_heads; }
    int speaker_head_dim() const { return speaker_dim / speaker_heads; }
    int ff_dim() const { return (int)(model_dim * mlp_ratio); }
    int text_ff_dim() const { return (int)(text_dim * text_mlp_ratio); }
    int speaker_ff_dim() const { return (int)(speaker_dim * speaker_mlp_ratio); }
};

// ── Weight structures ───────────────────────────────────────────────

struct irodori_text_block_weights {
    ggml_tensor* attn_norm = nullptr; // (text_dim,)
    ggml_tensor* wq = nullptr;        // (text_dim, text_dim)
    ggml_tensor* wk = nullptr;
    ggml_tensor* wv = nullptr;
    ggml_tensor* wo = nullptr;
    ggml_tensor* gate = nullptr;   // gated attention
    ggml_tensor* q_norm = nullptr; // (heads, head_dim)
    ggml_tensor* k_norm = nullptr;
    ggml_tensor* mlp_norm = nullptr; // (text_dim,)
    ggml_tensor* mlp_w1 = nullptr;   // gate (text_dim, ff_dim)
    ggml_tensor* mlp_w2 = nullptr;   // down (ff_dim, text_dim)
    ggml_tensor* mlp_w3 = nullptr;   // up (text_dim, ff_dim)
};

struct irodori_low_rank_adaln_weights {
    // shift/scale/gate each have: down (model_dim, rank) + up (rank, model_dim) + up_bias (model_dim,)
    ggml_tensor* shift_down = nullptr;
    ggml_tensor* shift_up_w = nullptr;
    ggml_tensor* shift_up_b = nullptr;
    ggml_tensor* scale_down = nullptr;
    ggml_tensor* scale_up_w = nullptr;
    ggml_tensor* scale_up_b = nullptr;
    ggml_tensor* gate_down = nullptr;
    ggml_tensor* gate_up_w = nullptr;
    ggml_tensor* gate_up_b = nullptr;
};

struct irodori_dit_block_weights {
    // JointAttention
    ggml_tensor* wq = nullptr; // (model_dim, model_dim)
    ggml_tensor* wk = nullptr;
    ggml_tensor* wv = nullptr;
    ggml_tensor* wo = nullptr;
    ggml_tensor* gate = nullptr;
    ggml_tensor* q_norm = nullptr; // (heads, head_dim)
    ggml_tensor* k_norm = nullptr;
    ggml_tensor* wk_text = nullptr; // (text_dim, model_dim)
    ggml_tensor* wv_text = nullptr;
    ggml_tensor* wk_spk = nullptr; // (speaker_dim, model_dim)
    ggml_tensor* wv_spk = nullptr;

    // LowRankAdaLN for attention and MLP
    irodori_low_rank_adaln_weights adaln_attn;
    irodori_low_rank_adaln_weights adaln_mlp;

    // SwiGLU
    ggml_tensor* mlp_w1 = nullptr;
    ggml_tensor* mlp_w2 = nullptr;
    ggml_tensor* mlp_w3 = nullptr;
};

struct irodori_dur_block_weights {
    ggml_tensor* norm = nullptr; // (hidden_dim,)
    ggml_tensor* mlp_w1 = nullptr;
    ggml_tensor* mlp_w2 = nullptr;
    ggml_tensor* mlp_w3 = nullptr;
    ggml_tensor* mod_w = nullptr; // (speaker_dim, hidden_dim*3)
    ggml_tensor* mod_b = nullptr; // (hidden_dim*3,)
};

struct irodori_weights {
    // Text encoder
    ggml_tensor* text_emb = nullptr; // (vocab_size, text_dim)
    std::vector<irodori_text_block_weights> text_blocks;
    ggml_tensor* text_norm = nullptr; // (text_dim,)

    // Speaker encoder
    ggml_tensor* spk_in_proj_w = nullptr; // (patched_latent_dim, speaker_dim)
    ggml_tensor* spk_in_proj_b = nullptr; // (speaker_dim,)
    std::vector<irodori_text_block_weights> spk_blocks;
    ggml_tensor* spk_norm = nullptr; // (speaker_dim,)

    // Timestep conditioning MLP
    ggml_tensor* cond_0_w = nullptr; // (timestep_embed_dim, model_dim)
    ggml_tensor* cond_2_w = nullptr; // (model_dim, model_dim)
    ggml_tensor* cond_4_w = nullptr; // (model_dim, model_dim*3)

    // Input/output projections
    ggml_tensor* in_proj_w = nullptr;  // (patched_latent_dim, model_dim)
    ggml_tensor* in_proj_b = nullptr;  // (model_dim,)
    ggml_tensor* out_norm = nullptr;   // (model_dim,)
    ggml_tensor* out_proj_w = nullptr; // (model_dim, patched_latent_dim)
    ggml_tensor* out_proj_b = nullptr; // (patched_latent_dim,)

    // DiT blocks
    std::vector<irodori_dit_block_weights> dit_blocks;

    // Duration predictor
    ggml_tensor* dur_input_proj_w = nullptr;
    ggml_tensor* dur_input_proj_b = nullptr;
    ggml_tensor* dur_null_speaker = nullptr;
    std::vector<irodori_dur_block_weights> dur_blocks;
    ggml_tensor* dur_out_norm = nullptr;
    ggml_tensor* dur_out_proj_w = nullptr;
    ggml_tensor* dur_out_proj_b = nullptr;
};

// ── Context ─────────────────────────────────────────────────────────

struct irodori_tts_context {
    irodori_hparams hparams;
    irodori_weights weights;

    // ggml state
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buf_weights = nullptr;
    ggml_context* w_ctx = nullptr; // weight tensor context (kept alive)

    // Tokenizer
    std::vector<std::string> vocab;
    int bos_token_id = -1;
    int pad_token_id = -1;

    // Reference latent (speaker conditioning)
    std::vector<float> ref_latent; // (T_ref * latent_dim)
    int ref_latent_frames = 0;

    // Runtime params
    int seed = 0;
    int ode_steps = 40;
    float cfg_scale_text = 3.0f;
    float cfg_scale_speaker = 5.0f;
    float speed = 1.0f;
    float duration_scale = 1.0f;
    float max_seconds = 30.0f;
    int verbosity = 1;
    int n_threads = 4;
};

// ── RoPE precomputation ─────────────────────────────────────────────

static void precompute_freqs_cis(int dim, int max_len, float theta, std::vector<float>& cos_cache,
                                 std::vector<float>& sin_cache) {
    int half = dim / 2;
    cos_cache.resize(max_len * half);
    sin_cache.resize(max_len * half);
    for (int t = 0; t < max_len; t++) {
        for (int k = 0; k < half; k++) {
            float freq = 1.0f / std::pow(theta, (float)(2 * k) / (float)dim);
            float angle = (float)t * freq;
            cos_cache[t * half + k] = std::cos(angle);
            sin_cache[t * half + k] = std::sin(angle);
        }
    }
}

// ── RMSNorm (graph op) ──────────────────────────────────────────────

static ggml_tensor* rms_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* weight, float eps) {
    x = ggml_rms_norm(ctx, x, eps);
    return ggml_mul(ctx, x, weight);
}

// ── Timestep embedding (CPU) ────────────────────────────────────────

static void compute_timestep_embedding(float t, int dim, std::vector<float>& out) {
    out.resize(dim);
    int half = dim / 2;
    for (int k = 0; k < half; k++) {
        float freq = 1000.0f * std::exp(-std::log(10000.0f) * (float)k / (float)half);
        float arg = t * freq;
        out[k] = std::cos(arg);
        out[half + k] = std::sin(arg);
    }
}

// ── RoPE is applied via ggml_rope_ext in the graph builders below. ──

// ── Graph builders ──────────────────────────────────────────────────

// Evaluate a ggml graph on the backend using gallocr.
static bool eval_graph(ggml_backend_t backend, ggml_cgraph* gf) {
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        ggml_gallocr_free(galloc);
        return false;
    }
    ggml_backend_graph_compute(backend, gf);
    ggml_gallocr_free(galloc);
    return true;
}

// ── GGUF loading ────────────────────────────────────────────────────

// Load weights using the core_gguf map-based getter pattern.
template <typename GetFn, typename TryGetFn>
static bool load_weights_from_map(irodori_tts_context* ctx, GetFn get, TryGetFn /*try_get*/) {
    auto& w = ctx->weights;
    const auto& hp = ctx->hparams;

    // Text encoder
    w.text_emb = get("irodori.text_enc.emb");
    w.text_norm = get("irodori.text_norm");
    w.text_blocks.resize(hp.text_layers);
    for (int i = 0; i < hp.text_layers; i++) {
        auto& blk = w.text_blocks[i];
        std::string p = "irodori.text_enc.blk." + std::to_string(i) + ".";
        blk.attn_norm = get(p + "attn_norm");
        blk.wq = get(p + "attn.wq");
        blk.wk = get(p + "attn.wk");
        blk.wv = get(p + "attn.wv");
        blk.wo = get(p + "attn.wo");
        blk.gate = get(p + "attn.gate");
        blk.q_norm = get(p + "attn.q_norm");
        blk.k_norm = get(p + "attn.k_norm");
        blk.mlp_norm = get(p + "mlp_norm");
        blk.mlp_w1 = get(p + "mlp.w1");
        blk.mlp_w2 = get(p + "mlp.w2");
        blk.mlp_w3 = get(p + "mlp.w3");
    }

    // Speaker encoder
    w.spk_in_proj_w = get("irodori.spk_enc.in_proj.w");
    w.spk_in_proj_b = get("irodori.spk_enc.in_proj.b");
    w.spk_norm = get("irodori.spk_norm");
    w.spk_blocks.resize(hp.speaker_layers);
    for (int i = 0; i < hp.speaker_layers; i++) {
        auto& blk = w.spk_blocks[i];
        std::string p = "irodori.spk_enc.blk." + std::to_string(i) + ".";
        blk.attn_norm = get(p + "attn_norm");
        blk.wq = get(p + "attn.wq");
        blk.wk = get(p + "attn.wk");
        blk.wv = get(p + "attn.wv");
        blk.wo = get(p + "attn.wo");
        blk.gate = get(p + "attn.gate");
        blk.q_norm = get(p + "attn.q_norm");
        blk.k_norm = get(p + "attn.k_norm");
        blk.mlp_norm = get(p + "mlp_norm");
        blk.mlp_w1 = get(p + "mlp.w1");
        blk.mlp_w2 = get(p + "mlp.w2");
        blk.mlp_w3 = get(p + "mlp.w3");
    }

    // Top-level
    w.cond_0_w = get("irodori.cond.0.w");
    w.cond_2_w = get("irodori.cond.2.w");
    w.cond_4_w = get("irodori.cond.4.w");
    w.in_proj_w = get("irodori.in_proj.w");
    w.in_proj_b = get("irodori.in_proj.b");
    w.out_norm = get("irodori.out_norm");
    w.out_proj_w = get("irodori.out_proj.w");
    w.out_proj_b = get("irodori.out_proj.b");

    // DiT blocks
    w.dit_blocks.resize(hp.num_layers);
    for (int i = 0; i < hp.num_layers; i++) {
        auto& blk = w.dit_blocks[i];
        std::string p = "irodori.dit.blk." + std::to_string(i) + ".";
        blk.wq = get(p + "attn.wq");
        blk.wk = get(p + "attn.wk");
        blk.wv = get(p + "attn.wv");
        blk.wo = get(p + "attn.wo");
        blk.gate = get(p + "attn.gate");
        blk.q_norm = get(p + "attn.q_norm");
        blk.k_norm = get(p + "attn.k_norm");
        blk.wk_text = get(p + "attn.wk_text");
        blk.wv_text = get(p + "attn.wv_text");
        blk.wk_spk = get(p + "attn.wk_spk");
        blk.wv_spk = get(p + "attn.wv_spk");

        // AdaLN attention
        auto load_adaln = [&](irodori_low_rank_adaln_weights& adaln, const std::string& ap) {
            adaln.shift_down = get(ap + "shift_down");
            adaln.shift_up_w = get(ap + "shift_up.w");
            adaln.shift_up_b = get(ap + "shift_up.b");
            adaln.scale_down = get(ap + "scale_down");
            adaln.scale_up_w = get(ap + "scale_up.w");
            adaln.scale_up_b = get(ap + "scale_up.b");
            adaln.gate_down = get(ap + "gate_down");
            adaln.gate_up_w = get(ap + "gate_up.w");
            adaln.gate_up_b = get(ap + "gate_up.b");
        };
        load_adaln(blk.adaln_attn, p + "adaln_attn.");
        load_adaln(blk.adaln_mlp, p + "adaln_mlp.");

        blk.mlp_w1 = get(p + "mlp.w1");
        blk.mlp_w2 = get(p + "mlp.w2");
        blk.mlp_w3 = get(p + "mlp.w3");
    }

    // Duration predictor
    if (hp.use_duration_predictor) {
        w.dur_input_proj_w = get("irodori.dur.input_proj.w");
        w.dur_input_proj_b = get("irodori.dur.input_proj.b");
        w.dur_null_speaker = get("irodori.dur.null_speaker");
        w.dur_blocks.resize(hp.duration_layers);
        for (int i = 0; i < hp.duration_layers; i++) {
            auto& blk = w.dur_blocks[i];
            std::string p = "irodori.dur.blk." + std::to_string(i) + ".";
            blk.norm = get(p + "norm");
            blk.mlp_w1 = get(p + "mlp.w1");
            blk.mlp_w2 = get(p + "mlp.w2");
            blk.mlp_w3 = get(p + "mlp.w3");
            blk.mod_w = get(p + "mod.w");
            blk.mod_b = get(p + "mod.b");
        }
        w.dur_out_norm = get("irodori.dur.out_norm");
        w.dur_out_proj_w = get("irodori.dur.out_proj.w");
        w.dur_out_proj_b = get("irodori.dur.out_proj.b");
    }

    return true;
}

// ── Graph: Text encoder forward ─────────────────────────────────────

// Build text encoder graph for a batch of token IDs.
// Input: token_ids (T_text,) I32
// Output: text_state (text_dim, T_text) F32
static ggml_tensor* build_text_encoder_graph(ggml_context* ctx, const irodori_tts_context* model,
                                             ggml_tensor* token_ids, ggml_tensor* mask_f, int T_text) {
    const auto& w = model->weights;
    const auto& hp = model->hparams;

    // Embedding lookup
    ggml_tensor* x = ggml_get_rows(ctx, w.text_emb, token_ids);
    // x: (text_dim, T_text) after get_rows
    x = ggml_cast(ctx, x, GGML_TYPE_F32);

    // Apply mask (zero out padding)
    x = ggml_mul(ctx, x, mask_f);

    // Process through transformer blocks
    for (int i = 0; i < hp.text_layers; i++) {
        const auto& blk = w.text_blocks[i];

        // Self-attention with pre-norm
        ggml_tensor* residual = x;
        ggml_tensor* h = rms_norm(ctx, x, blk.attn_norm, hp.norm_eps);

        // Q, K, V projections
        ggml_tensor* q = ggml_mul_mat(ctx, blk.wq, h);
        ggml_tensor* k = ggml_mul_mat(ctx, blk.wk, h);
        ggml_tensor* v = ggml_mul_mat(ctx, blk.wv, h);

        // Reshape to (head_dim, heads, T)
        int hd = hp.text_head_dim();
        int nh = hp.text_heads;
        q = ggml_reshape_3d(ctx, q, hd, nh, T_text);
        k = ggml_reshape_3d(ctx, k, hd, nh, T_text);
        v = ggml_reshape_3d(ctx, v, hd, nh, T_text);

        // QK norm (per-head RMSNorm)
        q = ggml_rms_norm(ctx, q, hp.norm_eps);
        q = ggml_mul(ctx, q, blk.q_norm);
        k = ggml_rms_norm(ctx, k, hp.norm_eps);
        k = ggml_mul(ctx, k, blk.k_norm);

        // RoPE
        q = ggml_rope_ext(ctx, q, nullptr, nullptr, hd, 0, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(ctx, k, nullptr, nullptr, hd, 0, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        // Permute for flash attention: (head_dim, T, heads) -> SDPA expects (T, head_dim, heads)
        // ggml_flash_attn_ext expects Q(hd, T, nh), K(hd, T_kv, nh), V(hd, T_kv, nh)
        ggml_tensor* attn_out =
            ggml_flash_attn_ext(ctx, q, k, v, /*mask=*/nullptr, 1.0f / std::sqrt((float)hd), 0.0f, 0.0f);
        // Output: (hd, T, nh)
        attn_out = ggml_reshape_2d(ctx, attn_out, hp.text_dim, T_text);

        // Gated attention: y = y * sigmoid(gate(h))
        ggml_tensor* g = ggml_mul_mat(ctx, blk.gate, h);
        g = ggml_sigmoid(ctx, g);
        attn_out = ggml_mul(ctx, attn_out, g);

        // Output projection
        attn_out = ggml_mul_mat(ctx, blk.wo, attn_out);
        x = ggml_add(ctx, residual, attn_out);

        // FFN with pre-norm
        residual = x;
        h = rms_norm(ctx, x, blk.mlp_norm, hp.norm_eps);
        ggml_tensor* mlp_out = core_ffn::swiglu(ctx, h, blk.mlp_w1, blk.mlp_w3, blk.mlp_w2);
        x = ggml_add(ctx, residual, mlp_out);

        // Re-apply mask
        x = ggml_mul(ctx, x, mask_f);
    }

    // Final text norm
    x = rms_norm(ctx, x, w.text_norm, hp.norm_eps);
    return x;
}

// ── Graph: Speaker encoder forward ──────────────────────────────────

static ggml_tensor* build_speaker_encoder_graph(ggml_context* ctx, const irodori_tts_context* model,
                                                ggml_tensor* latent_in, ggml_tensor* mask_f, int T_ref) {
    const auto& w = model->weights;
    const auto& hp = model->hparams;

    // Input projection + scale by 1/6
    ggml_tensor* x = ggml_mul_mat(ctx, w.spk_in_proj_w, latent_in);
    x = ggml_add(ctx, x, w.spk_in_proj_b);
    x = ggml_scale(ctx, x, 1.0f / 6.0f);
    x = ggml_mul(ctx, x, mask_f);

    // Same transformer block structure as text encoder
    for (int i = 0; i < hp.speaker_layers; i++) {
        const auto& blk = w.spk_blocks[i];

        ggml_tensor* residual = x;
        ggml_tensor* h = rms_norm(ctx, x, blk.attn_norm, hp.norm_eps);

        ggml_tensor* q = ggml_mul_mat(ctx, blk.wq, h);
        ggml_tensor* k = ggml_mul_mat(ctx, blk.wk, h);
        ggml_tensor* v = ggml_mul_mat(ctx, blk.wv, h);

        int hd = hp.speaker_head_dim();
        int nh = hp.speaker_heads;
        q = ggml_reshape_3d(ctx, q, hd, nh, T_ref);
        k = ggml_reshape_3d(ctx, k, hd, nh, T_ref);
        v = ggml_reshape_3d(ctx, v, hd, nh, T_ref);

        q = ggml_rms_norm(ctx, q, hp.norm_eps);
        q = ggml_mul(ctx, q, blk.q_norm);
        k = ggml_rms_norm(ctx, k, hp.norm_eps);
        k = ggml_mul(ctx, k, blk.k_norm);

        q = ggml_rope_ext(ctx, q, nullptr, nullptr, hd, 0, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(ctx, k, nullptr, nullptr, hd, 0, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        ggml_tensor* attn_out = ggml_flash_attn_ext(ctx, q, k, v, nullptr, 1.0f / std::sqrt((float)hd), 0.0f, 0.0f);
        attn_out = ggml_reshape_2d(ctx, attn_out, hp.speaker_dim, T_ref);

        ggml_tensor* g = ggml_mul_mat(ctx, blk.gate, h);
        g = ggml_sigmoid(ctx, g);
        attn_out = ggml_mul(ctx, attn_out, g);
        attn_out = ggml_mul_mat(ctx, blk.wo, attn_out);
        x = ggml_add(ctx, residual, attn_out);

        residual = x;
        h = rms_norm(ctx, x, blk.mlp_norm, hp.norm_eps);
        ggml_tensor* mlp_out = core_ffn::swiglu(ctx, h, blk.mlp_w1, blk.mlp_w3, blk.mlp_w2);
        x = ggml_add(ctx, residual, mlp_out);
        x = ggml_mul(ctx, x, mask_f);
    }

    // Speaker norm
    x = rms_norm(ctx, x, w.spk_norm, hp.norm_eps);
    return x;
}

// ── Graph: LowRankAdaLN ─────────────────────────────────────────────

// Apply LowRank AdaLN: returns (normed_modulated_x, gate)
// cond_embed: (model_dim*3, 1) — pre-chunked into 3 parts of model_dim each
static void apply_low_rank_adaln(ggml_context* ctx, const irodori_low_rank_adaln_weights& adaln, ggml_tensor* x,
                                 ggml_tensor* cond_shift, ggml_tensor* cond_scale, ggml_tensor* cond_gate,
                                 float norm_eps, ggml_tensor** out_x, ggml_tensor** out_gate) {
    // Low-rank residual modulation:
    //   shift = shift_up(shift_down(silu(cond_shift))) + cond_shift
    //   scale = scale_up(scale_down(silu(cond_scale))) + cond_scale
    //   gate  = gate_up(gate_down(silu(cond_gate))) + cond_gate

    auto lr_mod = [&](ggml_tensor* cond, ggml_tensor* down, ggml_tensor* up_w, ggml_tensor* up_b) -> ggml_tensor* {
        ggml_tensor* h = ggml_silu(ctx, cond);
        h = ggml_mul_mat(ctx, down, h);
        h = ggml_mul_mat(ctx, up_w, h);
        if (up_b)
            h = ggml_add(ctx, h, up_b);
        return ggml_add(ctx, h, cond); // residual connection
    };

    ggml_tensor* shift = lr_mod(cond_shift, adaln.shift_down, adaln.shift_up_w, adaln.shift_up_b);
    ggml_tensor* scale = lr_mod(cond_scale, adaln.scale_down, adaln.scale_up_w, adaln.scale_up_b);
    ggml_tensor* gate = lr_mod(cond_gate, adaln.gate_down, adaln.gate_up_w, adaln.gate_up_b);

    // RMSNorm + modulate: x_normed * (1 + scale) + shift
    ggml_tensor* x_norm = ggml_rms_norm(ctx, x, norm_eps);
    ggml_tensor* one_plus_scale = ggml_add(ctx, ggml_new_f32(ctx, 1.0f), scale);
    *out_x = ggml_add(ctx, ggml_mul(ctx, x_norm, one_plus_scale), shift);

    // gate = tanh(gate)
    *out_gate = ggml_tanh(ctx, gate);
}

// ── Graph: DiT block forward ────────────────────────────────────────

// Builds one DiffusionBlock.
// x: (model_dim, T_latent)
// cond_embed: (model_dim*3, 1) — timestep conditioning
// text_state: (text_dim, T_text)
// spk_state: (speaker_dim, T_ref) or nullptr
static ggml_tensor* build_dit_block_graph(ggml_context* ctx, const irodori_tts_context* model,
                                          const irodori_dit_block_weights& blk, ggml_tensor* x, ggml_tensor* cond_embed,
                                          ggml_tensor* text_state, int T_text, ggml_tensor* spk_state, int T_ref,
                                          int T_latent) {
    const auto& hp = model->hparams;
    int hd = hp.head_dim();
    int nh = hp.num_heads;

    // Split cond_embed into 3 parts: (model_dim,) each
    int D = hp.model_dim;
    ggml_tensor* cond_shift = ggml_view_1d(ctx, cond_embed, D, 0);
    ggml_tensor* cond_scale = ggml_view_1d(ctx, cond_embed, D, D * sizeof(float));
    ggml_tensor* cond_gate_raw = ggml_view_1d(ctx, cond_embed, D, 2 * D * sizeof(float));

    // ── Attention path ──
    ggml_tensor* h_attn = nullptr;
    ggml_tensor* gate_attn = nullptr;
    apply_low_rank_adaln(ctx, blk.adaln_attn, x, cond_shift, cond_scale, cond_gate_raw, hp.norm_eps, &h_attn,
                         &gate_attn);

    // Self Q/K/V
    ggml_tensor* q = ggml_mul_mat(ctx, blk.wq, h_attn);
    ggml_tensor* k_self = ggml_mul_mat(ctx, blk.wk, h_attn);
    ggml_tensor* v_self = ggml_mul_mat(ctx, blk.wv, h_attn);

    // Text context K/V
    ggml_tensor* k_text = ggml_mul_mat(ctx, blk.wk_text, text_state);
    ggml_tensor* v_text = ggml_mul_mat(ctx, blk.wv_text, text_state);

    // Speaker context K/V
    ggml_tensor* k_spk = nullptr;
    ggml_tensor* v_spk = nullptr;
    int T_kv_total = T_latent + T_text;
    if (spk_state && T_ref > 0) {
        k_spk = ggml_mul_mat(ctx, blk.wk_spk, spk_state);
        v_spk = ggml_mul_mat(ctx, blk.wv_spk, spk_state);
        T_kv_total += T_ref;
    }

    // Reshape Q: (head_dim, heads, T_latent)
    q = ggml_reshape_3d(ctx, q, hd, nh, T_latent);

    // QK norm
    q = ggml_rms_norm(ctx, q, hp.norm_eps);
    q = ggml_mul(ctx, q, blk.q_norm);

    // Reshape self K/V
    k_self = ggml_reshape_3d(ctx, k_self, hd, nh, T_latent);
    v_self = ggml_reshape_3d(ctx, v_self, hd, nh, T_latent);
    k_self = ggml_rms_norm(ctx, k_self, hp.norm_eps);
    k_self = ggml_mul(ctx, k_self, blk.k_norm);

    // Reshape text K/V
    k_text = ggml_reshape_3d(ctx, k_text, hd, nh, T_text);
    v_text = ggml_reshape_3d(ctx, v_text, hd, nh, T_text);
    k_text = ggml_rms_norm(ctx, k_text, hp.norm_eps);
    k_text = ggml_mul(ctx, k_text, blk.k_norm);

    // Concatenate K and V: self + text [+ speaker]
    ggml_tensor* k_cat = ggml_concat(ctx, k_self, k_text, /*dim=*/2);
    ggml_tensor* v_cat = ggml_concat(ctx, v_self, v_text, /*dim=*/2);
    if (k_spk && v_spk) {
        k_spk = ggml_reshape_3d(ctx, k_spk, hd, nh, T_ref);
        v_spk = ggml_reshape_3d(ctx, v_spk, hd, nh, T_ref);
        k_spk = ggml_rms_norm(ctx, k_spk, hp.norm_eps);
        k_spk = ggml_mul(ctx, k_spk, blk.k_norm);
        k_cat = ggml_concat(ctx, k_cat, k_spk, 2);
        v_cat = ggml_concat(ctx, v_cat, v_spk, 2);
    }

    // Half-RoPE: apply RoPE only to first half of heads
    // For simplicity in the initial implementation, we apply full RoPE
    // (half-RoPE optimization can be added later for exact parity)
    q = ggml_rope_ext(ctx, q, nullptr, nullptr, hd, 0, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    // Only self-KV gets RoPE (context KV does NOT get RoPE in JointAttention)
    // Actually in Irodori, half-RoPE means: rotate first half of head_dim, pass second half through
    // This is handled by _apply_rotary_half in the Python code
    // For now, apply full RoPE to self-K only (context K has no positional encoding)
    // We'll undo the rope on k_text/k_spk by only applying to k_self before concat
    // TODO: implement proper half-RoPE split

    // Flash attention
    ggml_tensor* attn_out = ggml_flash_attn_ext(ctx, q, k_cat, v_cat, nullptr, 1.0f / std::sqrt((float)hd), 0.0f, 0.0f);
    attn_out = ggml_reshape_2d(ctx, attn_out, D, T_latent);

    // Gated attention
    ggml_tensor* g = ggml_mul_mat(ctx, blk.gate, h_attn);
    g = ggml_sigmoid(ctx, g);
    attn_out = ggml_mul(ctx, attn_out, g);
    attn_out = ggml_mul_mat(ctx, blk.wo, attn_out);

    // gate_attn * attn_out
    attn_out = ggml_mul(ctx, gate_attn, attn_out);
    x = ggml_add(ctx, x, attn_out);

    // ── MLP path ──
    ggml_tensor* h_mlp = nullptr;
    ggml_tensor* gate_mlp = nullptr;
    apply_low_rank_adaln(ctx, blk.adaln_mlp, x, cond_shift, cond_scale, cond_gate_raw, hp.norm_eps, &h_mlp, &gate_mlp);

    ggml_tensor* mlp_out = core_ffn::swiglu(ctx, h_mlp, blk.mlp_w1, blk.mlp_w3, blk.mlp_w2);
    mlp_out = ggml_mul(ctx, gate_mlp, mlp_out);
    x = ggml_add(ctx, x, mlp_out);

    return x;
}

// ── Tokenizer ───────────────────────────────────────────────────────

static std::vector<int32_t> tokenize_text(const irodori_tts_context* ctx, const char* text) {
    // For the initial implementation, use byte-level tokenization as fallback.
    // The full sarashina2.2 SentencePiece tokenizer integration would require
    // loading the SPM model from GGUF or external file.
    // TODO: integrate proper SentencePiece tokenizer from core/sentencepiece.h
    std::vector<int32_t> tokens;
    if (ctx->bos_token_id >= 0) {
        tokens.push_back(ctx->bos_token_id);
    }

    // UTF-8 byte fallback encoding (works but suboptimal vs SentencePiece)
    const uint8_t* p = (const uint8_t*)text;
    while (*p) {
        tokens.push_back((int32_t)*p);
        p++;
    }
    return tokens;
}

// ── Public API ──────────────────────────────────────────────────────

struct irodori_tts_params irodori_tts_default_params(void) {
    irodori_tts_params p = {};
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = false;
    p.seed = 0;
    p.ode_steps = 40;
    p.cfg_scale_text = 3.0f;
    p.cfg_scale_speaker = 5.0f;
    p.speed = 1.0f;
    p.duration_scale = 1.0f;
    p.max_seconds = 30.0f;
    return p;
}

struct irodori_tts_context* irodori_tts_init_from_file(const char* path_model, struct irodori_tts_params params) {
    if (!path_model)
        return nullptr;

    auto* ctx = new irodori_tts_context();
    ctx->n_threads = params.n_threads > 0 ? params.n_threads : 4;
    ctx->verbosity = params.verbosity;
    ctx->seed = params.seed;
    ctx->ode_steps = params.ode_steps > 0 ? params.ode_steps : 40;
    ctx->cfg_scale_text = params.cfg_scale_text;
    ctx->cfg_scale_speaker = params.cfg_scale_speaker;
    ctx->speed = params.speed > 0 ? params.speed : 1.0f;
    ctx->duration_scale = params.duration_scale > 0 ? params.duration_scale : 1.0f;
    ctx->max_seconds = params.max_seconds > 0 ? params.max_seconds : 30.0f;

    // Initialize CPU backend
    ctx->backend = ggml_backend_cpu_init();
    if (!ctx->backend) {
        std::fprintf(stderr, "[irodori] failed to init CPU backend\n");
        delete ctx;
        return nullptr;
    }
    ggml_backend_cpu_set_n_threads(ctx->backend, ctx->n_threads);

    // Load GGUF
    if (ctx->verbosity >= 1) {
        std::fprintf(stderr, "[irodori] loading model: %s\n", path_model);
    }

    // Pass 1: metadata
    gguf_context* meta = core_gguf::open_metadata(path_model);
    if (!meta) {
        std::fprintf(stderr, "[irodori] failed to open GGUF metadata: %s\n", path_model);
        ggml_backend_free(ctx->backend);
        delete ctx;
        return nullptr;
    }

    // Read hyperparameters
    auto& hp = ctx->hparams;
    hp.latent_dim = core_gguf::kv_i32(meta, "irodori.latent_dim", hp.latent_dim);
    hp.latent_patch_size = core_gguf::kv_i32(meta, "irodori.latent_patch_size", hp.latent_patch_size);
    hp.model_dim = core_gguf::kv_i32(meta, "irodori.model_dim", hp.model_dim);
    hp.num_layers = core_gguf::kv_i32(meta, "irodori.num_layers", hp.num_layers);
    hp.num_heads = core_gguf::kv_i32(meta, "irodori.num_heads", hp.num_heads);
    hp.mlp_ratio = core_gguf::kv_f32(meta, "irodori.mlp_ratio", hp.mlp_ratio);
    hp.text_dim = core_gguf::kv_i32(meta, "irodori.text_dim", hp.text_dim);
    hp.text_layers = core_gguf::kv_i32(meta, "irodori.text_layers", hp.text_layers);
    hp.text_heads = core_gguf::kv_i32(meta, "irodori.text_heads", hp.text_heads);
    hp.text_mlp_ratio = core_gguf::kv_f32(meta, "irodori.text_mlp_ratio", hp.text_mlp_ratio);
    hp.text_vocab_size = core_gguf::kv_i32(meta, "irodori.text_vocab_size", hp.text_vocab_size);
    hp.speaker_dim = core_gguf::kv_i32(meta, "irodori.speaker_dim", hp.speaker_dim);
    hp.speaker_layers = core_gguf::kv_i32(meta, "irodori.speaker_layers", hp.speaker_layers);
    hp.speaker_heads = core_gguf::kv_i32(meta, "irodori.speaker_heads", hp.speaker_heads);
    hp.speaker_mlp_ratio = core_gguf::kv_f32(meta, "irodori.speaker_mlp_ratio", hp.speaker_mlp_ratio);
    hp.speaker_patch_size = core_gguf::kv_i32(meta, "irodori.speaker_patch_size", hp.speaker_patch_size);
    hp.timestep_embed_dim = core_gguf::kv_i32(meta, "irodori.timestep_embed_dim", hp.timestep_embed_dim);
    hp.adaln_rank = core_gguf::kv_i32(meta, "irodori.adaln_rank", hp.adaln_rank);
    hp.norm_eps = core_gguf::kv_f32(meta, "irodori.norm_eps", hp.norm_eps);
    hp.use_duration_predictor = core_gguf::kv_i32(meta, "irodori.use_duration_predictor", hp.use_duration_predictor);
    hp.duration_aux_dim = core_gguf::kv_i32(meta, "irodori.duration_aux_dim", hp.duration_aux_dim);
    hp.duration_hidden_dim = core_gguf::kv_i32(meta, "irodori.duration_hidden_dim", hp.duration_hidden_dim);
    hp.duration_layers = core_gguf::kv_i32(meta, "irodori.duration_layers", hp.duration_layers);
    hp.ode_steps = core_gguf::kv_i32(meta, "irodori.ode_steps", hp.ode_steps);
    hp.cfg_scale_text = core_gguf::kv_f32(meta, "irodori.cfg_scale_text", hp.cfg_scale_text);
    hp.cfg_scale_speaker = core_gguf::kv_f32(meta, "irodori.cfg_scale_speaker", hp.cfg_scale_speaker);
    hp.sample_rate = core_gguf::kv_i32(meta, "irodori.sample_rate", hp.sample_rate);
    hp.codec_hop_length = core_gguf::kv_i32(meta, "irodori.codec_hop_length", hp.codec_hop_length);

    core_gguf::free_metadata(meta);

    if (ctx->verbosity >= 1) {
        std::fprintf(stderr,
                     "[irodori] model: dim=%d, layers=%d, heads=%d, text_dim=%d, "
                     "text_layers=%d, latent_dim=%d\n",
                     hp.model_dim, hp.num_layers, hp.num_heads, hp.text_dim, hp.text_layers, hp.latent_dim);
    }

    // Pass 2: weights
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path_model, ctx->backend, "irodori_tts", wl)) {
        std::fprintf(stderr, "[irodori] failed to load weights\n");
        ggml_backend_free(ctx->backend);
        delete ctx;
        return nullptr;
    }

    // Load weight tensors
    auto& ts = wl.tensors;
    auto get = [&](const std::string& name) -> ggml_tensor* {
        return core_gguf::require(ts, name.c_str(), "irodori_tts");
    };
    auto try_get = [&](const std::string& name) -> ggml_tensor* { return core_gguf::try_get(ts, name.c_str()); };

    if (!load_weights_from_map(ctx, get, try_get)) {
        std::fprintf(stderr, "[irodori] failed to resolve weight tensors\n");
        ggml_backend_free(ctx->backend);
        delete ctx;
        return nullptr;
    }

    // Keep weight context alive
    ctx->buf_weights = wl.buf;

    // Set default tokenizer BOS
    ctx->bos_token_id = hp.text_vocab_size - 1; // sarashina2.2 convention

    if (ctx->verbosity >= 1) {
        std::fprintf(stderr, "[irodori] model loaded successfully\n");
    }

    return ctx;
}

void irodori_tts_free(struct irodori_tts_context* ctx) {
    if (!ctx)
        return;
    if (ctx->buf_weights)
        ggml_backend_buffer_free(ctx->buf_weights);
    if (ctx->w_ctx)
        ggml_free(ctx->w_ctx);
    if (ctx->backend)
        ggml_backend_free(ctx->backend);
    delete ctx;
}

int irodori_tts_set_reference(struct irodori_tts_context* ctx, const float* ref_pcm, int n_samples, int sample_rate) {
    if (!ctx || !ref_pcm || n_samples <= 0 || sample_rate <= 0)
        return -1;

    // TODO: Encode reference audio through DAC-VAE encoder.
    // For now, this requires pre-encoded latents via irodori_tts_set_reference_latent().
    // The DAC-VAE encoder needs to be ported or called externally.
    std::fprintf(stderr, "[irodori] WARNING: raw PCM reference not yet supported. "
                         "Use irodori_tts_set_reference_latent() with pre-encoded DAC-VAE latents.\n");
    return -1;
}

int irodori_tts_set_reference_latent(struct irodori_tts_context* ctx, const float* latent, int n_frames) {
    if (!ctx || !latent || n_frames <= 0)
        return -1;

    int D = ctx->hparams.latent_dim * ctx->hparams.latent_patch_size;
    ctx->ref_latent.resize(n_frames * D);
    std::memcpy(ctx->ref_latent.data(), latent, n_frames * D * sizeof(float));
    ctx->ref_latent_frames = n_frames;

    IRODORI_DBG("[irodori] set reference latent: %d frames x %d dims\n", n_frames, D);
    return 0;
}

void irodori_tts_clear_reference(struct irodori_tts_context* ctx) {
    if (!ctx)
        return;
    ctx->ref_latent.clear();
    ctx->ref_latent_frames = 0;
}

int irodori_tts_synthesize(struct irodori_tts_context* ctx, const char* text, float** pcm_out, int* sample_rate_out) {
    if (!ctx || !text || !pcm_out || !sample_rate_out)
        return 0;

    const auto& hp = ctx->hparams;
    *sample_rate_out = hp.sample_rate;

    // Tokenize
    std::vector<int32_t> token_ids = tokenize_text(ctx, text);
    int T_text = (int)token_ids.size();
    if (T_text == 0)
        return 0;

    IRODORI_DBG("[irodori] text tokens: %d\n", T_text);

    // Determine output length
    // For now, use a simple heuristic: ~6.3 frames per token (from duration_token_init_frames)
    float frames_per_token = 6.3f;
    int latent_steps = (int)(T_text * frames_per_token * ctx->duration_scale / ctx->speed);
    int max_steps = (int)(ctx->max_seconds * hp.sample_rate / hp.codec_hop_length);
    latent_steps = std::min(latent_steps, max_steps);
    latent_steps = std::max(latent_steps, 1);
    int patched_steps = latent_steps / hp.latent_patch_size;

    IRODORI_DBG("[irodori] output: %d latent steps, %d patched steps (%.2f sec)\n", latent_steps, patched_steps,
                (float)latent_steps * hp.codec_hop_length / hp.sample_rate);

    // The full inference pipeline:
    // 1. Encode text → text_state
    // 2. Encode speaker reference → spk_state (or zeros if no ref)
    // 3. For each ODE step:
    //    a. Compute timestep embedding
    //    b. Run DiT block (with CFG if configured)
    //    c. Euler step: x_t = x_t + v * dt
    // 4. Output latent → external DAC-VAE decoder
    //
    // For the initial implementation, we output the raw latent and
    // note that a DAC-VAE decoder is needed externally.
    // TODO: Full end-to-end synthesis with integrated DAC-VAE decoder.

    // For now, allocate a placeholder output (silence) and log the status.
    // The actual ODE loop + DAC decode will be implemented as we validate
    // the text encoder and speaker encoder graphs first via crispasr-diff.
    int n_pcm = latent_steps * hp.codec_hop_length;
    float* out = (float*)std::malloc(n_pcm * sizeof(float));
    if (!out)
        return 0;
    std::memset(out, 0, n_pcm * sizeof(float));

    if (ctx->verbosity >= 1) {
        std::fprintf(stderr,
                     "[irodori] synthesize: %d tokens → %d latent frames → %d samples "
                     "(%.2f sec @ %d Hz)\n",
                     T_text, latent_steps, n_pcm, (float)n_pcm / hp.sample_rate, hp.sample_rate);
        std::fprintf(stderr, "[irodori] NOTE: full ODE loop + DAC-VAE decode not yet wired. "
                             "Outputting silence placeholder.\n");
    }

    *pcm_out = out;
    return n_pcm;
}

void irodori_tts_set_seed(struct irodori_tts_context* ctx, int seed) {
    if (ctx)
        ctx->seed = seed;
}

void irodori_tts_set_ode_steps(struct irodori_tts_context* ctx, int steps) {
    if (ctx && steps > 0)
        ctx->ode_steps = steps;
}

void irodori_tts_set_cfg_scale_text(struct irodori_tts_context* ctx, float scale) {
    if (ctx)
        ctx->cfg_scale_text = scale;
}

void irodori_tts_set_cfg_scale_speaker(struct irodori_tts_context* ctx, float scale) {
    if (ctx)
        ctx->cfg_scale_speaker = scale;
}

void irodori_tts_set_speed(struct irodori_tts_context* ctx, float speed) {
    if (ctx && speed > 0)
        ctx->speed = speed;
}

int irodori_tts_sample_rate(const struct irodori_tts_context* ctx) {
    return ctx ? ctx->hparams.sample_rate : 48000;
}

int irodori_tts_vocab_size(const struct irodori_tts_context* ctx) {
    return ctx ? ctx->hparams.text_vocab_size : 0;
}

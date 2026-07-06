// crispasr_backend_irodori_tts.cpp — adapter for Aratako/Irodori-TTS.
//
// Flow-matching RF-DiT TTS with zero-shot voice cloning via DAC-VAE latents.
// Japanese-focused, 48 kHz output.
//
// Usage:
//   crispasr --tts irodori-tts --voice ref.wav -t "こんにちは" -o out.wav

#include "crispasr_backend.h"
#include "whisper_params.h"

#include "irodori_tts.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

class IrodoriTTSBackend : public CrispasrBackend {
public:
    IrodoriTTSBackend() = default;
    ~IrodoriTTSBackend() override { IrodoriTTSBackend::shutdown(); }

    const char* name() const override { return "irodori-tts"; }

    uint32_t capabilities() const override { return CAP_TTS | CAP_VOICE_CLONING; }

    std::vector<crispasr_segment> transcribe(const float* /*samples*/, int /*n_samples*/, int64_t /*t_offset_cs*/,
                                             const whisper_params& /*params*/) override {
        return {}; // TTS-only backend
    }

    bool init(const whisper_params& p) override {
        irodori_tts_params fp = irodori_tts_default_params();
        fp.n_threads = p.n_threads;
        fp.verbosity = p.no_prints ? 0 : 1;
        fp.seed = p.seed;

        ctx_ = irodori_tts_init_from_file(p.model.c_str(), fp);
        if (!ctx_) {
            std::fprintf(stderr, "crispasr[irodori-tts]: failed to load model '%s'\n", p.model.c_str());
            return false;
        }

        // Load DAC-VAE codec (companion model for audio decode)
        // Try --codec-model first, then look next to the model, then auto-download
        std::string codec_path = p.tts_codec_model;
        if (codec_path.empty()) {
            // Look for dacvae GGUF next to the model
            std::string model_dir = p.model.substr(0, p.model.find_last_of("/\\"));
            std::string candidate = model_dir + "/dacvae-ja-32dim-f16.gguf";
            FILE* f = std::fopen(candidate.c_str(), "rb");
            if (f) {
                std::fclose(f);
                codec_path = candidate;
            }
        }
        if (!codec_path.empty()) {
            if (irodori_tts_set_codec_path(ctx_, codec_path.c_str()) != 0) {
                std::fprintf(stderr, "crispasr[irodori-tts]: warning: codec load failed, output will be silent\n");
            }
        } else {
            std::fprintf(stderr, "crispasr[irodori-tts]: no DAC-VAE codec found. "
                                 "Use --codec-model <dacvae.gguf> for audio output.\n");
        }

        return true;
    }

    std::vector<float> synthesize(const std::string& text, const whisper_params& /*p*/) override {
        if (!ctx_ || text.empty())
            return {};

        float* pcm = nullptr;
        int sr = 0;
        int n = irodori_tts_synthesize(ctx_, text.c_str(), &pcm, &sr);
        if (n <= 0 || !pcm)
            return {};

        std::vector<float> out(pcm, pcm + n);
        std::free(pcm);
        return out;
    }

    int tts_sample_rate() const override { return ctx_ ? irodori_tts_sample_rate(ctx_) : 48000; }

    void shutdown() override {
        if (ctx_) {
            irodori_tts_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    irodori_tts_context* ctx_ = nullptr;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_irodori_tts_backend() {
    return std::make_unique<IrodoriTTSBackend>();
}

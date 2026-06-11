// crispasr_backend_lfm2_audio.cpp — adapter for LiquidAI LFM2.5-Audio.
//
// End-to-end multimodal ASR: FastConformer encoder → audio adapter →
// LFM2 hybrid conv+attention backbone → greedy text decode.
// Japanese (LFM2.5-Audio-1.5B-JP) and English (LFM2.5-Audio-1.5B).

#include "crispasr_backend.h"
#include "crispasr_backend_utils.h"
#include "whisper_params.h"

#include "lfm2_audio.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

class Lfm2AudioBackend : public CrispasrBackend {
public:
    Lfm2AudioBackend() = default;
    ~Lfm2AudioBackend() override { Lfm2AudioBackend::shutdown(); }

    const char* name() const override { return "lfm2-audio"; }

    uint32_t capabilities() const override { return CAP_AUTO_DOWNLOAD | CAP_UNBOUNDED_INPUT; }

    bool init(const whisper_params& p) override {
        lfm2_audio_context_params lp = lfm2_audio_context_default_params();
        lp.n_threads = p.n_threads;
        lp.verbosity = p.no_prints ? 0 : 1;
        lp.use_gpu = crispasr_backend_should_use_gpu(p);

        ctx_ = lfm2_audio_init_from_file(p.model.c_str(), lp);
        if (!ctx_) {
            fprintf(stderr, "crispasr[lfm2-audio]: failed to load model '%s'\n", p.model.c_str());
            return false;
        }
        return true;
    }

    std::vector<crispasr_segment> transcribe(const float* samples, int n_samples, int64_t t_offset_cs,
                                             const whisper_params& params) override {
        std::vector<crispasr_segment> out;
        if (!ctx_)
            return out;

        char* text = lfm2_audio_transcribe(ctx_, samples, n_samples, nullptr, 0);
        if (!text)
            return out;

        crispasr_segment seg;
        seg.text = text;
        seg.t0 = t_offset_cs;
        seg.t1 = t_offset_cs + (int64_t)((double)n_samples * 100.0 / 16000.0);
        free(text);

        out.push_back(std::move(seg));
        return out;
    }

    void shutdown() override {
        if (ctx_) {
            lfm2_audio_free(ctx_);
            ctx_ = nullptr;
        }
    }

private:
    lfm2_audio_context* ctx_ = nullptr;
};

} // namespace

std::unique_ptr<CrispasrBackend> crispasr_make_lfm2_audio_backend() {
    return std::make_unique<Lfm2AudioBackend>();
}

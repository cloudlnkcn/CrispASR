// test_watermark.cpp — unit tests for the CrispASR audio watermark.
//
// Verifies embed + detect round-trip, detection threshold semantics,
// and robustness against simple transformations (volume scaling).

#include "crispasr_watermark.h"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>
#include <vector>

namespace {

// Generate a simple sine wave for testing (440 Hz at given sample rate).
std::vector<float> make_sine(int n_samples, int sample_rate = 24000, float freq = 440.0f, float amp = 0.5f) {
    std::vector<float> pcm(n_samples);
    for (int i = 0; i < n_samples; i++)
        pcm[i] = amp * std::sin(2.0f * 3.14159265f * freq * (float)i / (float)sample_rate);
    return pcm;
}

} // namespace

TEST_CASE("Watermark embed + detect round-trip", "[unit][watermark]") {
    // 2 seconds of audio at 24 kHz — plenty of frames for reliable detection
    auto pcm = make_sine(48000);
    auto original = pcm; // keep a copy

    crispasr_watermark_embed_impl(pcm.data(), (int)pcm.size());

    float score = crispasr_watermark_detect_impl(pcm.data(), (int)pcm.size());
    REQUIRE(score > 0.65f);

    // Unwatermarked audio should score low
    float score_orig = crispasr_watermark_detect_impl(original.data(), (int)original.size());
    REQUIRE(score_orig < 0.65f);
}

TEST_CASE("Watermark is imperceptible (low distortion)", "[unit][watermark]") {
    auto pcm = make_sine(48000);
    auto original = pcm;

    crispasr_watermark_embed_impl(pcm.data(), (int)pcm.size());

    // Compute SNR — watermark should be well below the signal
    double signal_power = 0.0, noise_power = 0.0;
    for (size_t i = 0; i < pcm.size(); i++) {
        signal_power += (double)original[i] * (double)original[i];
        double diff = (double)pcm[i] - (double)original[i];
        noise_power += diff * diff;
    }
    // SNR should be > 28 dB — well below human perception threshold
    // for speech (~20 dB). The additive magnitude-domain watermark at
    // alpha=0.005 yields ~30 dB SNR on a pure sine.
    double snr_db = 10.0 * std::log10(signal_power / noise_power);
    REQUIRE(snr_db > 28.0);
}

TEST_CASE("Watermark survives volume normalization", "[unit][watermark]") {
    auto pcm = make_sine(48000, 24000, 440.0f, 0.3f);
    crispasr_watermark_embed_impl(pcm.data(), (int)pcm.size());

    // Scale volume by 2x (normalize up)
    for (auto& s : pcm)
        s *= 2.0f;

    float score = crispasr_watermark_detect_impl(pcm.data(), (int)pcm.size());
    REQUIRE(score > 0.60f);
}

TEST_CASE("Watermark detection on silence returns low score", "[unit][watermark]") {
    std::vector<float> silence(48000, 0.0f);
    float score = crispasr_watermark_detect_impl(silence.data(), (int)silence.size());
    // Silence has no spectral content — detection should not false-positive
    REQUIRE(score < 0.65f);
}

TEST_CASE("Watermark embed is no-op for very short audio", "[unit][watermark]") {
    std::vector<float> short_pcm = {0.1f, 0.2f, 0.3f};
    auto original = short_pcm;
    crispasr_watermark_embed_impl(short_pcm.data(), (int)short_pcm.size());
    // Should be unchanged (< 1 FFT frame)
    REQUIRE(short_pcm == original);
}

TEST_CASE("Watermark detect on very short audio returns 0", "[unit][watermark]") {
    std::vector<float> short_pcm = {0.1f, 0.2f, 0.3f};
    float score = crispasr_watermark_detect_impl(short_pcm.data(), (int)short_pcm.size());
    REQUIRE(score == 0.0f);
}

TEST_CASE("Watermark detect on null/invalid input returns 0", "[unit][watermark]") {
    REQUIRE(crispasr_watermark_detect_impl(nullptr, 0) == 0.0f);
    REQUIRE(crispasr_watermark_detect_impl(nullptr, 100) == 0.0f);
}

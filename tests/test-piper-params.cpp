// test-piper-params.cpp — unit tests for piper_tts_params defaults
// and null-guard coverage. No GGUF required.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "piper_tts.h"

TEST_CASE("piper_params: default values are sensible", "[unit][piper]") {
    struct piper_tts_params p = piper_tts_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
    REQUIRE(p.speaker_id >= 0);
    // noise_scale/length_scale/noise_w default to -1 (sentinel = use model default)
    REQUIRE(p.noise_scale != 0.0f);
    REQUIRE(p.length_scale != 0.0f);
}

TEST_CASE("piper_init_from_file: null path returns nullptr", "[unit][piper]") {
    struct piper_tts_params p = piper_tts_default_params();
    struct piper_tts_context* ctx = piper_tts_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("piper_init_from_file: empty path returns nullptr", "[unit][piper]") {
    struct piper_tts_params p = piper_tts_default_params();
    struct piper_tts_context* ctx = piper_tts_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("piper_free: NULL context is a no-op", "[unit][piper]") {
    piper_tts_free(nullptr);
    SUCCEED("piper_tts_free tolerated a NULL ctx.");
}

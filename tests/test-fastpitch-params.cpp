// test-fastpitch-params.cpp — unit tests for fastpitch_tts_params defaults
// and null-guard coverage. No GGUF required.

#include <catch2/catch_test_macros.hpp>
#include "fastpitch_tts.h"

TEST_CASE("fastpitch_params: default values are sensible", "[unit][fastpitch]") {
    struct fastpitch_tts_params p = fastpitch_tts_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
}

TEST_CASE("fastpitch_init_from_file: null path returns nullptr", "[unit][fastpitch]") {
    struct fastpitch_tts_params p = fastpitch_tts_default_params();
    struct fastpitch_tts_context* ctx = fastpitch_tts_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("fastpitch_init_from_file: empty path returns nullptr", "[unit][fastpitch]") {
    struct fastpitch_tts_params p = fastpitch_tts_default_params();
    struct fastpitch_tts_context* ctx = fastpitch_tts_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("fastpitch_free: NULL context is a no-op", "[unit][fastpitch]") {
    fastpitch_tts_free(nullptr);
    SUCCEED("fastpitch_tts_free tolerated a NULL ctx.");
}

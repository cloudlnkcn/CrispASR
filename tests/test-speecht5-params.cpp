// test-speecht5-params.cpp — unit tests for speecht5_tts_params defaults
// and null-guard coverage. No GGUF required.

#include <catch2/catch_test_macros.hpp>
#include "speecht5_tts.h"

TEST_CASE("speecht5_params: default values are sensible", "[unit][speecht5]") {
    struct speecht5_tts_params p = speecht5_tts_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
}

TEST_CASE("speecht5_init: null path returns nullptr", "[unit][speecht5]") {
    struct speecht5_tts_params p = speecht5_tts_default_params();
    struct speecht5_tts_context* ctx = speecht5_tts_init(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("speecht5_init: empty path returns nullptr", "[unit][speecht5]") {
    struct speecht5_tts_params p = speecht5_tts_default_params();
    struct speecht5_tts_context* ctx = speecht5_tts_init("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("speecht5_free: NULL context is a no-op", "[unit][speecht5]") {
    speecht5_tts_free(nullptr);
    SUCCEED("speecht5_tts_free tolerated a NULL ctx.");
}

// test-cosyvoice3-params.cpp — unit tests for cosyvoice3_tts_context_params defaults
// and null-guard coverage. No GGUF required.

#include <catch2/catch_test_macros.hpp>
#include "cosyvoice3_tts.h"

TEST_CASE("cosyvoice3_params: default values are sensible", "[unit][cosyvoice3]") {
    struct cosyvoice3_tts_context_params p = cosyvoice3_tts_context_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
    REQUIRE(p.temperature >= 0.0f);
    REQUIRE(p.max_tokens >= 0);
}

TEST_CASE("cosyvoice3_init_from_file: null path returns nullptr", "[unit][cosyvoice3]") {
    struct cosyvoice3_tts_context_params p = cosyvoice3_tts_context_default_params();
    struct cosyvoice3_tts_context* ctx = cosyvoice3_tts_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("cosyvoice3_init_from_file: empty path returns nullptr", "[unit][cosyvoice3]") {
    struct cosyvoice3_tts_context_params p = cosyvoice3_tts_context_default_params();
    struct cosyvoice3_tts_context* ctx = cosyvoice3_tts_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("cosyvoice3_free: NULL context is a no-op", "[unit][cosyvoice3]") {
    cosyvoice3_tts_free(nullptr);
    SUCCEED("cosyvoice3_tts_free tolerated a NULL ctx.");
}

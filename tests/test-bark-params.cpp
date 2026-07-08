// test-bark-params.cpp — unit tests for bark_context_params defaults
// and null-guard coverage. No GGUF required.

#include <catch2/catch_test_macros.hpp>
#include "bark_tts.h"

TEST_CASE("bark_params: default values are sensible", "[unit][bark]") {
    struct bark_context_params p = bark_context_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
    REQUIRE(p.temperature_semantic >= 0.0f);
    REQUIRE(p.temperature_coarse >= 0.0f);
    REQUIRE(p.temperature_fine >= 0.0f);
}

TEST_CASE("bark_init_from_file: null path returns nullptr", "[unit][bark]") {
    struct bark_context_params p = bark_context_default_params();
    struct bark_context* ctx = bark_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("bark_init_from_file: empty path returns nullptr", "[unit][bark]") {
    struct bark_context_params p = bark_context_default_params();
    struct bark_context* ctx = bark_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("bark_free: NULL context is a no-op", "[unit][bark]") {
    bark_free(nullptr);
    SUCCEED("bark_free tolerated a NULL ctx.");
}

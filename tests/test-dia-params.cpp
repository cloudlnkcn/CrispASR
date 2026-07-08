// test-dia-params.cpp — unit tests for dia_tts_context_params defaults
// and null-guard coverage. No GGUF required.

#include <catch2/catch_test_macros.hpp>
#include "dia_tts.h"

TEST_CASE("dia_params: default values are sensible", "[unit][dia]") {
    struct dia_tts_context_params p = dia_tts_context_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
}

TEST_CASE("dia_init_from_file: null path returns nullptr", "[unit][dia]") {
    struct dia_tts_context_params p = dia_tts_context_default_params();
    struct dia_tts_context* ctx = dia_tts_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("dia_init_from_file: empty path returns nullptr", "[unit][dia]") {
    struct dia_tts_context_params p = dia_tts_context_default_params();
    struct dia_tts_context* ctx = dia_tts_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("dia_free: NULL context is a no-op", "[unit][dia]") {
    dia_tts_free(nullptr);
    SUCCEED("dia_tts_free tolerated a NULL ctx.");
}

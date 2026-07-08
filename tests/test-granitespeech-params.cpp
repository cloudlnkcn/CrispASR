// test-granitespeech-params.cpp — unit tests for granite_speech_context_params defaults
// and null-guard coverage. No GGUF required.

#include <catch2/catch_test_macros.hpp>
#include "granite_speech.h"

TEST_CASE("granite_speech_params: default values are sensible", "[unit][granite_speech]") {
    struct granite_speech_context_params p = granite_speech_context_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
}

TEST_CASE("granite_speech_init_from_file: null path returns nullptr", "[unit][granite_speech]") {
    struct granite_speech_context_params p = granite_speech_context_default_params();
    struct granite_speech_context* ctx = granite_speech_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("granite_speech_init_from_file: empty path returns nullptr", "[unit][granite_speech]") {
    struct granite_speech_context_params p = granite_speech_context_default_params();
    struct granite_speech_context* ctx = granite_speech_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("granite_speech_free: NULL context is a no-op", "[unit][granite_speech]") {
    granite_speech_free(nullptr);
    SUCCEED("granite_speech_free tolerated a NULL ctx.");
}

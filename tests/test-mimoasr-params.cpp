// test-mimoasr-params.cpp — unit tests for mimo_asr_context_params defaults
// and null-guard coverage. No GGUF required.

#include <catch2/catch_test_macros.hpp>
#include "mimo_asr.h"

TEST_CASE("mimo_asr_params: default values are sensible", "[unit][mimo_asr]") {
    struct mimo_asr_context_params p = mimo_asr_context_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
}

TEST_CASE("mimo_asr_init_from_file: null path returns nullptr", "[unit][mimo_asr]") {
    struct mimo_asr_context_params p = mimo_asr_context_default_params();
    struct mimo_asr_context* ctx = mimo_asr_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("mimo_asr_init_from_file: empty path returns nullptr", "[unit][mimo_asr]") {
    struct mimo_asr_context_params p = mimo_asr_context_default_params();
    struct mimo_asr_context* ctx = mimo_asr_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("mimo_asr_free: NULL context is a no-op", "[unit][mimo_asr]") {
    mimo_asr_free(nullptr);
    SUCCEED("mimo_asr_free tolerated a NULL ctx.");
}

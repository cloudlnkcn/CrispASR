// test-gemma4e2b-params.cpp — unit tests for gemma4_e2b_context_params defaults
// and null-guard coverage. No GGUF required.

#include <catch2/catch_test_macros.hpp>
#include "gemma4_e2b.h"

TEST_CASE("gemma4_e2b_params: default values are sensible", "[unit][gemma4_e2b]") {
    struct gemma4_e2b_context_params p = gemma4_e2b_context_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
}

TEST_CASE("gemma4_e2b_init_from_file: null path returns nullptr", "[unit][gemma4_e2b]") {
    struct gemma4_e2b_context_params p = gemma4_e2b_context_default_params();
    struct gemma4_e2b_context* ctx = gemma4_e2b_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("gemma4_e2b_init_from_file: empty path returns nullptr", "[unit][gemma4_e2b]") {
    struct gemma4_e2b_context_params p = gemma4_e2b_context_default_params();
    struct gemma4_e2b_context* ctx = gemma4_e2b_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("gemma4_e2b_free: NULL context is a no-op", "[unit][gemma4_e2b]") {
    gemma4_e2b_free(nullptr);
    SUCCEED("gemma4_e2b_free tolerated a NULL ctx.");
}

// test-parakeet-params.cpp — unit tests for parakeet_context_params defaults
// and null-guard coverage. No GGUF required.

#include <catch2/catch_test_macros.hpp>
#include "parakeet.h"

TEST_CASE("parakeet_params: default values are sensible", "[unit][parakeet]") {
    struct parakeet_context_params p = parakeet_context_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
}

TEST_CASE("parakeet_init_from_file: null path returns nullptr", "[unit][parakeet]") {
    struct parakeet_context_params p = parakeet_context_default_params();
    struct parakeet_context* ctx = parakeet_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("parakeet_init_from_file: empty path returns nullptr", "[unit][parakeet]") {
    struct parakeet_context_params p = parakeet_context_default_params();
    struct parakeet_context* ctx = parakeet_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("parakeet_free: NULL context is a no-op", "[unit][parakeet]") {
    parakeet_free(nullptr);
    SUCCEED("parakeet_free tolerated a NULL ctx.");
}

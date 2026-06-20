// test-melotts-params.cpp — unit tests for melotts_params defaults
// and null-guard coverage. No GGUF required.

#include <catch2/catch_test_macros.hpp>
#include "melotts.h"

TEST_CASE("melotts_params: default values are sensible", "[unit][melotts]") {
    struct melotts_params p = melotts_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
}

TEST_CASE("melotts_init_from_file: null path returns nullptr", "[unit][melotts]") {
    struct melotts_params p = melotts_default_params();
    struct melotts_context* ctx = melotts_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("melotts_init_from_file: empty path returns nullptr", "[unit][melotts]") {
    struct melotts_params p = melotts_default_params();
    struct melotts_context* ctx = melotts_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("melotts_free: NULL context is a no-op", "[unit][melotts]") {
    melotts_free(nullptr);
    SUCCEED("melotts_free tolerated a NULL ctx.");
}

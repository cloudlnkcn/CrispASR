// test-indextts-params.cpp — unit tests for indextts_context_params defaults
// and null-guard coverage. No GGUF required.

#include <catch2/catch_test_macros.hpp>
#include "indextts.h"

TEST_CASE("indextts_params: default values are sensible", "[unit][indextts]") {
    struct indextts_context_params p = indextts_context_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
}

TEST_CASE("indextts_init_from_file: null path returns nullptr", "[unit][indextts]") {
    struct indextts_context_params p = indextts_context_default_params();
    struct indextts_context* ctx = indextts_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("indextts_init_from_file: empty path returns nullptr", "[unit][indextts]") {
    struct indextts_context_params p = indextts_context_default_params();
    struct indextts_context* ctx = indextts_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("indextts_free: NULL context is a no-op", "[unit][indextts]") {
    indextts_free(nullptr);
    SUCCEED("indextts_free tolerated a NULL ctx.");
}

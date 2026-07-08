// test-voxtral4b-params.cpp — unit tests for voxtral4b_context_params defaults
// and null-guard coverage. No GGUF required.

#include <catch2/catch_test_macros.hpp>
#include "voxtral4b.h"

TEST_CASE("voxtral4b_params: default values are sensible", "[unit][voxtral4b]") {
    struct voxtral4b_context_params p = voxtral4b_context_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
}

TEST_CASE("voxtral4b_init_from_file: null path returns nullptr", "[unit][voxtral4b]") {
    struct voxtral4b_context_params p = voxtral4b_context_default_params();
    struct voxtral4b_context* ctx = voxtral4b_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("voxtral4b_init_from_file: empty path returns nullptr", "[unit][voxtral4b]") {
    struct voxtral4b_context_params p = voxtral4b_context_default_params();
    struct voxtral4b_context* ctx = voxtral4b_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("voxtral4b_free: NULL context is a no-op", "[unit][voxtral4b]") {
    voxtral4b_free(nullptr);
    SUCCEED("voxtral4b_free tolerated a NULL ctx.");
}

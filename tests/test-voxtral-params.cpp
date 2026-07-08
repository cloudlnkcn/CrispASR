// test-voxtral-params.cpp — unit tests for voxtral_context_params defaults
// and null-guard coverage. No GGUF required.

#include <catch2/catch_test_macros.hpp>
#include "voxtral.h"

TEST_CASE("voxtral_params: default values are sensible", "[unit][voxtral]") {
    struct voxtral_context_params p = voxtral_context_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
}

TEST_CASE("voxtral_init_from_file: null path returns nullptr", "[unit][voxtral]") {
    struct voxtral_context_params p = voxtral_context_default_params();
    struct voxtral_context* ctx = voxtral_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("voxtral_init_from_file: empty path returns nullptr", "[unit][voxtral]") {
    struct voxtral_context_params p = voxtral_context_default_params();
    struct voxtral_context* ctx = voxtral_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("voxtral_free: NULL context is a no-op", "[unit][voxtral]") {
    voxtral_free(nullptr);
    SUCCEED("voxtral_free tolerated a NULL ctx.");
}

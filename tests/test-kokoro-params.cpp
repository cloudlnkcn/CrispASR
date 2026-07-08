// test-kokoro-params.cpp — unit tests for kokoro_context_params defaults
// and null-guard coverage. No GGUF required.

#include <catch2/catch_test_macros.hpp>
#include "kokoro.h"

TEST_CASE("kokoro_params: default values are sensible", "[unit][kokoro]") {
    struct kokoro_context_params p = kokoro_context_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
}

TEST_CASE("kokoro_init_from_file: null path returns nullptr", "[unit][kokoro]") {
    struct kokoro_context_params p = kokoro_context_default_params();
    struct kokoro_context* ctx = kokoro_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("kokoro_init_from_file: empty path returns nullptr", "[unit][kokoro]") {
    struct kokoro_context_params p = kokoro_context_default_params();
    struct kokoro_context* ctx = kokoro_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("kokoro_free: NULL context is a no-op", "[unit][kokoro]") {
    kokoro_free(nullptr);
    SUCCEED("kokoro_free tolerated a NULL ctx.");
}

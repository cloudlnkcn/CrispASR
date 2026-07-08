// test-csm-params.cpp — unit tests for csm_tts_context_params defaults
// and null-guard coverage. No GGUF required.

#include <catch2/catch_test_macros.hpp>
#include "csm_tts.h"

TEST_CASE("csm_params: default values are sensible", "[unit][csm]") {
    struct csm_tts_context_params p = csm_tts_context_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
}

TEST_CASE("csm_init_from_file: null path returns nullptr", "[unit][csm]") {
    struct csm_tts_context_params p = csm_tts_context_default_params();
    struct csm_tts_context* ctx = csm_tts_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("csm_init_from_file: empty path returns nullptr", "[unit][csm]") {
    struct csm_tts_context_params p = csm_tts_context_default_params();
    struct csm_tts_context* ctx = csm_tts_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("csm_free: NULL context is a no-op", "[unit][csm]") {
    csm_tts_free(nullptr);
    SUCCEED("csm_tts_free tolerated a NULL ctx.");
}

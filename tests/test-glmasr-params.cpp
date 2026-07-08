// test-glmasr-params.cpp — unit tests for glm_asr_context_params defaults
// and null-guard coverage. No GGUF required.

#include <catch2/catch_test_macros.hpp>
#include "glm_asr.h"

TEST_CASE("glm_asr_params: default values are sensible", "[unit][glm_asr]") {
    struct glm_asr_context_params p = glm_asr_context_default_params();

    REQUIRE(p.n_threads >= 1);
    REQUIRE(p.verbosity >= 0);
}

TEST_CASE("glm_asr_init_from_file: null path returns nullptr", "[unit][glm_asr]") {
    struct glm_asr_context_params p = glm_asr_context_default_params();
    struct glm_asr_context* ctx = glm_asr_init_from_file(nullptr, p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("glm_asr_init_from_file: empty path returns nullptr", "[unit][glm_asr]") {
    struct glm_asr_context_params p = glm_asr_context_default_params();
    struct glm_asr_context* ctx = glm_asr_init_from_file("", p);
    REQUIRE(ctx == nullptr);
}

TEST_CASE("glm_asr_free: NULL context is a no-op", "[unit][glm_asr]") {
    glm_asr_free(nullptr);
    SUCCEED("glm_asr_free tolerated a NULL ctx.");
}

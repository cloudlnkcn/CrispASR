// test-issue-114-chunk-context-gate.cpp — unit tests for the overlap-save
// gating logic (issue #114).
//
// The cad4c28a "feat(#89): overlap-save chunking with --chunk-overlap flag"
// change extended each transcribe() slice by ±chunk_overlap_seconds on
// each side whenever slices.size() > 1. That gate was correct for
// explicit `--chunk-seconds N` runs (where the chunks are arbitrary cuts
// through continuous speech and the encoder genuinely needs left/right
// context to span boundaries), but WRONG for VAD-derived multi-slice
// runs: VAD slices are separated by silence, so there is no boundary
// signal to recover. Adding 3 s of neighbour audio pulled the next
// utterance into the current encoder's context window, shifted the
// FastConformer features, and caused the TDT decoder to pick a worse
// token path. The user-visible symptom on parakeet-tdt-0.6b-ja was kanji
// collapsing to bare hiragana plus entire short slices being dropped
// (issue #114).
//
// The fix gates use_chunk_context on `effective_chunk_seconds > 0` so
// the extension is applied only when chunking is the actual reason for
// having multiple slices. These tests pin that invariant.

#include <catch2/catch_test_macros.hpp>

#include "crispasr_chunk_context_gate.h"

using crispasr_chunk_context::should_use_chunk_context;

TEST_CASE("issue #114: VAD-derived multi-slice run does NOT extend with context", "[unit][chunk-context][issue-114]") {
    // CAP_UNBOUNDED_INPUT backends (parakeet, canary, ...) set
    // effective_chunk_seconds = 0 unconditionally in crispasr_run.cpp.
    // With 56 VAD slices and the default chunk_overlap_seconds = 3.0,
    // the pre-fix gate evaluated to true (slices.size() > 1 && 3.0 > 0)
    // and produced the kanji → hiragana regression.
    constexpr int effective_chunk_seconds = 0;
    constexpr std::size_t n_slices = 56;
    constexpr float chunk_overlap_seconds = 3.0f;
    REQUIRE_FALSE(should_use_chunk_context(effective_chunk_seconds, n_slices, chunk_overlap_seconds));
}

TEST_CASE("explicit --chunk-seconds with multiple slices uses overlap-save", "[unit][chunk-context][issue-114]") {
    // The original intended use case of cad4c28a: the user (or the
    // default fallback for non-CAP_UNBOUNDED backends) requested
    // chunk_seconds=30 to cut long audio into 30 s slices. Here the
    // overlap is genuinely useful — adjacent slices are continuous
    // speech, so the encoder needs the ±3 s context to span the cut.
    constexpr int effective_chunk_seconds = 30;
    constexpr std::size_t n_slices = 10;
    constexpr float chunk_overlap_seconds = 3.0f;
    REQUIRE(should_use_chunk_context(effective_chunk_seconds, n_slices, chunk_overlap_seconds));
}

TEST_CASE("single slice never gets context (no boundary to mitigate)", "[unit][chunk-context][issue-114]") {
    // Even with explicit chunking + positive overlap, one slice means
    // there is no chunk boundary at all. Adding context would just be
    // expanding the slice to cover audio that isn't there.
    REQUIRE_FALSE(should_use_chunk_context(30, 1, 3.0f));
    REQUIRE_FALSE(should_use_chunk_context(0, 1, 3.0f));
}

TEST_CASE("--chunk-overlap 0 disables overlap-save", "[unit][chunk-context][issue-114]") {
    // The CLI gives the user an explicit knob to turn off the context
    // extension even when chunking is active. Useful as an escape hatch
    // if overlap-save itself proves problematic on a particular model.
    REQUIRE_FALSE(should_use_chunk_context(30, 10, 0.0f));
    // Negative values must also disable (defensive — should never happen).
    REQUIRE_FALSE(should_use_chunk_context(30, 10, -1.0f));
}

TEST_CASE("gate is purely a function of its three inputs", "[unit][chunk-context][issue-114]") {
    // The gate must remain stateless / referentially transparent so the
    // unit test stays trustworthy. Same inputs → same output, every time.
    for (int chunk_s : {0, 1, 5, 30, 120}) {
        for (std::size_t n : {std::size_t{0}, std::size_t{1}, std::size_t{2}, std::size_t{56}}) {
            for (float overlap : {-1.0f, 0.0f, 0.1f, 3.0f, 10.0f}) {
                const bool a = should_use_chunk_context(chunk_s, n, overlap);
                const bool b = should_use_chunk_context(chunk_s, n, overlap);
                REQUIRE(a == b);
                // Cross-check against the closed-form gate.
                const bool expected = (chunk_s > 0) && (n > 1) && (overlap > 0.0f);
                REQUIRE(a == expected);
            }
        }
    }
}

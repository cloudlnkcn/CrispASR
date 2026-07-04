// tests/test_align_only.cpp — unit + live tests for standalone alignment (issue #217).
//
// Unit tests: SRT parsing, tokenise_words, empty/null safety.
// Live tests: actual CTC alignment with canary-ctc-aligner or wav2vec2.
//
// Env vars:
//   CRISPASR_MODEL_ALIGNER       — CTC aligner GGUF (canary-ctc-aligner-q4_k, wav2vec2, etc.)
//   CRISPASR_TEST_AUDIO_JFK      — path to jfk.wav (defaults to samples/jfk.wav)

#include <catch2/catch_test_macros.hpp>

#include "crispasr_aligner.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Unit tests — no model needed
// ---------------------------------------------------------------------------

TEST_CASE("align-only: crispasr_align_words null/empty safety", "[unit][align]") {
    // All null/empty inputs must return an empty vector, not crash.
    auto r1 = crispasr_align_words("", "hello world", nullptr, 0, 0, 1);
    CHECK(r1.empty());

    auto r2 = crispasr_align_words("/nonexistent.gguf", "", nullptr, 0, 0, 1);
    CHECK(r2.empty());

    auto r3 = crispasr_align_words("/nonexistent.gguf", "hello", nullptr, 0, 0, 1);
    CHECK(r3.empty());

    float dummy = 0.0f;
    auto r4 = crispasr_align_words("/nonexistent.gguf", "hello", &dummy, 1, 0, 1);
    CHECK(r4.empty());
}

TEST_CASE("align-only: crispasr_aligner_free_cache does not crash when empty", "[unit][align]") {
    crispasr_aligner_free_cache(); // must not crash
    crispasr_aligner_free_cache(); // double-free safety
}

// SRT text extraction helper — matches the logic in crispasr_run.cpp --align-only.
// Re-implemented here so we can unit test the parsing independently.
static std::string extract_srt_text(const std::string& raw) {
    std::string text_only;
    enum { S_INDEX, S_TIME, S_TEXT } state = S_INDEX;
    size_t i = 0;
    while (i < raw.size()) {
        size_t nl = raw.find('\n', i);
        if (nl == std::string::npos)
            nl = raw.size();
        std::string line = raw.substr(i, nl - i);
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        i = nl + 1;

        if (state == S_INDEX) {
            if (line.empty())
                continue;
            state = S_TIME;
        } else if (state == S_TIME) {
            state = S_TEXT;
        } else {
            if (line.empty()) {
                state = S_INDEX;
            } else {
                if (!text_only.empty())
                    text_only += ' ';
                text_only += line;
            }
        }
    }
    return text_only;
}

TEST_CASE("align-only: SRT text extraction", "[unit][align]") {
    SECTION("standard SRT") {
        const char* srt = "1\r\n"
                          "00:00:00,000 --> 00:00:02,500\r\n"
                          "Hello world\r\n"
                          "\r\n"
                          "2\r\n"
                          "00:00:02,500 --> 00:00:05,000\r\n"
                          "This is a test\r\n"
                          "\r\n";
        std::string text = extract_srt_text(srt);
        CHECK(text == "Hello world This is a test");
    }

    SECTION("SRT with multi-line subtitle text") {
        const char* srt = "1\n"
                          "00:00:00,000 --> 00:00:03,000\n"
                          "Line one\n"
                          "Line two\n"
                          "\n"
                          "2\n"
                          "00:00:03,000 --> 00:00:05,000\n"
                          "Single line\n"
                          "\n";
        std::string text = extract_srt_text(srt);
        CHECK(text == "Line one Line two Single line");
    }

    SECTION("empty SRT") {
        CHECK(extract_srt_text("").empty());
    }

    SECTION("SRT with no trailing blank line") {
        const char* srt = "1\n"
                          "00:00:00,000 --> 00:00:01,000\n"
                          "No trailing blank";
        std::string text = extract_srt_text(srt);
        CHECK(text == "No trailing blank");
    }
}

// ---------------------------------------------------------------------------
// Live tests — need CRISPASR_MODEL_ALIGNER + audio
// ---------------------------------------------------------------------------

static std::string get_env(const char* name, const char* fallback = "") {
    const char* v = std::getenv(name);
    return v ? v : fallback;
}

static std::vector<float> load_wav_16k_mono(const std::string& path) {
    std::vector<float> pcm;
    FILE* f = fopen(path.c_str(), "rb");
    if (!f)
        return pcm;

    char riff[12];
    if (fread(riff, 1, 12, f) != 12 || memcmp(riff, "RIFF", 4) != 0 || memcmp(riff + 8, "WAVE", 4) != 0) {
        fclose(f);
        return pcm;
    }

    int channels = 0, sample_rate = 0, bits = 0;
    int32_t data_size = 0;
    bool found_fmt = false, found_data = false;

    while (!found_data) {
        char chunk_id[4];
        int32_t chunk_size;
        if (fread(chunk_id, 1, 4, f) != 4)
            break;
        if (fread(&chunk_size, 4, 1, f) != 1)
            break;
        if (memcmp(chunk_id, "fmt ", 4) == 0) {
            char fmt[16];
            if (chunk_size < 16 || fread(fmt, 1, 16, f) != 16)
                break;
            channels = *(int16_t*)(fmt + 2);
            sample_rate = *(int32_t*)(fmt + 4);
            bits = *(int16_t*)(fmt + 14);
            found_fmt = true;
            if (chunk_size > 16)
                fseek(f, chunk_size - 16, SEEK_CUR);
        } else if (memcmp(chunk_id, "data", 4) == 0) {
            data_size = chunk_size;
            found_data = true;
        } else {
            fseek(f, chunk_size, SEEK_CUR);
        }
    }

    if (!found_fmt || !found_data || bits != 16 || channels < 1) {
        fclose(f);
        return pcm;
    }

    int n_samples = data_size / (channels * (bits / 8));
    std::vector<int16_t> raw(n_samples * channels);
    if ((int)fread(raw.data(), sizeof(int16_t), n_samples * channels, f) != n_samples * channels) {
        fclose(f);
        return pcm;
    }
    fclose(f);

    pcm.resize(n_samples);
    for (int i = 0; i < n_samples; i++) {
        float sum = 0;
        for (int c = 0; c < channels; c++)
            sum += raw[i * channels + c];
        pcm[i] = sum / (channels * 32768.0f);
    }
    return pcm;
}

TEST_CASE("align-only: live alignment with canary-ctc-aligner", "[live][align]") {
    std::string model = get_env("CRISPASR_MODEL_ALIGNER");
    if (model.empty()) {
        // Fallback: try well-known paths.
        const char* dir = std::getenv("CRISPASR_MODELS_DIR");
        if (dir) {
            model = std::string(dir) + "/canary-ctc-aligner-q4_k.gguf";
            FILE* f = fopen(model.c_str(), "rb");
            if (f)
                fclose(f);
            else
                model.clear();
        }
    }
    if (model.empty())
        SKIP("CRISPASR_MODEL_ALIGNER not set and no fallback found");

    std::string audio_path = get_env("CRISPASR_TEST_AUDIO_JFK", "samples/jfk.wav");
    auto pcm = load_wav_16k_mono(audio_path);
    if (pcm.empty())
        SKIP("Cannot load audio: " + audio_path);

    REQUIRE(pcm.size() > 16000); // at least 1 second

    const std::string transcript = "And so my fellow Americans ask not what your country can do for you "
                                   "ask what you can do for your country";

    auto aligned = crispasr_align_words(model, transcript, pcm.data(), (int)pcm.size(), 0, 4);

    REQUIRE(!aligned.empty());

    // Verify basic properties:
    // 1. Number of words matches transcript word count.
    int word_count = 1;
    for (char c : transcript)
        if (c == ' ')
            word_count++;
    CHECK(aligned.size() == (size_t)word_count);

    // 2. Timestamps are monotonically non-decreasing.
    for (size_t i = 1; i < aligned.size(); i++) {
        CHECK(aligned[i].t0_cs >= aligned[i - 1].t0_cs);
    }

    // 3. All timestamps are within the audio duration.
    int64_t max_cs = (int64_t)(pcm.size() * 100 / 16000) + 10; // +10cs tolerance
    for (auto& w : aligned) {
        CHECK(w.t0_cs >= 0);
        CHECK(w.t0_cs <= max_cs);
        CHECK(w.t1_cs >= 0);
        CHECK(w.t1_cs <= max_cs);
    }

    // 4. First word starts near the beginning (within first 3 seconds).
    CHECK(aligned[0].t0_cs < 300);

    // 5. Words have non-empty text.
    for (auto& w : aligned) {
        CHECK(!w.text.empty());
    }

    crispasr_aligner_free_cache();
}

TEST_CASE("align-only: live alignment from SRT text input", "[live][align]") {
    std::string model = get_env("CRISPASR_MODEL_ALIGNER");
    if (model.empty()) {
        const char* dir = std::getenv("CRISPASR_MODELS_DIR");
        if (dir) {
            model = std::string(dir) + "/canary-ctc-aligner-q4_k.gguf";
            FILE* f = fopen(model.c_str(), "rb");
            if (f)
                fclose(f);
            else
                model.clear();
        }
    }
    if (model.empty())
        SKIP("CRISPASR_MODEL_ALIGNER not set and no fallback found");

    std::string audio_path = get_env("CRISPASR_TEST_AUDIO_JFK", "samples/jfk.wav");
    auto pcm = load_wav_16k_mono(audio_path);
    if (pcm.empty())
        SKIP("Cannot load audio: " + audio_path);

    // Simulate text extracted from an unaligned SRT.
    const char* srt_content = "1\n"
                              "00:00:00,000 --> 00:00:30,000\n"
                              "And so my fellow Americans ask not what your country can do for you "
                              "ask what you can do for your country\n"
                              "\n";
    std::string transcript = extract_srt_text(srt_content);
    REQUIRE(!transcript.empty());

    auto aligned = crispasr_align_words(model, transcript, pcm.data(), (int)pcm.size(), 0, 4);

    REQUIRE(!aligned.empty());
    // The alignment from SRT text should produce the same result as from plain text.
    CHECK(aligned.size() > 10); // at least 10 words aligned

    // Timestamps should span most of the audio duration.
    int64_t last_t1 = aligned.back().t1_cs;
    int64_t audio_dur_cs = (int64_t)(pcm.size() * 100 / 16000);
    // Last word should end within the last 3 seconds of the audio.
    CHECK(last_t1 > audio_dur_cs - 300);

    crispasr_aligner_free_cache();
}

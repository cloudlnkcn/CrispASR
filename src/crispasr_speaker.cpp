// crispasr_speaker.cpp — local speaker playback via miniaudio.
//
// Thin wrapper around miniaudio's ma_device playback mode. The
// MINIAUDIO_IMPLEMENTATION lives in crispasr_audio.cpp, so this TU
// just consumes the API. Mirrors the structure of crispasr_mic.cpp.

#include "crispasr_speaker.h"

// Same iOS/tvOS/watchOS guard as crispasr_mic.cpp: device IO isn't
// available on these platforms (MA_NO_DEVICE_IO is set), so stub out.
#if defined(__APPLE__)
#include <TargetConditionals.h>
#if TARGET_OS_IPHONE
#define CRISPASR_SPEAKER_STUB_ONLY 1
#endif
#endif

#ifndef CRISPASR_SPEAKER_STUB_ONLY

#include "miniaudio.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <new>

struct crispasr_speaker {
    ma_device device;
    const float* pcm = nullptr;
    int n_samples = 0;
    std::atomic<int> cursor{0};
    std::atomic<bool> stopped{false};
    ma_event done_event;
    bool event_init = false;
    bool started = false;
};

namespace {
void speaker_data_cb(ma_device* dev, void* output, const void* /*input*/, ma_uint32 frames) {
    auto* s = (crispasr_speaker*)dev->pUserData;
    auto* out = (float*)output;
    const int cur = s->cursor.load(std::memory_order_relaxed);
    const int rem = s->n_samples - cur;
    const int to_copy =
        (rem > 0 && !s->stopped.load(std::memory_order_relaxed)) ? (int)std::min((ma_uint32)rem, frames) : 0;
    if (to_copy > 0) {
        std::memcpy(out, s->pcm + cur, (size_t)to_copy * sizeof(float));
        s->cursor.fetch_add(to_copy, std::memory_order_relaxed);
    }
    // Zero-fill any remainder (device expects a full buffer each callback)
    if ((ma_uint32)to_copy < frames)
        std::memset(out + to_copy, 0, ((size_t)frames - (size_t)to_copy) * sizeof(float));
    // Signal completion once the buffer is exhausted or stop was requested
    if (to_copy < (int)frames || s->stopped.load(std::memory_order_relaxed)) {
        s->stopped.store(true, std::memory_order_relaxed);
        if (s->event_init)
            ma_event_signal(&s->done_event);
    }
}
} // namespace

extern "C" struct crispasr_speaker* crispasr_speaker_open(int sample_rate, int channels, int device_index) {
    if (sample_rate <= 0 || channels < 1 || channels > 2)
        return nullptr;

    auto* s = new (std::nothrow) crispasr_speaker();
    if (!s)
        return nullptr;

    if (ma_event_init(&s->done_event) != MA_SUCCESS) {
        delete s;
        return nullptr;
    }
    s->event_init = true;

    ma_device_config cfg = ma_device_config_init(ma_device_type_playback);
    cfg.playback.format = ma_format_f32;
    cfg.playback.channels = (ma_uint32)channels;
    cfg.sampleRate = (ma_uint32)sample_rate;
    cfg.dataCallback = speaker_data_cb;
    cfg.pUserData = s;

    // device_index -1 → nullptr (default device); otherwise use index
    // (miniaudio requires enumeration to get the ma_device_id — for now
    // we only support the default; numeric selection is a later addition)
    (void)device_index;

    if (ma_device_init(nullptr, &cfg, &s->device) != MA_SUCCESS) {
        ma_event_uninit(&s->done_event);
        delete s;
        return nullptr;
    }
    return s;
}

extern "C" int crispasr_speaker_play(struct crispasr_speaker* s, const float* pcm, int n_samples) {
    if (!s || !pcm || n_samples <= 0)
        return -1;
    if (s->started)
        return -2; // already playing; call stop() first

    s->pcm = pcm;
    s->n_samples = n_samples;
    s->cursor.store(0, std::memory_order_relaxed);
    s->stopped.store(false, std::memory_order_relaxed);
    // Reset the event so wait() will block properly
    // ma_event is auto-reset (manual reset not needed with ma_event_signal + ma_event_wait)

    if (ma_device_start(&s->device) != MA_SUCCESS)
        return -3;
    s->started = true;
    return 0;
}

extern "C" int crispasr_speaker_wait(struct crispasr_speaker* s) {
    if (!s || !s->started)
        return -1;
    ma_event_wait(&s->done_event);
    ma_device_stop(&s->device);
    s->started = false;
    return s->stopped.load() ? 0 : -1;
}

extern "C" int crispasr_speaker_stop(struct crispasr_speaker* s) {
    if (!s)
        return -1;
    s->stopped.store(true, std::memory_order_relaxed);
    if (s->event_init)
        ma_event_signal(&s->done_event); // unblock wait()
    if (s->started) {
        ma_device_stop(&s->device);
        s->started = false;
    }
    return 0;
}

extern "C" void crispasr_speaker_close(struct crispasr_speaker* s) {
    if (!s)
        return;
    if (s->started) {
        s->stopped.store(true, std::memory_order_relaxed);
        ma_device_stop(&s->device);
        s->started = false;
    }
    ma_device_uninit(&s->device);
    if (s->event_init)
        ma_event_uninit(&s->done_event);
    delete s;
}

extern "C" const char* crispasr_speaker_default_device_name(void) {
    static char name[MA_MAX_DEVICE_NAME_LENGTH + 1] = {0};
    ma_context ctx;
    if (ma_context_init(nullptr, 0, nullptr, &ctx) != MA_SUCCESS) {
        name[0] = '\0';
        return name;
    }
    ma_device_info* playback_infos = nullptr;
    ma_uint32 playback_count = 0;
    ma_device_info* capture_infos = nullptr;
    ma_uint32 capture_count = 0;
    if (ma_context_get_devices(&ctx, &playback_infos, &playback_count, &capture_infos, &capture_count) != MA_SUCCESS) {
        ma_context_uninit(&ctx);
        name[0] = '\0';
        return name;
    }
    name[0] = '\0';
    for (ma_uint32 i = 0; i < playback_count; i++) {
        if (playback_infos[i].isDefault) {
            std::strncpy(name, playback_infos[i].name, sizeof(name) - 1);
            break;
        }
    }
    if (name[0] == '\0' && playback_count > 0)
        std::strncpy(name, playback_infos[0].name, sizeof(name) - 1);
    ma_context_uninit(&ctx);
    return name;
}

#else // CRISPASR_SPEAKER_STUB_ONLY — iOS / tvOS / watchOS

struct crispasr_speaker {};

extern "C" struct crispasr_speaker* crispasr_speaker_open(int, int, int) {
    return nullptr;
}
extern "C" int crispasr_speaker_play(struct crispasr_speaker*, const float*, int) {
    return -1;
}
extern "C" int crispasr_speaker_wait(struct crispasr_speaker*) {
    return -1;
}
extern "C" int crispasr_speaker_stop(struct crispasr_speaker*) {
    return -1;
}
extern "C" void crispasr_speaker_close(struct crispasr_speaker*) {}
extern "C" const char* crispasr_speaker_default_device_name(void) {
    return "";
}

#endif // CRISPASR_SPEAKER_STUB_ONLY

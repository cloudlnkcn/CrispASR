// crispasr_speaker.h — local speaker playback via miniaudio.
//
// Cross-platform audio output using miniaudio's ma_device playback mode
// (Core Audio on macOS, ALSA/PulseAudio on Linux, WASAPI on Windows).
// The runtime symbols are already linked into libcrispasr because
// crispasr_audio.cpp defines MINIAUDIO_IMPLEMENTATION.
//
// Intended for CLI TTS (--tts-play) and future voice-loop use. Server
// mode routes audio to HTTP clients, not to local speakers.

#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct crispasr_speaker;

// Open a playback device without starting it.
//
//   sample_rate:   target output rate in Hz (24000 for most TTS backends)
//   channels:      1 (mono) or 2 (stereo). Pass 1 for TTS output.
//   device_index:  -1 = default device; 0-N = enumerate via
//                  crispasr_speaker_list_devices() (future).
//
// Returns nullptr on failure (no output device available, bad args).
// Call crispasr_speaker_close() when done regardless of whether play was called.
struct crispasr_speaker* crispasr_speaker_open(int sample_rate, int channels, int device_index);

// Begin playing `pcm` (float32, `n_samples` mono frames at the sample rate
// passed to open). Returns immediately — playback runs on the audio thread.
// The caller MUST keep `pcm` valid until crispasr_speaker_wait() or
// crispasr_speaker_stop() returns.
// Returns 0 on success, non-zero on driver error.
int crispasr_speaker_play(struct crispasr_speaker* s, const float* pcm, int n_samples);

// Block until playback of the current buffer is complete. Returns 0 when
// playback finished naturally, -1 if interrupted by crispasr_speaker_stop().
int crispasr_speaker_wait(struct crispasr_speaker* s);

// Interrupt in-flight playback and unblock any waiting crispasr_speaker_wait()
// call. Idempotent. The audio thread may emit a few more frames while draining.
int crispasr_speaker_stop(struct crispasr_speaker* s);

// Release the device and free the handle. Implies stop(). Always call this.
void crispasr_speaker_close(struct crispasr_speaker* s);

// Human-readable name of the default playback device, or empty string if no
// output device is available. Returns a static buffer; not thread-safe.
const char* crispasr_speaker_default_device_name(void);

#ifdef __cplusplus
}
#endif

// mGBA for the browser (gba_web.wasm): the real game, run frame by frame from JavaScript, with the picture (RGBA),
// the sound (stereo int16 at 48 kHz) and direct memory access for the AI. Same libmgba (0.10.2) and core
// settings as the Python host (emu/host/emulator.cpp). The ROM and save come from memory (the pack).

#include <mgba/core/blip_buf.h>
#include <mgba/core/core.h>
#include <mgba/core/config.h>
#include <mgba/core/log.h>
#include <mgba-util/vfs.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#define API extern "C" EMSCRIPTEN_KEEPALIVE
#else
#define API extern "C"
#endif

namespace {

void silentLog(struct mLogger*, int, enum mLogLevel, const char*, va_list) {}
struct mLogger gSilentLogger = {silentLog, nullptr};

constexpr int SAMPLE_RATE = 48000;
constexpr int AUDIO_CAP = 8192;      // stereo frames buffered between reads

struct Emu {
    struct mCore* core = nullptr;
    std::vector<color_t> video;
    std::vector<uint8_t> rgba;
    std::vector<uint8_t> rom, save;
    std::vector<int16_t> audio;      // interleaved stereo
    std::vector<uint8_t> state;
    unsigned width = 240, height = 160;
};

Emu* g = nullptr;

}  // namespace

// Load a ROM (and optionally a save) from memory. Returns 0 on success.
API int gba_init(const uint8_t* rom, int romSize, const uint8_t* save, int saveSize) {
    mLogSetDefaultLogger(&gSilentLogger);
    if (g) return -1;
    g = new Emu();
    g->rom.assign(rom, rom + romSize);
    struct VFile* vrom = VFileFromConstMemory(g->rom.data(), g->rom.size());
    g->core = mCoreFindVF(vrom);
    if (!g->core || !g->core->init(g->core)) return -2;
    mCoreInitConfig(g->core, nullptr);
    unsigned w, h;
    g->core->desiredVideoDimensions(g->core, &w, &h);
    g->width = w;
    g->height = h;
    g->video.resize(size_t(w) * h);
    g->rgba.resize(size_t(w) * h * 4);
    g->core->setVideoBuffer(g->core, g->video.data(), w);
    g->core->setAudioBufferSize(g->core, 2048);
    if (!g->core->loadROM(g->core, vrom)) return -3;
    if (save && saveSize > 0) {
        g->save.assign(save, save + saveSize);
        struct VFile* vsave = VFileMemChunk(g->save.data(), g->save.size());   // writes stay in memory
        if (!g->core->loadSave(g->core, vsave)) return -4;
    }
    g->core->reset(g->core);
    blip_set_rates(g->core->getAudioChannel(g->core, 0), g->core->frequency(g->core), SAMPLE_RATE);
    blip_set_rates(g->core->getAudioChannel(g->core, 1), g->core->frequency(g->core), SAMPLE_RATE);
    g->audio.resize(AUDIO_CAP * 2);
    return 0;
}

API void gba_reset(void) { g->core->reset(g->core); }

API void gba_set_keys(uint32_t keys) { g->core->setKeys(g->core, keys); }

// Run n frames (no rendering cost is skipped: the picture is the last frame's).
API void gba_run_frames(int n) {
    for (int i = 0; i < n; i++) g->core->runFrame(g->core);
}

API uint32_t gba_frame(void) { return g->core->frameCounter(g->core); }

// The last frame as RGBA (240x160x4), valid until the next call.
API const uint8_t* gba_video(void) {
    const size_t n = size_t(g->width) * g->height;
    for (size_t i = 0; i < n; i++) {
        uint32_t c = g->video[i];        // XBGR8888
        g->rgba[4 * i + 0] = c & 0xFF;
        g->rgba[4 * i + 1] = (c >> 8) & 0xFF;
        g->rgba[4 * i + 2] = (c >> 16) & 0xFF;
        g->rgba[4 * i + 3] = 0xFF;
    }
    return g->rgba.data();
}

// Audio produced since the last call: returns the number of stereo frames written to gba_audio_ptr().
API int gba_audio_read(void) {
    blip_t* l = g->core->getAudioChannel(g->core, 0);
    blip_t* r = g->core->getAudioChannel(g->core, 1);
    int n = blip_samples_avail(l);
    if (n > AUDIO_CAP) n = AUDIO_CAP;
    blip_read_samples(l, g->audio.data(), n, 1);
    blip_read_samples(r, g->audio.data() + 1, n, 1);
    return n;
}
API const int16_t* gba_audio_ptr(void) { return g->audio.data(); }

API uint32_t gba_read8(uint32_t a) { return g->core->busRead8(g->core, a); }
API uint32_t gba_read16(uint32_t a) { return g->core->busRead16(g->core, a); }
API uint32_t gba_read32(uint32_t a) { return g->core->busRead32(g->core, a); }
API void gba_read(uint32_t addr, uint8_t* out, int n) {
    for (int i = 0; i < n; i++) out[i] = uint8_t(g->core->busRead8(g->core, addr + uint32_t(i)));
}
API void gba_write(uint32_t addr, const uint8_t* data, int n) {
    for (int i = 0; i < n; i++) g->core->busWrite8(g->core, addr + uint32_t(i), data[i]);
}

API void gba_rtc_fake_epoch(double ms) {
    g->core->rtc.override = RTC_FAKE_EPOCH;
    g->core->rtc.value = int64_t(ms);
}

// Savestates: gba_state_save returns the size, the bytes at gba_state_ptr().
API int gba_state_size(void) { return int(g->core->stateSize(g->core)); }
API int gba_state_save(void) {
    g->state.resize(g->core->stateSize(g->core));
    return g->core->saveState(g->core, g->state.data()) ? int(g->state.size()) : -1;
}
API const uint8_t* gba_state_ptr(void) { return g->state.data(); }
API int gba_state_load(const uint8_t* data, int n) {
    if (size_t(n) != g->core->stateSize(g->core)) return -1;
    return g->core->loadState(g->core, data) ? 0 : -2;
}

// The cartridge save memory (flash), which savestates do not hold.
API int gba_save_size(void) {
    void* sram = nullptr;
    size_t size = g->core->savedataClone(g->core, &sram);
    g->state.assign(static_cast<uint8_t*>(sram), static_cast<uint8_t*>(sram) + size);
    free(sram);
    return int(size);
}
API int gba_save_restore(const uint8_t* data, int n) {
    return g->core->savedataRestore(g->core, data, size_t(n), true) ? 0 : -1;
}

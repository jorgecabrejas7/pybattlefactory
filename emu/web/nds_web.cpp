// DeSmuME for the browser (nds_web.wasm): a Nintendo DS run frame by frame from JavaScript, with the picture
// (RGBA, both screens), the sound (stereo int16 at 48 kHz), touch input and direct ARM9 memory access. Same idea
// and C API shape as gba_web.cpp (prefix nds_ instead of gba_).
//
// Core: the DeSmuME fork at github.com/44670/desmume-wasm (GPL-2.0), built by CMakeLists.txt (target nds_web) with
// the settings of its desmume/wasm-port/main.cpp: interpreter (no JIT), software 3D rasterizer, synchronous SPU,
// HLE BIOS and built-in firmware (no BIOS/firmware files), ROM and cartridge save in memory (no file IO).
// That fork reads the ROM through the globals romBuffer/romLen (ROMReader.cpp) and keeps the cartridge save in the
// global EMUFILE_MEMORY savFile (mc.cpp), so this file defines those globals (and the other front-end symbols the
// fork's wasm-port/main.cpp used to provide). Small source patches are applied by CMake to build-tree copies of
// the fork's files (CMakeLists.txt, _nds_patch; the clone is untouched):
//   mc.cpp       the backup device deleted savFile whenever MMU_Reset rebuilt it (use-after-free on every reset)
//   slot1/2.cpp  savestate loading dereferenced the slot devices the fork left NULL (null call, states never loaded)
//   ROMReader    reads starting past the end of the ROM copied a negative length
//   rtc.cpp      the clock comes from nds_web_rtc_now() (nds_set_rtc)
// DeSmuME logs a few lines with printf at init/state load (emscripten's print; pass print/printErr to silence).

#include "NDSSystem.h"
#include "MMU.h"
#include "SPU.h"
#include "GPU.h"
#include "mc.h"
#include "movie.h"
#include "rasterize.h"
#include "render3D.h"
#include "saves.h"
#include "emufile.h"
#include "utils/datetime.h"
#include "utils/decrypt/decrypt.h"
#include "utils/decrypt/header.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#define API extern "C" EMSCRIPTEN_KEEPALIVE
#else
#define API extern "C"
#endif

// ---- globals the desmume-wasm core expects from its front end (were in wasm-port/main.cpp) ----
u8* romBuffer = nullptr;
int romLen = 0;
EMUFILE_MEMORY* savFile = new EMUFILE_MEMORY();
int emuLastError = 0;
volatile bool execute = true;
volatile bool paused = false;
int DESMUME_SAMPLE_RATE = 48000;   // SPU output rate: samples_per_hline in SPU.cpp is set for 48000 Hz

extern "C" unsigned cpu_features_get_core_amount(void) { return 1; }   // no worker threads
extern "C" int bb(int, int, int, int) { return 0; }

u8 Mic_ReadSample() { return 64; }   // silent microphone (7-bit unsigned, centre 64)

// The fork removed the secure-area decryption (it needs the console's key tables), so only ROMs whose secure area
// is already decrypted (normal dumps, and decomp builds) or homebrew are accepted; encrypted ones fail with -1001.
bool DecryptSecureArea(u8* romheader, u8* secure) {
    const int type = DetectRomType(*reinterpret_cast<Header*>(romheader), reinterpret_cast<char*>(secure));
    if (type == ROMTYPE_INVALID) return false;
    if (type >= ROMTYPE_ENCRSECURE) { emuLastError = -1001; return false; }
    return true;
}

namespace {

constexpr int W = 256, H = 192;
constexpr int AUDIO_CAP = 16384;   // stereo frames kept between nds_audio_read calls (~0.34 s)

struct Emu {
    std::vector<uint8_t> rgba = std::vector<uint8_t>(size_t(W) * H * 2 * 4, 0);
    std::vector<int16_t> pending, audio;   // interleaved stereo
    std::vector<uint8_t> state;
    std::vector<uint8_t> savePlain;
    uint32_t keys = 0;
    bool touch = false;
    int tx = 0, ty = 0;
    bool rtcFixed = false;
    double rtcBaseMs = 0;
    int rtcBaseFrame = 0;
    bool ready = false;
};
Emu* g = nullptr;
u8* romAlloc = nullptr;
int romAllocSize = 0;

// ---- sound: the SPU hands each scanline's samples (48 kHz) to FetchSamples; keep them all ----
int sndInit(int) { return 0; }
void sndDeInit() {}
void sndUpdate(s16*, u32) {}
u32 sndSpace() { return 0; }   // SPU_Emulate_user (the pull path) is never used
void sndNop() {}
void sndVol(int) {}
void sndFetch(s16* buf, size_t n, ESynchMode, ISynchronizingAudioBuffer*) {
    if (!g) return;
    size_t have = g->pending.size() / 2;
    if (have + n > size_t(AUDIO_CAP)) n = have >= size_t(AUDIO_CAP) ? 0 : AUDIO_CAP - have;   // drop on overflow
    g->pending.insert(g->pending.end(), buf, buf + 2 * n);
}

}  // namespace

SoundInterface_struct SNDWeb = {1, "nds_web", sndInit, sndDeInit, sndUpdate, sndSpace,
                                sndNop, sndNop, sndVol, sndNop, sndFetch, nullptr};
SoundInterface_struct* SNDCoreList[] = {&SNDDummy, &SNDWeb, nullptr};
GPU3DInterface* core3DList[] = {&gpu3DNull, &gpu3DRasterize, nullptr};

// The DS clock (rtc.cpp, patched): wall clock, or a fixed start time that advances with emulated frames.
DateTime nds_web_rtc_now() {
    if (!g || !g->rtcFixed) return DateTime::get_Now();
    double ms = g->rtcBaseMs + (currFrameCounter - g->rtcBaseFrame) * (1000.0 / 59.8261);
    time_t t = time_t(ms / 1000.0);
    struct tm tmv;
    gmtime_r(&t, &tmv);
    return DateTime(tmv.tm_year + 1900, tmv.tm_mon + 1, tmv.tm_mday, tmv.tm_hour, tmv.tm_min, tmv.tm_sec);
}

namespace {

// The cartridge save as DeSmuME's .dsv: the raw bytes followed by a footer naming the size and type, so the backup
// device starts RUNNING with that content instead of auto-detecting an empty chip.
void putSave(const uint8_t* data, int n) {
    savFile->truncate(0);
    savFile->fseek(0, SEEK_SET);
    if (!data || n <= 0) return;   // empty: DeSmuME auto-detects the chip type from the game's first commands
    BackupDevice& bd = MMU_new.backupDevice;
    savFile->fwrite(data, size_t(n));
    u32 size = u32(n);
    u32 type = bd.searchFileSaveType(size);
    type = (type == 0xFF) ? 0 : type + 1;
    savFile->fprintf("|<--Snip above here to create a raw sav by excluding this DeSmuME savedata footer:");
    savFile->write_32LE(size);                                         // bytes actually written
    savFile->write_32LE(size);                                         // padded size
    savFile->write_32LE(type);                                         // save type
    savFile->write_32LE(BackupDevice::addr_size_for_old_save_size(int(size)));
    savFile->write_32LE(size);                                         // memory size
    savFile->write_32LE(0u);                                           // version
    savFile->fprintf("%s", "|-DESMUME SAVE-|");
    savFile->fseek(0, SEEK_SET);
}

// Size of the raw save inside savFile (without the .dsv footer, if one is there).
size_t rawSaveSize() {
    size_t n = size_t(savFile->size());
    const size_t foot = BackupDevice::GetDSVFooterSize();
    static const char cookie[] = "|-DESMUME SAVE-|";
    const size_t cl = sizeof(cookie) - 1;
    if (n >= foot && memcmp(savFile->buf() + n - cl, cookie, cl) == 0) return n - foot;
    return n;
}

// The cartridge header's own checksum (CRC-16/MODBUS of bytes 0..0x15D, stored at 0x15E): rejects non-ROM data,
// which DeSmuME would otherwise boot as "homebrew".
bool headerOk(const uint8_t* h) {
    uint16_t crc = 0xFFFF;
    for (int i = 0; i < 0x15E; i++) {
        crc ^= h[i];
        for (int b = 0; b < 8; b++) crc = (crc & 1) ? (crc >> 1) ^ 0xA001 : crc >> 1;
    }
    return crc == uint16_t(h[0x15E] | h[0x15F] << 8);
}

}  // namespace

// Optional zero-copy ROM upload: write the ROM into this buffer and pass it to nds_init (no second 128 MiB copy).
API uint8_t* nds_rom_alloc(int size) {
    if (romAlloc && romAllocSize >= size) return romAlloc;
    free(romAlloc);
    romAlloc = static_cast<u8*>(malloc(size_t(size)));
    romAllocSize = romAlloc ? size : 0;
    return romAlloc;
}

// Load a ROM (and optionally a raw cartridge save, e.g. 512 KiB flash) from memory. Returns 0 on success, else
// -1 already initialised, -2 not a DS ROM (bad header checksum), -3 out of memory, -4 NDS_Init failed,
// -1001 encrypted secure area (not supported by the fork), other negative: NDS_LoadROM failed.
// The ROM bytes are copied (unless they are the nds_rom_alloc buffer); the caller may free its own copy afterwards.
API int nds_init(const uint8_t* rom, int romSize, const uint8_t* save, int saveSize) {
    if (g) return -1;
    if (!rom || romSize < 0x200 || !headerOk(rom)) return -2;
    g = new Emu();
    g->audio.resize(AUDIO_CAP * 2);
    g->pending.reserve(AUDIO_CAP * 2);
    if (rom != romAlloc) {
        nds_rom_alloc(romSize);
        if (!romAlloc) return -3;
        memcpy(romAlloc, rom, size_t(romSize));
    }
    romBuffer = romAlloc;
    romLen = romSize;

    if (NDS_Init() != 0) return -4;
    SPU_ChangeSoundCore(1, 16384);
    GPU->Change3DRendererByID(RENDERID_SOFTRASTERIZER);
    putSave(save, saveSize);   // read by the backup device built in NDS_LoadROM's reset
    emuLastError = 0;
    if (NDS_LoadROM("rom.nds") < 1) return emuLastError ? emuLastError : -5;
    SPU_SetSynchMode(ESynchMode_Synchronous, ESynchMethod_N);
    SPU_SetVolume(100);
    savFile->changed = false;
    g->ready = true;
    return 0;
}

API void nds_reset(void) { NDS_Reset(); }

// Keys, one bit each (the hardware KEYINPUT order, then EXTKEYIN; same low 10 bits as gba_set_keys):
//   bit 0 A, 1 B, 2 SELECT, 3 START, 4 RIGHT, 5 LEFT, 6 UP, 7 DOWN, 8 R, 9 L, 10 X, 11 Y, 13 LID (closed).
// (wasm-port/main.cpp used a different order: 0 RIGHT 1 LEFT 2 DOWN 3 UP 4 SELECT 5 START 6 B 7 A 8 Y 9 X 10 L 11 R.)
API void nds_set_keys(uint32_t k) { if (g) g->keys = k; }

// Touch screen (bottom screen, 0..255 x 0..191). down=0 releases.
API void nds_touch(int down, int x, int y) {
    if (!g) return;
    g->touch = down != 0;
    g->tx = x < 0 ? 0 : x > 255 ? 255 : x;
    g->ty = y < 0 ? 0 : y > 191 ? 191 : y;
}

// Run one frame. draw=0 skips the 2D/3D rendering of this frame (faster; nds_video keeps the last picture).
API void nds_run_frame(int draw) {
    if (!g || !g->ready) return;
    const uint32_t k = g->keys;
    auto b = [k](int i) { return (k >> i & 1) != 0; };
    if (!draw) NDS_SkipNextFrame();
    if (g->touch) NDS_setTouchPos(u16(g->tx), u16(g->ty));
    else NDS_releaseTouch();
    //         R      L      D      U      Select Start  B      A      Y       X       L      R      debug  lid
    NDS_setPad(b(4), b(5), b(7), b(6), b(2), b(3), b(1), b(0), b(11), b(10), b(9), b(8), false, b(13));
    NDS_beginProcessingInput();
    NDS_endProcessingInput();
    NDS_exec<false>();
}

API uint32_t nds_frame(void) { return uint32_t(currFrameCounter); }

// The last drawn frame as RGBA, 256x384x4: top screen (rows 0..191) then bottom screen (rows 192..383).
API const uint8_t* nds_video(void) {
    if (!g) return nullptr;
    const NDSDisplayInfo& di = GPU->GetDisplayInfo();
    uint8_t* o = g->rgba.data();
    for (int s = 0; s < 2; s++) {
        const u16* p = static_cast<const u16*>(di.nativeBuffer[s == 0 ? NDSDisplayID_Main : NDSDisplayID_Touch]);
        for (int i = 0; i < W * H; i++) {
            const u16 c = p[i];
            const u8 r = c & 31, gg = (c >> 5) & 31, bl = (c >> 10) & 31;
            o[0] = u8(r << 3 | r >> 2);
            o[1] = u8(gg << 3 | gg >> 2);
            o[2] = u8(bl << 3 | bl >> 2);
            o[3] = 0xFF;
            o += 4;
        }
    }
    return g->rgba.data();
}

// Audio produced since the last call (48000 Hz, ~802 stereo frames per DS frame): returns the number of stereo
// frames written to nds_audio_ptr() (interleaved int16 L,R).
API int nds_audio_read(void) {
    if (!g) return 0;
    const int n = int(g->pending.size() / 2);
    std::copy(g->pending.begin(), g->pending.end(), g->audio.begin());
    g->pending.clear();
    return n;
}
API const int16_t* nds_audio_ptr(void) { return g ? g->audio.data() : nullptr; }

// ARM9 bus (main RAM 0x02000000-0x023FFFFF, etc.).
API uint32_t nds_read8(uint32_t a) { return _MMU_ARM9_read08(a); }
API uint32_t nds_read16(uint32_t a) { return _MMU_ARM9_read16(a); }
API uint32_t nds_read32(uint32_t a) { return _MMU_ARM9_read32(a); }
API void nds_read(uint32_t addr, uint8_t* out, int n) {
    for (int i = 0; i < n; i++) out[i] = _MMU_ARM9_read08(addr + uint32_t(i));
}
API void nds_write(uint32_t addr, const uint8_t* data, int n) {
    for (int i = 0; i < n; i++) _MMU_ARM9_write08(addr + uint32_t(i), data[i]);
}

// The DS clock: from now on it starts at unixMs (shown as that instant's UTC calendar time; pass
// Date.now() - new Date().getTimezoneOffset() * 60000 for local time) and advances with emulated frames
// (deterministic). A negative value goes back to the host's wall clock.
API void nds_set_rtc(double unixMs) {
    if (!g) return;
    g->rtcFixed = unixMs >= 0;
    g->rtcBaseMs = unixMs;
    g->rtcBaseFrame = currFrameCounter;
}

// Savestates: nds_state_save returns the size (or -1), the bytes at nds_state_ptr(). They include the cartridge save.
API int nds_state_save(void) {
    if (!g) return -1;
    EMUFILE_MEMORY os;
    if (!savestate_save(os, 0)) return -1;
    g->state.assign(os.buf(), os.buf() + os.size());
    return int(g->state.size());
}
API const uint8_t* nds_state_ptr(void) { return g ? g->state.data() : nullptr; }
API int nds_state_load(const uint8_t* data, int n) {
    if (!g) return -1;
    EMUFILE_MEMORY is(const_cast<uint8_t*>(data), n);
    return savestate_load(is) ? 0 : -2;
}

// The cartridge save (raw, e.g. 512 KiB flash for Pokemon Platinum; 0 bytes until the game first touches a
// chip of unknown type when started without a save). Pointer valid until the next emulator call.
API int nds_save_size(void) { return int(rawSaveSize()); }
API const uint8_t* nds_save_ptr(void) {
    const size_t n = rawSaveSize();
    if (!g) return nullptr;
    g->savePlain.assign(savFile->buf(), savFile->buf() + n);
    return g->savePlain.data();
}
// 1 if the game wrote its save since the last call (then read nds_save_ptr and store it).
API int nds_save_changed(void) {
    const int r = savFile->changed ? 1 : 0;
    savFile->changed = false;
    return r;
}
// Replace the cartridge save with raw bytes (the backup chip is rebuilt; call nds_reset to make the game reread it).
API int nds_save_restore(const uint8_t* data, int n) {
    if (!g) return -1;
    putSave(data, n);
    reconstruct(&MMU_new.backupDevice);
    savFile->changed = false;
    return 0;
}

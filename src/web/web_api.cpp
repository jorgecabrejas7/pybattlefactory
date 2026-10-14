// The C API of the browser build (gen3_web.wasm): a FactorySession behind plain functions that a
// JavaScript wrapper calls. Strings are returned as pointers valid until the next call on the same
// handle; errors return -1 (or NULL) and fs_last_error() says why. Nothing here throws.
//
// ABI version 1. Encoded battle observations are written to a fixed struct (fs_encoded_layout()
// describes it as JSON) so JavaScript reads the arrays straight from wasm memory.

#include "factory_session.hpp"

#include <cstdint>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#define API extern "C" EMSCRIPTEN_KEEPALIVE
#else
#define API extern "C"
#endif

using namespace pkmn;
using namespace pkmn::web;

namespace {

constexpr int ABI_VERSION = 1;
constexpr int MON_NUM_W = MON_NUM_V4;

// One battle observation, fp32 / int32 (the ONNX battler's inputs)
struct WebEncoded {
    int32_t mon_ids[6][MON_IDS];
    float mon_num[6][MON_NUM_W];
    float move_num[6][4][MOVE_NUM];
    int32_t ctx_ids[CTX_IDS];
    float ctx_num[CTX_NUM];
    uint8_t mask[8];          // 7 used: moves 0-3, switches to party 0-2
    int32_t active;
};

struct Handle {
    FactorySession session;
    std::vector<WebEncoded> batch;   // the search's observations to evaluate
    std::string text;         // the last returned string
    std::string state;        // the last fs_save
    WebEncoded enc;
    Handle(bool openLevel, int maxTurns) : session(openLevel, maxTurns) {}
};

std::string g_error;

template <class F> int guarded(F&& f) {
    try {
        f();
        return 0;
    } catch (const std::exception& e) {
        g_error = e.what();
    } catch (...) {
        g_error = "unknown error";
    }
    return -1;
}

const char* keep(Handle* h, std::string s) {
    h->text = std::move(s);
    return h->text.c_str();
}

}  // namespace

API int fs_abi(void) { return ABI_VERSION; }
API const char* fs_last_error(void) { return g_error.c_str(); }

API Handle* fs_new(int openLevel, int maxTurns) {
    try {
        return new Handle(openLevel != 0, maxTurns > 0 ? maxTurns : 500);
    } catch (const std::exception& e) {
        g_error = e.what();
        return nullptr;
    }
}

API void fs_free(Handle* h) { delete h; }

API int fs_reset(Handle* h, uint32_t seed, int winStreak, int rentsCount) {
    return guarded([&] { h->session.reset(seed, winStreak, rentsCount); });
}

API int fs_phase(Handle* h) { return int(h->session.phase()); }
API int fs_wins(Handle* h) {
    int w = 0;
    return guarded([&] { w = h->session.wins(); }) == 0 ? w : -1;
}
API int fs_last_battle_won(Handle* h) { return h->session.lastBattleWon(); }

API const char* fs_view_json(Handle* h) {
    std::string s;
    return guarded([&] { s = h->session.viewJson(); }) == 0 ? keep(h, std::move(s)) : nullptr;
}
API const char* fs_run_info_json(Handle* h) {
    std::string s;
    return guarded([&] { s = h->session.runInfoJson(); }) == 0 ? keep(h, std::move(s)) : nullptr;
}
API const char* fs_take_log_json(Handle* h) {
    std::string s;
    return guarded([&] { s = h->session.takeLogJson(); }) == 0 ? keep(h, std::move(s)) : nullptr;
}
API const char* fs_estimates_json(Handle* h) {
    std::string s;
    return guarded([&] { s = h->session.estimatesJson(); }) == 0 ? keep(h, std::move(s)) : nullptr;
}

// kind: 0 rental (a, b, c) | 1 swap (a = own slot or -1 keep, b = enemy slot) | 2 move (a) | 3 switch (a) | 4 forfeit
API int fs_act(Handle* h, int kind, int a, int b, int c) {
    return guarded([&] {
        FactorySession& s = h->session;
        switch (kind) {
        case 0: s.actRental(a, b, c); break;
        case 1: s.actSwap(a, b); break;
        case 2: s.actMove(a); break;
        case 3: s.actSwitch(a); break;
        case 4: s.actForfeit(); break;
        default: throw std::invalid_argument("unknown action kind");
        }
    });
}

namespace {
bool toWeb(const EncodedObs* e, WebEncoded& w) {
    std::memset(&w, 0, sizeof w);
    for (int i = 0; i < 6; i++)
        for (int k = 0; k < MON_IDS; k++) w.mon_ids[i][k] = int32_t(e->mon_ids[i][k]);
    if (e->mon_w != MON_NUM_W) {
        g_error = "unexpected encoding layout";
        return false;
    }
    std::memcpy(w.mon_num, e->mon_num, sizeof w.mon_num);
    std::memcpy(w.move_num, e->move_num, sizeof w.move_num);
    for (int k = 0; k < CTX_IDS; k++) w.ctx_ids[k] = int32_t(e->ctx_ids[k]);
    std::memcpy(w.ctx_num, e->ctx_num, sizeof w.ctx_num);
    for (int k = 0; k < 7; k++) w.mask[k] = e->mask[k] ? 1 : 0;
    w.active = int32_t(e->active);
    return true;
}
}  // namespace

// The battle observation (encoding v5 = v4 layout) of the current decision: a pointer to a WebEncoded, NULL when
// not in battle.
API const WebEncoded* fs_encode(Handle* h) {
    const EncodedObs* e = nullptr;
    if (guarded([&] { e = &h->session.encodeBattle(); }) != 0) return nullptr;
    return toWeb(e, h->enc) ? &h->enc : nullptr;
}

API const char* fs_encoded_layout(Handle* h) {
    std::string s = "{\"size\":" + std::to_string(sizeof(WebEncoded)) +
                    ",\"mon_ids\":{\"offset\":" + std::to_string(offsetof(WebEncoded, mon_ids)) +
                    ",\"shape\":[6," + std::to_string(MON_IDS) + "],\"type\":\"int32\"}" +
                    ",\"mon_num\":{\"offset\":" + std::to_string(offsetof(WebEncoded, mon_num)) +
                    ",\"shape\":[6," + std::to_string(MON_NUM_W) + "],\"type\":\"float32\"}" +
                    ",\"move_num\":{\"offset\":" + std::to_string(offsetof(WebEncoded, move_num)) +
                    ",\"shape\":[6,4," + std::to_string(MOVE_NUM) + "],\"type\":\"float32\"}" +
                    ",\"ctx_ids\":{\"offset\":" + std::to_string(offsetof(WebEncoded, ctx_ids)) +
                    ",\"shape\":[" + std::to_string(CTX_IDS) + "],\"type\":\"int32\"}" +
                    ",\"ctx_num\":{\"offset\":" + std::to_string(offsetof(WebEncoded, ctx_num)) +
                    ",\"shape\":[" + std::to_string(CTX_NUM) + "],\"type\":\"float32\"}" +
                    ",\"mask\":{\"offset\":" + std::to_string(offsetof(WebEncoded, mask)) +
                    ",\"shape\":[7],\"type\":\"uint8\"}" +
                    ",\"active\":{\"offset\":" + std::to_string(offsetof(WebEncoded, active)) +
                    ",\"shape\":[],\"type\":\"int32\"}}";
    return keep(h, std::move(s));
}

// Save: returns the size; fs_saved_ptr() points at the bytes (valid until the next fs_save on this handle).
API int fs_save(Handle* h) {
    if (guarded([&] { h->state = h->session.saveState(); }) != 0) return -1;
    return int(h->state.size());
}
API const uint8_t* fs_saved_ptr(Handle* h) { return reinterpret_cast<const uint8_t*>(h->state.data()); }

API int fs_load(Handle* h, const uint8_t* data, int size) {
    return guarded([&] { h->session.loadState(std::string(reinterpret_cast<const char*>(data), size_t(size))); });
}

// ---- mirror mode (the real game in an emulator; see FactorySession::mirror*) ----

// Comparison regions: n pairs of u32 (GBA address, size).
API void fs_mirror_regions(Handle* h, const uint32_t* pairs, int n) {
    std::vector<FactorySession::Region> v;
    for (int i = 0; i < n; i++) v.push_back({pairs[2 * i], pairs[2 * i + 1]});
    h->session.mirrorRegions(v);
}

// The battle's start (BeginBattleIntro), packed little-endian: rng u32, battleTypeFlags u32, trainerId u16,
// lvlMode u8, pad u8, battleNum u16, factoryStreaks 4 x u16, towerStreaks 8 x u16, playerParty 300, enemyParty 300,
// rentalMons 72  (= 4+4+2+1+1+2+8+16+300+300+72 = 710 bytes)
API int fs_mirror_begin(Handle* h, const uint8_t* b, int n, int hintType, int hintStyle) {
    return guarded([&] {
        if (n != 710) throw std::invalid_argument("battle start: 710 bytes expected");
        auto u16 = [&](int at) { return uint16_t(b[at] | b[at + 1] << 8); };
        auto u32 = [&](int at) { return uint32_t(u16(at)) | uint32_t(u16(at + 2)) << 16; };
        FactorySession::BattleStart st{};
        st.rng = u32(0);
        st.battleTypeFlags = u32(4);
        st.trainerId = u16(8);
        st.lvlMode = b[10];
        st.battleNum = u16(12);
        for (int i = 0; i < 4; i++) st.factoryStreaks[i] = u16(14 + 2 * i);
        for (int i = 0; i < 8; i++) st.towerStreaks[i] = u16(22 + 2 * i);
        st.playerParty.assign(reinterpret_cast<const char*>(b + 38), 300);
        st.enemyParty.assign(reinterpret_cast<const char*>(b + 338), 300);
        st.rentalMons.assign(reinterpret_cast<const char*>(b + 638), 72);
        h->session.mirrorBegin(st, hintType, hintStyle);
        if (!h->session.mirrorSynced()) throw std::runtime_error(h->session.mirrorError());
    });
}

// 1: a new decision (fs_view_json / fs_encode / fs_estimates are this decision's), 2: still the last decision,
// 0: not reproduced (no AI for this battle)
API int fs_mirror_sync(Handle* h, int forced, const uint8_t* snap, int n) {
    int r = 0;
    guarded([&] { r = h->session.mirrorSync(forced != 0, std::string(reinterpret_cast<const char*>(snap), size_t(n))); });
    return r;
}
API int fs_mirror_finish(Handle* h, const uint8_t* snap, int n) {
    bool ok = false;
    guarded([&] { ok = h->session.mirrorFinish(std::string(reinterpret_cast<const char*>(snap), size_t(n))); });
    return ok ? 1 : 0;
}
API const char* fs_mirror_error(Handle* h) { return keep(h, h->session.mirrorError()); }

API void fs_set_context(Handle* h, int streak, int battle, int challenge, int rents, int noland, int trainerId) {
    h->session.setContext(streak, battle, challenge, rents, noland != 0, trainerId);
}
// phase: 0 rental 1 swap 2 battle 3 forced 4 over (mirror mode: the emulator's)
API void fs_set_phase(Handle* h, int phase) { h->session.setPhase(Phase(phase)); }

API const char* fs_rental_view_json(Handle* h, const uint8_t* b, int n, int hintType, int hintStyle) {
    std::string s;
    return guarded([&] {
        if (n != 6 * 102) throw std::invalid_argument("rentals: 6 x 102 bytes expected");
        s = h->session.rentalViewJson(std::string(reinterpret_cast<const char*>(b), size_t(n)), hintType, hintStyle);
    }) == 0 ? keep(h, std::move(s)) : nullptr;
}
API const char* fs_swap_view_json(Handle* h, const uint8_t* own, const uint8_t* enemy, int hintType, int hintStyle) {
    std::string s;
    return guarded([&] {
        s = h->session.swapViewJson(std::string(reinterpret_cast<const char*>(own), 300),
                                    std::string(reinterpret_cast<const char*>(enemy), 300), hintType, hintStyle);
    }) == 0 ? keep(h, std::move(s)) : nullptr;
}

// ---- search (FactorySession::search*) ----

// attempts: K * R determinizations, 3 slots each, 22 x int32 per slot: partySlot, species, moves[4], item, ivs[6],
// evs[6], nature, abilityBit, hpFraction (float32 bits). hidden: 14 x int32 (sleepElapsed[2][3], confusion[2],
// wrap[2], uproar[2], rampage[2]). seeds: 2 x u32 per root (determinize, redraw_turn). cfg: nSims, batch, maxDecisions.
API int fs_search_begin(Handle* h, const int32_t* att, int K, int R, const int32_t* hid, const uint32_t* seeds,
                        const float* rootPriors, float rootValue, int nSims, int batch, float cPuct) {
    return guarded([&] {
        std::vector<std::vector<Gen3Game::DetMon>> attempts(size_t(K) * R);
        const int32_t* p = att;
        for (auto& a : attempts) {
            for (int s = 0; s < 3; s++) {
                Gen3Game::DetMon m{};
                m.partySlot = p[0];
                m.species = p[1];
                for (int i = 0; i < 4; i++) m.moves[i] = p[2 + i];
                m.item = p[6];
                for (int i = 0; i < 6; i++) m.ivs[i] = p[7 + i];
                for (int i = 0; i < 6; i++) m.evs[i] = p[13 + i];
                m.nature = p[19];
                m.abilityBit = p[20];
                std::memcpy(&m.hpFraction, &p[21], 4);
                a.push_back(m);
                p += 22;
            }
        }
        Gen3Game::DetHidden h2{};
        for (int b = 0; b < 2; b++)
            for (int i = 0; i < 3; i++) h2.sleepElapsed[b][i] = hid[b * 3 + i];
        for (int b = 0; b < 2; b++) {
            h2.confusionElapsed[b] = hid[6 + b];
            h2.wrapElapsed[b] = hid[8 + b];
            h2.uproarElapsed[b] = hid[10 + b];
            h2.rampageElapsed[b] = hid[12 + b];
        }
        std::vector<uint64_t> sd;
        for (int k = 0; k < 2 * K; k++) sd.push_back(seeds[k]);
        SearchConfig cfg;
        cfg.nSims = nSims;
        cfg.batch = batch;
        cfg.cPuct = cPuct;
        h->session.searchBegin(attempts, R, h2, sd, rootPriors, rootValue, cfg);
    });
}

// The next observations to evaluate (n WebEncoded at fs_search_batch(), stride fs_encoded_layout().size), or -1: done.
API int fs_search_prepare(Handle* h) {
    int n = -1;
    if (guarded([&] { n = h->session.searchPrepare(); }) != 0) return -2;
    if (n <= 0) return n;
    h->batch.resize(size_t(n));
    const EncodedObs* e = h->session.searchObservations();
    for (int i = 0; i < n; i++)
        if (!toWeb(&e[i], h->batch[i])) return -2;
    return n;
}
API const WebEncoded* fs_search_batch(Handle* h) { return h->batch.data(); }
API int fs_search_submit(Handle* h, const float* priors, const float* values) {
    return guarded([&] { h->session.searchSubmit(priors, values); });
}
// {"visits":[7],"q":[7],"leaves","nodes","errors","depth","ms"}
API const char* fs_search_stats(Handle* h) {
    std::string s;
    return guarded([&] {
        SearchStats st = h->session.searchStats();
        s = "{\"visits\":[";
        for (int a = 0; a < 7; a++) s += (a ? "," : "") + std::to_string(st.visits[a]);
        s += "],\"q\":[";
        for (int a = 0; a < 7; a++) s += (a ? "," : "") + std::to_string(st.q[a]);
        s += "],\"leaves\":" + std::to_string(st.leaves) + ",\"nodes\":" + std::to_string(st.nodes) +
             ",\"errors\":" + std::to_string(st.errors) + ",\"depth\":" + std::to_string(st.maxDepth) +
             ",\"ms\":" + std::to_string(st.msTotal) + "}";
    }) == 0 ? keep(h, std::move(s)) : nullptr;
}
API void fs_search_end(Handle* h) { h->session.searchEnd(); }

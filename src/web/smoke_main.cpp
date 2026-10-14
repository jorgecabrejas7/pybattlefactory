// Determinism check for the web build: plays Factory runs with a fixed pseudo-random policy and
// prints one digest line per decision (game RAM read by GBA address, RNG, Factory info and the C++
// observer's encoding). The native and the emscripten builds must print the same lines.
//
//   smoke_main [runs=3] [seed=1] [verbose] [trace-random-of-run]

#include "gen3_game.hpp"
#include "gen3_observer.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

using namespace pkmn;

namespace {

uint64_t fnv(uint64_t h, const void* data, size_t n) {
    const auto* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < n; i++) h = (h ^ p[i]) * 0x100000001b3ULL;
    return h;
}

struct Lcg {
    uint64_t s;
    uint32_t next() { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return uint32_t(s >> 33); }
    int below(int n) { return int(next() % uint32_t(n)); }
};

constexpr uint32_t G_BATTLE_MONS = 0x2024084, G_PLAYER_PARTY = 0x20244ec, G_ENEMY_PARTY = 0x2024744;
constexpr uint16_t FLAG_SILVER = 0x860 + 0x6C, FLAG_GOLD = 0x860 + 0x6D;

bool g_verbose = false;

uint64_t digest(Gen3Game& g, uint64_t h) {
    const uint64_t seed = 0xcbf29ce484222325ULL;
    std::string mons = g.readRam(G_BATTLE_MONS, 352), own = g.readRam(G_PLAYER_PARTY, 600),
                foe = g.readRam(G_ENEMY_PARTY, 600);
    uint32_t rng = g.rng();
    auto fi = g.factoryInfo();
    if (g_verbose) {
        std::string tf = g.readRam(0x2022fec, 4);
        uint32_t flags;
        std::memcpy(&flags, tf.data(), 4);
        std::printf("  typeFlags %08x frames %u\n", flags, g.frames());
    }
    if (g_verbose)
        std::printf("  mons %016llx own %016llx foe %016llx rng %08x phase %d wins %d hint %d/%d\n",
                    (unsigned long long)fnv(seed, mons.data(), mons.size()),
                    (unsigned long long)fnv(seed, own.data(), own.size()),
                    (unsigned long long)fnv(seed, foe.data(), foe.size()), rng, fi.phase, fi.wins, fi.hintType,
                    fi.hintStyle);
    h = fnv(h, mons.data(), mons.size());
    h = fnv(h, own.data(), own.size());
    h = fnv(h, foe.data(), foe.size());
    h = fnv(h, &rng, sizeof rng);
    return fnv(h, &fi, sizeof fi);
}

}  // namespace

int main(int argc, char** argv) {
    int runs = argc > 1 ? std::atoi(argv[1]) : 3;
    uint32_t seed0 = argc > 2 ? uint32_t(std::atoi(argv[2])) : 1;
    g_verbose = argc > 3;
    set_encode_version(4);
    int traceRun = argc > 4 ? std::atoi(argv[4]) : -1;
    for (int r = 0; r < runs; r++) {
        Gen3Game::traceRandom(r == traceRun);
        Lcg rnd{seed0 * 7919ULL + r};
        int streak = (r * 9) % 49;
        Gen3Game g;
        int symbols = int(streak >= 21) + int(streak >= 42);
        g.setFlag(FLAG_SILVER, symbols >= 1);
        g.setFlag(FLAG_GOLD, symbols >= 2);
        g.factoryBegin(true, uint16_t(streak), 0, seed0 * 1000003u + uint32_t(r));
        ObsMemory mem;
        obs_init(mem);
        int decisions = 0, turns = 0;
        uint64_t h = 0xcbf29ce484222325ULL;
        while (decisions < 5000) {
            int fp = g.factoryPhase();
            if (fp == Gen3Game::F_RUN_OVER) break;
            if (fp == Gen3Game::F_RENTAL) {
                h = digest(g, h);
                for (;;) {   // (argument evaluation order differs between compilers: draw first)
                    int a = rnd.below(6), b = rnd.below(6), c = rnd.below(6);
                    if (g.factoryRent(a, b, c)) break;
                }
                auto fi = g.factoryInfo();
                obs_init(mem, uint16_t(fi.hintType), uint16_t(fi.hintStyle));
            } else if (fp == Gen3Game::F_SWAP) {
                h = digest(g, h);
                int keep = rnd.below(2), own = rnd.below(3), foe = rnd.below(3);
                if (keep == 0 || !g.factorySwap(own, foe)) g.factorySwap(-1, 0);
                auto fi = g.factoryInfo();
                obs_init(mem, uint16_t(fi.hintType), uint16_t(fi.hintStyle));
            } else {
                int d = g.run(400000);
                if (d == Gen3Game::BATTLE_OVER) {
                    obs_finish(mem, g);
                    d = g.factoryRunBattle();
                }
                if (d == Gen3Game::BATTLE_OVER) { turns = 0; continue; }
                if (d == Gen3Game::TIMEOUT) { std::printf("TIMEOUT\n"); return 1; }
                bool forced = d == Gen3Game::SWITCH;
                uint8_t unusable = g.unusableMoves(0);
                bool canSwitch = g.canSwitch(0);
                obs_observe(mem, g, forced, unusable, canSwitch);
                EncodedObs enc;
                std::memset(&enc, 0, sizeof enc);
                auto fi = g.factoryInfo();
                encode_battle(mem, g, EncodeCtx{streak + fi.wins, fi.wins % 7, (streak + fi.wins) / 7, 0}, enc);
                h = digest(g, h);
                if (g_verbose)
                    std::printf("  enc ids %016llx num %016llx moves %016llx ctx %016llx mask %016llx\n",
                                (unsigned long long)fnv(0, enc.mon_ids, sizeof enc.mon_ids),
                                (unsigned long long)fnv(0, enc.mon_num, sizeof enc.mon_num),
                                (unsigned long long)fnv(0, enc.move_num, sizeof enc.move_num),
                                (unsigned long long)fnv(0, enc.ctx_num, sizeof enc.ctx_num),
                                (unsigned long long)fnv(0, enc.mask, sizeof enc.mask));
                h = fnv(h, &enc, sizeof enc);
                if (!forced && ++turns > 300) { g.forfeit(); decisions++; continue; }
                bool acted = false;
                for (int tries = 0; tries < 20 && !acted; tries++) {
                    try {
                        if (forced || rnd.below(5) == 0) g.chooseSwitch(rnd.below(3));
                        else g.chooseMove(rnd.below(4));
                        acted = true;
                    } catch (const std::exception&) {}
                }
                for (int i = 0; i < 3 && !acted; i++) {
                    try { g.chooseSwitch(i); acted = true; } catch (const std::exception&) {}
                }
                if (!acted) {
                    if (forced) { std::printf("no legal switch\n"); return 1; }
                    g.forfeit();
                }
            }
            decisions++;
            std::printf("run %d dec %d phase %d %016llx\n", r, decisions, fp, (unsigned long long)h);
        }
        std::printf("run %d done wins %d decisions %d %016llx\n", r, g.factoryInfo().wins, decisions,
                    (unsigned long long)h);
    }
    return 0;
}

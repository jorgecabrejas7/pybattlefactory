#pragma once

// A Battle Factory run for the browser: pybattle/backend.py SimBackend in C++ (same phases, same
// observer calls at the same moments, same views), on top of Gen3Game and the C++ observer.
//
//   FactorySession s;  s.reset(seed, streak, rents);
//   while (s.phase() != Phase::RUN_OVER) { s.viewJson(); ...; s.act*(...); }
//
// Views are JSON objects with the field names of the Python dataclasses (RentalView, SwapView,
// BattleView, RunInfo; pybattle/web_schema.py writes the same JSON from SimBackend, and
// tests/python/test_web_session.py checks they agree).

#include "gen3_game.hpp"
#include "gen3_observer.hpp"
#include "gen3_search.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace pkmn::web {

enum class Phase { RENTAL = 0, SWAP = 1, BATTLE = 2, FORCED_SWITCH = 3, RUN_OVER = 4 };

// pybattle/emu/decode.py decode_pokemon (the fields views use)
struct PartyMon {
    uint32_t personality, status;
    int species, item, moves[4], pp[4], evs[6], ivs[6], ability_num, level, hp, max_hp, stats[5];
    int nature() const { return int(personality % 25); }
};
PartyMon decode_pokemon(const uint8_t* data);   // 100 bytes, struct Pokemon
// Game text (host build: ASCII literals mixed with game-charmap bytes from the game's own conversions,
// control codes) as UTF-8; line breaks become '\n'.
std::string game_text_utf8(const std::string& text);

class FactorySession {
public:
    explicit FactorySession(bool openLevel = true, int maxTurns = 500);

    void reset(uint32_t seed, int winStreak, int rentsCount);
    Phase phase() const { return m_phase; }
    int wins();
    int lastBattleWon() const { return m_lastWon; }   // -1 none yet, 0 lost, 1 won

    std::string viewJson();       // the current phase's view; "null" when the run is over
    // The battle messages printed since the last call, in order, as the game words them, each with the HP /
    // status of both active Pokemon at that moment (the opponent's HP as its HP bar shows it: 0-48 pixels).
    std::string takeLogJson();
    std::string runInfoJson();

    // Actions (std::invalid_argument / std::out_of_range / std::runtime_error when refused, nothing changed)
    void actRental(int a, int b, int c);
    void actSwap(int ownSlot, int enemySlot);     // ownSlot < 0: keep the team
    void actMove(int slot);
    void actSwitch(int partyIndex);
    void actForfeit();

    // rl/damage.py battle_estimates (v4) of the current battle view, for display: {"own": [[[min,max,ko] x4] x3],
    // "foe": [{"move", "min", "max", "ko"}] (the active enemy's revealed moves vs our active Pokemon),
    // "threat": [[max, ko] x3], "speed": [[surely, maybe] x3]}
    std::string estimatesJson();

    // rl.encode.battle(view, ctx_of(run_info)) at a battle decision (encoding v4 layout; v5 battle = v4)
    const EncodedObs& encodeBattle();

    // Everything needed to resume: the game, the observers and the session's own fields.
    std::string saveState();
    void loadState(const std::string& bytes);

    // ---- mirror mode: the real game runs in an emulator; the simulator replays its battles ----------------------
    // (pybattle/emu/mirror.py: started from the battle's start, given the same actions, the simulator reproduces the
    // emulator's battle RAM byte for byte; the action the player took is found by trying each legal one.)
    struct BattleStart {
        uint32_t rng, battleTypeFlags;
        uint16_t trainerId;
        uint8_t lvlMode;
        uint16_t battleNum, factoryStreaks[4], towerStreaks[8];
        std::string playerParty, enemyParty, rentalMons;      // 300, 300, 72 bytes
    };
    struct Region { uint32_t addr, size; };
    void mirrorBegin(const BattleStart& start, int hintType, int hintStyle);
    void mirrorRegions(const std::vector<Region>& regions) { m_regions = regions; }
    // The emulator is at a battle decision (forced: a party menu after a faint / Baton Pass). `snapshot`: its gRngValue
    // (4 bytes LE), the regions in order, gPlayerParty (300, field order), gEnemyParty (300). Returns true when the
    // simulator matches it: 1 a new decision (the view / encoding / estimates are this decision's), 2 still the last
    // decision (nothing happened: a menu opened and closed), 0 not reproduced (the mirror stops for this battle).
    int mirrorSync(bool forced, const std::string& snapshot);
    // The emulator's battle is over (gBattleOutcome set): bring the simulator there; the observer records the last turn.
    bool mirrorFinish(const std::string& snapshot);
    bool mirrorSynced() const { return m_mirrorSynced; }
    const std::string& mirrorError() const { return m_mirrorError; }
    // The run context the emulator shows (encodings, views)
    void setContext(int streak, int battle, int challenge, int rents, bool noland, int trainerId);
    // Rental / swap views from the emulator's memory: 6 x (frontier mon id u16 + struct Pokemon), hints
    std::string rentalViewJson(const std::string& mons, int hintType, int hintStyle);
    // own / enemy party (300 bytes each); the defeated team's reveals come from the last mirrored battle
    std::string swapViewJson(const std::string& own, const std::string& enemy, int hintType, int hintStyle);
    void setPhase(Phase p) { m_phase = p; }

    // ---- search (rl/search.py SearchBattler, legal mode) on the mirrored battle -------------------------------
    // attempts[k * R + r]: the r-th determinization drawn for root k (3 opponent slots each: rl/determinize.py specs);
    // the first that leaves what the player can do unchanged is used (SearchBattler._roots redraws), the last otherwise.
    void searchBegin(const std::vector<std::vector<Gen3Game::DetMon>>& attempts, int R, const Gen3Game::DetHidden& hidden,
                     const std::vector<uint64_t>& seeds, const float rootPriors[7], float rootValue, const SearchConfig& cfg);
    int searchPrepare();                         // observations to evaluate (searchObservations), or -1: done
    const EncodedObs* searchObservations() const;
    void searchSubmit(const float* priors, const float* values);
    SearchStats searchStats() const;
    void searchEnd() { m_search.reset(); m_searchGames.clear(); }
    // The same roots searched by runSearch in one call (tests: the resumable session must grow the same trees).
    SearchStats searchRunNative(const std::vector<std::vector<Gen3Game::DetMon>>& attempts, int R,
                                const Gen3Game::DetHidden& hidden, const std::vector<uint64_t>& seeds,
                                const float rootPriors[7], float rootValue, const SearchConfig& cfg,
                                const BatchEvaluator& evaluator);

    Gen3Game& game() { return m_game; }
    const ObsMemory& observer() const { return m_obs; }

private:
    void sync();
    void newObserver();
    void beforeBattleAction();
    struct Info { int streak, battle, challenge, rents; bool noland; };
    Info runInfo();

    Gen3Game m_game;
    ObsMemory m_obs, m_obsDone;
    EncodedObs m_enc;
    Phase m_phase = Phase::RUN_OVER;
    bool m_openLevel;
    int m_maxTurns, m_turns = 0, m_startStreak = 0, m_lastWon = -1;
    bool m_hasDone = false;
    // mirror mode
    bool m_mirror = false, m_mirrorSynced = false, m_atDecision = false;
    std::string m_mirrorError;
    std::vector<Region> m_regions;
    struct Context { int streak, battle, challenge, rents; bool noland; int trainerId; } m_ctx{};
    bool matches(Gen3Game& g, const std::string& snapshot);
    std::unique_ptr<SearchSession> m_search;
    std::vector<std::unique_ptr<Gen3Game>> m_searchGames;
    std::vector<SearchRoot> searchRoots(const std::vector<std::vector<Gen3Game::DetMon>>& attempts, int R,
                                        const Gen3Game::DetHidden& hidden, const std::vector<uint64_t>& seeds);
    std::string swapJson(const std::string& own, const std::string& foe, int hintType, int hintStyle);
};

}  // namespace pkmn::web

#pragma once

// The game's own battle and Battle Factory code, headless (third_party/pokeemerald,
// src/gen3/host.c, src/gen3/factory_run.c).
//
// The game code keeps its state in globals, grouped into one memory section. Each Gen3Game
// owns a copy of that section and swaps it in while it runs, so any number of games can
// coexist (one thread at a time) and clone() is a plain copy.

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace pkmn {

class Gen3Game {
public:
    enum Decision { NONE = 0, ACTION = 1, SWITCH = 2, BATTLE_OVER = 3, TIMEOUT = 4 };

    Gen3Game();
    Gen3Game(const Gen3Game& other);
    Gen3Game& operator=(const Gen3Game& other);
    ~Gen3Game();

    // Setup (before start())
    void setBattle(uint32_t battleTypeFlags, uint16_t trainerId);
    void writeParty(int side, const std::string& raw);          // raw struct Pokemon[], game format
    void writeSaveBlock2(uint32_t offset, const std::string& data);
    std::string readSaveBlock2(uint32_t offset, size_t size);
    void setVar(uint16_t id, uint16_t value);
    void setFlag(uint16_t id, bool on);

    // Run the start of battle up to BeginBattleIntro, then seed gRngValue.
    bool start(uint32_t rng);

    // Advance until the player must decide; returns a Decision.
    int run(uint32_t maxFrames = 200000);
    void chooseMove(uint8_t slot);
    uint8_t unusableMoves(uint8_t battler = 0);   // CheckMoveLimitations bitmask
    bool canSwitch(uint8_t battler = 0);
    uint16_t choicedMove(uint8_t battler = 0);          // not trapped (Mean Look, wrap, Ingrain, Shadow Tag...)
    void chooseSwitch(uint8_t partyIndex);
    void forfeit();

    uint32_t rng();
    std::string readRam(uint32_t gbaAddress, size_t size);   // by GBA address, like the emulator
    void writeRam(uint32_t gbaAddress, const std::string& data);   // tests / debugging
    uint32_t frames();

    // Battle Factory run (singles). Phases: RENTAL -> BATTLE -> SWAP -> BATTLE ... -> RUN_OVER
    enum FactoryPhase { F_NONE = 0, F_RENTAL = 1, F_BATTLE = 2, F_SWAP = 3, F_RUN_OVER = 4 };
    void factoryBegin(bool openLevel, uint16_t winStreak, uint16_t rentsCount, uint32_t seed);
    int factoryPhase();
    std::string factoryRental(uint8_t i);        // struct Pokemon (game format)
    uint16_t factoryRentalMonId(uint8_t i);      // gBattleFrontierMons index
    bool factoryRent(uint8_t a, uint8_t b, uint8_t c);
    bool factorySwap(int playerSlot, int enemySlot);   // playerSlot < 0: keep
    int factoryRunBattle(uint32_t maxFrames = 400000); // Decision; BATTLE_OVER advances the run
    struct FactoryInfo {
        int phase, lvlMode, hintType, hintStyle, brainStatus, lastOutcome, trainerId, wins, swaps, challengesWon;
    };
    FactoryInfo factoryInfo();
    static void traceRandom(bool enable);

    // Search (src/gen3/search_host.c). These never advance the Factory phase.
    // simStep: kind 0 = move slot `index`, 1 = switch to party index `index`; then run() until the
    // next player decision or the battle's end. Returns {Decision, gBattleOutcome}.
    std::pair<int, int> simStep(int kind, int index);
    void setRng(uint32_t value);
    int battleOutcome();
    struct DetSlot { int partySlot, setId, iv, abilityBit; float hpFraction; };
    void determinize(const std::vector<DetSlot>& slots, int64_t hiddenSeed);

    // Make this game's state the live battle RAM (for direct readers such as the C++ observer).
    void makeLive() { activate(); }

private:
    void activate();
    std::vector<uint8_t> m_state;
};

}  // namespace pkmn

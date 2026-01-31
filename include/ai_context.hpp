#pragma once

#include "types.hpp"
#include "constants.hpp"
#include <vector>
#include <map>
#include <functional>

namespace pkmn {

class BattleEngine;

struct AIThinkingStruct {
    uint8_t aiState;
    uint8_t aiLogicId;
    uint8_t movesetIndex;
    uint16_t moveConsidered;
    int8_t score[4];
    uint32_t aiFlags;
    uint32_t aiAction;
    int funcResult; // For get_how_powerful etc
};

class AIContext {
public:
    AIContext(BattleEngine& engine, uint8_t aiBattler, uint8_t targetBattler);

    // AI Execution
    void execute(uint32_t logicId);
    
    // VM State
    std::vector<uint32_t> stack;
    
    // Context State
    AIThinkingStruct aiThinking;
    BattleEngine& engine;
    uint8_t battlerAI;
    uint8_t battlerTarget;

    // Command Helpers (Primitives for generated scripts)
    
    // Checks
    bool shouldFlee();
    bool shouldWatch();
    
    // Random
    bool randomLessThan(uint8_t val);
    bool randomGreaterThan(uint8_t val);
    
    // Scores
    void scoreOp(int val); // score +x or -x
    
    // HP
    bool hpLessThan(uint8_t battler, uint8_t percent);
    bool hpMoreThan(uint8_t battler, uint8_t percent);
    
    // Status
    bool hasStatus(uint8_t battler, uint32_t statusMask);
    bool hasStatus2(uint8_t battler, uint32_t statusMask);
    bool hasStatus3(uint8_t battler, uint32_t statusMask);
    bool hasSideStatus(uint8_t side, uint32_t statusMask);
    
    // Moves
    bool isMove(uint16_t move);
    bool hasMove(uint8_t battler, uint16_t move);
    bool hasMoveWithEffect(uint8_t battler, uint16_t effect); // Need to map effect ID?
    
    // Types & Effectiveness
    int getTypeEffectiveness(uint8_t multiplier); // Returns match with enum? No, script compares with constants.
    // Actually script does: if_type_effectiveness AI_EFFECTIVENESS_x2, label
    // So helper:
    bool typeEffectivenessEquals(int effectiveness);
    
    // Stats & Levels
    bool statLevelLessThan(uint8_t battler, uint8_t stat, uint8_t val);
    bool statLevelMoreThan(uint8_t battler, uint8_t stat, uint8_t val);
    bool statLevelEqual(uint8_t battler, uint8_t stat, uint8_t val);
    
    // Abilities
    uint8_t getAbility(uint8_t battler);
    bool hasAbility(uint8_t battler, uint8_t ability);
    
    // Effects
    bool isEffect(uint16_t effect);
    
    // Powerful Move Check
    void checkMostPowerfulMove(); // Sets funcResult
    
    // Helpers
    uint8_t getBattler(uint8_t scriptBattlerId); // AI_USER -> battlerAI
};

} // namespace pkmn

#include "ai_context.hpp"
#include "battle_engine.hpp"
#include "data.hpp"
#include <iostream>

namespace pkmn {

AIContext::AIContext(BattleEngine& eng, uint8_t ai, uint8_t target)
    : engine(eng), battlerAI(ai), battlerTarget(target) {
    // defaults
    aiThinking.aiState = 0;
    aiThinking.funcResult = 0;
    for(int i=0; i<4; ++i) aiThinking.score[i] = 0;
}

uint8_t AIContext::getBattler(uint8_t scriptBattlerId) {
    if (scriptBattlerId == AI_USER) return battlerAI;
    if (scriptBattlerId == AI_TARGET) return battlerTarget;
    if (scriptBattlerId == AI_USER_PARTNER) return battlerAI; // No doubles yet
    if (scriptBattlerId == AI_TARGET_PARTNER) return battlerTarget; // No doubles
    return battlerAI;
}

void AIContext::scoreOp(int val) {
    aiThinking.score[aiThinking.movesetIndex] += val;
    // Don't cap at 0 here? Decomp does: if score < 0 score = 0.
    if (aiThinking.score[aiThinking.movesetIndex] < 0) 
        aiThinking.score[aiThinking.movesetIndex] = 0;
}

bool AIContext::randomLessThan(uint8_t val) {
    return (engine.random() % 256) < val;
}

bool AIContext::randomGreaterThan(uint8_t val) {
    return (engine.random() % 256) > val;
}

bool AIContext::hpLessThan(uint8_t battlerId, uint8_t percent) {
    uint8_t id = getBattler(battlerId);
    const auto& mon = engine.getState().getActivePokemon(id);
    if (mon.maxHP == 0) return false;
    return (mon.currentHP * 100 / mon.maxHP) < percent;
}

bool AIContext::hpMoreThan(uint8_t battlerId, uint8_t percent) {
    uint8_t id = getBattler(battlerId);
    const auto& mon = engine.getState().getActivePokemon(id);
    if (mon.maxHP == 0) return false;
    return (mon.currentHP * 100 / mon.maxHP) > percent;
}

static uint32_t statusToStatus1(Status s) {
    if (s == Status::None) return STATUS1_NONE;
    if (s >= Status::Sleep1 && s <= Status::Sleep7) return static_cast<uint32_t>(s);
    if (s == Status::Poison) return STATUS1_POISON;
    if (s == Status::Burn) return STATUS1_BURN;
    if (s == Status::Freeze) return STATUS1_FREEZE;
    if (s == Status::Paralysis) return STATUS1_PARALYSIS;
    if (s == Status::BadPoison) return STATUS1_TOXIC_POISON;
    return 0;
}

bool AIContext::hasStatus(uint8_t battlerId, uint32_t statusMask) {
    uint8_t id = getBattler(battlerId);
    uint32_t s1 = statusToStatus1(engine.getState().getActivePokemon(id).status);
    return (s1 & statusMask) != 0;
}

bool AIContext::hasStatus2(uint8_t battlerId, uint32_t statusMask) {
     uint8_t id = getBattler(battlerId);
     // ActiveMon has 'volatileStatus'? 
     // I need to check ActiveMon struct in types.hpp again.
     // Assuming some volatile status field exists.
     // For now placeholder.
     return false; 
}

bool AIContext::hasStatus3(uint8_t battlerId, uint32_t statusMask) {
    return false; // placeholder
}

bool AIContext::hasSideStatus(uint8_t side, uint32_t statusMask) {
    // Side status checks
    return false; // placeholder
}

bool AIContext::isMove(uint16_t move) {
    return aiThinking.moveConsidered == move;
}

bool AIContext::hasMove(uint8_t battlerId, uint16_t move) {
    uint8_t id = getBattler(battlerId);
    const auto& mon = engine.getState().getActivePokemon(id);
    // Need to check moves known (which are in the Pokemon struct, not ActiveBattler?)
    // ActiveBattler has 'moves' array?
    // Let's check types.hpp later.
    // Assuming yes:
    // for (int i=0; i<4; ++i) if (mon.moves[i] == move) return true;
    return false;
}

bool AIContext::hasMoveWithEffect(uint8_t battlerId, uint16_t effect) {
    // iterate moves, check MoveData
    return false; 
}

// Type Effectiveness
bool AIContext::typeEffectivenessEquals(int effectiveness) {
    // Calculate effectiveness of considered move against target
    // 1. Get Move Type
    const auto& moveData = getMoveData(aiThinking.moveConsidered);
    Type moveType = moveData.type;
    
    // 2. Get Target Types
    const auto& target = engine.getState().getActivePokemon(battlerTarget);
    SpeciesData spec = getSpeciesData(target.species); // Wait, stored in ActiveMon?
    // ActiveBattler usually stores types (because of Color Change/Conversion)
    // Assuming engine.getBattlerTypes(target) or similar.
    // For now use species data.
    
    float mult = engine.getTypeEffectiveness(moveType, spec.type1, spec.type2);
    
    // Convert float multiplier to AI_EFFECTIVENESS constant
    // x4 = 160, x2 = 80, x1 = 40, x0.5 = 20, x0.25 = 10, x0 = 0
    int aiEff = 40;
    if (mult >= 4.0) aiEff = 160;
    else if (mult >= 2.0) aiEff = 80;
    else if (mult >= 1.0) aiEff = 40;
    else if (mult >= 0.5) aiEff = 20;
    else if (mult >= 0.25) aiEff = 10;
    else aiEff = 0;
    
    return aiEff == effectiveness;
}

bool AIContext::statLevelLessThan(uint8_t battlerId, uint8_t stat, uint8_t val) {
    uint8_t id = getBattler(battlerId);
    const auto& mon = engine.getState().active[id];
    // statStages are int8_t (-6 to +6). 
    // val is usually 6 + stage (0-12)?
    // Decomp constants: DEFAULT_STAT_STAGE = 6.
    // So usually comparison is against 6-based index.
    return (mon.statStages[stat] + 6) < val;
}

bool AIContext::statLevelMoreThan(uint8_t battlerId, uint8_t stat, uint8_t val) {
    uint8_t id = getBattler(battlerId);
    const auto& mon = engine.getState().active[id];
    return (mon.statStages[stat] + 6) > val;
}

bool AIContext::statLevelEqual(uint8_t battlerId, uint8_t stat, uint8_t val) {
    uint8_t id = getBattler(battlerId);
    const auto& mon = engine.getState().active[id];
    return (mon.statStages[stat] + 6) == val;
}

uint8_t AIContext::getAbility(uint8_t battlerId) {
    uint8_t id = getBattler(battlerId);
    const auto& mon = engine.getState().getActivePokemon(id);
    return mon.ability;
}

bool AIContext::hasAbility(uint8_t battlerId, uint8_t ability) {
    return getAbility(battlerId) == ability;
}

bool AIContext::isEffect(uint16_t effect) {
    const auto& moveData = getMoveData(aiThinking.moveConsidered);
    // Need to cast MoveEffect enum to uint16 match or handled by parser
    return static_cast<uint16_t>(moveData.effect) == effect;
}

void AIContext::checkMostPowerfulMove() {
    // See Cmd_get_how_powerful_move_is implementation plan
    // Iterate all moves of user
    // Calculate damage for each
    // Compare with current considered move logic
    // Set funcResult
    
    // Placeholder implementation
    int consideredDmg = engine.calculateDamage(battlerAI, battlerTarget, aiThinking.moveConsidered);
    int maxDmg = 0;
    
    // Find max damage among all usable moves
    // ...
    
    if (consideredDmg >= maxDmg && maxDmg > 0) 
        aiThinking.funcResult = MOVE_MOST_POWERFUL;
    else 
        aiThinking.funcResult = MOVE_NOT_MOST_POWERFUL;
}

} // namespace pkmn

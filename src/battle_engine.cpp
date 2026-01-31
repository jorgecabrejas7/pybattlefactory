#include "battle_engine.hpp"
#include "ai.hpp"
#include "data.hpp"
#include "constants.hpp"
#include <algorithm>

namespace pkmn {

// ============================================================================
// RNG (same LCG as the game)
// ============================================================================

uint16_t BattleEngine::random() {
    // Same formula as pokeemerald: gRngValue = 1103515245 * gRngValue + 24691
    m_state.rngState = 1103515245 * m_state.rngState + 24691;
    return m_state.rngState >> 16;
}

uint16_t BattleEngine::randomRange(uint16_t max) {
    return random() % max;
}

// ============================================================================
// Damage Calculation (Gen 3 formula)
// ============================================================================

int BattleEngine::calculateDamage(uint8_t attackerSide, uint8_t defenderSide, uint16_t moveId) {
    const Pokemon& attacker = m_state.getActivePokemon(attackerSide);
    const Pokemon& defender = m_state.getActivePokemon(defenderSide);
    const ActiveMon& attackerActive = m_state.active[attackerSide];
    const ActiveMon& defenderActive = m_state.active[defenderSide];
    const MoveData& move = getMoveData(moveId);
    
    if (move.power == 0) return 0;  // Status move
    
    // Get attack and defense stats
    int attackStat, defenseStat;
    int attackStage, defenseStage;
    
    if (move.isPhysical) {
        attackStat = attacker.stats[Stat::Attack];
        defenseStat = defender.stats[Stat::Defense];
        attackStage = attackerActive.statStages[BattleStat::ATK];
        defenseStage = defenderActive.statStages[BattleStat::DEF];
    } else {
        attackStat = attacker.stats[Stat::SpAttack];
        defenseStat = defender.stats[Stat::SpDefense];
        attackStage = attackerActive.statStages[BattleStat::SPA];
        defenseStage = defenderActive.statStages[BattleStat::SPD];
    }
    
    // Apply stat stages
    int atkStageIdx = attackStage + 6;  // Convert -6..+6 to 0..12
    int defStageIdx = defenseStage + 6;
    
    attackStat = attackStat * STAT_STAGE_NUMERATORS[atkStageIdx] / STAT_STAGE_DENOMINATORS[atkStageIdx];
    defenseStat = defenseStat * STAT_STAGE_NUMERATORS[defStageIdx] / STAT_STAGE_DENOMINATORS[defStageIdx];
    
    // Base damage formula: ((2 * level / 5 + 2) * power * attack / defense / 50 + 2)
    int level = attacker.level;
    int damage = (2 * level / 5 + 2) * move.power * attackStat / defenseStat / 50 + 2;
    
    // Weather modifiers (simplified)
    // TODO: Full weather implementation
    
    // Critical hit check
    bool isCrit = false;
    int critStage = 0;  // TODO: Track crit stage from moves like Focus Energy
    if (move.effect == MoveEffect::HIGH_CRITICAL) critStage++;
    critStage = std::min(critStage, 4);
    
    if (randomRange(CRIT_CHANCE_DENOMINATORS[critStage]) < CRIT_CHANCE_NUMERATORS[critStage]) {
        isCrit = true;
        damage = damage * 2;  // Gen 3: 2x multiplier
    }
    
    // Random factor (85-100%)
    int randFactor = 85 + randomRange(16);  // 85 to 100
    damage = damage * randFactor / 100;
    
    // STAB (Same Type Attack Bonus)
    const SpeciesData& attackerSpecies = getSpeciesData(attacker.species);
    if (move.type == attackerSpecies.type1 || move.type == attackerSpecies.type2) {
        damage = damage * 150 / 100;  // 1.5x
    }
    
    // Type effectiveness
    const SpeciesData& defenderSpecies = getSpeciesData(defender.species);
    Type defType1 = defenderSpecies.type1;
    Type defType2 = defenderSpecies.type2;
    
    // Check for type overrides (from moves like Conversion)
    if (defenderActive.typesOverridden) {
        defType1 = defenderActive.types[0];
        defType2 = defenderActive.types[1];
    }
    
    int typeEff = getTypeEffectivenessDual(move.type, defType1, defType2);
    damage = damage * typeEff / 100;
    
    // Minimum 1 damage if move would do damage
    if (damage == 0 && typeEff > 0) damage = 1;
    
    return damage;
}

// ============================================================================
// Type Effectiveness
// ============================================================================

float BattleEngine::getTypeEffectiveness(Type attackType, Type defType1, Type defType2) {
    int eff = getTypeEffectivenessDual(attackType, defType1, defType2);
    return eff / 100.0f;
}

// ============================================================================
// Constructor / Setup
// ============================================================================

BattleEngine::BattleEngine() {
    reset(0);
}

BattleEngine::~BattleEngine() = default;

void BattleEngine::reset(uint32_t seed) {
    m_state = BattleState{};
    m_state.rngState = seed;
    m_state.turnNumber = 0;
    m_state.weather = Weather::None;
    m_state.weatherTurns = 0;
    
    for (int i = 0; i < 2; i++) {
        m_state.teamSizes[i] = 0;
        m_state.active[i].reset();
        m_state.sides[i] = SideState{};
    }
}

void BattleEngine::setPlayerTeam(const Pokemon* mons, uint8_t count) {
    count = std::min(count, static_cast<uint8_t>(MAX_PARTY_SIZE));
    m_state.teamSizes[0] = count;
    for (uint8_t i = 0; i < count; i++) {
        m_state.teams[0][i] = mons[i];
    }
    m_state.active[0].partyIndex = 0;
    m_state.active[0].reset();
}

void BattleEngine::setOpponentTeam(const Pokemon* mons, uint8_t count) {
    count = std::min(count, static_cast<uint8_t>(MAX_PARTY_SIZE));
    m_state.teamSizes[1] = count;
    for (uint8_t i = 0; i < count; i++) {
        m_state.teams[1][i] = mons[i];
    }
    m_state.active[1].partyIndex = 0;
    m_state.active[1].reset();
}

// ============================================================================
// Legal Actions
// ============================================================================

std::vector<Action> BattleEngine::getLegalActions() const {
    std::vector<Action> actions;
    const Pokemon& active = m_state.getActivePokemon(0);
    
    // Check moves
    bool hasUsableMove = false;
    for (int i = 0; i < MAX_MOVES; i++) {
        if (active.moves[i] != MOVE_NONE && active.pp[i] > 0) {
            actions.push_back(Action{static_cast<ActionType>(i)});
            hasUsableMove = true;
        }
    }
    
    // If no usable moves, Struggle is the only option
    if (!hasUsableMove) {
        actions.push_back(Action{ActionType::Struggle});
        return actions;
    }
    
    // Check switches
    for (uint8_t i = 0; i < m_state.teamSizes[0]; i++) {
        if (i != m_state.active[0].partyIndex && m_state.teams[0][i].currentHP > 0) {
            actions.push_back(Action{static_cast<ActionType>(static_cast<int>(ActionType::Switch1) + i)});
        }
    }
    
    return actions;
}

// ============================================================================
// Turn Execution
// ============================================================================

BattleEngine::TurnOrder BattleEngine::determineTurnOrder(Action playerAction, Action opponentAction) {
    TurnOrder order;
    
    int playerPriority = 0, opponentPriority = 0;
    int playerSpeed = m_state.getActivePokemon(0).stats[Stat::Speed];
    int opponentSpeed = m_state.getActivePokemon(1).stats[Stat::Speed];
    
    // Switches always go first (priority +6 equivalent)
    if (playerAction.isSwitch()) playerPriority = 6;
    else if (playerAction.isMove()) {
        uint16_t moveId = playerAction.type == ActionType::Struggle ? 
            MOVE_STRUGGLE : m_state.getActivePokemon(0).moves[playerAction.getMoveIndex()];
        if (moveId != MOVE_NONE) {
            playerPriority = getMoveData(moveId).priority;
        }
    }
    
    if (opponentAction.isSwitch()) opponentPriority = 6;
    else if (opponentAction.isMove()) {
        uint16_t moveId = opponentAction.type == ActionType::Struggle ?
            MOVE_STRUGGLE : m_state.getActivePokemon(1).moves[opponentAction.getMoveIndex()];
        if (moveId != MOVE_NONE) {
            opponentPriority = getMoveData(moveId).priority;
        }
    }
    
    // Apply speed stat stages
    int playerStage = m_state.active[0].statStages[BattleStat::SPE] + 6;
    int opponentStage = m_state.active[1].statStages[BattleStat::SPE] + 6;
    playerSpeed = playerSpeed * STAT_STAGE_NUMERATORS[playerStage] / STAT_STAGE_DENOMINATORS[playerStage];
    opponentSpeed = opponentSpeed * STAT_STAGE_NUMERATORS[opponentStage] / STAT_STAGE_DENOMINATORS[opponentStage];
    
    // Determine order
    bool playerFirst;
    if (playerPriority != opponentPriority) {
        playerFirst = playerPriority > opponentPriority;
    } else if (playerSpeed != opponentSpeed) {
        playerFirst = playerSpeed > opponentSpeed;
    } else {
        // Speed tie: 50/50
        playerFirst = randomRange(2) == 0;
    }
    
    if (playerFirst) {
        order.first = 0;
        order.second = 1;
        order.firstAction = playerAction;
        order.secondAction = opponentAction;
    } else {
        order.first = 1;
        order.second = 0;
        order.firstAction = opponentAction;
        order.secondAction = playerAction;
    }
    
    return order;
}

void BattleEngine::executeSwitch(uint8_t side, uint8_t newPartyIndex) {
    // Clear volatile status
    m_state.active[side].reset();
    m_state.active[side].partyIndex = newPartyIndex;
    
    // TODO: Entry hazards (Spikes, Stealth Rock)
}

void BattleEngine::executeMove(uint8_t attackerSide, uint8_t defenderSide, uint16_t moveId) {
    Pokemon& attacker = m_state.getActivePokemon(attackerSide);
    Pokemon& defender = m_state.getActivePokemon(defenderSide);
    const MoveData& move = getMoveData(moveId);
    
    // Deduct PP
    for (int i = 0; i < MAX_MOVES; i++) {
        if (attacker.moves[i] == moveId && attacker.pp[i] > 0) {
            attacker.pp[i]--;
            break;
        }
    }
    
    // Accuracy check
    if (move.accuracy > 0) {
        int accStage = m_state.active[attackerSide].statStages[BattleStat::ACC] -
                       m_state.active[defenderSide].statStages[BattleStat::EVA] + 6;
        accStage = std::clamp(accStage, 0, 12);
        
        int accuracy = move.accuracy * ACC_STAGE_NUMERATORS[accStage] / ACC_STAGE_DENOMINATORS[accStage];
        if (randomRange(100) >= accuracy) {
            return;  // Miss!
        }
    }
    
    // Calculate and apply damage
    if (move.power > 0) {
        int damage = calculateDamage(attackerSide, defenderSide, moveId);
        defender.currentHP = std::max(0, static_cast<int>(defender.currentHP) - damage);
        
        // Handle Recoil
        if (move.effect == MoveEffect::RECOIL || move.effect == MoveEffect::DOUBLE_EDGE) {
            int recoil = damage / 4;
            if (recoil == 0 && damage > 0) recoil = 1;
            attacker.currentHP = std::max(0, static_cast<int>(attacker.currentHP) - recoil);
        }
    }
    
    // TODO: Apply secondary effects based on move.effect (Status, etc.)
}

void BattleEngine::applyEndOfTurnEffects() {
    // Weather damage
    if (m_state.weather == Weather::Sandstorm || m_state.weather == Weather::Hail) {
        for (int side = 0; side < 2; side++) {
            Pokemon& mon = m_state.getActivePokemon(side);
            const SpeciesData& species = getSpeciesData(mon.species);
            
            bool immune = false;
            if (m_state.weather == Weather::Sandstorm) {
                immune = species.type1 == Type::Rock || species.type2 == Type::Rock ||
                         species.type1 == Type::Ground || species.type2 == Type::Ground ||
                         species.type1 == Type::Steel || species.type2 == Type::Steel;
            } else {
                immune = species.type1 == Type::Ice || species.type2 == Type::Ice;
            }
            
            if (!immune) {
                int damage = mon.maxHP / 16;
                if (damage == 0) damage = 1;
                mon.currentHP = std::max(0, static_cast<int>(mon.currentHP) - damage);
            }
        }
    }
    
    // Status damage (burn, poison)
    for (int side = 0; side < 2; side++) {
        Pokemon& mon = m_state.getActivePokemon(side);
        
        if (mon.status == Status::Burn || mon.status == Status::Poison) {
            int damage = mon.maxHP / 8;
            if (damage == 0) damage = 1;
            mon.currentHP = std::max(0, static_cast<int>(mon.currentHP) - damage);
        } else if (mon.status == Status::BadPoison) {
            // TODO: Track toxic counter for escalating damage
            int damage = mon.maxHP / 16;
            if (damage == 0) damage = 1;
            mon.currentHP = std::max(0, static_cast<int>(mon.currentHP) - damage);
        }
    }
    
    // Leftovers recovery
    for (int side = 0; side < 2; side++) {
        Pokemon& mon = m_state.getActivePokemon(side);
        if (mon.heldItem == ITEM_LEFTOVERS && mon.currentHP > 0) {
            int heal = mon.maxHP / 16;
            if (heal == 0) heal = 1;
            mon.currentHP = std::min(static_cast<int>(mon.maxHP), static_cast<int>(mon.currentHP) + heal);
        }
    }
    
    // Decrement weather turns
    if (m_state.weatherTurns > 0) {
        m_state.weatherTurns--;
        if (m_state.weatherTurns == 0) {
            m_state.weather = Weather::None;
        }
    }
}

void BattleEngine::executeTurn(Action playerAction, Action opponentAction) {
    TurnOrder order = determineTurnOrder(playerAction, opponentAction);
    
    // First action
    if (order.firstAction.isSwitch()) {
        executeSwitch(order.first, order.firstAction.getSwitchTarget());
    } else if (order.firstAction.isMove()) {
        uint16_t moveId = order.firstAction.type == ActionType::Struggle ?
            MOVE_STRUGGLE : m_state.getActivePokemon(order.first).moves[order.firstAction.getMoveIndex()];
        executeMove(order.first, 1 - order.first, moveId);
    }
    
    // Check if defender fainted
    if (!m_state.isTerminal() && m_state.getActivePokemon(order.second).currentHP > 0) {
        // Second action
        if (order.secondAction.isSwitch()) {
            executeSwitch(order.second, order.secondAction.getSwitchTarget());
        } else if (order.secondAction.isMove()) {
            uint16_t moveId = order.secondAction.type == ActionType::Struggle ?
                MOVE_STRUGGLE : m_state.getActivePokemon(order.second).moves[order.secondAction.getMoveIndex()];
            executeMove(order.second, 1 - order.second, moveId);
        }
    }
    
    // End of turn effects
    if (!m_state.isTerminal()) {
        applyEndOfTurnEffects();
    }
    
    m_state.turnNumber++;
}

StepResult BattleEngine::step(Action playerAction) {
    // Get AI action for opponent (battler 1)
    Action opponentAction = chooseAIAction(*this, 1);
    
    executeTurn(playerAction, opponentAction);
    
    StepResult result;
    result.done = m_state.isTerminal();
    result.winner = m_state.getWinner();
    result.reward = result.done ? (result.winner == 0 ? 1.0f : -1.0f) : 0.0f;
    
    return result;
}

// ============================================================================
// Vectorized Environment
// ============================================================================

VecBattleEnv::VecBattleEnv(size_t numEnvs) : m_envs(numEnvs) {}

void VecBattleEnv::reset(const uint32_t* seeds, size_t count) {
    for (size_t i = 0; i < m_envs.size() && i < count; i++) {
        m_envs[i].reset(seeds[i]);
    }
}

void VecBattleEnv::step(const Action* actions, float* rewards, bool* dones, size_t count) {
    size_t n = std::min(m_envs.size(), count);
    for (size_t i = 0; i < n; i++) {
        StepResult result = m_envs[i].step(actions[i]);
        rewards[i] = result.reward;
        dones[i] = result.done;
    }
}

void VecBattleEnv::setPlayerTeam(size_t idx, const Pokemon* mons, uint8_t count) {
    if (idx < m_envs.size()) {
        m_envs[idx].setPlayerTeam(mons, count);
    }
}

void VecBattleEnv::setOpponentTeam(size_t idx, const Pokemon* mons, uint8_t count) {
    if (idx < m_envs.size()) {
        m_envs[idx].setOpponentTeam(mons, count);
    }
}

}  // namespace pkmn

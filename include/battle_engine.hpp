#pragma once

#include "types.hpp"
#include <vector>
#include <memory>

namespace pkmn {

// Forward declarations
class AIScriptInterpreter;

// ============================================================================
// Battle Engine - Main simulator class
// ============================================================================
class BattleEngine {
public:
    BattleEngine();
    ~BattleEngine();
    
    // ========================================================================
    // Setup
    // ========================================================================
    
    /// Reset engine to initial state with given RNG seed
    void reset(uint32_t seed);
    
    /// Set up teams (Factory-style: 3 mons each in singles)
    void setPlayerTeam(const Pokemon* mons, uint8_t count);
    void setOpponentTeam(const Pokemon* mons, uint8_t count);
    
    // ========================================================================
    // RL Interface
    // ========================================================================
    
    /// Get current battle state (for observation)
    const BattleState& getState() const { return m_state; }
    
    /// Get legal actions for player
    std::vector<Action> getLegalActions() const;
    
    /// Execute one turn: player takes action, AI responds
    /// Returns reward and done flag
    StepResult step(Action playerAction);
    
    /// Check if battle is over
    bool isTerminal() const { return m_state.isTerminal(); }
    
    /// Get winner (0 = player, 1 = opponent)
    int getWinner() const { return m_state.getWinner(); }
    
    // ========================================================================
    // Internals (public for testing)
    // ========================================================================
    
    /// RNG: returns 0-65535
    uint16_t random();
    
    /// RNG: returns 0 to max-1
    uint16_t randomRange(uint16_t max);
    
    /// Calculate damage for a move
    int calculateDamage(uint8_t attacker, uint8_t defender, uint16_t moveId);
    
    /// Get type effectiveness multiplier (0, 0.25, 0.5, 1, 2, 4)
    float getTypeEffectiveness(Type attackType, Type defType1, Type defType2);
    
    /// Get current turn count
    uint16_t getTurnCount() const { return m_state.turnNumber; }
    
private:
    BattleState m_state;
    
    // Internal turn execution
    void executeTurn(Action playerAction, Action opponentAction);
    void executeMove(uint8_t attacker, uint8_t defender, uint16_t moveId);
    void executeSwitch(uint8_t side, uint8_t newPartyIndex);
    void applyEndOfTurnEffects();
    
    // Determine turn order
    struct TurnOrder {
        uint8_t first, second;
        Action firstAction, secondAction;
    };
    TurnOrder determineTurnOrder(Action playerAction, Action opponentAction);
};

// ============================================================================
// Vectorized Environment for RL Training
// ============================================================================
class VecBattleEnv {
public:
    explicit VecBattleEnv(size_t numEnvs);
    
    /// Reset all environments with given seeds
    void reset(const uint32_t* seeds, size_t count);
    
    /// Step all environments
    /// Actions, observations, rewards, dones must be pre-allocated
    void step(const Action* actions, float* rewards, bool* dones, size_t count);
    
    /// Set teams for a specific environment
    void setPlayerTeam(size_t idx, const Pokemon* mons, uint8_t count);
    void setOpponentTeam(size_t idx, const Pokemon* mons, uint8_t count);
    
    /// Get state for a specific environment
    const BattleState& getState(size_t idx) const { return m_envs[idx].getState(); }
    
    /// Get legal actions for a specific environment
    std::vector<Action> getLegalActions(size_t idx) const { return m_envs[idx].getLegalActions(); }
    
    size_t size() const { return m_envs.size(); }
    
private:
    std::vector<BattleEngine> m_envs;
};

}  // namespace pkmn

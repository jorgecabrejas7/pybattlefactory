#include "ai.hpp"
#include "ai_context.hpp"
#include <vector>
#include <algorithm>
#include <iostream>

namespace pkmn {

Action chooseAIAction(BattleEngine& engine, uint8_t battlerID) {
    // 1. Setup Context
    // Target is opponent (singles only for now).
    // If battlerID=0 (Player), target=1. If 1, target=0.
    uint8_t targetID = (battlerID == 0) ? 1 : 0;
    AIContext ctx(engine, battlerID, targetID);
    
    // 2. Initialize scores
    const auto& mon = engine.getState().getActivePokemon(battlerID);
    
    std::vector<uint8_t> validMoves;
    for (int i=0; i<4; ++i) {
        if (mon.moves[i] != 0 && mon.pp[i] > 0) {
            ctx.aiThinking.score[i] = 100;
            validMoves.push_back(i);
        } else {
            ctx.aiThinking.score[i] = 0;
        }
    }
    
    // If no moves, Struggle (handled by returning Struggle action or let engine handle it)
    if (validMoves.empty()) {
        Action a; 
        a.type = ActionType::Struggle; // Should be handled by engine if index is special?
        // ActionType enum has Struggle? Yes.
        return a;
    }
    
    // 3. Run Scripts
    // AI Flags: CheckBadMove | Viability | TryToFaint | SetupFirstTurn
    // Logic IDs:
    // 0: AI_CheckBadMove
    // 1: AI_TryToFaint
    // 2: AI_Viability
    // 3: AI_SetupFirstTurn
    
    bool runCheckBadMove = true;
    bool runViability = true;
    bool runTryToFaint = true;
    bool runSetupFirstTurn = true;
    
    // Phase 1: Check Bad Move (Script 0)
    if (runCheckBadMove) {
        for (uint8_t i : validMoves) {
            if (ctx.aiThinking.score[i] == 0) continue;
            
            ctx.aiThinking.movesetIndex = i;
            ctx.aiThinking.moveConsidered = mon.moves[i];
            ctx.execute(0);
        }
    }
    
    // Phase 2: Viability (Script 2)
    if (runViability) {
        for (uint8_t i : validMoves) {
            if (ctx.aiThinking.score[i] == 0) continue;
            
            ctx.aiThinking.movesetIndex = i;
            ctx.aiThinking.moveConsidered = mon.moves[i];
            ctx.execute(2);
        }
    }
    
    // Phase 3: Try To Faint (Script 1)
    if (runTryToFaint) {
        for (uint8_t i : validMoves) {
            if (ctx.aiThinking.score[i] == 0) continue;
            
            ctx.aiThinking.movesetIndex = i;
            ctx.aiThinking.moveConsidered = mon.moves[i];
            ctx.execute(1);
        }
    }
    
    // Phase 4: Setup First Turn (Script 3)
    if (runSetupFirstTurn) {
         for (uint8_t i : validMoves) {
            if (ctx.aiThinking.score[i] == 0) continue;
            
            ctx.aiThinking.movesetIndex = i;
            ctx.aiThinking.moveConsidered = mon.moves[i];
            ctx.execute(3);
        }
    }

    // 4. Pick best move
    int bestScore = -1;
    
    for (uint8_t i : validMoves) {
        if (ctx.aiThinking.score[i] > bestScore) {
            bestScore = ctx.aiThinking.score[i];
        }
    }
    
    // Collect ties
    std::vector<int> ties;
    for (uint8_t i : validMoves) {
        if (ctx.aiThinking.score[i] == bestScore) ties.push_back(i);
    }
    
    int bestMoveIdx = -1;
    if (!ties.empty()) {
        // Deterministic random from engine?
        // engine.random() returns uint32_t
        bestMoveIdx = ties[engine.random() % ties.size()];
    }
    
    Action action;
    if (bestMoveIdx >= 0) {
        action.type = static_cast<ActionType>((int)ActionType::Move1 + bestMoveIdx);
    } else {
        action.type = ActionType::Struggle;
    }
    
    return action;
}

} // namespace pkmn

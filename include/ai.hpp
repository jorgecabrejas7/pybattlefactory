#pragma once
#include "battle_engine.hpp"

namespace pkmn {

// Main AI entry point
Action chooseAIAction(BattleEngine& engine, uint8_t battlerID);

} // namespace pkmn

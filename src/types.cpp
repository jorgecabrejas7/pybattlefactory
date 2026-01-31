#include "types.hpp"
#include "data.hpp"
#include "constants.hpp"

namespace pkmn {

// Calculate Pokemon stats from base stats, IVs, EVs, nature
void Pokemon::calculateStats(const SpeciesData& base) {
    // HP formula: floor((2*base + IV + EV/4) * level/100) + level + 10
    int hpCalc = (2 * base.baseHP + ivs[HP] + evs[HP] / 4) * level / 100 + level + 10;
    stats[HP] = static_cast<uint16_t>(hpCalc);
    maxHP = stats[HP];
    if (currentHP == 0) currentHP = maxHP;  // Initialize if not set

    // Other stats: floor((2*base + IV + EV/4) * level/100 + 5) * nature
    const uint8_t* bases = &base.baseAttack;  // Attack, Defense, Speed, SpAttack, SpDefense are contiguous
    for (int i = 1; i < 6; i++) {
        int baseStat = bases[i - 1];
        int calc = (2 * baseStat + ivs[i] + evs[i] / 4) * level / 100 + 5;
        int natureMod = getNatureModifier(nature, static_cast<Stat>(i));
        stats[i] = static_cast<uint16_t>(calc * natureMod / 100);
    }
}

void ActiveMon::reset() {
    for (int i = 0; i < BATTLE_STAT_COUNT; i++) {
        statStages[i] = 0;
    }
    isConfused = false;
    confusionTurns = 0;
    isTaunted = false;
    tauntTurns = 0;
    isSeeded = false;
    hasSubstitute = false;
    substituteHP = 0;
    protectUses = 0;
    protectedThisTurn = false;
    typesOverridden = false;
}

uint8_t BattleState::countRemaining(uint8_t side) const {
    uint8_t count = 0;
    for (uint8_t i = 0; i < teamSizes[side]; i++) {
        if (teams[side][i].currentHP > 0) count++;
    }
    return count;
}

bool BattleState::isTerminal() const {
    return countRemaining(0) == 0 || countRemaining(1) == 0;
}

int BattleState::getWinner() const {
    if (!isTerminal()) return -1;
    return countRemaining(0) == 0 ? 1 : 0;
}

}  // namespace pkmn

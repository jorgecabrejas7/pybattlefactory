#ifndef GEN3_SEARCH_HOST_H
#define GEN3_SEARCH_HOST_H

// Search support for the gen3 battle core (src/gen3/search_host.c): battle outcome and
// determinization of the opponent's hidden information. Usable from C and C++.

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct Gen3DetSlot {
    int32_t partySlot;      // gEnemyParty index 0-2
    int32_t setId;          // gBattleFrontierMons index, or -1: keep this slot as it is
    int32_t iv;             // fixed IV for all six stats (0-31)
    int32_t abilityBit;     // personality & 1 (only matters when the species has two abilities)
    float hpFraction;       // (0,1]: HP = round(maxHP * f) (min 1); < 0: keep the current HP fraction
};

uint8_t Gen3_BattleOutcome(void);   // gBattleOutcome (0 while the battle runs)

// Rebuild gEnemyParty[slot] for each spec as FillFactoryFrontierTrainerParty would, keeping status
// and HP (see hpFraction); patches gBattleMons[1] when the slot is the enemy's active battler.
// hiddenSeed >= 0 also resamples the hidden sleep and confusion counters of both sides.
// Returns 0 on success, -1 on a bad spec (nothing is changed then).
int Gen3Search_Determinize(const struct Gen3DetSlot *slots, int count, int64_t hiddenSeed);

#ifdef __cplusplus
}
#endif

#endif

#ifndef GEN3_OBSERVER_HOST_H
#define GEN3_OBSERVER_HOST_H

// Battle RAM the C++ observer reads at a decision (src/gen3/observer_host.c), and the game's static
// tables it uses. Usable from C and C++.
//
// Structures that the Python observer reads as bytes by GBA address (pybattle/view.py _Snap) are
// copied here as the same bytes (the host keeps the GBA layout of battle RAM, see
// generated/rom_layout_check.c), so both decode them with the same offsets. Party Pokemon are
// decrypted here (pybattle/emu/decode.py decode_pokemon); only the first 3 of each party are read,
// like _Snap.

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GEN3_OBS_NUM_SPECIES 412
#define GEN3_OBS_MOVES_COUNT 355
#define GEN3_OBS_ITEMS_COUNT 377
#define GEN3_OBS_TYPE_CHART 336

struct Gen3ObsPartyMon {
    uint32_t status;
    uint16_t species, item, moves[4], hp, maxHp, stats[5];   // stats: Atk Def Spe SpA SpD
    uint8_t pp[4], abilityNum, level;
};

struct Gen3ObsRaw {
    uint8_t mons[2][0x58];          // gBattleMons[0..1] (struct BattlePokemon)
    uint8_t idx[4];                 // gBattlerPartyIndexes[0..1] (u16 x2)
    struct Gen3ObsPartyMon party[2][3];
    uint8_t dis[2][0x1C];           // gDisableStructs[0..1]
    uint32_t status3[2];
    uint16_t sides[2];              // gSideStatuses
    uint8_t timers[24];             // gSideTimers
    uint16_t weather;
    uint8_t wfk[44];                // gWishFutureKnock
    uint8_t lastMoves[4];           // gLastMoves[0..1] (u16 x2)
    uint16_t lastPrinted[2];
    int32_t bide[2];                // gBideDmg
    uint8_t crit;                   // gCritMultiplier
    uint16_t locked[2];             // gLockedMoves
    uint8_t results[0x26];          // gBattleResults
    uint16_t rndTurn;               // gRandomTurnNumber
    uint8_t order[2];               // gBattlerByTurnOrder
};

// Copy the live game's battle RAM (the Gen3Game must be the active one).
void Gen3Obs_Read(struct Gen3ObsRaw *out);

struct Gen3ObsSpecies { uint8_t base[6], types[2], abilities[2]; };
struct Gen3ObsMove { uint8_t effect, power, type, accuracy, pp, secondaryChance, flags; int8_t priority; };
struct Gen3ObsItem { uint8_t holdEffect, holdEffectParam; };

struct Gen3ObsTables {
    struct Gen3ObsSpecies species[GEN3_OBS_NUM_SPECIES];
    struct Gen3ObsMove moves[GEN3_OBS_MOVES_COUNT];
    struct Gen3ObsItem items[GEN3_OBS_ITEMS_COUNT];
    uint8_t typeChart[GEN3_OBS_TYPE_CHART];      // gTypeEffectiveness: (attacker, defender, x10) triples
    uint8_t statStageRatios[13][2];              // gStatStageRatios
    int abilitiesCount, effectsCount;            // ABILITIES_COUNT, NUM_BATTLE_MOVE_EFFECTS
};

// The game's static tables (read-only data, independent of any game state).
const struct Gen3ObsTables *Gen3Obs_Tables(void);

#ifdef __cplusplus
}
#endif

#endif

// Battle RAM and static tables for the C++ observer (src/gen3/observer.cpp, gen3_observer.hpp).

#include "global.h"
#include "battle.h"
#include "battle_main.h"
#include "item.h"
#include "pokemon.h"
#include "random.h"
#include "constants/abilities.h"
#include "constants/battle_move_effects.h"
#include "constants/items.h"
#include "constants/moves.h"
#include "constants/species.h"

#include "gen3/observer_host.h"

#include <string.h>

_Static_assert(sizeof(struct BattlePokemon) == 0x58, "BattlePokemon layout");
_Static_assert(sizeof(struct DisableStruct) == 0x1C, "DisableStruct layout");
_Static_assert(sizeof(gSideTimers) == 24, "SideTimer layout");
_Static_assert(sizeof(struct WishFutureKnock) == 44, "WishFutureKnock layout");
_Static_assert(sizeof(struct BattleResults) >= 0x26, "BattleResults layout");
_Static_assert(sizeof(struct Pokemon) == 100, "Pokemon layout");
_Static_assert(NUM_SPECIES == GEN3_OBS_NUM_SPECIES, "species count");
_Static_assert(MOVES_COUNT == GEN3_OBS_MOVES_COUNT, "moves count");
_Static_assert(ITEMS_COUNT == GEN3_OBS_ITEMS_COUNT, "items count");
_Static_assert(sizeof(gTypeEffectiveness) == GEN3_OBS_TYPE_CHART, "type chart size");

// GetSubstruct (pokemon.c): slot of substruct type t for personality % 24
static const u8 sSubstructSlot[24][4] = {
    {0, 1, 2, 3}, {0, 1, 3, 2}, {0, 2, 1, 3}, {0, 3, 1, 2}, {0, 2, 3, 1}, {0, 3, 2, 1},
    {1, 0, 2, 3}, {1, 0, 3, 2}, {2, 0, 1, 3}, {3, 0, 1, 2}, {2, 0, 3, 1}, {3, 0, 2, 1},
    {1, 2, 0, 3}, {1, 3, 0, 2}, {2, 1, 0, 3}, {3, 1, 0, 2}, {2, 3, 0, 1}, {3, 2, 0, 1},
    {1, 2, 3, 0}, {1, 3, 2, 0}, {2, 1, 3, 0}, {3, 1, 2, 0}, {2, 3, 1, 0}, {3, 2, 1, 0},
};

static u16 Rd16(const u8 *p) { return (u16)(p[0] | (p[1] << 8)); }
static u32 Rd32(const u8 *p) { return (u32)p[0] | ((u32)p[1] << 8) | ((u32)p[2] << 16) | ((u32)p[3] << 24); }

// pybattle/emu/decode.py decode_pokemon, the fields the observer uses
static void DecodeMon(const struct Pokemon *mon, struct Gen3ObsPartyMon *out)
{
    const u8 *raw = (const u8 *)mon;
    u32 personality = Rd32(raw), key = personality ^ Rd32(raw + 4);
    u32 secure[12];
    const u8 *order = sSubstructSlot[personality % 24];
    const u32 *growth, *attacks, *misc;
    int i;

    for (i = 0; i < 12; i++)
        secure[i] = Rd32(raw + 0x20 + 4 * i) ^ key;
    growth = &secure[order[0] * 3];
    attacks = &secure[order[1] * 3];
    misc = &secure[order[3] * 3];
    out->species = (u16)(growth[0] & 0xFFFF);
    out->item = (u16)(growth[0] >> 16);
    for (i = 0; i < 4; i++)
    {
        out->moves[i] = (u16)(attacks[i / 2] >> (16 * (i & 1)));
        out->pp[i] = (u8)(attacks[2] >> (8 * i));
    }
    out->abilityNum = (u8)((misc[1] >> 31) & 1);
    out->status = Rd32(raw + 0x50);
    out->level = raw[0x54];
    out->hp = Rd16(raw + 0x56);
    out->maxHp = Rd16(raw + 0x58);
    for (i = 0; i < 5; i++)
        out->stats[i] = Rd16(raw + 0x5A + 2 * i);
}

void Gen3Obs_Read(struct Gen3ObsRaw *out)
{
    int i;
    memcpy(out->mons, gBattleMons, sizeof(out->mons));
    memcpy(out->idx, gBattlerPartyIndexes, sizeof(out->idx));
    for (i = 0; i < 3; i++)
    {
        DecodeMon(&gPlayerParty[i], &out->party[0][i]);
        DecodeMon(&gEnemyParty[i], &out->party[1][i]);
    }
    memcpy(out->dis, gDisableStructs, sizeof(out->dis));
    out->status3[0] = gStatuses3[0];
    out->status3[1] = gStatuses3[1];
    out->sides[0] = gSideStatuses[0];
    out->sides[1] = gSideStatuses[1];
    memcpy(out->timers, gSideTimers, sizeof(out->timers));
    out->weather = gBattleWeather;
    memcpy(out->wfk, &gWishFutureKnock, sizeof(out->wfk));
    memcpy(out->lastMoves, gLastMoves, sizeof(out->lastMoves));
    out->lastPrinted[0] = gLastPrintedMoves[0];
    out->lastPrinted[1] = gLastPrintedMoves[1];
    out->bide[0] = gBideDmg[0];
    out->bide[1] = gBideDmg[1];
    out->crit = gCritMultiplier;
    out->locked[0] = gLockedMoves[0];
    out->locked[1] = gLockedMoves[1];
    memcpy(out->results, &gBattleResults, sizeof(out->results));
    out->rndTurn = gRandomTurnNumber;
    out->order[0] = gBattlerByTurnOrder[0];
    out->order[1] = gBattlerByTurnOrder[1];
}

static struct Gen3ObsTables sTables;
static int sTablesReady;

const struct Gen3ObsTables *Gen3Obs_Tables(void)
{
    int i;
    if (sTablesReady)
        return &sTables;
    for (i = 0; i < NUM_SPECIES; i++)
    {
        const struct SpeciesInfo *s = &gSpeciesInfo[i];
        struct Gen3ObsSpecies *o = &sTables.species[i];
        o->base[0] = s->baseHP;
        o->base[1] = s->baseAttack;
        o->base[2] = s->baseDefense;
        o->base[3] = s->baseSpeed;
        o->base[4] = s->baseSpAttack;
        o->base[5] = s->baseSpDefense;
        o->types[0] = s->types[0];
        o->types[1] = s->types[1];
        o->abilities[0] = s->abilities[0];
        o->abilities[1] = s->abilities[1];
    }
    for (i = 0; i < MOVES_COUNT; i++)
    {
        const struct BattleMove *m = &gBattleMoves[i];
        struct Gen3ObsMove *o = &sTables.moves[i];
        o->effect = m->effect;
        o->power = m->power;
        o->type = m->type;
        o->accuracy = m->accuracy;
        o->pp = m->pp;
        o->secondaryChance = m->secondaryEffectChance;
        o->flags = m->flags;
        o->priority = m->priority;
    }
    for (i = 0; i < ITEMS_COUNT; i++)
    {
        sTables.items[i].holdEffect = gItems[i].holdEffect;
        sTables.items[i].holdEffectParam = gItems[i].holdEffectParam;
    }
    memcpy(sTables.typeChart, gTypeEffectiveness, sizeof(sTables.typeChart));
    memcpy(sTables.statStageRatios, gStatStageRatios, sizeof(sTables.statStageRatios));
    sTables.abilitiesCount = ABILITIES_COUNT;
    sTables.effectsCount = NUM_BATTLE_MOVE_EFFECTS;
    sTablesReady = 1;
    return &sTables;
}

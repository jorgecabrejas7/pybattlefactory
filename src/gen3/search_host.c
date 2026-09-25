// Search support: battle outcome and determinization of the opponent's hidden information.
//
// Gen3Search_Determinize rebuilds opponent party slots exactly as the game builds Battle Factory
// opponents (FillFactoryFrontierTrainerParty -> CreateMonWithEVSpreadNatureOTID,
// battle_tower.c / pokemon.c), but with a chosen set, IV and ability bit and without touching
// gRngValue, then patches the active battler (gBattleMons[1]) the way the opponent controller
// fills it at switch-in (host_ctrl_opponent.c REQUEST_ALL_BATTLE, battle_main.c).
//
// Hidden state that is NOT resampled:
//   - wrap / uproar / thrash-lock turn counters (status2), Disable and Encore timers
//     (gDisableStructs), which also have "start value" fields; the player can count elapsed turns
//     but the remaining ones are random in the game;
//   - the AI's state (gBattleResources->battleHistory, AI_THINKING_STRUCT): left as is;
//   - gBattleStruct fields that reference the old mon (choicedMove, usedHeldItems,
//     hpOnSwitchout...): left as is; the caller should pick sets consistent with what was seen;
//   - personality-derived details other than nature/ability/gender (shininess, Hidden Power is
//     not used by Factory sets).

#include "global.h"
#include "battle.h"
#include "battle_factory.h"
#include "battle_tower.h"
#include "main.h"
#include "pokemon.h"
#include "random.h"
#include "constants/battle_frontier_mons.h"
#include "constants/items.h"
#include "constants/moves.h"

#include "gen3/search_host.h"

#include <string.h>

u8 Gen3_BattleOutcome(void)
{
    return gBattleOutcome;
}

// splitmix64: a local generator so determinization never touches gRngValue.
static u64 sDetState;

static u32 DetRandom(void)
{
    u64 z = (sDetState += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return (u32)((z ^ (z >> 31)) >> 32);
}

// A personality with the set's nature and the requested ability bit (and gender when asked),
// like the do/while Random32() loop of CreateMonWithEVSpreadNatureOTID.
static u32 PickPersonality(u16 species, u8 nature, u8 abilityBit, s32 gender)
{
    u32 tries;
    for (tries = 0; tries < 1000000; tries++)
    {
        u32 p = DetRandom();
        if (GetNatureFromPersonality(p) != nature)
            continue;
        if (gSpeciesInfo[species].abilities[1] && (p & 1) != abilityBit)
            continue;
        if (gender >= 0 && GetGenderFromSpeciesAndPersonality(species, p) != gender)
            continue;
        return p;
    }
    return nature;   // unreachable in practice
}

// CreateMonWithEVSpreadNatureOTID with a given personality, then the rest of
// FillFactoryFrontierTrainerParty's per-mon setup.
static void BuildFactoryMon(struct Pokemon *mon, u16 setId, u8 level, u8 fixedIV, u32 personality, u32 otId)
{
    const struct FacilityMon *fm = &gBattleFrontierMons[setId];
    s32 i, statCount = 0;
    u8 evsBits;
    u16 evAmount;
    u8 friendship = 0;

    CreateMon(mon, fm->species, level, fixedIV, TRUE, personality, OT_ID_PRESET, otId);
    evsBits = fm->evSpread;
    for (i = 0; i < NUM_STATS; i++)
    {
        if (evsBits & 1)
            statCount++;
        evsBits >>= 1;
    }
    evAmount = statCount ? MAX_TOTAL_EVS / statCount : 0;
    evsBits = 1;
    for (i = 0; i < NUM_STATS; i++)
    {
        if (fm->evSpread & evsBits)
            SetMonData(mon, MON_DATA_HP_EV + i, &evAmount);
        evsBits <<= 1;
    }
    CalculateMonStats(mon);

    for (i = 0; i < MAX_MON_MOVES; i++)
        SetMonMoveAvoidReturn(mon, fm->moves[i], i);
    SetMonData(mon, MON_DATA_FRIENDSHIP, &friendship);
    SetMonData(mon, MON_DATA_HELD_ITEM, &gBattleFrontierHeldItems[fm->itemTableId]);
}

static u16 ScaleHp(u16 newMax, float fraction, u16 oldHp, u16 oldMax)
{
    u32 hp;
    if (fraction > 0)
        hp = (u32)(newMax * fraction + 0.5f);
    else if (oldHp == 0 || oldMax == 0)
        return 0;
    else
        hp = (newMax * (u32)oldHp * 2 + oldMax) / (2 * (u32)oldMax);   // round(newMax * oldHp / oldMax)
    if (hp < 1)
        hp = 1;
    if (hp > newMax)
        hp = newMax;
    return hp;
}

static u32 ResampleSleep(u32 status1)
{
    // 2-5 turns when inflicted (Rest: 3), decremented on each move attempt: 1-5 can remain.
    if (status1 & STATUS1_SLEEP)
        status1 = (status1 & ~STATUS1_SLEEP) | STATUS1_SLEEP_TURN(1 + DetRandom() % 5);
    return status1;
}

static void Determinize(struct Pokemon *mon, const struct Gen3DetSlot *spec, int active)
{
    struct Pokemon old = *mon;
    struct BattlePokemon *bm = &gBattleMons[1];
    const struct FacilityMon *fm = &gBattleFrontierMons[spec->setId];
    u16 oldSpecies = GetMonData(&old, MON_DATA_SPECIES);
    u16 oldMaxHp = GetMonData(&old, MON_DATA_MAX_HP);
    u16 oldHp = GetMonData(&old, MON_DATA_HP);
    u32 status = GetMonData(&old, MON_DATA_STATUS);
    u16 item, hp, newMax;
    s32 gender = -1;
    u32 personality;
    s32 i, j;

    if (active)
    {
        oldHp = bm->hp;
        oldMaxHp = bm->maxHP;
        status = bm->status1;
    }
    if (fm->species == oldSpecies)
        gender = GetMonGender(&old);   // visible: keep it
    personality = PickPersonality(fm->species, fm->nature, spec->abilityBit & 1, gender);
    BuildFactoryMon(mon, spec->setId, GetMonData(&old, MON_DATA_LEVEL), spec->iv, personality,
                    GetMonData(&old, MON_DATA_OT_ID));

    // PP already spent on a move the new set shares is kept
    for (i = 0; i < MAX_MON_MOVES; i++)
    {
        u16 move = GetMonData(mon, MON_DATA_MOVE1 + i);
        for (j = 0; j < MAX_MON_MOVES; j++)
        {
            if (move != MOVE_NONE && GetMonData(&old, MON_DATA_MOVE1 + j) == move)
            {
                u8 pp = GetMonData(&old, MON_DATA_PP1 + j);
                SetMonData(mon, MON_DATA_PP1 + i, &pp);
                break;
            }
        }
    }
    // A consumed or knocked-off item stays gone (Factory sets always hold an item)
    if (GetMonData(&old, MON_DATA_HELD_ITEM) == ITEM_NONE)
    {
        item = ITEM_NONE;
        SetMonData(mon, MON_DATA_HELD_ITEM, &item);
    }
    newMax = GetMonData(mon, MON_DATA_MAX_HP);
    hp = ScaleHp(newMax, spec->hpFraction, oldHp, oldMaxHp);
    SetMonData(mon, MON_DATA_HP, &hp);
    SetMonData(mon, MON_DATA_STATUS, &status);

    if (!active)
        return;

    // gBattleMons[1]: what the controller copies at switch-in, keeping stat stages, status1/2 and
    // everything visible that changed since (Transform, Conversion/Color Change types).
    {
        struct BattlePokemon prev = *bm;
        u8 abilityNum = GetMonData(mon, MON_DATA_ABILITY_NUM);
        if (!(prev.status2 & STATUS2_TRANSFORMED))
        {
            u16 newSpecies = fm->species;
            bm->species = newSpecies;
            bm->attack = GetMonData(mon, MON_DATA_ATK);
            bm->defense = GetMonData(mon, MON_DATA_DEF);
            bm->speed = GetMonData(mon, MON_DATA_SPEED);
            bm->spAttack = GetMonData(mon, MON_DATA_SPATK);
            bm->spDefense = GetMonData(mon, MON_DATA_SPDEF);
            for (i = 0; i < MAX_MON_MOVES; i++)
            {
                bm->moves[i] = GetMonData(mon, MON_DATA_MOVE1 + i);
                bm->pp[i] = GetMonData(mon, MON_DATA_PP1 + i);
                for (j = 0; j < MAX_MON_MOVES; j++)
                {
                    if (bm->moves[i] != MOVE_NONE && prev.moves[j] == bm->moves[i])
                    {
                        bm->pp[i] = prev.pp[j];
                        break;
                    }
                }
            }
            bm->ppBonuses = GetMonData(mon, MON_DATA_PP_BONUSES);
            bm->abilityNum = abilityNum;
            // Trace / Role Play / Skill Swap: the changed ability is visible; keep it
            if (prev.ability == GetAbilityBySpecies(prev.species, prev.abilityNum))
                bm->ability = GetAbilityBySpecies(newSpecies, abilityNum);
            if (prev.types[0] == gSpeciesInfo[prev.species].types[0]
             && prev.types[1] == gSpeciesInfo[prev.species].types[1])
            {
                bm->types[0] = gSpeciesInfo[newSpecies].types[0];
                bm->types[1] = gSpeciesInfo[newSpecies].types[1];
            }
            bm->personality = personality;
            bm->hpIV = spec->iv;
            bm->attackIV = spec->iv;
            bm->defenseIV = spec->iv;
            bm->speedIV = spec->iv;
            bm->spAttackIV = spec->iv;
            bm->spDefenseIV = spec->iv;
            bm->experience = GetMonData(mon, MON_DATA_EXP);
            bm->friendship = GetMonData(mon, MON_DATA_FRIENDSHIP);
            GetMonData(mon, MON_DATA_NICKNAME, bm->nickname);
        }
        bm->item = prev.item == ITEM_NONE ? ITEM_NONE : GetMonData(mon, MON_DATA_HELD_ITEM);
        bm->maxHP = newMax;
        bm->hp = hp;
    }
}

int Gen3Search_Determinize(const struct Gen3DetSlot *slots, int count, int64_t hiddenSeed)
{
    int k, b, i;
    u8 enemyActive = 0xFF;

    for (k = 0; k < count; k++)
    {
        const struct Gen3DetSlot *s = &slots[k];
        if (s->partySlot < 0 || s->partySlot >= FRONTIER_PARTY_SIZE)
            return -1;
        if (s->setId >= NUM_FRONTIER_MONS || (s->setId >= 0 && (s->iv < 0 || s->iv > MAX_PER_STAT_IVS)))
            return -1;
        if (s->setId >= 0 && s->hpFraction > 1.0f)
            return -1;
    }
    if (gMain.inBattle && gBattleMons[1].species != SPECIES_NONE)
        enemyActive = gBattlerPartyIndexes[1];

    sDetState = (u64)(hiddenSeed < 0 ? 0 : hiddenSeed) * 0x2545F4914F6CDD1Dull + 0x1234567ull;
    for (k = 0; k < count; k++)
    {
        const struct Gen3DetSlot *s = &slots[k];
        if (s->setId < 0)
            continue;
        Determinize(&gEnemyParty[s->partySlot], s, s->partySlot == enemyActive);
    }

    if (hiddenSeed < 0)
        return 0;
    // Hidden counters, both sides (the game shows neither player how many turns remain)
    for (i = 0; i < PARTY_SIZE; i++)
    {
        struct Pokemon *parties[2] = {&gPlayerParty[i], &gEnemyParty[i]};
        for (b = 0; b < 2; b++)
        {
            u32 status = GetMonData(parties[b], MON_DATA_STATUS);
            if (GetMonData(parties[b], MON_DATA_SPECIES) != SPECIES_NONE && (status & STATUS1_SLEEP))
            {
                status = ResampleSleep(status);
                SetMonData(parties[b], MON_DATA_STATUS, &status);
            }
        }
    }
    if (gMain.inBattle)
    {
        for (b = 0; b < gBattlersCount; b++)
        {
            struct Pokemon *party = GetBattlerSide(b) == B_SIDE_PLAYER ? gPlayerParty : gEnemyParty;
            if (gBattleMons[b].species == SPECIES_NONE)
                continue;
            if (gBattleMons[b].status1 & STATUS1_SLEEP)
            {
                u32 status = ResampleSleep(gBattleMons[b].status1);
                gBattleMons[b].status1 = status;
                SetMonData(&party[gBattlerPartyIndexes[b]], MON_DATA_STATUS, &status);
            }
            // 2-5 turns when inflicted, decremented on each move attempt: 1-5 can remain
            if (gBattleMons[b].status2 & STATUS2_CONFUSION)
                gBattleMons[b].status2 = (gBattleMons[b].status2 & ~STATUS2_CONFUSION)
                                       | STATUS2_CONFUSION_TURN(1 + DetRandom() % 5);
        }
    }
    return 0;
}

int Gen3Search_HasChoice(uint8_t unusable, int canSwitch)
{
    int i;
    for (i = 0; i < MAX_MON_MOVES; i++)
        if (gBattleMons[0].moves[i] != MOVE_NONE && !((unusable >> i) & 1))
            return 1;
    if (canSwitch)
        for (i = 0; i < 3; i++)
            if (i != gBattlerPartyIndexes[0] && GetMonData(&gPlayerParty[i], MON_DATA_HP, NULL) > 0)
                return 1;
    return 0;
}

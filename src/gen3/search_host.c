// Search support: battle outcome, determinization of the opponent's hidden information and of the
// hidden random state of the turn.
//
// Gen3Search_Determinize rebuilds opponent party slots from explicit specs (species, moves, item,
// IVs, EVs, nature, ability bit, HP) drawn by rl/determinize.py from what a player knows -- never
// from the Battle Factory's set list or the true state -- the way the game builds a Pokemon
// (CreateMon + EVs + CalculateMonStats, pokemon.c), without touching gRngValue, then patches the
// active battler (gBattleMons[1]) the way the opponent controller fills it at switch-in
// (host_ctrl_opponent.c REQUEST_ALL_BATTLE, battle_main.c). With a hidden seed it also resamples
// the hidden counters (see ResampleHiddenCounters).
//
// Gen3Search_RedrawTurn gives a root a fresh RNG seed, redraws this turn's Quick Claw roll and makes
// the opponent's AI choose again (its choice for this turn is made while the player's controller
// waits, so a copy of the live game would otherwise know it).
//
// Hidden state that is NOT resampled (docs/RL_DECISIONS.md §15):
//   - the Bide damage stored for an opponent using Bide (gBideDmg), and this turn's physical /
//     special damage records used by Counter / Mirror Coat (reset every turn);
//   - the AI's memory of the player's side (gBattleResources->battleHistory): what the AI knows is
//     not the player's hidden information;
//   - personality-derived details other than nature / ability / gender (shininess; Hidden Power's
//     type and power follow the drawn IVs);
//   - friendship (0, as the Battle Frontier builds its Pokemon: only Return / Frustration read it).

#include "global.h"
#include "battle.h"
#include "battle_ai_switch_items.h"
#include "battle_controllers.h"
#include "item.h"
#include "main.h"
#include "pokemon.h"
#include "random.h"
#include "constants/abilities.h"
#include "constants/battle.h"
#include "constants/battle_frontier.h"
#include "constants/hold_effects.h"
#include "constants/items.h"
#include "constants/moves.h"

#include "gen3/host.h"
#include "gen3/search_host.h"

#include <string.h>

// HandleTurnActionSelectionState's per-battler states (battle_main.c, a local enum there)
#define SEL_STATE_BEFORE_ACTION_CHOSEN 1
#define SEL_STATE_WAIT_ACTION_CONFIRMED 5

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

// A counter started uniformly in [lo, hi], `elapsed` steps of `step` already taken and still running
// (>= 1 left): the remaining value, drawn from that posterior. Falls back to 1 if nothing fits.
static u32 DrawRemaining(u32 lo, u32 hi, s32 elapsed, u32 step)
{
    u32 options[8], n = 0, s;
    if (elapsed < 0)
        elapsed = 0;
    for (s = lo; s <= hi && n < 8; s++)
        if ((s32)s - elapsed * (s32)step >= 1)
            options[n++] = s - elapsed * step;
    return n ? options[DetRandom() % n] : 1;
}

// A personality with the requested nature and ability bit (and gender when asked),
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

static void BuildMon(struct Pokemon *mon, const struct Gen3DetMon *spec, u8 level, u32 personality, u32 otId)
{
    s32 i;
    u8 friendship = 0;
    u16 item = spec->item;

    CreateMon(mon, spec->species, level, 0, TRUE, personality, OT_ID_PRESET, otId);
    for (i = 0; i < NUM_STATS; i++)
    {
        u8 iv = spec->ivs[i], ev = spec->evs[i];
        SetMonData(mon, MON_DATA_HP_IV + i, &iv);
        SetMonData(mon, MON_DATA_HP_EV + i, &ev);
    }
    CalculateMonStats(mon);
    for (i = 0; i < MAX_MON_MOVES; i++)
        SetMonMoveSlot(mon, spec->moves[i], i);
    SetMonData(mon, MON_DATA_FRIENDSHIP, &friendship);
    SetMonData(mon, MON_DATA_HELD_ITEM, &item);
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

static void Determinize(struct Pokemon *mon, const struct Gen3DetMon *spec, int active)
{
    struct Pokemon old = *mon;
    struct BattlePokemon *bm = &gBattleMons[1];
    u16 oldSpecies = GetMonData(&old, MON_DATA_SPECIES);
    u16 oldMaxHp = GetMonData(&old, MON_DATA_MAX_HP);
    u16 oldHp = GetMonData(&old, MON_DATA_HP);
    u32 status = GetMonData(&old, MON_DATA_STATUS);
    u16 item, hp, newMax, species = spec->species;
    s32 gender = -1;
    u32 personality;
    s32 i, j;

    if (active)
    {
        oldHp = bm->hp;
        oldMaxHp = bm->maxHP;
        status = bm->status1;
    }
    if (species == oldSpecies)
        gender = GetMonGender(&old);   // visible: keep it
    personality = PickPersonality(species, spec->nature, spec->abilityBit & 1, gender);
    BuildMon(mon, spec, GetMonData(&old, MON_DATA_LEVEL), personality, GetMonData(&old, MON_DATA_OT_ID));

    // PP already spent on a move the new set shares is kept (revealed moves: the player counted them)
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
    // A consumed or knocked-off item stays gone (the game announced it)
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
            bm->species = species;
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
                bm->ability = GetAbilityBySpecies(species, abilityNum);
            if (prev.types[0] == gSpeciesInfo[prev.species].types[0]
             && prev.types[1] == gSpeciesInfo[prev.species].types[1])
            {
                bm->types[0] = gSpeciesInfo[species].types[0];
                bm->types[1] = gSpeciesInfo[species].types[1];
            }
            bm->personality = personality;
            bm->hpIV = spec->ivs[0];
            bm->attackIV = spec->ivs[1];
            bm->defenseIV = spec->ivs[2];
            bm->speedIV = spec->ivs[3];
            bm->spAttackIV = spec->ivs[4];
            bm->spDefenseIV = spec->ivs[5];
            bm->experience = GetMonData(mon, MON_DATA_EXP);
            bm->friendship = GetMonData(mon, MON_DATA_FRIENDSHIP);
            GetMonData(mon, MON_DATA_NICKNAME, bm->nickname);
        }
        bm->item = prev.item == ITEM_NONE ? ITEM_NONE : GetMonData(mon, MON_DATA_HELD_ITEM);
        bm->maxHP = newMax;
        bm->hp = hp;
        if (gBattleStruct)
        {
            // AI memory of its own HP at the last switch-out, in the new max HP's scale
            u16 *hpOut = &gBattleStruct->hpOnSwitchout[1];
            if (oldMaxHp)
                *hpOut = (u16)((u32)*hpOut * newMax / oldMaxHp);
        }
    }
}

static u32 ResampleSleep(u32 status1, s32 elapsed, u16 ability)
{
    // 2-5 when inflicted (Rest: 3, not told apart here), -1 per attempt to move (-2 with Early Bird)
    if (status1 & STATUS1_SLEEP)
        status1 = (status1 & ~STATUS1_SLEEP)
                | STATUS1_SLEEP_TURN(DrawRemaining(2, 5, elapsed, ability == ABILITY_EARLY_BIRD ? 2 : 1));
    return status1;
}

static void ResampleHiddenCounters(const struct Gen3DetHidden *h)
{
    static const struct Gen3DetHidden sNone;
    s32 i, b;

    if (h == NULL)
        h = &sNone;
    // Sleep, both sides, every party Pokemon (the counter survives switching)
    for (i = 0; i < 3; i++)
    {
        for (b = 0; b < 2; b++)
        {
            struct Pokemon *mon = b == 0 ? &gPlayerParty[i] : &gEnemyParty[i];
            u32 status = GetMonData(mon, MON_DATA_STATUS);
            if (GetMonData(mon, MON_DATA_SPECIES) != SPECIES_NONE && (status & STATUS1_SLEEP))
            {
                u16 ability = GetAbilityBySpecies(GetMonData(mon, MON_DATA_SPECIES), GetMonData(mon, MON_DATA_ABILITY_NUM));
                status = ResampleSleep(status, h->sleepElapsed[b][i], ability);
                SetMonData(mon, MON_DATA_STATUS, &status);
            }
        }
    }
    if (!gMain.inBattle)
        return;
    for (b = 0; b < gBattlersCount && b < 2; b++)
    {
        struct BattlePokemon *m = &gBattleMons[b];
        struct DisableStruct *d = &gDisableStructs[b];
        struct Pokemon *party = GetBattlerSide(b) == B_SIDE_PLAYER ? gPlayerParty : gEnemyParty;
        u8 slot = gBattlerPartyIndexes[b];
        if (m->species == SPECIES_NONE)
            continue;
        if (m->status1 & STATUS1_SLEEP)
        {
            u32 status = ResampleSleep(m->status1, slot < 3 ? h->sleepElapsed[b][slot] : 0, m->ability);
            m->status1 = status;
            SetMonData(&party[slot], MON_DATA_STATUS, &status);
        }
        // 2-5 when inflicted (also at the end of a rampage), -1 per attempt to move
        if (m->status2 & STATUS2_CONFUSION)
            m->status2 = (m->status2 & ~STATUS2_CONFUSION)
                       | STATUS2_CONFUSION_TURN(DrawRemaining(2, 5, h->confusionElapsed[b], 1));
        // Wrap & co. 3-6, Uproar 2-5, Thrash lock 2-3: -1 per turn
        if (m->status2 & STATUS2_WRAPPED)
            m->status2 = (m->status2 & ~STATUS2_WRAPPED)
                       | STATUS2_WRAPPED_TURN(DrawRemaining(3, 6, h->wrapElapsed[b], 1));
        if (m->status2 & STATUS2_UPROAR)
            m->status2 = (m->status2 & ~STATUS2_UPROAR)
                       | STATUS2_UPROAR_TURN(DrawRemaining(2, 5, h->uproarElapsed[b], 1));
        if (m->status2 & STATUS2_LOCK_CONFUSE)
            m->status2 = (m->status2 & ~STATUS2_LOCK_CONFUSE)
                       | STATUS2_LOCK_CONFUSE_TURN(DrawRemaining(2, 3, h->rampageElapsed[b], 1));
        // Disable 2-5, Encore 3-6: the start value is stored, so elapsed = start - left (the player counted it)
        if (d->disableTimer)
        {
            s32 elapsed = (s32)d->disableTimerStartValue - d->disableTimer;
            u32 left = DrawRemaining(2, 5, elapsed, 1);
            d->disableTimer = left;
            d->disableTimerStartValue = left + (elapsed > 0 ? elapsed : 0);
        }
        if (d->encoreTimer)
        {
            s32 elapsed = (s32)d->encoreTimerStartValue - d->encoreTimer;
            u32 left = DrawRemaining(3, 6, elapsed, 1);
            d->encoreTimer = left;
            d->encoreTimerStartValue = left + (elapsed > 0 ? elapsed : 0);
        }
    }
    // The opponent's Substitute: maxHP / 4 when made; if it was hit (the player saw it), any HP it could have left
    if (gBattleMons[1].status2 & STATUS2_SUBSTITUTE)
    {
        struct DisableStruct *d = &gDisableStructs[1];
        u32 full = gBattleMons[1].maxHP / 4;
        if (full == 0)
            full = 1;
        if (full > 255)
            full = 255;
        if (d->substituteHP < full && full > 1)
            d->substituteHP = 1 + DetRandom() % (full - 1);
        else
            d->substituteHP = full;
    }
    // Future Sight / Doom Desire: the damage was computed with the true stats of the Pokemon involved
    // when it was used; recompute it with the Pokemon out now (the true values are not known)
    for (b = 0; b < 2 && b < gBattlersCount; b++)
    {
        if (gWishFutureKnock.futureSightCounter[b] != 0)
        {
            u8 atk = gWishFutureKnock.futureSightAttacker[b];
            if (atk < gBattlersCount && gBattleMons[atk].species != SPECIES_NONE)
                gWishFutureKnock.futureSightDmg[b] = CalculateBaseDamage(&gBattleMons[atk], &gBattleMons[b],
                                                                         gWishFutureKnock.futureSightMove[b],
                                                                         gSideStatuses[GET_BATTLER_SIDE(b)], 0, 0, atk, b);
        }
    }
}

// The opponent's Choice Band lock follows its drawn item: the move it used since it came in, if it
// holds a Choice Band in this world (the true lock would reveal a hidden Choice Band).
static void FixOpponentChoiceLock(void)
{
    u16 *choiced, last;
    s32 i;
    if (!gMain.inBattle || gBattleStruct == NULL || gBattleMons[1].species == SPECIES_NONE)
        return;
    choiced = &gBattleStruct->choicedMove[1];
    last = gLastMoves[1];
    *choiced = MOVE_NONE;
    if (GetItemHoldEffect(gBattleMons[1].item) != HOLD_EFFECT_CHOICE_BAND)
        return;
    if (last == MOVE_NONE || last == MOVE_UNAVAILABLE || last == MOVE_STRUGGLE)
        return;
    for (i = 0; i < MAX_MON_MOVES; i++)
        if (gBattleMons[1].moves[i] == last)
            *choiced = last;
}

int Gen3Search_Determinize(const struct Gen3DetMon *mons, int count, const struct Gen3DetHidden *hidden,
                           int64_t hiddenSeed)
{
    int k, i, full;
    u8 enemyActive = 0xFF, covered = 0;

    for (k = 0; k < count; k++)
    {
        const struct Gen3DetMon *s = &mons[k];
        s32 evTotal = 0;
        if (s->partySlot < 0 || s->partySlot >= FRONTIER_PARTY_SIZE)
            return -1;
        if (s->species <= 0)
            continue;
        if (s->species >= NUM_SPECIES || s->nature < 0 || s->nature >= NUM_NATURES || s->hpFraction > 1.0f
            || s->item < 0 || s->item >= ITEMS_COUNT)
            return -1;
        for (i = 0; i < MAX_MON_MOVES; i++)
            if (s->moves[i] < 0 || s->moves[i] >= MOVES_COUNT)
                return -1;
        if (s->moves[0] == MOVE_NONE)
            return -1;
        for (i = 0; i < NUM_STATS; i++)
        {
            if (s->ivs[i] < 0 || s->ivs[i] > MAX_PER_STAT_IVS || s->evs[i] < 0 || s->evs[i] > 255)
                return -1;
            evTotal += s->evs[i];
        }
        if (evTotal > MAX_TOTAL_EVS)
            return -1;
    }
    if (gMain.inBattle && gBattleMons[1].species != SPECIES_NONE)
        enemyActive = gBattlerPartyIndexes[1];

    sDetState = (u64)(hiddenSeed < 0 ? 0 : hiddenSeed) * 0x2545F4914F6CDD1Dull + 0x1234567ull;
    for (k = 0; k < count; k++)
    {
        const struct Gen3DetMon *s = &mons[k];
        if (s->species <= 0)
            continue;
        Determinize(&gEnemyParty[s->partySlot], s, s->partySlot == enemyActive);
        covered |= 1 << s->partySlot;
    }
    full = 1;
    for (i = 0; i < FRONTIER_PARTY_SIZE; i++)
        if (!(covered & (1 << i)) && GetMonData(&gEnemyParty[i], MON_DATA_HP) != 0)
            full = 0;

    if (hiddenSeed >= 0)
    {
        ResampleHiddenCounters(hidden);
        FixOpponentChoiceLock();
    }
    return full;
}

static void RedrawOpponentChoice(void)
{
    struct Gen3HostState *host = Gen3_Host();
    u8 saved = gActiveBattler;

    if (!gMain.inBattle || gBattleStruct == NULL || gBattlersCount < 2)
        return;
    if (host->pendingDecision == GEN3_DECISION_ACTION)
    {
        // HandleTurnActionSelectionState: back to "choose an action"; the opponent's controller is
        // asked again on the next frame and its AI decides from this (determinized) state.
        u8 st = gBattleCommunication[1];
        if (st > SEL_STATE_BEFORE_ACTION_CHOSEN && st <= SEL_STATE_WAIT_ACTION_CONFIRMED)
        {
            gBattleCommunication[1] = SEL_STATE_BEFORE_ACTION_CHOSEN;
            gChosenActionByBattler[1] = B_ACTION_NONE;
            gChosenMoveByBattler[1] = MOVE_NONE;
            gBattleStruct->chosenMovePositions[1] = 0;
            gBattleStruct->moveTarget[1] = 0;
            gBattleStruct->monToSwitchIntoId[1] = PARTY_SIZE;
            gBattleStruct->AI_monToSwitchIntoId[1] = PARTY_SIZE;
            memset(gBattleBufferB[1], 0, 4);
        }
    }
    else if (host->pendingDecision == GEN3_DECISION_SWITCH
             && gBattleMons[1].hp == 0 && gBattleStruct->monToSwitchIntoId[1] < PARTY_SIZE)
    {
        // Both sides fainted: the opponent's replacement (OpponentHandleChoosePokemon) was already
        // chosen from its hidden team. Choose it again from this state, the same way.
        s32 chosen, i;
        gActiveBattler = 1;
        chosen = GetMostSuitableMonToSwitchInto();
        if (chosen == PARTY_SIZE)
        {
            for (i = 0; i < PARTY_SIZE; i++)
            {
                if (GetMonData(&gEnemyParty[i], MON_DATA_HP) != 0 && i != gBattlerPartyIndexes[1])
                {
                    chosen = i;
                    break;
                }
            }
        }
        if (chosen < PARTY_SIZE)
        {
            gBattleStruct->monToSwitchIntoId[1] = chosen;
            BtlController_EmitChosenMonReturnValue(B_COMM_TO_ENGINE, chosen, NULL);
        }
        gActiveBattler = saved;
    }
}

void Gen3Search_RedrawTurn(uint32_t seed)
{
    gRngValue = seed;
    if (!gMain.inBattle)
        return;
    gRandomTurnNumber = Random();   // battle_main.c: drawn when the turn's action selection starts
    RedrawOpponentChoice();
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

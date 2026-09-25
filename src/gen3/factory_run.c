// A Battle Factory run, driven like the game's map scripts drive it.
//
// Every step calls the same special functions the scripts call, in the same order
// (data/maps/BattleFrontier_BattleFactory{Lobby,PreBattleRoom,BattleRoom}/scripts.inc), so
// rental generation, opponent generation, IVs, Noland, swaps and streak bookkeeping are the
// game's own code. Only the overworld's per-frame RNG advance between these steps (which
// depends on how many frames the player spends walking and reading text) is not modelled;
// the RNG simply carries over from one step to the next.

#include "global.h"
#include "battle.h"
#include "battle_factory.h"
#include "battle_setup.h"
#include "battle_tower.h"
#include "event_data.h"
#include "frontier_util.h"
#include "pokemon.h"
#include "random.h"
#include "script_pokemon_util.h"
#include "constants/battle_factory.h"
#include "constants/battle_frontier.h"
#include "constants/frontier_util.h"
#include "constants/trainers.h"

#include "gen3/host.h"

#include <string.h>

// Extracted from battle_factory_screen.c / added to battle_tower.c (third_party/pokeemerald/src)
void Host_FactoryCreateRentals(void);
struct Pokemon *Host_FactoryRental(u8 i);
u16 Host_FactoryRentalMonId(u8 i);
bool8 Host_FactoryRent(u8 slot);
void Host_FactoryConfirmRentals(void);
bool8 Host_FactorySwap(u8 playerSlot, u8 enemySlot);
void Host_StartSpecialTrainerBattle(void);
extern void BattleMainCB2(void);

EWRAM_DATA static struct Gen3FactoryState sRun = {0};

struct Gen3FactoryState *Gen3Factory_State(void)
{
    return &sRun;
}

// ---------------------------------------------------------------------------
// Script macros (asm/macros/battle_frontier/*.inc)
// ---------------------------------------------------------------------------

static u16 FactoryFunc(u16 func, u16 arg5, u16 arg6)
{
    gSpecialVar_0x8004 = func;
    gSpecialVar_0x8005 = arg5;
    gSpecialVar_0x8006 = arg6;
    CallBattleFactoryFunction();
    return gSpecialVar_Result;
}

static u16 FrontierUtil(u16 func, u16 arg5, u16 arg6)
{
    gSpecialVar_0x8004 = func;
    gSpecialVar_0x8005 = arg5;
    gSpecialVar_0x8006 = arg6;
    CallFrontierUtilFunc();
    return gSpecialVar_Result;
}

#define factory_get(data)            FactoryFunc(BATTLE_FACTORY_FUNC_GET_DATA, data, 0)
#define factory_set(data, val)       FactoryFunc(BATTLE_FACTORY_FUNC_SET_DATA, data, val)
#define frontier_get(data)           FrontierUtil(FRONTIER_UTIL_FUNC_GET_DATA, data, 0)
#define frontier_set(data, val)      FrontierUtil(FRONTIER_UTIL_FUNC_SET_DATA, data, val)

// ---------------------------------------------------------------------------
// Rooms
// ---------------------------------------------------------------------------

static void GenerateNextOpponent(void)
{
    // factory_generateopponentmons, factory_getopponentmontype, factory_getopponentstyle
    FactoryFunc(BATTLE_FACTORY_FUNC_GENERATE_OPPONENT_MONS, 0, 0);
    sRun.hintType = FactoryFunc(BATTLE_FACTORY_FUNC_GET_OPPONENT_MON_TYPE, 0, 0);
    sRun.hintStyle = FactoryFunc(BATTLE_FACTORY_FUNC_GET_OPPONENT_STYLE, 0, 0);
}

// Lobby: BattleFrontier_BattleFactoryLobby_EventScript_SaveBeforeChallenge, then the pre-battle
// room's first entry (EnterRoom) up to factory_rentmons.
static void EnterChallenge(void)
{
    VarSet(VAR_FRONTIER_FACILITY, FRONTIER_FACILITY_FACTORY);
    VarSet(VAR_FRONTIER_BATTLE_MODE, FRONTIER_MODE_SINGLES);
    frontier_set(FRONTIER_DATA_LVL_MODE, sRun.lvlMode);
    FactoryFunc(BATTLE_FACTORY_FUNC_INIT, 0, 0);
    frontier_set(FRONTIER_DATA_CHALLENGE_STATUS, CHALLENGE_STATUS_SAVING);
    factory_set(FACTORY_DATA_WIN_STREAK_ACTIVE, TRUE);
    frontier_set(FRONTIER_DATA_PAUSED, FALSE);

    FactoryFunc(BATTLE_FACTORY_FUNC_GENERATE_RENTAL_MONS, 0, 0);
    GenerateNextOpponent();
    FactoryFunc(BATTLE_FACTORY_FUNC_SET_SWAPPED, 0, 0);
    Host_FactoryCreateRentals();
    sRun.phase = GEN3_FACTORY_RENTAL;
}

// Battle room: OnTransition + EnterRoom + BattleOpponent / DoNolandBattle
static void StartBattle(void)
{
    frontier_set(FRONTIER_DATA_RECORD_DISABLED, FALSE);
    FrontierUtil(FRONTIER_UTIL_FUNC_SET_TRAINERS, 0, 0);
    sRun.brainStatus = FrontierUtil(FRONTIER_UTIL_FUNC_GET_BRAIN_STATUS, 0, 0);
    if (sRun.brainStatus != FRONTIER_BRAIN_NOT_READY)
        FrontierUtil(FRONTIER_UTIL_FUNC_SET_BRAIN_OBJECT, 0, 0);   // gTrainerBattleOpponent_A = Noland
    HealPlayerParty();
    gSpecialVar_0x8004 = SPECIAL_BATTLE_FACTORY;
    gSpecialVar_0x8005 = 0;
    DoSpecialTrainerBattle();
    // The game clears gBattleOutcome during the intro (BattleStartClearSetData); clear it now so
    // the previous battle's result is not mistaken for this one's while it starts up.
    gBattleOutcome = 0;
    Host_StartSpecialTrainerBattle();
    sRun.trainerId = gTrainerBattleOpponent_A;
    sRun.phase = GEN3_FACTORY_BATTLE;
}

// After the battle: DefeatedOpponent / DefeatedNoland, then the pre-battle room's
// ReturnToRoomFromBattle and AskSwapMon, or the end of the challenge.
static void FinishBattle(void)
{
    u16 battleNum;

    sRun.lastOutcome = gSpecialVar_Result;
    if (gSpecialVar_Result != B_OUTCOME_WON)
    {
        frontier_set(FRONTIER_DATA_CHALLENGE_STATUS, CHALLENGE_STATUS_LOST);
        sRun.phase = GEN3_FACTORY_RUN_OVER;
        return;
    }

    if (sRun.brainStatus != FRONTIER_BRAIN_NOT_READY)
    {
        // DefeatedNolandSilver / DefeatedNolandGold
        u16 symbols = FrontierUtil(FRONTIER_UTIL_FUNC_GET_FACILITY_SYMBOLS, 0, 0);
        if ((sRun.brainStatus == FRONTIER_BRAIN_GOLD || sRun.brainStatus == FRONTIER_BRAIN_STREAK_LONG)
            ? symbols != 2 : symbols == 0)
            FrontierUtil(FRONTIER_UTIL_FUNC_GIVE_FACILITY_SYMBOL, 0, 0);
    }
    if (factory_get(FACTORY_DATA_WIN_STREAK_SWAPS) != MAX_STREAK)
        factory_set(FACTORY_DATA_WIN_STREAK_SWAPS, factory_get(FACTORY_DATA_WIN_STREAK_SWAPS) + 1);
    FrontierUtil(FRONTIER_UTIL_FUNC_INCREMENT_STREAK, 0, 0);
    battleNum = frontier_get(FRONTIER_DATA_BATTLE_NUM) + 1;
    frontier_set(FRONTIER_DATA_BATTLE_NUM, battleNum);
    sRun.wins++;

    if (battleNum == FRONTIER_STAGES_PER_CHALLENGE)
    {
        // WarpToLobbyWon; the streak goes on with a new challenge (new rentals).
        frontier_set(FRONTIER_DATA_CHALLENGE_STATUS, CHALLENGE_STATUS_WON);
        sRun.challengesWon++;
        EnterChallenge();
        return;
    }

    // ReturnToRoomFromBattle: the defeated team becomes the swap candidates
    FactoryFunc(BATTLE_FACTORY_FUNC_SET_OPPONENT_MONS, 0, 0);
    FactoryFunc(BATTLE_FACTORY_FUNC_RESET_HELD_ITEMS, 0, 0);
    if (FrontierUtil(FRONTIER_UTIL_FUNC_GET_BRAIN_STATUS, 0, 0) != FRONTIER_BRAIN_NOT_READY)
    {
        // AskReadyForHead -> AskSwapBeforeHead: no opponent is generated before Noland's battle
        // and the attendant "can't tell anything" about him (FillFactoryBrainParty draws his team
        // when the battle starts)
        sRun.hintType = GEN3_FACTORY_NO_HINT_TYPE;
        sRun.hintStyle = GEN3_FACTORY_NO_HINT_STYLE;
    }
    else
    {
        // AskSwapMon: the next opponent is generated (and hinted at) before the swap question
        GenerateNextOpponent();
    }
    sRun.phase = GEN3_FACTORY_SWAP;
}

// ---------------------------------------------------------------------------
// API
// ---------------------------------------------------------------------------

void Gen3Factory_Begin(u8 lvlMode, u16 winStreak, u16 rentsCount, u32 seed)
{
    u32 lvl = lvlMode ? FRONTIER_LVL_OPEN : FRONTIER_LVL_50;

    memset(&sRun, 0, sizeof(sRun));
    sRun.lvlMode = lvl;
    gRngValue = seed;
    gSaveBlock2Ptr->frontier.lvlMode = lvl;
    if (winStreak || rentsCount)
    {
        // An ongoing streak (as if earlier challenges had been won); InitFactoryChallenge resets
        // the streak and the rent count otherwise
        gSaveBlock2Ptr->frontier.winStreakActiveFlags |= lvl == FRONTIER_LVL_OPEN ? STREAK_FACTORY_SINGLES_OPEN
                                                                                   : STREAK_FACTORY_SINGLES_50;
    }
    gSaveBlock2Ptr->frontier.factoryWinStreaks[FRONTIER_MODE_SINGLES][lvl] = winStreak;
    gSaveBlock2Ptr->frontier.factoryRentsCount[FRONTIER_MODE_SINGLES][lvl] = rentsCount;
    EnterChallenge();
}

int Gen3Factory_Phase(void)
{
    return sRun.phase;
}

void Gen3Factory_ReadRental(u8 i, void *dst)
{
    memcpy(dst, Host_FactoryRental(i), sizeof(struct Pokemon));
}

u16 Gen3Factory_RentalFrontierMonId(u8 i)
{
    return Host_FactoryRentalMonId(i);
}

// Rent three mons (select-screen positions, in party order) and start the next battle.
int Gen3Factory_Rent(u8 a, u8 b, u8 c)
{
    u16 sa, sb, sc;

    // Checked before touching the select screen, so a refused rental leaves it as it was
    if (sRun.phase != GEN3_FACTORY_RENTAL || a >= 6 || b >= 6 || c >= 6 || a == b || a == c || b == c)
        return FALSE;
    sa = GetMonData(Host_FactoryRental(a), MON_DATA_SPECIES);
    sb = GetMonData(Host_FactoryRental(b), MON_DATA_SPECIES);
    sc = GetMonData(Host_FactoryRental(c), MON_DATA_SPECIES);
    if (sa == sb || sa == sc || sb == sc)
        return FALSE;
    if (!Host_FactoryRent(a) || !Host_FactoryRent(b) || !Host_FactoryRent(c))
        return FALSE;
    Host_FactoryConfirmRentals();
    StartBattle();
    return TRUE;
}

// playerSlot < 0: keep the team. Otherwise trade gPlayerParty[playerSlot] for the defeated
// team's gEnemyParty[enemySlot]. Starts the next battle.
int Gen3Factory_Swap(int playerSlot, int enemySlot)
{
    if (sRun.phase != GEN3_FACTORY_SWAP || playerSlot > 2 || (playerSlot >= 0 && (enemySlot < 0 || enemySlot > 2)))
        return FALSE;
    if (playerSlot >= 0)
    {
        if (!Host_FactorySwap(playerSlot, enemySlot))
            return FALSE;
        FactoryFunc(BATTLE_FACTORY_FUNC_SET_SWAPPED, 0, 0);
        sRun.swaps++;
    }
    StartBattle();
    return TRUE;
}

// In GEN3_FACTORY_BATTLE: run the battle until the player decides (returns GEN3_DECISION_ACTION
// or _SWITCH) or it is over, in which case the run advances (phase changes) and
// GEN3_DECISION_BATTLE_OVER is returned.
int Gen3Factory_RunBattle(u32 maxFrames)
{
    int decision;
    u32 f;

    if (sRun.phase != GEN3_FACTORY_BATTLE)
        return GEN3_DECISION_NONE;
    decision = Gen3_RunUntilDecision(maxFrames);
    if (decision != GEN3_DECISION_BATTLE_OVER)
        return decision;
    // Let the battle wind down (messages, freeing battle resources, HandleSpecialTrainerBattleEnd)
    for (f = 0; f < maxFrames && (gMain.inBattle || gMain.callback2 == BattleMainCB2); f++)
        Gen3_RunFrame();
    FinishBattle();
    return GEN3_DECISION_BATTLE_OVER;
}

#ifndef GEN3_HOST_H
#define GEN3_HOST_H

// C interface of the gen3 battle core (src/gen3/host.c). Usable from C and C++.

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    GEN3_DECISION_NONE = 0,
    GEN3_DECISION_ACTION = 1,       // choose a move or a switch
    GEN3_DECISION_SWITCH = 2,       // forced replacement after a faint
    GEN3_DECISION_BATTLE_OVER = 3,
    GEN3_DECISION_TIMEOUT = 4,
};

struct Gen3HostState {
    uint32_t frames;
    uint8_t pendingDecision;
    uint8_t decisionBattler;
    uint8_t chosenMoveSlot;
    uint8_t chosenPartyIndex;
    uint8_t switchPending;
    uint8_t partyAction;
    uint8_t moveRejected;
    uint8_t answeredMove;
};

enum {
    GEN3_FACTORY_NONE = 0,
    GEN3_FACTORY_RENTAL = 1,     // pick 3 of the 6 rentals
    GEN3_FACTORY_BATTLE = 2,     // a battle is running (see Gen3Factory_RunBattle)
    GEN3_FACTORY_SWAP = 3,       // after a win: keep the team or trade one mon for the defeated team's
    GEN3_FACTORY_RUN_OVER = 4,   // lost: the streak is over
};

struct Gen3FactoryState {
    uint8_t phase;
    uint8_t lvlMode;
    uint8_t hintType;            // factory_getopponentmontype: most common type of the next opponent
    uint8_t hintStyle;           // factory_getopponentstyle: FACTORY_STYLE_*
    uint8_t brainStatus;         // FRONTIER_BRAIN_* for the current battle
    uint8_t lastOutcome;         // B_OUTCOME_* of the last battle
    uint16_t trainerId;
    uint16_t wins;
    uint16_t swaps;
    uint16_t challengesWon;
};

void Gen3_InitSystem(void);
void Gen3_RunFrame(void);
size_t Gen3_StateSize(void);
void Gen3_SaveState(void *dst);
void Gen3_LoadState(const void *src);
struct Gen3HostState *Gen3_Host(void);

void Gen3_SetBattleParams(uint32_t battleTypeFlags, uint16_t trainerA);
void Gen3_WriteParty(int side, const void *raw, size_t size);   // raw struct Pokemon[<=6] (game format)
void Gen3_WriteSaveBlock2(uint32_t offset, const void *data, size_t size);
void Gen3_ReadSaveBlock2(uint32_t offset, void *dst, size_t size);
void Gen3_SetVar(uint16_t id, uint16_t value);
void Gen3_SetFlag(uint16_t id, uint8_t on);
uint32_t Gen3_GetRng(void);
void Gen3_SetRng(uint32_t value);
int Gen3_ReadRam(uint32_t gbaAddress, void *dst, size_t size);
int Gen3_WriteRam(uint32_t gbaAddress, const void *src, size_t size);   // tests / debugging
int Gen3_StartBattle(uint32_t maxFrames);   // 1 once gBattleMainFunc == BeginBattleIntro
int Gen3_RunUntilDecision(uint32_t maxFrames);
void Gen3_ChooseMove(uint8_t slot);
uint8_t Gen3_UnusableMoves(uint8_t battler);
uint8_t Gen3_CanSwitch(uint8_t battler);
uint16_t Gen3_ChoicedMove(uint8_t battler);    // gBattleStruct->choicedMove (Choice Band lock)       // voluntary switch allowed (not trapped)   // bitmask of move slots the game refuses
void Gen3_ChooseSwitch(uint8_t partyIndex);
void Gen3_Forfeit(void);                 // RUN, then "forfeit?" YES: loses the battle
void Gen3_DebugDump(void);

// Battle Factory run (src/gen3/factory_run.c)
void Gen3Factory_Begin(uint8_t openLevel, uint16_t winStreak, uint16_t rentsCount, uint32_t seed);
struct Gen3FactoryState *Gen3Factory_State(void);
int Gen3Factory_Phase(void);
void Gen3Factory_ReadRental(uint8_t i, void *dst);          // struct Pokemon, game format
uint16_t Gen3Factory_RentalFrontierMonId(uint8_t i);
int Gen3Factory_Rent(uint8_t a, uint8_t b, uint8_t c);
int Gen3Factory_Swap(int playerSlot, int enemySlot);
int Gen3Factory_RunBattle(uint32_t maxFrames);
void Gen3_TraceRandom(int enable);   // log every Random() caller to stderr   // callbacks and controller state to stderr

#ifdef __cplusplus
}
#endif

#endif

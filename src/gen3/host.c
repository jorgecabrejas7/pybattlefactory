#define _GNU_SOURCE  // dladdr
// Host layer for the vendored pokeemerald battle code (third_party/pokeemerald).
//
// Provides what the game's other subsystems would: the frame loop, headless battle
// controllers (the player's decisions come from the caller, the opponent's from the game's
// own AI), script address translation, BIOS helpers and event vars. All battle state lives
// in the "gen3_ram" section, so an instance is saved/restored/cloned by copying that section.

#include "global.h"
#include "battle.h"
#include "battle_ai_script_commands.h"
#include "battle_controllers.h"
#include "battle_setup.h"
#include "event_data.h"
#include "malloc.h"
#include "palette.h"
#include "party_menu.h"
#include "pokemon.h"
#include "random.h"
#include "util.h"
#include "constants/battle_frontier.h"
#include "constants/moves.h"
#include "constants/abilities.h"

#include "gen3/host.h"

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Scripts address these structures by GBA offset and party data is copied raw from the real
// game, so their host layout must match the GBA layout.
_Static_assert(sizeof(struct Pokemon) == 100, "struct Pokemon layout");
_Static_assert(sizeof(struct BattlePokemon) == 0x58, "struct BattlePokemon layout");
_Static_assert(sizeof(struct DisableStruct) == 0x1C, "struct DisableStruct layout");
_Static_assert(__builtin_offsetof(struct BattleScripting, dmgMultiplier) == 0x0E, "struct BattleScripting layout");
_Static_assert(__builtin_offsetof(struct BattleScripting, animTargetsHit) == 0x19, "struct BattleScripting layout");
_Static_assert(__builtin_offsetof(struct SaveBlock2, frontier) == 0x64C, "struct SaveBlock2 layout");

// ---------------------------------------------------------------------------
// Battle RAM section and instance management
// ---------------------------------------------------------------------------

extern u8 __start_gen3_ram[];
extern u8 __stop_gen3_ram[];

size_t Gen3_StateSize(void)
{
    return (size_t)(__stop_gen3_ram - __start_gen3_ram);
}

void Gen3_SaveState(void *dst)
{
    memcpy(dst, __start_gen3_ram, Gen3_StateSize());
}

void Gen3_LoadState(const void *src)
{
    memcpy(__start_gen3_ram, src, Gen3_StateSize());
}

// Writes to GBA hardware (VRAM, OAM, palettes, IO registers) land here and are ignored.
unsigned char gGen3FakeHardware[0x80000] __attribute__((aligned(16)));

// Save blocks. SaveBlock2 holds the Frontier state and is per instance; SaveBlock1 (bag, map,
// flags...) is only read incidentally by battle code and is shared.
EWRAM_DATA static struct SaveBlock2 sSaveBlock2 = {0};
static struct SaveBlock1 sSaveBlock1;

// Event vars/flags used by Frontier code (VAR_FRONTIER_*).
EWRAM_DATA static u16 sVars[VARS_COUNT] = {0};
EWRAM_DATA static u8 sFlags[(FLAGS_COUNT + 7) / 8] = {0};

// Host bookkeeping, also per instance.
EWRAM_DATA static struct Gen3HostState sHost = {0};

struct Gen3HostState *Gen3_Host(void)
{
    return &sHost;
}

// ---------------------------------------------------------------------------
// Script address translation (see scripts/extract_rom_scripts.py)
// ---------------------------------------------------------------------------

extern const u8 Gen3_RomScripts[];
extern const u32 gGen3RomScriptsBase;
extern const u32 gGen3RomScriptsSize;
struct Gen3RamSymbol { u32 gbaAddress; u32 size; void *host; };
extern const struct Gen3RamSymbol gGen3RamSymbols[];
extern const u32 gGen3RamSymbolsCount;

void *Gen3_ResolveAddress(u32 addr)
{
    u32 i;

    if (addr - gGen3RomScriptsBase < gGen3RomScriptsSize)
        return (void *)(Gen3_RomScripts + (addr - gGen3RomScriptsBase));
    if (addr < 0x4000)
    {
        // BIOS region. Scripts pass NULL for unused animation arguments, which the GBA reads
        // from the BIOS (open bus); only animation code sees the value.
        static const u8 sFakeBios[0x4000];
        return (void *)(sFakeBios + addr);
    }
    for (i = 0; i < gGen3RamSymbolsCount; i++)
    {
        const struct Gen3RamSymbol *s = &gGen3RamSymbols[i];
        if (addr - s->gbaAddress < s->size)
            return (u8 *)s->host + (addr - s->gbaAddress);
    }
    fprintf(stderr, "gen3: unresolved script address %08x\n", addr);
    abort();
}

void Gen3_OutOfMemory(u32 size)
{
    fprintf(stderr, "gen3: battle heap exhausted allocating %u bytes (HEAP_SIZE %#x)\n", size, HEAP_SIZE);
    abort();
}

// ---------------------------------------------------------------------------
// Palette fades. Nothing is drawn, but battle code sequences work on fades (e.g. the battle's
// resources are freed while the closing fade runs), so a fade stays active for a few frames.
// ---------------------------------------------------------------------------

IWRAM_DATA struct PaletteFadeControl gPaletteFade = {0};
EWRAM_DATA static u8 sFadeFramesLeft = 0;
#define FADE_FRAMES 3

bool8 BeginNormalPaletteFade(u32 selectedPalettes, s8 delay, u8 startY, u8 targetY, u16 blendColor)
{
    if (gPaletteFade.active)
        return FALSE;
    gPaletteFade.active = TRUE;
    sFadeFramesLeft = FADE_FRAMES;
    return TRUE;
}

void BeginFastPaletteFade(u8 submode)
{
    gPaletteFade.active = TRUE;
    sFadeFramesLeft = FADE_FRAMES;
}

u8 UpdatePaletteFade(void)
{
    if (sFadeFramesLeft && --sFadeFramesLeft)
        return PALETTE_FADE_STATUS_LOADING;
    gPaletteFade.active = FALSE;
    return PALETTE_FADE_STATUS_DONE;
}

void ResetPaletteFade(void)
{
    gPaletteFade.active = FALSE;
}

void ResetPaletteFadeControl(void)
{
    gPaletteFade.active = FALSE;
}

// ---------------------------------------------------------------------------
// BIOS / system helpers
// ---------------------------------------------------------------------------

void CpuSet(const void *src, void *dest, u32 control)
{
    u32 count = control & 0x1FFFFF;
    bool32 fill = (control >> 24) & 1;
    bool32 words = (control >> 26) & 1;
    u32 i;

    if (words)
    {
        const u32 *s = src;
        u32 *d = dest;
        for (i = 0; i < count; i++)
            d[i] = fill ? s[0] : s[i];
    }
    else
    {
        const u16 *s = src;
        u16 *d = dest;
        for (i = 0; i < count; i++)
            d[i] = fill ? s[0] : s[i];
    }
}

void CpuFastSet(const void *src, void *dest, u32 control)
{
    CpuSet(src, dest, (control & 0x1FFFFF) | (1 << 26) | (control & (1 << 24)));
}

u16 Sqrt(u32 num)
{
    u32 r = 0, bit = 1u << 30;

    while (bit > num)
        bit >>= 2;
    while (bit)
    {
        if (num >= r + bit)
        {
            num -= r + bit;
            r = (r >> 1) + bit;
        }
        else
        {
            r >>= 1;
        }
        bit >>= 2;
    }
    return r;
}

u16 *GetVarPointer(u16 id)
{
    if (id >= VARS_START && id < VARS_END)
        return &sVars[id - VARS_START];
    if (id == VAR_RESULT)
        return &gSpecialVar_Result;
    if (id == VAR_0x8004)
        return &gSpecialVar_0x8004;
    if (id == VAR_0x8005)
        return &gSpecialVar_0x8005;
    if (id == VAR_0x8006)
        return &gSpecialVar_0x8006;
    return NULL;
}

u16 VarGet(u16 id)
{
    u16 *ptr = GetVarPointer(id);
    return ptr ? *ptr : id;
}

bool8 VarSet(u16 id, u16 value)
{
    u16 *ptr = GetVarPointer(id);
    if (!ptr)
        return FALSE;
    *ptr = value;
    return TRUE;
}

u8 FlagGet(u16 id)
{
    return id < FLAGS_COUNT && (sFlags[id / 8] >> (id % 8)) & 1;
}

u8 FlagSet(u16 id)
{
    if (id < FLAGS_COUNT)
        sFlags[id / 8] |= 1 << (id % 8);
    return 0;
}

u8 FlagClear(u16 id)
{
    if (id < FLAGS_COUNT)
        sFlags[id / 8] &= ~(1 << (id % 8));
    return 0;
}

EWRAM_DATA u16 gSpecialVar_0x8004 = 0;
EWRAM_DATA u16 gSpecialVar_0x8005 = 0;
EWRAM_DATA u16 gSpecialVar_0x8006 = 0;
EWRAM_DATA u16 gSpecialVar_Result = 0;
EWRAM_DATA u16 gSpecialVar_LastTalked = 0;
EWRAM_DATA u16 gTrainerBattleOpponent_A = 0;
EWRAM_DATA u16 gTrainerBattleOpponent_B = 0;
EWRAM_DATA u16 gPartnerTrainerId = 0;

#include "data/text/species_names.h"
#include "data/text/move_names.h"

// ---------------------------------------------------------------------------
// Headless battle controllers
// ---------------------------------------------------------------------------

// Extracted verbatim from the game's controllers (third_party/pokeemerald/src/host_ctrl_*.c).
void Host_PlayerGetMonData(void);
void Host_PlayerGetRawMonData(void);
void Host_PlayerSetMonData(void);
void Host_PlayerSetRawMonData(void);
void Host_OpponentGetMonData(void);
void Host_OpponentGetRawMonData(void);
void Host_OpponentSetMonData(void);
void Host_OpponentSetRawMonData(void);
void Host_OpponentChooseAction(void);
void Host_OpponentChooseMove(void);
void Host_OpponentChooseItem(void);
void Host_OpponentChoosePokemon(void);
void Host_BattlePartyMenuChoose(u8 partyIndex);

// PlayerBufferExecCompleted / OpponentBufferExecCompleted outside link battles.
void Host_ControllerExecCompleted(void)
{
    gBattleControllerExecFlags &= ~gBitTable[gActiveBattler];
}

static void Host_PlayerBufferRunCommand(void);
static void Host_OpponentBufferRunCommand(void);

static void Host_PlayerDummy(void)
{
}

// Commands whose only effect is on graphics/sound/text complete immediately. The few that
// touch battle state do what the game's handler does to that state.
static bool8 Host_CommonCommand(u8 cmd, bool8 isPlayer)
{
    switch (cmd)
    {
    case CONTROLLER_SWITCHINANIM:
        if (!isPlayer)
            gBattleStruct->monToSwitchIntoId[gActiveBattler] = PARTY_SIZE;
        gBattlerPartyIndexes[gActiveBattler] = gBattleBufferA[gActiveBattler][1];
        if (isPlayer)
        {
            gActionSelectionCursor[gActiveBattler] = 0;
            gMoveSelectionCursor[gActiveBattler] = 0;
        }
        return TRUE;
    case CONTROLLER_MOVEANIMATION:
        {
            struct DisableStruct *disable = (struct DisableStruct *)&gBattleBufferA[gActiveBattler][16];
            gTransformedPersonalities[gActiveBattler] = disable->transformedMonPersonality;
        }
        return TRUE;
    case CONTROLLER_INTROSLIDE:
        gIntroSlideFlags |= 1;
        return TRUE;
    default:
        return TRUE;
    }
}

static void Host_PlayerBufferRunCommand(void)
{
    u8 cmd;

    if (!(gBattleControllerExecFlags & gBitTable[gActiveBattler]))
        return;
    cmd = gBattleBufferA[gActiveBattler][0];
    switch (cmd)
    {
    case CONTROLLER_GETMONDATA:
        Host_PlayerGetMonData();
        return;
    case CONTROLLER_GETRAWMONDATA:
        Host_PlayerGetRawMonData();
        return;
    case CONTROLLER_SETMONDATA:
        Host_PlayerSetMonData();
        return;
    case CONTROLLER_SETRAWMONDATA:
        Host_PlayerSetRawMonData();
        return;
    case CONTROLLER_CHOOSEACTION:
        sHost.answeredMove = FALSE;
        sHost.moveRejected = FALSE;
        sHost.pendingDecision = GEN3_DECISION_ACTION;
        sHost.decisionBattler = gActiveBattler;
        gBattlerControllerFuncs[gActiveBattler] = Host_PlayerDummy;
        return;
    case CONTROLLER_PRINTSTRINGPLAYERONLY:
        // A selection message. After we answered the move menu it means "can't use that move"
        // and the game reopens the move menu (see CONTROLLER_CHOOSEMOVE below). Without a move
        // answer it is "no moves left!": the game uses Struggle and nothing is reopened.
        sHost.moveRejected = sHost.answeredMove;
        Host_ControllerExecCompleted();
        return;
    case CONTROLLER_CHOOSEMOVE:
        if (sHost.moveRejected)
        {
            // Back out of the reopened move menu (B): the player gets a fresh action decision.
            sHost.moveRejected = FALSE;
            BtlController_EmitTwoReturnValues(B_COMM_TO_ENGINE, 10, 0xFFFF);
            Host_ControllerExecCompleted();
            return;
        }
        // The action (FIGHT) was already chosen together with the move; answer immediately.
        {
            struct ChooseMoveStruct *moveInfo = (struct ChooseMoveStruct *)(&gBattleBufferA[gActiveBattler][4]);
            u8 slot = sHost.chosenMoveSlot;
            u8 moveTarget;

            gMoveSelectionCursor[gActiveBattler] = slot;
            if (moveInfo->moves[slot] == MOVE_CURSE)
                moveTarget = (moveInfo->monTypes[0] != TYPE_GHOST && moveInfo->monTypes[1] != TYPE_GHOST)
                           ? MOVE_TARGET_USER : MOVE_TARGET_SELECTED;
            else
                moveTarget = gBattleMoves[moveInfo->moves[slot]].target;
            if (moveTarget & MOVE_TARGET_USER)
                gMultiUsePlayerCursor = gActiveBattler;
            else
                gMultiUsePlayerCursor = GetBattlerAtPosition(BATTLE_OPPOSITE(GET_BATTLER_SIDE(gActiveBattler)));
            BtlController_EmitTwoReturnValues(B_COMM_TO_ENGINE, 10, slot | (gMultiUsePlayerCursor << 8));
            sHost.answeredMove = TRUE;
            Host_ControllerExecCompleted();
        }
        return;
    case CONTROLLER_CHOOSEPOKEMON:
        {
            s32 i;
            for (i = 0; i < (int)ARRAY_COUNT(gBattlePartyCurrentOrder); i++)
                gBattlePartyCurrentOrder[i] = gBattleBufferA[gActiveBattler][4 + i];
            gBattleStruct->battlerPreventingSwitchout = gBattleBufferA[gActiveBattler][1] >> 4;
            gBattleStruct->prevSelectedPartySlot = gBattleBufferA[gActiveBattler][2];
            gBattleStruct->abilityPreventingSwitchout = gBattleBufferA[gActiveBattler][3];
            gBattlerInMenuId = gActiveBattler;
            sHost.partyAction = gBattleBufferA[gActiveBattler][1] & 0xF;
            if (sHost.switchPending)
            {
                // Voluntary switch: the target was chosen together with the SWITCH action.
                sHost.switchPending = FALSE;
                Host_BattlePartyMenuChoose(sHost.chosenPartyIndex);
                BtlController_EmitChosenMonReturnValue(B_COMM_TO_ENGINE, gSelectedMonPartyId, gBattlePartyCurrentOrder);
                Host_ControllerExecCompleted();
            }
            else
            {
                // Forced replacement after a faint.
                sHost.pendingDecision = GEN3_DECISION_SWITCH;
                sHost.decisionBattler = gActiveBattler;
                gBattlerControllerFuncs[gActiveBattler] = Host_PlayerDummy;
            }
        }
        return;
    case CONTROLLER_YESNOBOX:
        // Only asked after RUN in a Frontier battle: "Would you like to forfeit?" -> YES
        gMultiUsePlayerCursor = 0;
        BtlController_EmitTwoReturnValues(B_COMM_TO_ENGINE, 0xD, 0);
        Host_ControllerExecCompleted();
        return;
    case CONTROLLER_OPENBAG:
        // Not reachable in Frontier singles.
        fprintf(stderr, "gen3: unexpected player controller command %u\n", cmd);
        abort();
    default:
        Host_CommonCommand(cmd, TRUE);
        Host_ControllerExecCompleted();
        return;
    }
}

static void Host_OpponentBufferRunCommand(void)
{
    u8 cmd;

    if (!(gBattleControllerExecFlags & gBitTable[gActiveBattler]))
        return;
    cmd = gBattleBufferA[gActiveBattler][0];
    switch (cmd)
    {
    case CONTROLLER_GETMONDATA:
        Host_OpponentGetMonData();
        return;
    case CONTROLLER_GETRAWMONDATA:
        Host_OpponentGetRawMonData();
        return;
    case CONTROLLER_SETMONDATA:
        Host_OpponentSetMonData();
        return;
    case CONTROLLER_SETRAWMONDATA:
        Host_OpponentSetRawMonData();
        return;
    case CONTROLLER_CHOOSEACTION:
        Host_OpponentChooseAction();
        return;
    case CONTROLLER_CHOOSEMOVE:
        Host_OpponentChooseMove();
        return;
    case CONTROLLER_OPENBAG:
        Host_OpponentChooseItem();
        return;
    case CONTROLLER_CHOOSEPOKEMON:
        Host_OpponentChoosePokemon();
        return;
    default:
        Host_CommonCommand(cmd, FALSE);
        Host_ControllerExecCompleted();
        return;
    }
}

void SetControllerToPlayer(void)
{
    gBattlerControllerFuncs[gActiveBattler] = Host_PlayerBufferRunCommand;
}

void SetControllerToOpponent(void)
{
    gBattlerControllerFuncs[gActiveBattler] = Host_OpponentBufferRunCommand;
}

void BattleControllerDummy(void)
{
}

// ---------------------------------------------------------------------------
// Driving the battle
// ---------------------------------------------------------------------------

void Gen3_InitSystem(void)
{
    memset(__start_gen3_ram, 0, Gen3_StateSize());
    InitHeap(gHeap, HEAP_SIZE);
    gSaveBlock2Ptr = &sSaveBlock2;
    gSaveBlock1Ptr = &sSaveBlock1;
}

void Gen3_RunFrame(void)
{
    if (gMain.callback1)
        gMain.callback1();
    if (gMain.callback2)
        gMain.callback2();
}

extern void BattleMainCB2(void);
extern void CB2_InitBattle(void);

void Gen3_SetBattleParams(u32 battleTypeFlags, u16 trainerA)
{
    gBattleTypeFlags = battleTypeFlags;
    gTrainerBattleOpponent_A = trainerA;
}

void Gen3_WriteParty(int side, const void *raw, size_t size)
{
    struct Pokemon *party = side == 0 ? gPlayerParty : gEnemyParty;
    if (size > sizeof(gPlayerParty))
        size = sizeof(gPlayerParty);
    memset(party, 0, sizeof(gPlayerParty));
    memcpy(party, raw, size);
    CalculatePlayerPartyCount();
    CalculateEnemyPartyCount();
}

void Gen3_WriteSaveBlock2(u32 offset, const void *data, size_t size)
{
    if (offset + size <= sizeof(struct SaveBlock2))
        memcpy((u8 *)gSaveBlock2Ptr + offset, data, size);
}

void Gen3_ReadSaveBlock2(u32 offset, void *dst, size_t size)
{
    if (offset + size <= sizeof(struct SaveBlock2))
        memcpy(dst, (u8 *)gSaveBlock2Ptr + offset, size);
}

void Gen3_SetFlag(u16 id, u8 on)
{
    if (on)
        FlagSet(id);
    else
        FlagClear(id);
}

void Gen3_SetVar(u16 id, u16 value)
{
    VarSet(id, value);
}

u32 Gen3_GetRng(void)
{
    return gRngValue;
}

void Gen3_SetRng(u32 value)
{
    gRngValue = value;
}

// Read battle RAM by GBA address (only variables the host build defines).
int Gen3_ReadRam(u32 addr, void *dst, size_t size)
{
    u32 i;
    for (i = 0; i < gGen3RamSymbolsCount; i++)
    {
        const struct Gen3RamSymbol *s = &gGen3RamSymbols[i];
        if (addr - s->gbaAddress < s->size && addr + size - s->gbaAddress <= s->size)
        {
            memcpy(dst, (u8 *)s->host + (addr - s->gbaAddress), size);
            return 1;
        }
    }
    return 0;
}

static const char *FuncName(void *fn)
{
    Dl_info info;
    if (fn && dladdr(fn, &info) && info.dli_sname)
        return info.dli_sname;
    return fn ? "?" : "NULL";
}

// Optional Random() call tracing (debugging RNG-order divergences).
void (*gGen3RandomHook)(void *) = NULL;

static void LogRandomCaller(void *returnAddress)
{
    fprintf(stderr, "gen3: Random() #%u from %s -> %08x\n", sHost.frames, FuncName(returnAddress), gRngValue);
}

void Gen3_TraceRandom(int enable)
{
    gGen3RandomHook = enable ? LogRandomCaller : NULL;
}

void Gen3_DebugDump(void)
{
    int i;
    fprintf(stderr, "gen3: frames=%u cb1=%s cb2=%s mainFunc=%s state=%u comm0=%u typeFlags=%08x execFlags=%08x outcome=%u\n",
            sHost.frames, FuncName((void *)gMain.callback1), FuncName((void *)gMain.callback2),
            FuncName((void *)gBattleMainFunc), gMain.state, gBattleCommunication[0], gBattleTypeFlags,
            gBattleControllerExecFlags, gBattleOutcome);
    for (i = 0; i < gBattlersCount; i++)
        fprintf(stderr, "  battler %d ctrl=%s cmd=%u\n", i, FuncName((void *)gBattlerControllerFuncs[i]),
                gBattleBufferA[i][0]);
}

int Gen3_WriteRam(u32 addr, const void *src, size_t size)
{
    u32 i;
    for (i = 0; i < gGen3RamSymbolsCount; i++)
    {
        const struct Gen3RamSymbol *s = &gGen3RamSymbols[i];
        if (addr - s->gbaAddress < s->size && addr + size - s->gbaAddress <= s->size)
        {
            memcpy((u8 *)s->host + (addr - s->gbaAddress), src, size);
            return 1;
        }
    }
    return 0;
}

extern void BeginBattleIntro(void);

// Run CB2_InitBattle and the start-of-battle callbacks up to gBattleMainFunc == BeginBattleIntro,
// the point where trace recordings take gRngValue.
int Gen3_StartBattle(u32 maxFrames)
{
    u32 f;

    gMain.inBattle = FALSE;
    gMain.callback1 = NULL;
    gBattleOutcome = 0;
    SetMainCallback2(CB2_InitBattle);
    sHost.pendingDecision = GEN3_DECISION_NONE;
    sHost.switchPending = FALSE;
    for (f = 0; f < maxFrames; f++)
    {
        if (gMain.callback2 == BattleMainCB2 && gBattleMainFunc == BeginBattleIntro)
            return 1;
        Gen3_RunFrame();
    }
    Gen3_DebugDump();
    return 0;
}

// Run frames until the player must decide or the battle ends.
// Returns GEN3_DECISION_* (GEN3_DECISION_BATTLE_OVER when gBattleOutcome is set).
static u32 Host_ProgressHash(void)
{
    u32 h = gRngValue ^ (gBattleControllerExecFlags * 0x9E3779B1u);
    int i;
    for (i = 0; i < 8; i++)
        h = h * 31 + gBattleCommunication[i];
    return h ^ (u32)(uintptr_t)gBattleMainFunc;
}

// While the player's controller waits for input the rest of the battle keeps running (the
// opponent's AI picks its move a few frames later). A decision is returned only once nothing
// else changes, so the state handed out -- and compared against the real game -- is well defined.
#define QUIET_FRAMES 8

int Gen3_RunUntilDecision(u32 maxFrames)
{
    u32 f, quiet = 0, hash = 0;

    for (f = 0; f < maxFrames; f++)
    {
        if (gBattleOutcome != 0 && gMain.inBattle)
            return GEN3_DECISION_BATTLE_OVER;
        if (sHost.pendingDecision != GEN3_DECISION_NONE
            && gBattleControllerExecFlags == gBitTable[sHost.decisionBattler])
        {
            u32 h = Host_ProgressHash();
            quiet = (h == hash) ? quiet + 1 : 0;
            hash = h;
            if (quiet >= QUIET_FRAMES)
                return sHost.pendingDecision;
        }
        Gen3_RunFrame();
        sHost.frames++;
    }
    Gen3_DebugDump();
    return GEN3_DECISION_TIMEOUT;
}

static void Host_FinishDecision(void)
{
    gActiveBattler = sHost.decisionBattler;
    gBattlerControllerFuncs[gActiveBattler] = Host_PlayerBufferRunCommand;
    sHost.pendingDecision = GEN3_DECISION_NONE;
}

// Moves the game will refuse at selection (no PP, Disable, Torment, Taunt, Imprison, Choice
// Band): bit i set = move slot i unusable. All four set means the player can only Struggle.
u8 Gen3_UnusableMoves(u8 battler)
{
    return CheckMoveLimitations(battler, 0, MOVE_LIMITATIONS_ALL);
}

// Whether `battler` may switch out voluntarily: the B_ACTION_SWITCH checks of
// HandleTurnActionSelectionState (battle_main.c), without their side effect on gLastUsedAbility.
u16 Gen3_ChoicedMove(u8 battler)
{
    return gBattleStruct ? gBattleStruct->choicedMove[battler] : 0;
}

bool8 Gen3_CanSwitch(u8 battler)
{
    s32 i;
    bool8 steel = IS_BATTLER_OF_TYPE(battler, TYPE_STEEL);
    bool8 grounded = !IS_BATTLER_OF_TYPE(battler, TYPE_FLYING) && gBattleMons[battler].ability != ABILITY_LEVITATE;

    if (gBattleMons[battler].status2 & (STATUS2_WRAPPED | STATUS2_ESCAPE_PREVENTION)
        || gBattleTypeFlags & BATTLE_TYPE_ARENA
        || gStatuses3[battler] & STATUS3_ROOTED)
        return FALSE;
    for (i = 0; i < gBattlersCount; i++)
    {
        u8 ability = gBattleMons[i].ability;
        if (GetBattlerSide(i) != GetBattlerSide(battler)
            && (ability == ABILITY_SHADOW_TAG || (ability == ABILITY_ARENA_TRAP && grounded)))
            return FALSE;
        if (i != battler && ability == ABILITY_MAGNET_PULL && steel)
            return FALSE;
    }
    return TRUE;
}

void Gen3_ChooseMove(u8 slot)
{
    sHost.chosenMoveSlot = slot;
    Host_FinishDecision();
    BtlController_EmitTwoReturnValues(B_COMM_TO_ENGINE, B_ACTION_USE_MOVE, 0);
    Host_ControllerExecCompleted();
}

// RUN: in a Frontier battle the game asks whether to forfeit; the answer is YES (a loss).
void Gen3_Forfeit(void)
{
    Host_FinishDecision();
    BtlController_EmitTwoReturnValues(B_COMM_TO_ENGINE, B_ACTION_RUN, 0);
    Host_ControllerExecCompleted();
}

void Gen3_ChooseSwitch(u8 partyIndex)
{
    u8 decision = sHost.pendingDecision;

    sHost.chosenPartyIndex = partyIndex;
    Host_FinishDecision();
    if (decision == GEN3_DECISION_ACTION)
    {
        sHost.switchPending = TRUE;
        BtlController_EmitTwoReturnValues(B_COMM_TO_ENGINE, B_ACTION_SWITCH, 0);
        Host_ControllerExecCompleted();
    }
    else
    {
        Host_BattlePartyMenuChoose(partyIndex);
        BtlController_EmitChosenMonReturnValue(B_COMM_TO_ENGINE, gSelectedMonPartyId, gBattlePartyCurrentOrder);
        Host_ControllerExecCompleted();
    }
}

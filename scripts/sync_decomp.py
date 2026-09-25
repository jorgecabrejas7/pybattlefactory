#!/usr/bin/env python3
"""Vendor the pokeemerald code the gen3 battle core is built from.

third_party/pokeemerald/ receives, verbatim:
  include/                  all decomp headers (a few are patched for the host build, see HEADER_PATCHES)
  src/<file>.c              whole battle-logic source files (SOURCE_FILES)
  src/<data>.h              data tables those files #include
  src/host_extracted.c      individual logic functions/data pulled out of UI-heavy files
                            (EXTRACT), plus the same-file static helpers they depend on

Everything else the vendored code references (graphics, sound, text windows, link cable,
save file...) is stubbed by scripts/gen_stubs.py.

Usage: sync_decomp.py [--pokeemerald ~/Dev/pokeemerald]
"""

import argparse
import os
import re
import shutil

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DST = os.path.join(REPO, "third_party", "pokeemerald")

SOURCE_FILES = [
    "battle_main", "battle_script_commands", "battle_util", "battle_util2", "battle_ai_script_commands",
    "battle_ai_switch_items", "battle_controllers", "battle_message", "random", "pokemon", "item",
    "battle_factory", "battle_tower", "frontier_util", "string_util", "malloc",
]

DATA_HEADERS = [
    "data/text/abilities.h", "data/text/item_descriptions.h", "data/items.h", "data/battle_moves.h",
    "data/battle_frontier/battle_frontier_trainer_mons.h", "data/battle_frontier/battle_frontier_trainers.h",
    "data/battle_frontier/battle_frontier_mons.h", "data/battle_frontier/battle_tent.h",
    "data/pokemon/item_effects.h", "data/pokemon/tmhm_learnsets.h", "data/pokemon/trainer_class_lookups.h",
    "data/pokemon/cry_ids.h", "data/pokemon/experience_tables.h", "data/pokemon/species_info.h",
    "data/pokemon/level_up_learnsets.h", "data/pokemon/evolution.h", "data/pokemon/level_up_learnset_pointers.h",
    "data/text/species_names.h", "data/text/move_names.h", "data/text/trainer_class_names.h",
]

# Generated file -> (decomp source, names to copy with their same-file static dependencies,
#                   {name: replacement code} for helpers that must not be copied, code appended at the end)
EXTRACT = {
    "host_extracted.c": [
        ("battle_anim_mons.c", ["GetBattlerSide", "GetBattlerPosition", "GetBattlerAtPosition", "IsDoubleBattle"],
         {}, ""),
        ("util.c", ["gBitTable", "CountTrailingZeroBits"], {}, ""),
        ("battle_interface.c", ["GetScaledHPFraction"], {}, ""),
        ("new_game.c", ["CopyTrainerId", "SetTrainerId"], {}, ""),
        ("recorded_battle.c", ["gBattlePalaceMoveSelectionRngValue"], {}, ""),
        ("script_pokemon_util.c", ["ReducePlayerPartyToSelectedMons"], {}, ""),
        ("battle_gfx_sfx_util.c", ["AllocateBattleSpritesData", "FreeBattleSpritesData"], {}, ""),
        ("main.c", ["gMain", "SetMainCallback2"], {}, ""),
        ("script_pokemon_util.c", ["HealPlayerParty"], {}, ""),
        ("berry.c", ["gBerries", "GetBerryInfo", "ItemIdToBerryType", "IsEnigmaBerryValid", "GetEnigmaBerryChecksum"],
         {}, ""),
        ("load_save.c", ["gSaveBlock1Ptr", "gSaveBlock2Ptr"], {}, ""),
    ],
    "host_party_menu.c": [
        ("party_menu.c", ["gBattlePartyCurrentOrder", "gSelectedOrderFromParty", "gSelectedMonPartyId",
                          "gPartyMenuUseExitCallback", "GetPartyIdFromBattlePartyId",
                          "BufferBattlePartyCurrentOrderBySide", "SwitchPartyMonSlots", "SwitchPartyOrderLinkMulti",
                          "ClearSelectedPartyOrder", "BufferBattlePartyCurrentOrder", "UpdatePartyToBattleOrder",
                          "UpdatePartyToFieldOrder", "SwapPartyPokemon", "GetPartyIdFromBattleSlot",
                          "GetMonNickname"], {},
         """
// Host: the state changes of choosing `partyIndex` in the in-battle party menu
// (OpenPartyMenuInBattle -> TrySwitchInPokemon -> Task_ClosePartyMenuAndSetCB2), minus UI.
void Host_BattlePartyMenuChoose(u8 partyIndex)
{
    u8 slot, newSlot;

    UpdatePartyToBattleOrder();
    for (slot = 0; slot < PARTY_SIZE; slot++)
        if (GetPartyIdFromBattleSlot(slot) == partyIndex)
            break;
    gSelectedMonPartyId = GetPartyIdFromBattleSlot(slot);
    gPartyMenuUseExitCallback = TRUE;
    newSlot = GetPartyIdFromBattlePartyId(gBattlerPartyIndexes[gBattlerInMenuId]);
    SwitchPartyMonSlots(newSlot, slot);
    SwapPartyPokemon(&gPlayerParty[newSlot], &gPlayerParty[slot]);
    UpdatePartyToFieldOrder();
}
"""),
    ],
    "host_factory_screen.c": [
        ("battle_factory_screen.c",
         ["CreateFrontierFactorySelectableMons", "Select_CopyMonsToPlayerParty", "Select_AreSpeciesValid",
          "CopySwappedMonData", "Swap_AlreadyHasSameSpecies"], {},
         """
// Host: the rental select screen and the swap screen without their UI.
EWRAM_DATA static struct FactorySelectScreen sHostSelectScreen = {0};
EWRAM_DATA static struct FactorySwapScreen sHostSwapScreen = {0};

// factory_rentmons: build the 6 rental Pokemon from rentalMons[] (DoBattleFactorySelectScreen)
void Host_FactoryCreateRentals(void)
{
    memset(&sHostSelectScreen, 0, sizeof(sHostSelectScreen));
    sFactorySelectScreen = &sHostSelectScreen;
    sFactorySelectScreen->selectingMonsState = 1;
    CreateFrontierFactorySelectableMons(0);
}

struct Pokemon *Host_FactoryRental(u8 i)
{
    return &sHostSelectScreen.mons[i].monData;
}

u16 Host_FactoryRentalMonId(u8 i)
{
    return sHostSelectScreen.mons[i].monId;
}

// Rent `slot` as the next party member; FALSE if the game would refuse (same species).
bool8 Host_FactoryRent(u8 slot)
{
    sFactorySelectScreen = &sHostSelectScreen;
    if (sFactorySelectScreen->mons[slot].selectedId != 0
        || !Select_AreSpeciesValid(sFactorySelectScreen->mons[slot].monId))
        return FALSE;
    sFactorySelectScreen->mons[slot].selectedId = sFactorySelectScreen->selectingMonsState++;
    return TRUE;
}

// "Is this team OK?" -> YES (Select_Task_Exit)
void Host_FactoryConfirmRentals(void)
{
    sFactorySelectScreen = &sHostSelectScreen;
    Select_CopyMonsToPlayerParty();
    sFactorySelectScreen = NULL;
}

// factory_swapmons with a swap chosen (Swap_Task_Exit); FALSE if refused (same species).
bool8 Host_FactorySwap(u8 playerSlot, u8 enemySlot)
{
    memset(&sHostSwapScreen, 0, sizeof(sHostSwapScreen));
    sFactorySwapScreen = &sHostSwapScreen;
    sFactorySwapScreen->playerMonId = playerSlot;
    if (Swap_AlreadyHasSameSpecies(enemySlot))
    {
        sFactorySwapScreen = NULL;
        return FALSE;
    }
    sFactorySwapScreen->monSwapped = TRUE;
    sFactorySwapScreen->enemyMonId = enemySlot;
    CopySwappedMonData();
    sFactorySwapScreen = NULL;
    return TRUE;
}
"""),
    ],
    "host_ctrl_player.c": [
        ("battle_controller_player.c",
         ["PlayerHandleGetMonData", "PlayerHandleGetRawMonData", "PlayerHandleSetMonData", "PlayerHandleSetRawMonData"],
         {"PlayerBufferExecCompleted": "static void PlayerBufferExecCompleted(void) { Host_ControllerExecCompleted(); }"},
         """
void Host_PlayerGetMonData(void) { PlayerHandleGetMonData(); }
void Host_PlayerGetRawMonData(void) { PlayerHandleGetRawMonData(); }
void Host_PlayerSetMonData(void) { PlayerHandleSetMonData(); }
void Host_PlayerSetRawMonData(void) { PlayerHandleSetRawMonData(); }
"""),
    ],
    "host_ctrl_opponent.c": [
        ("battle_controller_opponent.c",
         ["OpponentHandleGetMonData", "OpponentHandleGetRawMonData", "OpponentHandleSetMonData",
          "OpponentHandleSetRawMonData", "OpponentHandleChooseAction", "OpponentHandleChooseMove",
          "OpponentHandleChooseItem", "OpponentHandleChoosePokemon"],
         {"OpponentBufferExecCompleted": "static void OpponentBufferExecCompleted(void) { Host_ControllerExecCompleted(); }"},
         """
void Host_OpponentGetMonData(void) { OpponentHandleGetMonData(); }
void Host_OpponentGetRawMonData(void) { OpponentHandleGetRawMonData(); }
void Host_OpponentSetMonData(void) { OpponentHandleSetMonData(); }
void Host_OpponentSetRawMonData(void) { OpponentHandleSetRawMonData(); }
void Host_OpponentChooseAction(void) { OpponentHandleChooseAction(); }
void Host_OpponentChooseMove(void) { OpponentHandleChooseMove(); }
void Host_OpponentChooseItem(void) { OpponentHandleChooseItem(); }
void Host_OpponentChoosePokemon(void) { OpponentHandleChoosePokemon(); }
"""),
    ],
}

EXTRACT_INCLUDES = [
    "global.h", "battle.h", "battle_ai_script_commands.h", "battle_ai_switch_items.h", "battle_anim.h",
    "battle_controllers.h", "battle_interface.h", "battle_message.h", "battle_setup.h", "battle_tv.h",
    "data.h", "event_data.h", "item.h", "new_game.h", "party_menu.h", "pokemon.h", "random.h",
    "recorded_battle.h", "script_pokemon_util.h", "string_util.h", "util.h", "malloc.h",
    "constants/battle_ai.h", "constants/battle_frontier.h", "constants/items.h", "constants/moves.h",
    "constants/party_menu.h", "berry.h", "item_menu.h", "constants/berry.h",
    "battle_factory.h", "battle_factory_screen.h", "battle_tower.h", "frontier_util.h", "constants/battle_frontier_mons.h",
]

# Host-build patches to vendored headers: (path, old, new)
HEADER_PATCHES = [
    ("include/gba/defines.h",
     '#define IWRAM_DATA __attribute__((section("iwram_data")))\n'
     '#define EWRAM_DATA __attribute__((section("ewram_data")))\n'
     '#define COMMON_DATA __attribute__((section("common_data")))',
     '#ifdef GEN3_HOST\n'
     '// pybattle host build: all battle RAM lives in one section so a battle instance can be\n'
     '// saved, restored and cloned as a single block (see src/gen3/host.c).\n'
     '#define IWRAM_DATA __attribute__((section("gen3_ram")))\n'
     '#define EWRAM_DATA __attribute__((section("gen3_ram")))\n'
     '#define COMMON_DATA __attribute__((section("gen3_ram")))\n'
     '#else\n'
     '#define IWRAM_DATA __attribute__((section("iwram_data")))\n'
     '#define EWRAM_DATA __attribute__((section("ewram_data")))\n'
     '#define COMMON_DATA __attribute__((section("common_data")))\n'
     '#endif'),
    ("include/global.h",
     "#define T1_READ_PTR(ptr) (u8 *) T1_READ_32(ptr)",
     "#ifdef GEN3_HOST\n"
     "// Scripts store 32-bit GBA addresses; translate them to host pointers.\n"
     "void *Gen3_ResolveAddress(u32 gbaAddress);\n"
     "#define T1_READ_PTR(ptr) ((u8 *) Gen3_ResolveAddress(T1_READ_32(ptr)))\n"
     "#else\n"
     "#define T1_READ_PTR(ptr) (u8 *) T1_READ_32(ptr)\n"
     "#endif"),
    ("include/global.h",
     "#define T2_READ_PTR(ptr) (void *) T2_READ_32(ptr)",
     "#ifdef GEN3_HOST\n"
     "#define T2_READ_PTR(ptr) Gen3_ResolveAddress(T2_READ_32(ptr))\n"
     "#else\n"
     "#define T2_READ_PTR(ptr) (void *) T2_READ_32(ptr)\n"
     "#endif"),
    ("include/gba/defines.h", "#define UNUSED __attribute__((unused))",
     "#define UNUSED __attribute__((unused))\n\n"
     "#ifdef GEN3_HOST\n#define GEN3_APCS __attribute__((aligned(4))) // see scripts/sync_decomp.py\n"
     "#else\n#define GEN3_APCS\n#endif"),
    # GBA hardware (palette RAM, VRAM, OAM, IO registers, BIOS-reserved IWRAM) -> host dummy buffers
    ("include/gba/defines.h", "#define SOUND_INFO_PTR (*(struct SoundInfo **)0x3007FF0)\n"
     "#define INTR_CHECK     (*(u16 *)0x3007FF8)\n#define INTR_VECTOR    (*(void **)0x3007FFC)",
     "#ifdef GEN3_HOST\n#include <stdint.h>\nextern unsigned char gGen3FakeHardware[];\n"
     "#define GEN3_FAKE_HW(off) ((uintptr_t)gGen3FakeHardware + (off))\n"
     "#define SOUND_INFO_PTR (*(struct SoundInfo **)GEN3_FAKE_HW(0x70000))\n"
     "#define INTR_CHECK     (*(u16 *)GEN3_FAKE_HW(0x70008))\n#define INTR_VECTOR    (*(void **)GEN3_FAKE_HW(0x70010))\n"
     "#else\n#define SOUND_INFO_PTR (*(struct SoundInfo **)0x3007FF0)\n"
     "#define INTR_CHECK     (*(u16 *)0x3007FF8)\n#define INTR_VECTOR    (*(void **)0x3007FFC)\n#endif"),
    ("include/gba/defines.h", "#define PLTT          0x5000000",
     "#ifdef GEN3_HOST\n#define PLTT          GEN3_FAKE_HW(0x40000)\n#else\n#define PLTT          0x5000000\n#endif"),
    ("include/gba/defines.h", "#define VRAM      0x6000000",
     "#ifdef GEN3_HOST\n#define VRAM      GEN3_FAKE_HW(0x0)\n#else\n#define VRAM      0x6000000\n#endif"),
    ("include/gba/defines.h", "#define OAM      0x7000000",
     "#ifdef GEN3_HOST\n#define OAM      GEN3_FAKE_HW(0x50000)\n#else\n#define OAM      0x7000000\n#endif"),
    ("include/gba/io_reg.h", "#define REG_BASE 0x4000000 // I/O register base address",
     "#ifdef GEN3_HOST\n#define REG_BASE GEN3_FAKE_HW(0x60000)\n#else\n#define REG_BASE 0x4000000 // I/O register base address\n#endif"),
    ("include/malloc.h", "#define HEAP_SIZE 0x1C000",
     "#ifdef GEN3_HOST\n#define HEAP_SIZE 0xA000 // battle allocations only; lives in the swappable battle RAM\n"
     "#else\n#define HEAP_SIZE 0x1C000\n#endif"),
    # Host text: the build has no charmap preprocessor, so strings stay ASCII, but they must end
    # with EOS (0xFF) like game text or StringCopy & co. run past them.
    ("include/global.h",
     "#if defined(__APPLE__) || defined(__CYGWIN__) || defined(__INTELLISENSE__)\n"
     "// We define these when using certain IDEs to fool preproc\n"
     "#define _(x)        {x}\n"
     "#define __(x)       {x}",
     "#if defined(__APPLE__) || defined(__CYGWIN__) || defined(__INTELLISENSE__) || defined(GEN3_HOST)\n"
     "// We define these when using certain IDEs to fool preproc\n"
     "#ifdef GEN3_HOST\n#define _(x)        {x \"\\xFF\"}\n#define __(x)       {x \"\\xFF\"}\n#else\n"
     "#define _(x)        {x}\n"
     "#define __(x)       {x}\n#endif"),
]

# agbcc (the compiler the ROM was built with) follows the old ARM APCS rules: every struct and
# union is at least 4-byte aligned and its size a multiple of 4. Host compilers pack small
# structs tighter, which would shift fields the scripts and the emulator address by offset.
STRUCT_DEF = re.compile(r"\b(struct|union)(\s+)(?!GEN3_APCS\b)(\w+\s*)?(?=\{)")


def apcs_align(text):
    out, pos = [], 0
    for m in STRUCT_DEF.finditer(text):
        # find the matching closing brace; packed structs keep their packed layout
        depth, i = 0, m.end()
        while i < len(text):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        if re.match(r"\s*__attribute__\s*\(\(\s*packed", text[i + 1:i + 40]):
            continue
        out.append(text[pos:m.start()])
        out.append(f"{m.group(1)} GEN3_APCS {m.group(3) or ''}")
        pos = m.end()
    out.append(text[pos:])
    return "".join(out)


# Code appended to vendored sources (host entry points that need file-static functions)
SOURCE_APPENDS = {
    "battle_tower.c": """
#ifdef GEN3_HOST
// Task_StartBattleAfterTransition without the transition: start the battle DoSpecialTrainerBattle
// set up, ending in HandleSpecialTrainerBattleEnd like the game.
void Host_StartSpecialTrainerBattle(void)
{
    gMain.savedCallback = HandleSpecialTrainerBattleEnd;
    SetMainCallback2(CB2_InitBattle);
}
#endif
""",
}

# Host-build patches to vendored sources: (file, old, new)
SOURCE_PATCHES = [
    # IWRAM static without a section attribute: move it into the swappable battle RAM.
    ("src/battle_factory.c", "static bool8 sPerformedRentalSwap;", "IWRAM_DATA static bool8 sPerformedRentalSwap;"),
    ("src/random.c", "    gRngValue = ISO_RANDOMIZE1(gRngValue);\n    sRandCount++;",
     "    gRngValue = ISO_RANDOMIZE1(gRngValue);\n    sRandCount++;\n#ifdef GEN3_HOST\n"
     "    { extern void (*gGen3RandomHook)(void *); if (gGen3RandomHook) gGen3RandomHook(__builtin_return_address(0)); }\n"
     "#endif"),
    # The opponent's controller passes NULL here (the GBA reads BIOS open-bus bytes); the
    # bytes are only used in link multi battles.
    ("src/battle_controllers.c",
     "    for (i = 0; i < (int)ARRAY_COUNT(gBattlePartyCurrentOrder); i++)\n        sBattleBuffersTransferData[2 + i] = battlePartyOrder[i];",
     "    for (i = 0; i < (int)ARRAY_COUNT(gBattlePartyCurrentOrder); i++)\n#ifdef GEN3_HOST\n"
     "        sBattleBuffersTransferData[2 + i] = battlePartyOrder ? battlePartyOrder[i] : 0;\n#else\n"
     "        sBattleBuffersTransferData[2 + i] = battlePartyOrder[i];\n#endif"),
    ("src/malloc.c", "void *Alloc(u32 size)\n{\n    return AllocInternal(sHeapStart, size);\n}",
     "void *Alloc(u32 size)\n{\n#ifdef GEN3_HOST\n    void *p = AllocInternal(sHeapStart, size);\n"
     "    extern void Gen3_OutOfMemory(u32 size);\n    if (!p)\n        Gen3_OutOfMemory(size);\n    return p;\n"
     "#else\n    return AllocInternal(sHeapStart, size);\n#endif\n}"),
    ("src/malloc.c", "void *AllocZeroed(u32 size)\n{\n    return AllocZeroedInternal(sHeapStart, size);\n}",
     "void *AllocZeroed(u32 size)\n{\n#ifdef GEN3_HOST\n    void *p = AllocZeroedInternal(sHeapStart, size);\n"
     "    extern void Gen3_OutOfMemory(u32 size);\n    if (!p)\n        Gen3_OutOfMemory(size);\n    return p;\n"
     "#else\n    return AllocZeroedInternal(sHeapStart, size);\n#endif\n}"),
    ("src/malloc.c", "static void *sHeapStart;\nstatic u32 sHeapSize;",
     "IWRAM_DATA static void *sHeapStart;\nIWRAM_DATA static u32 sHeapSize;"),
]

# ---------------------------------------------------------------------------
# Minimal top-level C splitter
# ---------------------------------------------------------------------------

IDENT = re.compile(r"[A-Za-z_]\w*")


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def top_level_items(text):
    """Yield (name, source) for top-level definitions; preprocessor lines yield (None, line)."""
    code = strip_comments(text)
    i, n = 0, len(code)
    start = 0
    depth = 0
    while i < n:
        c = code[i]
        if depth == 0 and c == "#" and code[start:i].strip() == "":
            end = i
            while True:
                end = code.find("\n", end)
                if end == -1:
                    end = n
                    break
                if code[end - 1] != "\\":
                    break
                end += 1
            yield None, code[i:end]
            i = start = end + 1
            continue
        if c in "\"'":
            q = c
            i += 1
            while i < n and code[i] != q:
                i += 2 if code[i] == "\\" else 1
        elif c in "{(":
            depth += 1
        elif c in "})":
            depth -= 1
            if depth == 0 and c == "}":
                # function body ends here unless this closes an initializer (followed by ';')
                j = i + 1
                while j < n and code[j] in " \t\n":
                    j += 1
                if j >= n or code[j] != ";":
                    item = code[start:i + 1]
                    yield item_name(item), item
                    start = i + 1
        elif c == ";" and depth == 0:
            item = code[start:i + 1]
            yield item_name(item), item
            start = i + 1
        i += 1


def item_name(item):
    head = item.split("{", 1)[0].split("=", 1)[0]
    if "(" in head and not re.search(r"\(\s*\*", head):
        head = head.split("(", 1)[0]          # function: name before '('
    else:
        head = re.split(r"[\[;]", head, 1)[0]  # data: name before '[' / ';'
    names = IDENT.findall(head)
    return names[-1] if names else None


def extract(path, seeds, replace=None):
    replace = replace or {}
    text = open(path).read()
    items = list(top_level_items(text))
    defs = {}
    for name, src in items:
        if name and "(" in src.split("{", 1)[0] and src.rstrip().endswith(";") and "=" not in src:
            continue  # prototype
        if name:
            defs.setdefault(name, src)
    wanted, todo = [], list(seeds)
    while todo:
        name = todo.pop()
        if name in wanted or name not in defs or name in replace:
            continue
        wanted.append(name)
        for ident in set(IDENT.findall(defs[name])):
            if ident in defs and ident not in wanted and (
                    re.match(r"\s*(static|EWRAM_DATA static)", defs[ident])
                    or re.match(r"\s*(struct|union|enum)\s+(GEN3_APCS\s+)?\w+\s*\{", defs[ident])):
                todo.append(ident)
    missing = [s for s in list(seeds) + list(replace) if s not in defs]
    if missing:
        raise SystemExit(f"{path}: not found: {missing}")
    defines = [src for name, src in items if name is None and src.lstrip().startswith(("#define", "#undef"))]
    order = [name for name, _ in items if name in wanted]
    order = list(dict.fromkeys(order))

    def kind(name):   # types, then data, then functions
        src = defs[name].lstrip()
        if re.match(r"(struct|union|enum)\s+(GEN3_APCS\s+)?\w+\s*\{", src):
            return 0
        header = src.split("{", 1)[0]
        return 2 if "(" in header and src.rstrip().endswith("}") and "=" not in header else 1
    order.sort(key=kind)
    protos = []
    for name in order:
        src = defs[name]
        header = src.split("{", 1)[0].strip()
        if "(" in header and src.rstrip().endswith("}") and header.startswith("static"):
            protos.append(header + ";")
    return defines, protos, [defs[name].strip() for name in order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pokeemerald", default=os.path.expanduser("~/Dev/pokeemerald"))
    args = ap.parse_args()
    src_root = args.pokeemerald

    if os.path.exists(os.path.join(DST, "include")):
        shutil.rmtree(os.path.join(DST, "include"))
    shutil.copytree(os.path.join(src_root, "include"), os.path.join(DST, "include"))
    for root, _, files in os.walk(os.path.join(DST, "include")):
        for fname in files:
            if fname.endswith(".h") and not root.endswith("constants"):
                full = os.path.join(root, fname)
                text = open(full, errors="replace").read()
                patched = apcs_align(text)
                if patched != text:
                    open(full, "w").write(patched)
    for path, old, new in HEADER_PATCHES:
        full = os.path.join(DST, path)
        text = open(full).read()
        if old not in text:
            raise SystemExit(f"patch target not found in {path}")
        open(full, "w").write(text.replace(old, new, 1))

    os.makedirs(os.path.join(DST, "src"), exist_ok=True)
    for name in SOURCE_FILES:
        shutil.copy(os.path.join(src_root, "src", name + ".c"), os.path.join(DST, "src", name + ".c"))
    for path, old, new in SOURCE_PATCHES:
        full = os.path.join(DST, path)
        text = open(full).read()
        if old not in text:
            raise SystemExit(f"patch target not found in {path}")
        open(full, "w").write(text.replace(old, new, 1))
    for name in SOURCE_FILES:
        full = os.path.join(DST, "src", name + ".c")
        text = open(full).read() + SOURCE_APPENDS.get(name + ".c", "")
        open(full, "w").write(apcs_align(text))
    for rel in DATA_HEADERS:
        os.makedirs(os.path.dirname(os.path.join(DST, "src", rel)), exist_ok=True)
        shutil.copy(os.path.join(src_root, "src", rel), os.path.join(DST, "src", rel))

    n_defs = 0
    for out_name, parts in EXTRACT.items():
        out = ["// GENERATED by scripts/sync_decomp.py: logic pulled verbatim out of UI-heavy decomp files.",
               *[f'#include "{h}"' for h in EXTRACT_INCLUDES], "",
               "void Host_ControllerExecCompleted(void);", ""]
        for fname, seeds, replace, append in parts:
            defines, protos, bodies = extract(os.path.join(src_root, "src", fname), seeds, replace)
            n_defs += len(bodies)
            used = [d for d in defines if len(IDENT.findall(d)) > 1
                    and any(re.search(r"\b" + re.escape(IDENT.findall(d)[1]) + r"\b", b) for b in bodies)]
            out.append(f"// ---- from src/{fname}")
            out += used
            out += protos + list(replace.values()) + bodies
            out.append(append)
            out += [f"#undef {IDENT.findall(d)[1]}" for d in used if d.lstrip().startswith("#define")]
            out.append("")
        text = apcs_align("\n".join(out) + "\n")
        # file-scope statics without a section attribute live in IWRAM on the GBA; keep them in
        # the swappable battle RAM too
        text = re.sub(r"^static (?!(?:const|inline)\b)([^(\n]*?[;=][^\n]*)$",
                      lambda m: "IWRAM_DATA static " + m.group(1) if "(" not in m.group(1).split("=")[0] else m.group(0),
                      text, flags=re.M)
        open(os.path.join(DST, "src", out_name), "w").write(text)
    print("vendored", len(SOURCE_FILES), "sources,", len(DATA_HEADERS), "data headers; extracted",
          n_defs, "definitions")


if __name__ == "__main__":
    main()

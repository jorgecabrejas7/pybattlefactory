#!/usr/bin/env python3
"""Extract the battle and battle-AI scripts from pokeemerald.elf for the gen3 host build.

The scripts run exactly as in the ROM: their bytes are embedded verbatim and every label C
refers to becomes an address constant pointing into the blob. Scripts contain 32-bit GBA addresses (jump
targets, RAM variables); at runtime Gen3_ResolveAddress maps them to host pointers using the
tables generated here.

Outputs (src/gen3/generated/):
  rom_scripts.bin         script bytes (reference copy; the build uses rom_scripts_data.c)
  rom_scripts_data.c      the blob as a C array (Gen3_RomScripts)
  rom_script_labels.h     #define Label (Gen3_RomScripts + offset) for every label C declares
  rom_script_tables.c     host-pointer versions of the script pointer tables, ROM window bounds
  rom_ram_map.c           GBA RAM address -> host variable map (gGen3RamSymbols)
  battle_placeholders.h   battle text placeholder codes (charmap.txt), for the host's message text

Everything is plain C (no assembly), so the same files build natively and with emscripten.

Usage: extract_rom_scripts.py [--pokeemerald ~/Dev/pokeemerald]
"""

import argparse
import os
import re
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_game_data import Rom  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "src", "gen3", "generated")
VENDORED_SRC = os.path.join(REPO, "third_party", "pokeemerald", "src")

SCRIPT_OBJECTS = ["data/battle_scripts_1.o", "data/battle_scripts_2.o", "data/battle_ai_scripts.o"]

# Pointer tables inside the script data. C code indexes them as `const u8 *const x[]`, so they
# must be host-pointer arrays rather than blob bytes.
POINTER_TABLES = ["gBattleScriptsForMoveEffects", "gBattlescriptsForBallThrow", "gBattlescriptsForUsingItem",
                  "gBattlescriptsForRunningByItem", "gBattlescriptsForSafariActions", "gBattleAI_ScriptsTable"]


# Pointers are 8 bytes on the host; these hold pointers (or UI structs containing them) and are
# never compared raw or addressed by scripts.
HOST_SIZE_DIFFERS = {
    "gBattleAnimBgTileBuffer", "gBattleAnimBgTilemapBuffer", "gBattlescriptCurrInstr", "gSelectionBattleScripts",
    "gPalaceSelectionBattleScripts", "gBattleStruct", "gLinkBattleSendBuffer", "gLinkBattleRecvBuffer",
    "gBattleResources", "gBattleSpritesDataPtr", "gMonSpritesGfxPtr", "gBattleControllerOpponentHealthboxData",
    "gBattleControllerOpponentFlankHealthboxData", "gMultiuseSpriteTemplate", "gBagPockets", "gBattleMsgDataPtr",
    "gFacilityTrainers", "gFacilityTrainerMons", "gSaveBlock1Ptr", "gSaveBlock2Ptr", "gMain", "gAIScriptPtr",
    "gBattleMainFunc", "gBattlerControllerFuncs", "gPreBattleCallback1",
}


def script_ranges(map_path):
    text = open(map_path).read()
    ranges = []
    for obj in SCRIPT_OBJECTS:
        m = re.search(r"script_data\s+0x([0-9a-f]+)\s+0x([0-9a-f]+)\s+" + re.escape(obj), text)
        if not m:
            raise SystemExit(f"{obj} not found in map")
        ranges.append((int(m.group(1), 16), int(m.group(2), 16)))
    return ranges


def all_symbols(rom):
    """name -> address for every symbol (any type) in the ELF."""
    from elftools.elf.sections import SymbolTableSection
    out = {}
    for section in rom.elf.iter_sections():
        if isinstance(section, SymbolTableSection):
            for sym in section.iter_symbols():
                if sym.name and sym["st_info"]["type"] in ("STT_NOTYPE", "STT_OBJECT"):
                    out.setdefault(sym.name, sym["st_value"])
    return out


def vendored_ram_globals():
    """Non-static RAM globals defined by the vendored sources (EWRAM/IWRAM/COMMON_DATA)."""
    names = set()
    decl = re.compile(r"^(?:EWRAM_DATA|IWRAM_DATA|COMMON_DATA)\s+(?!static)[^=;(]*?\b(\w+)\s*(?:\[[^\]]*\]\s*)*(?:=|;)",
                      re.M)
    for fname in os.listdir(VENDORED_SRC):
        if fname.endswith(".c"):
            names.update(decl.findall(open(os.path.join(VENDORED_SRC, fname)).read()))
    return names


def vendored_const_arrays():
    """Non-static const integer arrays defined by the vendored sources (string-id tables etc.),
    which scripts may address directly."""
    decl = re.compile(r"^const\s+(?:u8|u16|s8|s16|u32|s32)\s+(\w+)\s*\[[^\]]*\]\s*=", re.M)
    names = set()
    for fname in os.listdir(VENDORED_SRC):
        if fname.endswith(".c"):
            names.update(decl.findall(open(os.path.join(VENDORED_SRC, fname)).read()))
    return names


def vendored_script_externs():
    """Script labels the vendored headers declare as `extern const u8 Label[];`."""
    path = os.path.join(REPO, "third_party", "pokeemerald", "include", "battle_scripts.h")
    return set(re.findall(r"^extern const u8 (\w+)\[\];", open(path).read(), re.M))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pokeemerald", default=os.path.expanduser("~/Dev/pokeemerald"))
    args = ap.parse_args()
    rom = Rom(os.path.join(args.pokeemerald, "pokeemerald.elf"))
    ranges = script_ranges(os.path.join(args.pokeemerald, "pokeemerald.map"))
    base = min(a for a, _ in ranges)
    end = max(a + n for a, n in ranges)
    blob = rom.read(base, end - base)
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "rom_scripts.bin"), "wb").write(blob)

    syms = all_symbols(rom)
    labels = {name: addr for name, addr in syms.items()
              if base <= addr < end and re.fullmatch(r"[A-Za-z_]\w*", name) and not name.startswith("$")}
    starts = sorted(set(labels.values()) | {end})

    # --- C: blob + label address constants ----------------------------------------------
    data = ["// GENERATED by scripts/extract_rom_scripts.py. Do not edit.",
            "typedef unsigned char u8;",
            f"const u8 Gen3_RomScripts[{len(blob)}] __attribute__((aligned(4))) = {{"]
    for i in range(0, len(blob), 24):
        data.append("    " + ",".join(str(b) for b in blob[i:i + 24]) + ",")
    data.append("};")
    open(os.path.join(OUT, "rom_scripts_data.c"), "w").write("\n".join(data) + "\n")

    addr_to_label = {}
    for name, addr in labels.items():
        if name not in POINTER_TABLES:
            addr_to_label.setdefault(addr, name)
    table_labels = set()
    for table in POINTER_TABLES:
        addr = labels[table]
        stop = next(s for s in starts if s > addr)
        entries = struct.unpack(f"<{(stop - addr) // 4}I", rom.read(addr, stop - addr - (stop - addr) % 4))
        table_labels |= {addr_to_label[e] for e in entries}
    # Only labels C names get a macro (others, e.g. sMovesTable, may clash with C identifiers).
    declared = vendored_script_externs() | table_labels
    hdr = ["// GENERATED by scripts/extract_rom_scripts.py. Do not edit.",
           "// Battle script labels as addresses inside the embedded script blob. Included at the end",
           "// of battle_scripts.h, after its GBA `extern const u8 Label[];` declarations.",
           "#ifndef GEN3_ROM_SCRIPT_LABELS_H", "#define GEN3_ROM_SCRIPT_LABELS_H",
           "extern const unsigned char Gen3_RomScripts[];"]
    for name, addr in sorted(labels.items(), key=lambda kv: (kv[1], kv[0])):
        if name in declared and name not in POINTER_TABLES:
            hdr.append(f"#define {name} (Gen3_RomScripts + {addr - base:#x})")
    hdr.append("#endif")
    open(os.path.join(OUT, "rom_script_labels.h"), "w").write("\n".join(hdr) + "\n")

    # --- C: pointer tables + address maps ------------------------------------------------
    c = ["// GENERATED by scripts/extract_rom_scripts.py. Do not edit.",
         '#include "global.h"', '#include "gen3/generated/rom_script_labels.h"', "",
         f"const u32 gGen3RomScriptsBase = {base:#010x};",
         f"const u32 gGen3RomScriptsSize = {end - base:#x};", ""]
    for table in POINTER_TABLES:
        addr = labels[table]
        stop = next(s for s in starts if s > addr)
        entries = struct.unpack(f"<{(stop - addr) // 4}I", rom.read(addr, stop - addr - (stop - addr) % 4))
        c.append(f"const u8 *const {table}[] = {{")
        c += [f"    {addr_to_label[e]}," for e in entries]
        c.append("};\n")

    ram = vendored_ram_globals()
    ram_syms = sorted((rom.symbols[n][0], rom.symbols[n][1], n) for n in ram if n in rom.symbols)
    const_syms = sorted((rom.symbols[n][0], rom.symbols[n][1], n) for n in vendored_const_arrays()
                        if n in rom.symbols and rom.symbols[n][1])
    # RAM map: it only needs the symbols' addresses, so this file includes no headers and
    # declares every symbol as a byte array (pointer width follows the target).
    cram = ["// GENERATED by scripts/extract_rom_scripts.py. Do not edit.",
            "struct Gen3RamSymbol { unsigned int gbaAddress; unsigned int size; void *host; };"]
    cram += [f"extern unsigned char {n}[];" for _, _, n in sorted(ram_syms + const_syms, key=lambda t: t[2])]
    cram.append("const struct Gen3RamSymbol gGen3RamSymbols[] = {")
    cram += [f"    {{{a:#010x}, {sz:#x}, {n}}}," for a, sz, n in ram_syms + const_syms]
    cram += ["};", f"const unsigned int gGen3RamSymbolsCount = {len(ram_syms) + len(const_syms)};"]
    open(os.path.join(OUT, "rom_ram_map.c"), "w").write("\n".join(cram) + "\n")
    open(os.path.join(OUT, "rom_script_tables.c"), "w").write("\n".join(c) + "\n")

    # Battle text placeholders ({B_ATK_NAME_WITH_PREFIX} = FD 0F): the host build has no charmap preprocessor,
    # so its battle strings keep them spelled out; src/gen3/host.c encodes them for display (Gen3_BattleMessageText).
    ph = re.findall(r"^(B_\w+)\s*=\s*FD ([0-9A-Fa-f]{2})\s*(?:@.*)?$", open(os.path.join(args.pokeemerald, "charmap.txt")).read(), re.M)
    hp = ["// GENERATED by scripts/extract_rom_scripts.py from charmap.txt. Do not edit.",
          "static const struct { const char *name; unsigned char code; } sGen3BattlePlaceholders[] = {"]
    hp += [f'    {{"{n}", 0x{c.lower()}}},' for n, c in ph]
    hp.append("};")
    open(os.path.join(OUT, "battle_placeholders.h"), "w").write("\n".join(hp) + "\n")

    # Compile-time layout check: every battle RAM variable must have its GBA size on the host.
    inc_dir = os.path.join(REPO, "third_party", "pokeemerald", "include")
    declared, headers = set(), set()
    for root, _, files in os.walk(inc_dir):
        for fname in files:
            if fname.endswith(".h"):
                text = open(os.path.join(root, fname), errors="replace").read()
                for n, dims in re.findall(r"^extern[^;(]*?\b(\w+)\s*((?:\[[^\]]*\]\s*)*);", text, re.M):
                    if not re.search(r"\[\s*\]", dims):   # incomplete arrays have no sizeof
                        declared.add(n)
                    headers.add(os.path.relpath(os.path.join(root, fname), inc_dir))
    chk = ["// GENERATED by scripts/extract_rom_scripts.py. Do not edit.",
           "// Host sizes of battle RAM variables must equal their sizes in pokeemerald.elf.",
           '#include "global.h"']
    chk += [f'#include "{h}"' for h in sorted(headers)
            if ("/" not in h or h.startswith("gba/")) and h not in ("AgbRfu_LinkManager.h", "link_rfu.h")]
    chk += [f'_Static_assert(sizeof({n}) == {sz:#x}, "{n}: host layout differs from GBA");'
            for _, sz, n in ram_syms if n in declared and n not in HOST_SIZE_DIFFERS]
    open(os.path.join(OUT, "rom_layout_check.c"), "w").write("\n".join(chk) + "\n")
    print(f"scripts {base:#x}-{end:#x} ({end - base} bytes), {len(labels)} labels, {len(ram_syms)} RAM symbols, "
          f"{len(const_syms)} ROM tables")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Extract game data straight from a built pokeemerald ELF.

Tables are read as bytes from the ELF (so values are bit-exact with the ROM), and
constant names come from pokeemerald's headers. Everything is keyed by the game's
own internal IDs, which is what the emulator bridge reads from RAM.

Outputs (all generated, do not edit by hand):
  include/gen/game_constants.hpp   SPECIES_/MOVE_/ITEM_/ABILITY_/HOLD_EFFECT_ enums
  include/gen/move_effects.inc     body of `enum class MoveEffect`
  src/data/species_data.cpp        gSpeciesInfo
  src/data/move_data.cpp           gBattleMoves
  src/data/item_data.cpp           gItems hold effects
  src/data/frontier_mons.cpp       gBattleFrontierMons + gBattleFrontierHeldItems
  src/data/type_chart.cpp          gTypeEffectiveness, gNatureStatTable, gStatStageRatios
  src/data/factory_tables.cpp      Battle Factory / Frontier tables
  pybattle/data/game_data.json     same data for Python

Usage: extract_game_data.py [--pokeemerald ~/Dev/pokeemerald] [--check]
"""

import argparse
import json
import os
import re
import struct
import sys

from elftools.elf.elffile import ELFFile
from elftools.elf.sections import SymbolTableSection

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Header constants
# ---------------------------------------------------------------------------

DEFINE_RE = re.compile(r"^\s*#define\s+(\w+)\s+(.+?)\s*(?://.*)?$")


def parse_defines(paths, prefix_filter=None):
    """Parse simple integer #defines, resolving references to earlier defines."""
    raw = {}
    order = []
    for path in paths:
        with open(path) as f:
            for line in f:
                m = DEFINE_RE.match(line)
                if not m:
                    continue
                name, value = m.group(1), m.group(2).split("/*")[0].strip()
                if name not in raw:
                    order.append(name)
                raw[name] = value

    resolved = {}

    def evaluate(name, depth=0):
        if name in resolved:
            return resolved[name]
        if depth > 50 or name not in raw:
            raise ValueError(name)
        expr = raw[name]
        expr = re.sub(r"\b(\w+)\b",
                      lambda t: str(evaluate(t.group(1), depth + 1))
                      if not re.fullmatch(r"(0x[0-9a-fA-F]+|\d+)", t.group(1)) else t.group(1),
                      expr)
        if not re.fullmatch(r"[\s\d()+\-*/<>|&~x0-9a-fA-F]+", expr):
            raise ValueError(name)
        value = int(eval(expr.replace("/", "//")))
        resolved[name] = value
        return value

    out = {}
    for name in order:
        if prefix_filter and not name.startswith(prefix_filter):
            continue
        try:
            out[name] = evaluate(name)
        except (ValueError, SyntaxError, RecursionError):
            pass
    return out


def names_by_value(defines, prefix, exclude=()):
    """value -> first name with that prefix (keeps game-order aliases stable)."""
    by_value = {}
    for name, value in defines.items():
        if name.startswith(prefix) and name not in exclude and value not in by_value:
            by_value[value] = name
    return by_value


# ---------------------------------------------------------------------------
# ELF access
# ---------------------------------------------------------------------------

class Rom:
    def __init__(self, elf_path):
        self.f = open(elf_path, "rb")
        self.elf = ELFFile(self.f)
        self.symbols = {}          # global name -> (addr, size)
        self.local_symbols = {}    # (source file, name) -> (addr, size)
        self.functions = {}        # name -> address (Thumb bit set, as stored in function pointers)
        current_file = None
        for section in self.elf.iter_sections():
            if not isinstance(section, SymbolTableSection):
                continue
            for sym in section.iter_symbols():
                kind = sym["st_info"]["type"]
                if kind == "STT_FILE":
                    current_file = sym.name
                    continue
                if kind == "STT_FUNC":
                    key = sym.name if sym.name not in self.functions else f"{sym.name}@{current_file}"
                    self.functions[key] = sym["st_value"]
                    continue
                if kind != "STT_OBJECT":
                    continue
                entry = (sym["st_value"], sym["st_size"])
                if sym["st_info"]["bind"] == "STB_LOCAL":
                    self.local_symbols[(current_file, sym.name)] = entry
                else:
                    self.symbols[sym.name] = entry

    def sym(self, name, source=None):
        if source:
            return self.local_symbols[(source, name)]
        if name in self.symbols:
            return self.symbols[name]
        matches = [v for (src, n), v in self.local_symbols.items() if n == name]
        if len(matches) != 1:
            raise KeyError(f"{name}: {len(matches)} matches, pass source=")
        return matches[0]

    def read(self, addr, size):
        for seg in self.elf.iter_segments():
            start = seg["p_vaddr"]
            if start <= addr and addr + size <= start + seg["p_filesz"]:
                self.f.seek(seg["p_offset"] + addr - start)
                return self.f.read(size)
        raise ValueError(f"address {addr:#x} not in a loaded segment")

    def table(self, name, source=None, extra=0):
        addr, size = self.sym(name, source)
        return self.read(addr, size + extra), addr

    def u16_list_at(self, addr, terminator=0):
        out = []
        while True:
            (v,) = struct.unpack("<H", self.read(addr, 2))
            if v == terminator:
                return out
            out.append(v)
            addr += 2


def chunks(data, stride):
    assert len(data) % stride == 0, (len(data), stride)
    return [data[i:i + stride] for i in range(0, len(data), stride)]


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract(root):
    inc = os.path.join(root, "include")
    cst = os.path.join(inc, "constants")
    defs = parse_defines([
        os.path.join(cst, "species.h"), os.path.join(cst, "moves.h"),
        os.path.join(cst, "items.h"), os.path.join(cst, "abilities.h"),
        os.path.join(cst, "battle_move_effects.h"), os.path.join(cst, "hold_effects.h"),
        os.path.join(cst, "pokemon.h"), os.path.join(inc, "battle.h"),
        os.path.join(cst, "battle_frontier.h"), os.path.join(cst, "battle_frontier_mons.h"),
        os.path.join(cst, "battle_ai.h"),
    ])
    rom = Rom(os.path.join(root, "pokeemerald.elf"))
    d = {}

    species_names = names_by_value(defs, "SPECIES_")
    move_names = names_by_value(defs, "MOVE_", exclude={"MOVE_TARGET_SELECTED"} | {n for n in defs if n.startswith("MOVE_TARGET_")})
    item_names = names_by_value(defs, "ITEM_")
    ability_names = names_by_value(defs, "ABILITY_")
    effect_names = names_by_value(defs, "EFFECT_")
    hold_names = names_by_value(defs, "HOLD_EFFECT_")
    type_names = names_by_value(defs, "TYPE_")
    nature_names = names_by_value(defs, "NATURE_")

    d["counts"] = {
        "NUM_SPECIES": defs["NUM_SPECIES"], "MOVES_COUNT": defs["MOVES_COUNT"],
        "ITEMS_COUNT": defs["ITEMS_COUNT"], "ABILITIES_COUNT": defs["ABILITIES_COUNT"],
        "NUM_BATTLE_MOVE_EFFECTS": max(effect_names) + 1,
        "NUM_FRONTIER_MONS": defs["NUM_FRONTIER_MONS"],
        "FRONTIER_MONS_HIGH_TIER": defs["FRONTIER_MONS_HIGH_TIER"],
    }
    d["names"] = {
        "species": {v: n for v, n in species_names.items() if v < defs["NUM_SPECIES"]},
        "moves": {v: n for v, n in move_names.items() if v < defs["MOVES_COUNT"]},
        "items": {v: n for v, n in item_names.items() if v < defs["ITEMS_COUNT"]},
        "abilities": {v: n for v, n in ability_names.items() if v < defs["ABILITIES_COUNT"]},
        "effects": effect_names,
        "hold_effects": hold_names,
        "types": {v: n for v, n in type_names.items() if v < 18},
        "natures": {v: n for v, n in nature_names.items() if v < 25},
    }

    # gSpeciesInfo: 28-byte SpeciesInfo, indexed by internal species ID
    data, _ = rom.table("gSpeciesInfo")
    d["species"] = []
    for c in chunks(data, 28):
        d["species"].append({
            "base": list(c[0:6]),                      # HP Atk Def Spe SpA SpD
            "types": [c[6], c[7]],
            "gender_ratio": c[0x10],
            "abilities": [c[0x16], c[0x17]],
        })
    assert len(d["species"]) == defs["NUM_SPECIES"], len(d["species"])

    # gBattleMoves: 12-byte BattleMove (9 fields + 3 padding)
    data, _ = rom.table("gBattleMoves")
    d["moves"] = []
    for c in chunks(data, 12):
        effect, power, mtype, acc, pp, chance, target, prio, flags = struct.unpack("<BBBBBBBbB", c[:9])
        d["moves"].append({"effect": effect, "power": power, "type": mtype, "accuracy": acc, "pp": pp,
                           "secondary_chance": chance, "target": target, "priority": prio, "flags": flags})
    assert len(d["moves"]) == defs["MOVES_COUNT"], len(d["moves"])

    # gItems: 44-byte Item
    data, _ = rom.table("gItems")
    d["items"] = []
    for c in chunks(data, 44):
        item_id, price, hold, param = struct.unpack("<HHBB", c[14:20])
        d["items"].append({"hold_effect": hold, "hold_effect_param": param})
    assert len(d["items"]) == defs["ITEMS_COUNT"], len(d["items"])

    # gBattleFrontierMons: 16-byte FacilityMon
    data, _ = rom.table("gBattleFrontierMons")
    d["frontier_mons"] = []
    for c in chunks(data, 16):
        species, m0, m1, m2, m3, item_tid, ev_spread, nature = struct.unpack("<HHHHHBBB", c[:13])
        d["frontier_mons"].append({"species": species, "moves": [m0, m1, m2, m3],
                                   "item_table_id": item_tid, "ev_spread": ev_spread, "nature": nature})
    assert len(d["frontier_mons"]) == defs["NUM_FRONTIER_MONS"]

    data, _ = rom.table("gBattleFrontierHeldItems")
    d["frontier_held_items"] = list(struct.unpack(f"<{len(data)//2}H", data))

    # gTypeEffectiveness: (atk, def, multiplier) triplets, TYPE_FORESIGHT(0xFE) separator,
    # TYPE_ENDTABLE(0xFF) terminator. Order matters: the game applies entries sequentially.
    data, _ = rom.table("gTypeEffectiveness")
    d["type_effectiveness"] = [list(c) for c in chunks(data, 3)]

    data, _ = rom.table("gNatureStatTable")
    d["nature_stat_table"] = [list(struct.unpack("<5b", c)) for c in chunks(data, 5)]

    data, _ = rom.table("gStatStageRatios")
    d["stat_stage_ratios"] = [list(c) for c in chunks(data, 2)]

    data, _ = rom.table("sCriticalHitChance")
    d["critical_hit_chance"] = list(struct.unpack(f"<{len(data)//2}H", data))

    # Battle Factory tables (battle_factory.c)
    data, _ = rom.table("sInitialRentalMonRanges")
    d["factory_rental_ranges"] = [list(struct.unpack("<HH", c)) for c in chunks(data, 4)]
    # sFixedIVTable is read one row past its end at challengeNum 8 in vanilla
    # (GetFactoryMonFixedIV uses `>` instead of `>=`); keep that row too.
    data, _ = rom.table("sFixedIVTable", extra=2)
    d["factory_fixed_ivs"] = [list(c) for c in chunks(data, 2)]
    data, _ = rom.table("sRequiredMoveCounts")
    d["factory_style_required_counts"] = list(data)
    data, _ = rom.table("sMoveStyles")
    d["factory_style_moves"] = [rom.u16_list_at(p) for p in struct.unpack(f"<{len(data)//4}I", data)]

    # Frontier trainers (battle_tower.c)
    data, _ = rom.table("sFrontierTrainerIdRanges")
    d["frontier_trainer_ranges"] = [list(struct.unpack("<HH", c)) for c in chunks(data, 4)]
    data, _ = rom.table("sFrontierTrainerIdRangesHard")
    d["frontier_trainer_ranges_hard"] = [list(struct.unpack("<HH", c)) for c in chunks(data, 4)]
    data, _ = rom.table("gBattleFrontierTrainers")
    d["frontier_trainers"] = [{"facility_class": c[0]} for c in chunks(data, 52)]

    data, _ = rom.table("sFrontierBrainStreakAppearances", source="frontier_util.o")
    d["frontier_brain_streak_appearances"] = [list(c) for c in chunks(data, 4)]

    d["_symbols"] = symbol_table(rom)
    d["defines"] = {k: defs[k] for k in sorted(defs) if k.startswith((
        "MOVE_TARGET_", "FLAG_", "F_EV_SPREAD_", "FACTORY_", "FRONTIER_LVL_", "AI_SCRIPT_",
        "MAX_", "NUM_STATS", "STAT_"))}
    return d


# ---------------------------------------------------------------------------
# C++ generation
# ---------------------------------------------------------------------------

HEADER = "// GENERATED by scripts/extract_game_data.py from pokeemerald.elf. Do not edit.\n"


def enum_block(comment, names, count_aliases=()):
    lines = [f"// {comment}", "enum : uint16_t {"]
    for value in sorted(names):
        lines.append(f"    {names[value]} = {value},")
    for alias, value in count_aliases:
        lines.append(f"    {alias} = {value},")
    lines.append("};\n")
    return "\n".join(lines)


def gen_constants(d):
    n, c = d["names"], d["counts"]
    out = [HEADER, "#pragma once\n", "#include <cstdint>\n", "namespace pkmn {\n"]
    out.append(enum_block("Species (internal IDs, not national dex)", n["species"],
                          [("SPECIES_COUNT", c["NUM_SPECIES"])]))
    out.append(enum_block("Moves", n["moves"], [("MOVES_COUNT", c["MOVES_COUNT"])]))
    out.append(enum_block("Items", n["items"], [("ITEMS_COUNT", c["ITEMS_COUNT"]),
                                                ("ITEM_COUNT", c["ITEMS_COUNT"])]))
    out.append(enum_block("Abilities", n["abilities"], [("ABILITIES_COUNT", c["ABILITIES_COUNT"])]))
    out.append(enum_block("Item hold effects", n["hold_effects"]))
    out.append(enum_block("Move effects (battle_move_effects.h); MoveEffect enum class mirrors these",
                          n["effects"], [("EFFECT_COUNT", c["NUM_BATTLE_MOVE_EFFECTS"])]))
    out.append(f"constexpr uint16_t NUM_FRONTIER_MONS_TOTAL = {c['NUM_FRONTIER_MONS']};")
    out.append(f"constexpr uint16_t FRONTIER_MONS_HIGH_TIER = {c['FRONTIER_MONS_HIGH_TIER']};\n")
    for k, v in d["defines"].items():
        if k.startswith(("MOVE_TARGET_", "FLAG_", "STAT_")) and k != "STAT_BUFF_NEGATIVE":
            out.append(f"constexpr uint8_t {k} = {v};")
    out.append("\n}  // namespace pkmn\n")
    return "\n".join(out)


def gen_move_effects(d):
    lines = [HEADER.replace("// ", "// ")]
    for value in sorted(d["names"]["effects"]):
        lines.append(f"    {d['names']['effects'][value][len('EFFECT_'):]} = {value},")
    return "\n".join(lines) + "\n"


def type_expr(t):
    return f"static_cast<Type>({t})"


def gen_species(d):
    rows = []
    for i, s in enumerate(d["species"]):
        b = s["base"]
        rows.append(f"    /* {d['names']['species'].get(i, i)} */ "
                    f"{{{b[0]}, {b[1]}, {b[2]}, {b[3]}, {b[4]}, {b[5]}, {type_expr(s['types'][0])}, "
                    f"{type_expr(s['types'][1])}, {{{s['abilities'][0]}, {s['abilities'][1]}}}, {s['gender_ratio']}}},")
    return f"""{HEADER}#include "data.hpp"

namespace pkmn {{

// Format: {{HP, Atk, Def, Spe, SpA, SpD, Type1, Type2, {{Ability1, Ability2}}, GenderRatio}}
static const SpeciesData SPECIES_DATA[] = {{
{chr(10).join(rows)}
}};

const SpeciesData& getSpeciesData(uint16_t speciesId) {{
    if (speciesId >= SPECIES_COUNT) return SPECIES_DATA[0];
    return SPECIES_DATA[speciesId];
}}

}}  // namespace pkmn
"""


def gen_moves(d):
    rows = []
    flag_contact = d["defines"]["FLAG_MAKES_CONTACT"]
    for i, m in enumerate(d["moves"]):
        physical = "true" if m["type"] < 9 else "false"   # Gen 3: type < TYPE_MYSTERY is physical
        contact = "true" if m["flags"] & flag_contact else "false"
        rows.append(f"    /* {d['names']['moves'].get(i, i)} */ "
                    f"{{{m['power']}, {m['accuracy']}, {m['pp']}, {type_expr(m['type'])}, "
                    f"static_cast<MoveEffect>({m['effect']}), {m['secondary_chance']}, {m['priority']}, "
                    f"{physical}, {contact}, {m['target']}, {m['flags']}}},")
    return f"""{HEADER}#include "data.hpp"

namespace pkmn {{

// Format: {{Power, Accuracy, PP, Type, Effect, SecondaryChance, Priority, IsPhysical, MakesContact, Target, Flags}}
static const MoveData MOVE_DATA[] = {{
{chr(10).join(rows)}
}};

const MoveData& getMoveData(uint16_t moveId) {{
    if (moveId >= MOVES_COUNT) return MOVE_DATA[0];
    return MOVE_DATA[moveId];
}}

}}  // namespace pkmn
"""


def gen_items(d):
    rows = [f"    /* {d['names']['items'].get(i, i)} */ {{{it['hold_effect']}, {it['hold_effect_param']}}},"
            for i, it in enumerate(d["items"])]
    return f"""{HEADER}#include "data.hpp"

namespace pkmn {{

static const ItemData ITEM_DATA[] = {{
{chr(10).join(rows)}
}};

const ItemData& getItemData(uint16_t itemId) {{
    if (itemId >= ITEMS_COUNT) return ITEM_DATA[0];
    return ITEM_DATA[itemId];
}}

}}  // namespace pkmn
"""


def gen_frontier(d):
    rows = []
    for i, m in enumerate(d["frontier_mons"]):
        mv = ", ".join(str(x) for x in m["moves"])
        rows.append(f"    /* {i} */ {{{m['species']}, {{{mv}}}, {m['item_table_id']}, {m['ev_spread']}, "
                    f"static_cast<Nature>({m['nature']})}},")
    items = ", ".join(str(x) for x in d["frontier_held_items"])
    n_items = len(d["frontier_held_items"])
    return f"""{HEADER}#include "data.hpp"

namespace pkmn {{

static const FrontierMon FRONTIER_MONS[] = {{
{chr(10).join(rows)}
}};

// gBattleFrontierHeldItems: itemTableId -> ITEM_*
static const uint16_t FRONTIER_ITEMS[{n_items}] = {{{items}}};

const FrontierMon& getFrontierMon(uint16_t id) {{
    if (id >= NUM_FRONTIER_MONS) return FRONTIER_MONS[0];
    return FRONTIER_MONS[id];
}}

uint16_t getFrontierItem(uint8_t id) {{
    if (id >= {n_items}) return ITEM_NONE;
    return FRONTIER_ITEMS[id];
}}

}}  // namespace pkmn
"""


def gen_type_chart(d):
    triplets = ", ".join(f"{{{a}, {b}, {m}}}" for a, b, m in d["type_effectiveness"])
    natures = ",\n    ".join("{" + ", ".join(str(x) for x in row) + "}" for row in d["nature_stat_table"])
    ratios = ", ".join(f"{{{a}, {b}}}" for a, b in d["stat_stage_ratios"])
    return f"""{HEADER}#include "data.hpp"

namespace pkmn {{

// gTypeEffectiveness, in game order. {{atk, def, multiplier(x10)}};
// atk == TYPE_FORESIGHT (0xFE) separates the Foresight-ignorable entries, 0xFF ends the table.
const TypeEffectivenessEntry TYPE_EFFECTIVENESS[{len(d['type_effectiveness'])}] = {{{triplets}}};
const size_t TYPE_EFFECTIVENESS_COUNT = {len(d['type_effectiveness'])};

// gNatureStatTable: [nature][Atk, Def, Spe, SpA, SpD] -> +1 / -1 / 0
static const int8_t NATURE_STAT_TABLE[25][5] = {{
    {natures}
}};

// gStatStageRatios: numerator/denominator for stages -6..+6
const uint8_t STAT_STAGE_RATIOS[13][2] = {{{ratios}}};

int getTypeEffectiveness(Type attackType, Type defendType) {{
    for (size_t i = 0; i < TYPE_EFFECTIVENESS_COUNT; i++) {{
        const auto& e = TYPE_EFFECTIVENESS[i];
        if (e.attacker == TYPE_ENDTABLE) break;
        if (e.attacker == static_cast<uint8_t>(attackType) && e.defender == static_cast<uint8_t>(defendType))
            return e.multiplier * 10;  // x10 multiplier -> x100 scale
    }}
    return 100;
}}

int getTypeEffectivenessDual(Type attackType, Type defType1, Type defType2) {{
    int eff1 = getTypeEffectiveness(attackType, defType1);
    if (defType1 == defType2) return eff1;
    return eff1 * getTypeEffectiveness(attackType, defType2) / 100;
}}

int getNatureModifier(Nature nature, Stat stat) {{
    if (stat == Stat::HP) return 100;
    return 100 + 10 * NATURE_STAT_TABLE[static_cast<int>(nature)][static_cast<int>(stat) - 1];
}}

void calculateEVsFromSpread(uint8_t evSpread, uint8_t* outEVs) {{
    // CreateMonWithEVSpreadNatureOTID: MAX_TOTAL_EVS / number of flagged stats
    int count = 0;
    for (int i = 0; i < 6; i++)
        if (evSpread & (1 << i)) count++;
    int amount = count ? 510 / count : 0;
    for (int i = 0; i < 6; i++)
        outEVs[i] = (evSpread & (1 << i)) ? static_cast<uint8_t>(amount) : 0;
}}

}}  // namespace pkmn
"""


def gen_factory_tables(d):
    def pairs(rows):
        return ", ".join(f"{{{a}, {b}}}" for a, b in rows)

    styles = []
    for i, moves in enumerate(d["factory_style_moves"]):
        styles.append(f"static const uint16_t STYLE_MOVES_{i}[] = {{{', '.join(map(str, moves + [0]))}}};")
    style_ptrs = ", ".join(f"STYLE_MOVES_{i}" for i in range(len(d["factory_style_moves"])))
    brain = ", ".join("{" + ", ".join(map(str, r)) + "}" for r in d["frontier_brain_streak_appearances"])
    return f"""{HEADER}#include "game_tables.hpp"

namespace pkmn {{

// battle_factory.c
const uint16_t FACTORY_RENTAL_RANGES[{len(d['factory_rental_ranges'])}][2] = {{{pairs(d['factory_rental_ranges'])}}};
// Last row is the out-of-bounds read at challengeNum 8 (vanilla GetFactoryMonFixedIV bug).
const uint8_t FACTORY_FIXED_IVS[{len(d['factory_fixed_ivs'])}][2] = {{{pairs(d['factory_fixed_ivs'])}}};
const uint8_t FACTORY_STYLE_REQUIRED_COUNTS[{len(d['factory_style_required_counts'])}] = {{{', '.join(map(str, d['factory_style_required_counts']))}}};
{chr(10).join(styles)}
const uint16_t* const FACTORY_STYLE_MOVES[{len(d['factory_style_moves'])}] = {{{style_ptrs}}};

// battle_tower.c
const uint16_t FRONTIER_TRAINER_RANGES[{len(d['frontier_trainer_ranges'])}][2] = {{{pairs(d['frontier_trainer_ranges'])}}};
const uint16_t FRONTIER_TRAINER_RANGES_HARD[{len(d['frontier_trainer_ranges_hard'])}][2] = {{{pairs(d['frontier_trainer_ranges_hard'])}}};
const uint16_t NUM_FRONTIER_TRAINERS = {len(d['frontier_trainers'])};

// frontier_util.c
const uint8_t FRONTIER_BRAIN_STREAK_APPEARANCES[{len(d['frontier_brain_streak_appearances'])}][4] = {{{brain}}};

// battle_script_commands.c
const uint16_t CRITICAL_HIT_CHANCE[{len(d['critical_hit_chance'])}] = {{{', '.join(map(str, d['critical_hit_chance']))}}};

}}  // namespace pkmn
"""


def gen_game_tables_header(d):
    return f"""{HEADER}#pragma once

#include <cstdint>

namespace pkmn {{

extern const uint16_t FACTORY_RENTAL_RANGES[{len(d['factory_rental_ranges'])}][2];  // [challenge + 8*openLevel]
extern const uint8_t FACTORY_FIXED_IVS[{len(d['factory_fixed_ivs'])}][2];          // [challenge][useHigherIV]
extern const uint8_t FACTORY_STYLE_REQUIRED_COUNTS[{len(d['factory_style_required_counts'])}];
extern const uint16_t* const FACTORY_STYLE_MOVES[{len(d['factory_style_moves'])}];  // MOVE_NONE terminated
extern const uint16_t FRONTIER_TRAINER_RANGES[{len(d['frontier_trainer_ranges'])}][2];
extern const uint16_t FRONTIER_TRAINER_RANGES_HARD[{len(d['frontier_trainer_ranges_hard'])}][2];
extern const uint16_t NUM_FRONTIER_TRAINERS;
extern const uint8_t FRONTIER_BRAIN_STREAK_APPEARANCES[{len(d['frontier_brain_streak_appearances'])}][4];
extern const uint16_t CRITICAL_HIT_CHANCE[{len(d['critical_hit_chance'])}];

}}  // namespace pkmn
"""


def symbol_table(rom):
    """RAM objects and all functions, for the emulator bridge (pybattle/emu/symbols.json)."""
    objects = {}
    for name, (addr, size) in rom.symbols.items():
        if 0x02000000 <= addr < 0x04000000:
            objects[name] = [addr, size]
    by_name = {}
    for (src, name), entry in rom.local_symbols.items():
        by_name.setdefault(name, []).append((src, entry))
    for name, entries in by_name.items():
        for src, (addr, size) in entries:
            if 0x02000000 <= addr < 0x04000000:
                key = name if len(entries) == 1 and name not in objects else f"{name}@{src}"
                objects[key] = [addr, size]
    return {"objects": dict(sorted(objects.items())), "functions": dict(sorted(rom.functions.items()))}


def outputs(d):
    symbols = d.pop("_symbols")
    return {
        "pybattle/emu/symbols.json": json.dumps(symbols, indent=0, sort_keys=True) + "\n",
        "include/gen/game_constants.hpp": gen_constants(d),
        "include/gen/move_effects.inc": gen_move_effects(d),
        "include/game_tables.hpp": gen_game_tables_header(d),
        "src/data/species_data.cpp": gen_species(d),
        "src/data/move_data.cpp": gen_moves(d),
        "src/data/item_data.cpp": gen_items(d),
        "src/data/frontier_mons.cpp": gen_frontier(d),
        "src/data/type_chart.cpp": gen_type_chart(d),
        "src/data/factory_tables.cpp": gen_factory_tables(d),
        "pybattle/data/game_data.json": json.dumps(d, indent=1, sort_keys=True) + "\n",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pokeemerald", default=os.path.expanduser("~/Dev/pokeemerald"))
    ap.add_argument("--check", action="store_true", help="fail if generated files are stale")
    args = ap.parse_args()

    stale = []
    for rel, text in outputs(extract(args.pokeemerald)).items():
        path = os.path.join(REPO, rel)
        old = open(path).read() if os.path.exists(path) else None
        if old == text:
            continue
        stale.append(rel)
        if not args.check:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(text)
    if args.check and stale:
        sys.exit("stale generated files: " + ", ".join(stale))
    print("updated:" if not args.check else "ok", ", ".join(stale) or "nothing")


if __name__ == "__main__":
    main()

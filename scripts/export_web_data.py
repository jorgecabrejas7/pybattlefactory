#!/usr/bin/env python3
"""Display data and game tables for the browser build, as one JSON file (game.json in the web pack).

Everything is keyed by the game's internal ids (the ids views and the engine use):
    species[id]   {name, types, base, abilities}
    moves[id]     {name, type, power, accuracy, pp, priority, effect, target, flags, desc}
    items[id]     {name, desc, hold_effect, hold_effect_param}
    abilities[id] {name, desc}
    types[id], natures[id] (+ nature_stats: [atk, def, spe, spa, spd] multipliers x10)
    trainers[id]  {name, class}: Battle Frontier trainers (Factory opponents), FactoryInfo.trainerId
    styles        FACTORY_STYLE_* names (the attendant's hint)
    frontier_mons Battle Factory sets (pybattle/data/game_data.json)
and planner.json next to it: the tactician's tables (encoder tokens, Factory set tables, hint rule, planner constants).
Names come from the decomp's text (pokeemerald src/data/text, src/data/items.h, battle frontier trainers), in
display case ("THUNDERBOLT" -> "Thunderbolt", "{PKMN}" -> "Pokémon").

    export_web_data.py [--pokeemerald ~/Dev/pokeemerald] [--out dist/data/game.json]
"""

import argparse
import json
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAME_DATA = os.path.join(REPO, "pybattle", "data", "game_data.json")

STYLES = ["No style", "Preparation", "Slow & steady", "Endurance", "High risk", "Weakening", "Unpredictable",
          "Weather"]   # FACTORY_STYLE_* (battle_factory.c sFactoryStyleNames order)

# words kept as written by display_case
KEEP = {"HP", "PP", "TM", "HM", "KO", "OHKO", "EXP", "PC", "PSN", "PAR", "SLP", "BRN", "FRZ", "UV", "RPM", "LV"}


def read(path):
    return open(path, encoding="utf-8", errors="replace").read()


def defines(*paths):
    """#define NAME value (ints, hex, other names, simple + / - / parens) from C headers."""
    raw = {}
    for p in paths:
        for n, v in re.findall(r"^#define[ \t]+(\w+)[ \t]+([^\n/]+?)[ \t]*(?://.*)?$", read(p), re.M):
            raw[n] = v.strip()
    out = {}

    def ev(n, depth=0):
        if n in out:
            return out[n]
        v = raw.get(n)
        if v is None or depth > 20:
            return None
        expr = re.sub(r"\b([A-Za-z_]\w*)\b", lambda m: str(ev(m.group(1), depth + 1)), v)
        try:
            out[n] = int(eval(expr, {"__builtins__": {}}))   # noqa: S307 (C constant arithmetic only)
        except Exception:
            out[n] = None
        return out[n]

    for n in raw:
        ev(n)
    return {k: v for k, v in out.items() if v is not None}


def game_text(s):
    """Decomp text -> display text: placeholders and control escapes resolved, case kept."""
    s = s.replace("{PKMN}", "Pokémon").replace("POKéMON", "Pokémon").replace("{FEMALE}", "♀").replace("{MALE}", "♂")
    s = s.replace("é", "é")
    s = re.sub(r"\\[nlp]", " ", s)
    s = re.sub(r"\{[A-Z_0-9 ]+\}", "", s)
    return re.sub(r"\s+", " ", s).strip()


def display_case(s):
    s = game_text(s)

    def word(w):
        core = re.sub(r"[^A-Za-z]", "", w)
        if core.upper() in KEEP or not core.isupper():
            return w
        out, up = [], True
        for ch in w:
            if ch.isalpha():
                out.append(ch.upper() if up else ch.lower())
                up = False
            else:
                out.append(ch)
                up = ch in "-. " or (ch == "'" and False)
        return "".join(out)

    return " ".join(word(w) for w in s.split(" ")).replace("Pokémon", "Pokémon")


def table(text, key_prefix, consts):
    """[PREFIX_NAME] = _("TEXT") entries -> {id: text}."""
    out = {}
    for k, v in re.findall(r"\[(" + key_prefix + r"\w+)\]\s*=\s*_\(\"((?:[^\"\\]|\\.)*)\"\)", text):
        if k in consts:
            out[consts[k]] = v
    return out


def pointer_table(text, array, key_prefix, consts, offset=0):
    """static const u8 sX[] = _("..."); + array[] = {[KEY] = sX, ...} -> {id: text}."""
    strings = {n: "".join(re.findall(r"\"((?:[^\"\\]|\\.)*)\"", body))
               for n, body in re.findall(r"static const u8 (\w+)\[\]\s*=\s*_\(\s*((?:\"(?:[^\"\\]|\\.)*\"\s*)+)\)", text)}
    m = re.search(re.escape(array) + r"[^=]*=\s*\{(.*?)\n\};", text, re.S)
    out = {}
    for k, s in re.findall(r"\[(" + key_prefix + r"\w+)(?:\s*-\s*1)?\]\s*=\s*(\w+)", m.group(1)):
        if k in consts and s in strings:
            out[consts[k] - offset] = strings[s]
    return out


def planner_tables():
    """The tactician's tables (rl/encode.py v5 token ids and context, rl/determinize.py set tables and hint rule,
    rl/team_eval.py planner constants), dumped from the Python modules themselves for the TypeScript port."""
    import sys
    sys.path.insert(0, REPO)
    import numpy as np
    from rl import determinize as DET, encode as E, team_eval as TE
    E.set_version(5)
    pad4 = lambda mv: (list(mv) + [0] * 4)[:4]                                   # noqa: E731
    return {
        "tokens": {k: int(getattr(E, k)) for k in ("SPECIES_UNK", "MOVE_UNK", "ITEM_UNK", "ABILITY_UNK", "EFFECT_NONE",
                                                   "EFFECT_UNK", "TYPE_NONE", "TYPE_UNK", "N_TYPES")},
        "encode": {"MON_NUM": E.MON_NUM, "MOVE_NUM": E.MOVE_NUM, "CTX_NUM": E.CTX_NUM, "FIELD_NUM": E.FIELD_NUM,
                   "HINT_NUM": E.HINT_NUM, "N_HINT_TYPES": E.N_HINT_TYPES, "N_STYLES": E.N_STYLES,
                   "N_ROUNDS": E.N_ROUNDS, "N_ACTIVE": E.N_ACTIVE, "N_DEFEATED": E.N_DEFEATED,
                   "N_MAJOR_STATUSES": 7, "RENT_RANKS": list(E.RENT_RANKS), "PAIRS": [list(p) for p in E.PAIRS]},
        "sets": {"species": DET.SET_SPECIES.tolist(), "item": DET.SET_ITEM.tolist(),
                 "moves_as_built": [pad4(m) for m in DET.SET_MOVES_AS_BUILT],
                 "moves_raw": DET.FRONTIER_MOVES_RAW.tolist()},
        "factory": {"ranges": DET.RANGES, "high_tier": DET.HIGH_TIER, "species_unown": int(DET.SPECIES_UNOWN),
                    "hint_tries": DET.HINT_TRIES_FACTORY},
        "hint": {"move_style": DET.MOVE_STYLE.tolist(), "style_required": list(DET.STYLE_REQUIRED),
                 "species_types": DET.SPECIES_TYPES.tolist(), "type_none": DET.HINT_TYPE_NONE,
                 "style_many": DET.HINT_STYLE_MANY},
        "team": {"N_ROUNDS": TE.N_ROUNDS, "UNKNOWN": TE.UNKNOWN, "MON_NUM": TE.MON_NUM, "HINT_UNK_TYPE": TE.HINT_UNK_TYPE,
                 "HINT_UNK_STYLE": TE.HINT_UNK_STYLE, "WIN_SCALE": TE.WIN_SCALE, "WIN_COLUMNS": list(TE.WIN_COLUMNS),
                 "REVEAL": TE.REVEAL, "OPT_FEATURES": list(TE.OPT_FEATURES)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pokeemerald", default=os.path.expanduser("~/Dev/pokeemerald"))
    ap.add_argument("--out", default=os.path.join(REPO, "dist", "data", "game.json"))
    args = ap.parse_args()
    pe = args.pokeemerald
    inc = lambda *p: os.path.join(pe, "include", *p)     # noqa: E731
    src = lambda *p: os.path.join(pe, "src", *p)         # noqa: E731
    gd = json.load(open(GAME_DATA))
    names = gd["names"]
    by_name = {t: {v: int(k) for k, v in names[t].items()} for t in names}
    consts = defines(inc("constants", "species.h"), inc("constants", "moves.h"), inc("constants", "items.h"),
                     inc("constants", "abilities.h"), inc("constants", "trainers.h"), inc("constants", "pokemon.h"),
                     inc("constants", "battle_frontier_trainers.h"))
    for t in by_name.values():
        consts.update(t)

    sp_names = table(read(src("data", "text", "species_names.h")), "SPECIES_", consts)
    mv_names = table(read(src("data", "text", "move_names.h")), "MOVE_", consts)
    ab_text = read(src("data", "text", "abilities.h"))
    ab_names = table(ab_text, "ABILITY_", consts)
    ab_desc = pointer_table(ab_text, "gAbilityDescriptionPointers", "ABILITY_", consts)
    mv_desc = pointer_table(read(src("data", "text", "move_descriptions.h")), "gMoveDescriptionPointers", "MOVE_",
                            consts)
    item_desc_strings = {n: "".join(re.findall(r"\"((?:[^\"\\]|\\.)*)\"", b)) for n, b in re.findall(
        r"static const u8 (\w+)\[\]\s*=\s*_\(\s*((?:\"(?:[^\"\\]|\\.)*\"\s*)+)\)", read(src("data", "text", "item_descriptions.h")))}
    items = {}
    for k, body in re.findall(r"\[(ITEM_\w+)\]\s*=\s*\{(.*?)\n    \}", read(src("data", "items.h")), re.S):
        nm = re.search(r"\.name\s*=\s*_\(\"((?:[^\"\\]|\\.)*)\"\)", body)
        ds = re.search(r"\.description\s*=\s*(\w+)", body)
        if k in consts and nm:
            items[consts[k]] = (nm.group(1), item_desc_strings.get(ds.group(1), "") if ds else "")
    nature_text = read(src("data", "text", "nature_names.h"))
    nature_names = re.findall(r"static const u8 s\w+NatureName\[\]\s*=\s*_\(\"(\w+)\"\)", nature_text)
    tc_names = table(read(src("data", "text", "trainer_class_names.h")), "TRAINER_CLASS_", consts)
    fc_to_tc = {consts[f]: consts[t] for f, t in re.findall(r"\[(FACILITY_CLASS_\w+)\]\s*=\s*(TRAINER_CLASS_\w+)",
                                                            read(src("data", "pokemon", "trainer_class_lookups.h")))
                if f in consts and t in consts}
    trainers = {}
    for k, body in re.findall(r"^\s+\[(FRONTIER_TRAINER_\w+)\]\s*=\s*\{(.*?)\n    \}",
                              read(src("data", "battle_frontier", "battle_frontier_trainers.h")), re.S | re.M):
        fc = re.search(r"\.facilityClass\s*=\s*(FACILITY_CLASS_\w+)", body)
        nm = re.search(r"\.trainerName\s*=\s*_\(\"((?:[^\"\\]|\\.)*)\"\)", body)
        if k in consts and fc and nm:
            tc = fc_to_tc.get(consts.get(fc.group(1)))
            trainers[consts[k]] = {"name": display_case(nm.group(1)),
                                   "class": display_case(tc_names.get(tc, "")) if tc is not None else ""}

    n_species = len(gd["species"])
    species = []
    for i, s in enumerate(gd["species"]):
        species.append({"name": display_case(sp_names.get(i, names["species"].get(str(i), "?")[8:])),
                        "types": s["types"], "base": s["base"], "abilities": s["abilities"]})
    moves = []
    for i, m in enumerate(gd["moves"]):
        moves.append({"name": display_case(mv_names.get(i, "")) if i else "—",
                      "type": m["type"], "power": m["power"], "accuracy": m["accuracy"], "pp": m["pp"],
                      "priority": m["priority"], "effect": m["effect"], "target": m["target"], "flags": m["flags"],
                      "chance": m["secondary_chance"], "desc": game_text(mv_desc.get(i, ""))})
    item_list = []
    for i, it in enumerate(gd["items"]):
        nm, ds = items.get(i, ("", ""))
        item_list.append({"name": display_case(nm) if i else "None", "desc": game_text(ds),
                          "hold_effect": it["hold_effect"], "hold_effect_param": it["hold_effect_param"]})
    n_ab = len(names["abilities"])
    abilities = [{"name": display_case(ab_names.get(i, "")) if i else "—", "desc": game_text(ab_desc.get(i, ""))}
                 for i in range(n_ab)]
    types = [display_case(names["types"][str(i)][5:].replace("_", " ")) for i in range(len(names["types"]))]
    out = {
        "version": 1,
        "species": species, "moves": moves, "items": item_list, "abilities": abilities, "types": types,
        "natures": [display_case(n) for n in nature_names], "nature_stats": gd["nature_stat_table"],
        "trainers": [trainers.get(i, {"name": "?", "class": ""}) for i in range(len(gd["frontier_trainers"]))],
        "noland": {"name": "Noland", "class": "Factory Head"},
        "styles": STYLES,
        "type_effectiveness": gd["type_effectiveness"],
        "frontier_mons": gd["frontier_mons"], "frontier_held_items": gd["frontier_held_items"],
        "factory_style_moves": gd.get("factory_style_moves"),
    }
    assert len(species) == n_species
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    with open(os.path.join(os.path.dirname(os.path.abspath(args.out)), "planner.json"), "w") as f:
        json.dump(planner_tables(), f, separators=(",", ":"))
    print(f"{args.out}: {os.path.getsize(args.out) / 1e3:.0f} KB; {len(species)} species, {len(moves)} moves, "
          f"{len(item_list)} items, {len(abilities)} abilities, {len(trainers)} trainers")


if __name__ == "__main__":
    main()

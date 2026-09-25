#!/usr/bin/env python3
"""
Extract every species' learnable moves from the pokeemerald sources (player knowledge: what a species can know).

For each species: level-up moves (with their level), TM/HM, move tutor and egg moves, each also inherited from
its pre-evolutions (a Pokemon can know what it learned before evolving). Used by rl/determinize.py to draw the
opponent's unrevealed moves without the Battle Factory's set list.

Output (generated, do not edit): pybattle/data/learnsets.json
    {"<species id>": {"level": [[level, move], ...], "other": [move, ...]}}

Usage: extract_learnsets.py [--pokeemerald ~/Dev/pokeemerald] [--check]
"""

import argparse
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "pybattle", "data", "learnsets.json")


def load_names():
    data = json.load(open(os.path.join(REPO, "pybattle", "data", "game_data.json")))
    species = {v: int(k) for k, v in data["names"]["species"].items()}
    moves = {v: int(k) for k, v in data["names"]["moves"].items()}
    return species, moves


def read(root, rel):
    with open(os.path.join(root, rel)) as f:
        return f.read()


def extract(root):
    species, moves = load_names()
    mons = os.path.join("src", "data", "pokemon")

    # level-up: arrays by name, then the species -> array pointer table
    src = read(root, os.path.join(mons, "level_up_learnsets.h"))
    arrays = {}
    for name, body in re.findall(r"static const u16 (\w+)\[\] = \{(.*?)\};", src, re.S):
        arrays[name] = [(int(l), moves[m]) for l, m in re.findall(r"LEVEL_UP_MOVE\(\s*(\d+),\s*(MOVE_\w+)\)", body)]
    level = {}
    for sp, arr in re.findall(r"\[(SPECIES_\w+)\]\s*=\s*(\w+)", read(root, os.path.join(mons, "level_up_learnset_pointers.h"))):
        if sp != "SPECIES_NONE":
            level[species[sp]] = arrays[arr]

    other = {sid: set() for sid in species.values()}
    # TM / HM: ".MOVE_NAME = TRUE" inside each species' block
    src = read(root, os.path.join(mons, "tmhm_learnsets.h"))
    for sp, body in re.findall(r"\[(SPECIES_\w+)\]\s*=\s*\{\s*\.learnset\s*=\s*\{(.*?)\}\s*\}", src, re.S):
        for m in re.findall(r"\.(\w+)\s*=\s*TRUE", body):
            other[species[sp]].add(moves["MOVE_" + m])
    # tutor: TUTOR(MOVE_X) inside each species' expression
    src = read(root, os.path.join(mons, "tutor_learnsets.h"))
    for sp, body in re.findall(r"\[(SPECIES_\w+)\]\s*=\s*\((.*?)\),", src, re.S):
        for m in re.findall(r"TUTOR\((MOVE_\w+)\)", body):
            other[species[sp]].add(moves[m])
    # egg moves: egg_moves(NAME, MOVE_..., ...)
    src = read(root, os.path.join(mons, "egg_moves.h"))
    for sp, body in re.findall(r"egg_moves\((\w+),(.*?)\)", src, re.S):
        for m in re.findall(r"MOVE_\w+", body):
            other[species["SPECIES_" + sp]].add(moves[m])

    # pre-evolutions: [SPECIES_A] = {{EVO_..., param, SPECIES_B}, ...}
    pre = {}
    src = read(root, os.path.join(mons, "evolution.h"))
    for sp, body in re.findall(r"\[(SPECIES_\w+)\]\s*=\s*\{(.*?)\}\},", src, re.S):
        for target in re.findall(r"(SPECIES_\w+)\s*\}", body + "}"):
            pre[species[target]] = species[sp]

    out = {}
    for sid in sorted(species.values()):
        if sid == 0:
            continue
        lv, oth, s = set(level.get(sid, [])), set(other.get(sid, set())), sid
        while s in pre:
            s = pre[s]
            lv |= set(level.get(s, []))
            oth |= other.get(s, set())
        out[str(sid)] = {"level": sorted(lv), "other": sorted(oth)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pokeemerald", default=os.path.expanduser("~/Dev/pokeemerald"))
    ap.add_argument("--check", action="store_true", help="fail if the output file differs")
    args = ap.parse_args()
    text = json.dumps(extract(args.pokeemerald), separators=(",", ":"), sort_keys=True) + "\n"
    if args.check:
        same = os.path.exists(OUT) and open(OUT).read() == text
        print("learnsets.json up to date" if same else "learnsets.json differs")
        sys.exit(0 if same else 1)
    with open(OUT, "w") as f:
        f.write(text)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Pokemon sprites for the browser build, from the pokeemerald decomp's graphics (one small PNG per species, so a
page loads only the ones it shows):

    sprites/front/<species id>.png   64x128: the two front frames (anim_front), stacked
    sprites/back/<species id>.png    64x64
    sprites/icon/<species id>.png    32x64: the two party-icon frames, stacked

Species -> files through the game's own tables (src/data/pokemon_graphics/*_table.h, src/data/graphics/pokemon.h,
src/anim_mon_front_pics.c, src/pokemon_icon.c), not folder names. The PNGs are indexed with their palette; index 0 becomes transparent.

    build_sprites.py [--pokeemerald ~/Dev/pokeemerald] [--out dist/sprites]
"""

import argparse
import json
import os
import re

from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def incbins(text):
    """symbol -> graphics path (without the .4bpp/.lz suffixes)"""
    return {n: p.split(".")[0] for n, p in re.findall(r"(\w+)\[\]\s*=\s*INCBIN_U\d+\(\s*\"([^\"]+)\"\s*\)", text)}


def rgba(path, crop=None):
    im = Image.open(path)
    if im.mode != "P":
        raise SystemExit(f"{path}: not an indexed PNG")
    pal = im.getpalette()
    data = im.tobytes()
    out = Image.new("RGBA", im.size)
    px = [(pal[3 * i], pal[3 * i + 1], pal[3 * i + 2], 0 if i == 0 else 255) for i in range(len(pal) // 3)]
    out.putdata([px[b] for b in data])
    return out.crop(crop) if crop else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pokeemerald", default=os.path.expanduser("~/Dev/pokeemerald"))
    ap.add_argument("--out", default=os.path.join(REPO, "dist", "sprites"))
    args = ap.parse_args()
    pe = args.pokeemerald
    names = json.load(open(os.path.join(REPO, "pybattle", "data", "game_data.json")))["names"]["species"]
    species_id = {v[len("SPECIES_"):]: int(k) for k, v in names.items()}
    syms = incbins(open(os.path.join(pe, "src", "data", "graphics", "pokemon.h")).read())
    syms.update(incbins(open(os.path.join(pe, "src", "anim_mon_front_pics.c")).read()))

    def table(fname, macro):
        text = open(os.path.join(pe, "src", "data", "pokemon_graphics", fname)).read()
        return {species_id[s]: sym for s, sym in re.findall(macro + r"\((\w+),\s*(\w+)\)", text) if s in species_id}

    front = table("front_pic_table.h", "SPECIES_SPRITE")
    back = table("back_pic_table.h", "SPECIES_SPRITE")
    icon_text = open(os.path.join(pe, "src", "pokemon_icon.c")).read()
    icons = {species_id[s]: sym for s, sym in re.findall(r"\[SPECIES_(\w+)\]\s*=\s*(gMonIcon_\w+)", icon_text)
             if s in species_id}
    os.makedirs(args.out, exist_ok=True)
    counts = {}
    for kind, tab, fname in (("front", front, "anim_front.png"), ("back", back, "back.png"), ("icon", icons, "icon.png")):
        d = os.path.join(args.out, kind)
        os.makedirs(d, exist_ok=True)
        n = 0
        for sp, sym in tab.items():
            base = syms.get(sym)
            if base is None:
                continue
            folder = os.path.dirname(base)
            path = os.path.join(pe, folder, fname)
            for alt in (os.path.join(pe, folder, "normal", fname),                      # Castform's forms
                        os.path.join(pe, folder, "front.png") if kind == "front" else path):   # no animation sheet
                if not os.path.exists(path):
                    path = alt
            if not os.path.exists(path):
                continue
            rgba(path).save(os.path.join(d, f"{sp}.png"), optimize=True)
            n += 1
        counts[kind] = n
    total = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(args.out) for f in fs)
    print(f"{args.out}: {counts}, {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()

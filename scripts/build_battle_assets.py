#!/usr/bin/env python3
"""The battle screen's graphics for the browser build, from the pokeemerald decomp (what the GBA shows in a Battle
Factory battle), into dist/battle/:

    bg.png               240x160: the Frontier battle room (battle_environment/building tiles + map, stadium
                         battle_frontier palette: what DrawMainBattleBackground loads for BATTLE_TYPE_FRONTIER)
    textbox_tiles.png    BG0 tiles as palette indices (red = 16 * index); battle.json textbox_map: the 32x64 tilemap
                         (message box at BG0_Y 0, action menu at 160, move menu at 320); palettes: text (0-1, 5),
                         frames (the 20 window frame palettes: palette 1 is the player's Options frame type)
    cursor_tiles.png     the menu cursor (textbox tiles 1-2, palette 1)
    frame_tiles.png      the 20 window frames' 9 tiles (row i = frame type i + 1), over BG0 tiles 0x12 / 0x22
    healthbox_player.png, healthbox_opponent.png, hpbar.png, status.png (PSN PAR SLP FRZ BRN, 8px rows)
    font_{normal,small,narrow}.png  glyphs (16x16 cells) as palette indices: red = 64 * index (1 text, 2 shadow)
    battle.json          glyph widths, the charmap (character -> glyph), per-species sprite offsets
                         (front y_offset - elevation, back y_offset), screen coordinates (src/battle_anim_mons.c,
                         src/battle_interface.c)

    build_battle_assets.py [--pokeemerald ~/Dev/pokeemerald] [--out dist/battle]
"""

import argparse
import json
import os
import re
import struct

from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def gbapal(path):
    """A GBA palette file (BGR555) as RGB, converted like the hardware / mGBA show it (c << 3 | c >> 2)."""
    raw = open(path, "rb").read()
    out = []
    for i in range(0, len(raw) - 1, 2):
        v = raw[i] | raw[i + 1] << 8
        out.append(tuple((c << 3) | (c >> 2) for c in (v & 31, (v >> 5) & 31, (v >> 10) & 31)))
    return out


def jasc(path):
    v = open(path).read().split()[3:]
    return [tuple(int(x) for x in v[i:i + 3]) for i in range(0, len(v), 3)]


def render_map(tiles_png, map_bin, palette, pal_base, size, backdrop=None):
    """A 32-tile-wide text BG: 4bpp tiles (indexed PNG), u16 map entries (tile, hflip, vflip, palette)."""
    tiles = Image.open(tiles_png)
    tw, tb = tiles.size[0] // 8, tiles.tobytes()
    m = open(map_bin, "rb").read()
    img = Image.new("RGBA", (256, (len(m) // 64) * 8))
    px = img.load()
    for i in range(len(m) // 2):
        e = struct.unpack_from("<H", m, 2 * i)[0]
        ti, hf, vf, pn = e & 0x3FF, (e >> 10) & 1, (e >> 11) & 1, (e >> 12) & 15
        tx, ty = (ti % tw) * 8, (ti // tw) * 8
        for y in range(8):
            for x in range(8):
                c = tb[(ty + (7 - y if vf else y)) * tiles.size[0] + tx + (7 - x if hf else x)] & 15
                k = (pn - pal_base) * 16 + c
                if c == 0 and backdrop is not None:
                    px[(i % 32) * 8 + x, (i // 32) * 8 + y] = tuple(backdrop) + (255,)
                    continue
                px[(i % 32) * 8 + x, (i // 32) * 8 + y] = palette[k] + (255,) if 0 <= k < len(palette) else (0, 0, 0, 0)
    return img.crop(size)


def indexed_rgba(path, transparent0=True):
    im = Image.open(path)
    pal = im.getpalette()
    out = Image.new("RGBA", im.size)
    out.putdata([(0, 0, 0, 0) if (transparent0 and b == 0) else (pal[3 * b], pal[3 * b + 1], pal[3 * b + 2], 255)
                 for b in im.tobytes()])
    return out


def coords(text, key="y_offset"):
    out = {}
    for sp, body in re.findall(r"\[SPECIES_(\w+)\]\s*=\s*\{([^}]*)\}", text):
        m = re.search(key + r"\s*=\s*(\d+)", body)
        if m:
            out[sp] = int(m.group(1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pokeemerald", default=os.path.expanduser("~/Dev/pokeemerald"))
    ap.add_argument("--out", default=os.path.join(REPO, "dist", "battle"))
    args = ap.parse_args()
    pe = args.pokeemerald
    g = lambda *p: os.path.join(pe, "graphics", *p)       # noqa: E731
    os.makedirs(args.out, exist_ok=True)

    # BG3: the room; its rows under the message box use BG palette 0 (the textbox palette)
    BACKDROP = (74, 66, 82)     # what the hardware shows through transparent pixels of every layer in these battles
    bgpal = [BACKDROP] + gbapal(g("battle_interface", "textbox.gbapal"))[1:16] + [(0, 0, 0)] * 16 + \
        gbapal(g("battle_environment", "stadium", "battle_frontier.gbapal"))
    render_map(g("battle_environment", "building", "tiles.png"), g("battle_environment", "building", "map.bin"),
               bgpal, 0, (0, 0, 240, 160), BACKDROP).save(os.path.join(args.out, "bg.png"), optimize=True)
    # BG0 (messages, action menu, move menu: the game scrolls BG0 by 0 / 160 / 320). Palette 0 = textbox, 1 = the
    # player's window frame (Options: frame type, LoadUserWindowBorderGfx), 5 = window text: composed at runtime.
    tb = Image.open(g("battle_interface", "textbox.png"))
    idx = Image.new("RGBA", tb.size)
    idx.putdata([((b & 15) * 16, 0, 0, 255) for b in tb.tobytes()])
    idx.save(os.path.join(args.out, "textbox_tiles.png"))
    tmap = list(struct.unpack(f"<{len(open(g('battle_interface', 'textbox_map.bin'), 'rb').read()) // 2}H",
                              open(g("battle_interface", "textbox_map.bin"), "rb").read()))
    text_pals = gbapal(g("battle_interface", "textbox.gbapal")) + [(0, 0, 0)] * 48 + gbapal(g("battle_interface", "text.gbapal"))
    frames = [gbapal(g("text_window", f"{i}.gbapal")) for i in range(1, 21)]
    # the frames' 9 tiles (LoadUserWindowBorderGfx loads them over BG0 tiles 0x12 and 0x22): one row of 9 tiles per frame
    ft = Image.new("RGBA", (72, 8 * 20))
    for i in range(20):
        fim = Image.open(g("text_window", f"{i + 1}.png"))
        fb = fim.tobytes()
        for t in range(9):
            for y in range(8):
                for x in range(8):
                    c = fb[((t // 3) * 8 + y) * 24 + (t % 3) * 8 + x] & 15
                    ft.putpixel((t * 8 + x, i * 8 + y), (c * 16, 0, 0, 255))
    ft.save(os.path.join(args.out, "frame_tiles.png"))
    # the menu cursor: textbox tiles 1 and 2 stacked, palette 1 (ActionSelectionCreateCursorAt)
    tb_tiles = Image.open(g("battle_interface", "textbox.png"))
    tw, tbb = tb_tiles.size[0] // 8, tb_tiles.tobytes()
    cur = Image.new("RGBA", (8, 16))
    for k, ti in enumerate((1, 2)):
        for y in range(8):
            for x in range(8):
                c = tbb[((ti // tw) * 8 + y) * tb_tiles.size[0] + (ti % tw) * 8 + x] & 15
                cur.putpixel((x, 8 * k + y), (c * 16, 0, 0, 255))
    cur.save(os.path.join(args.out, "cursor_tiles.png"))     # palette indices (palette 1: the window frame)
    hb_pal = gbapal(g("battle_interface", "ball_status_bar.gbapal"))
    bar_pal = gbapal(g("battle_interface", "ball_display.gbapal"))
    def with_pal(src, pal, dst):
        im = Image.open(g("battle_interface", src))
        out = Image.new("RGBA", im.size)
        out.putdata([(0, 0, 0, 0) if b == 0 else pal[b] + (255,) for b in im.tobytes()])
        out.save(os.path.join(args.out, dst), optimize=True)
    with_pal("healthbox_singles_player.png", hb_pal, "healthbox_player.png")
    with_pal("healthbox_singles_opponent.png", hb_pal, "healthbox_opponent.png")
    indexed_rgba(g("battle_interface", "status2.png")).save(os.path.join(args.out, "status.png"), optimize=True)
    with_pal("hpbar.png", bar_pal, "hpbar.png")
    pp_pal = gbapal(g("battle_interface", "text_pp.gbapal"))

    # fonts as palette indices (red channel = index * 64: 0 background, 1 text, 2 shadow, 3 alt)
    fonts_c = open(os.path.join(pe, "src", "fonts.c")).read()
    fonts = {}
    for name, png, table in (("normal", "latin_normal.png", "gFontNormalLatinGlyphWidths"),
                             ("small", "latin_small.png", "gFontSmallLatinGlyphWidths"),
                             ("narrow", "latin_narrow.png", "gFontNarrowLatinGlyphWidths")):
        f = Image.open(g("fonts", png))
        out = Image.new("RGBA", f.size)
        out.putdata([(b * 64, 0, 0, 255 if b else 0) for b in f.tobytes()])
        out.save(os.path.join(args.out, f"font_{name}.png"), optimize=True)
        widths = [int(x) for x in re.search(table + r"\[\]\s*=\s*\{([^}]*)\}", fonts_c).group(1)
                  .replace("\n", " ").split(",") if x.strip()]
        fonts[name] = {"file": f"font_{name}.png", "widths": widths}
    charmap = {}
    for line in open(os.path.join(pe, "charmap.txt"), encoding="utf-8"):
        m = re.match(r"^'(\\'|[^'])'\s*=\s*([0-9A-Fa-f]{2})\s*(?:@.*)?$", line.strip())
        if m:
            ch = "'" if m.group(1) == "\\'" else m.group(1)
            charmap.setdefault(ch, int(m.group(2), 16))

    names = json.load(open(os.path.join(REPO, "pybattle", "data", "game_data.json")))["names"]["species"]
    sid = {v[len("SPECIES_"):]: int(k) for k, v in names.items()}
    gfx = os.path.join(pe, "src", "data", "pokemon_graphics")
    front = coords(open(os.path.join(gfx, "front_pic_coordinates.h")).read())
    back = coords(open(os.path.join(gfx, "back_pic_coordinates.h")).read())
    elev = {sp: int(v) for sp, v in re.findall(r"\[SPECIES_(\w+)\]\s*=\s*(\d+)", open(os.path.join(gfx, "enemy_mon_elevation.h")).read())}
    mons = {}
    for sp, i in sid.items():
        if sp in front or sp in back:
            mons[i] = [front.get(sp, 0), back.get(sp, 0), elev.get(sp, 0)]
    meta = {
        "screen": [240, 160], "backdrop": list(BACKDROP),
        "fonts": fonts, "cell": 16, "per_row": 16, "charmap": charmap,
        "palettes": {"text": text_pals, "frames": frames, "healthbox": hb_pal, "healthbar": bar_pal, "pp": pp_pal},
        "textbox_map": tmap,
        "coords": {"player": [72, 80], "opponent": [176, 40], "player_box": [158, 88], "opponent_box": [44, 30],
                   "message": [16, 120]},
        "mons": mons,       # species -> [front y_offset, back y_offset, elevation]
    }
    with open(os.path.join(args.out, "battle.json"), "w") as fh:
        json.dump(meta, fh, ensure_ascii=False, separators=(",", ":"))
    print(f"{args.out}: {sorted(os.listdir(args.out))}, {len(mons)} species offsets, {len(charmap)} chars")


if __name__ == "__main__":
    main()

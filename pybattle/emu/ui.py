"""Drawing for the watch window (pybattle/emu/viewer.py): theme, fonts, widgets and the info panels.

Everything is measured in `u`, the body font size in pixels, so the whole window scales with it.
Panels are drawn from the structured views (RentalView / SwapView / BattleView) and, when the agent
gives them, `Choice.details` (probabilities, P(win), damage estimates); an agent that only gives a
reason string still gets the same panels, without the numbers it didn't provide.
"""

import os
from typing import Dict, List, Optional, Tuple

import pygame

from ..view import (ACTION_CANT_MOVE, ACTION_MOVE, ACTION_SWITCH, HP_BAR_PIXELS, MAJOR_STATUSES, MOVES, NAMES,
                    NO_HINT_TYPE, WEATHERS, BattleView, RentalView, SwapView)

# ---- theme ------------------------------------------------------------------------------------------------

DARK = dict(
    bg=(12, 14, 20), panel=(21, 24, 33), card=(29, 33, 45), card_hi=(36, 41, 56), border=(46, 52, 68),
    text=(232, 234, 240), muted=(146, 153, 172), faint=(96, 103, 122), accent=(255, 196, 61),
    on_accent=(24, 20, 8), good=(74, 222, 128), warn=(250, 204, 21), bad=(248, 113, 113), info=(110, 170, 255),
    bar_bg=(44, 49, 64), chosen=(255, 196, 61), chosen_bg=(54, 47, 24), button=(38, 43, 58),
    button_hover=(54, 61, 82), button_down=(30, 34, 46), button_on=(38, 88, 60), danger=(120, 46, 46),
    screen_bg=(0, 0, 0))
LIGHT = dict(
    bg=(232, 234, 240), panel=(248, 249, 252), card=(238, 240, 246), card_hi=(228, 232, 242),
    border=(206, 211, 224), text=(24, 27, 36), muted=(92, 99, 118), faint=(150, 156, 172), accent=(196, 132, 0),
    on_accent=(255, 255, 255), good=(22, 150, 72), warn=(190, 140, 0), bad=(210, 50, 50), info=(40, 110, 220),
    bar_bg=(218, 222, 232), chosen=(196, 132, 0), chosen_bg=(255, 240, 200), button=(226, 229, 238),
    button_hover=(212, 217, 230), button_down=(200, 205, 220), button_on=(190, 230, 205), danger=(240, 200, 200),
    screen_bg=(0, 0, 0))
THEMES = {"dark": DARK, "light": LIGHT}

# game type order (TYPE_NORMAL = 0 ... TYPE_DARK = 17), the usual type palette
TYPE_COLORS = [(168, 168, 120), (192, 48, 40), (168, 144, 240), (160, 64, 160), (224, 192, 104), (184, 160, 56),
               (168, 184, 32), (112, 88, 152), (184, 184, 208), (104, 160, 144), (240, 128, 48), (104, 144, 240),
               (120, 200, 80), (248, 208, 48), (248, 88, 136), (152, 216, 216), (112, 56, 248), (112, 88, 72)]
HP_GREEN, HP_YELLOW, HP_RED = (72, 200, 120), (240, 200, 48), (232, 72, 56)
STATUS_COLORS = {1: (140, 140, 160), 2: (160, 64, 160), 3: (240, 128, 48), 4: (120, 190, 230), 5: (220, 180, 40),
                 6: (130, 40, 150)}
STATUS_SHORT = {1: "SLP", 2: "PSN", 3: "BRN", 4: "FRZ", 5: "PAR", 6: "TOX"}
STYLES = ["none", "preparation", "slow & steady", "endurance", "high risk", "weakening", "unpredictable", "weather"]
STAGE_NAMES = ("Atk", "Def", "Spe", "SpA", "SpD", "Acc", "Eva")


def hp_color(pixels: int) -> Tuple[int, int, int]:
    """The game's thresholds on the 48-pixel bar."""
    return HP_GREEN if pixels > 24 else HP_YELLOW if pixels > 9 else HP_RED


def type_text_color(t: int):
    r, g, b = TYPE_COLORS[t]
    return (20, 20, 20) if 0.299 * r + 0.587 * g + 0.114 * b > 150 else (250, 250, 250)


def nm(kind: str, i) -> str:
    n = NAMES[kind].get(str(i), str(i))
    for p in ("SPECIES_", "MOVE_", "ITEM_", "ABILITY_", "TYPE_"):
        n = n.replace(p, "")
    return n.replace("_", " ").title()


def type_name(t: int) -> str:
    return nm("types", t) if t < 18 else "mixed"


def move_type(mv: int) -> int:
    try:
        return MOVES[mv]["type"]
    except (IndexError, KeyError, TypeError):
        return 0


def pct(x: float) -> str:
    return f"{x:.0%}" if x >= 0.01 or x == 0 else "<1%"


# ---- fonts --------------------------------------------------------------------------------------------------

def _font_files() -> Dict[str, Optional[str]]:
    """Inter if installed (Regular/Medium/SemiBold/Bold), else DejaVu Sans, else pygame's default font."""
    out: Dict[str, Optional[str]] = {}
    inter = pygame.font.match_font("inter")
    if inter and os.path.basename(inter).lower().startswith("inter"):
        d = os.path.dirname(inter)
        for w, f in (("regular", "Inter-Regular"), ("medium", "Inter-Medium"), ("semibold", "Inter-SemiBold"),
                     ("bold", "Inter-Bold")):
            for ext in (".otf", ".ttf"):
                if os.path.exists(os.path.join(d, f + ext)):
                    out[w] = os.path.join(d, f + ext)
                    break
    if "regular" not in out:
        reg = pygame.font.match_font("dejavusans,notosans,liberationsans,arial,sans")
        bold = pygame.font.match_font("dejavusans,notosans,liberationsans,arial,sans", bold=True)
        out["regular"] = reg
        out["bold"] = bold or reg
    out.setdefault("medium", out["regular"])
    out.setdefault("semibold", out.get("bold") or out["regular"])
    out.setdefault("bold", out["semibold"])
    return out


class Fonts:
    def __init__(self):
        self.files = _font_files()
        self._cache: Dict[Tuple[str, int], pygame.font.Font] = {}
        self._text: Dict[tuple, pygame.Surface] = {}

    def get(self, weight: str, px: int) -> pygame.font.Font:
        px = max(8, int(round(px)))
        key = (weight, px)
        f = self._cache.get(key)
        if f is None:
            path = self.files.get(weight)
            try:
                f = pygame.font.Font(path, px)
            except (OSError, FileNotFoundError):
                f = pygame.font.Font(None, int(px * 1.3))
            self._cache[key] = f
        return f

    def render(self, text: str, font: pygame.font.Font, color) -> pygame.Surface:
        key = (text, id(font), color)
        s = self._text.get(key)
        if s is None:
            if len(self._text) > 4000:
                self._text.clear()
            s = font.render(text, True, color)
            self._text[key] = s
        return s


# ---- painter -------------------------------------------------------------------------------------------------

class Painter:
    """Draws on one surface at one scale (u = body font px) with one theme."""

    def __init__(self, surf: pygame.Surface, fonts: Fonts, u: float, theme: dict):
        self.s, self.fonts, self.u, self.t = surf, fonts, u, theme
        self.body = fonts.get("regular", u)
        self.medium = fonts.get("medium", u)
        self.strong = fonts.get("semibold", u)
        self.small = fonts.get("regular", u * 0.82)
        self.small_b = fonts.get("semibold", u * 0.82)
        self.tiny = fonts.get("semibold", u * 0.7)
        self.title = fonts.get("semibold", u * 1.12)
        self.big = fonts.get("bold", u * 1.7)
        self.pad = max(4, round(u * 0.55))
        self.gap = max(3, round(u * 0.4))
        self.radius = max(3, round(u * 0.35))

    # text
    def ellipsize(self, text: str, font, max_w: float) -> str:
        if max_w <= 0:
            return ""
        if font.size(text)[0] <= max_w:
            return text
        lo, hi = 0, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if font.size(text[:mid].rstrip() + "…")[0] <= max_w:
                lo = mid
            else:
                hi = mid - 1
        return (text[:lo].rstrip() + "…") if lo > 0 else ""

    def text(self, text: str, x: float, y: float, font=None, color=None, max_w: float = None, align="left",
             valign="top") -> int:
        """Draw one line (ellipsized to max_w). Returns its width."""
        font = font or self.body
        color = color or self.t["text"]
        if max_w is not None:
            text = self.ellipsize(text, font, max_w)
        if not text:
            return 0
        surf = self.fonts.render(text, font, color)
        w, h = surf.get_size()
        if align == "right":
            x -= w
        elif align == "center":
            x -= w / 2
        if valign == "center":
            y -= h / 2
        self.s.blit(surf, (round(x), round(y)))
        return w

    def wrap(self, text: str, font, max_w: float) -> List[str]:
        out = []
        for para in text.split("\n"):
            words, line = para.split(" "), ""
            for w in words:
                cand = w if not line else line + " " + w
                if font.size(cand)[0] <= max_w:
                    line = cand
                    continue
                if line:
                    out.append(line)
                while font.size(w)[0] > max_w and len(w) > 1:       # a word longer than the line: cut it
                    k = len(w)
                    while k > 1 and font.size(w[:k])[0] > max_w:
                        k -= 1
                    out.append(w[:k])
                    w = w[k:]
                line = w
            out.append(line)
        return out

    def paragraph(self, text: str, x, y, w, font=None, color=None) -> float:
        font = font or self.body
        lh = font.get_linesize()
        for line in self.wrap(text, font, w):
            self.text(line, x, y, font, color)
            y += lh
        return y

    # shapes
    def rect(self, color, r, radius=None, width=0):
        pygame.draw.rect(self.s, color, pygame.Rect(r), width, border_radius=self.radius if radius is None else radius)

    def chip(self, text: str, x, y, bg, fg, font=None, max_w=None) -> int:
        """A rounded label; returns its width (0 if it doesn't fit in max_w)."""
        font = font or self.tiny
        px = max(3, round(self.u * 0.35))
        tw = font.size(text)[0]
        if max_w is not None and tw + 2 * px > max_w:
            text = self.ellipsize(text, font, max_w - 2 * px)
            if not text:
                return 0
            tw = font.size(text)[0]
        h = self.chip_h(font)
        self.rect(bg, (x, y, tw + 2 * px, h), radius=h // 2)
        self.text(text, x + px, y + h / 2, font, fg, valign="center")
        return tw + 2 * px

    def chip_w(self, text: str, font=None) -> int:
        font = font or self.tiny
        return font.size(text)[0] + 2 * max(3, round(self.u * 0.35))

    def chip_h(self, font=None) -> int:
        font = font or self.tiny
        return font.get_height() + max(2, round(self.u * 0.12))

    def bar(self, x, y, w, h, frac, color, bg=None):
        bg = bg or self.t["bar_bg"]
        r = max(1, h // 2)
        self.rect(bg, (x, y, w, h), radius=r)
        fw = round(max(0.0, min(1.0, frac)) * w)
        if fw > 0:
            self.rect(color, (x, y, max(fw, min(w, 2 * r)), h), radius=r)

    def type_chip(self, t: int, x, y, max_w=None) -> int:
        if t >= 18:
            return 0
        return self.chip(type_name(t).upper(), x, y, TYPE_COLORS[t], type_text_color(t), max_w=max_w)

    def section(self, title: str, x, y, w, right: str = "", right_color=None) -> float:
        """A small caps section title with a rule; returns the y below it."""
        f = self.small_b
        tw = self.text(title.upper(), x, y, f, self.t["muted"], max_w=w * 0.6)
        if right:
            self.text(right, x + w, y, f, right_color or self.t["muted"], max_w=w - tw - self.gap * 2, align="right")
        y += f.get_linesize()
        pygame.draw.line(self.s, self.t["border"], (x, y), (x + w, y))
        return y + self.gap


# ---- cards ---------------------------------------------------------------------------------------------------

def _chips_row(p: Painter, x, y, w, chips) -> float:
    """chips: [(text, bg, fg)]; wraps onto more rows. Returns the y below."""
    if not chips:
        return y
    h = p.chip_h()
    cx = x
    for text, bg, fg in chips:
        cw = p.chip_w(text)
        if cx > x and cx + cw > x + w:
            cx, y = x, y + h + p.gap // 2
        cx += p.chip(text, cx, y, bg, fg, max_w=w) + p.gap // 2
    return y + h + p.gap // 2


def own_card(p: Painter, x, y, w, mon, *, active=False, badge=None, badge_color=None, est=None,
             highlight_move=None, show_stats=False, dim=False, footer=0) -> float:
    """One of our Pokemon (or a rental). est: per move (min, max, can_ko) vs the foe. show_stats: the stats
    and ability instead of the HP bar (rentals, swaps: HP is full). footer: extra room at the bottom (px).
    Returns the bottom y."""
    t, pad = p.t, p.pad
    x0, y0, iw = x + pad, y + pad, w - 2 * pad
    h = _own_card_height(p, w, mon, show_stats) + footer
    border = t["accent"] if active else t["border"]
    p.rect(t["card_hi"] if active else t["card"], (x, y, w, h))
    p.rect(border, (x, y, w, h), width=2 if active else 1)
    fainted = mon.hp == 0 and mon.max_hp > 0
    # row 1: name + badge
    bx = x0 + iw
    if badge:
        bw = p.chip_w(badge)
        p.chip(badge, bx - bw, y0 + 1, badge_color or t["accent"], t["on_accent"])
        bx -= bw + p.gap
    if fainted:
        bw = p.chip_w("FAINTED")
        p.chip("FAINTED", bx - bw, y0 + 1, t["bad"], (255, 255, 255))
        bx -= bw + p.gap
    name_col = t["faint"] if (dim or fainted) else t["text"]
    p.text(nm("species", mon.species), x0, y0, p.strong, name_col, max_w=bx - x0)
    y1 = y0 + p.strong.get_linesize()
    # row 2: types + item
    cx = x0
    for ty in dict.fromkeys(mon.types):
        cx += p.type_chip(ty, cx, y1, max_w=iw / 2) + p.gap // 2
    if mon.item:
        p.text("@ " + nm("items", mon.item), x0 + iw, y1, p.small, t["muted"], max_w=x0 + iw - cx - p.gap,
               align="right")
    y1 += p.chip_h() + p.gap
    if not show_stats:
        # row 3: HP
        frac = mon.hp / mon.max_hp if mon.max_hp else 0
        pixels = round(frac * HP_BAR_PIXELS) if mon.hp else 0
        if mon.hp and pixels == 0:
            pixels = 1
        bh = max(4, round(p.u * 0.42))
        p.bar(x0, y1, iw, bh, frac, hp_color(pixels))
        y1 += bh + 2
        p.text(f"{mon.hp}/{mon.max_hp} HP", x0, y1, p.small, t["text"])
        rx = x0 + iw
        if mon.status:
            s = STATUS_SHORT.get(mon.status, MAJOR_STATUSES[mon.status][:3].upper())
            rx -= p.chip(s, rx - p.chip_w(s), y1 + 1, STATUS_COLORS.get(mon.status, t["faint"]), (255, 255, 255))
            rx -= p.gap
        p.text(pct(frac), rx, y1, p.small, t["muted"], align="right")
        y1 += p.small.get_linesize() + p.gap // 2
    else:
        for line in _stat_lines(p, mon, iw):
            p.text(line, x0, y1, p.small, t["muted"], max_w=iw)
            y1 += p.small.get_linesize()
        y1 += p.gap // 2
    if fainted:
        est = None
    # moves
    for i, mv in enumerate(mon.moves[:4]):
        y1 = _move_row(p, x0, y1, iw, mv, mon.pp[i] if i < len(mon.pp) else None,
                       est[i] if est is not None and i < len(est) else None, highlight=highlight_move == i,
                       dim=dim or fainted)
    return y + h


def _stat_lines(p: Painter, mon, w) -> List[str]:
    st = mon.stats
    parts = [f"HP {mon.max_hp}", f"Atk {st[0]}", f"Def {st[1]}", f"SpA {st[3]}", f"SpD {st[4]}", f"Spe {st[2]}"]
    lines = p.wrap("  ".join(parts).replace(" ", "\u00a0").replace("\u00a0\u00a0", "  "), p.small, w)
    lines = [ln.replace("\u00a0", " ") for ln in lines]
    if mon.ability:
        lines.append(nm("abilities", mon.ability))
    return lines


def _own_card_height(p: Painter, w, mon, show_stats) -> int:
    h = p.pad * 2 + p.strong.get_linesize() + p.chip_h() + p.gap
    if show_stats:
        h += p.small.get_linesize() * len(_stat_lines(p, mon, w - 2 * p.pad)) + p.gap // 2
    else:
        h += max(4, round(p.u * 0.42)) + 2 + p.small.get_linesize() + p.gap // 2
    h += _move_row_h(p) * len(mon.moves[:4])
    return h


def _move_row_h(p: Painter) -> int:
    return p.small.get_linesize() + max(2, round(p.u * 0.18))


def _move_row(p: Painter, x, y, w, mv, pp, est, highlight=False, dim=False, pp_label=True) -> float:
    t = p.t
    h = _move_row_h(p)
    if not mv:
        p.text("—", x, y, p.small, t["faint"])
        return y + h
    ty = move_type(mv)
    if highlight:
        p.rect(t["chosen_bg"], (x - 3, y, w + 6, h - 1), radius=3)
    sw = max(3, round(p.u * 0.22))
    p.rect(TYPE_COLORS[ty], (x, y + 2, sw, h - 5), radius=2)
    rx = x + w
    col = t["faint"] if dim else t["text"]
    if est is not None:
        lo, hi, ko = est[0], est[1], est[2]
        if ko:
            rx -= p.chip("KO", rx - p.chip_w("KO"), y + (h - p.chip_h()) / 2, t["bad"], (255, 255, 255)) + p.gap // 2
        if hi > 0:
            txt = f"{min(lo, 1):.0%}–{min(hi, 1):.0%}" if hi < 1.0 or lo < 1.0 else "100%"
            rx -= p.text(txt, rx, y, p.small_b, t["good"] if hi >= 0.5 else col, align="right") + p.gap
    if pp is not None and pp_label:
        mx = MOVES[mv]["pp"] if mv < len(MOVES) else 0
        ppt = f"{pp}/{mx}" if mx else f"{pp}"
        ppw = p.small.size(ppt)[0]
        if rx - x - ppw > w * 0.45:
            rx -= p.text(ppt, rx, y, p.small, t["bad"] if pp == 0 else t["faint"], align="right") + p.gap
    p.text(nm("moves", mv), x + sw + p.gap, y, p.small, col, max_w=rx - x - sw - p.gap)
    return y + h


def foe_card(p: Painter, x, y, w, mon, *, active=False, est=None, h=None, label=None) -> float:
    """One of the opponent's Pokemon, as far as we've seen it. est: per revealed move (min, max, can_ko) vs us."""
    t, pad = p.t, p.pad
    x0, y0, iw = x + pad, y + pad, w - 2 * pad
    h = h or foe_card_height(p, [mon])
    p.rect(t["card_hi"] if active else t["card"], (x, y, w, h))
    p.rect(t["bad"] if active else t["border"], (x, y, w, h), width=2 if active else 1)
    known = mon.seen or mon.species
    bx = x0 + iw
    if label:
        bw = p.chip_w(label)
        p.chip(label, bx - bw, y0 + 1, t["bad"], (255, 255, 255))
        bx -= bw + p.gap
    if mon.fainted:
        bw = p.chip_w("FAINTED")
        p.chip("FAINTED", bx - bw, y0 + 1, t["faint"], (255, 255, 255))
        bx -= bw + p.gap
    p.text(nm("species", mon.species) if known and mon.species else "?  not seen yet", x0, y0, p.strong,
           t["text"] if known and not mon.fainted else t["faint"], max_w=bx - x0)
    y1 = y0 + p.strong.get_linesize()
    cx = x0
    if known:
        for ty in dict.fromkeys(mon.types):
            cx += p.type_chip(ty, cx, y1, max_w=iw / 2) + p.gap // 2
        if mon.revealed_item:
            p.text("@ " + nm("items", mon.revealed_item), x0 + iw, y1, p.small, t["muted"],
                   max_w=x0 + iw - cx - p.gap, align="right")
    y1 += p.chip_h() + p.gap
    bh = max(4, round(p.u * 0.42))
    if mon.hp_pixels >= 0:
        p.bar(x0, y1, iw, bh, mon.hp_pixels / HP_BAR_PIXELS, hp_color(mon.hp_pixels))
    else:
        p.bar(x0, y1, iw, bh, 0, t["faint"])
    y1 += bh + 2
    p.text(f"{round(100 * mon.hp_pixels / HP_BAR_PIXELS)}% HP" if mon.hp_pixels >= 0 else "HP unknown", x0, y1,
           p.small, t["text"] if mon.hp_pixels >= 0 else t["faint"])
    if mon.status:
        s = STATUS_SHORT.get(mon.status, "?")
        p.chip(s, x0 + iw - p.chip_w(s), y1 + 1, STATUS_COLORS.get(mon.status, t["faint"]), (255, 255, 255))
    y1 += p.small.get_linesize() + p.gap // 2
    if known:
        if mon.revealed_ability:
            ab = nm("abilities", mon.revealed_ability)
        else:
            ab = " / ".join(nm("abilities", a) for a in mon.possible_abilities if a) + " (?)"
        p.text(ab, x0, y1, p.small, t["muted"], max_w=iw)
    y1 += p.small.get_linesize()
    moves = mon.revealed_moves[:4]
    for i, mv in enumerate(moves):
        y1 = _move_row(p, x0, y1, iw, mv, None, est[i] if est and i < len(est) else None)
    if len(moves) < 4:
        p.text("no moves seen yet" if not moves else f"+ {4 - len(moves)} not seen yet", x0, y1, p.small, t["faint"])
    return y + h


def foe_card_height(p: Painter, mons) -> int:
    n = max(len(m.revealed_moves[:4]) + (len(m.revealed_moves) < 4) for m in mons)
    return (p.pad * 2 + p.strong.get_linesize() + p.chip_h() + p.gap + max(4, round(p.u * 0.42)) + 2
            + p.small.get_linesize() + p.gap // 2 + p.small.get_linesize() + n * _move_row_h(p))


def _grid(n_cols, x, w, gap):
    cw = (w - gap * (n_cols - 1)) / n_cols
    return [(round(x + i * (cw + gap)), round(cw)) for i in range(n_cols)]


# ---- state panel -----------------------------------------------------------------------------------------------

def draw_state(p: Painter, x, y, w, scene) -> float:
    """What the agent sees: returns the bottom y."""
    view = scene.view
    if isinstance(view, BattleView):
        return _battle_state(p, x, y, w, scene)
    if isinstance(view, RentalView):
        return _rental_state(p, x, y, w, scene)
    if isinstance(view, SwapView):
        return _swap_state(p, x, y, w, scene)
    return y


def _hint(view) -> str:
    if view.hint_type == NO_HINT_TYPE and not view.hint_style:
        return ""
    ty = type_name(view.hint_type) if view.hint_type < 18 else "mixed types"
    return f"attendant's hint: {ty} · {STYLES[view.hint_style] if view.hint_style < len(STYLES) else '?'}"


def _chosen_move_slot(scene):
    a = scene.choice.action if scene.choice is not None else None
    if isinstance(a, tuple) and len(a) == 2 and a[0] == "move":
        return a[1]
    return None


def _battle_state(p: Painter, x, y, w, scene) -> float:
    v, t = scene.view, p.t
    est = scene.estimates
    a_own, a_foe = v.own_active.party_index, v.enemy_active.party_index
    y = p.section("Our team", x, y, w, f"turn {v.turn + 1}" + ("  ·  FORCED SWITCH" if v.forced_switch else ""),
                  t["warn"] if v.forced_switch else None)
    cols = _grid(3, x, w, p.gap)
    order = [a_own] + [i for i in range(3) if i != a_own]
    bottom = y
    for (cx, cw), i in zip(cols, order):
        mon = v.own_party[i]
        e = est[0][i] if est is not None else None
        b = own_card(p, cx, y, cw, mon, active=i == a_own and not v.forced_switch, est=e,
                     highlight_move=_chosen_move_slot(scene) if i == a_own else None,
                     badge="ACTIVE" if i == a_own else None)
        bottom = max(bottom, b)
    y = bottom + p.gap
    y = _field_row(p, x, y, w, v.own_active, v.own_side, mine=True)
    y += p.gap
    y = p.section("Opponent", x, y, w, _hint(v))
    order = [a_foe] + [i for i in range(3) if i != a_foe]
    fh = foe_card_height(p, v.enemy_party[:3])
    for (cx, cw), i in zip(cols, order):
        foe_card(p, cx, y, cw, v.enemy_party[i], active=i == a_foe, h=fh,
                 est=est[1] if est is not None and i == a_foe else None, label="ACTIVE" if i == a_foe else None)
    y += fh + p.gap
    y = _field_row(p, x, y, w, v.enemy_active, v.enemy_side, mine=False)
    chips = []
    if v.weather:
        chips.append((f"{WEATHERS[v.weather].upper()}" + (" (ability)" if v.weather_permanent else
                                                          f" {v.weather_turns_left} turns"),
                      t["info"], (255, 255, 255)))
    lt = _last_turn(v)
    if chips or lt:
        y += p.gap // 2
        cx = x
        for text, bg, fg in chips:
            cx += p.chip(text, cx, y, bg, fg) + p.gap
        if lt:
            p.text(lt, cx, y, p.small, t["muted"], max_w=x + w - cx)
        y += max(p.chip_h(), p.small.get_linesize())
    return y


def _field_row(p: Painter, x, y, w, act, side, mine) -> float:
    t = p.t
    chips = []
    for name, v in zip(STAGE_NAMES, act.stat_stages):
        if v:
            chips.append((f"{name} {v:+d}", t["good"] if v > 0 else t["bad"], (15, 15, 15)))
    flags = [("confused", "Confused"), ("infatuated", "Infatuated"), ("substitute", "Substitute"),
             ("leech_seeded", "Leech Seed"), ("cursed", "Curse"), ("trapped", "Trapped"), ("taunted", "Taunt"),
             ("yawn", "Drowsy"), ("focus_energy", "Focus Energy"), ("must_recharge", "Recharging"),
             ("perish_song", "Perish"), ("rooted", "Ingrain"), ("torment", "Torment")]
    for attr, label in flags:
        if getattr(act, attr, False):
            chips.append((label, t["warn"], (20, 20, 20)))
    if getattr(act, "encored_move", 0):
        chips.append(("Encore", t["warn"], (20, 20, 20)))
    if getattr(act, "disabled_move", 0):
        chips.append(("Disable: " + nm("moves", act.disabled_move), t["warn"], (20, 20, 20)))
    for attr, label in (("reflect_turns", "Reflect"), ("light_screen_turns", "Light Screen"),
                        ("safeguard_turns", "Safeguard"), ("mist_turns", "Mist")):
        n = getattr(side, attr, 0)
        if n:
            chips.append((f"{label} {n}", t["info"], (255, 255, 255)))
    if side.spikes:
        chips.append((f"Spikes ×{side.spikes}", t["faint"], (255, 255, 255)))
    if getattr(side, "future_sight_turns", 0):
        chips.append((f"Future Sight {side.future_sight_turns}", t["info"], (255, 255, 255)))
    if getattr(side, "wish_turns", 0):
        chips.append((f"Wish {side.wish_turns}", t["info"], (255, 255, 255)))
    if not chips:
        return y
    return _chips_row(p, x, y, w, chips)


def _last_turn(v) -> str:
    ev = v.last_turn
    if ev.turn < 0:
        return ""

    def act(a, mv, crit):
        if a == ACTION_MOVE:
            return nm("moves", mv) + (" (crit)" if crit == 1 else "")
        return {ACTION_SWITCH: "switched", ACTION_CANT_MOVE: "couldn't move"}.get(a, "didn't act")
    first = {0: "we moved first", 1: "foe moved first"}.get(ev.first, "")
    ours = "we " + act(ev.own_action, ev.own_move, ev.own_crit)
    if ev.own_action == ACTION_MOVE:
        ours += f" (−{round(100 * ev.damage_dealt_pixels / HP_BAR_PIXELS)}%)"
    theirs = "foe " + act(ev.enemy_action, ev.enemy_move, ev.enemy_crit)
    if ev.enemy_action == ACTION_MOVE:
        theirs += f" (−{ev.damage_taken} HP)"
    return f"last turn: {ours}, {theirs}" + (f", {first}" if first else "")


def _rental_state(p: Painter, x, y, w, scene) -> float:
    v, t = scene.view, p.t
    d = scene.details
    team = list(d.get("team") or []) if d else []
    if not team and scene.choice is not None and isinstance(scene.choice.action, (tuple, list)):
        team = list(scene.choice.action)
    lead_probs = d.get("lead_probs") if d else None
    y = p.section("Rental — pick 3 of 6", x, y, w, _hint(v))
    ncols = 3 if w > p.u * 30 else 2
    cols = _grid(ncols, x, w, p.gap)
    row_bottom = y
    for i, mon in enumerate(v.candidates):
        cx, cw = cols[i % ncols]
        if i % ncols == 0 and i:
            y = row_bottom + p.gap
        badge = None
        if i in team:
            k = team.index(i)
            badge = "LEAD" if k == 0 else f"#{k + 1}"
        bh = max(3, round(p.u * 0.25))
        foot = (bh + p.small.get_linesize()) if lead_probs is not None else 0
        b = own_card(p, cx, y, cw, mon, active=bool(team) and i == team[0], badge=badge, show_stats=True,
                     dim=bool(team) and i not in team, footer=foot)
        if lead_probs is not None:
            fy = b - p.pad - foot
            p.text("lead", cx + p.pad, fy, p.small, t["muted"])
            p.text(pct(lead_probs[i]), cx + cw - p.pad, fy, p.small_b, t["text"], align="right")
            p.bar(cx + p.pad, fy + p.small.get_linesize(), cw - 2 * p.pad, bh, lead_probs[i], t["accent"])
        row_bottom = max(row_bottom, b)
    return row_bottom + p.gap


def _swap_state(p: Painter, x, y, w, scene) -> float:
    v, t = scene.view, p.t
    act = scene.choice.action if scene.choice is not None else None
    out_i, in_j = (act[0], act[1]) if isinstance(act, (tuple, list)) and len(act) == 2 else (None, None)
    y = p.section("Swap — our team", x, y, w, "")
    cols = _grid(3, x, w, p.gap)
    bottom = y
    for (cx, cw), (i, mon) in zip(cols, enumerate(v.own_party[:3])):
        bottom = max(bottom, own_card(p, cx, y, cw, mon, badge="OUT" if i == out_i else None,
                                      badge_color=t["bad"], show_stats=True))
    y = bottom + p.gap
    y = p.section("Beaten team (can take one)", x, y, w, "next: " + _hint(v).replace("attendant's hint: ", ""))
    fh = foe_card_height(p, v.enemy_party[:3])
    for (cx, cw), (j, mon) in zip(cols, enumerate(v.enemy_party[:3])):
        foe_card(p, cx, y, cw, mon, h=fh, label="IN" if j == in_j else None)
    return y + fh + p.gap


# ---- decision panel --------------------------------------------------------------------------------------------

def _fallback_actions(scene):
    """Actions for an agent that gave no details: the legal options, the chosen one marked, no probabilities."""
    v, a = scene.view, scene.choice.action if scene.choice is not None else None
    acts = []
    if isinstance(v, BattleView):
        me = v.own_party[v.own_active.party_index]
        est = scene.estimates
        for la in v.legal_actions or []:
            if la[0] == "move":
                d = {"label": nm("moves", me.moves[la[1]]), "move": me.moves[la[1]], "action": la}
                if est is not None:
                    d["damage"] = est[0][v.own_active.party_index][la[1]]
            elif la[0] == "switch":
                d = {"label": "→ " + nm("species", v.own_party[la[1]].species), "action": la,
                     "species": v.own_party[la[1]].species}
                if est is not None:
                    d["threat"] = est[2][la[1]]
            else:
                continue
            acts.append(d)
    elif isinstance(v, SwapView):
        acts.append({"label": "Keep the team", "action": None})
        if isinstance(a, (tuple, list)):
            acts.append({"label": f"{nm('species', v.own_party[a[0]].species)} → "
                                  f"{nm('species', v.enemy_party[a[1]].species)}", "action": tuple(a)})
    elif isinstance(v, RentalView) and isinstance(a, (tuple, list)):
        for k, i in enumerate(a):
            acts.append({"label": ("lead: " if k == 0 else "") + nm("species", v.candidates[i].species),
                         "action": i, "chosen_all": True})
    norm = (lambda z: tuple(z) if isinstance(z, list) else z)
    chosen = next((n for n, d in enumerate(acts) if norm(d["action"]) == norm(a)), None)
    return acts, chosen


def draw_decision(p: Painter, x, y, w, scene, waiting: bool) -> float:
    t = p.t
    d = scene.details or {}
    title = "What the agent would do (you're playing)" if scene.suggestion else "Agent's decision"
    right, rc = ("waiting for Next", t["accent"]) if waiting else ("", None)
    if scene.choice is None:
        right, rc = "thinking…", t["muted"]
    y = p.section(title, x, y, w, right, rc)
    if scene.choice is None:
        return y
    # value
    if d.get("p_win") is not None:
        y = _gauge(p, x, y, w, "P(win this battle)", d["p_win"], pct(d["p_win"]))
    elif d.get("expected_wins") is not None:
        ew = d["expected_wins"]
        y = _gauge(p, x, y, w, "Expected wins ahead (the tactician's estimate)", None, f"≈ {ew:.1f}", color=t["info"])
    actions, chosen = (d.get("actions"), d.get("chosen")) if d.get("actions") else _fallback_actions(scene)
    kind = d.get("kind") or ("rental" if isinstance(scene.view, RentalView) else "")
    if actions:
        label = {"rental": "Lead", "swap": "Options"}.get(kind, "Options")
        if not d.get("actions"):
            label += " (this agent gives no probabilities)"
        y = p.section(label, x, y, w, "")
        y = _action_bars(p, x, y, w, actions, chosen)
    if d.get("pairs"):
        pairs = sorted(enumerate(d["pairs"]), key=lambda z: -z[1]["p"])[:4]
        cp = d.get("chosen_pair")
        y += p.gap
        y = p.section("Partners for the lead", x, y, w, "")
        y = _action_bars(p, x, y, w, [q for _, q in pairs],
                         next((n for n, (k, _) in enumerate(pairs) if k == cp), None))
    if scene.choice.reason:
        y += p.gap
        y = p.paragraph(scene.choice.reason, x, y, w, p.small, t["muted"])
    return y


def _gauge(p: Painter, x, y, w, label, frac, value, color=None) -> float:
    t = p.t
    if color is None:
        color = HP_GREEN if frac >= 0.6 else HP_YELLOW if frac >= 0.35 else HP_RED
    vw = p.text(value, x + w, y, p.big, color, align="right")
    lh = p.big.get_linesize()
    p.text(label, x, y + (lh - p.body.get_linesize()) / 2, p.medium, t["text"], max_w=w - vw - p.gap)
    y += lh
    if frac is None:
        return y + p.gap
    bh = max(5, round(p.u * 0.5))
    p.bar(x, y, w, bh, frac, color)
    return y + bh + p.gap * 2


def _action_bars(p: Painter, x, y, w, actions, chosen) -> float:
    t = p.t
    order = list(range(len(actions)))
    hidden = 0
    if any(a.get("p") is not None for a in actions):
        order.sort(key=lambda i: -(actions[i].get("p") or 0))
        if len(order) > 6:                              # long lists: fold the options under 1%
            keep = [i for i in order if (actions[i].get("p") or 0) >= 0.01 or i == chosen][:6]
            hidden = len(order) - len(keep)
            order = keep
    rh = p.body.get_linesize() + max(4, round(p.u * 0.45))
    sw = max(3, round(p.u * 0.22)) + p.gap
    label_w = min(w * 0.5, max(p.u * 6, 6 + sw + max(p.strong.size(a["label"])[0] for a in actions) + p.gap))
    extra_w = 0
    for a in actions:
        dmg, thr = a.get("damage"), a.get("threat")
        if dmg is not None and dmg[1] > 0:
            extra_w = max(extra_w, p.small_b.size("100%–100%")[0] + (p.chip_w("KO") + p.gap if dmg[2] else 0))
        elif thr is not None and thr[0] > 0:
            extra_w = max(extra_w, p.small.size("takes ≤100%")[0] + (p.chip_w("KO risk") + p.gap if thr[1] else 0))
    extra_w = min(extra_w, w * 0.3)
    for i in order:
        a = actions[i]
        is_c = i == chosen or a.get("chosen_all")
        if is_c:
            p.rect(t["chosen_bg"], (x - 4, y, w + 8, rh - 2), radius=4)
            p.rect(t["chosen"], (x - 4, y, 4, rh - 2), radius=2)
        cy = y + (rh - 2) / 2
        lx = x + 6
        if a.get("move"):
            sw = max(3, round(p.u * 0.22))
            p.rect(TYPE_COLORS[move_type(a["move"])], (lx, y + 3, sw, rh - 8), radius=2)
            lx += sw + p.gap
        font = p.strong if is_c else p.body
        p.text(a["label"], lx, cy, font, t["text"], max_w=x + label_w - lx, valign="center")
        # extra (damage / threat) on the right
        ex = x + w
        dmg, thr = a.get("damage"), a.get("threat")
        if dmg is not None and dmg[1] > 0:
            if dmg[2]:
                ex -= p.chip("KO", ex - p.chip_w("KO"), cy - p.chip_h() / 2, t["bad"], (255, 255, 255)) + p.gap // 2
            txt = f"{min(dmg[0], 1):.0%}–{min(dmg[1], 1):.0%}"
            p.text(txt, ex, cy, p.small_b, t["good"] if dmg[1] >= 0.5 else t["text"], max_w=extra_w,
                   align="right", valign="center")
        elif thr is not None and thr[0] > 0:
            txt = f"takes ≤{min(thr[0], 1):.0%}"
            if thr[1]:
                ex -= p.chip("KO risk", ex - p.chip_w("KO risk"), cy - p.chip_h() / 2, t["bad"],
                             (255, 255, 255)) + p.gap // 2
            p.text(txt, ex, cy, p.small, t["muted"], max_w=extra_w, align="right", valign="center")
        # probability bar
        bx0, bx1 = x + label_w + p.gap, x + w - extra_w - (p.gap * 2 if extra_w else 0)
        if a.get("p") is not None and bx1 - bx0 > p.u * 3:
            pw = p.small_b.size("100%")[0]
            bh = max(6, round(p.u * 0.62))
            bw = bx1 - bx0 - pw - p.gap
            p.bar(bx0, cy - bh / 2, bw, bh, a["p"], t["chosen"] if is_c else t["info"])
            p.text(pct(a["p"]), bx1, cy, p.small_b, t["text"] if is_c else t["muted"], align="right",
                   valign="center")
        elif is_c:
            p.text("chosen", bx0, cy, p.small_b, t["chosen"], valign="center")
        y += rh
    if hidden:
        p.text(f"+ {hidden} more under 1%", x + 6, y, p.small, t["faint"])
        y += p.small.get_linesize()
    return y


# ---- text-only scene (booting, walking, messages) --------------------------------------------------------------

def draw_message(p: Painter, x, y, w, lines: List[str]) -> float:
    for line in lines:
        y = p.paragraph(line, x, y, w, p.body, p.t["muted"])
    return y

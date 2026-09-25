"""A window to watch the real game while agents (or you) play it.

Layout (it follows the window size; resize, maximize or go fullscreen freely):

    ┌ header: phase · round · battle n/7 (Noland) · streak · run wins · rentals · mode · speed ──────┐
    │ GBA screen                     │ what the agent sees: our team, the opponent, field          │
    │                                │                                                             │
    ├────────────────────────────────┤ the agent's decision: P(win) / expected wins, every option  │
    │ decision log (mouse wheel)     │ as a probability bar (chosen one highlighted), damage       │
    └ controls: Pause · Next · Step │ − speed + │ Human │ A− A+ · Theme · Fullscreen · Quit ───────┘

Threads. The window lives in the main thread (`Viewer.run`), which handles clicks and keys and
redraws ~60 times a second no matter what the game is doing. The game — emulator, driver, agents —
runs in a worker thread. `ViewedMachine` wraps the headless emulator: after every emulated frame it
calls `Viewer.frame_done`, which paces the game to the chosen speed (read fresh every frame, so a
speed change applies at once), holds it while paused, and hands the latest picture to the window.
So the controls respond at every speed, while the agent is thinking, and in step mode.

The keyboard drives the GBA only in human mode: arrows, Z = A, X = B, Enter = Start,
Backspace = Select, A = L, S = R. Outside human mode a few shortcuts also work: Space pause,
+ / - speed, Enter / N next, F11 fullscreen.
"""

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pygame

from . import native
from . import ui

K = native
SPEEDS = [1, 2, 4, 8, 0]           # 0 = as fast as possible
UI_FPS = 60
FONT_SCALES = [0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4, 1.6]

HUMAN_KEYS = {
    pygame.K_UP: K.KEY_UP, pygame.K_DOWN: K.KEY_DOWN, pygame.K_LEFT: K.KEY_LEFT, pygame.K_RIGHT: K.KEY_RIGHT,
    pygame.K_z: K.KEY_A, pygame.K_x: K.KEY_B, pygame.K_RETURN: K.KEY_START, pygame.K_BACKSPACE: K.KEY_SELECT,
    pygame.K_a: K.KEY_L, pygame.K_s: K.KEY_R,
}
HUMAN_HELP = "Keyboard → GBA:  arrows = D-pad · Z = A · X = B · Enter = Start · Backspace = Select · A / S = L / R"


class Quit(BaseException):
    """Raised in the game thread when the window closes (BaseException so `except Exception` doesn't swallow it)."""


@dataclass
class Scene:
    """One decision as shown in the panels."""
    phase: str                          # "rental" / "battle" / "forced_switch" / "swap"
    view: Any
    info: Any = None                    # RunInfo
    choice: Any = None                  # Choice (None while the agent is thinking)
    suggestion: bool = False            # human mode: what the agents would do
    estimates: Any = None               # rl.damage.battle_estimates(...) if available

    @property
    def details(self) -> Optional[dict]:
        return getattr(self.choice, "details", None) if self.choice is not None else None


class Button:
    def __init__(self, name: str, rect: pygame.Rect, compact: bool):
        self.name, self.rect, self.compact = name, rect, compact


class Viewer:
    deadline = 0.0

    def __init__(self, speed: int = 1, step: bool = False, human: bool = False, size: Tuple[int, int] = None,
                 theme: str = "dark"):
        pygame.init()
        pygame.display.set_caption("pybattlefactory — Battle Factory")
        if size is None:
            try:
                dw, dh = pygame.display.get_desktop_sizes()[0]
            except Exception:
                dw, dh = 1600, 900
            size = (max(1024, min(1600, int(dw * 0.9))), max(600, min(920, int(dh * 0.88))))
        self.windowed_size = size
        self.fullscreen = False
        self.screen = pygame.display.set_mode(size, pygame.RESIZABLE)
        self.fonts = ui.Fonts()
        self.theme = theme if theme in ui.THEMES else "dark"
        self.font_idx = FONT_SCALES.index(1.0)
        self.speed_idx = SPEEDS.index(speed) if speed in SPEEDS else 0
        self.paused = False
        self.step_mode = step
        self.human = human
        self.quit = False
        self.waiting = False                  # step mode: an action is shown and waits for Next
        # what the panels show (set from the game thread; each assignment replaces a whole object)
        self.scene: Optional[Scene] = None
        self.info = None                      # last RunInfo
        self.status = ""                      # e.g. "walking to the attendant..."
        self.run_no = 0
        self.log: List[Tuple[str, str]] = []  # (text, tone)
        self._scene_ver = 0
        self._log_ver = 0
        self.log_scroll = 0                   # lines scrolled up from the bottom (0 = follow)
        self._next = threading.Event()        # Next pressed
        self._wake = threading.Event()        # any state change: wakes the game thread early
        self._human_mask = 0
        self._frame: Optional[np.ndarray] = None
        self._frame_id = 0
        self._shown_key = None
        self._frame_surface = None
        self._last_capture = 0.0
        self._clock_t = 0.0                   # pacing: when the next frame is due
        self._clock_speed = None
        self._pressed: Optional[str] = None   # button under a held mouse press
        self._clock = pygame.time.Clock()
        self._cache: Dict[str, Tuple[Any, pygame.Surface]] = {}
        self._layout_key = None
        self.error: Optional[BaseException] = None
        self.relayout()

    # --- content (called from the game thread) --------------------------------------------------------------

    def set_scene(self, scene: Optional[Scene]):
        self.scene = scene
        if scene is not None and scene.info is not None:
            self.info = scene.info
        self._scene_ver += 1

    def set_status(self, text: str):
        self.status = text
        self._scene_ver += 1

    def set_run(self, n: int):
        self.run_no = n
        self._scene_ver += 1

    def add_log(self, line: str, tone: str = ""):
        self.log.append((line, tone))
        del self.log[:-500]
        self._log_ver += 1

    # --- state changes (buttons and shortcuts) --------------------------------------------------------------

    @property
    def speed(self) -> int:
        return SPEEDS[self.speed_idx]

    @property
    def u(self) -> float:
        """Body font size in pixels: follows the window size and the A−/A+ buttons."""
        w, h = self.screen.get_size()
        base = min(h / 46, w / 84) if w >= 1.2 * h else min(h / 60, w / 62)   # landscape / portrait
        return max(10.0, min(46.0, base * FONT_SCALES[self.font_idx]))

    def _changed(self):
        self._wake.set()

    def toggle_pause(self):
        self.paused = not self.paused
        self._changed()

    def faster(self):
        self.speed_idx = min(self.speed_idx + 1, len(SPEEDS) - 1)
        self._changed()

    def slower(self):
        self.speed_idx = max(self.speed_idx - 1, 0)
        self._changed()

    def toggle_step(self):
        self.step_mode = not self.step_mode
        self._changed()

    def press_next(self):
        self._next.set()
        self._changed()

    def toggle_human(self):
        self.human = not self.human
        if not self.human:
            self._human_mask = 0
        self._changed()

    def font_smaller(self):
        self.font_idx = max(0, self.font_idx - 1)
        self.relayout()

    def font_bigger(self):
        self.font_idx = min(len(FONT_SCALES) - 1, self.font_idx + 1)
        self.relayout()

    def toggle_theme(self):
        self.theme = "light" if self.theme == "dark" else "dark"
        self.relayout()

    def toggle_fullscreen(self):
        try:
            if not self.fullscreen:
                self.windowed_size = self.screen.get_size()
                self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                self.fullscreen = True
            else:
                self.screen = pygame.display.set_mode(self.windowed_size, pygame.RESIZABLE)
                if self.screen.get_size() != tuple(self.windowed_size):     # some SDL drivers need a 2nd call
                    self.screen = pygame.display.set_mode(self.windowed_size, pygame.RESIZABLE)
                self.fullscreen = False
        except pygame.error:
            self.screen = pygame.display.set_mode(self.windowed_size, pygame.RESIZABLE)
            self.fullscreen = False
        self.relayout()

    def request_quit(self):
        self.quit = True
        self._next.set()
        self._changed()

    ACTIONS = {"pause": "toggle_pause", "slower": "slower", "faster": "faster", "step": "toggle_step",
               "next": "press_next", "human": "toggle_human", "quit": "request_quit", "font_down": "font_smaller",
               "font_up": "font_bigger", "theme": "toggle_theme", "fullscreen": "toggle_fullscreen"}

    def click(self, name: str):
        getattr(self, self.ACTIONS[name])()

    def resize(self, w: int, h: int):
        if not self.fullscreen:
            w, h = max(640, w), max(400, h)
            self.screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)
            self.windowed_size = (w, h)
        self.relayout()

    # --- layout -----------------------------------------------------------------------------------------------

    def relayout(self):
        self._cache.clear()
        W, H = self.screen.get_size()
        u = self.u
        pad = max(6, round(u * 0.6))
        header_h = round(u * 2.9)
        bar_h = round(u * 2.9)
        top, bottom = header_h, H - bar_h
        avail_h = bottom - top - 2 * pad
        self.r_header = pygame.Rect(0, 0, W, header_h)
        self.r_bar = pygame.Rect(0, bottom, W, bar_h)
        if W >= 1.2 * H:                               # landscape: game + log left, panels right
            log_min = u * 7
            gw = min((W - 3 * pad) * 0.46, (avail_h - log_min - pad) * 1.5)
            gw, gh = self._game_size(gw)
            self.r_game = pygame.Rect(pad, top + pad, gw, gh)
            self.r_log = pygame.Rect(pad, self.r_game.bottom + pad, gw, bottom - pad - self.r_game.bottom - pad)
            self.r_right = pygame.Rect(self.r_game.right + pad, top + pad, W - self.r_game.right - 2 * pad, avail_h)
        else:                                          # portrait: game on top, panels, log at the bottom
            gw = min(W - 2 * pad, avail_h * 0.36 * 1.5)
            gw, gh = self._game_size(gw)
            self.r_game = pygame.Rect((W - gw) // 2, top + pad, gw, gh)
            rest = bottom - pad - self.r_game.bottom - pad
            log_h = max(round(u * 5), round(rest * 0.2))
            self.r_right = pygame.Rect(pad, self.r_game.bottom + pad, W - 2 * pad, rest - log_h - pad)
            self.r_log = pygame.Rect(pad, self.r_right.bottom + pad, W - 2 * pad, log_h)
        self.buttons = self._layout_buttons(W, u)
        self._layout_key = (W, H, self.font_idx, self.theme)

    @staticmethod
    def _game_size(gw: float) -> Tuple[int, int]:
        """Width/height for the 240x160 screen: snap to an integer scale when that loses little space."""
        k = gw / 240
        if k >= 1 and k - int(k) < 0.15:
            k = int(k)
        gw = max(120, int(240 * k))
        return gw, gw * 2 // 3

    BUTTONS = [["pause", "next", "step"], ["slower", "speed", "faster"], ["human"],
               ["font_down", "font_up", "theme", "fullscreen", "quit"]]

    def _layout_buttons(self, W: int, u: float) -> List[Button]:
        p = ui.Painter(self.screen, self.fonts, u, ui.THEMES[self.theme])
        bh = round(u * 2.0)
        pad, gap, group_gap = max(6, round(u * 0.6)), max(4, round(u * 0.35)), round(u * 1.1)
        font = p.strong
        icon = round(u * 1.05)

        def width(name, compact):
            if name == "speed":
                return font.size("max")[0] + round(u * 1.4)
            if compact or not self._label(name)[0]:
                return bh
            return icon + gap + max(font.size(t)[0] for t in self._label_variants(name)) + round(u * 1.4)

        for compact in (False, True):
            widths = [[width(n, compact) for n in g] for g in self.BUTTONS]
            total = sum(sum(g) + gap * (len(g) - 1) for g in widths) + group_gap * (len(widths) - 1) + 2 * pad
            if total <= W:
                break
        out = []
        y = self.r_bar.y + (self.r_bar.h - bh) // 2
        # the first three groups from the left, the last one from the right
        x = pad
        for g, ws in zip(self.BUTTONS[:-1], widths[:-1]):
            for n, w in zip(g, ws):
                out.append(Button(n, pygame.Rect(x, y, w, bh), compact))
                x += w + gap
            x += group_gap - gap
        x = W - pad - (sum(widths[-1]) + gap * (len(widths[-1]) - 1))
        for n, w in zip(self.BUTTONS[-1], widths[-1]):
            out.append(Button(n, pygame.Rect(x, y, w, bh), compact))
            x += w + gap
        return out

    def button(self, name: str) -> Button:
        return next(b for b in self.buttons if b.name == name)

    def _label_variants(self, name: str) -> List[str]:
        return {"pause": ["Pause", "Resume"], "step": ["Step: on", "Step: off"], "human": ["Human: on", "Human: off"],
                "theme": ["Light", "Dark"], "fullscreen": ["Fullscreen", "Window"]}.get(name, [self._label(name)[0]])

    def _label(self, name: str):
        """(text, enabled, active) of a button."""
        if name == "pause":
            return ("Resume" if self.paused else "Pause"), True, self.paused
        if name == "slower":
            return "", self.speed_idx > 0, False
        if name == "faster":
            return "", self.speed_idx < len(SPEEDS) - 1, False
        if name == "speed":
            return ("max" if self.speed == 0 else f"{self.speed}×"), False, False
        if name == "step":
            return ("Step: on" if self.step_mode else "Step: off"), True, self.step_mode
        if name == "next":
            return "Next", self.waiting and not self.human, self.waiting and not self.human
        if name == "human":
            return ("Human: on" if self.human else "Human: off"), True, self.human
        if name == "font_down":
            return "", self.font_idx > 0, False
        if name == "font_up":
            return "", self.font_idx < len(FONT_SCALES) - 1, False
        if name == "theme":
            return ("Light" if self.theme == "dark" else "Dark"), True, False
        if name == "fullscreen":
            return ("Window" if self.fullscreen else "Fullscreen"), True, self.fullscreen
        return "Quit", True, False

    # --- the UI loop (main thread) ------------------------------------------------------------------------------

    def run(self, game: Callable[[], None]):
        """Run `game` in a worker thread and the window here until the user quits."""

        def target():
            try:
                game()
            except Quit:
                pass
            except BaseException as e:        # shown in the window and re-raised by run()
                self.error = e
                self.add_log(f"ERROR: {type(e).__name__}: {e}", "bad")
                import traceback
                traceback.print_exc()

        worker = threading.Thread(target=target, name="game", daemon=True)
        worker.start()
        try:
            while not self.quit:               # if the game ends by itself the window stays until closed
                self.tick()
        except KeyboardInterrupt:
            pass
        self.request_quit()
        worker.join(timeout=2.0)
        if self.error is not None:
            raise self.error

    def tick(self):
        """One UI iteration: handle every pending event, redraw, wait for the next UI frame."""
        self.handle_events()
        if self.deadline and time.time() > self.deadline:
            self.request_quit()
        self.draw()
        self._clock.tick(UI_FPS)

    def handle_events(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                self.request_quit()
            elif ev.type == pygame.VIDEORESIZE:
                self.resize(ev.w, ev.h)
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                self._pressed = self._hit(ev.pos)
            elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                hit = self._hit(ev.pos)
                if hit is not None and hit == self._pressed:
                    self.click(hit)
                self._pressed = None
            elif ev.type == pygame.MOUSEWHEEL:
                if self.r_log.collidepoint(pygame.mouse.get_pos()):
                    self.log_scroll = max(0, self.log_scroll + 3 * ev.y)
            elif ev.type == pygame.KEYDOWN and not self.human:
                self._shortcut(ev)
        if self.screen.get_size() != self._layout_key[:2] or pygame.display.get_surface() is not self.screen:
            self.screen = pygame.display.get_surface()
            self.relayout()
        if self.human:
            pressed = pygame.key.get_pressed()
            self._human_mask = sum(bit for key, bit in HUMAN_KEYS.items() if pressed[key])

    def _hit(self, pos) -> Optional[str]:
        for b in self.buttons:
            if b.name != "speed" and b.rect.collidepoint(pos) and self._label(b.name)[1]:
                return b.name
        return None

    def _shortcut(self, ev):
        """Agent mode only (in human mode every key belongs to the GBA). Layout-independent via ev.unicode."""
        ch = ev.unicode
        if ev.key == pygame.K_SPACE:
            self.toggle_pause()
        elif ch == "+" or ev.key in (pygame.K_KP_PLUS, pygame.K_EQUALS):
            self.faster()
        elif ch == "-" or ev.key == pygame.K_KP_MINUS:
            self.slower()
        elif ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_n):
            self.press_next()
        elif ev.key == pygame.K_F11:
            self.toggle_fullscreen()

    # --- drawing (main thread) ---------------------------------------------------------------------------------

    def _painter(self, surf, u=None) -> ui.Painter:
        return ui.Painter(surf, self.fonts, u or self.u, ui.THEMES[self.theme])

    def _cached(self, name: str, key, render: Callable[[], pygame.Surface]) -> pygame.Surface:
        hit = self._cache.get(name)
        if hit is not None and hit[0] == key:
            return hit[1]
        surf = render()
        self._cache[name] = (key, surf)
        return surf

    def draw(self):
        s, t = self.screen, ui.THEMES[self.theme]
        s.fill(t["bg"])
        hdr_key = (self._scene_ver, self.speed_idx, self.paused, self.step_mode, self.human, self.waiting,
                   self._layout_key)
        s.blit(self._cached("header", hdr_key, self._render_header), self.r_header)
        s.blit(self._cached("right", (self._scene_ver, self.waiting, self._layout_key), self._render_right),
               self.r_right)
        s.blit(self._cached("log", (self._log_ver, self.log_scroll, self._layout_key), self._render_log), self.r_log)
        self._draw_game()
        self._draw_bar()
        pygame.display.flip()

    def _draw_game(self):
        s, t, r = self.screen, ui.THEMES[self.theme], self.r_game
        pygame.draw.rect(s, t["screen_bg"], r)
        key = (self._frame_id, r.size)
        if self._frame is not None and key != self._shown_key:
            self._shown_key = key
            raw = pygame.surfarray.make_surface(self._frame.swapaxes(0, 1))
            k = r.w / 240
            if abs(k - round(k)) < 1e-6:
                self._frame_surface = pygame.transform.scale(raw, r.size)
            else:                              # sharp scaling: nearest up to the next integer, smooth down
                n = int(k) + 1
                big = pygame.transform.scale(raw, (240 * n, 160 * n))
                self._frame_surface = pygame.transform.smoothscale(big, r.size)
        p = self._painter(s)
        if self._frame_surface is not None:
            s.blit(self._frame_surface, r)
        else:
            p.text(self.status or "starting…", r.centerx, r.centery, p.medium, (200, 200, 200), align="center",
                   valign="center", max_w=r.w - 20)
        pygame.draw.rect(s, t["border"], r.inflate(2, 2), 1)
        # overlays: paused badge, human-mode keyboard help
        if self.paused:
            w = p.chip_w("PAUSED", p.strong)
            p.chip("PAUSED", r.right - w - p.pad, r.y + p.pad, t["accent"], t["on_accent"], font=p.strong)
        if self.human:
            lines = p.wrap(HUMAN_HELP, p.small, r.w - 2 * p.pad)
            lh = p.small.get_linesize()
            hh = lh * len(lines) + p.gap * 2
            band = pygame.Surface((r.w, hh), pygame.SRCALPHA)
            band.fill((0, 0, 0, 170))
            s.blit(band, (r.x, r.bottom - hh))
            for i, line in enumerate(lines):
                p.text(line, r.x + p.pad, r.bottom - hh + p.gap + i * lh, p.small, (255, 255, 255))

    def _render_header(self) -> pygame.Surface:
        t = ui.THEMES[self.theme]
        surf = pygame.Surface(self.r_header.size)
        surf.fill(t["panel"])
        pygame.draw.line(surf, t["border"], (0, surf.get_height() - 1), (surf.get_width(), surf.get_height() - 1))
        p = self._painter(surf)
        W, H = surf.get_size()
        x, cy = p.pad * 2, H / 2
        # right: mode + speed chips
        chips = []
        if self.human:
            chips.append(("HUMAN", t["info"], (255, 255, 255)))
        elif self.step_mode:
            chips.append(("STEP · press Next" if self.waiting else "STEP", t["accent"], t["on_accent"]))
        else:
            chips.append(("AGENTS", t["good"], (15, 15, 15)))
        chips.append((("speed max" if self.speed == 0 else f"speed {self.speed}×"), t["card_hi"], t["text"]))
        if self.paused:
            chips.append(("PAUSED", t["bad"], (255, 255, 255)))
        rx = W - p.pad * 2
        for text, bg, fg in reversed(chips):
            w = p.chip_w(text, p.small_b)
            rx -= w
            p.chip(text, rx, cy - p.chip_h(p.small_b) / 2, bg, fg, font=p.small_b)
            rx -= p.gap
        # left: title + phase
        x += p.text("Battle Factory", x, cy, p.title, t["text"], valign="center") + p.pad * 2
        sc = self.scene
        phase = {"rental": "RENTAL", "battle": "BATTLE", "forced_switch": "FORCED SWITCH",
                 "swap": "SWAP"}.get(sc.phase if sc else "", "")
        if phase and x + p.chip_w(phase, p.small_b) < rx:
            x += p.chip(phase, x, cy - p.chip_h(p.small_b) / 2, t["accent"], t["on_accent"], font=p.small_b) + p.pad * 2
        # stats
        stats = []
        if self.info is not None:
            i = self.info
            rnd = i.challenge_num + 1
            stats += [("ROUND", f"{rnd}/6" if rnd <= 6 else f"{rnd}"),
                      ("BATTLE", f"{i.battle_in_challenge + 1}/7" + ("  NOLAND" if getattr(i, "noland", False) else "")),
                      ("STREAK", str(i.win_streak)), ("RUN WINS", str(i.wins)), ("RENTALS", str(i.rents))]
        if self.run_no:
            stats.append(("RUN", f"#{self.run_no}"))
        for label, value in stats:
            lw = p.tiny.size(label)[0]
            vw = p.strong.size(value)[0]
            w = max(lw, vw)
            if x + w > rx - p.pad:
                break
            p.text(label, x, cy - p.tiny.get_linesize() + 1, p.tiny, t["muted"])
            if value.endswith("NOLAND"):
                base = value[:-len("  NOLAND")]
                bw = p.text(base, x, cy, p.strong, t["text"])
                p.chip("NOLAND", x + bw + p.gap, cy + 1, t["bad"], (255, 255, 255))
            else:
                p.text(value, x, cy, p.strong, t["text"])
            x += w + p.pad * 3
        if self.status and x < rx - p.u * 4:
            p.text(self.status, x, cy, p.body, t["muted"], max_w=rx - x - p.pad, valign="center")
        return surf

    def _render_right(self) -> pygame.Surface:
        """The state panel (what the agent sees) above the decision panel; shrinks the text until it fits."""
        t = ui.THEMES[self.theme]
        W, H = self.r_right.size
        surf = pygame.Surface((W, H))
        sc = self.scene
        u0 = self.u
        for shrink in (1.0, 0.92, 0.85, 0.78, 0.72, 0.66, 0.6, 0.55):
            surf.fill(t["bg"])
            p = self._painter(surf, max(8.0, u0 * shrink))
            inner = W - 2 * p.pad
            if sc is None:
                p.rect(t["panel"], (0, 0, W, H))
                ui.draw_message(p, p.pad, p.pad, inner, [self.status or "waiting for the first decision…"])
                break
            # measure: state, then decision, drawn once each on a scratch pass
            hs = ui.draw_state(p, p.pad, p.pad, inner, sc) + p.pad
            surf.fill(t["bg"])
            y_dec = hs + p.gap
            hd = ui.draw_decision(p, p.pad, y_dec + p.pad, inner, sc, self.waiting) + p.pad
            if hd <= H or shrink == 0.55:
                surf.fill(t["bg"])
                if hs > p.pad * 2:
                    p.rect(t["panel"], (0, 0, W, hs))
                    ui.draw_state(p, p.pad, p.pad, inner, sc)
                else:
                    y_dec = 0
                p.rect(t["panel"], (0, y_dec, W, H - y_dec))
                ui.draw_decision(p, p.pad, y_dec + p.pad, inner, sc, self.waiting)
                break
        return surf

    def _render_log(self) -> pygame.Surface:
        t = ui.THEMES[self.theme]
        W, H = self.r_log.size
        surf = pygame.Surface((max(1, W), max(1, H)))
        surf.fill(t["panel"])
        p = self._painter(surf)
        right = "wheel: scroll" if not self.log_scroll else f"scrolled ↑ {self.log_scroll} · wheel down to follow"
        y0 = p.section("Decisions", p.pad, p.pad, W - 2 * p.pad, right, t["accent"] if self.log_scroll else None)
        tones = {"run": t["accent"], "won": t["good"], "bad": t["bad"], "lost": t["bad"], "human": t["info"],
                 "": t["text"], "muted": t["muted"]}
        font = p.small
        lh = font.get_linesize()
        lines = []
        for text, tone in list(self.log):
            for k, line in enumerate(p.wrap(text, font, W - 2 * p.pad - p.u)):
                lines.append(((" " * 3 if k else "") + line, tones.get(tone, t["text"])))
        room = max(0, (H - y0 - p.pad) // lh)
        self.log_scroll = min(self.log_scroll, max(0, len(lines) - room))
        end = len(lines) - self.log_scroll
        shown = lines[max(0, end - room):end]
        y = y0
        for line, col in shown:
            p.text(line, p.pad, y, font, col, max_w=W - 2 * p.pad)
            y += lh
        return surf

    def _draw_bar(self):
        s, t = self.screen, ui.THEMES[self.theme]
        pygame.draw.rect(s, t["panel"], self.r_bar)
        pygame.draw.line(s, t["border"], self.r_bar.topleft, self.r_bar.topright)
        p = self._painter(s)
        mouse = pygame.mouse.get_pos()
        for b in self.buttons:
            text, enabled, active = self._label(b.name)
            r = b.rect
            if b.name == "speed":
                p.rect(t["bg"], r)
                p.text(text, r.centerx, r.centery, p.strong, t["accent"], align="center", valign="center")
                continue
            if not enabled:
                bg, fg = t["panel"], t["faint"]
            else:
                bg = t["button_on"] if active else t["button"]
                if b.name == "quit":
                    bg = t["danger"]
                if r.collidepoint(mouse):
                    bg = t["button_hover"] if not active else tuple(min(255, c + 20) for c in bg)
                if self._pressed == b.name:
                    bg = t["button_down"]
                fg = t["text"]
            if b.name == "next" and enabled:
                bg, fg = t["accent"], t["on_accent"]
                if self._pressed == b.name:
                    bg = tuple(max(0, c - 40) for c in bg)
            p.rect(bg, r)
            p.rect(t["border"] if enabled else t["panel"], r, width=1)
            isz = round(p.u * 1.0)
            if b.compact or not text:
                _icon(s, b.name, self, pygame.Rect(r.centerx - isz // 2, r.centery - isz // 2, isz, isz), fg)
            else:
                ix = r.x + round(p.u * 0.7)
                _icon(s, b.name, self, pygame.Rect(ix, r.centery - isz // 2, isz, isz), fg)
                p.text(text, ix + isz + p.gap, r.centery, p.strong, fg, valign="center",
                       max_w=r.right - ix - isz - p.gap - 4)

    # --- called from the game thread ----------------------------------------------------------------------------

    def check_quit(self):
        if self.quit:
            raise Quit()

    def human_keys(self) -> int:
        return self._human_mask if self.human else 0

    def wait_enter(self):
        """Step mode: hold here until Next is clicked (or step mode / human mode is switched on/off)."""
        self._next.clear()
        self.waiting = True
        try:
            while self.step_mode and not self.human and not self._next.is_set():
                self.check_quit()
                self._next.wait(0.05)
            self.check_quit()
        finally:
            self.waiting = False

    def frame_done(self, emu):
        """Called after every emulated frame: quit check, pause, pacing, picture for the window."""
        self.check_quit()
        while self.paused:
            self._wake.wait(0.05)
            self._wake.clear()
            self.check_quit()
            self._clock_speed = None           # restart pacing after a pause
        now = time.perf_counter()
        speed = self.speed
        if speed == 0:
            self._clock_speed = None
            if now - self._last_capture >= 1 / 30:
                self._capture(emu, now)
            return
        if speed != self._clock_speed or now - self._clock_t > 0.25:   # speed change or long stall: resync
            self._clock_speed, self._clock_t = speed, now
        self._clock_t += 1 / (60 * speed)
        if now - self._last_capture >= 1 / UI_FPS:
            self._capture(emu, now)
        ahead = self._clock_t - now
        if ahead > 0.004:
            self._wake.clear()
            # interruptible: a click (speed, pause, quit) wakes us and the new state applies at once
            if self._wake.wait(ahead):
                self._clock_speed = None

    def _capture(self, emu, now):
        self._frame = np.array(emu.screenshot(), copy=True)
        self._frame_id += 1
        self._last_capture = now


def _icon(s: pygame.Surface, name: str, v: Viewer, r: pygame.Rect, c):
    """Small vector icons for the control buttons."""
    x, y, w, h = r
    lw = max(2, w // 8)
    if name == "pause":
        if v.paused:                                    # play triangle
            pygame.draw.polygon(s, c, [(x + w * 0.2, y), (x + w * 0.2, y + h), (x + w * 0.95, y + h / 2)])
        else:
            pygame.draw.rect(s, c, (x + w * 0.15, y, w * 0.25, h), border_radius=2)
            pygame.draw.rect(s, c, (x + w * 0.6, y, w * 0.25, h), border_radius=2)
    elif name == "next":
        pygame.draw.polygon(s, c, [(x, y), (x, y + h), (x + w * 0.65, y + h / 2)])
        pygame.draw.rect(s, c, (x + w * 0.72, y, w * 0.2, h), border_radius=1)
    elif name == "step":
        for k in range(3):                              # three dots on a line: one action at a time
            pygame.draw.circle(s, c, (x + w * (0.15 + 0.35 * k), y + h * 0.45), max(2, w // 8))
        pygame.draw.line(s, c, (x, y + h * 0.85), (x + w, y + h * 0.85), lw)
    elif name in ("slower", "faster", "font_down", "font_up"):
        if name.startswith("font"):
            f = v.fonts.get("bold", h * 0.95)
            img = f.render("A", True, c)
            s.blit(img, img.get_rect(center=(x + w * 0.4, y + h / 2)))
            sx, sy, sw = x + w * 0.72, y + h * 0.3, w * 0.28
        else:
            sx, sy, sw = x + w * 0.1, y + h / 2, w * 0.8
        pygame.draw.line(s, c, (sx, sy), (sx + sw, sy), lw)
        if name in ("faster", "font_up"):
            pygame.draw.line(s, c, (sx + sw / 2, sy - sw / 2), (sx + sw / 2, sy + sw / 2), lw)
    elif name == "human":                               # a little gamepad
        body = pygame.Rect(x, y + h * 0.25, w, h * 0.55)
        pygame.draw.rect(s, c, body, lw, border_radius=max(2, h // 4))
        cx, cy = x + w * 0.3, y + h * 0.52
        pygame.draw.line(s, c, (cx - w * 0.12, cy), (cx + w * 0.12, cy), lw)
        pygame.draw.line(s, c, (cx, cy - w * 0.12), (cx, cy + w * 0.12), lw)
        pygame.draw.circle(s, c, (x + w * 0.72, cy), max(2, w // 10))
    elif name == "theme":
        cx, cy, rad = x + w / 2, y + h / 2, w * 0.45
        pygame.draw.circle(s, c, (cx, cy), rad, lw)
        pygame.draw.circle(s, c, (cx, cy), rad, draw_top_left=True, draw_bottom_left=True)
    elif name == "fullscreen":
        k = w * 0.35
        for (px, py, dx, dy) in ((x, y, 1, 1), (x + w, y, -1, 1), (x, y + h, 1, -1), (x + w, y + h, -1, -1)):
            pygame.draw.line(s, c, (px, py), (px + dx * k, py), lw)
            pygame.draw.line(s, c, (px, py), (px, py + dy * k), lw)
    elif name == "quit":
        pygame.draw.line(s, c, (x + w * 0.1, y + h * 0.1), (x + w * 0.9, y + h * 0.9), lw + 1)
        pygame.draw.line(s, c, (x + w * 0.9, y + h * 0.1), (x + w * 0.1, y + h * 0.9), lw + 1)


class ViewedMachine:
    """The headless emulator, drawn into a Viewer frame by frame. Same interface as Emulator."""

    def __init__(self, rom: str, save: str, viewer: Viewer):
        self.emu = native.Emulator(rom, save)
        self.viewer = viewer

    def run_frames(self, n: int = 1):
        for _ in range(n):
            self.emu.run_frames(1)
            self.viewer.frame_done(self.emu)

    def __getattr__(self, name):          # read/write/set_keys/save_state/... pass through
        return getattr(self.emu, name)

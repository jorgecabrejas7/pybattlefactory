"""Drive Battle Factory in the real game through the headless emulator.

Every decision point is detected from RAM (callbacks, controller functions, task
functions) rather than from frame timing, so the driver is robust to text speed and
animation length. Menu cursors are read back from RAM before confirming.
"""

import enum
import struct
from dataclasses import dataclass
from typing import List, Optional

from . import native
from .decode import (SYMBOLS as S, SB2_FACTORY_RENTS_COUNT, SB2_FACTORY_WIN_STREAKS, SB2_FRONTIER_WIN_STREAK_ACTIVE,
                     SB2_RENTAL_MONS, BattleMon, PartyMon, decode_battle_mon, decode_party,
                     decode_pokemon, decode_rental_mons)

K = native

LAYOUT_LOBBY, LAYOUT_PRE_BATTLE_ROOM, LAYOUT_BATTLE_ROOM = 346, 347, 348

# RAM layout details (see pokeemerald/src)
_SELECT_MONS_OFFSET = 12          # FactorySelectScreen.mons
_SELECTABLE_MON_SIZE = 108        # FactorySelectableMon
_SELECT_CURSOR_POS = 3
_SELECT_MENU_CURSOR_POS = 0
_SELECT_MENU_RENT = 1
_SELECT_YES_NO_CURSOR_POS = 7
# FactorySwapScreen (GBA layout)
_SWAP_MENU_CURSOR_POS = 0
_SWAP_CURSOR_POS = 3
_SWAP_IN_ENEMY_SCREEN = 20
_SWAP_YES_NO_CURSOR_POS = 22
_SWAP_MENU_SWAP = 1
_FRONTIER_CHALLENGE_STATUS = 0xCA8
_CHALLENGE_STATUS_WON = 3
_PARTY_MENU_SLOT_ID = 9           # PartyMenu.slotId
_TASK_SIZE = 40
_OBJECT_EVENT_SIZE = 36
_MAP_OFFSET = 7
_MAPGRID_COLLISION_MASK = 0x0C00
_DIR_NORTH = 2
_SINGLES_ATTENDANT_SPOT = (4, 8)          # attendant stands at (4, 7), BattleFactoryLobby/map.json
_LOBBY_EXIT_WARPS = ((9, 11), (10, 11))
_STREAK_FACTORY_SINGLES = (1 << 8, 1 << 9)   # winStreakActiveFlags: [lv50, open]
_NUM_TASKS = 16


class Decision(enum.Enum):
    NONE = "none"
    RENTAL_SELECT = "rental_select"
    BATTLE_ACTION = "battle_action"
    PARTY_MENU = "party_menu"          # voluntary switch or forced replacement
    SWAP_SCREEN = "swap_screen"
    BATTLE_OVER = "battle_over"
    SWAP_QUESTION = "swap_question"      # pre-battle room: "Would you like to swap a Pokemon?"
    CHALLENGE_WON = "challenge_won"      # back in the lobby after 7 wins, streak still active
    LOBBY_IDLE = "lobby_idle"            # back in the lobby, free to walk (after a loss)


@dataclass
class RentalCandidate:
    frontier_mon_id: int
    mon: PartyMon


class FactoryDriver:
    def __init__(self, rom_path: str = "", save_path: str = "", machine=None):
        """Headless emulator for rom_path/save_path, or an existing `machine` with the same
        interface (e.g. pybattle.emu.lua_machine.LuaMachine for a live mgba-qt)."""
        self.emu = machine if machine is not None else native.Emulator(rom_path, save_path)
        self._func_names = {v: k for k, v in S.functions.items()}

    # -- low level ----------------------------------------------------------

    def fn_name(self, ptr: int) -> str:
        return self._func_names.get(ptr, hex(ptr))

    def callback2(self) -> str:
        return self.fn_name(self.emu.read32(S.addr("gMain") + 4))

    def player_controller(self) -> str:
        return self.fn_name(self.emu.read32(S.addr("gBattlerControllerFuncs")))

    def active_tasks(self) -> List[str]:
        base = S.addr("gTasks")
        return [self.fn_name(self.emu.read32(base + i * _TASK_SIZE))
                for i in range(_NUM_TASKS) if self.emu.read8(base + i * _TASK_SIZE + 4)]

    def layout(self) -> int:
        return self.emu.read16(S.addr("gMapHeader") + 0x12)

    def rng(self) -> int:
        return self.emu.read32(S.addr("gRngValue"))

    def saveblock2(self) -> int:
        return self.emu.read32(S.addr("gSaveBlock2Ptr"))

    def tap(self, keys: int, hold: int = 4, wait: int = 16):
        self.emu.set_keys(keys)
        self.emu.run_frames(hold)
        self.emu.set_keys(0)
        self.emu.run_frames(wait)

    # While any of these is current, a decision screen is loading: a key press could be
    # consumed by the menu task on the very frame it starts, so input is withheld.
    _QUIET_CALLBACKS = {"CB2_InitSelectScreen", "CB2_SelectScreen", "CB2_InitSwapScreen", "Swap_CB2",
                        "CB2_UpdatePartyMenu", "CB2_InitPartyMenu"}
    _QUIET_CONTROLLERS = {"HandleChooseActionAfterDma3", "HandleInputChooseAction", "HandleChooseMoveAfterDma3",
                          "HandleInputChooseMove", "OpenPartyMenuToChooseMon", "WaitForMonSelection"}

    def _quiet(self) -> bool:
        cb2 = self.callback2()
        if cb2 in self._QUIET_CALLBACKS:
            return True
        if self._in_pre_battle_yes_no():
            return True
        return cb2 == "BattleMainCB2" and self.player_controller() in self._QUIET_CONTROLLERS

    def mash_until(self, predicate, keys: int = K.KEY_A, max_frames: int = 20000) -> bool:
        """Alternate pressing `keys` (4 frames on / 4 off) until `predicate` holds. The predicate
        is checked every frame before input is applied, and input is withheld while a decision
        screen is loading, so a press never lands on the decision being waited for."""
        for f in range(max_frames):
            if predicate():
                self.emu.set_keys(0)
                return True
            press = keys and (f // 4) % 2 == 0 and not self._quiet()
            self.emu.set_keys(keys if press else 0)
            self.emu.run_frames(1)
        self.emu.set_keys(0)
        return False

    def _press_until(self, keys: int, predicate, attempts: int = 10, wait: int = 20):
        for _ in range(attempts):
            self.tap(keys, wait=wait)
            if predicate():
                return
        raise TimeoutError(f"input had no effect (cb2={self.callback2()}, tasks={self.active_tasks()})")

    def save_state(self) -> bytes:
        return self.emu.save_state()

    def load_state(self, state: bytes):
        self.emu.load_state(state)

    # -- decision detection -------------------------------------------------

    def decision(self) -> Decision:
        cb2 = self.callback2()
        if cb2 == "CB2_SelectScreen" and "Select_Task_HandleChooseMons" in self.active_tasks():
            return Decision.RENTAL_SELECT
        if cb2 == "Swap_CB2" and "Swap_Task_HandleChooseMons" in self.active_tasks():
            return Decision.SWAP_SCREEN
        if cb2 == "CB2_UpdatePartyMenu" and "Task_HandleChooseMonInput" in self.active_tasks():
            return Decision.PARTY_MENU
        if cb2 == "BattleMainCB2":
            if self.emu.read8(S.addr("gBattleOutcome")) != 0:
                return Decision.BATTLE_OVER
            if self.player_controller() == "HandleInputChooseAction":
                return Decision.BATTLE_ACTION
        return Decision.NONE

    def advance(self, max_frames: int = 60000) -> Decision:
        """Mash A through text/animations until the player must decide something, then let the
        rest of the battle settle (the opponent's AI picks its move while the player's menu is
        open), so the state at a decision is well defined. The simulator does the same."""
        if not self.mash_until(lambda: self.decision() != Decision.NONE, max_frames=max_frames):
            raise TimeoutError(f"no decision point reached (cb2={self.callback2()}, tasks={self.active_tasks()})")
        dec = self.decision()
        if dec in (Decision.BATTLE_ACTION, Decision.PARTY_MENU):
            self.settle()
        return dec

    def _progress(self):
        emu = self.emu
        return (self.rng(), emu.read32(S.addr("gBattleControllerExecFlags")),
                emu.read(S.addr("gBattleCommunication"), 8), emu.read32(S.addr("gBattleMainFunc")))

    def settle(self, quiet_frames: int = 30, max_frames: int = 2000):
        """Run without input until battle progress (RNG, controller flags, engine state) stops."""
        last, quiet = self._progress(), 0
        for _ in range(max_frames):
            self.emu.run_frames(1)
            cur = self._progress()
            quiet = quiet + 1 if cur == last else 0
            last = cur
            if quiet >= quiet_frames:
                return
        raise TimeoutError("battle state did not settle")

    def _in_pre_battle_yes_no(self) -> bool:
        return (self.callback2() == "CB2_Overworld" and self.layout() == LAYOUT_PRE_BATTLE_ROOM
                and "Task_HandleYesNoInput" in self.active_tasks())

    def advance_factory(self, max_frames: int = 120000) -> Decision:
        """Like advance(), across the whole Factory loop: battles, the pre-battle room (the
        Continue menu is confirmed, the swap question is returned as a decision) and the lobby."""
        in_battle_prev = False
        for _ in range(max_frames // 8):
            dec = self.decision()
            if dec == Decision.BATTLE_OVER:
                if not self._outcome_reported:
                    self._outcome_reported = True
                    return dec
                dec = Decision.NONE
            if dec in (Decision.BATTLE_ACTION, Decision.PARTY_MENU):
                self._outcome_reported = False
                self.settle()
                return dec
            if dec != Decision.NONE:
                return dec
            if self._in_pre_battle_yes_no():
                self.emu.run_frames(10)
                return Decision.SWAP_QUESTION
            if self._leftover_menu():
                self.tap(K.KEY_B, wait=20)          # a sub-menu left open (e.g. by a human): close it
                continue
            if self.layout() == LAYOUT_LOBBY and self.callback2() == "CB2_Overworld" and self._lobby_idle():
                won = self.emu.read8(self.saveblock2() + _FRONTIER_CHALLENGE_STATUS) == _CHALLENGE_STATUS_WON
                return Decision.CHALLENGE_WON if won else Decision.LOBBY_IDLE
            self.mash_until(lambda: False, max_frames=8)
        raise TimeoutError(f"factory flow stuck (cb2={self.callback2()}, layout={self.layout()}, tasks={self.active_tasks()})")

    _outcome_reported = False

    _SUB_MENU_TASKS = {"Select_Task_HandleMenu", "Select_Task_HandleYesNo", "Swap_Task_HandleMenu",
                       "Swap_Task_HandleYesNo"}

    def _leftover_menu(self) -> bool:
        return self.callback2() in ("CB2_SelectScreen", "Swap_CB2") and bool(self._SUB_MENU_TASKS & set(self.active_tasks()))

    def _lobby_idle(self) -> bool:
        """No script is running (sGlobalScriptContextStatus == CONTEXT_SHUTDOWN): the player can walk."""
        return (self.emu.read8(S.addr("sGlobalScriptContextStatus")) == 2
                and "Task_HandleMultichoiceInput" not in self.active_tasks())

    def hints(self):
        """(most common type, battle style) of the next opponent, as the attendant announced
        them (scripts keep them in VAR_0x8005 / VAR_0x8006)."""
        return (self.emu.read16(S.addr("gSpecialVar_0x8005")), self.emu.read16(S.addr("gSpecialVar_0x8006")))

    def answer_swap_question(self, yes: bool):
        assert self._in_pre_battle_yes_no()
        if yes:
            self._press_until(K.KEY_A, lambda: not self._in_pre_battle_yes_no())
            if not self.mash_until(lambda: self.decision() == Decision.SWAP_SCREEN, keys=0, max_frames=3000):
                raise TimeoutError("swap screen did not open")
        else:
            self._press_until(K.KEY_B, lambda: not self._in_pre_battle_yes_no())

    def _swap_screen(self) -> int:
        return self.emu.read32(S.addr("sFactorySwapScreen"))

    def swap(self, own_slot: int, enemy_slot: int):
        """On the swap screen: trade our `own_slot` for the defeated team's `enemy_slot`."""
        scr = self._swap_screen
        if "Swap_Task_HandleChooseMons" not in self.active_tasks():
            self._swap_normalize()            # e.g. a human left a menu open before handing over
        choose = lambda: "Swap_Task_HandleChooseMons" in self.active_tasks()
        self.mash_until(choose, keys=0, max_frames=600)
        self.emu.run_frames(10)
        while self.emu.read8(scr() + _SWAP_CURSOR_POS) != own_slot:
            self.tap(K.KEY_RIGHT)
        self._press_until(K.KEY_A, lambda: "Swap_Task_HandleMenu" in self.active_tasks())
        self.mash_until(lambda: False, keys=0, max_frames=30)
        while self.emu.read8(scr() + _SWAP_MENU_CURSOR_POS) != _SWAP_MENU_SWAP:
            self.tap(K.KEY_DOWN)
        self._press_until(K.KEY_A, lambda: self.emu.read8(scr() + _SWAP_IN_ENEMY_SCREEN) == 1)
        self.mash_until(choose, keys=0, max_frames=600)
        self.emu.run_frames(10)
        while self.emu.read8(scr() + _SWAP_CURSOR_POS) != enemy_slot:
            self.tap(K.KEY_RIGHT)
        self._press_until(K.KEY_A, lambda: "Swap_Task_HandleYesNo" in self.active_tasks())
        self.mash_until(lambda: False, keys=0, max_frames=30)
        assert self.emu.read8(scr() + _SWAP_YES_NO_CURSOR_POS) == 0
        self._press_until(K.KEY_A, lambda: "Swap_Task_HandleYesNo" not in self.active_tasks())

    def _swap_normalize(self):
        """From anywhere inside the swap screen (a submenu, a yes/no, the defeated team's page), back to
        choosing among our own Pokemon. B closes submenus and answers NO, so it never commits a swap."""
        scr = self._swap_screen
        for _ in range(20):
            tasks = set(self.active_tasks())
            if "Swap_Task_HandleChooseMons" in tasks and self.emu.read8(scr() + _SWAP_IN_ENEMY_SCREEN) == 0:
                self.emu.run_frames(10)
                return
            if tasks & {"Swap_Task_HandleYesNo", "Swap_Task_HandleMenu"} or \
               self.emu.read8(scr() + _SWAP_IN_ENEMY_SCREEN) == 1:
                self.tap(K.KEY_B)
            self.mash_until(lambda: False, keys=0, max_frames=20)
        raise TimeoutError("could not get back to the swap screen's own-team page")

    def quit_swap_screen(self):
        """On the swap screen: leave it keeping the current team (B, then YES to "Quit swapping?")."""
        self._swap_normalize()
        self._press_until(K.KEY_B, lambda: "Swap_Task_HandleYesNo" in self.active_tasks())
        self.mash_until(lambda: False, keys=0, max_frames=30)
        scr = self._swap_screen
        while self.emu.read8(scr() + _SWAP_YES_NO_CURSOR_POS) != 0:
            self.tap(K.KEY_UP)
        self._press_until(K.KEY_A, lambda: self.decision() != Decision.SWAP_SCREEN)

    def forfeit(self):
        """At the action prompt: RUN, then YES to "forfeit the match?"."""
        assert self.decision() == Decision.BATTLE_ACTION
        self.emu.write(S.addr("gActionSelectionCursor"), bytes([3]))
        self.tap(K.KEY_A, wait=2)
        # yes/no box in battle: YES is the default
        self.mash_until(lambda: self.player_controller() == "PlayerHandleYesNoInput", keys=0, max_frames=600)
        self.tap(K.KEY_A, wait=2)

    def finish_lost_challenge(self):
        """After a lost battle: let the game return to the lobby."""
        self.advance_factory()

    # -- boot / lobby -------------------------------------------------------

    def boot_to_overworld(self):
        """Title screen -> CONTINUE -> overworld."""
        self.emu.run_frames(600)
        self.mash_until(lambda: self.callback2() == "CB2_Overworld", keys=K.KEY_START | K.KEY_A,
                        max_frames=5000)
        # Let on-frame map scripts finish (e.g. the lobby's "quit without saving" handler, which
        # resets the Factory streak) before anything is written to the save block.
        self.emu.run_frames(120)
        self.mash_until(lambda: False, keys=K.KEY_B, max_frames=600)

    def set_factory_streak(self, streak: int, rents: int = 0, open_level: bool = True):
        """Pretend an active singles win streak (the challenge keeps it instead of resetting).
        Must be called in the lobby, before start_challenge()."""
        sb2 = self.saveblock2()
        lvl = 1 if open_level else 0
        flags = self.emu.read32(sb2 + SB2_FRONTIER_WIN_STREAK_ACTIVE)
        self.emu.write32(sb2 + SB2_FRONTIER_WIN_STREAK_ACTIVE, flags | _STREAK_FACTORY_SINGLES[lvl])
        self.emu.write(sb2 + SB2_FACTORY_WIN_STREAKS + lvl * 2, struct.pack("<H", streak))
        self.emu.write(sb2 + SB2_FACTORY_RENTS_COUNT + lvl * 2, struct.pack("<H", rents))

    # -- walking ----------------------------------------------------------------

    _DIR_KEYS = {(0, 1): (K.KEY_DOWN, 1), (0, -1): (K.KEY_UP, 2), (-1, 0): (K.KEY_LEFT, 3), (1, 0): (K.KEY_RIGHT, 4)}

    def _player_object(self) -> int:
        obj_id = self.emu.read8(S.addr("gPlayerAvatar") + 5)          # PlayerAvatar.objectEventId
        return S.addr("gObjectEvents") + obj_id * _OBJECT_EVENT_SIZE

    def player_pos(self):
        """Player map coordinates (x, y) and facing direction (DIR_*)."""
        obj = self._player_object()
        x, y = struct.unpack("<hh", self.emu.read(obj + 0x10, 4))
        return x - _MAP_OFFSET, y - _MAP_OFFSET, self.emu.read8(obj + 0x18) & 0xF

    def _blocked_tiles(self):
        width, height, grid = struct.unpack("<iiI", self.emu.read(S.addr("gBackupMapLayout"), 12))
        cells = struct.unpack(f"<{width * height}H", self.emu.read(grid, 2 * width * height))
        blocked = set()
        for gy in range(height):
            for gx in range(width):
                if cells[gy * width + gx] & _MAPGRID_COLLISION_MASK:
                    blocked.add((gx - _MAP_OFFSET, gy - _MAP_OFFSET))
        me = self._player_object()
        for i in range(16):                                            # other people
            obj = S.addr("gObjectEvents") + i * _OBJECT_EVENT_SIZE
            if obj != me and self.emu.read8(obj) & 1:
                x, y = struct.unpack("<hh", self.emu.read(obj + 0x10, 4))
                blocked.add((x - _MAP_OFFSET, y - _MAP_OFFSET))
        return blocked, (width - 2 * _MAP_OFFSET, height - 2 * _MAP_OFFSET)

    def walk_to(self, tx: int, ty: int, facing: Optional[int] = None, avoid=(), max_steps: int = 80):
        """Walk to map tile (tx, ty) along a shortest free path, then face `facing` (DIR_*)."""
        from collections import deque
        for _ in range(max_steps):
            x, y, _ = self.player_pos()
            if (x, y) == (tx, ty):
                break
            blocked, (w, h) = self._blocked_tiles()
            blocked |= set(avoid)
            prev, q = {(x, y): None}, deque([(x, y)])
            while q:
                cur = q.popleft()
                if cur == (tx, ty):
                    break
                for d in self._DIR_KEYS:
                    nxt = (cur[0] + d[0], cur[1] + d[1])
                    if 0 <= nxt[0] < w and 0 <= nxt[1] < h and nxt not in prev and (nxt not in blocked or nxt == (tx, ty)):
                        prev[nxt] = cur
                        q.append(nxt)
            if (tx, ty) not in prev:
                raise RuntimeError(f"no path from {(x, y)} to {(tx, ty)}")
            step = (tx, ty)
            while prev[step] != (x, y):
                step = prev[step]
            key, _ = self._DIR_KEYS[(step[0] - x, step[1] - y)]
            self._walk_one(key, (x, y))
        else:
            raise RuntimeError(f"could not reach {(tx, ty)}")
        if facing is not None:
            key = next(k for k, d in self._DIR_KEYS.values() if d == facing)
            for _ in range(10):
                if self.player_pos()[2] == facing:
                    break
                self.tap(key, hold=2, wait=10)                            # a short tap turns in place

    def _walk_one(self, key: int, start):
        self.emu.set_keys(key)
        for _ in range(40):
            self.emu.run_frames(1)
            if self.player_pos()[:2] != start:
                break
        self.emu.set_keys(0)
        self.emu.run_frames(12)                                           # finish the step

    def go_to_singles_attendant(self):
        """Stand in front of the Battle Factory singles attendant (lobby), facing her."""
        assert self.layout() == LAYOUT_LOBBY, "not in the Battle Factory lobby"
        self.walk_to(*_SINGLES_ATTENDANT_SPOT, facing=_DIR_NORTH, avoid=_LOBBY_EXIT_WARPS)

    def start_challenge(self, open_level: bool = True):
        """In the lobby: walk to the singles attendant, take the challenge, pick the level mode,
        save, and walk to the rental screen."""
        assert self.layout() == LAYOUT_LOBBY
        self.go_to_singles_attendant()
        # Talk until the CHALLENGE/INFO/EXIT menu, choose CHALLENGE; then LV.50 / OPEN LEVEL / EXIT.
        # Other questions on the way (e.g. "record your last battle?" after a won challenge) are
        # yes/no menus: answer NO.
        wanted = [0, 1 if open_level else 0]
        while wanted:
            self._mash_to_multichoice()
            if self.emu.read8(S.addr("sMenu@menu.o") + 4) == 1:          # maxCursorPos 1: YES/NO
                self._choose_multichoice(1)
                continue
            self._choose_multichoice(wanted.pop(0))
        # Save prompt (YES is default), text, walk into the pre-battle room, rental screen
        if not self.mash_until(lambda: self.decision() == Decision.RENTAL_SELECT, max_frames=30000):
            raise TimeoutError("rental screen not reached")

    def _choose_multichoice(self, index: int):
        """Move the open multichoice menu's cursor (sMenu.cursorPos, read back from RAM) to
        `index` and confirm."""
        cursor = lambda: self.emu.read8(S.addr("sMenu@menu.o") + 2)
        for _ in range(20):
            if cursor() == index:
                break
            self.tap(K.KEY_DOWN if cursor() < index else K.KEY_UP, wait=8)
        if cursor() != index:
            raise RuntimeError(f"menu cursor stuck at {cursor()}, wanted {index}")
        self._press_until(K.KEY_A, lambda: "Task_HandleMultichoiceInput" not in self.active_tasks())

    def _mash_to_multichoice(self):
        if not self.mash_until(lambda: "Task_HandleMultichoiceInput" in self.active_tasks(), max_frames=5000):
            raise TimeoutError("multichoice menu not reached")
        self.emu.run_frames(10)

    # -- rental select ------------------------------------------------------

    def _select_screen(self) -> int:
        return self.emu.read32(S.addr("sFactorySelectScreen"))

    def rental_candidates(self) -> List[RentalCandidate]:
        base = self._select_screen() + _SELECT_MONS_OFFSET
        out = []
        for i in range(6):
            addr = base + i * _SELECTABLE_MON_SIZE
            out.append(RentalCandidate(self.emu.read16(addr), decode_pokemon(self.emu.read(addr + 8, 100))))
        return out

    def pick_rentals(self, picks: List[int]):
        """Rent the mons at select-screen positions `picks` (in party order) and confirm."""
        assert len(picks) == 3 and len(set(picks)) == 3
        for n, pick in enumerate(picks):
            assert self.decision() == Decision.RENTAL_SELECT, (self.callback2(), self.active_tasks())
            scr = self._select_screen()
            selected = lambda: self.emu.read8(scr + _SELECT_MONS_OFFSET + pick * _SELECTABLE_MON_SIZE + 4)
            while self.emu.read8(scr + _SELECT_CURSOR_POS) != pick:
                self.tap(K.KEY_RIGHT)
            # Menu input is ignored while it animates open, so confirm each step from RAM and retry.
            self._press_until(K.KEY_A, lambda: "Select_Task_HandleMenu" in self.active_tasks())
            self.emu.run_frames(20)   # menu init resets menuCursorPos a few frames after opening
            while self.emu.read8(scr + _SELECT_MENU_CURSOR_POS) != _SELECT_MENU_RENT:
                self.tap(K.KEY_DOWN)
            self._press_until(K.KEY_A, lambda: "Select_Task_HandleMenu" not in self.active_tasks())
            want = "Select_Task_HandleYesNo" if n == 2 else "Select_Task_HandleChooseMons"
            if not self.mash_until(lambda: want in self.active_tasks(), keys=0, max_frames=600):
                raise TimeoutError(f"select screen stuck after pick {n}: {self.active_tasks()}")
            if selected() != n + 1:
                raise RuntimeError(f"pick {n} (slot {pick}) not registered")
        # "Is this team OK?" — YES is the default
        self.emu.run_frames(20)
        assert self.emu.read8(self._select_screen() + _SELECT_YES_NO_CURSOR_POS) == 0  # YES
        self._press_until(K.KEY_A, lambda: "Select_Task_HandleYesNo" not in self.active_tasks())

    # -- battle -------------------------------------------------------------

    def battle_mons(self) -> List[BattleMon]:
        base = S.addr("gBattleMons")
        return [decode_battle_mon(self.emu.read(base + b * 0x58, 0x58)) for b in range(2)]

    def player_party_raw(self) -> bytes:
        """gPlayerParty in field order. While the in-battle party menu is open the game keeps the
        party in display order (UpdatePartyToBattleOrder); undo that so indices are stable."""
        raw = self.emu.read(S.addr("gPlayerParty"), 600)
        if self.callback2() != "CB2_UpdatePartyMenu":
            return raw
        order = self._battle_order()
        field = bytearray(600)
        for slot in range(6):
            field[order[slot] * 100:(order[slot] + 1) * 100] = raw[slot * 100:(slot + 1) * 100]
        return bytes(field)

    def player_party(self) -> List[PartyMon]:
        return decode_party(self.player_party_raw())[:3]

    def _battle_order(self) -> List[int]:
        """gBattlePartyCurrentOrder unpacked: display slot -> party index."""
        order = []
        for byte in self.emu.read(S.addr("gBattlePartyCurrentOrder"), 3):
            order += [byte >> 4, byte & 0xF]
        return order

    def enemy_party(self) -> List[PartyMon]:
        return decode_party(self.emu.read(S.addr("gEnemyParty"), 600))[:3]

    def battler_party_indexes(self) -> List[int]:
        return list(struct.unpack("<2H", self.emu.read(S.addr("gBattlerPartyIndexes"), 4)))

    def battle_outcome(self) -> int:
        return self.emu.read8(S.addr("gBattleOutcome"))

    def choose_move(self, slot: int) -> bool:
        """At the action prompt: FIGHT, then the move in `slot` (0-3). Returns False when the game
        picked the move itself (Encore, no PP left -> Struggle): FIGHT then skips the move menu."""
        assert self.decision() == Decision.BATTLE_ACTION
        self.emu.write(S.addr("gActionSelectionCursor"), bytes([0]))
        self.tap(K.KEY_A, wait=2)
        action_phase = {"HandleInputChooseAction", "HandleChooseActionAfterDma3", "HandleChooseMoveAfterDma3"}
        if not self.mash_until(lambda: self.player_controller() not in action_phase, keys=0, max_frames=240):
            raise TimeoutError("FIGHT was not accepted")
        if self.player_controller() != "HandleInputChooseMove":
            return False
        self.emu.write(S.addr("gMoveSelectionCursor"), bytes([slot]))
        self.tap(K.KEY_A, wait=2)
        return True

    def choose_switch(self, party_index: int):
        """At the action prompt: POKéMON, then send out `party_index`."""
        assert self.decision() == Decision.BATTLE_ACTION
        self.emu.write(S.addr("gActionSelectionCursor"), bytes([2]))
        self.tap(K.KEY_A, wait=2)
        if not self.mash_until(lambda: self.decision() == Decision.PARTY_MENU, keys=0, max_frames=600):
            raise TimeoutError("party menu did not open")
        self.choose_party_slot(party_index)

    def choose_party_slot(self, party_index: int):
        """In the battle party menu (voluntary or forced): pick `party_index` and SHIFT/SEND OUT."""
        assert self.decision() == Decision.PARTY_MENU
        slot = self._display_slot(party_index)
        menu = S.addr("gPartyMenu")
        for _ in range(12):
            if self.emu.read8(menu + _PARTY_MENU_SLOT_ID) == slot:
                break
            self.tap(K.KEY_DOWN, wait=6)
        else:
            raise RuntimeError(f"could not move party cursor to slot {slot}")
        self.tap(K.KEY_A, wait=20)       # opens SHIFT / SUMMARY / CANCEL
        self.tap(K.KEY_A, wait=20)       # SHIFT (first option)

    def _display_slot(self, party_index: int) -> int:
        return self._battle_order().index(party_index)

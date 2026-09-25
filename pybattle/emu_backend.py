"""FactoryBackend on the real game (mGBA, headless) — same views, same actions as SimBackend.

Views are built by the same code (pybattle.view) from the same RAM addresses, so a trained
agent sees identical observations in both. The emulator walks through the game's menus
(pybattle.emu.driver) to perform each action.
"""

import struct
from typing import Optional

from .backend import NO_HINT, Action, FactoryBackend, Phase, factory_brain_status, rents_offset
from .emu.decode import SYMBOLS as S, decode_party
from .emu.driver import LAYOUT_LOBBY, LAYOUT_PRE_BATTLE_ROOM, Decision, FactoryDriver, K
from .view import MOVES, BattleObserver, OwnMon, RentalView, RunInfo, SwapView

HOLD_EFFECT_CHOICE_BAND = 29
STATUS2_TORMENT = 1 << 31
STATUS3_IMPRISONED_OTHERS = 1 << 13
MOVE_UNAVAILABLE = 0xFFFF
_BATTLE_STRUCT_CHOICED_MOVE = 200      # offsetof(struct BattleStruct, choicedMove) on the GBA

_ITEMS = None


def _hold_effect(item: int) -> int:
    global _ITEMS
    if _ITEMS is None:
        import json, os
        _ITEMS = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "game_data.json")))["items"]
    return _ITEMS[item]["hold_effect"] if item < len(_ITEMS) else 0


def unusable_moves(ram, battler: int = 0) -> int:
    """CheckMoveLimitations(battler, 0, MOVE_LIMITATIONS_ALL) computed from RAM."""
    mon = ram.read(S.addr("gBattleMons") + battler * 0x58, 0x58)
    moves = struct.unpack_from("<4H", mon, 0x0C)
    pp = mon[0x24:0x28]
    item = struct.unpack_from("<H", mon, 0x2E)[0]
    status2 = struct.unpack_from("<I", mon, 0x50)[0]
    dis = ram.read(S.addr("gDisableStructs") + battler * 0x1C, 0x1C)
    disabled_move, encored_move = struct.unpack_from("<HH", dis, 4)
    encore_timer = dis[0x0E] & 0xF
    taunt_timer = dis[0x13] & 0xF
    last_move = struct.unpack("<H", ram.read(S.addr("gLastMoves") + battler * 2, 2))[0]
    battlers = ram.read(S.addr("gBattlersCount"), 1)[0]
    status3 = struct.unpack(f"<{battlers}I", ram.read(S.addr("gStatuses3"), 4 * battlers))
    battle_struct = struct.unpack("<I", ram.read(S.addr("gBattleStruct"), 4))[0]
    choiced = struct.unpack("<H", ram.read(battle_struct + _BATTLE_STRUCT_CHOICED_MOVE + battler * 2, 2))[0]
    hold = _hold_effect(item)          # (Enigma Berry never appears in the Factory)

    def imprisoned(move):
        side = battler & 1
        for b in range(battlers):
            if (b & 1) != side and status3[b] & STATUS3_IMPRISONED_OTHERS:
                other = struct.unpack("<4H", ram.read(S.addr("gBattleMons") + b * 0x58 + 0x0C, 8))
                if move in other:
                    return True
        return False

    out = 0
    for i, mv in enumerate(moves):
        if (mv == 0 or pp[i] == 0 or mv == disabled_move
                or (mv == last_move and status2 & STATUS2_TORMENT)
                or (taunt_timer and MOVES[mv]["power"] == 0)
                or (mv and imprisoned(mv))
                or (encore_timer and encored_move != mv)
                or (hold == HOLD_EFFECT_CHOICE_BAND and choiced not in (0, MOVE_UNAVAILABLE) and choiced != mv)):
            out |= 1 << i
    return out


ABILITY_LEVITATE, ABILITY_SHADOW_TAG, ABILITY_ARENA_TRAP, ABILITY_MAGNET_PULL = 26, 23, 71, 42
TYPE_FLYING, TYPE_STEEL = 2, 8
STATUS2_WRAPPED = 7 << 13
STATUS2_ESCAPE_PREVENTION = 1 << 26
STATUS3_ROOTED = 1 << 10
BATTLE_TYPE_ARENA = 1 << 18


def can_switch(ram, battler: int = 0) -> bool:
    """Gen3_CanSwitch (src/gen3/host.c) computed from RAM: not trapped."""
    battlers = ram.read(S.addr("gBattlersCount"), 1)[0]
    mons = [ram.read(S.addr("gBattleMons") + b * 0x58, 0x58) for b in range(battlers)]
    status2 = struct.unpack_from("<I", mons[battler], 0x50)[0]
    status3 = struct.unpack("<I", ram.read(S.addr("gStatuses3") + battler * 4, 4))[0]
    type_flags = struct.unpack("<I", ram.read(S.addr("gBattleTypeFlags"), 4))[0]
    if status2 & (STATUS2_WRAPPED | STATUS2_ESCAPE_PREVENTION) or type_flags & BATTLE_TYPE_ARENA \
            or status3 & STATUS3_ROOTED:
        return False
    types = mons[battler][0x21:0x23]
    grounded = TYPE_FLYING not in types and mons[battler][0x20] != ABILITY_LEVITATE
    for i, m in enumerate(mons):
        ability = m[0x20]
        if (i & 1) != (battler & 1) and (ability == ABILITY_SHADOW_TAG or (ability == ABILITY_ARENA_TRAP and grounded)):
            return False
        if i != battler and ability == ABILITY_MAGNET_PULL and TYPE_STEEL in types:
            return False
    return True


class EmuBackend(FactoryBackend):
    def __init__(self, rom: str = "", save: str = "", open_level: bool = True, machine=None, boot=None):
        """Headless (rom/save), or on an existing `machine` (the viewer's ViewedMachine, or a
        LuaMachine for mgba-qt). `boot` (default: only for a fresh headless emulator) loads the
        save from the title screen; otherwise the game must already be in the Factory lobby."""
        self.d = FactoryDriver(rom, save, machine=machine)
        if boot if boot is not None else machine is None:
            self.d.boot_to_overworld()
        self._lobby = self.d.save_state()
        self.open_level = open_level
        self.phase = Phase.RUN_OVER
        self._observer = BattleObserver()
        self.last_battle_won: Optional[bool] = None
        self.wins = 0
        self.start_streak = 0

    # --- lifecycle --------------------------------------------------------------------

    def reset(self, seed: int = 0, win_streak: Optional[int] = 0, rents_count: int = 0, rewind: bool = True) -> None:
        """Start a new run.

        rewind=True: go back to the lobby as it was when the backend started (reproducible runs;
        `win_streak`/`rents_count` are written into the save). rewind=False: carry on from where the
        game is now, like a player would (after a loss the player is back in the lobby with the
        streak reset by the game); `win_streak=None` keeps the game's own streak.
        """
        d = self.d
        if rewind:
            d.load_state(self._lobby)
            d.emu.run_frames(1 + seed % 997)           # the overworld RNG advances every frame
        else:
            d.advance_factory()                         # finish any lobby dialogue first
        if win_streak is not None:
            d.set_factory_streak(win_streak, rents_count, self.open_level)
        sb2 = d.saveblock2()
        self.start_streak = d.emu.read16(sb2 + 0xDE2 + 2 * (1 if self.open_level else 0)) if win_streak is None else win_streak
        self.wins = 0
        d.start_challenge(self.open_level)
        self._observer = BattleObserver()
        self._sync()

    def _sync(self):
        """Advance through text and menus to the next decision."""
        d = self.d
        while True:
            dec = d.advance_factory()
            if dec == Decision.RENTAL_SELECT:
                self.phase = Phase.RENTAL
                return
            if dec == Decision.BATTLE_ACTION:
                self.phase = Phase.BATTLE
                return
            if dec == Decision.PARTY_MENU:
                self.phase = Phase.FORCED_SWITCH
                return
            if dec == Decision.SWAP_QUESTION:
                self.phase = Phase.SWAP
                self._swap_screen_open = False
                return
            if dec == Decision.SWAP_SCREEN:
                # the swap screen is already open (a human said YES, then handed over to the agents)
                self.phase = Phase.SWAP
                self._swap_screen_open = True
                return
            if dec == Decision.BATTLE_OVER:
                self._observer.finish(_FieldOrderRam(d))     # the last turn, at the moment the battle is decided
                won = d.battle_outcome() == 1
                self.last_battle_won = won
                self.wins += won
                if not won:
                    d.finish_lost_challenge()
                    self.phase = Phase.RUN_OVER
                    return
                continue
            if dec == Decision.LOBBY_IDLE:
                # free to walk in the lobby with no challenge running (e.g. a human handed over there)
                self.phase = Phase.RUN_OVER
                return
            if dec == Decision.CHALLENGE_WON:
                d.start_challenge(self.open_level)       # the streak goes on
                self._observer = BattleObserver()
                continue
            raise RuntimeError(f"unexpected {dec}")

    # --- observations -------------------------------------------------------------------

    def view(self):
        d = self.d
        if self.phase == Phase.RENTAL:
            cands = d.rental_candidates()
            return RentalView([OwnMon.from_party(c.mon) for c in cands], [c.frontier_mon_id for c in cands],
                              *self._hints())
        if self.phase == Phase.SWAP:
            own = [OwnMon.from_party(m) for m in decode_party(d.emu.read(S.addr("gPlayerParty"), 300))[:3]]
            return SwapView(own, self._observer.swap_candidates(d.emu), *self._hints(), self._observer.foe_records())
        if self.phase in (Phase.BATTLE, Phase.FORCED_SWITCH):
            ram = _FieldOrderRam(d)
            return self._observer.observe(ram, self.phase == Phase.FORCED_SWITCH, unusable_moves(d.emu), can_switch(d.emu))
        return None

    def _noland_next(self) -> bool:
        sb2 = self.d.saveblock2()
        streak = self.d.emu.read16(sb2 + 0xDE2 + 2 * (1 if self.open_level else 0))
        return factory_brain_status(streak, self.d.factory_symbols()) != 0

    def _hints(self):
        """The attendant's hints about the next opponent. Before Noland's battle the game generates no opponent
        and says nothing (AskSwapBeforeHead): the variables hold stale values, so the canonical "no hint" is used,
        as in the simulator (src/gen3/factory_run.c)."""
        return NO_HINT if self._noland_next() else self.d.hints()

    def run_info(self) -> RunInfo:
        """Read from the save block, so it is right however the run was started."""
        sb2 = self.d.saveblock2()
        streak = self.d.emu.read16(sb2 + 0xDE2 + 2 * (1 if self.open_level else 0))
        battle_num = self.d.emu.read16(sb2 + 0xCB2)
        rents = self.d.emu.read16(sb2 + rents_offset(self.open_level))
        if self.phase in (Phase.BATTLE, Phase.FORCED_SWITCH):
            noland = self.d.emu.read16(S.addr("gTrainerBattleOpponent_A")) == 1022    # TRAINER_FRONTIER_BRAIN
        else:
            noland = factory_brain_status(streak, self.d.factory_symbols()) != 0
        return RunInfo(streak, battle_num, streak // 7, self.open_level, self.wins, rents, noland)

    # --- actions ------------------------------------------------------------------------

    def act(self, action: Action) -> None:
        d = self.d
        if self.phase == Phase.RENTAL:
            self._observer = BattleObserver(*self._hints())  # the attendant's hints about the first opponent
            d.pick_rentals(list(action))
        elif self.phase == Phase.SWAP:
            self._observer = BattleObserver(*self._hints())  # hints about the next opponent
            if getattr(self, "_swap_screen_open", False):
                self._swap_screen_open = False
                if action is None:
                    d.quit_swap_screen()
                else:
                    d.swap(action[0], action[1])
            elif action is None:
                d.answer_swap_question(False)
            else:
                d.answer_swap_question(True)
                d.swap(action[0], action[1])
        elif self.phase == Phase.BATTLE:
            self.view()                                     # the observer must see every decision
            if action[0] == "move":
                d.choose_move(action[1])
            elif action[0] == "switch":
                d.choose_switch(action[1])
            else:
                d.forfeit()
        elif self.phase == Phase.FORCED_SWITCH:
            self.view()
            d.choose_party_slot(action[1])
        else:
            raise RuntimeError("run is over")
        self._sync()


class _FieldOrderRam:
    """RAM reader that returns gPlayerParty in field order even while the party menu has it
    rearranged (the simulator never rearranges it)."""

    def __init__(self, driver: FactoryDriver):
        self.d = driver
        self.party = S.addr("gPlayerParty")

    def read(self, addr, size):
        if self.party <= addr < self.party + 600:
            raw = self.d.player_party_raw()
            off = addr - self.party
            return raw[off:off + size]
        return self.d.emu.read(addr, size)

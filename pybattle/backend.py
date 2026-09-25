"""A Battle Factory run behind one interface, whether it runs in the simulator or in mGBA.

    backend.reset(seed, win_streak)
    while backend.phase is not Phase.RUN_OVER:
        view = backend.view()        # RentalView | SwapView | BattleView
        backend.act(action)

Actions by phase:
    RENTAL          (a, b, c)                   select-screen positions to rent, in party order
    SWAP            None | (own_slot, enemy_slot)  keep the team, or trade
    BATTLE          ("move", slot) | ("switch", party_index) | ("forfeit",)
    FORCED_SWITCH   ("switch", party_index)
"""

import enum
from typing import Optional, Tuple, Union

from .emu.decode import SB2_FACTORY_RENTS_COUNT, SYMBOLS as S, decode_party, decode_pokemon
from .pybattle_native import Gen3Game
from .view import BattleObserver, BattleView, OwnMon, RentalView, RunInfo, SwapView


class Phase(enum.Enum):
    RENTAL = "rental"
    SWAP = "swap"
    BATTLE = "battle"               # choose a move or a switch
    FORCED_SWITCH = "forced_switch"  # choose who comes in: after a faint, or after our Baton Pass
    RUN_OVER = "run_over"


Action = Union[Tuple[int, int, int], None, Tuple[int, int], Tuple[str, int], Tuple[str]]

# ---- Battle Factory rules shared by both backends ------------------------------------------------------------------

FLAG_SYS_FACTORY_SILVER = 0x860 + 0x6C       # SYSTEM_FLAGS + 0x6C (constants/flags.h)
FLAG_SYS_FACTORY_GOLD = 0x860 + 0x6D
NO_HINT = (18, 0)                           # hint_type NUMBER_OF_MON_TYPES, hint_style FACTORY_STYLE_NONE


def factory_symbols_for_streak(streak: int) -> int:
    """Symbols a player with this Factory singles streak holds: Noland's silver is won at battle 21, the gold at 42
    (a streak of 21+ went through battle 21, so the silver symbol is necessarily there)."""
    return int(streak >= 21) + int(streak >= 42)


def factory_brain_status(streak: int, symbols: int) -> int:
    """GetFrontierBrainStatus (frontier_util.c) for the Factory's next battle, with `streak` wins so far:
    0 not Noland, 1 silver (21), 2 gold (42), 3 / 4 again at 21 / 42 and every 21 after with both symbols."""
    s = streak + 1
    if symbols < 2:
        return symbols + 1 if s == (21, 42)[symbols] else 0
    if s == 21:
        return 3
    return 4 if s == 42 or (s > 42 and (s - 42) % 21 == 0) else 0


def rents_offset(open_level: bool) -> int:
    """SaveBlock2 offset of factoryRentsCount[singles][lvlMode]."""
    return SB2_FACTORY_RENTS_COUNT + 2 * int(bool(open_level))


class FactoryBackend:
    phase: Phase

    def reset(self, seed: int = 0, win_streak: int = 0) -> None: ...
    def view(self): ...
    def act(self, action: Action) -> None: ...
    def run_info(self) -> RunInfo: ...

    wins: int   # battles won since reset()


class SimBackend(FactoryBackend):
    """The game's own code, headless (Gen3Game). ~1000 battles/s."""

    def __init__(self, open_level: bool = True, max_turns: int = 500):
        self.open_level = open_level
        self.max_turns = max_turns      # safety net: Gen3 battles can stall forever; forfeit then
        self.game = Gen3Game()
        self.phase = Phase.RUN_OVER
        self._observer = BattleObserver()
        self._turns = 0
        self.last_battle_won: Optional[bool] = None

    # --- lifecycle --------------------------------------------------------------------

    def reset(self, seed: int = 0, win_streak: int = 0, rents_count: int = 0) -> None:
        """A new run from `win_streak` (the symbols a player with that streak holds are set: see
        factory_symbols_for_streak) and `rents_count`. Nothing of a previous run is kept."""
        self.game = Gen3Game()
        symbols = factory_symbols_for_streak(win_streak)
        self.game.set_flag(FLAG_SYS_FACTORY_SILVER, symbols >= 1)
        self.game.set_flag(FLAG_SYS_FACTORY_GOLD, symbols >= 2)
        self.game.factory_begin(self.open_level, win_streak, rents_count, seed & 0xFFFFFFFF)
        self._start_streak = win_streak
        self._turns = 0
        self._observer = BattleObserver()
        self._observer_done = None
        self.last_battle_won = None
        self._sync()

    def clone(self) -> "SimBackend":
        other = SimBackend.__new__(SimBackend)
        other.__dict__.update(self.__dict__)
        other.game = self.game.clone()
        other._observer = _copy_observer(self._observer)
        if getattr(self, "_observer_done", None) is not None:
            other._observer_done = other._observer if self._observer_done is self._observer \
                else _copy_observer(self._observer_done)
        return other

    def _sync(self):
        """Advance the game to the next decision and set self.phase."""
        P, D = Gen3Game.FactoryPhase, Gen3Game.Decision
        while True:
            fp = self.game.factory_phase
            if fp == P.RENTAL:
                self.phase = Phase.RENTAL
                return
            if fp == P.SWAP:
                self.phase = Phase.SWAP
                return
            if fp == P.RUN_OVER:
                self.phase = Phase.RUN_OVER
                self.last_battle_won = False
                return
            # the battle first runs to its end (gBattleOutcome set) without winding down, so the observer sees the
            # last turn at the same moment as on the emulator; factory_run_battle then finishes it off
            decision = self.game.run(400000)
            if decision == D.BATTLE_OVER:
                self._observer.finish(self.game)
                decision = self.game.factory_run_battle()
            if decision == D.BATTLE_OVER:
                self.last_battle_won = self.game.factory_info.last_outcome == 1
                self._observer_done = self._observer
                self._observer = BattleObserver() if self.game.factory_phase != P.SWAP else self._observer
                self._turns = 0
                continue
            if decision == D.TIMEOUT:
                raise RuntimeError("gen3 battle did not reach a decision")
            self.phase = Phase.FORCED_SWITCH if decision == D.SWITCH else Phase.BATTLE
            return

    # --- observations -------------------------------------------------------------------

    def view(self):
        if self.phase == Phase.RENTAL:
            info = self.game.factory_info
            mons = [decode_pokemon(self.game.factory_rental(i)) for i in range(6)]
            return RentalView([OwnMon.from_party(m) for m in mons],
                              [self.game.factory_rental_mon_id(i) for i in range(6)], info.hint_type, info.hint_style)
        if self.phase == Phase.SWAP:
            info = self.game.factory_info
            own = [OwnMon.from_party(m) for m in decode_party(self.game.read(S.addr("gPlayerParty"), 300))]
            return SwapView(own, self._observer.swap_candidates(self.game), info.hint_type, info.hint_style,
                            self._observer.foe_records())
        if self.phase in (Phase.BATTLE, Phase.FORCED_SWITCH):
            return self._observer.observe(self.game, self.phase == Phase.FORCED_SWITCH, self.game.unusable_moves(0),
                                          self.game.can_switch(0))
        return None

    def _new_observer(self) -> BattleObserver:
        """Memory for the next battle; it starts with the hints heard before it."""
        info = self.game.factory_info
        return BattleObserver(info.hint_type, info.hint_style)

    @property
    def wins(self) -> int:
        return self.game.factory_info.wins

    def run_info(self) -> RunInfo:
        info = self.game.factory_info
        streak = self._start_streak + info.wins
        battle_num = self.game.read_saveblock2(0xCB2, 2)
        rents = int.from_bytes(self.game.read_saveblock2(rents_offset(self.open_level), 2), "little")
        if self.phase in (Phase.BATTLE, Phase.FORCED_SWITCH):
            noland = info.brain_status != 0
        else:
            noland = factory_brain_status(streak, factory_symbols_for_streak(streak)) != 0
        return RunInfo(streak, int.from_bytes(battle_num, "little"), streak // 7, self.open_level, info.wins, rents,
                       noland)

    # --- actions ------------------------------------------------------------------------

    def act(self, action: Action) -> None:
        g = self.game
        if self.phase == Phase.RENTAL:
            self._observer = self._new_observer()
            if not g.factory_rent(*action):
                raise ValueError(f"rental {action} refused (same species twice?)")
        elif self.phase == Phase.SWAP:
            ok = g.factory_swap(-1) if action is None else g.factory_swap(action[0], action[1])
            if not ok:
                raise ValueError(f"swap {action} refused (species already on the team)")
            self._observer = self._new_observer()
        elif self.phase == Phase.BATTLE:
            self.view()                     # the observer must see every decision (cached if already seen)
            kind = action[0]
            self._turns += 1
            if kind == "forfeit" or self._turns > self.max_turns:
                g.forfeit()
            elif kind == "move":
                g.choose_move(action[1])
            elif kind == "switch":
                g.choose_switch(action[1])
            else:
                raise ValueError(action)
        elif self.phase == Phase.FORCED_SWITCH:
            self.view()
            g.choose_switch(action[1])
        else:
            raise RuntimeError("run is over")
        self._sync()


def _copy_observer(o: BattleObserver) -> BattleObserver:
    """Everything the observer remembers about the battle (reveals, counters, turn start)."""
    return o.fast_copy()

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

from .emu.decode import SYMBOLS as S, decode_party, decode_pokemon
from .pybattle_native import Gen3Game
from .view import BattleObserver, BattleView, OwnMon, RentalView, RunInfo, SwapView


class Phase(enum.Enum):
    RENTAL = "rental"
    SWAP = "swap"
    BATTLE = "battle"               # choose a move or a switch
    FORCED_SWITCH = "forced_switch"  # choose who comes in: after a faint, or after our Baton Pass
    RUN_OVER = "run_over"


Action = Union[Tuple[int, int, int], None, Tuple[int, int], Tuple[str, int], Tuple[str]]


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
        self.game = Gen3Game()
        self.game.factory_begin(self.open_level, win_streak, rents_count, seed & 0xFFFFFFFF)
        self._start_streak = win_streak
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
            return SwapView(own, self._observer.swap_candidates(self.game), info.hint_type, info.hint_style)
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
        rents = int.from_bytes(self.game.read_saveblock2(0xDF4, 2), "little")   # factoryRentsCount[singles][open]
        return RunInfo(streak, int.from_bytes(battle_num, "little"), streak // 7, self.open_level, info.wins, rents)

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

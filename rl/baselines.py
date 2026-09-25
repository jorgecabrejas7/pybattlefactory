"""Baseline policies (docs/RL_DECISIONS.md §8).

Battlers act on a BattleView, tacticians on a RentalView / SwapView. All only use player-visible information.
"""

import random
from itertools import combinations

from pybattle.backend import Phase
from .gamedata import MOVES, SPECIES, effectiveness


def legal_battle_actions(view):
    acts = [("move", i) for i, ok in enumerate(view.usable_moves) if ok]
    acts += [("switch", i) for i in view.switch_targets]
    return acts or [("move", 0)]        # nothing selectable: the game uses the forced move / Struggle


def move_score(move_id: int, user_types, target_types) -> float:
    """power x STAB x type effectiveness against the target."""
    m = MOVES[move_id]
    if not move_id or m["power"] <= 1:          # status moves; power 1 = variable-power moves
        return 0.0
    stab = 1.5 if m["type"] in user_types else 1.0
    return m["power"] * stab * effectiveness(m["type"], target_types) * m["accuracy"] / 100 if m["accuracy"] \
        else m["power"] * stab * effectiveness(m["type"], target_types)


class RandomBattler:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def __call__(self, view):
        return self.rng.choice(legal_battle_actions(view))


class MaxDamageBattler:
    """Best power x STAB x effectiveness move against the enemy's active Pokemon; never switches voluntarily.
    After a faint, sends the party member with the best such move."""

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def __call__(self, view):
        target = view.enemy_active.types
        if view.forced_switch:
            def best(i):
                mon = view.own_party[i]
                return max(move_score(mv, mon.types, target) for mv in mon.moves)
            return ("switch", max(view.switch_targets, key=best))
        me = view.own_party[view.own_active.party_index]
        usable = [i for i, ok in enumerate(view.usable_moves) if ok]
        if not usable:
            return legal_battle_actions(view)[0]
        scores = {i: move_score(me.moves[i], me.types, target) for i in usable}
        top = max(scores.values())
        return ("move", self.rng.choice([i for i in usable if scores[i] == top]))


class RandomTactician:
    """Random rental (distinct species), never swaps."""

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def __call__(self, phase, view):
        if phase == Phase.SWAP:
            return None
        sp = [m.species for m in view.candidates]
        while True:
            pick = self.rng.sample(range(6), 3)
            if len({sp[i] for i in pick}) == 3:
                return tuple(pick)


def run_streak(backend, tactician, battler, seed: int, win_streak: int = 0, max_decisions: int = 300):
    """Play one run until it is lost (or a battle is truncated). Returns (wins, truncated)."""
    backend.reset(seed=seed, win_streak=win_streak)
    n = 0
    while backend.phase != Phase.RUN_OVER:
        view = backend.view()
        if backend.phase in (Phase.RENTAL, Phase.SWAP):
            backend.act(tactician(backend.phase, view))
            n = 0
            continue
        n += 1
        if n > max_decisions:
            return backend.wins, True
        backend.act(battler(view))
    return backend.wins, False

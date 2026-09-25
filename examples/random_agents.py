"""Two dummy agents that take random legal actions — a tactician (rentals, swaps) and a battler
(moves, switches). Useful to watch the game loop (scripts/watch.py) and as a template for real
agents: see pybattle/agent.py for the interface.
"""

import random

from pybattle.agent import Choice
from pybattle.backend import Phase
from pybattle.view import NAMES

_species = lambda s: NAMES["species"].get(str(s), str(s)).replace("SPECIES_", "").title()
_move = lambda m: NAMES["moves"].get(str(m), str(m)).replace("MOVE_", "").replace("_", " ").title()


class RandomTactician:
    """Rents 3 random Pokemon (distinct species, as the game requires); after a win keeps the
    team or makes a random allowed trade, 50/50."""

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def act(self, phase, view, info):
        if phase == Phase.RENTAL:
            slots = list(range(6))
            self.rng.shuffle(slots)
            picks = []
            for s in slots:
                if view.candidates[s].species not in {view.candidates[p].species for p in picks}:
                    picks.append(s)
                if len(picks) == 3:
                    break
            names = ", ".join(_species(view.candidates[p].species) for p in picks)
            return Choice(tuple(picks), f"rent {names} (random)")
        # SWAP
        trades = [(p, e) for p in range(3) for e in range(3)
                  if view.enemy_party[e].species not in {m.species for i, m in enumerate(view.own_party) if i != p}]
        if not trades or self.rng.random() < 0.5:
            return Choice(None, "keep the team (random)")
        p, e = self.rng.choice(trades)
        return Choice((p, e), f"trade {_species(view.own_party[p].species)} for "
                              f"{_species(view.enemy_party[e].species)} (random)")


class RandomBattler:
    """Picks uniformly among the legal moves and switches; replacements after a faint are random.
    Forfeits after `max_turns` turns so a stalled battle ends (Gen 3 has no turn limit)."""

    def __init__(self, seed=None, switch_prob=0.1, max_turns=300):
        self.rng = random.Random(seed)
        self.switch_prob = switch_prob
        self.max_turns = max_turns

    def act(self, phase, view, info):
        if phase == Phase.FORCED_SWITCH:
            i = self.rng.choice(view.switch_targets)
            return Choice(("switch", i), f"send out {_species(view.own_party[i].species)} (random)")
        if view.turn >= self.max_turns:
            return Choice(("forfeit",), f"turn {view.turn}: forfeit")
        active = view.own_party[view.own_active.party_index]
        moves = [i for i, ok in enumerate(view.usable_moves) if ok]
        if view.switch_targets and (not moves or self.rng.random() < self.switch_prob):
            i = self.rng.choice(view.switch_targets)
            return Choice(("switch", i), f"switch to {_species(view.own_party[i].species)} (random)")
        if not moves:                   # every move blocked: the game uses Struggle
            return Choice(("move", 0), "Struggle")
        m = self.rng.choice(moves)
        return Choice(("move", m), f"{_move(active.moves[m])} (random)")

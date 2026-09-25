"""Static game knowledge (species, moves, type chart) shared by encoders and baselines."""

import numpy as np

from pybattle.view import _DATA

SPECIES = _DATA["species"]
MOVES = _DATA["moves"]
N_SPECIES = len(SPECIES)
N_MOVES = len(MOVES)
N_ITEMS = _DATA["counts"]["ITEMS_COUNT"]
N_ABILITIES = _DATA["counts"]["ABILITIES_COUNT"]
N_EFFECTS = _DATA["counts"]["NUM_BATTLE_MOVE_EFFECTS"]
N_TYPES = 18
TYPE_MYSTERY = 9

# TYPE_EFFECTIVENESS[atk, def] as a multiplier (the "Foresight" section is ignored: it only applies to Ghosts
# under Foresight/Odor Sleuth, which the baseline does not model)
TYPE_EFFECTIVENESS = np.ones((N_TYPES, N_TYPES), dtype=np.float32)
for atk, dfn, mult in _DATA["type_effectiveness"]:
    if atk < N_TYPES and dfn < N_TYPES:
        TYPE_EFFECTIVENESS[atk, dfn] = mult / 10.0


def effectiveness(move_type: int, def_types) -> float:
    m = 1.0
    for t in set(def_types):
        m *= TYPE_EFFECTIVENESS[move_type, t]
    return m

"""The interface between the game and an agent.

At every decision the backend is in a `Phase` and gives a view; the agent answers with an action:

    Phase           view          action
    RENTAL          RentalView    (a, b, c)            select-screen slots 0-5, party order (a leads)
    SWAP            SwapView      None                 keep the team
                                  (own_slot, enemy_slot)  trade our party slot for the defeated team's
    BATTLE          BattleView    ("move", slot)       slot 0-3, must be in view.usable_moves
                                  ("switch", party_index)  must be in view.switch_targets
                                  ("forfeit",)         give up the battle (ends the run)
    FORCED_SWITCH   BattleView    ("switch", party_index)  who comes in after a faint or our Baton Pass
                                  (Roar/Whirlwind pick the replacement at random: no decision)

`RunInfo` (backend.run_info()) has the streak, the battle number in the challenge and the wins so
far. How an agent turns views into actions is entirely up to it.
"""

from dataclasses import dataclass
from typing import Any, Optional, Protocol

from .backend import Action, Phase
from .view import RunInfo


@dataclass
class Choice:
    action: Action
    reason: str = ""       # optional, shown by the viewer / written to the log
    details: Optional[dict] = None   # optional, structured, for the viewer's decision panel (see below)


# `Choice.details` (all keys optional; the watch window falls back to `reason` for anything missing):
#   "actions":  [{"label": str, "p": float, "action": Action, ...}, ...]   the options the agent weighed; extra
#               keys the viewer understands: "move" (move id), "damage" ([min, max] fraction of the foe's HP,
#               can_ko), "species" (species id), "threat" (worst revealed foe move vs it, as a fraction, can_ko)
#   "chosen":   index into "actions" of the option taken
#   "p_win":    the agent's estimated chance of winning this battle (0-1)
#   "expected_wins": expected wins ahead in the run
#   rentals:    "lead_probs" (6 floats), "team" (the 3 slots taken), "pairs" (like "actions", for the two partners)


class Agent(Protocol):
    def act(self, phase: Phase, view: Any, info: RunInfo) -> Choice: ...

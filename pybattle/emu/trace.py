"""Record Battle Factory battles from the real game as JSON traces.

A trace holds the battle's starting point (both parties as raw `struct Pokemon` bytes plus
gRngValue while gBattleMainFunc == BeginBattleIntro, the last point where the RNG is stable),
then a snapshot of the full battle state at every player decision together with the action
taken. The simulator replays the player's actions from the same start and must reproduce
every snapshot, including gRngValue, bit for bit.
"""

import json
import struct
from typing import Callable, Dict, List

from .decode import SYMBOLS as S, decode_battle_mon, decode_party
from .driver import Decision, FactoryDriver

TRACE_VERSION = 1

# Battle state regions captured raw at every decision (name -> (symbol, size or None=symbol size))
_REGIONS = [
    "gBattleMons", "gBattlerPartyIndexes", "gDisableStructs", "gSideStatuses", "gSideTimers",
    "gStatuses3", "gBattleWeather", "gWishFutureKnock", "gProtectStructs", "gLastMoves",
    "gLastLandedMoves", "gLastHitBy", "gChosenMoveByBattler", "gRandomTurnNumber", "gBattleResults",
]

_AI_FLAGS_OFFSET = 12            # AI_ThinkingStruct.aiFlags
_RESOURCES_AI = 5 * 4            # BattleResources.ai
_RESOURCES_FLAGS = 1 * 4         # BattleResources.flags (Flash Fire etc.)


def _hex(b: bytes) -> str:
    return b.hex()


def snapshot(d: FactoryDriver) -> Dict:
    emu = d.emu
    regions = {name: _hex(emu.read(S.addr(name), S.size(name))) for name in _REGIONS}
    res = emu.read32(S.addr("gBattleResources"))
    flags_ptr = emu.read32(res + _RESOURCES_FLAGS)
    regions["resourceFlags"] = _hex(emu.read(flags_ptr, 16))
    return {
        "rng": d.rng(),
        "regions": regions,
        "player_party": _hex(d.player_party_raw()[:300]),
        "enemy_party": _hex(emu.read(S.addr("gEnemyParty"), 300)),
    }


def battle_start(d: FactoryDriver) -> Dict:
    emu = d.emu
    sb2 = d.saveblock2()
    return {
        "rng": d.rng(),
        "battle_type_flags": emu.read32(S.addr("gBattleTypeFlags")),
        "trainer_id": emu.read16(S.addr("gTrainerBattleOpponent_A")),
        "lvl_mode": emu.read8(sb2 + 0xCA9) & 3,
        "battle_num": emu.read16(sb2 + 0xCB2),
        "factory_win_streaks": list(struct.unpack("<4H", emu.read(sb2 + 0xDE2, 8))),
        "tower_win_streaks": list(struct.unpack("<8H", emu.read(sb2 + 0xCE0, 16))),
        "player_party": _hex(emu.read(S.addr("gPlayerParty"), 300)),
        "enemy_party": _hex(emu.read(S.addr("gEnemyParty"), 300)),
    }


def ai_flags(d: FactoryDriver) -> int:
    res = d.emu.read32(S.addr("gBattleResources"))
    return d.emu.read32(d.emu.read32(res + _RESOURCES_AI) + _AI_FLAGS_OFFSET)


# A policy maps (driver, decision) -> action dict: {"move": slot} | {"switch": party_index}
Policy = Callable[[FactoryDriver, Decision], Dict]


def record_battle(d: FactoryDriver, policy: Policy, max_decisions: int = 500) -> Dict:
    """From anywhere before the battle starts, play one battle with `policy` and record it."""
    main_func = lambda: d.fn_name(d.emu.read32(S.addr("gBattleMainFunc")))
    if not d.mash_until(lambda: d.callback2() in ("CB2_HandleStartBattle", "BattleMainCB2")
                        and main_func() == "BeginBattleIntro", max_frames=60000):
        raise TimeoutError("battle did not start")
    trace = {"version": TRACE_VERSION, "start": battle_start(d), "decisions": []}

    for _ in range(max_decisions):
        dec = d.advance()
        if dec == Decision.BATTLE_OVER:
            break
        if dec not in (Decision.BATTLE_ACTION, Decision.PARTY_MENU):
            raise RuntimeError(f"unexpected decision {dec} inside battle")
        snap = snapshot(d)
        action = policy(d, dec)
        trace["decisions"].append({"kind": dec.value, "state": snap, "action": action})
        if "move" in action:
            if not d.choose_move(action["move"]):
                action["forced"] = True   # Encore / Struggle: the game chose the move
        elif dec == Decision.PARTY_MENU:
            d.choose_party_slot(action["switch"])
        else:
            d.choose_switch(action["switch"])

    # Final state once the battle has been decided
    trace["end"] = {"outcome": d.battle_outcome(), "state": snapshot(d)}
    return trace


def random_policy(rng) -> Policy:
    """Uniform over legal moves and switches (mostly moves, so battles progress)."""
    def policy(d: FactoryDriver, dec: Decision) -> Dict:
        party = d.player_party()
        active = d.battler_party_indexes()[0]
        bench = [i for i, m in enumerate(party) if m.hp > 0 and i != active]
        if dec == Decision.PARTY_MENU:
            return {"switch": rng.choice(bench)}
        from ..emu_backend import can_switch, unusable_moves
        bad = unusable_moves(d.emu)
        moves = [i for i in range(4) if not (bad >> i) & 1]
        if not can_switch(d.emu):
            bench = []
        if bench and (not moves or rng.random() < 0.15):
            return {"switch": rng.choice(bench)}
        return {"move": rng.choice(moves)} if moves else {"move": 0}
    return policy


def save_trace(trace: Dict, path: str):
    with open(path, "w") as f:
        json.dump(trace, f, indent=1)


def load_trace(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)

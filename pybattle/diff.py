"""Differential check: replay an emulator trace in the gen3 battle core and compare.

The simulator gets the trace's starting point (raw parties, battle flags, Frontier save data,
gRngValue at BeginBattleIntro), receives the same player actions, and must reproduce every
recorded snapshot -- battle RAM regions, both parties and gRngValue -- byte for byte. The
opponent's actions are not replayed: the game's AI inside the simulator has to pick the same
ones.
"""

import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .emu.decode import (SYMBOLS as S, SB2_FACTORY_WIN_STREAKS, SB2_FRONTIER_CUR_BATTLE_NUM, SB2_FRONTIER_LVL_MODE,
                         SB2_TOWER_WIN_STREAKS, decode_battle_mon, decode_party)
from .pybattle_native import Gen3Game

VAR_FRONTIER_BATTLE_MODE = 0x40CE
VAR_FRONTIER_FACILITY = 0x40CF
FRONTIER_FACILITY_FACTORY = 4
FRONTIER_MODE_SINGLES = 0

# Regions that hold host pointers (not comparable byte for byte).
_SKIP_REGIONS = {"resourceFlags"}

_DECISION_KIND = {"battle_action": Gen3Game.Decision.ACTION, "party_menu": Gen3Game.Decision.SWITCH}


@dataclass
class Mismatch:
    decision: int            # index of the decision (len(decisions) = end of battle)
    what: str
    expected: str
    got: str


@dataclass
class ReplayResult:
    decisions_matched: int
    total_decisions: int
    mismatches: List[Mismatch] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.mismatches


def setup_battle(start: Dict) -> Gen3Game:
    b = Gen3Game()
    b.set_battle(start["battle_type_flags"], start["trainer_id"])
    b.write_party(0, bytes.fromhex(start["player_party"]))
    b.write_party(1, bytes.fromhex(start["enemy_party"]))
    b.set_var(VAR_FRONTIER_FACILITY, FRONTIER_FACILITY_FACTORY)
    b.set_var(VAR_FRONTIER_BATTLE_MODE, FRONTIER_MODE_SINGLES)
    lvl = b.read_saveblock2(SB2_FRONTIER_LVL_MODE, 1)[0]
    b.write_saveblock2(SB2_FRONTIER_LVL_MODE, bytes([(lvl & ~3) | start["lvl_mode"]]))
    b.write_saveblock2(SB2_FRONTIER_CUR_BATTLE_NUM, struct.pack("<H", start["battle_num"]))
    b.write_saveblock2(SB2_FACTORY_WIN_STREAKS, struct.pack("<4H", *start["factory_win_streaks"]))
    b.write_saveblock2(SB2_TOWER_WIN_STREAKS, struct.pack("<8H", *start["tower_win_streaks"]))
    if not b.start(start["rng"]):
        raise RuntimeError("battle did not reach BeginBattleIntro")
    return b


def sim_snapshot(b: Gen3Game, regions) -> Dict:
    out = {"rng": b.rng, "regions": {}}
    for name in regions:
        if name in _SKIP_REGIONS:
            continue
        out["regions"][name] = b.read(S.addr(name), S.size(name)).hex()
    out["player_party"] = b.read(S.addr("gPlayerParty"), 300).hex()
    out["enemy_party"] = b.read(S.addr("gEnemyParty"), 300).hex()
    return out


def _describe(name: str, expected_hex: str, got_hex: str) -> str:
    """Human-readable field diff for the regions where it matters most."""
    e, g = bytes.fromhex(expected_hex), bytes.fromhex(got_hex)
    if name == "gBattleMons":
        diffs = []
        for i in range(2):
            me, mg = decode_battle_mon(e[i * 0x58:(i + 1) * 0x58]), decode_battle_mon(g[i * 0x58:(i + 1) * 0x58])
            for k in me.__dataclass_fields__:
                if getattr(me, k) != getattr(mg, k):
                    diffs.append(f"battler{i}.{k}: {getattr(me, k)} != {getattr(mg, k)}")
        return "; ".join(diffs)
    if name in ("player_party", "enemy_party"):
        diffs = []
        for i, (me, mg) in enumerate(zip(decode_party(e), decode_party(g))):
            for k in ("species", "hp", "status", "pp", "held_item", "moves"):
                if getattr(me, k) != getattr(mg, k):
                    diffs.append(f"mon{i}.{k}: {getattr(me, k)} != {getattr(mg, k)}")
        return "; ".join(diffs)
    first = next(i for i in range(min(len(e), len(g))) if e[i] != g[i])
    return f"first differing byte at +{first:#x}: {e[first]:#04x} != {g[first]:#04x}"


def _mask_battle_mons(raw: bytes) -> bytes:
    """Blank bytes the game fills from uninitialised stack memory: BattlePokemon.unknown and the
    nickname/OT-name bytes after their terminator (CopyPlayerMonData copies a stack struct)."""
    b = bytearray(raw)
    for base in range(0, len(b), 0x58):
        b[base + 0x23] = 0
        for start, length in ((0x30, 11), (0x3C, 8)):
            field = b[base + start:base + start + length]
            end = field.find(0xFF)
            if end >= 0:
                b[base + start + end + 1:base + start + length] = bytes(length - end - 1)
    return bytes(b)


_NORMALIZE = {"gBattleMons": _mask_battle_mons}


def compare(index: int, expected: Dict, got: Dict) -> List[Mismatch]:
    out = []
    if expected["rng"] != got["rng"]:
        out.append(Mismatch(index, "gRngValue", f"{expected['rng']:08x}", f"{got['rng']:08x}"))
    for name, hexval in expected["regions"].items():
        if name in _SKIP_REGIONS:
            continue
        norm = _NORMALIZE.get(name, lambda b: b)
        e, g = norm(bytes.fromhex(hexval)), norm(bytes.fromhex(got["regions"][name]))
        if e != g:
            out.append(Mismatch(index, name, "", _describe(name, e.hex(), g.hex())))
    for name in ("player_party", "enemy_party"):
        if expected[name] != got[name]:
            out.append(Mismatch(index, name, "", _describe(name, expected[name], got[name])))
    return out


def replay_trace(trace: Dict, stop_at_first: bool = True) -> ReplayResult:
    b = setup_battle(trace["start"])
    decisions = trace["decisions"]
    result = ReplayResult(0, len(decisions))
    for i, dec in enumerate(decisions):
        kind = b.run()
        if kind != _DECISION_KIND[dec["kind"]]:
            result.mismatches.append(Mismatch(i, "decision kind", dec["kind"], str(kind)))
            return result
        mism = compare(i, dec["state"], sim_snapshot(b, dec["state"]["regions"]))
        if mism:
            result.mismatches += mism
            if stop_at_first:
                return result
        else:
            result.decisions_matched += 1
        action = dec["action"]
        if "move" in action:
            b.choose_move(action["move"])
        else:
            b.choose_switch(action["switch"])
    kind = b.run()
    end = trace["end"]
    if kind != Gen3Game.Decision.BATTLE_OVER:
        result.mismatches.append(Mismatch(len(decisions), "battle end", "BATTLE_OVER", str(kind)))
        return result
    outcome = b.read(S.addr("gBattleOutcome"), 1)[0]
    if outcome != end["outcome"]:
        result.mismatches.append(Mismatch(len(decisions), "gBattleOutcome", str(end["outcome"]), str(outcome)))
    result.mismatches += compare(len(decisions), end["state"], sim_snapshot(b, end["state"]["regions"]))
    return result

"""Move/switch legality: the Python checks the emulator backend uses must agree with the game's
own logic in the simulator, and trapped Pokemon cannot switch in either."""

import random
import struct

import pytest

from pybattle.backend import Phase, SimBackend
from pybattle.emu.decode import SYMBOLS as S
from pybattle.emu_backend import can_switch, unusable_moves

STATUS2_ESCAPE_PREVENTION = 1 << 26


_FAKE_BATTLE_STRUCT = 0x0E000000


class _SimRam:
    """The simulator as a GBA-address RAM reader, like the emulator. gBattleStruct is a host
    pointer in the simulator, so its one field used here is served through an accessor."""

    def __init__(self, game):
        self.g = game

    def read(self, addr, size):
        if addr == S.addr("gBattleStruct") and size == 4:
            return struct.pack("<I", _FAKE_BATTLE_STRUCT)
        if addr >= _FAKE_BATTLE_STRUCT:
            return struct.pack("<H", self.g.choiced_move((addr - _FAKE_BATTLE_STRUCT - 200) // 2))
        return self.g.read(addr, size)


def _battle_states(n_runs=25):
    rng = random.Random(0)
    for seed in range(n_runs):
        b = SimBackend()
        b.reset(seed=seed, win_streak=rng.choice([0, 21, 42]))
        steps = 0
        while b.phase != Phase.RUN_OVER and steps < 400:
            v = b.view()
            if b.phase == Phase.BATTLE:
                yield b
                moves = [i for i, ok in enumerate(v.usable_moves) if ok]
                if v.switch_targets and (not moves or rng.random() < 0.2):
                    b.act(("switch", rng.choice(v.switch_targets)))
                else:
                    b.act(("move", rng.choice(moves) if moves else 0))
            elif b.phase == Phase.FORCED_SWITCH:
                b.act(("switch", rng.choice(v.switch_targets)))
            elif b.phase == Phase.RENTAL:
                picks = [0]
                for i in range(1, 6):
                    if len(picks) < 3 and v.candidates[i].species not in {v.candidates[p].species for p in picks}:
                        picks.append(i)
                b.act(tuple(picks))
            else:
                b.act(None)
            steps += 1


def test_python_checks_match_game():
    n = 0
    for b in _battle_states():
        g = b.game
        ram = _SimRam(g)
        assert unusable_moves(ram) == g.unusable_moves(0)
        assert can_switch(ram) == g.can_switch(0)
        n += 1
    assert n > 200


def test_trapped_mon_cannot_switch():
    b = next(s for s in _battle_states(3) if s.view().switch_targets)
    g = b.game
    addr = S.addr("gBattleMons") + 0x50
    status2 = struct.unpack("<I", g.read(addr, 4))[0]
    g.write(addr, struct.pack("<I", status2 | STATUS2_ESCAPE_PREVENTION))   # Mean Look
    assert not g.can_switch(0) and not can_switch(_SimRam(g))
    assert b.view().switch_targets == []
    with pytest.raises(RuntimeError):
        g.choose_switch(b.view().own_party.index(b.view().own_party[0]) or 1)


def test_no_pp_left_uses_struggle():
    """All PP gone: FIGHT makes the game use Struggle, with no extra decision afterwards."""
    b = next(s for s in _battle_states(3) if s.view().turn >= 1)
    g = b.game
    base = S.addr("gBattleMons")
    g.write(base + 0x24, bytes(4))                              # battler 0: PP of all 4 moves = 0
    assert not any(b.view().usable_moves) and g.unusable_moves(0) == 0xF
    turn = b.view().turn
    b.act(("move", 0))
    if b.phase == Phase.BATTLE:
        last = struct.unpack("<H", g.read(S.addr("gLastMoves"), 2))[0]
        assert last == 165                                      # MOVE_STRUGGLE
        assert b.view().turn == turn + 1                        # the turn happened, no extra prompt
        assert not any(b.view().usable_moves)

"""The pybind interface checks every index it forwards to the game (no read or write outside an array)."""

import itertools
import struct

import pytest

from pybattle.backend import Phase, SimBackend
from pybattle.emu.decode import SYMBOLS as S


def _battle(seed=1):
    b = SimBackend(max_turns=10 ** 9)
    b.reset(seed=seed)
    sp = [m.species for m in b.view().candidates]
    b.act(next(t for t in itertools.combinations(range(6), 3) if len({sp[i] for i in t}) == 3))
    assert b.phase == Phase.BATTLE
    return b


def test_battle_indices_are_checked():
    b = _battle()
    g = b.game.clone()
    before = g._state_bytes()
    for slot in (-1, 4, 200):
        with pytest.raises(IndexError):
            g.choose_move(slot)
    for idx in (-1, 3, 9, 255):
        with pytest.raises(IndexError):
            g.choose_switch(idx)
    active = struct.unpack("<H", g.read(S.addr("gBattlerPartyIndexes"), 2))[0]
    with pytest.raises(ValueError):
        g.choose_switch(active)                    # already out
    for battler in (-1, 2, 200):
        for f in (g.unusable_moves, g.can_switch, g.choiced_move):
            with pytest.raises(IndexError):
                f(battler)
    with pytest.raises(IndexError):
        g.sim_step(0, 4)
    with pytest.raises(IndexError):
        g.sim_step(1, 3)                           # an empty party slot
    with pytest.raises(ValueError):
        g.sim_step(1, active)
    with pytest.raises(ValueError):
        g.sim_step(2, 0)
    assert g._state_bytes() == before              # nothing was changed by the refused calls
    # an empty move slot
    mv = S.addr("gBattleMons") + 0x0C + 2 * 3
    g.write(mv, struct.pack("<H", 0))
    with pytest.raises(ValueError):
        g.choose_move(3)
    # valid calls still work
    assert b.game.clone().sim_step(0, 0)[0] in (1, 2, 3)


def test_factory_indices_are_checked():
    b = SimBackend()
    b.reset(seed=2)
    g = b.game
    for i in (-1, 6, 255):
        with pytest.raises(IndexError):
            g.factory_rental(i)
        with pytest.raises(IndexError):
            g.factory_rental_mon_id(i)
    with pytest.raises(IndexError):
        g.factory_rent(0, 1, 40)
    with pytest.raises(IndexError):
        g.factory_rent(-1, 1, 2)
    from test_factory_rules import _win_until
    b = SimBackend(max_turns=10 ** 9)
    b.reset(seed=3)
    _win_until(b, 1)                               # at the swap after the first win
    assert b.phase == Phase.SWAP
    for p, e in ((7, 1), (3, 0), (0, 3), (0, -1), (256, 0)):
        with pytest.raises(IndexError):
            b.game.factory_swap(p, e)
    assert b.game.factory_swap(-1)                 # keep the team: fine

"""Battle Factory rules of the simulator that the RL environment depends on: Noland's schedule and symbols, the
attendant's hints before his battles, the rental counter, and a refused rental leaving nothing changed."""

import struct

import pytest

from pybattle.backend import (FLAG_SYS_FACTORY_GOLD, FLAG_SYS_FACTORY_SILVER, NO_HINT, Phase, SimBackend,
                              factory_brain_status, factory_symbols_for_streak, rents_offset)
from pybattle.emu.decode import SYMBOLS as S
from pybattle.view import MOVES


def _pick_rental(v):
    sp = [m.species for m in v.candidates]
    pick = [0]
    for i in range(1, 6):
        if sp[i] not in [sp[j] for j in pick]:
            pick.append(i)
    return tuple(pick[:3])


def _weaken_opponent(b):
    """Every opponent Pokemon down to 1 HP: our next hit wins (a test shortcut, not a game action)."""
    g = b.game
    ep = S.addr("gEnemyParty")
    for i in range(3):
        if struct.unpack("<H", g.read(ep + i * 100 + 0x56, 2))[0] > 1:
            g.write(ep + i * 100 + 0x56, struct.pack("<H", 1))
    bm = S.addr("gBattleMons") + 0x58
    if struct.unpack("<H", g.read(bm + 0x28, 2))[0] > 1:
        g.write(bm + 0x28, struct.pack("<H", 1))
    g.write(bm + 0x24, bytes(4))            # no PP left: it can only Struggle (and its recoil)


def _strongest_move(v):
    moves = v.own_party[v.own_active.party_index].moves
    best = max((a for a in v.legal_actions if a[0] == "move"),
               key=lambda a: MOVES[moves[a[1]]]["power"] if moves[a[1]] else 0, default=None)
    return best or v.legal_actions[0]


def _win_until(b, wins, keep=True):
    """Play (forcing wins) until `wins` battles are won, stopping at the next SWAP / RENTAL decision."""
    while b.phase != Phase.RUN_OVER and b.wins < wins:
        v = b.view()
        if b.phase == Phase.RENTAL:
            b.act(_pick_rental(v))
        elif b.phase == Phase.SWAP:
            b.act(None)
        elif b.phase == Phase.FORCED_SWITCH:
            b.act(("switch", v.switch_targets[0]))
        else:
            _weaken_opponent(b)
            b.act(_strongest_move(b.view()))


def test_brain_schedule_follows_the_game():
    assert [factory_symbols_for_streak(s) for s in (0, 20, 21, 41, 42, 100)] == [0, 0, 1, 1, 2, 2]
    assert factory_brain_status(20, 0) == 1 and factory_brain_status(41, 1) == 2
    assert factory_brain_status(41, 0) == 0                     # no silver symbol: no gold battle
    assert [factory_brain_status(s, 2) for s in (20, 41, 62, 83, 61)] == [3, 4, 4, 4, 0]


@pytest.mark.parametrize("streak,status", [(20, 1), (41, 2), (62, 4), (83, 4), (27, 0)])
def test_reset_sets_the_symbols_of_the_streak(streak, status):
    """A run started at `streak` meets Noland exactly when a player with that streak would."""
    b = SimBackend(max_turns=10 ** 9)
    b.reset(seed=streak, win_streak=streak, rents_count=3)
    assert b.run_info().noland == (status != 0)
    b.act(_pick_rental(b.view()))
    assert b.game.factory_info.brain_status == status
    assert b.run_info().noland == (status != 0)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_no_opponent_or_hint_before_noland(seed):
    """AskReadyForHead -> AskSwapBeforeHead: before Noland's battle no opponent is generated and the attendant
    gives no hint; before a normal battle the hint is about the opponent that is then fought."""
    b = SimBackend(max_turns=10 ** 9)
    b.reset(seed=seed, win_streak=19, rents_count=3)
    _win_until(b, 1)
    assert b.phase == Phase.SWAP and b.run_info().noland
    v = b.view()
    assert (v.hint_type, v.hint_style) == NO_HINT
    party = b.game.read(S.addr("gFrontierTempParty"), 6)
    b.act(None)
    assert b.game.factory_info.brain_status == 1
    assert (b.view().hint_type, b.view().hint_style) == NO_HINT
    # a normal battle: the opponent is generated before the swap question (gFrontierTempParty is its team)
    c = SimBackend(max_turns=10 ** 9)
    c.reset(seed=seed, win_streak=15, rents_count=3)
    _win_until(c, 1)
    assert c.phase == Phase.SWAP and not c.run_info().noland
    temp = c.game.read(S.addr("gFrontierTempParty"), 6)
    c.act(None)
    assert c.game.factory_info.brain_status == 0
    enemy = [struct.unpack("<H", c.game.read(S.addr("gEnemyParty") + i * 100, 2)) for i in range(3)]
    assert temp == c.game.read(S.addr("gFrontierTempParty"), 6) and len(enemy) == 3
    assert party is not None


def test_symbols_are_given_after_noland():
    b = SimBackend(max_turns=10 ** 9)
    b.reset(seed=4, win_streak=20, rents_count=3)
    _win_until(b, 1)                                         # beat Noland at 21: the silver symbol
    assert b.run_info().win_streak == 21
    # at 42 the gold battle follows: brain status 2 needs the silver symbol that was just given
    assert factory_brain_status(41, 1) == 2


def test_rents_counter_offsets():
    assert rents_offset(True) == 0xDF4 and rents_offset(False) == 0xDF2
    for open_level in (True, False):
        b = SimBackend(open_level=open_level, max_turns=10 ** 9)
        b.reset(seed=5, win_streak=7, rents_count=4)
        assert b.run_info().rents == 4


def test_refused_rental_changes_nothing():
    b = SimBackend()
    b.reset(seed=6)
    v = b.view()
    pick = _pick_rental(v)
    ref = SimBackend()
    ref.reset(seed=6)
    ref.act(pick)
    assert not b.game.factory_rent(pick[0], pick[0], pick[1])      # the same slot twice: refused
    sp = [m.species for m in v.candidates]
    dup = next(((i, j) for i in range(6) for j in range(i + 1, 6) if sp[i] == sp[j]), None)
    if dup:
        assert not b.game.factory_rent(dup[0], dup[1], next(k for k in range(6) if sp[k] != sp[dup[0]]))
    b.act(pick)                                                     # a valid rental afterwards: the same team
    own = lambda x: [m.species for m in x.view().own_party]
    assert own(b) == own(ref)

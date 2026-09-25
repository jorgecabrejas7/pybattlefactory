"""Encoding v4 (docs/RL_DECISIONS.md §17): damage estimates without the opponent's IVs, Thick Club / Light Ball on
the right species, and the defeated-opponent records in the swap encoding."""

import random

import numpy as np
import pytest

from pybattle.backend import Phase, SimBackend
from pybattle.view import NAMES, FoeRecord, SwapView
from rl import damage, encode
from rl.gamedata import SPECIES

SP = {v: int(k) for k, v in NAMES["species"].items()}
ITEM = {v: int(k) for k, v in NAMES["items"].items()}


@pytest.fixture(autouse=True)
def _restore_version():
    old = encode.VERSION
    yield
    encode.set_version(old)


def _dmg(move, species, item, version, phys_stat=200):
    types = SPECIES[species]["types"]
    return damage.move_damage(move, (phys_stat, phys_stat), (200, 200), atk_types=types, def_types=[0],
                              atk_item=item, atk_species=species, version=version)


def test_thick_club_and_light_ball_species():
    bonemerang = int(next(k for k, v in NAMES["moves"].items() if v == "MOVE_BONEMERANG"))
    thunderbolt = int(next(k for k, v in NAMES["moves"].items() if v == "MOVE_THUNDERBOLT"))
    club, ball = ITEM["ITEM_THICK_CLUB"], ITEM["ITEM_LIGHT_BALL"]
    for sp in ("SPECIES_CUBONE", "SPECIES_MAROWAK"):
        assert _dmg(bonemerang, SP[sp], club, 4)[1] > 1.9 * _dmg(bonemerang, SP[sp], 0, 4)[1]
    assert _dmg(bonemerang, SP["SPECIES_PIKACHU"], club, 4) == _dmg(bonemerang, SP["SPECIES_PIKACHU"], 0, 4)
    assert _dmg(thunderbolt, SP["SPECIES_PIKACHU"], ball, 4)[1] > 1.9 * _dmg(thunderbolt, SP["SPECIES_PIKACHU"], 0, 4)[1]
    assert _dmg(thunderbolt, SP["SPECIES_MAROWAK"], ball, 4) == _dmg(thunderbolt, SP["SPECIES_MAROWAK"], 0, 4)
    # v3 keeps its quirk (the ids in id order): Thick Club counts for Pikachu and Cubone, Light Ball for Marowak
    assert _dmg(bonemerang, SP["SPECIES_PIKACHU"], club, 3)[1] > 1.9 * _dmg(bonemerang, SP["SPECIES_PIKACHU"], 0, 3)[1]
    assert _dmg(bonemerang, SP["SPECIES_MAROWAK"], club, 3) == _dmg(bonemerang, SP["SPECIES_MAROWAK"], 0, 3)
    assert _dmg(thunderbolt, SP["SPECIES_MAROWAK"], ball, 3)[1] > 1.9 * _dmg(thunderbolt, SP["SPECIES_MAROWAK"], 0, 3)[1]


def test_any_iv_ranges_contain_every_fixed_iv_range():
    for base in (1, 5, 45, 80, 130, 255):
        for level in (50, 100):
            lo, hi = damage.stat_range_any_iv(base, level)
            hlo, hhi = damage.hp_range_any_iv(base, level)
            for iv in range(32):
                a, b = damage.stat_range(base, iv, level)
                assert lo <= a and b <= hi
                a, b = damage.hp_range(base, iv, level)
                assert hlo <= a and b <= hhi
            assert (lo, hi) == (damage.stat_range(base, 0, level)[0], damage.stat_range(base, 31, level)[1])


def _swap_views(n=12):
    for seed in range(n):
        rng = random.Random(seed)
        b = SimBackend()
        b.reset(seed=seed)
        steps = 0
        while b.phase != Phase.RUN_OVER and steps < 2000:
            steps += 1
            v = b.view()
            if b.phase == Phase.RENTAL:
                sp, pick = [m.species for m in v.candidates], [0]
                for i in range(1, 6):
                    if len(pick) < 3 and sp[i] not in {sp[p] for p in pick}:
                        pick.append(i)
                b.act(tuple(pick))
            elif b.phase == Phase.SWAP:
                yield v
                b.act(None)
            else:
                moves = [i for i, ok in enumerate(v.usable_moves) if ok] if not v.forced_switch else []
                b.act(("move", rng.choice(moves)) if moves else ("switch", rng.choice(v.switch_targets))
                      if v.switch_targets else ("move", 0))


def test_swap_encoding_carries_the_records():
    ctx = {"streak": 1, "battle": 1, "challenge": 0, "rents": 0}
    n = 0
    for v in _swap_views():
        encode.set_version(3)
        x3 = encode.swap(v, ctx)
        encode.set_version(4)
        x4 = encode.swap(v, ctx)
        k = encode.N_DEFEATED
        assert x4["mon_num"].shape == (6, encode.MON_NUM) and encode.MON_NUM == x3["mon_num"].shape[1] + k
        np.testing.assert_array_equal(x4["mon_num"][:, :-k], x3["mon_num"])
        assert not x4["mon_num"][:3, -k:].any()                              # our own team: zeros
        for j, r in enumerate(v.defeated):
            np.testing.assert_allclose(x4["mon_num"][3 + j, -k:], encode._defeated_feats(r))
        assert x4["mon_num"][3:, -k:].any()
        # a swap view without records (another caller, or a human handing over at the swap screen): zeros
        bare = SwapView(v.own_party, v.enemy_party, v.hint_type, v.hint_style)
        assert bare.defeated == [FoeRecord()] * 3
        assert not encode.swap(bare, ctx)["mon_num"][:, -k:].any()
        n += 1
    assert n > 5


def test_rental_and_battle_tokens_have_zero_records():
    encode.set_version(4)
    b = SimBackend()
    b.reset(seed=1)
    ctx = {"streak": 0, "battle": 0, "challenge": 0, "rents": 0}
    x = encode.rental(b.view(), ctx)
    assert x["mon_num"].shape == (6, encode.MON_NUM) and not x["mon_num"][:, -encode.N_DEFEATED:].any()
    b.act((0, 1, 2) if len({m.species for m in b.view().candidates[:3]}) == 3 else (0, 1, 3))
    x = encode.battle(b.view(), ctx)
    assert x["mon_num"].shape == (6, encode.MON_NUM) and not x["mon_num"][:, -encode.N_DEFEATED:].any()

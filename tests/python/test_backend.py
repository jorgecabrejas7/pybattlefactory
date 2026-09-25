import random

from pybattle.backend import Phase, SimBackend
from pybattle.view import BattleView, RentalView, SwapView


def random_action(backend, view, rng):
    if backend.phase == Phase.RENTAL:
        while True:
            pick = tuple(rng.sample(range(6), 3))
            species = [view.candidates[i].species for i in pick]
            if len(set(species)) == 3:
                return pick
    if backend.phase == Phase.SWAP:
        if rng.random() < 0.5:
            return None
        own = {m.species for m in view.own_party}
        for _ in range(20):
            p, e = rng.randrange(3), rng.randrange(3)
            others = {m.species for i, m in enumerate(view.own_party) if i != p}
            if view.enemy_party[e].species not in others:
                return (p, e)
        return None
    if backend.phase == Phase.FORCED_SWITCH:
        return ("switch", rng.choice(view.switch_targets))
    moves = [i for i, ok in enumerate(view.usable_moves) if ok]
    if view.switch_targets and (not moves or rng.random() < 0.1):
        return ("switch", rng.choice(view.switch_targets))
    return ("move", rng.choice(moves)) if moves else ("move", 0)


def test_random_runs_complete_and_views_are_consistent():
    rng = random.Random(0)
    phases = set()
    for seed in range(40):
        b = SimBackend()
        b.reset(seed=seed, win_streak=rng.choice([0, 20, 35]))
        steps = 0
        while b.phase != Phase.RUN_OVER:
            v = b.view()
            phases.add(b.phase)
            if b.phase == Phase.RENTAL:
                assert isinstance(v, RentalView) and len(v.candidates) == 6
                assert all(m.level == 100 for m in v.candidates)
            elif b.phase == Phase.SWAP:
                assert isinstance(v, SwapView) and all(m.species for m in v.enemy_party)
            else:
                assert isinstance(v, BattleView)
                assert 0 <= v.enemy_party[v.enemy_active.party_index].hp_pixels <= 48
                if b.phase == Phase.FORCED_SWITCH:
                    assert v.switch_targets and not any(v.usable_moves)
            b.act(random_action(b, v, rng))
            steps += 1
            assert steps < 20000
    assert {Phase.RENTAL, Phase.BATTLE, Phase.FORCED_SWITCH, Phase.SWAP} <= phases


def test_clone_diverges_independently():
    b = SimBackend()
    b.reset(seed=3)
    b.act((0, 1, 2) if len({m.species for m in b.view().candidates[:3]}) == 3 else (0, 1, 3))
    assert b.phase == Phase.BATTLE
    c = b.clone()
    rng_before = c.game.rng
    b.act(("move", b.view().usable_moves.index(True)))
    assert c.game.rng == rng_before

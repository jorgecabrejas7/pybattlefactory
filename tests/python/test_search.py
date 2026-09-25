"""Decision-time search, Python side (rl/search.py, rl/determinize.py, BattleObserver.fast_copy / rebase)."""

import glob
import inspect
import random
import struct
import time

import numpy as np
import pytest

from pybattle.backend import Phase, SimBackend
from pybattle.emu.decode import SB2_RENTAL_MONS, SYMBOLS as S, decode_party, decode_rental_mons
from rl import determinize as D
from rl import search

CKPTS = sorted(glob.glob("runs/ppo_joint_v3/latest.pt")) + sorted(glob.glob("runs/*/latest.pt"))
FLAG_SYS_FACTORY_SILVER = 0x860 + 0x6C


def _random_action(b, view, rng):
    if b.phase == Phase.RENTAL:
        while True:
            pick = tuple(rng.sample(range(6), 3))
            if len({view.candidates[i].species for i in pick}) == 3:
                return pick
    if b.phase == Phase.SWAP:
        if rng.random() < 0.5:
            return None
        for _ in range(20):
            p, e = rng.randrange(3), rng.randrange(3)
            if view.enemy_party[e].species not in {m.species for i, m in enumerate(view.own_party) if i != p}:
                return (p, e)
        return None
    if b.phase == Phase.FORCED_SWITCH:
        return ("switch", rng.choice(view.switch_targets))
    moves = [i for i, ok in enumerate(view.usable_moves) if ok]
    if view.switch_targets and (not moves or rng.random() < 0.15):
        return ("switch", rng.choice(view.switch_targets))
    return ("move", rng.choice(moves)) if moves else ("move", 0)


def _decisions(n_runs, seed=0, streaks=(0, 7, 14, 20, 28, 35), max_steps=400):
    """Battle decisions of random-play Factory runs: yields (backend, view, tracker)."""
    rng = random.Random(seed)
    for r in range(n_runs):
        b = SimBackend(max_turns=10 ** 9)
        streak = streaks[r % len(streaks)]
        b.reset(seed=seed * 1000 + r, win_streak=streak, rents_count=streak // 7)
        if streak >= 21:
            b.game.set_flag(FLAG_SYS_FACTORY_SILVER, True)
        tracker = D.ExclusionTracker()
        steps = 0
        while b.phase != Phase.RUN_OVER and steps < max_steps:
            v = b.view()
            a = _random_action(b, v, rng)
            if b.phase == Phase.RENTAL:
                tracker.on_rental(v)
                tracker.set_own_ids([v.frontier_ids[i] for i in a])
            elif b.phase == Phase.SWAP:
                own = [m.mon_id for m in decode_rental_mons(b.game.read_saveblock2(SB2_RENTAL_MONS, 72))[:3]]
                tracker.on_swap(v, own)
            else:
                yield b, v, tracker
            b.act(a)
            steps += 1


# ---- fast_copy ------------------------------------------------------------------------------------------------

def test_fast_copy_gives_the_same_views_as_copy():
    rng = random.Random(1)
    n = 0
    t_copy = t_fast = 0.0
    for seed in range(12):
        b = SimBackend(max_turns=10 ** 9)
        b.reset(seed=seed, win_streak=7 * (seed % 5))
        chains = []          # (deepcopied observer, fast-copied observer) followed for a few decisions
        steps = 0
        while b.phase != Phase.RUN_OVER and steps < 300:
            v = b.view()
            if b.phase in (Phase.BATTLE, Phase.FORCED_SWITCH):
                forced = b.phase == Phase.FORCED_SWITCH
                args = (b.game, forced, b.game.unusable_moves(0), b.game.can_switch(0))
                for dc, fc in chains:
                    vd, vf = dc.observe(*args), fc.observe(*args)
                    assert vd == vf == v
                    n += 1
                chains = [c for c in chains if rng.random() < 0.7]
                o = b._observer
                t = time.perf_counter(); dc = o.copy(); t_copy += time.perf_counter() - t
                t = time.perf_counter(); fc = o.fast_copy(); t_fast += time.perf_counter() - t
                assert dc.observe(*args) == fc.observe(*args) == v          # same decision: the cached view
                chains.append((dc, fc))
            elif chains:
                chains = []
            b.act(_random_action(b, v, rng))
            steps += 1
    assert n > 500
    speedup = t_copy / t_fast
    print(f"\nfast_copy: {n} views compared; copy {t_copy * 1e6 / n:.0f} us, fast_copy {t_fast * 1e6 / n:.0f} us, "
          f"speedup x{speedup:.1f}")
    assert speedup > 3


def test_fast_copy_does_not_share_mutable_memory():
    for b, v, _ in _decisions(3, seed=4):
        o = b._observer
        f = o.fast_copy()
        f.seen.add(99)
        f.revealed_moves.setdefault(0, []).append(12345)
        for rec in f._sleep.values():
            rec[0] += 7
        f._vol[1]["x"] = 1
        assert 99 not in o.seen and 12345 not in o.revealed_moves.get(0, [])
        assert all(rec[0] < 7 for rec in o._sleep.values()) and "x" not in o._vol[1]


def test_clone_keeps_views():
    rng = random.Random(3)
    for b, v, _ in _decisions(2, seed=5):
        c = b.clone()
        assert c.view() == v
        if rng.random() < 0.2:
            a = _random_action(b, v, rng)
            b2 = b.clone()
            b2.act(a)
            c.act(a)
            if c.phase in (Phase.BATTLE, Phase.FORCED_SWITCH):
                assert c.view() == b2.view()


def test_rebase_on_an_unchanged_state_changes_nothing():
    rng = random.Random(2)
    n = 0
    for b, v, _ in _decisions(4, seed=6):
        forced = b.phase == Phase.FORCED_SWITCH
        f = b._observer.fast_copy()
        f.rebase(b.game, forced)
        assert f.observe(b.game, forced, b.game.unusable_moves(0), b.game.can_switch(0)) == v
        a = _random_action(b, v, rng)
        c = b.clone()
        c.act(a)
        if c.phase in (Phase.BATTLE, Phase.FORCED_SWITCH):
            g = c.game
            vf = f.observe(g, c.phase == Phase.FORCED_SWITCH, g.unusable_moves(0), g.can_switch(0))
            assert vf == c.view()
            n += 1
    assert n > 50


# ---- determinization ------------------------------------------------------------------------------------------

def _true_team(b, ctx):
    """(set id, party mon) of the opponent's team: gFrontierTempParty for a normal trainer; Noland's sets are not
    stored, so they are matched by species and moves in his pool."""
    party = decode_party(b.game.read(S.addr("gEnemyParty"), 300))[:3]
    if not ctx["brain"]:
        ids = struct.unpack("<3H", b.game.read(S.addr("gFrontierTempParty"), 6))
        assert [int(D.SET_SPECIES[i]) for i in ids] == [m.species for m in party]
        return list(zip(ids, party))
    pool = D.pool(ctx["challenge"])
    out = []
    for m in party:
        ids = [int(i) for i in pool if D.SET_SPECIES[i] == m.species and D.FRONTIER_MONS[i]["moves"] == m.moves]
        assert len(ids) == 1
        out.append((ids[0], m))
    return out


def test_pools_follow_the_game():
    for c in range(4):
        p = D.pool(c)
        assert len(p) == 96 and len(set(D.SET_SPECIES[p])) == 96          # rounds 1-4: one set per species
        assert (p[0], p[-1]) == (372 + 96 * c, 467 + 96 * c)
    for c in (4, 5, 6, 7, 12):
        p = D.pool(c)
        assert (p[0], p[-1], len(p)) == (372, 881, 510)
    assert D.fixed_iv(0, False) == 3 and D.fixed_iv(0, True) == 6
    assert D.enemy_iv({"challenge": 5, "battle": 3}) == 3 and D.enemy_iv({"challenge": 5, "battle": 6}) == 6
    assert D.enemy_iv({"challenge": 2, "battle": 6, "brain": 1}) == 15           # Noland, battle 21
    assert D.enemy_iv({"challenge": 5, "battle": 6, "brain": 2}) == 31           # Noland, battle 42


def test_candidates_contain_the_true_sets_and_respect_exclusions():
    rng = random.Random(0)
    n = noland = 0
    for b, v, tracker in _decisions(24, seed=7):
        ri = b.run_info()
        ctx = {"challenge": ri.challenge_num, "battle": ri.battle_in_challenge,
               "brain": b.game.factory_info.brain_status}
        true = _true_team(b, ctx)
        noland += bool(ctx["brain"])
        own = tracker.species | {m.species for m in v.own_party}
        cands = D.candidates(v, ctx, own, tracker.own_ids)
        seen_species = {m.species for m in v.enemy_party if m.seen}
        revealed = {m.revealed_item for m in v.enemy_party if m.seen and m.revealed_item}
        for i, (tid, pm) in enumerate(true):
            assert tid in cands[i], (i, tid, v.enemy_party[i])
            c = cands[i]
            if v.enemy_party[i].seen:
                assert set(D.SET_SPECIES[c]) == {pm.species}
            else:
                assert not set(D.SET_SPECIES[c]) & seen_species
                if not ctx["brain"]:
                    assert not set(D.SET_SPECIES[c]) & own
                assert not set(D.SET_ITEM[c]) & revealed
        # IV and HP of the true team
        assert all(pm.ivs == [D.enemy_iv(ctx)] * 6 for _, pm in true)
        assert all(D.max_hp(tid, pm.ivs[0]) == pm.max_hp for tid, pm in true)
        specs = D.sample_determinization(v, ctx, own, rng, own_ids=tracker.own_ids)
        # fainted Pokemon are kept as they are (they no longer matter); the rest form a valid team with them
        ids = [s[1] for s in specs if s[1] >= 0]
        kept = [true[s[0]][0] for s in specs if s[1] < 0]
        assert len({int(D.SET_SPECIES[i]) for i in ids + kept}) == 3
        items = [int(D.SET_ITEM[i]) for i in ids if D.SET_ITEM[i]]
        assert len(items) == len(set(items))
        for (slot, sid, iv, bit, hp), m in zip(specs, v.enemy_party):
            assert iv == D.enemy_iv(ctx) and bit in (0, 1)
            if m.seen and m.fainted:
                assert sid == hp == D.KEEP
                continue
            assert sid in cands[slot]
            if m.seen:
                mh = D.max_hp(sid, iv)
                hp_int = round(hp * mh)
                from pybattle.view import hp_bar_pixels
                assert hp_bar_pixels(hp_int, mh) == m.hp_pixels or len(
                    [h for h in range(1, mh + 1) if hp_bar_pixels(h, mh) == m.hp_pixels]) == 0
                if m.revealed_ability:
                    a = D.SPECIES[m.species]["abilities"]
                    assert bit == (1 if (a[1] and a[1] != a[0] and m.revealed_ability == a[1]) else 0)
            else:
                assert hp == 1.0
        n += 1
    assert n > 300 and noland > 0


# ---- the perfect-information guard ------------------------------------------------------------------------------

class _FakePolicy:
    net = None
    device = "cpu"
    value_norm = {}


def test_perfect_mode_needs_allow_perfect(monkeypatch):
    monkeypatch.delenv(search.TRAINING_ENV, raising=False)
    with pytest.raises(PermissionError):
        search.SearchBattler(_FakePolicy(), mode="perfect")
    search.SearchBattler(_FakePolicy(), mode="perfect", allow_perfect=True)
    monkeypatch.setenv(search.TRAINING_ENV, "1")
    with pytest.raises(PermissionError):
        search.SearchBattler(_FakePolicy(), mode="perfect", allow_perfect=True)
    with pytest.raises(ValueError):
        search.SearchBattler(_FakePolicy(), mode="cheat", allow_perfect=True)


def test_training_entry_points_mark_training():
    from rl import train, train_rainbow
    for mod in (train, train_rainbow):
        src = inspect.getsource(mod.main)
        assert "mark_training()" in src.split("parse()")[0], mod.__name__


# ---- trees ------------------------------------------------------------------------------------------------------

def _bandit(tree, best=2, sims=300, batch=8):
    tree.reset()
    tree.select(1)
    legal = [True] * 6 + [False]
    tree.expand(0, [1 / 6] * 6 + [0.0], legal)
    tree.backup(0, 0.5)
    done = 1
    while done < sims:
        for leaf, parent, a in tree.select(batch):
            if not tree.is_expanded(leaf):
                v = 1.0 if a == best else 0.2
                tree.set_terminal(leaf, v)
            tree.backup(leaf, 1.0 if a == best else 0.2)
        done += batch
    return np.asarray(tree.root_visits())


@pytest.mark.parametrize("native", [False, True])
def test_tree_finds_the_best_action(native):
    if native and search.NativeMctsTree is None:
        pytest.skip("native MctsTree not built yet")
    tree = search.make_tree(1.5, 1.0, native=native)
    v = _bandit(tree)
    assert v.argmax() == 2 and v[2] > 0.5 * v.sum() and v[6] == 0


# ---- search end to end ----------------------------------------------------------------------------------------

def _policy():
    if not CKPTS:
        pytest.skip("no checkpoint")
    import torch
    from rl.policy import Policy
    torch.set_num_threads(2)
    return Policy(CKPTS[0], torch.device("cpu"))


def _search_decisions(sb, n, seed=0, streak=0):
    from rl.envs import NO_ACTION, FactoryEnv, decode_action
    from rl import encode
    env = FactoryEnv(seed, win_streak=streak)
    ev = env._advance(NO_ACTION)
    done = 0
    while done < n:
        if ev["truncate"]:
            ev = env.step("reset")
            continue
        be = env.backend
        if ev["kind"] == "battle":
            view = be.view()
            a = sb.act_on(be, decisions=env.decisions - 1)
            assert a in view.legal_actions and a != ("forfeit",)
            st = sb.last_stats
            assert len(st["visits"]) == 7 and st["ms"] > 0 and st["errors"] == 0
            if st["searched"]:
                vis = np.asarray(st["visits"])
                mask = encode.battle(view, search.run_ctx(be))["mask"]
                assert vis[~mask].sum() == 0 and vis.sum() > 0
            done += 1
        else:
            a = decode_action(ev["kind"], sb.policy.act(ev["kind"], encode.collate([ev["obs"]], sb.device))[0])
        ev = env.step(a)


def test_perfect_search_runs_end_to_end(monkeypatch):
    monkeypatch.delenv(search.TRAINING_ENV, raising=False)
    pol = _policy()
    for sims in (8, 64):                                   # more simulations must not break anything
        sb = search.SearchBattler(pol, n_sims=sims, n_determinizations=2, batch=8, mode="perfect",
                                  allow_perfect=True, seed=1)
        _search_decisions(sb, 12, seed=11)


def test_search_with_the_python_tree(monkeypatch):
    monkeypatch.delenv(search.TRAINING_ENV, raising=False)
    pol = _policy()
    sb = search.SearchBattler(pol, n_sims=32, n_determinizations=2, batch=8, mode="perfect", allow_perfect=True,
                              native_tree=False)
    _search_decisions(sb, 6, seed=12)


@pytest.mark.skipif(not search.HAS_DETERMINIZE, reason="Gen3Game.determinize not built yet")
def test_legal_search_runs_end_to_end():
    pol = _policy()
    for sims in (16, 128):
        sb = search.SearchBattler(pol, n_sims=sims, n_determinizations=4, batch=16, mode="legal", seed=2)
        _search_decisions(sb, 10, seed=13, streak=28)


@pytest.mark.skipif(not search.HAS_DETERMINIZE, reason="Gen3Game.determinize not built yet")
def test_determinized_root_shows_the_same_view():
    """After determinize + rebase, the player's view at the root is unchanged and the next views never mark an
    unseen enemy as seen."""
    rng = random.Random(9)
    n = 0
    for b, v, tracker in _decisions(6, seed=8, streaks=(28, 35)):
        forced = b.phase == Phase.FORCED_SWITCH
        ri = b.run_info()
        ctx = {"challenge": ri.challenge_num, "battle": ri.battle_in_challenge,
               "brain": b.game.factory_info.brain_status}
        g = b.game.clone()
        g.determinize(D.sample_determinization(v, ctx, tracker.species, rng, own_ids=tracker.own_ids),
                      rng.getrandbits(32))
        o = b._observer.fast_copy()
        o.rebase(g, forced)
        assert o.observe(g, forced, g.unusable_moves(0), g.can_switch(0)) == v
        a = _random_action(b, v, rng)
        kind, idx = (0, a[1]) if a[0] == "move" else (1, a[1])
        d, _ = search.sim_step(g, kind, idx)
        if d in (search.ACTION, search.SWITCH):
            v2 = o.observe(g, d == search.SWITCH, g.unusable_moves(0), g.can_switch(0))
            for i, m in enumerate(v.enemy_party):
                if not m.seen and i != v2.enemy_active.party_index:
                    assert not v2.enemy_party[i].seen
                if m.seen:
                    assert v2.enemy_party[i].species == m.species
                    assert set(m.revealed_moves) <= set(v2.enemy_party[i].revealed_moves)
            n += 1
    assert n > 20


@pytest.mark.skipif(not search.HAS_DETERMINIZE, reason="Gen3Game.determinize not built yet")
def test_identity_determinization_keeps_the_game_and_the_views():
    """Determinizing with the true sets, ability bits and HP (hidden counters kept) and rebasing the observer must
    not change what happens next nor what the player sees next, at BATTLE and FORCED_SWITCH decisions."""
    rng = random.Random(10)
    n = forced_n = differ = 0
    for b, v, tracker in _decisions(10, seed=9, streaks=(0, 14, 28, 35)):
        forced = b.phase == Phase.FORCED_SWITCH
        ri = b.run_info()
        ctx = {"challenge": ri.challenge_num, "battle": ri.battle_in_challenge,
               "brain": b.game.factory_info.brain_status}
        true = _true_team(b, ctx)
        bm = decode_battle_mons(b.game)
        active = struct.unpack("<2H", b.game.read(S.addr("gBattlerPartyIndexes"), 4))[1]
        specs = []
        for i, (tid, pm) in enumerate(true):
            hp = bm.hp if i == active else pm.hp
            specs.append((i, tid, pm.ivs[0], pm.ability_num, hp / pm.max_hp) if hp > 0 else (i, -1, 3, 0, -1.0))
        g = b.game.clone()
        g.determinize(specs, -1)
        o = b._observer.fast_copy()
        o.rebase(g, forced)
        a = _random_action(b, v, rng)
        c = b.clone()
        c.act(a)
        kind, idx = (0, a[1]) if a[0] == "move" else (1, a[1])
        d, _ = search.sim_step(g, kind, idx)
        if c.phase in (Phase.BATTLE, Phase.FORCED_SWITCH) and d in (search.ACTION, search.SWITCH):
            v2 = o.observe(g, d == search.SWITCH, g.unusable_moves(0), g.can_switch(0))
            if v2 != c.view():
                differ += 1
            n += 1
            forced_n += forced
    print(f"\nidentity determinization: {n} next views, {forced_n} from forced switches, {differ} differ")
    assert n > 100 and forced_n > 0 and differ <= 0.02 * n


def decode_battle_mons(game):
    from pybattle.emu.decode import decode_battle_mon
    return decode_battle_mon(game.read(S.addr("gBattleMons") + 0x58, 0x58))

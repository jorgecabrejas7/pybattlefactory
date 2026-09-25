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
            swap = b.phase == Phase.SWAP
            b.act(a)
            if swap:
                tracker.set_own_ids([m.mon_id for m in decode_rental_mons(b.game.read_saveblock2(SB2_RENTAL_MONS, 72))[:3]])
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


def test_legal_sampler_uses_player_knowledge_only():
    """The legal sampler reads the view only (no set list, pool, fixed IVs or true state) and its draws respect
    what the player saw and the visible rules."""
    from pybattle.view import hp_bar_pixels
    assert list(inspect.signature(D.sample_determinization).parameters) == ["view", "ctx", "rng", "open_level"]
    src = inspect.getsource(D.sample_determinization)
    for banned in ("FRONTIER_MONS", "SET_", "pool(", "fixed_iv", "enemy_iv", "gEnemyParty", ".game"):
        assert banned not in src, banned
    rng = random.Random(0)
    n = 0
    species_all = set(D.frontier_species(True))
    for b, v, _ in _decisions(24, seed=7):
        specs = D.sample_determinization(v, None, rng)
        level = D.team_level(v)
        drawn_species, items = [], []
        for (slot, sp, moves, item, ivs, evs, nature, bit, hp), m in zip(specs, v.enemy_party):
            if m.seen and m.fainted:
                assert sp == D.KEEP and hp < 0
                continue
            assert len(moves) == 4 and len(ivs) == 6 and len(evs) == 6
            assert all(0 <= x <= 31 for x in ivs) and all(0 <= x <= 252 for x in evs) and sum(evs) <= 510
            assert 0 <= nature < 25 and bit in (0, 1)
            real_moves = [x for x in moves if x]
            assert real_moves and len(set(real_moves)) == len(real_moves)
            if m.seen:
                assert sp == m.species
                assert set(x for x in m.revealed_moves if x) <= set(real_moves)
                if m.revealed_item:
                    assert item == m.revealed_item
                mh = D.max_hp_of(sp, ivs[0], evs[0], level)
                assert hp_bar_pixels(round(hp * mh), mh) == m.hp_pixels or m.hp_pixels >= 48
            else:
                assert sp in species_all and hp == 1.0
                assert set(real_moves) <= set(D.learnable(sp, level))
            if item:
                items.append(item)
            drawn_species.append(sp)
        assert len(drawn_species) == len(set(drawn_species))
        assert len(items) == len(set(items))
        n += 1
    assert n > 300


def _redrawn_state(game):
    """RAM that the next turn depends on, to compare two roots."""
    regions = [("gBattleMons", 0x58 * 2), ("gEnemyParty", 300), ("gPlayerParty", 300), ("gRngValue", 4),
               ("gRandomTurnNumber", 2), ("gChosenActionByBattler", 2), ("gChosenMoveByBattler", 4),
               ("gBattleCommunication", 2)]
    return b"".join(game.read(S.addr(name), n) for name, n in regions)


@pytest.mark.skipif(not search.HAS_REDRAW_TURN, reason="Gen3Game.redraw_turn not built")
def test_roots_do_not_know_the_opponents_choice():
    """The opponent's AI chooses its action while the player decides. Two games that differ only in that choice
    (the AI rerun with other random numbers) must give the same roots after redraw_turn, and the same outcome
    after the same action; with only a new RNG seed they differ (the leak redraw_turn closes)."""
    rng = random.Random(3)
    n = leak_without_redraw = 0
    chosen = lambda g: g.read(S.addr("gChosenActionByBattler"), 2) + g.read(S.addr("gChosenMoveByBattler"), 4)
    for b, v, _ in _decisions(16, seed=21, streaks=(0, 14, 28)):
        if b.phase != Phase.BATTLE:
            continue
        g1, g2 = b.game.clone(), b.game.clone()
        for g in (g1, g2):
            g.redraw_turn(rng.getrandbits(32))
            assert g.run() == search.ACTION                   # the opponent's AI chooses again, the player waits
        if chosen(g1) == chosen(g2):
            continue
        a = _random_action(b, v, rng)
        kind, idx = (0, a[1]) if a[0] == "move" else (1, a[1])
        seed = rng.getrandbits(32)
        # without redrawing the turn, the pre-chosen action is played
        s1, s2 = g1.clone(), g2.clone()
        s1.set_rng(seed)
        s2.set_rng(seed)
        search.sim_step(s1, kind, idx)
        search.sim_step(s2, kind, idx)
        leak_without_redraw += _redrawn_state(s1) != _redrawn_state(s2)
        # with it, the roots and what follows are the same
        r1, r2 = g1.clone(), g2.clone()
        search.redraw_turn(r1, seed)
        search.redraw_turn(r2, seed)
        assert _redrawn_state(r1) == _redrawn_state(r2)
        o1, o2 = search.sim_step(r1, kind, idx), search.sim_step(r2, kind, idx)
        assert o1 == o2 and _redrawn_state(r1) == _redrawn_state(r2)
        n += 1
    assert n >= 20 and leak_without_redraw > 0.5 * n


@pytest.mark.skipif(not search.HAS_REDRAW_TURN, reason="Gen3Game.redraw_turn not built")
@pytest.mark.parametrize("mode", ["legal", "perfect"])
def test_search_visits_do_not_depend_on_the_opponents_choice(mode, monkeypatch):
    monkeypatch.delenv(search.TRAINING_ENV, raising=False)

    def evaluator(batch):
        x = batch["mon_num"].reshape(len(batch["mask"]), -1).sum(1)
        pri = np.full((len(x), 7), 1 / 7, np.float32)
        return pri, (np.sin(x) * 0.5 + 0.5).astype(np.float32)
    rng = random.Random(4)
    n = 0
    for b, v, _ in _decisions(10, seed=22, streaks=(0, 21)):
        if b.phase != Phase.BATTLE or len(v.legal_actions) < 3:
            continue
        c1, c2 = b.clone(), b.clone()
        for c in (c1, c2):
            c.game.redraw_turn(rng.getrandbits(32))
            assert c.game.run() == search.ACTION
        visits = []
        for c in (c1, c2):
            sb = search.SearchBattler(None, n_sims=48, n_determinizations=4, batch=8, mode=mode,
                                      allow_perfect=True, seed=5, evaluator=evaluator)
            sb.act_on(c)
            visits.append(sb.last_stats["visits"])
        assert visits[0] == visits[1]
        n += 1
        if n >= 12:
            break
    assert n >= 12


def test_legal_determinization_is_full():
    rng = random.Random(6)
    n = 0
    for b, v, _ in _decisions(6, seed=23, streaks=(0, 28)):
        g = b.game.clone()
        assert not g.determinized
        full = g.determinize(D.sample_determinization(v, None, rng), D.hidden_counters(v), rng.getrandbits(32))
        assert full and g.determinized and g.clone().determinized
        n += 1
    assert n > 50


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


def test_perfect_mode_cannot_be_switched_on_later(monkeypatch):
    monkeypatch.delenv(search.TRAINING_ENV, raising=False)
    sb = search.SearchBattler(_FakePolicy(), mode="legal")
    with pytest.raises(AttributeError):
        sb.mode = "perfect"
    sb_perfect = search.SearchBattler(_FakePolicy(), mode="perfect", allow_perfect=True)
    b = next(b for b, v, _ in _decisions(1, seed=30))
    monkeypatch.setenv(search.TRAINING_ENV, "1")
    with pytest.raises(PermissionError):           # built before training was marked: refused when used
        sb_perfect._roots(b, b.view(), None, False)


@pytest.mark.skipif(search.NativeSearcher is None, reason="pybattle_native.Searcher not built")
def test_native_searcher_refuses_true_state_roots_in_training(monkeypatch):
    b = next(b for b, v, _ in _decisions(1, seed=31))
    ctx = search.run_ctx(b)
    ev = lambda batch: (np.full((len(batch["mask"]), 7), 1 / 7, np.float32),
                        np.full(len(batch["mask"]), 0.5, np.float32))
    s = search.NativeSearcher(8, 4, 1.5, 1.0, 300)
    monkeypatch.setenv(search.TRAINING_ENV, "1")
    with pytest.raises(PermissionError):
        s.search([(b.game.clone(), b._observer.fast_copy(), 0)], ctx, [1 / 7] * 7, [True] * 7, 0.5, ev)
    g = b.game.clone()
    v = b.view()
    g.determinize(D.sample_determinization(v, None, random.Random(0)), D.hidden_counters(v), 1)
    o = b._observer.fast_copy()
    o.rebase(g, b.phase == Phase.FORCED_SWITCH)
    s.search([(g, o, 0)], ctx, [1 / 7] * 7, [True] * 7, 0.5, ev)      # a determinized root is fine


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
        truth = (b.game.unusable_moves(0), b.game.can_switch(0))
        for _ in range(8):                      # as SearchBattler: redraw what would change the player's options
            g = b.game.clone()
            g.determinize(D.sample_determinization(v, ctx, rng), D.hidden_counters(v), rng.getrandbits(32))
            if forced or (g.unusable_moves(0), g.can_switch(0)) == truth:
                break
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
            specs.append(D.set_spec(i, tid, pm.ivs[0], pm.ability_num, hp / pm.max_hp) if hp > 0
                         else (i, D.KEEP, [0] * 4, 0, [0] * 6, [0] * 6, 0, 0, -1.0))
        g = b.game.clone()
        g.determinize(specs, None, -1)
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

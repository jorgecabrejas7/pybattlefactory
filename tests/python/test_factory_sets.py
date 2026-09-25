"""The training-time opponent sampler "factory_sets" (rl/determinize.py sample_factory_sets, docs/RL_DECISIONS.md §18)
and the simulated swap screens of the tactician's search (rl/tactician_search.py next_tactician_obs)."""

import inspect
import random
import struct

import numpy as np
import pytest
import torch

from pybattle.backend import OpponentKnowledge, Phase, SimBackend
from pybattle.emu.decode import SYMBOLS as S, decode_party
from pybattle.view import BattleObserver, hp_bar_pixels
from rl import alphazero as AZ
from rl import determinize as D
from rl import encode, search
from rl import tactician_search as TS
from rl.model import FactoryNet

from test_alphazero import _battle_to_swap
from test_search import _decisions, _random_action


@pytest.fixture(autouse=True)
def _keep_encode_version():
    v = encode.VERSION
    yield
    encode.set_version(v)


@pytest.fixture
def training(monkeypatch):
    monkeypatch.setenv(search.TRAINING_ENV, "1")


def _net(seed=0, version=4):
    torch.manual_seed(seed)
    encode.set_version(version)
    return FactoryNet(share="embeddings").eval()


def _tactician_decision(kind, seed, streak=7):
    rng = random.Random(seed)
    for s in range(seed, seed + 80):
        b = SimBackend(max_turns=10 ** 9)
        b.reset(seed=s, win_streak=streak, rents_count=streak // 7)
        if kind == "swap" and not _battle_to_swap(b, rng):
            continue
        ctx = search.run_ctx(b)
        v = b.view()
        return b, (encode.rental(v, ctx) if kind == "rental" else encode.swap(v, ctx))
    raise RuntimeError(f"no {kind} decision reached")


# ---- the policy lock ------------------------------------------------------------------------------------------------

def test_factory_sets_is_training_only(monkeypatch):
    monkeypatch.delenv(search.TRAINING_ENV, raising=False)
    assert not search.in_training()
    from types import SimpleNamespace
    from pybattle.view import SeenMon
    view = SimpleNamespace(enemy_party=[SeenMon() for _ in range(3)], own_party=[SimpleNamespace(level=100)])
    with pytest.raises(PermissionError):
        D.determinization(view, OpponentKnowledge(0), random.Random(0), True, "factory_sets")
    with pytest.raises(ValueError):
        D.check_prior("oracle")
    ev = AZ.local_battler_evaluator(_net(), None)
    with pytest.raises(PermissionError):
        search.SearchBattler(None, n_sims=8, n_determinizations=2, evaluator=ev, opponent_prior="factory_sets")
    with pytest.raises(PermissionError):
        AZ.AZBattler(ev, n_sims=8, n_determinizations=2, opponent_prior="factory_sets")
    with pytest.raises(PermissionError):
        TS.TacticianSearch(ev, None, bootstrap=False, opponent_prior="factory_sets")
    b, obs = _tactician_decision("rental", 3)
    with pytest.raises(PermissionError):
        TS.start_simulated_battle(b, "rental", TS.options_of("rental", obs)[0], random.Random(0), "factory_sets")
    # defaults: the strict sampler everywhere except alphazero's training search
    assert inspect.signature(search.SearchBattler).parameters["opponent_prior"].default == "strict"
    assert inspect.signature(TS.TacticianSearch).parameters["opponent_prior"].default == "strict"
    assert search.SearchBattler(None, n_sims=8, n_determinizations=2, evaluator=ev).opponent_prior == "strict"
    assert AZ.parse([]).opponent_prior == "factory_sets"
    assert AZ.parse(["--opponent-prior", "strict"]).opponent_prior == "strict"
    assert '"opponent_prior": "strict"' in inspect.getsource(AZ.evaluate_rounds)
    import rl.eval_rounds
    import rl.evaluate
    for mod in (rl.eval_rounds, rl.evaluate):
        assert "factory_sets" not in inspect.getsource(mod)
    # once constructed in training, leaving training is refused at search time too
    monkeypatch.setenv(search.TRAINING_ENV, "1")
    sb = search.SearchBattler(None, n_sims=8, n_determinizations=2, evaluator=ev, opponent_prior="factory_sets")
    monkeypatch.delenv(search.TRAINING_ENV)
    b.act(next(t for t in __import__("itertools").combinations(range(6), 3)
               if len({b.view().candidates[i].species for i in t}) == 3))
    with pytest.raises(PermissionError):
        sb.act_on(b)


# ---- what the player knows of the opponent's generation --------------------------------------------------------------

def _true_ids(b):
    return list(struct.unpack("<3H", b.game.read(S.addr("gFrontierTempParty"), 6)))


def test_opponent_knowledge_matches_the_games_generation():
    """The real opponent (read here only to check) was drawn from exactly the pool the knowledge describes."""
    n = noland = 0
    last_battle = None
    for b, v, _ in _decisions(80, seed=5, streaks=(0, 7, 14, 20, 28, 35, 41)):
        key = (id(b), b.run_info().win_streak)
        if key == last_battle:
            continue
        last_battle = key
        k = b.opponent_knowledge
        info = b.run_info()
        assert k is not None and k.challenge == info.challenge_num and k.noland == info.noland
        party = decode_party(b.game.read(S.addr("gEnemyParty"), 300))[:3]
        ids = D.factory_pool(k, b.open_level)
        if not k.noland:
            real = _true_ids(b)
            assert [int(D.SET_SPECIES[i]) for i in real] == [m.species for m in party]
            assert all(i in ids for i in real), (real, k)
            assert 3 <= len(k.species) <= 6
        else:
            noland += 1
            assert 3 <= len(k.set_ids) <= 6 and not k.species
            for m in party:
                assert any(int(D.SET_SPECIES[i]) == m.species and set(D.SET_MOVES_AS_BUILT[i]) == set(m.moves)
                           for i in ids), m.species
        n += 1
    assert n > 60 and noland >= 1


def test_opponent_knowledge_never_reads_the_real_team():
    src = inspect.getsource(SimBackend.opponent_knowledge_for)
    assert "gFrontierTempParty" not in src
    b, _ = _tactician_decision("swap", 21)
    c = b.clone()
    c.game.write(S.addr("gFrontierTempParty"), struct.pack("<3H", 1, 2, 3))
    for act in (None, (0, 1), (2, 0)):
        assert b.opponent_knowledge_for(act) == c.opponent_knowledge_for(act)


# ---- the sampler ---------------------------------------------------------------------------------------------------------

def test_factory_sets_sampler_respects_what_the_player_saw(training):
    rng = random.Random(1)
    n = 0
    D.factory_stats.update(seen_slots=0, fallback=0)
    for b, v, _ in _decisions(30, seed=11, streaks=(0, 7, 14, 20, 28, 35, 41)):
        k = b.opponent_knowledge
        pool = set(int(i) for i in D.factory_pool(k, b.open_level))
        specs = D.determinization(v, k, rng, b.open_level, "factory_sets")
        level = D.team_level(v)
        species, items = [], []
        for (slot, sp, moves, item, ivs, evs, nature, bit, hp), m in zip(specs, v.enemy_party):
            if m.seen and m.fainted:
                assert sp == D.KEEP
                continue
            assert all(0 <= x <= 31 for x in ivs) and all(0 <= x <= 252 for x in evs) and sum(evs) <= 510
            assert 0 <= nature < 25 and bit in (0, 1)
            sets = [i for i in pool if int(D.SET_SPECIES[i]) == sp and list(D.SET_MOVES_AS_BUILT[i]) == moves
                    and int(D.SET_ITEM[i]) == item]
            if m.seen:
                assert sp == m.species
                assert set(x for x in m.revealed_moves if x) <= set(moves)
                if m.revealed_item:
                    assert item == m.revealed_item
                mh = D.max_hp_of(sp, ivs[0], evs[0], level)
                assert hp_bar_pixels(round(hp * mh), mh) == m.hp_pixels or m.hp_pixels >= 48
            else:
                assert sets, "an unseen Pokemon is a set of the pool"
                assert hp == 1.0
            species.append(sp)
            if item:
                items.append(item)
        assert len(species) == len(set(species)) and len(items) == len(set(items))
        n += 1
    assert n > 300
    assert D.factory_stats["seen_slots"] > 300
    assert D.factory_stats["fallback"] <= 0.05 * D.factory_stats["seen_slots"], D.factory_stats


def test_factory_sets_battler_search_never_reads_the_real_team(training):
    """The battler's roots use backend.opponent_knowledge (screens), never gFrontierTempParty."""
    ev = AZ.local_battler_evaluator(_net(3), None)
    rng = random.Random(4)
    b = SimBackend(max_turns=10 ** 9)
    b.reset(seed=8, win_streak=14, rents_count=2)
    b.act(_random_action(b, b.view(), rng))
    assert b.phase in (Phase.BATTLE, Phase.FORCED_SWITCH)
    c = b.clone()
    c.game.write(S.addr("gFrontierTempParty"), struct.pack("<3H", 5, 6, 7))
    out = []
    for backend in (b, c):
        sb = AZ.AZBattler(ev, n_sims=32, n_determinizations=4, seed=2, opponent_prior="factory_sets")
        out.append(sb.search_root(backend, 0, noise=False))
    assert np.array_equal(out[0]["visits"], out[1]["visits"]) and np.allclose(out[0]["q"], out[1]["q"])


@pytest.mark.parametrize("kind", ["rental", "swap"])
def test_factory_sets_tactician_search_never_reads_the_real_team(kind, training):
    net = _net(4)
    ev = AZ.local_battler_evaluator(net, None)
    b, obs = _tactician_decision(kind, 11)
    b2 = b.clone()
    addr = S.addr("gFrontierTempParty")
    ids = struct.unpack("<3H", b2.game.read(addr, 6))
    b2.game.write(addr, struct.pack("<3H", *[(i + 37 * (j + 1)) % 800 + 1 for j, i in enumerate(ids)]))
    opts = TS.options_of(kind, obs)
    s1 = TS.start_simulated_battle(b, kind, opts[-1], random.Random(5), "factory_sets")
    s2 = TS.start_simulated_battle(b2, kind, opts[-1], random.Random(5), "factory_sets")
    assert s1.game.read(S.addr("gEnemyParty"), 600) == s2.game.read(S.addr("gEnemyParty"), 600)
    assert s1.opponent_knowledge == s2.opponent_knowledge
    # the simulated team is made of sets of the round's pool
    enemy = decode_party(s1.game.read(S.addr("gEnemyParty"), 300))[:3]
    pool = D.factory_pool(s1.opponent_knowledge, b.open_level)
    for m in enemy:
        assert any(int(D.SET_SPECIES[i]) == m.species and list(D.SET_MOVES_AS_BUILT[i]) == m.moves
                   and int(D.SET_ITEM[i]) == m.held_item for i in pool)
    logp = np.log(np.full(len(opts), 1.0 / len(opts)))
    res = []
    for backend in (b, b2):
        ts = TS.TacticianSearch(ev, lambda k, o: np.full(len(o), 1.5), budget=48, max_considered=6, seed=7,
                                opponent_prior="factory_sets")
        res.append(ts.search(backend, kind, obs, logp, noise=True))
    assert res[0]["visits"].sum() > 0
    assert np.array_equal(res[0]["visits"], res[1]["visits"])
    assert np.allclose(res[0]["q"], res[1]["q"], equal_nan=True)


# ---- the simulated swap screen shows what the player saw --------------------------------------------------------------

def _greedy(ev, x):
    pri, _ = ev({k: np.stack([x[k]]) for k in x})
    return int(np.argmax(np.where(x["mask"], pri[0], -1.0)))


def _replay_with_the_backend(be, ev):
    """Play the battle of a simulated start `be` through SimBackend's own loop (Python observer) with the same
    greedy policy as BattleSimulator. -> the backend at the next phase."""
    be._observer = BattleObserver(*TS.NO_HINT)
    be._observer_done = None
    be._sync()
    n = 0
    while be.phase in (Phase.BATTLE, Phase.FORCED_SWITCH):
        v = be.view()
        forced = be.phase == Phase.FORCED_SWITCH
        if not forced and not any(v.usable_moves) and not v.switch_targets:
            be.act(("move", 0))
        else:
            a = _greedy(ev, encode.battle(v, search.run_ctx(be)))
            be.act(("move", a) if a < 4 else ("switch", a - 4))
        n += 1
        assert n < 300
    return be


@pytest.mark.parametrize("cpp_obs", [True, False])
def test_simulated_swap_screen_matches_the_real_one(cpp_obs):
    """After a won simulated battle, the swap observation used for the tactician's bootstrap is the one SimBackend
    shows after the same battle: revealed moves / items / abilities and the v4 defeated-foe records."""
    if cpp_obs and TS.NativeObsMemory is None:
        pytest.skip("no C++ observer")
    net = _net(6, version=4)
    ev = AZ.local_battler_evaluator(net, None)
    sim = TS.BattleSimulator(ev, cpp_obs=cpp_obs)
    checked = records = reveals = 0
    for seed in range(20):
        b, obs = _tactician_decision("rental", 40 + seed, streak=0)
        for j, opt in enumerate(TS.options_of("rental", obs)[:6]):
            be = TS.start_simulated_battle(b, "rental", opt, random.Random(seed * 100 + j))
            be2 = be.clone()
            s = TS._Sim(be, 0, sim.cpp)
            sim.run([s])
            if not s.won or s.trunc:
                continue
            kind, x = TS.next_tactician_obs(s.be, s.obs)
            if kind != "swap":
                continue
            be2 = _replay_with_the_backend(be2, ev)
            assert be2.phase == Phase.SWAP
            y = encode.swap(be2.view(), search.run_ctx(be2))
            assert x.keys() == y.keys()
            for key in x:
                assert np.array_equal(x[key], y[key]), key
            done = be2._observer
            rec = [list(r) for r in (s.obs.records if cpp_obs else s.obs._records)]
            assert rec == done._records
            records += any(r[2] > 0 for r in rec)
            reveals += any(done.revealed_moves.values())
            checked += 1
            if checked >= 4:
                break
        if checked >= 4:
            break
    assert checked >= 2 and records == checked and reveals >= 1

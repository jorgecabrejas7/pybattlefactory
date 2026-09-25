"""The C++ search loop (src/gen3/search.cpp, pybattle_native.Searcher) against the Python one (rl/search.py):
same seeds and evaluator -> same root visits and the same action."""

import glob

import numpy as np
import pytest

from pybattle.backend import Phase
from rl import search
from test_search import _decisions

pytestmark = pytest.mark.skipif(search.NativeSearcher is None or not search.HAS_DETERMINIZE,
                                reason="pybattle_native.Searcher / Gen3Game.determinize not built")

CKPTS = sorted(glob.glob("runs/ppo_joint_v3/latest.pt"))
OBSERVERS = ["python"] + (["cpp"] if search.NativeObsMemory is not None else [])


def fake_evaluator(batch):
    """A deterministic per-sample function of the observation (no network): priors and values that depend on
    everything the encoder writes, so a wrong observation / order shows up in the visits."""
    n = len(batch["mask"])
    feats = [batch["mon_num"].reshape(n, -1).astype(np.float64), batch["move_num"].reshape(n, -1),
             batch["ctx_num"].reshape(n, -1), batch["mon_ids"].reshape(n, -1) % 97, batch["ctx_ids"] % 89,
             batch["mask"].astype(np.float64), batch["active"][:, None]]
    x = np.concatenate([np.asarray(f, np.float64) for f in feats], 1)
    h = x @ np.cos(np.arange(x.shape[1]) * 0.37)
    lg = 2.0 * np.sin(h[:, None] * np.arange(1, 8) * 0.013 + np.arange(7))
    p = np.exp(lg)
    p /= p.sum(1, keepdims=True)
    v = 1 / (1 + np.exp(-np.sin(h * 0.01)))
    return p.astype(np.float32), v.astype(np.float32)


def _pair(observer="python", evaluator=fake_evaluator, policy=None, **kw):
    kw.setdefault("seed", 5)
    py = search.SearchBattler(policy, impl="python", evaluator=evaluator, **kw)
    cpp = search.SearchBattler(policy, impl="cpp", observer=observer, evaluator=evaluator, **kw)
    return py, cpp


def _compare(py, cpp, runs, seed, streaks, limit, decisions=lambda b: 0):
    """The C++ observer's floats agree with the Python encoder within 1e-6 (tests/python/test_observer_cpp.py), so
    with it Q agrees within float tolerance; with the Python observer everything is bit-identical."""
    q_tol = 1e-6 if cpp.observer == "python" else 1e-4
    n = searched = 0
    for b, v, tracker in _decisions(runs, seed=seed, streaks=streaks):
        d = decisions(b)
        a_py = py.act_on(b, tracker, d)
        a_cpp = cpp.act_on(b, tracker, d)
        sp, sc = py.last_stats, cpp.last_stats
        assert sc["impl"] == "cpp" and sp["impl"] == "python"
        assert sp["visits"] == sc["visits"], (n, sp["visits"], sc["visits"])
        np.testing.assert_allclose(sp["q"], sc["q"], atol=q_tol)
        assert (sp["leaves"], sp["nodes"], sp["errors"]) == (sc["leaves"], sc["nodes"], sc["errors"])
        if a_py != a_cpp:           # only on an exact tie of visits + prior tie-break
            s = np.asarray(sp["visits"]) + 1e-6 * np.asarray(sp["prior"])
            assert abs(s[sp["action"]] - s[sc["action"]]) < 1e-9
        searched += sp["searched"]
        n += 1
        if n >= limit:
            break
    return n, searched


@pytest.mark.parametrize("observer", OBSERVERS)
def test_same_visits_as_python_legal(observer):
    py, cpp = _pair(observer, n_sims=32, n_determinizations=4, batch=16)
    n, searched = _compare(py, cpp, 40, seed=21, streaks=(0, 14, 28, 35), limit=300)
    print(f"\nlegal, observer={observer}: {n} decisions ({searched} searched) identical; "
          f"ms/decision python {py.total_ms / n:.1f}, cpp {cpp.total_ms / n:.1f}")
    assert n >= 300 and searched > 200


@pytest.mark.parametrize("observer", OBSERVERS)
def test_same_visits_as_python_256_sims(observer, monkeypatch):
    monkeypatch.delenv(search.TRAINING_ENV, raising=False)
    for mode in ("legal", "perfect"):
        py, cpp = _pair(observer, n_sims=256, n_determinizations=8, batch=32, mode=mode, allow_perfect=True)
        n, searched = _compare(py, cpp, 6, seed=22, streaks=(7, 21, 42), limit=25)
        assert n == 25 and searched > 15


@pytest.mark.parametrize("observer", OBSERVERS)
def test_truncation_keeps_the_network_value(observer):
    """At max_decisions - 1 every child of a root is truncated (or the battle ends): terminal after one
    evaluation, so the trees cannot grow deeper, and both implementations agree."""
    lim = 20
    py, cpp = _pair(observer, n_sims=64, n_determinizations=4, batch=16, max_decisions=lim)
    n, searched = _compare(py, cpp, 4, seed=23, streaks=(0, 28), limit=40, decisions=lambda b: lim - 1)
    assert searched > 20
    py2, cpp2 = _pair(observer, n_sims=64, n_determinizations=4, batch=16, max_decisions=lim)
    for b, v, tracker in _decisions(3, seed=24):
        cpp2.act_on(b, tracker, lim - 1)
        st = cpp2.last_stats
        if st["searched"]:
            assert st["nodes"] <= 4 * 8                 # root + at most 7 children per tree
            assert st["leaves"] <= 4 * 7
            assert sum(st["visits"]) == 4 * (64 // 4) - 4
        # two decisions before the limit the trees grow one level deeper at most
        cpp2.act_on(b, tracker, lim - 2)
        assert cpp2.last_stats["nodes"] <= 4 * (1 + 7 + 49)


def test_battle_end_leaves_are_terminal():
    """Near the end of battles some leaves end the battle (won 1 / lost 0): they are never evaluated again, so
    the evaluated leaves are fewer than the simulations, and the Python loop agrees."""
    py, cpp = _pair("python", n_sims=64, n_determinizations=2, batch=8)
    fewer = 0
    for b, v, tracker in _decisions(6, seed=25, streaks=(0,)):
        own_alive = sum(m.hp > 0 for m in v.own_party[:3])
        foe_alive = sum(not (m.seen and m.fainted) for m in v.enemy_party[:3])
        if own_alive + foe_alive > 3:
            continue
        py.act_on(b, tracker)
        cpp.act_on(b, tracker)
        sp, sc = py.last_stats, cpp.last_stats
        assert sp["visits"] == sc["visits"] and sp["leaves"] == sc["leaves"]
        if sc["searched"]:
            assert all(0.0 <= q <= 1.0 for q in sc["q"])
            fewer += sc["leaves"] < sum(sc["visits"]) - 2
    assert fewer > 3


@pytest.mark.parametrize("observer", OBSERVERS)
def test_no_state_leaks(observer):
    """The roots, the backend and the observer are only cloned from; identical roots give identical trees (no
    state shared between trees or nodes); a search repeated on the same roots gives the same result."""
    sb = search.SearchBattler(None, impl="cpp", observer=observer, evaluator=fake_evaluator, n_determinizations=1,
                              seed=7)
    n = 0
    for b, v, tracker in _decisions(4, seed=26, streaks=(0, 28)):
        x = search.encode.battle(v, search.run_ctx(b))
        if x["mask"].sum() <= 1:
            continue
        pri, val = fake_evaluator({k: np.asarray(x[k])[None] for k in x})
        (g, obs), = sb._roots(b, v, tracker, b.phase == Phase.FORCED_SWITCH, native_obs=observer == "cpp")
        ram = b.game.read(search.S.addr("gBattleMons"), 0x58 * 2) + b.game.read(search.S.addr("gEnemyParty"), 600)
        groot = g.read(search.S.addr("gBattleMons"), 0x58 * 2) + g.read(search.S.addr("gEnemyParty"), 600)
        rng = g.rng
        ctx = search.run_ctx(b)
        args = (ctx, pri[0].tolist(), x["mask"].tolist(), float(val[0]), fake_evaluator)
        one = search.NativeSearcher(8, 4).search([(g, obs, 0)], *args)
        four = search.NativeSearcher(32, 16).search([(g, obs, 0)] * 2 + [(g.clone(), obs.copy() if observer ==
                                                     "cpp" else obs.fast_copy(), 0)] * 2, *args)
        again = search.NativeSearcher(8, 4).search([(g, obs, 0)], *args)
        assert four["visits"] == [4 * x for x in one["visits"]]
        np.testing.assert_allclose(four["q"], one["q"], atol=1e-6)
        assert again["visits"] == one["visits"] and again["q"] == one["q"]
        assert g.rng == rng and g.read(search.S.addr("gBattleMons"), 0x58 * 2) + \
            g.read(search.S.addr("gEnemyParty"), 600) == groot
        assert b.game.read(search.S.addr("gBattleMons"), 0x58 * 2) + b.game.read(search.S.addr("gEnemyParty"),
                                                                                   600) == ram
        assert b.view() == v
        n += 1
        if n >= 40:
            break
    assert n >= 40


def test_evaluator_batches_match_encode_collate():
    """The evaluator gets the stacked encode.battle arrays (same keys, dtypes and shapes as encode.collate)."""
    seen = []

    def ev(batch):
        seen.append({k: (v.dtype, v.shape, v.flags["C_CONTIGUOUS"]) for k, v in batch.items()})
        return fake_evaluator(batch)

    sb = search.SearchBattler(None, impl="cpp", observer=OBSERVERS[-1], evaluator=ev, n_sims=16,
                              n_determinizations=2, batch=8)
    for b, v, tracker in _decisions(1, seed=27):
        ref = search.encode.battle(v, search.run_ctx(b))
        seen.clear()
        sb.act_on(b, tracker)
        for s in seen[1:]:                              # [0]: the root, stacked in Python
            assert set(s) == set(ref)
            for k, (dt, shape, contig) in s.items():
                assert dt == np.asarray(ref[k]).dtype and shape[1:] == np.asarray(ref[k]).shape and contig
        if len(seen) > 1:
            break
    assert len(seen) > 1


@pytest.mark.skipif(not CKPTS, reason="no checkpoint")
@pytest.mark.parametrize("observer", OBSERVERS)
def test_same_visits_with_the_network(observer):
    import torch
    from rl.policy import Policy
    torch.set_num_threads(1)
    pol = Policy(CKPTS[0], torch.device("cpu"))
    py = search.SearchBattler(pol, impl="python", n_sims=64, n_determinizations=4, batch=16, seed=9)
    cpp = search.SearchBattler(pol, impl="cpp", observer=observer, n_sims=64, n_determinizations=4, batch=16,
                               seed=9)
    n, searched = _compare(py, cpp, 4, seed=28, streaks=(14, 35), limit=40)
    assert n == 40 and searched > 25


def test_impl_selection_and_perfect_guard(monkeypatch):
    monkeypatch.delenv(search.TRAINING_ENV, raising=False)
    assert search.SearchBattler(None, evaluator=fake_evaluator).impl == "cpp"
    assert search.SearchBattler(None, evaluator=fake_evaluator, native_tree=False).impl == "python"
    with pytest.raises(ValueError):
        search.SearchBattler(None, evaluator=fake_evaluator, impl="rust")
    with pytest.raises(ValueError):
        search.SearchBattler(None, evaluator=fake_evaluator, impl="cpp", native_tree=False)
    for impl in ("cpp", "python"):
        with pytest.raises(PermissionError):
            search.SearchBattler(None, evaluator=fake_evaluator, impl=impl, mode="perfect")
        monkeypatch.setenv(search.TRAINING_ENV, "1")
        with pytest.raises(PermissionError):
            search.SearchBattler(None, evaluator=fake_evaluator, impl=impl, mode="perfect", allow_perfect=True)
        monkeypatch.delenv(search.TRAINING_ENV)

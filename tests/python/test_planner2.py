"""The improved planner (rl/planner2.py, docs/RL_DECISIONS.md §27.x): Noland's rules, option switches, common random
numbers and determinism."""

import os
import random
import sys

import numpy as np
import pytest

from pybattle.backend import Phase, SimBackend, factory_brain_status, factory_symbols_for_streak
from pybattle.emu.decode import SYMBOLS as S, decode_party
from rl import determinize as DET
from rl import planner2 as P2
from rl import team_eval as T

sys.path.insert(0, os.path.dirname(__file__))
from test_team_eval_v3 import _FakeW, _decision, _weaken_and_win  # noqa: E402

OPP_MODEL = "runs/opp_model_v3/model.pt"
needs_model = pytest.mark.skipif(not os.path.exists(OPP_MODEL), reason=f"{OPP_MODEL} not present")


@pytest.fixture(scope="module")
def model():
    from rl.opponent_model import ExperienceModel
    return ExperienceModel.load(OPP_MODEL)


# ---- Noland's rules ----------------------------------------------------------------------------------------------

def test_noland_schedule_and_iv():
    # battle 7 of rounds 3 and 6 (battles 21 and 42 of the streak), then every 21 battles; nowhere else
    for r in range(1, 13):
        want = r % 3 == 0
        assert (T.battle_types(r)[6] == T.NOLAND) == want, r
        assert all(t == T.NORMAL for t in T.battle_types(r)[:6])
    assert [s + 1 for s in range(80) if factory_brain_status(s, factory_symbols_for_streak(s))] == [21, 42, 63]
    # his IVs (GetFactoryMonFixedIV(challengeNum + 2, FALSE)): 15 in round 3, 31 from round 6
    assert P2.noland_iv(2) == 15 and P2.noland_iv(5) == 31 and P2.noland_iv(8) == 31


def _pre_noland_swap(streak=19, seeds=range(1, 60)):
    rng = random.Random(0)
    for s in seeds:
        b = SimBackend(max_turns=10 ** 9)
        b.reset(seed=s, win_streak=streak, rents_count=2)
        b.act(T.rental_pick(rng.choice(T.legal_rentals(b.view()))))
        _weaken_and_win(b)
        if b.phase == Phase.SWAP and b.run_info().noland:
            return b
    raise RuntimeError("no swap screen before Noland reached")


@pytest.mark.parametrize("streak,challenge", [(19, 2), (40, 5)])
def test_noland_iv_matches_the_game(streak, challenge):
    """The game's Noland (read here only to check the rule) has every IV = noland_iv, and none of the sets excluded
    by the swap screen (our team after the swap + the defeated team)."""
    b = _pre_noland_swap(streak)
    know = b.opponent_knowledge_for(None)
    assert know.noland and know.challenge == challenge
    c = b.clone()
    c.act(None)                                 # keep: the game generates Noland's team
    foe = decode_party(c.game.read(S.addr("gEnemyParty"), 300))[:3]
    assert all(iv == P2.noland_iv(challenge) for m in foe for iv in m.ivs)
    assert len({m.species for m in foe}) == 3
    # the excluded sets are exactly the 6 Pokemon on the swap screen; giving one away makes its set possible
    k2 = b.opponent_knowledge_for((0, 1))
    assert len(know.set_ids) == 6 and len(k2.set_ids) == 5 and k2.set_ids < know.set_ids


def test_noland_exclusion_of_the_swap_screen():
    b = _pre_noland_swap()
    v = b.view()
    own0, seen0 = P2.noland_exclusion(v, 0)
    assert len(own0) == 3 and len(seen0) == 3
    own4, seen4 = P2.noland_exclusion(v, 4)           # own slot 1 for the defeated slot 0: slot 1 given away
    assert own4 == [own0[0], own0[2]] and seen4 == seen0


@needs_model
def test_noland_draws_follow_his_rules(model):
    s = P2.OpponentSampler("experience", model)
    g = np.random.default_rng(0)
    t = model.compile()
    tab = t.joint_table(5, frozenset(), 100, set())
    # exclude the most likely known configuration (as if it were ours) and a whole seen species' consistent ones
    known = [i for i, c in enumerate(tab["cfg"]) if c is not None]
    top = max(known, key=lambda i: tab["w"][i])
    sp_top = int(tab["sp"][top])
    own = [(sp_top, frozenset(int(x) for x in tab["cfg"][top][0] if x), int(tab["cfg"][top][1]))]
    d = s.noland(5, 3000, g, lambda tb: P2.exclusion_mask(tb, own, []), 100)
    assert len(d) == 3000
    assert (d.species[:, 0] != d.species[:, 1]).all() and (d.species[:, 1] != d.species[:, 2]).all() \
        and (d.species[:, 0] != d.species[:, 2]).all()
    for a, c in ((0, 1), (0, 2), (1, 2)):
        assert ((d.item[:, a] == 0) | (d.item[:, a] != d.item[:, c])).all()
    cfgs = {(int(d.species[i, j]), frozenset(int(x) for x in d.moves[i, j] if x), int(d.item[i, j]))
            for i in range(len(d)) for j in range(3)}
    assert own[0] not in cfgs
    # the species itself is not excluded (Noland excludes sets, not species), when it has other configurations
    others = [i for i in known if int(tab["sp"][i]) == sp_top and i != top]
    if others:
        assert sp_top in set(d.species.ravel().tolist())
    # without the exclusion that configuration does appear
    d0 = s.noland(5, 3000, np.random.default_rng(0), None, 100)
    assert own[0] in {(int(d0.species[i, j]), frozenset(int(x) for x in d0.moves[i, j] if x), int(d0.item[i, j]))
                      for i in range(len(d0)) for j in range(3)}


def test_noland_factory_sets_draws_exclude_set_ids(monkeypatch):
    monkeypatch.setenv("PYBATTLE_DIAGNOSTIC_PRIOR", "factory_sets")
    s = P2.OpponentSampler("factory_sets")
    pool = DET.pool(2)
    excl = set(int(x) for x in pool[:20])
    d = s.noland(2, 500, np.random.default_rng(1), excl)
    assert not (set(d.set_ids.ravel().tolist()) & excl)
    assert set(d.set_ids.ravel().tolist()) <= set(pool.tolist())


def test_factory_sets_needs_the_diagnostic_flag(monkeypatch):
    monkeypatch.delenv("PYBATTLE_DIAGNOSTIC_PRIOR", raising=False)
    with pytest.raises(PermissionError):
        P2.OpponentSampler("factory_sets")


# ---- option switches ---------------------------------------------------------------------------------------------

def test_options_parse_and_tag():
    c = P2.parse_options("n=32,h=1,d=2,nol=mc,p=fs,ns=16,nw=4")
    assert (c.n_opp, c.hint_next, c.depth, c.noland, c.prior, c.noland_sims, c.noland_w_n) == \
        (32, True, 2, "mc", "factory_sets", 16, 4.0)
    assert P2.parse_options(P2.options_tag(c)) == c
    assert P2.options_tag(P2.parse_options("")) == "" and P2.parse_options("") == P2.Planner2Config()
    assert P2.file_tag("n=32,h=1") == "n32-h1"
    with pytest.raises(ValueError):
        P2.parse_options("zz=1")
    with pytest.raises(ValueError):
        P2.Planner2Config(noland="x")


@pytest.mark.parametrize("kind", ["rental", "swap"])
def test_factory_prior_reproduces_the_old_planner(kind, monkeypatch):
    """prior factory_sets, depth 1, no hint_next: exactly rl.team_eval.plan_values (same draws, same values)."""
    monkeypatch.setenv("PYBATTLE_DIAGNOSTIC_PRIOR", "factory_sets")
    b, v, ctx, options = _decision(kind, 3, 15)
    ev = _FakeW()
    hint = (int(v.hint_type), int(v.hint_style))
    old = T.plan_values(ev, v, kind, ctx, options, 4.0, hint, T.PlannerConfig(seed=11), with_current=True)
    new = P2.plan_values2(ev, v, kind, ctx, options, 4.0, hint, P2.Planner2Config(prior="factory_sets", seed=11),
                          P2.OpponentSampler("factory_sets"), with_current=True)
    for key in ("q_plan", "phi", "e_round", "w_hint", "w_unknown"):
        assert np.allclose(old[key], new[key], atol=1e-12), key


@needs_model
@pytest.mark.parametrize("kind", ["rental", "swap"])
def test_switches_and_common_random_numbers(kind, model):
    b, v, ctx, options = _decision(kind, 3, 15)
    ev = _FakeW()
    hint = (int(v.hint_type), int(v.hint_style))
    samp = P2.OpponentSampler("experience", model)

    def q(opts, **kw):
        cfg = P2.Planner2Config(**dict(dict(seed=5, n_opp=6), **kw))
        return P2.plan_values2(ev, v, kind, ctx, opts, 3.0, hint, cfg, samp)["q_plan"]

    base = q(options)
    # determinism: the same seed gives the same values; another seed other draws
    assert np.array_equal(base, q(options))
    assert not np.allclose(base, q(options, seed=6))
    # common random numbers: an option's value does not depend on the other options evaluated with it
    sub = options[::3]
    assert np.allclose(q(sub), base[::3], atol=1e-12)
    assert np.allclose(q(sub, hint_next=True), q(options, hint_next=True)[::3], atol=1e-12)
    # depth 0 is Phi with the hint; depth 2 >= depth 1 with the same level-1 draws (keeping is always possible)
    assert np.allclose(q(options, depth=0), P2.plan_values2(ev, v, kind, ctx, options, 3.0, hint,
                                                            P2.Planner2Config(seed=5), samp)["phi"])
    assert (q(options, depth=2, n_opp2=3) >= base - 1e-9).all()
    # hint_next changes the values with a hint-aware W, and is a no-op with a W that ignores the hint
    assert not np.allclose(q(options, hint_next=True), base)
    ev0 = _FakeW(hint=False)
    a0 = P2.plan_values2(ev0, v, kind, ctx, options, 3.0, hint, P2.Planner2Config(seed=5, n_opp=6), samp)["q_plan"]
    a1 = P2.plan_values2(ev0, v, kind, ctx, options, 3.0, hint, P2.Planner2Config(seed=5, n_opp=6, hint_next=True),
                         samp)["q_plan"]
    assert np.allclose(a0, a1, atol=1e-12)
    # more opponents: other values, same definition (finite, positive)
    q32 = q(options, n_opp=32)
    assert q32.shape == base.shape and np.isfinite(q32).all() and (q32 > 0).all()


@needs_model
def test_hint_next_by_brute_force(model):
    """hint_next, depth 1, by the definition with loops: p_k(T | h) (1 + mean_n max_o V_0(T + o | h'_n))."""
    b, v, ctx, options = _decision("swap", 3, 15)
    ev = _FakeW()
    hint = (int(v.hint_type), int(v.hint_style))
    samp = P2.OpponentSampler("experience", model)
    cfg = P2.Planner2Config(seed=9, n_opp=4, hint_next=True)
    q = P2.plan_values2(ev, v, "swap", ctx, options, 2.0, hint, cfg, samp)["q_plan"]
    r, k = T.round_of(ctx)
    types = T.battle_types(r)
    g = np.random.default_rng(9)
    vis = T._visible_species(v, "swap")
    lvl = P2.team_level(v, "swap")
    opp = samp.normal(ctx["challenge"], 4, g, vis, hint, lvl)
    table = T._MonTable()
    for o in options:                                  # (the table only orders mons: no random calls)
        [table.add(m) for m in T.team_of_option(v, "swap", o).mons]
    seen = [[P2.seen_mon(opp.species[i, j], opp.moves[i, j], opp.item[i, j], g) for j in range(3)] for i in range(4)]
    nxt = [samp.normal(ctx["challenge"], 1, g, vis | set(int(x) for x in opp.species[n]), None, lvl) for n in range(4)]
    hn = [tuple(int(a[0]) for a in P2.hint_codes(d.species, d.moves)) for d in nxt]

    def leaf(team, j, h):
        wf, wu = ev.w([team], r, h)[0], ev.w([team], r)[0]
        run = e = wf[types[j - 1]]
        for i in range(j + 1, 8):
            run *= wu[types[i - 1]]
            e += run
        return e + run * 2.0

    for oi, o in enumerate(options):
        team = T.team_of_option(v, "swap", o)
        best = []
        for n in range(4):
            kids = [team] + [T.Team(tuple(seen[n][jj] if ii == i else team.mons[ii] for ii in range(3)))
                             for i in range(3) for jj in range(3)]
            best.append(max(leaf(t, k + 1, hn[n]) for t in kids))
        want = ev.w([team], r, hint)[0][types[k - 1]] * (1 + np.mean(best))
        assert abs(q[oi] - want) < 1e-9


# ---- Noland, simulated ---------------------------------------------------------------------------------------------

def _dummy_evaluator(batch):
    n = len(batch["mask"])
    return np.full((n, 7), 1.0 / 7), np.full(n, 0.5)


@needs_model
def test_noland_mc_is_deterministic_and_uses_his_iv(model):
    b = _pre_noland_swap()
    v = b.view()
    samp = P2.OpponentSampler("experience", model)
    mc = P2.NolandMC(_dummy_evaluator, samp, sims=4)
    opts = [0, 1, 5]
    p1, n1 = mc.p_win(b, opts, v, 2, np.random.default_rng(3), 100)
    p2, n2 = mc.p_win(b, opts, v, 2, np.random.default_rng(3), 100)
    assert np.array_equal(p1, p2) and (n1 == 4).all() and np.array_equal(n1, n2)
    # a simulated battle: our party, Noland's IV on every simulated opponent, the game's own team overwritten
    pool = samp.noland(2, 4, np.random.default_rng(0), None, 100)
    spec = P2.NolandMC._specs(pool, 0, random.Random(0), P2.noland_iv(2), model)
    be = P2.NolandMC._start(b, 0, spec, 123)
    foe = decode_party(be.game.read(S.addr("gEnemyParty"), 300))[:3]
    assert [m.species for m in foe] == [int(x) for x in pool.species[0]]
    assert all(iv == 15 for m in foe for iv in m.ivs)
    assert be.phase == Phase.BATTLE
    # inside the planner: only at the swap right before Noland, with the W shrinkage
    # (the run started mid-round at streak 19: run_ctx counts battles from the run's start, so the context of
    # battle 21 is given from the streak; the probes start runs at a round's first battle)
    ctx = {"streak": 20, "challenge": 2, "battle": 6, "rents": 3}
    ev = _FakeW()
    cfg = P2.Planner2Config(seed=3, noland="mc", noland_sims=4, noland_w_n=4.0)
    out = P2.plan_values2(ev, v, "swap", ctx, opts, 2.0, None, cfg, samp, backend=b, noland_mc=mc)
    w = out["w_hint"][:, T.NOLAND]
    assert np.allclose(out["q_plan"], (4.0 * w + 4 * out["noland_mc"]) / 8.0 * 3.0)


def test_probe_spec():
    from rl import agent_probe as AP
    assert AP.parse_spec("planner2:runs/x.pt:h=1,n=32") == ("planner2", "runs/x.pt", "n=32,h=1")
    assert AP.parse_spec("planner2:runs/x.pt") == ("planner2", "runs/x.pt", "")
    assert AP.parse_spec("planner:runs/x.pt") == ("planner", "runs/x.pt", None)

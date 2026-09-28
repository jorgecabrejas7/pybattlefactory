"""alphazero_v3's evaluator side (docs/RL_DECISIONS.md §20): the attendant's hint (rl.determinize.hint_of_team), the
hint-consistent opponent samplers, W conditioned on the hint and the tactician's planner (rl.team_eval.plan_values,
option_features_v3 / phi_v3 / q_eval_v3)."""

import random
import re
import struct

import numpy as np
import pytest
import torch

from pybattle.backend import OpponentKnowledge, Phase, SimBackend
from pybattle.emu.decode import SYMBOLS as S, decode_party
from pybattle.view import NAMES, SeenMon
from rl import determinize as DET
from rl import encode, search
from rl import team_eval as T
from rl import team_eval_data as TD
from rl.tactician_search import _unseen_view, options_of

_MOVE_ID = {v: int(k) for k, v in NAMES["moves"].items()}


@pytest.fixture(autouse=True)
def _keep_encode_version():
    v = encode.VERSION
    yield
    encode.set_version(v)


def _temp_party(b):
    return list(struct.unpack("<3H", b.game.read(S.addr("gFrontierTempParty"), 6)))


def _weaken_and_win(b):
    """Play (forcing wins: every opponent Pokemon at 1 HP) until the next SWAP / RENTAL decision or the end."""
    ep = S.addr("gEnemyParty")
    while b.phase in (Phase.BATTLE, Phase.FORCED_SWITCH):
        v = b.view()
        if b.phase == Phase.FORCED_SWITCH:
            b.act(("switch", v.switch_targets[0]))
            continue
        for i in range(3):
            if struct.unpack("<H", b.game.read(ep + i * 100 + 0x56, 2))[0] > 1:
                b.game.write(ep + i * 100 + 0x56, struct.pack("<H", 1))
        bm = S.addr("gBattleMons") + 0x58
        if struct.unpack("<H", b.game.read(bm + 0x28, 2))[0] > 1:
            b.game.write(bm + 0x28, struct.pack("<H", 1))
        b.game.write(bm + 0x24, bytes(4))
        moves = [a for a in v.legal_actions if a[0] == "move"]
        b.act(moves[0] if moves else v.legal_actions[0])


# ---- the hint: the game's own computation ------------------------------------------------------------------------------

def test_style_tables_match_the_decomp():
    src = open("third_party/pokeemerald/src/battle_factory.c").read()
    order = ["TotalPreparation", "SlowAndSteady", "Endurance", "HighRiskHighReturn", "WeakeningTheFoe",
             "ImpossibleToPredict", "DependsOnTheBattlesFlow"]
    for s, name in enumerate(order):
        body = re.search(r"sMoves_%s\[\] =\s*\{(.*?)\};" % name, src, re.S).group(1)
        names = [x.strip() for x in body.replace("\n", " ").split(",") if x.strip() and x.strip() != "MOVE_NONE"]
        assert tuple(names) == DET._STYLE_MOVES[s]
    counts = re.search(r"sRequiredMoveCounts\[FACTORY_NUM_STYLES - 1\] =\s*\{(.*?)\};", src, re.S).group(1)
    assert tuple(int(x.split("=")[1]) for x in counts.split(",") if "=" in x) == DET.STYLE_REQUIRED
    assert DET.MOVE_STYLE[_MOVE_ID["MOVE_SNATCH"]] == 1          # in two lists: the first one counts
    assert DET.MOVE_STYLE[_MOVE_ID["MOVE_TACKLE"]] == 0


def test_hint_rules_on_hand_made_counts():
    def one(tc, sp=None):
        t, s = DET._hint_from_counts(np.array([tc]), np.array([sp or [0] * 8]))
        return int(t[0]), int(s[0])
    tc = [0] * 18
    assert one(tc)[0] == 18                                   # (never in a real team) no type counted
    tc = [0] * 18; tc[11] = 2; tc[4] = 1
    assert one(tc)[0] == 11                                   # strictly most common
    tc = [0] * 18; tc[11] = 2; tc[4] = 2
    assert one(tc)[0] == 18                                   # a tie: no hint type
    tc = [0] * 18; tc[0] = 2; tc[5] = 1
    assert one(tc)[0] == 0                                    # Normal (index 0) leading, never tied
    tc = [0] * 18; tc[3] = 1; tc[5] = 1; tc[12] = 3
    assert one(tc)[0] == 12                                   # a later type overtakes an earlier tie
    # styles: thresholds 3/3/3/2/2/2/2; one or two styles -> the last; three or more -> 8
    assert one([0] * 18, [9, 2, 2, 2, 1, 1, 1, 1])[1] == 0
    assert one([0] * 18, [0, 3, 0, 0, 0, 0, 0, 0])[1] == 1
    assert one([0] * 18, [0, 3, 0, 0, 2, 0, 0, 0])[1] == 4
    assert one([0] * 18, [0, 3, 3, 0, 2, 0, 0, 0])[1] == 8


def test_hint_of_team_accepts_ids_pairs_and_specs():
    ids = [int(x) for x in DET.pool(3)[:3]]
    h = DET.hint_of_team(ids)
    pairs = [(int(DET.SET_SPECIES[i]), list(DET.FRONTIER_MONS[i]["moves"])) for i in ids]
    specs = [DET.set_spec(j, i, 3) for j, i in enumerate(ids)]         # (Return is built as Frustration)
    assert DET.hint_of_team(pairs) == h == DET.hint_of_team(specs)


def test_hint_of_team_matches_the_game_at_rentals():
    """The hint the game computes (RentalView) for the first opponent of a round vs hint_of_team on its real sets
    (gFrontierTempParty; read in this test only)."""
    seen = set()
    for seed in range(600):
        b = SimBackend(max_turns=10 ** 9)
        b.reset(seed=seed, win_streak=7 * (seed % 7), rents_count=seed % 7)
        v = b.view()
        assert DET.hint_of_team(_temp_party(b)) == (v.hint_type, v.hint_style)
        seen.add((v.hint_type, v.hint_style))
    assert len(seen) > 20                                    # many different hints were checked


def test_hint_of_team_matches_the_game_at_swaps():
    n = 0
    rng = random.Random(0)
    for seed in range(40):
        b = SimBackend(max_turns=10 ** 9)
        b.reset(seed=seed, win_streak=7 * (seed % 6) + seed % 5, rents_count=3)
        v = b.view()
        b.act(T.rental_pick(rng.choice(T.legal_rentals(v))))
        for _ in range(3):
            _weaken_and_win(b)
            if b.phase != Phase.SWAP:
                break
            v = b.view()
            if b.run_info().noland:
                assert (v.hint_type, v.hint_style) == DET.NO_HINT
            else:
                assert DET.hint_of_team(_temp_party(b)) == (v.hint_type, v.hint_style)
                n += 1
            b.act(None)
    assert n > 40


# ---- samplers ---------------------------------------------------------------------------------------------------------

def _seen_view(level=100):
    """An opponent team in battle: slot 0 seen (Salamence? whatever a real set is) with two revealed moves, slot 1
    seen and fainted, slot 2 unseen."""
    ids = DET.pool(2)
    a, bb = int(ids[5]), int(ids[40])
    mv_a = [x for x in DET.SET_MOVES_AS_BUILT[a] if x][:2]
    m0 = SeenMon(species=int(DET.SET_SPECIES[a]), seen=True, revealed_moves=mv_a, hp_pixels=30, level=level)
    m1 = SeenMon(species=int(DET.SET_SPECIES[bb]), seen=True, fainted=True, level=level)
    v = _unseen_view(level)
    v.enemy_party = [m0, m1, SeenMon()]
    return v


@pytest.mark.parametrize("seen", [False, True])
def test_factory_sets_sampler_draws_teams_with_the_heard_hint(seen, monkeypatch):
    monkeypatch.setenv(search.TRAINING_ENV, "1")
    v = _seen_view() if seen else _unseen_view(100)
    kn = OpponentKnowledge(2, False, frozenset(), battle=1)
    rng = random.Random(1)
    # the hints this sampler produces without a hint, then each of the common ones asked for
    hints = [DET.hint_of_team(DET.sample_factory_sets(v, kn, rng)) for _ in range(300)] if not seen else None
    targets = [(18, 0), (18, 4), (11, 0)] if not seen else [(18, 0), (18, 4)]
    for h in targets:
        DET.hint_stats.update(calls=0, draws=0, fallback=0)
        for _ in range(40):
            specs = DET.sample_factory_sets(v, kn, rng, hint=h)
            assert len(specs) == 3
            if not seen:
                assert DET.hint_of_team(specs) == h
            else:
                assert specs[1][1] == DET.KEEP                        # the fainted one is kept
                assert specs[0][1] == v.enemy_party[0].species        # the seen one keeps its species and moves
                assert set(v.enemy_party[0].revealed_moves) <= set(specs[0][2])
        if not seen:
            assert DET.hint_stats["fallback"] == 0
    if hints is not None:
        assert (18, 0) in hints
    # Noland: no hint, the argument is ignored
    kn_n = OpponentKnowledge(2, True, set_ids=frozenset(), battle=6)
    assert len(DET.sample_factory_sets(_unseen_view(100), kn_n, rng, hint=(3, 3))) == 3


def test_hint_rejection_keeps_the_sampler_distribution(monkeypatch):
    """Conditioning by rejection: the species frequencies among accepted teams equal those of unconditioned draws
    that happen to have the hint (both are samples of the same conditional distribution)."""
    monkeypatch.setenv(search.TRAINING_ENV, "1")
    v = _unseen_view(100)
    kn = OpponentKnowledge(1, False, frozenset(), battle=0)
    rng = random.Random(3)
    h = (18, 4)
    free = [s for s in (DET.sample_factory_sets(v, kn, rng) for _ in range(4000)) if DET.hint_of_team(s) == h]
    cond = [DET.sample_factory_sets(v, kn, rng, hint=h) for _ in range(len(free))]
    count = lambda xs: np.bincount([x[1] for s in xs for x in s], minlength=450)    # noqa: E731
    a, b = count(free), count(cond)
    top = np.argsort(-(a + b))[:15]
    # the most common species appear at similar rates (loose: sampling noise)
    assert np.abs(a[top] - b[top]).max() <= 6 * np.sqrt((a[top] + b[top]).max() / 2) + 5


def test_strict_sampler_draws_teams_with_the_heard_hint():
    rng = random.Random(2)
    v = _unseen_view(100)
    for h in [(18, 0), (18, 4), (11, 0), (18, 2)]:
        DET.hint_stats.update(calls=0, draws=0, fallback=0)
        for _ in range(15):
            specs = DET.sample_determinization(v, None, rng, hint=h)
            assert DET.hint_of_team(specs) == h
            for s in specs:                                      # still legal: learnable moves, no repeats
                assert set(x for x in s[2] if x) <= set(DET.learnable(s[1], 100))
            assert len({s[1] for s in specs}) == 3
        assert DET.hint_stats["fallback"] == 0
    v = _seen_view()
    specs = DET.sample_determinization(v, None, rng, hint=(18, 0))
    assert specs[1][1] == DET.KEEP and specs[0][1] == v.enemy_party[0].species
    assert set(v.enemy_party[0].revealed_moves) <= set(specs[0][2])
    # an impossible hint falls back after max_tries and is counted
    DET.hint_stats.update(calls=0, draws=0, fallback=0)
    DET.sample_determinization(_unseen_view(100), None, rng, hint=(9, 0), max_tries=64)      # TYPE_MYSTERY
    assert DET.hint_stats["fallback"] == 1


def test_sample_pool_teams():
    g = np.random.default_rng(0)
    ids = DET.pool(4)
    excl = {int(DET.SET_SPECIES[ids[0]]), int(DET.SET_SPECIES[ids[1]])}
    t = DET.sample_pool_teams(ids, 50, g, excl, (18, 4))
    assert t.shape == (50, 3)
    for row in t:
        sp = DET.SET_SPECIES[row]
        assert len(set(sp.tolist())) == 3 and not (set(sp.tolist()) & excl)
        it = [x for x in DET.SET_ITEM[row].tolist() if x]
        assert len(set(it)) == len(it)
        assert DET.hint_of_team(row.tolist()) == (18, 4)
    assert DET.sample_pool_teams(ids, 7, g).shape == (7, 3)


# ---- data: each battle records its opponent's hint ------------------------------------------------------------------------

def test_start_battle_records_the_opponents_hint(monkeypatch):
    monkeypatch.setenv(search.TRAINING_ENV, "1")
    rng = random.Random(5)
    for rt in [(1, 0), (4, 1), (3, 2)]:
        be, own, know, rec = TD.make_case(rng, p_swap=0.5, round_type=rt)
        c, hint = TD.start_battle(be, own, know, rng, with_hint=True)
        foe = decode_party(c.game.read(S.addr("gEnemyParty"), 300))[:3]
        if rt[1] == 2:
            assert hint == DET.NO_HINT
        else:
            assert hint == DET.hint_of_team([(m.species, list(m.moves)) for m in foe])
    d = TD.to_arrays([rec], [(0, True, False, 10), (0, False, False, 12, 11, 4)])
    assert d["b_hint_type"].tolist() == [-1, 11] and d["b_hint_style"].tolist() == [-1, 4]


# ---- W conditioned on the hint --------------------------------------------------------------------------------------------

def _hint_evaluator(seed=0):
    torch.manual_seed(seed)
    return T.TeamEvaluator(T.TeamNet(d=32, d_emb=8, layers=1, hint=True))


def _teams(n=6, seed=6):
    b = SimBackend(max_turns=10 ** 9)
    b.reset(seed=seed, win_streak=14, rents_count=2)
    v = b.view()
    return [T.team_of_option(v, "rental", o) for o in T.legal_rentals(v)[:n]]


def test_hint_network_shapes_and_unknown(tmp_path):
    ev = _hint_evaluator()
    teams = _teams()
    assert ev.uses_hint
    wu = ev.w(teams, 3)
    wh = ev.w(teams, 3, (11, 4))
    assert wu.shape == wh.shape == (6, 3) and not np.allclose(wu, wh)
    assert np.allclose(ev.w(teams, 3, None), wu)
    per = ev.w(teams, 3, [(11, 4)] * 3 + [None] * 3)
    assert np.allclose(per[:3], wh[:3]) and np.allclose(per[3:], wu[3:])
    path = str(tmp_path / "w3.pt")
    ev.save(path)
    ev2 = T.TeamEvaluator.load(path)
    assert ev2.uses_hint and np.allclose(ev2.w(teams, 3, (11, 4)), wh, atol=1e-6)
    # a v1 checkpoint (no hint) still loads and ignores the hint
    torch.manual_seed(1)
    ev1 = T.TeamEvaluator(T.TeamNet(d=32, d_emb=8, layers=1))
    assert not ev1.uses_hint and np.allclose(ev1.w(teams, 3, (11, 4)), ev1.w(teams, 3))


def _tiny_data(n_teams=12, seed=0):
    rng = random.Random(seed)
    recs = []
    while len(recs) < n_teams:
        try:
            recs.append(TD.make_case(rng)[3])
        except RuntimeError:
            continue
    r = np.random.default_rng(seed)
    bat = [(t, bool(r.random() < 0.7), bool(r.random() < 0.05), 10, int(r.integers(19)), int(r.integers(9)))
           for t in range(n_teams) for _ in range(5)]
    return TD.to_arrays(recs, bat)


def test_battle_loss_is_the_binomial_loss_without_a_hint():
    d = _tiny_data()
    torch.manual_seed(0)
    net = T.TeamNet(d=32, d_emb=8, layers=1)
    idx = np.arange(len(d["round"]))
    loss, nb = T.battle_loss(net, d, idx, np.random.default_rng(1), "cpu")
    ids, num = T.hide_swapped(d, idx, np.random.default_rng(1))
    played, won = T.team_targets(d)
    logit = net(torch.from_numpy(ids), torch.from_numpy(num), torch.from_numpy(T.round_index(d["round"])))
    lg = logit.gather(1, torch.from_numpy(d["btype"].astype(np.int64))[:, None])[:, 0]
    pl, wn = torch.from_numpy(played.astype(np.float32)), torch.from_numpy(won.astype(np.float32))
    ref = (pl * torch.nn.functional.softplus(lg) - wn * lg).sum() / pl.sum()
    assert nb == int(played.sum()) and torch.isclose(loss, ref, atol=1e-5)
    # the hint model: Noland battles and dropped hints use the unknown token
    ht, hs = T.battle_hints(d, np.arange(len(d["b_won"])), np.random.default_rng(0), p_unknown=0.0)
    nol = d["btype"][d["b_team"]] == T.NOLAND
    assert (ht[nol] == T.HINT_UNK_TYPE).all() and (ht[~nol] == d["b_hint_type"][~nol]).all()
    ht, _ = T.battle_hints(d, np.arange(len(d["b_won"])), np.random.default_rng(0), p_unknown=1.0)
    assert (ht == T.HINT_UNK_TYPE).all()
    net_h = T.TeamNet(d=32, d_emb=8, layers=1, hint=True)
    loss_h, _ = T.battle_loss(net_h, d, idx, np.random.default_rng(1), "cpu", p_unknown=0.3)
    loss_h.backward()
    assert net_h.hint_type.weight.grad is not None


# ---- the planner ----------------------------------------------------------------------------------------------------------

class _FakeW(T.TeamEvaluator):
    """A deterministic W that depends on every Pokemon, the lead and the hint (for the planner's formulas)."""

    def __init__(self, hint=True):
        self.net, self.meta, self.device, self._hint = None, {}, torch.device("cpu"), hint

    @property
    def uses_hint(self):
        return self._hint

    def w_features(self, ids, num, rounds, hint_type=None, hint_style=None):
        ids = np.asarray(ids, np.float64)
        x = (ids[..., 0] * (1 + 2 * np.asarray(num)[..., 7])).sum(-1) + 0.37 * ids[..., 3:].sum((-1, -2))
        if hint_type is not None:
            x = x + 3.1 * np.asarray(hint_type) + 1.7 * np.asarray(hint_style)
        base = 0.55 + 0.4 * np.sin(x)
        return np.stack([base, 0.9 * base, 0.5 * base], -1)


def _decision(kind, seed, streak):
    rng = random.Random(seed)
    for s in range(seed, seed + 50):
        b = SimBackend(max_turns=10 ** 9)
        b.reset(seed=s, win_streak=streak, rents_count=2)
        if kind == "swap":
            b.act(T.rental_pick(rng.choice(T.legal_rentals(b.view()))))
            _weaken_and_win(b)
            if b.phase != Phase.SWAP or b.run_info().noland:
                continue
        ctx = search.run_ctx(b)
        v = b.view()
        return b, v, ctx, options_of(kind, encode.rental(v, ctx) if kind == "rental" else encode.swap(v, ctx))
    raise RuntimeError("no decision reached")


def _brute_force_q1(ev, v, kind, ctx, option, v_next, hint, cfg):
    """V_1 of one option by the definition, with loops over Team objects (the same draws as plan_values)."""
    r, k = T.round_of(ctx)
    g = np.random.default_rng(cfg.seed)
    team = T.team_of_option(v, kind, option)
    opp = DET.sample_pool_teams(DET.pool(ctx["challenge"]), cfg.n_opp, g, T._visible_species(v, kind), hint)
    seen = [[T.seen_set_mon(int(x), g) for x in row] for row in opp]
    types = T.battle_types(r)

    def leaf(tm, j, h):
        wf = ev.w([tm], r, h)[0]
        wu = ev.w([tm], r)[0]
        run, e = wf[types[j - 1]], wf[types[j - 1]]
        for i in range(j + 1, 8):
            run *= wu[types[i - 1]]
            e += run
        return e + run * v_next

    best = []
    for n in range(cfg.n_opp):
        vals = [leaf(team, k + 1, None)]
        for i in range(3):
            for j in range(3):
                mons = list(team.mons)
                mons[i] = seen[n][j]
                vals.append(leaf(T.Team(tuple(mons)), k + 1, None))
        best.append(max(vals))
    p = ev.w([team], r, hint)[0][types[k - 1]]
    return p * (1 + np.mean(best)), leaf(team, k, hint)


@pytest.mark.parametrize("kind", ["rental", "swap"])
def test_planner_matches_its_definition(kind):
    ev = _FakeW()
    b, v, ctx, opts = _decision(kind, 11, 14)
    hint = (v.hint_type, v.hint_style)
    cfg = T.PlannerConfig(depth=1, n_opp=4, seed=123)
    out = T.plan_values(ev, v, kind, ctx, opts, 5.0, hint, cfg)
    for i in [0, len(opts) // 2, len(opts) - 1]:
        q1, phi = _brute_force_q1(ev, v, kind, ctx, opts[i], 5.0, hint, cfg)
        assert np.isclose(out["q_plan"][i], q1, rtol=1e-9) and np.isclose(out["phi"][i], phi, rtol=1e-9)


@pytest.mark.parametrize("kind", ["rental", "swap"])
def test_planner_depths_are_ordered_and_depth_0_is_phi(kind):
    ev = _FakeW()
    b, v, ctx, opts = _decision(kind, 12, 7)
    hint = (v.hint_type, v.hint_style)
    q = [T.plan_values(ev, v, kind, ctx, opts, 4.0, hint, T.PlannerConfig(depth=dd, n_opp=4, n_opp2=3, seed=7))
         for dd in (0, 1, 2)]
    assert np.allclose(q[0]["q_plan"], q[0]["phi"])
    # V_2 >= V_1 >= V_0 on the same draws (keeping is always an option; the beam keeps the best by V_0)
    assert (q[1]["q_plan"] >= q[0]["q_plan"] - 1e-9).all() and (q[2]["q_plan"] >= q[1]["q_plan"] - 1e-9).all()
    # Phi without a hint is rl.team_eval.phi_options (§19.2's Q_eval)
    ev_nh = _FakeW(hint=False)
    q0 = T.plan_values(ev_nh, v, kind, ctx, opts, 4.0, hint, T.PlannerConfig(depth=0))
    assert np.allclose(q0["q_plan"], T.phi_options(ev_nh, v, kind, ctx, opts, 4.0))


def test_planner_at_battle_7():
    ev = _FakeW()
    b, v, ctx, opts = _decision("swap", 13, 7)
    ctx = dict(ctx, battle=6)                                  # read as the swap screen before battle 7 of round 2
    r, k = T.round_of(ctx)
    assert (r, k) == (2, 7)
    out = T.plan_values(ev, v, "swap", ctx, opts, 2.0, (11, 4), T.PlannerConfig(depth=2, seed=1))
    # no swap after the last battle: the plan is Phi = W(last | hint) (1 + v_next)
    W = ev.w([T.team_of_option(v, "swap", o) for o in opts], r, (11, 4))
    assert np.allclose(out["q_plan"], W[:, T.LAST] * 3.0)
    # before Noland's battle the hint is ignored (the game gives none)
    types = T.battle_types(3)
    ctx3 = dict(ctx, challenge=2)
    out3 = T.plan_values(ev, v, "swap", ctx3, opts, 2.0, (11, 4), T.PlannerConfig(depth=1, seed=1))
    W3 = ev.w([T.team_of_option(v, "swap", o) for o in opts], 3)
    assert types[6] == T.NOLAND and np.allclose(out3["q_plan"], W3[:, T.NOLAND] * 3.0)


def test_v3_interface():
    ev = _FakeW()
    table = [0.0] + [6.0] * 11
    for kind, streak in (("rental", 14), ("swap", 14)):
        b, v, ctx, opts = _decision(kind, 14, streak)
        hint = (v.hint_type, v.hint_style)
        cfg = {"depth": 1, "n_opp": 4, "seed": 5}
        f = T.option_features_v3(ev, v, kind, ctx, opts, table, hint, cfg)
        assert f.shape == (len(opts), T.N_OPT_FEATURES) and f.dtype == np.float32 and np.isfinite(f).all()
        q = T.q_eval_v3(f)
        assert np.allclose(q, f[:, T.Q_PLAN] * T.WIN_SCALE)
        assert np.allclose(f[:, 5], f[:, 4] - f[:, 4].mean(), atol=1e-5)
        out = T.plan_values(ev, v, kind, ctx, opts, 6.0, hint, T.PlannerConfig(**cfg), with_current=True)
        assert np.allclose(q, out["q_plan"], atol=1e-4) and np.allclose(f[:, 6] * T.WIN_SCALE, out["phi"], atol=1e-4)
        assert np.allclose(f[:, 0] * T.WIN_SCALE, out["e_round"], atol=1e-4) and np.abs(f).max() < 5
        phi = T.phi_v3(ev, v, kind, ctx, table, hint, cfg)
        if kind == "swap":
            assert np.isclose(phi, q[opts.index(0)], atol=1e-4)                # keeping = the current team
            assert np.isclose(f[opts.index(0), 3], 0.0, atol=1e-6)
        else:
            assert np.isclose(phi, q.mean(), atol=1e-4)
            assert abs(f[:, 3].mean()) < 1e-5
        T._LAST_PLAN.clear()                                   # without the cached plan: the same value
        assert np.isclose(T.phi_v3(ev, v, kind, ctx, table, hint, cfg), phi, atol=1e-4)


def test_v3_features_never_read_the_opponents_hidden_state():
    from tests.python.test_team_eval import _rebuild_defeated_team
    ev = _FakeW()
    b, v, ctx, opts = _decision("swap", 15, 14)
    cfg = {"depth": 1, "n_opp": 4, "seed": 9}
    hint = (v.hint_type, v.hint_style)
    f1 = T.option_features_v3(ev, v, "swap", ctx, opts, None, hint, cfg)
    c = _rebuild_defeated_team(b, random.Random(0))
    T._LAST_PLAN.clear()
    f2 = T.option_features_v3(ev, c.view(), "swap", search.run_ctx(c), opts, None, hint, cfg)
    assert np.array_equal(f1, f2)


def test_planner_runs_with_a_network():
    ev = _hint_evaluator(3)
    b, v, ctx, opts = _decision("rental", 16, 7)
    f = T.option_features_v3(ev, v, "rental", ctx, opts, [0.0] + [5.0] * 11, (v.hint_type, v.hint_style),
                             {"depth": 2, "n_opp": 3, "n_opp2": 2, "beam": 2})
    assert f.shape == (len(opts), T.N_OPT_FEATURES) and np.isfinite(f).all()
    assert (f[:, 4] >= f[:, 6] - 1e-5).all()                   # planning never loses value


def test_team_features_v3_in_an_environment():
    from rl import tactician_features as TF
    ev = _FakeW()
    for kind, streak in (("rental", 7), ("swap", 7)):
        b, v, ctx, opts = _decision(kind, 17, streak)
        obs = encode.rental(v, ctx) if kind == "rental" else encode.swap(v, ctx)
        tf = TF.TeamFeaturesV3("unused", "rl.team_eval", planner_cfg={"depth": 1, "n_opp": 3, "seed": 2})
        tf._mod, tf._ev = T, ev
        tf.set_v_next([0.0] + [5.0] * 11)
        out = tf.compute(b, kind, obs, ctx)
        shape = ((6, 15) if kind == "rental" else (10,)) + (T.N_OPT_FEATURES,)
        assert out["opt_feat"].shape == shape and tf.n_feat == T.N_OPT_FEATURES
        f = T.option_features_v3(ev, v, kind, ctx, opts, [0.0] + [5.0] * 11, (v.hint_type, v.hint_style),
                                 {"depth": 1, "n_opp": 3, "seed": 2})
        assert np.allclose(out["feats"], f) and np.allclose(out["q_eval"], f[:, T.Q_PLAN] * T.WIN_SCALE)
        cell = tuple(opts[0]) if kind == "rental" else opts[0]
        assert np.allclose(out["opt_feat"][cell], f[0])
        assert np.isfinite(out["phi"])
        obs2 = tf(b, kind, obs, ctx)
        assert obs2["opt_feat"].shape == shape


def test_real_battle_log_records_the_games_hint():
    from rl.team_eval_refresh import RealBattleLog, merge_arrays
    log = RealBattleLog()
    for seed in range(6):
        b = SimBackend(max_turns=10 ** 9)
        b.reset(seed=seed, win_streak=7, rents_count=2)
        v = b.view()
        opt = T.legal_rentals(v)[0]
        log.on_tactician("rental", 0)
        b.act(T.rental_pick(opt))
        log.start(b, search.run_ctx(b))
        assert log.hint == (v.hint_type, v.hint_style)       # what the attendant said at the rental screen
        log.end(True, False, 5)
    d = log.take()
    assert (d["b_hint_type"] >= 0).all() and len(d["b_hint_type"]) == 6
    # merged with data without recorded hints (v1): those battles get -1 (unknown)
    old = {k: v for k, v in d.items() if not k.startswith("b_hint")}
    m = merge_arrays([d, old])
    assert m["b_hint_type"].tolist() == d["b_hint_type"].tolist() + [-1] * 6

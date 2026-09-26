"""alphazero_v2 (rl/alphazero_v2.py): the Gumbel battler (rl/gumbel.py + the C++ Gumbel root), the per-option
features and the hybrid tactician (rl/tactician_features.py, rl/team_eval_stub.py), and a tiny run end to end."""

import inspect
import itertools
import math
import os

import numpy as np
import pytest
import torch

from pybattle.backend import Phase, SimBackend
from rl import alphazero as AZ
from rl import alphazero_v2 as V2
from rl import encode, gumbel as G, search
from rl import tactician_features as TF
from rl import tactician_search as TS
from rl import team_eval_stub as STUB
from rl.model import FactoryNet

pytestmark = pytest.mark.skipif(G.halving_plan is None or not hasattr(search.NativeSearcher, "search_gumbel"),
                                reason="needs the extension's Gumbel search (rebuild)")


@pytest.fixture(autouse=True)
def _keep_encode_version():
    v = encode.VERSION
    yield
    encode.set_version(v)


def _net(seed=0, opt_feat=0):
    torch.manual_seed(seed)
    encode.set_version(4)
    return FactoryNet(share="embeddings", opt_feat=opt_feat).eval()


def _battle_backend(seed=4):
    b = SimBackend(max_turns=10 ** 9)
    b.reset(seed=seed)
    v = b.view()
    b.act(next(t for t in itertools.combinations(range(6), 3) if len({v.candidates[i].species for i in t}) == 3))
    assert b.phase in (Phase.BATTLE, Phase.FORCED_SWITCH)
    return b


def _battler(seed=1, sims=64, dets=4, **kw):
    net = _net(2)
    return G.GumbelBattler(AZ.local_battler_evaluator(net, None), n_sims=sims, n_determinizations=dets, seed=seed,
                           batch=16, **kw)


# ---- sequential halving ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("m", range(1, 17))
def test_halving_plan_spends_the_budget(m):
    for n in (16, 64, 256, 1000):
        per, tot = G.plan(m, n)
        assert len(per) == (1 if m == 1 else math.ceil(math.log2(m)))
        r = m
        for i, x in enumerate(per):
            assert x >= 1
            if i < len(per) - 1:
                assert x == max(1, (n // len(per)) // r)       # n / phases per phase, split evenly
            r = (r + 1) // 2
        if n >= m * len(per):
            assert sum(tot) == n                                # exactly the budget


@pytest.mark.parametrize("non_root", ["gumbel", "puct"])
def test_gumbel_search_allocates_the_budget_by_halving(non_root):
    b = _battle_backend()
    gb = _battler(non_root=non_root, sims=128)
    r = gb.search_root(b, 0)
    legal = r["legal"]
    m = int(legal.sum())
    per, tot = G.plan(m, 128)
    assert r["searched"] and r["considered"][legal].all() and not r["considered"][~legal].any()
    assert r["phase_sims"] == tot and r["visits"].sum() == 128
    # the visits of an action = the simulations of the phases it survived: sorted, they are the plan's partial sums
    cum = np.cumsum(per)
    survivors = [m]
    for _ in per[:-1]:
        survivors.append((survivors[-1] + 1) // 2)
    expect = []
    for i in range(len(per)):
        n_out = survivors[i] - (survivors[i + 1] if i + 1 < len(per) else 0)
        expect += [cum[i]] * n_out
    extra = 128 - sum(tot[:-1]) - per[-1] * survivors[-1]
    got = sorted(r["visits"][legal].astype(int))
    expect = sorted(expect)
    if extra:                                                  # the last phase's remainder: one more to the best
        expect[-1] += extra
    assert got == sorted(expect)
    # the winner is the better of the last two survivors by g + logits + sigma(completed Q)
    last2 = np.argsort(-r["visits"])[:2]
    score = G.halving_winner_score(r["prior"], r["visits"], r["q"], legal, r["value"], r["g"])
    assert r["action"] in last2 and score[r["action"]] >= score[[a for a in last2 if a != r["action"]][0]] - 1e-4


def test_gumbel_search_is_deterministic_given_seeds():
    b = _battle_backend(7)
    out = []
    for _ in range(2):
        gb = _battler(seed=5)
        r = gb.search_root(b.clone(), 0)
        out.append(r)
    assert out[0]["action"] == out[1]["action"]
    assert np.array_equal(out[0]["visits"], out[1]["visits"])
    assert np.allclose(out[0]["q"], out[1]["q"]) and np.allclose(out[0]["target"], out[1]["target"])
    gb = _battler(seed=6)
    assert not np.array_equal(gb.search_root(b.clone(), 0)["g"], out[0]["g"])


def test_gumbel_self_play_never_plays_an_illegal_action():
    torch.set_num_threads(2)
    gb = _battler(sims=32, dets=2)
    for seed in (1, 2):
        b = _battle_backend(seed)
        steps = 0
        while b.phase in (Phase.BATTLE, Phase.FORCED_SWITCH) and steps < 40:
            ctx = search.run_ctx(b)
            x = encode.battle(b.view(), ctx)
            r = gb.search_root(b, steps, x)
            assert x["mask"][r["action"]]
            assert (r["target"][~x["mask"]] == 0).all() and np.isclose(r["target"].sum(), 1.0)
            assert (r["visits"][~x["mask"]] == 0).all()
            b.act(search.to_backend_action(r["action"]))
            steps += 1
    assert gb.errors == 0


# ---- targets -------------------------------------------------------------------------------------------------------

def test_improved_policy_puts_more_mass_on_better_actions():
    legal = np.array([1, 1, 1, 0, 1, 0, 0], bool)
    prior = np.array([0.25, 0.25, 0.25, 0, 0.25, 0, 0])
    visits = np.array([40, 40, 20, 0, 0, 0, 0], float)
    q = np.array([0.8, 0.6, 0.3, 0, 0, 0, 0])
    t = G.improved_policy(prior, visits, q, legal, v_hat=0.5)
    assert np.isclose(t.sum(), 1) and (t[~legal] == 0).all()
    assert t[0] > t[1] > t[2]
    # the unvisited action gets v_mix: (v_hat + sum N * sum pi q / sum pi) / (1 + sum N)
    vmix = (0.5 + 100 * (0.25 * (0.8 + 0.6 + 0.3)) / 0.75) / 101
    assert np.isclose(G.completed_q(visits, q, prior, legal, 0.5)[4], vmix)
    assert t[2] < t[4] < t[1]                                   # 0.3 < v_mix ~ 0.566 < 0.6
    # sigma = (c_visit + max N) c_scale q_rescaled: the best gets (50 + 40) * 0.1 = 9 more than the worst
    s = G.sigma(G.completed_q(visits, q, prior, legal, 0.5), visits, legal)
    assert np.isclose(s[0] - s[2], 9.0)
    # equal Q: the target is the prior
    t2 = G.improved_policy(prior, visits, np.where(legal, 0.5, 0), legal, 0.5)
    assert np.allclose(t2[legal], 0.25)


def test_gumbel_rule_below_the_root_is_the_papers():
    from pybattle.pybattle_native import MctsTree
    t = MctsTree(3, 1.5, 1.0)
    t.set_gumbel_rule(True, 50.0, 0.1, True)
    t.select(1)
    t.expand(0, [0.5, 0.3, 0.2], [True, True, True])
    t.backup(0, 0.5)
    # forced root action 1: its child is created
    (leaf, parent, a), = t.select_forced(1)
    assert (parent, a) == (0, 1)
    t.expand(leaf, [0.6, 0.3, 0.1], [True, True, True])
    t.backup(leaf, 0.4)
    assert np.isclose(t.net_value(leaf), 0.4)
    # below the root, with no child visited: pi' = prior, all N = 0 -> the largest prior (action 0)
    (l2, p2, a2), = t.select_forced(1)
    assert (p2, a2) == (leaf, 0)
    t.expand(l2, [1 / 3] * 3, [True] * 3)
    t.backup(l2, 0.0)
    # now action 0 (q 0) vs unvisited (v_mix): score pi'(a) - N(a) / (1 + sum N)
    pri = np.array([0.6, 0.3, 0.1])
    n = np.array([1.0, 0, 0])
    vmix = (0.4 + 1 * 0.0) / 2
    cq = np.array([0.0, vmix, vmix])
    cq = (cq - cq.min()) / (cq.max() - cq.min())
    z = np.log(pri) + (50 + 1) * 0.1 * cq
    pi = np.exp(z - z.max()); pi /= pi.sum()
    expect = int(np.argmax(pi - n / (1 + n.sum())))
    (l3, p3, a3), = t.select_forced(1)
    assert (p3, a3) == (leaf, expect)


# ---- no perfect information in training ------------------------------------------------------------------------------

def test_perfect_mode_impossible_in_training(monkeypatch):
    assert "rl.alphazero_v2" in search._TRAINING_MAINS
    src = inspect.getsource(V2.main)
    assert "mark_training()" in src.split("parse(argv)")[0]
    assert "mode" not in inspect.signature(G.GumbelBattler).parameters
    monkeypatch.setenv(search.TRAINING_ENV, "1")
    gb = _battler(sims=16, dets=2, opponent_prior="factory_sets")
    assert gb.mode == "legal"
    b = _battle_backend(3)
    r = gb.search_root(b, 0)
    assert r["searched"]
    with pytest.raises(PermissionError):                         # a true-state root is refused in training
        gb.searcher.search_gumbel([(b.game.clone(), b._observer.fast_copy(), 0)], search.run_ctx(b),
                                  [1 / 7] * 7, [True] * 7, 0.5, gb.evaluate_batch, [0.0] * 7)
    monkeypatch.delenv(search.TRAINING_ENV)
    with pytest.raises(PermissionError):                         # factory_sets is training-only
        _battler(opponent_prior="factory_sets")


# ---- the network's per-option features -------------------------------------------------------------------------------

def _tactician_obs(kind, seed=3):
    """A SimBackend at a RENTAL / SWAP decision and its encoded observation (test_alphazero's helper)."""
    import random
    rng = random.Random(seed)
    for s in range(seed, seed + 50):
        b = SimBackend(max_turns=500)            # (a stalled battle is forfeited)
        b.reset(seed=s, win_streak=7, rents_count=2)
        if kind == "swap" and not _battle_to_swap(b, rng):
            continue
        ctx = search.run_ctx(b)
        v = b.view()
        return b, (encode.rental(v, ctx) if kind == "rental" else encode.swap(v, ctx))
    raise RuntimeError("no swap decision reached")


def _battle_to_swap(b, rng):
    while b.phase != Phase.SWAP:
        if b.phase == Phase.RUN_OVER:
            return False
        v = b.view()
        if b.phase == Phase.RENTAL:
            while True:
                pick = tuple(rng.sample(range(6), 3))
                if len({v.candidates[i].species for i in pick}) == 3:
                    break
            b.act(pick)
        elif b.phase == Phase.FORCED_SWITCH:
            b.act(("switch", v.switch_targets[0]))
        else:
            moves = [i for i, ok in enumerate(v.usable_moves) if ok]
            b.act(("move", moves[0]) if moves else ("switch", v.switch_targets[0]) if v.switch_targets else ("move", 0))
    return True


def test_option_features_tilt_the_joint_rental_policy():
    net = _net(1, opt_feat=4)
    with torch.no_grad():
        for p in net.t_feat_rental.parameters():
            p.normal_(0, 0.5)
    b, obs = _tactician_obs("rental")
    rng = np.random.default_rng(0)
    obs_f = dict(obs, opt_feat=rng.normal(size=(6, 15, 4)).astype(np.float32))
    x0, x1 = encode.collate([obs], "cpu"), encode.collate([obs_f], "cpu")
    with torch.no_grad():
        ll0, pl0, _ = net.rental_joint(x0)
        ll1, pl1, _ = net.rental_joint(x1)
        f = net.t_feat_rental(x1["opt_feat"])[..., 0]
        legal = x0["pair_mask"] & x0["lead_mask"][:, :, None]
        j0 = (torch.log_softmax(ll0, -1)[:, :, None] + torch.log_softmax(pl0, -1)).masked_fill(~legal, -1e9)
        j1 = (torch.log_softmax(ll1, -1)[:, :, None] + torch.log_softmax(pl1, -1)).masked_fill(~legal, -1e9)
        tilt = torch.log_softmax((j0 + f).masked_fill(~legal, -1e9).flatten(1), -1)
        assert torch.allclose(j1.flatten(1)[legal.flatten(1)], tilt[legal.flatten(1)], atol=1e-4)
        # rental() (lead, then the pair given the lead) is the same distribution
        for lead in range(6):
            if not obs["lead_mask"][lead]:
                continue
            l, p, _, _ = net.rental(x1, lead=torch.tensor([lead]))
            assert torch.allclose(torch.log_softmax(l, -1), torch.log_softmax(ll1, -1), atol=1e-5)
            assert torch.allclose(torch.log_softmax(p, -1)[0][obs["pair_mask"][lead]],
                                  torch.log_softmax(pl1[0, lead], -1)[obs["pair_mask"][lead]], atol=1e-5)


def test_v1_state_dict_loads_and_zero_features_change_nothing():
    v1 = _net(3)
    v2 = _net(4, opt_feat=4)
    v2.load_state_dict(v1.state_dict())                   # new parameters optional
    for kind in ("rental", "swap"):
        _, obs = _tactician_obs(kind, 5)
        shape = (6, 15, 4) if kind == "rental" else (10, 4)
        x0 = encode.collate([obs], "cpu")
        x1 = encode.collate([dict(obs, opt_feat=np.zeros(shape, np.float32))], "cpu")
        with torch.no_grad():
            if kind == "rental":
                a, b_, c = v1.rental_joint(x0)
                d, e, f = v2.rental_joint(x1)
                legal = x0["pair_mask"]
                assert torch.allclose(torch.log_softmax(a, -1)[x0["lead_mask"]],
                                      torch.log_softmax(d, -1)[x0["lead_mask"]], atol=1e-5)
                assert torch.allclose(torch.log_softmax(b_, -1)[legal], torch.log_softmax(e, -1)[legal], atol=1e-5)
                assert torch.allclose(c, f, atol=1e-6)
            else:
                a, c = v1.swap(x0)
                d, f = v2.swap(x1)
                m = x0["mask"]
                assert torch.allclose(torch.log_softmax(a, -1)[m], torch.log_softmax(d, -1)[m], atol=1e-5)
                assert torch.allclose(c, f, atol=1e-6)
    # a v1 checkpoint through rl.policy keeps loading (no opt_feat in its args)
    with pytest.raises(RuntimeError):
        v1.load_state_dict(v2.state_dict())                # (the other way round is not allowed: unexpected keys)


@pytest.mark.parametrize("kind", ["rental", "swap"])
def test_team_features_fill_the_action_space(kind):
    b, obs = _tactician_obs(kind, 11)
    ctx = search.run_ctx(b)
    tf = TF.TeamFeatures("stub:varied", "rl.team_eval_stub", v_next=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    out = tf.compute(b, kind, obs, ctx)
    opts = TS.options_of(kind, obs)
    assert out["options"] == opts and out["feats"].shape == (len(opts), 4)
    legal = (obs["pair_mask"] & obs["lead_mask"][:, None]) if kind == "rental" else obs["mask"]
    assert out["opt_feat"].shape == legal.shape + (4,)
    assert (out["opt_feat"][~legal] == 0).all()
    for o, f in zip(opts, out["feats"]):
        assert np.array_equal(out["opt_feat"][tuple(o) if kind == "rental" else o], f)
    # Q_eval = E_round + W_n^(7-k) W_l v_next, v_next = table[round + 1]
    k = ctx["battle"] + 1
    vn = ctx["challenge"] + 2
    assert np.allclose(out["q_eval"], out["feats"][:, 0] + out["feats"][:, 1] ** (7 - k) * out["feats"][:, 2] * vn)
    ev = STUB.TeamEvaluator.load("stub:varied")
    assert np.isclose(out["phi"], STUB.phi(ev, b.view(), kind, ctx, vn))
    # no evaluator: zeros
    z = TF.TeamFeatures(None).compute(b, kind, obs, ctx)
    assert not z["opt_feat"].any() and z["phi"] == 0.0 and not z["q_eval"].any()
    # the FactoryEnv hook adds exactly "opt_feat"
    h = tf(b, kind, obs, ctx)
    assert set(h) == set(obs) | {"opt_feat"} and np.array_equal(h["opt_feat"], out["opt_feat"])


def test_load_evaluator_accepts_both_load_styles():
    class Inst:
        class TeamEvaluator:
            def load(self, path, device):
                self.path = path

    class Cls:
        TeamEvaluator = STUB.TeamEvaluator

    assert TF.load_evaluator(Inst, "p").path == "p"
    assert isinstance(TF.load_evaluator(Cls, "stub"), STUB.TeamEvaluator)


# ---- the tactician's trajectories -----------------------------------------------------------------------------------

def test_gae_and_v_next():
    r, v = [1.0, 0.5, 2.0], [3.0, 2.0, 1.5]
    adv = V2.gae(r, v, [False, False, True], 0.0, lam=1.0)
    assert np.allclose(adv + v, [3.5, 2.5, 2.0])              # lambda 1: the return
    adv = V2.gae(r, v, [False, False, False], 4.0, lam=0.0)
    assert np.allclose(adv, [1 + 2 - 3, 0.5 + 1.5 - 2, 2 + 4 - 1.5])
    vn = V2.VNext(rate=0.5)
    vn.update([(2, 4.0), (2, 6.0), (3, 1.0), (40, 0.0)])
    assert vn.table[2] == 5.0 and vn.table[3] == 1.0 and vn.seen[len(vn.table) - 1]
    vn.update([(2, 1.0)])
    assert vn.table[2] == 3.0
    assert TF.v_next_of(vn.table, {"challenge": 0}) == 3.0     # round 1 -> table[2]


@pytest.fixture(scope="module")
def v2_selfplay():
    torch.set_num_threads(2)
    v = encode.VERSION
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(search.TRAINING_ENV, "1")
        args = V2.parse(["--sims", "16", "--dets", "2", "--holdout", "0.3", "--team-eval", "stub:varied",
                         "--team-eval-module", "rl.team_eval_stub"])
        sp = V2.SelfPlayerV2(V2.worker_cfg(args), seed=3)
        sp.set_v_next([0.0, 0.0, 2.0, 1.5, 1.0, 0.5, 0.2, 0.0, 0, 0, 0, 0])
        runs = []
        orig = sp._end_run

        def spy(extra=0.0, truncated=False):
            traj = list(sp.traj)                     # (the same dicts: orig closes the last one)
            orig(extra, truncated)
            runs.append(traj + [{"final_wins": sp.run_wins, "extra": extra}])
        sp._end_run = spy
        outs = []
        for _ in range(30):
            out, st = sp.play(80)
            outs.append(out)
            if len(runs) >= 3 and any(o["swap"] is not None for o in outs):
                break
    encode.set_version(v)
    return sp, outs, runs, args


def test_shaping_telescopes_over_a_run(v2_selfplay):
    sp, outs, runs, _ = v2_selfplay
    full = [r for r in runs if r[:-1] and not r[0]["stale"] and r[0]["wins_before"] == 0]
    assert full
    for r in full:
        traj, end = r[:-1], r[-1]
        total = sum(t["reward"] for t in traj)
        # sum of r + Phi(s') - Phi(s) = wins of the run - Phi(s_0) (+ the truncation bootstrap)
        assert np.isclose(total, end["final_wins"] - traj[0]["phi"] + end["extra"], atol=1e-5)
        assert any(t["phi"] != 0 for t in traj)


def test_selfplay_samples(v2_selfplay):
    sp, outs, runs, args = v2_selfplay
    b = AZ.merge([{"battle": o["battle"]} for o in outs])["battle"]
    n = len(b["z"])
    assert b["pi"].shape == (n, 7) and np.allclose(b["pi"].sum(1), 1, atol=1e-5)
    assert (b["pi"][~b["obs"]["mask"]] == 0).all()
    t = V2.merge_t(outs)
    for kind in ("rental", "swap"):
        d = t[kind]
        assert d is not None, kind
        m = len(d["ret"])
        assert d["obs"]["opt_feat"].shape == (m,) + ((6, 15, 4) if kind == "rental" else (10, 4))
        assert d["obs"]["opt_feat"].any()
        assert np.allclose(d["pi_eval"].reshape(m, -1).sum(1), 1, atol=1e-5)
        assert np.allclose(d["ret"], d["adv"] + d["value"], atol=1e-5)
        assert (d["logp"] <= 0).all()
        if kind == "rental":
            legal = d["obs"]["pair_mask"] & d["obs"]["lead_mask"][:, :, None]
            assert legal[np.arange(m), d["action"][:, 0], d["action"][:, 1]].all()
        else:
            assert d["obs"]["mask"][np.arange(m), d["action"]].all()
    assert sum(len(o["v_next"]) for o in outs) > 0


def test_ppo_update_on_selfplay_samples(v2_selfplay):
    sp, outs, runs, args = v2_selfplay
    args.t_imitation, args.t_minibatch = 0.5, 16
    net = _net(5, opt_feat=4)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    norm = V2.ValueNorm(0.3)
    data = V2.merge_t(outs)
    before = [p.detach().clone() for p in net.parameters()]
    out = V2.tactician_ppo(net, opt, data, norm, args, torch.device("cpu"), np.random.default_rng(0))
    for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac", "imitation_ce"):
        assert np.isfinite(out[f"tactician/{k}"]), k
    assert out["tactician/samples"] == sum(int((~d["held"]).sum()) for d in data.values() if d is not None)
    assert any(not torch.equal(a, b) for a, b in zip(before, net.parameters()))
    assert norm.seen
    V2.tactician_heldout(net, data, norm, torch.device("cpu"))


# ---- one tiny run end to end ---------------------------------------------------------------------------------------------

def test_tiny_run_end_to_end(tmp_path, monkeypatch):
    monkeypatch.delenv(search.TRAINING_ENV, raising=False)
    run = str(tmp_path / "az2")
    V2.main(["--run-dir", run, "--iterations", "2", "--decisions-per-iter", "60", "--workers", "2",
             "--inference", "cpu", "--device", "cpu", "--eval-runs", "2", "--eval-rounds", "1",
             "--eval-workers", "1", "--eval-envs-per-worker", "2", "--batch-b", "16", "--sims", "16", "--dets", "2",
             "--holdout", "0.3", "--team-eval", "stub:varied", "--team-eval-module", "rl.team_eval_stub",
             "--t-imitation", "0.1", "--t-minibatch", "8"])
    assert os.environ.get(search.TRAINING_ENV) == "1"
    files = sorted(os.listdir(run))
    assert "latest.pt" in files and len([f for f in files if f.startswith("ckpt_")]) == 2
    ck = torch.load(os.path.join(run, "latest.pt"), map_location="cpu")
    assert ck["args"]["algo"] == "alphazero" and ck["args"]["version"] == 2 and ck["args"]["opt_feat"] == 4
    assert ck["iteration"] == 1 and ck["battler_steps"] >= 120 and "v_next" in ck
    from rl.policy import Policy
    pol = Policy(os.path.join(run, "latest.pt"), torch.device("cpu"))
    assert pol.net.opt_feat == 4
    b, obs = _tactician_obs("rental", 1)
    a = pol.act("rental", encode.collate([dict(obs, opt_feat=np.zeros((6, 15, 4), np.float32))], "cpu"))
    assert a.shape == (1, 2)


# ---- periodic refresh of the team evaluator W (rl/team_eval_refresh.py, docs §19.6) -------------------------------------

from rl import team_eval_refresh as REF  # noqa: E402


def _tiny_team_eval(path, seed=0):
    from rl import team_eval as T
    torch.manual_seed(seed)
    T.TeamEvaluator(T.TeamNet(d=32, d_emb=8, layers=1)).save(str(path))
    return str(path)


def test_refresh_trigger():
    ref = REF.parse_reference("66.0,66.8,41.8,56.6,21.1,19.5")
    assert np.isclose(ref[1], 0.66) and len(ref) == 6
    assert REF.parse_reference("none") is None and REF.parse_reference("0") == {}
    s = REF.RefreshSchedule(ref, every=20, window=3)
    target = np.mean(list(ref.values()))
    low = {k: v - 0.05 for k, v in ref.items()}
    high = {k: v + 0.02 for k, v in ref.items()}
    for it in range(5):
        s.observe(it, low)
        assert not s.due(it)
    s.observe(5, high)
    s.observe(6, high)
    assert s.armed_it is None                    # mean of the last 3: (low + 2 high) / 3 < reference
    s.observe(7, {k: v + 0.2 for k, v in ref.items()})
    assert s.armed_it == 7 and s.running_mean() >= target
    assert s.due(7)
    s.done(7)
    assert not s.due(8) and not s.due(26) and s.due(27)
    # only the rounds evaluated are compared (with the same rounds of the reference)
    s2 = REF.RefreshSchedule(ref, every=1, window=1)
    s2.observe(0, {3: 0.5})
    assert s2.armed_it == 0
    s3 = REF.RefreshSchedule(None, every=20)
    s3.observe(0, high)
    assert not s3.enabled and not s3.due(0)
    s4 = REF.RefreshSchedule(ref, every=20, window=3)
    s4.load(s.state())
    assert s4.armed_it == 7 and s4.last_it == 7 and s4.count == 1 and s4.due(27) and not s4.due(26)


def test_blend_params(tmp_path):
    from rl import team_eval as T
    a = T.TeamEvaluator.load(_tiny_team_eval(tmp_path / "a.pt", 0))
    b = T.TeamEvaluator.load(_tiny_team_eval(tmp_path / "b.pt", 1))
    for alpha, ref in ((0.0, a), (1.0, b)):
        m = REF.blend_params(a, b, alpha)
        for k, v in m.net.state_dict().items():
            assert torch.allclose(v, ref.net.state_dict()[k]), k
    m = REF.blend_params(a, b, 0.5)
    k = next(n for n, _ in m.net.named_parameters())
    assert torch.allclose(m.net.state_dict()[k], 0.5 * (a.net.state_dict()[k] + b.net.state_dict()[k]))


def _refresh_args(te, extra=()):
    return V2.parse(["--sims", "8", "--dets", "2", "--team-eval", te, "--w-refresh-battles", "48",
                     "--w-refresh-battles-per-team", "4", "--w-refresh-concurrency", "16", "--w-refresh-probe", "8",
                     "--w-refresh-real-iters", "0", "--w-refresh-epochs", "1", "--w-refresh-batch", "8",
                     *extra])


def test_refresh_swaps_the_evaluator_in_the_workers_and_is_deterministic(tmp_path, monkeypatch):
    monkeypatch.setenv(search.TRAINING_ENV, "1")        # factory_sets opponents: training only
    torch.set_num_threads(2)
    te = _tiny_team_eval(tmp_path / "te.pt")
    args = _refresh_args(te)
    assert V2.refresh_enabled(args)
    assert not V2.refresh_enabled(_refresh_args("stub", ["--team-eval-module", "rl.team_eval_stub"]))
    probe = REF.probe_teams(8, seed=3)
    pool = AZ.WorkerPool(2, V2.worker_cfg(args), None, seed=1, player=V2.SelfPlayerV2)
    try:
        assert pool.broadcast("call", ("team_eval_path", ())) == [te, te]
        outs = []
        for name in ("r1.pt", "r2.pt"):
            path = str(tmp_path / name)
            m = V2.run_refresh(pool, 2, te, path, [], args, 3, "cpu", probe)
            assert pool.broadcast("call", ("team_eval_path", ())) == [path, path]
            outs.append((path, m))
        assert outs[0][1]["battles"] >= 48 and outs[0][1]["heldout_teams"] >= 1
        for k in ("brier_before", "brier_after", "bce_before", "bce_after", "probe_abs_dw"):
            assert np.isfinite(outs[0][1][k]), k
            assert outs[0][1][k] == outs[1][1][k], k            # same seeds: the same refresh
        from rl import team_eval as T
        a, b, old = (T.TeamEvaluator.load(p) for p in (outs[0][0], outs[1][0], te))
        sa, sb, so = a.net.state_dict(), b.net.state_dict(), old.net.state_dict()
        assert all(torch.equal(sa[k], sb[k]) for k in sa)
        assert any(not torch.equal(sa[k], so[k]) for k in sa)
        assert a.meta["refreshed_from"] == te and a.meta["refresh_iteration"] == 3
        # the new evaluator gives other tactician features
        b0, obs = _tactician_obs("swap", 11)
        f_new = V2.TeamFeatures(outs[0][0]).compute(b0, "swap", obs, search.run_ctx(b0))["opt_feat"]
        f_old = V2.TeamFeatures(te).compute(b0, "swap", obs, search.run_ctx(b0))["opt_feat"]
        assert not np.allclose(f_new, f_old)
    finally:
        pool.close()


def test_real_battle_log(v2_selfplay):
    sp, outs, runs, args = v2_selfplay
    d = REF.merge_arrays([o.get("real") for o in outs])
    assert d is not None and len(d["round"]) == len(d["b_team"]) >= 3
    assert set(np.unique(d["btype"])) <= {0, 1, 2} and (d["battle"] >= 1).all() and (d["battle"] <= 7).all()
    assert (d["species"] > 0).all() and d["stats"].shape == (len(d["round"]), 3, 6)
    assert np.array_equal(d["b_team"], np.arange(len(d["round"])))


def test_tiny_run_with_refresh_and_resume(tmp_path, monkeypatch):
    monkeypatch.delenv(search.TRAINING_ENV, raising=False)
    te = _tiny_team_eval(tmp_path / "te.pt")
    run = str(tmp_path / "az2w")
    common = ["--run-dir", run, "--decisions-per-iter", "60", "--workers", "2", "--inference", "cpu", "--device",
              "cpu", "--eval-runs", "2", "--eval-rounds", "1", "--eval-workers", "1", "--eval-envs-per-worker", "2",
              "--batch-b", "16", "--sims", "16", "--dets", "2", "--holdout", "0.3", "--team-eval", te,
              "--t-minibatch", "8", "--eval-search-every", "0", "--w-refresh-after", "0", "--w-refresh-every", "2",
              "--w-refresh-window", "1", "--w-refresh-battles", "48", "--w-refresh-battles-per-team", "4",
              "--w-refresh-concurrency", "16", "--w-refresh-probe", "8"]
    V2.main(common + ["--iterations", "3"])
    files = os.listdir(run)
    assert "team_eval_0000.pt" in files and "team_eval_0002.pt" in files
    ck = torch.load(os.path.join(run, "latest.pt"), map_location="cpu", weights_only=False)
    assert ck["team_eval"].endswith("team_eval_0002.pt")
    assert ck["w_refresh"]["count"] == 2 and ck["w_refresh"]["last_it"] == 2 and ck["w_refresh"]["armed_it"] == 0
    # ckpt_<steps> is written before the iteration's refresh (the W its self-play used), latest.pt after it
    cks = [torch.load(os.path.join(run, f), map_location="cpu", weights_only=False)
           for f in sorted(f for f in files if f.startswith("ckpt_"))]
    assert cks[0]["team_eval"] == te and cks[0]["w_refresh"]["count"] == 0
    assert cks[1]["team_eval"].endswith("team_eval_0000.pt") and cks[1]["w_refresh"]["count"] == 1
    # resume: the refreshed W and the schedule come back (next refresh at iteration 4)
    V2.main(common + ["--iterations", "5", "--init-from", os.path.join(run, "latest.pt")])
    ck = torch.load(os.path.join(run, "latest.pt"), map_location="cpu", weights_only=False)
    assert ck["iteration"] == 4 and ck["w_refresh"]["count"] == 3
    assert ck["team_eval"].endswith("team_eval_0004.pt") and "team_eval_0003.pt" not in os.listdir(run)
    from rl import team_eval as T
    assert T.TeamEvaluator.load(ck["team_eval"]).meta["refreshed_from"].endswith("team_eval_0002.pt")

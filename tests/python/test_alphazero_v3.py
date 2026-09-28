"""alphazero_v3 (rl/alphazero_v3.py, docs/RL_DECISIONS.md §20): the warmup-stable-decay learning rate, encoding
version 5 (hints in the tactician's input), per-option features of width N_OPT_FEATURES from the v3 evaluator API
(rl/az3_stub.py until the real module is merged), the tactician's PPO options, and a tiny run end to end."""

import os
import sys
import types
from argparse import Namespace

import numpy as np
import pytest
import torch

from rl import alphazero_v2 as V2
from rl import alphazero_v3 as V3
from rl import az3_stub as S3
from rl import encode, gumbel as G, search
from rl import tactician_search as TS
from rl.model import FactoryNet

sys.path.insert(0, os.path.dirname(__file__))
from test_alphazero_v2 import _tactician_obs  # noqa: E402

pytestmark = pytest.mark.skipif(G.halving_plan is None or not hasattr(search.NativeSearcher, "search_gumbel"),
                                reason="needs the extension's Gumbel search (rebuild)")

V2_CKPT = os.path.join(os.path.dirname(__file__), "..", "..", "runs", "alphazero_v2", "latest.pt")


@pytest.fixture(autouse=True)
def _keep_encode_version():
    v = encode.VERSION
    yield
    encode.set_version(v)


# ---- 1. warmup-stable-decay ------------------------------------------------------------------------------------------

def _wsd_args(**kw):
    a = dict(lr=1e-3, iterations=100, lr_warmup=3, decay_frac=0.15, lr_decay="cosine", lr_min_frac=0.05)
    a.update(kw)
    return Namespace(**a)


@pytest.mark.parametrize("shape", ["cosine", "linear"])
def test_wsd_values(shape):
    s = V3.WSD(1e-3, 100, warmup=3, decay_frac=0.15, shape=shape, min_frac=0.05)
    assert s.decay_start == 85
    assert np.allclose([s.lr(0), s.lr(1), s.lr(2)], [2.5e-4, 5e-4, 7.5e-4])       # warmup: lr (it + 1) / 4
    assert all(s.lr(it) == 1e-3 for it in range(3, 85))                           # stable
    dec = [s.lr(it) for it in range(85, 100)]
    assert all(a > b for a, b in zip(dec, dec[1:])) and dec[0] < 1e-3            # strictly decaying
    assert np.isclose(s.lr(99), 5e-5) and np.isclose(s.lr(150), 5e-5)             # the floor at the last iteration
    mid = s.lr(85 + 15 // 2 - 1)                                                   # frac = 7/15
    f = 7 / 15
    want = 5e-5 + 0.5 * (1e-3 - 5e-5) * (1 + np.cos(np.pi * f)) if shape == "cosine" else 1e-3 - (1e-3 - 5e-5) * f
    assert np.isclose(mid, want)
    assert V3.WSD(1e-3, 600).decay_start == 510                                   # the default plan: 90 decaying
    assert V3.WSD(1e-3, 10, decay_frac=0.0).lr(9) == 1e-3


def test_wsd_extension_on_resume():
    s, note = V3.WSD.plan(_wsd_args(), None, 0)
    assert s.decay_start == 85 and note is None
    # resumed before the decay with more iterations: the decay moves to the new end
    s2, note = V3.WSD.plan(_wsd_args(iterations=200), s.state(), 60)
    assert s2.decay_start == 170 and s2.iterations == 200 and note is None
    assert s2.lr(100) == 1e-3 and s2.lr(60) == s.lr(60)
    # still possible at the last stable iteration (iteration 85 not trained yet)
    assert V3.WSD.plan(_wsd_args(iterations=200), s.state(), 85)[0].decay_start == 170
    # the decay started (iteration 85 trained): the checkpoint's plan is kept
    s3, note = V3.WSD.plan(_wsd_args(iterations=200), s.state(), 86)
    assert s3.state() == s.state() and note and "decay started" in note
    assert s3.lr(86) == s.lr(86) and np.isclose(s3.lr(150), 5e-5)
    # same plan, resumed during the decay: no note, the same values
    s4, note = V3.WSD.plan(_wsd_args(), s.state(), 90)
    assert s4.state() == s.state() and note is None
    # a shorter plan whose decay would already be under way: it starts at the resumed iteration
    s5, note = V3.WSD.plan(_wsd_args(iterations=70), s.state(), 65)
    assert s5.decay_start == 65 and s5.lr(64) == 1e-3 and np.isclose(s5.lr(69), 5e-5) and note


# ---- 2. hints in the tactician's observation (encoding version 5) ----------------------------------------------------

def _hint_slice(obs):
    return obs["ctx_num"][encode.FIELD_NUM - encode.HINT_NUM:encode.FIELD_NUM]


@pytest.mark.parametrize("kind", ["rental", "swap"])
def test_v5_hints_in_tactician_obs(kind):
    encode.set_version(5)
    b, obs5 = _tactician_obs(kind, 3)
    v, ctx = b.view(), search.run_ctx(b)
    h = _hint_slice(obs5)
    assert h.sum() == 2 and h[v.hint_type] == 1 and h[encode.N_HINT_TYPES + v.hint_style] == 1
    assert (obs5["ctx_num"][:encode.FIELD_NUM - encode.HINT_NUM] == 0).all()
    encode.set_version(4)
    obs4 = encode.rental(v, ctx) if kind == "rental" else encode.swap(v, ctx)
    assert not _hint_slice(obs4).any()                          # v4: no hints for the tactician
    for k in obs4:                                              # and nothing else changes
        a, c = obs4[k], obs5[k]
        if k == "ctx_num":
            a, c = np.delete(a, np.s_[encode.FIELD_NUM - encode.HINT_NUM:encode.FIELD_NUM]), \
                np.delete(c, np.s_[encode.FIELD_NUM - encode.HINT_NUM:encode.FIELD_NUM])
        assert a.shape == c.shape and np.array_equal(a, c), k


def test_v5_no_hint_and_layout():
    encode.set_version(4)
    sizes4 = (encode.MON_NUM, encode.MOVE_NUM, encode.CTX_NUM)
    encode.set_version(5)
    assert (encode.MON_NUM, encode.MOVE_NUM, encode.CTX_NUM) == sizes4
    b, obs = _tactician_obs("rental", 3)
    v = b.view()
    v.hint_type, v.hint_style = 18, 0                           # the attendant says nothing (before Noland)
    h = _hint_slice(encode.rental(v, search.run_ctx(b)))
    assert h[18] == 1 and h[encode.N_HINT_TYPES] == 1 and h.sum() == 2
    from pybattle import pybattle_native as N
    if hasattr(N, "encode_version"):
        assert N.encode_version() == 4                          # the C++ battle encoder runs as v4
    assert encode.native_version(5) == 4 and 5 in encode.CPP_VERSIONS


def test_v5_battle_obs_is_v4():
    from test_alphazero_v2 import _battle_backend
    b = _battle_backend()
    ctx = search.run_ctx(b)
    encode.set_version(4)
    o4 = encode.battle(b.view(), ctx)
    encode.set_version(5)
    o5 = encode.battle(b.view(), ctx)
    assert all(np.array_equal(o4[k], o5[k]) for k in o4)


@pytest.mark.skipif(not os.path.exists(V2_CKPT), reason="runs/alphazero_v2/latest.pt not present")
def test_v2_checkpoint_is_hint_safe():
    """alphazero_v2's checkpoint keeps loading and acting (encoding 4: no hints), and initializes a v3 net."""
    from rl.policy import Policy
    pol = Policy(V2_CKPT, torch.device("cpu"))
    assert encode.VERSION == 4 and pol.net.opt_feat == 4
    b, obs = _tactician_obs("rental", 3)
    assert not _hint_slice(obs).any()
    a = pol.act("rental", encode.collate([dict(obs, opt_feat=np.zeros((6, 15, 4), np.float32))], "cpu"))
    assert a.shape == (1, 2)
    ck = torch.load(V2_CKPT, map_location="cpu", weights_only=False)
    encode.set_version(5)
    net = FactoryNet(ck["args"]["d_emb"], ck["args"]["d"], ck["args"]["layers"], ck["args"]["heads"],
                     share="embeddings", opt_feat=S3.N_OPT_FEATURES)
    init = {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(ck["net"])                              # the 4-wide option layers keep their initialization
    for k, v in net.state_dict().items():
        want = init[k] if k.startswith("t_feat_") and ck["net"][k].shape != v.shape else ck["net"][k]
        assert torch.equal(v, want), k
    assert V3.features_hook(ck) is None                         # (a v2 checkpoint gets no v3 hook)


# ---- 3. per-option features of width N_OPT_FEATURES ------------------------------------------------------------------

def _fake_module(n):
    """A v3 evaluator module with n features (the stub's 6 and zeros after)."""
    m = types.ModuleType(f"fake_v3_{n}")
    m.N_OPT_FEATURES = n
    m.TeamEvaluator = S3.TeamEvaluator

    def feats(ev, view, kind, ctx, options, table, hint, cfg):
        f = S3.option_features_v3(ev, view, kind, ctx, options, table, hint, cfg)
        return np.concatenate([f, np.full((len(f), n - 6), 0.25, np.float32)], 1)
    m.option_features_v3 = feats
    m.phi_v3 = S3.phi_v3
    m.q_eval_v3 = S3.q_eval_v3
    sys.modules[m.__name__] = m
    return m.__name__


@pytest.mark.parametrize("kind", ["rental", "swap"])
def test_opt_feat_width_follows_n_opt_features(kind):
    name = _fake_module(9)
    assert V3.resolve_module(name, "stub") == name
    import importlib
    mod = importlib.import_module(name)
    assert V3.n_opt_features(mod) == 9
    ev = S3.TeamEvaluator.load("stub")
    ev.N_OPT_FEATURES = 11                                     # the loaded evaluator (its checkpoint) wins
    assert V3.n_opt_features(mod, ev) == 11
    encode.set_version(5)
    b, obs = _tactician_obs(kind, 11)
    tf = V3.TeamFeaturesV3("stub:varied", name, 9, v_next=list(range(12)))
    out = tf.compute(b, kind, obs, search.run_ctx(b))
    opts = TS.options_of(kind, obs)
    assert out["feats"].shape == (len(opts), 9)
    assert out["opt_feat"].shape == ((6, 15, 9) if kind == "rental" else (10, 9))
    assert np.allclose(out["q_eval"], out["feats"][:, 4])
    v = b.view()
    assert out["hint"] == (v.hint_type, v.hint_style)
    with pytest.raises(ValueError):                            # a network of another width is refused
        V3.TeamFeaturesV3("stub:varied", name, 6).compute(b, kind, obs, search.run_ctx(b))
    net = FactoryNet(share="embeddings", opt_feat=9).eval()
    x = encode.collate([dict(obs, opt_feat=out["opt_feat"])], "cpu")
    with torch.no_grad():
        (net.rental_joint if kind == "rental" else net.swap)(x)
    assert net.t_feat_rental[0].in_features == 9 and net.t_feat_value.in_features == 18
    z = V3.TeamFeaturesV3(None, name, 9).compute(b, kind, obs, search.run_ctx(b))
    assert z["opt_feat"].shape[-1] == 9 and not z["opt_feat"].any() and z["phi"] == 0.0


def test_stub_features_and_hint():
    encode.set_version(5)
    b, obs = _tactician_obs("swap", 11)
    ctx = search.run_ctx(b)
    ev = S3.TeamEvaluator.load("stub:varied")
    opts = TS.options_of("swap", obs)
    tab = np.arange(12, dtype=float)
    f = S3.option_features_v3(ev, b.view(), "swap", ctx, opts, tab, (18, 0), {})
    assert f.shape == (len(opts), S3.N_OPT_FEATURES)
    assert np.allclose(f[:, 5], f[:, 4] - f[:, 4].mean())
    g = S3.option_features_v3(ev, b.view(), "swap", ctx, opts, tab, (9, 3), {})
    assert not np.allclose(f[:, 1], g[:, 1])                  # the hint moves W(normal) in the stub
    phi = S3.phi_v3(ev, b.view(), "swap", ctx, tab, (18, 0), {})
    assert np.isclose(phi, f[0, 4])                            # Phi(s) = Q_plan of keeping the team (the stub)


# ---- 4. the tactician's PPO: round weights, adaptive epochs ----------------------------------------------------------

def test_round_weights():
    r = np.array([1, 1, 5, 6, 8])
    assert (V3.round_weights(r, {1: 0.8, 5: 0.3, 6: 0.2}, 0.0) == 1).all()
    w = V3.round_weights(r, {1: 0.8, 5: 0.3, 6: 0.2}, 1.0)
    assert np.isclose(w.mean(), 1.0) and w[2] > w[0] and w[3] == w[4]      # round 8 counts as round 6
    w5 = V3.round_weights(r, {1: 0.8, 5: 0.3, 6: 0.2}, 0.5)
    assert np.isclose(w5.mean(), 1.0) and np.allclose(w5, 0.5 + 0.5 * w)
    assert (V3.round_weights(r, {}, 1.0) == 1).all()                         # nothing evaluated: uniform


@pytest.fixture(scope="module")
def v3_selfplay():
    torch.set_num_threads(2)
    v = encode.VERSION
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(search.TRAINING_ENV, "1")
        args = V3.parse(["--sims", "16", "--dets", "2", "--holdout", "0.3", "--team-eval", "stub:varied"])
        args.team_eval_module = V3.resolve_module(None, args.team_eval)
        sp = V3.SelfPlayerV3(V3.worker_cfg(args, S3.N_OPT_FEATURES), seed=3)
        sp.set_v_next([0.0, 0.0, 2.0, 1.5, 1.0, 0.5, 0.2, 0.0, 0, 0, 0, 0])
        outs = []
        for _ in range(30):
            out, st = sp.play(80)
            outs.append(out)
            if sum(o["rental"] is not None for o in outs) >= 2 and any(o["swap"] is not None for o in outs):
                break
    encode.set_version(v)
    return sp, outs, args


def test_v3_selfplay_samples(v3_selfplay):
    sp, outs, args = v3_selfplay
    assert args.team_eval_module == V3.STUB_MODULE and sp.net.opt_feat == S3.N_OPT_FEATURES
    t = V2.merge_t(outs)
    for kind in ("rental", "swap"):
        d = t[kind]
        assert d is not None, kind
        m = len(d["ret"])
        assert d["obs"]["opt_feat"].shape == (m,) + ((6, 15, 6) if kind == "rental" else (10, 6))
        assert d["obs"]["opt_feat"].any()
        assert (_hint_slice_batch(d["obs"]).sum(1) == 2).all()           # every tactician sample has the hints
        assert np.allclose(d["pi_eval"].reshape(m, -1).sum(1), 1, atol=1e-5)
        assert ((d["round"] >= 1) & (d["round"] <= 12)).all()


def _hint_slice_batch(obs):
    return obs["ctx_num"][:, encode.FIELD_NUM - encode.HINT_NUM:encode.FIELD_NUM]


def test_v3_ppo_adaptive_epochs_and_weights(v3_selfplay):
    sp, outs, args = v3_selfplay
    encode.set_version(5)
    data = V2.merge_t(outs)
    V3.add_round_weights(data, {1: 0.9, 2: 0.5, 3: 0.1}, 1.0)
    ws = np.concatenate([d["weight"] for d in data.values() if d is not None])
    assert np.isclose(ws.mean(), 1.0) and ws.std() > 0
    runs = {}
    for target, max_ep in ((0.0, 4), (1e9, 4), (1e-12, 4)):
        a = Namespace(**vars(args))
        a.t_minibatch, a.t_kl_target, a.t_max_epochs, a.t_imitation = 16, target, max_ep, 0.5
        torch.manual_seed(5)
        net = FactoryNet(share="embeddings", opt_feat=6).eval()
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
        out = V2.tactician_ppo(net, opt, data, V2.ValueNorm(0.3), a, torch.device("cpu"), np.random.default_rng(0))
        runs[target] = out
        for k in ("policy_loss", "value_loss", "approx_kl", "imitation_ce"):
            assert np.isfinite(out[f"tactician/{k}"]), k
        assert out["tactician/weight_max"] > out["tactician/weight_min"]
    assert "tactician/epochs" not in runs[0.0]                              # target 0: exactly --t-ppo-epochs
    assert runs[1e9]["tactician/epochs"] == 4                                # KL always below: up to the maximum
    assert runs[1e-12]["tactician/epochs"] == 2                              # KL above after the base epochs
    assert runs[1e9]["tactician/ppo_steps"] == 2 * runs[1e-12]["tactician/ppo_steps"]


def test_v2_ppo_unchanged_without_v3_options(v3_selfplay):
    """v2's args (no t_kl_target, no weights): the same result as before the generalization (2 epochs)."""
    sp, outs, args = v3_selfplay
    encode.set_version(5)
    data = V2.merge_t(outs)
    for d in data.values():
        if d is not None:
            d.pop("weight", None)
    a = V2.parse(["--t-minibatch", "16"])
    assert not hasattr(a, "t_kl_target")
    torch.manual_seed(5)
    net = FactoryNet(share="embeddings", opt_feat=6).eval()
    out = V2.tactician_ppo(net, torch.optim.AdamW(net.parameters(), lr=1e-3), data, V2.ValueNorm(0.3), a,
                           torch.device("cpu"), np.random.default_rng(0))
    n = out["tactician/samples"]
    assert out["tactician/ppo_steps"] == 2 * max(1, -(-n // 16)) and "tactician/epochs" not in out


# ---- 5. a tiny run end to end, and a resume that lengthens it --------------------------------------------------------

def test_tiny_run_end_to_end(tmp_path, monkeypatch):
    monkeypatch.delenv(search.TRAINING_ENV, raising=False)
    run = str(tmp_path / "az3")
    common = ["--run-dir", run, "--decisions-per-iter", "60", "--workers", "2", "--inference", "cpu", "--device",
              "cpu", "--eval-runs", "2", "--eval-rounds", "1", "--eval-workers", "1", "--eval-envs-per-worker", "2",
              "--batch-b", "16", "--sims", "16", "--dets", "2", "--holdout", "0.3", "--team-eval", "stub:varied",
              "--t-imitation", "0.1", "--t-minibatch", "8", "--t-round-weight", "0.5", "--lr-warmup", "1",
              "--decay-frac", "0.5"]
    V3.main(common + ["--iterations", "2"])
    files = sorted(os.listdir(run))
    assert "latest.pt" in files and len([f for f in files if f.startswith("ckpt_")]) == 2
    ck = torch.load(os.path.join(run, "latest.pt"), map_location="cpu", weights_only=False)
    a = ck["args"]
    assert a["algo"] == "alphazero" and a["version"] == 3 and a["opt_feat"] == S3.N_OPT_FEATURES
    assert a["encode_version"] == 5 and a["team_eval_module"] == V3.STUB_MODULE
    assert ck["iteration"] == 1 and ck["battler_steps"] >= 120 and "v_next" in ck
    assert ck["lr_schedule"]["decay_start"] == 1 and ck["lr_schedule"]["iterations"] == 2
    from rl.policy import Policy
    pol = Policy(os.path.join(run, "latest.pt"), torch.device("cpu"))
    assert pol.net.opt_feat == 6 and encode.VERSION == 5
    hook = V3.features_hook(os.path.join(run, "latest.pt"))
    b, obs = _tactician_obs("rental", 1)
    obs = hook(b, "rental", obs, search.run_ctx(b))
    assert obs["opt_feat"].shape == (6, 15, 6) and obs["opt_feat"].any()
    assert pol.act("rental", encode.collate([obs], "cpu")).shape == (1, 2)
    # resume with more iterations: the decay had started (iteration 1 trained), so the plan is kept
    V3.main(common + ["--iterations", "3", "--init-from", os.path.join(run, "latest.pt")])
    ck = torch.load(os.path.join(run, "latest.pt"), map_location="cpu", weights_only=False)
    assert ck["iteration"] == 2 and ck["lr_schedule"]["iterations"] == 2

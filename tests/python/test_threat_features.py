"""Threat features from the learned opponent model (rl/threat_features.py, docs/RL_DECISIONS.md §27.x): the
information rule (source scan; two states that look the same to the player give the same features), the closed-form
novel-atom formulas against brute force, cache consistency, and ThreatNet (a FactoryNet state_dict loads into it and
computes the same function until the threat weights move)."""

import itertools
import math
import os
import random
import sys

import numpy as np
import pytest
import torch

from pybattle.backend import Phase
from pybattle.emu.decode import SYMBOLS as S, decode_party
from rl import encode
from rl import opponent_model as OM
from rl import threat_features as TF
from rl.model import FactoryNet

sys.path.insert(0, os.path.dirname(__file__))
from test_opponent_model import BANNED, _battle_start, _code_only, _overwrite_hidden  # noqa: E402
from test_search import _random_action  # noqa: E402

MODEL = os.path.join(os.path.dirname(__file__), "..", "..", "runs", "opp_model_v3", "model.pt")


@pytest.fixture(autouse=True)
def _keep_encode_version():
    v = encode.VERSION
    yield
    encode.set_version(v)


@pytest.fixture(scope="module")
def model():
    if os.path.exists(MODEL):
        return OM.ExperienceModel.load(MODEL)
    pytest.skip("runs/opp_model_v3/model.pt missing")


# ---- the information rule -------------------------------------------------------------------------------------------

def test_source_never_reads_the_true_opponent_or_the_set_list():
    src = _code_only(TF)
    for name in BANNED + ("factory_sets", "SET_SPECIES", "SET_MOVES", "SET_ITEM", "gBattleMons"):
        assert name not in src, name
    # the featurizer itself (everything before the dataset / benchmark harness) never touches a backend or IVs / EVs
    feat_src = src.split("V4_CKPT =")[0]
    assert "class ThreatFeaturizer" in feat_src
    for name in (".ivs", ".evs", "backend", "_observer", ".be."):
        assert name not in feat_src, name


def test_features_are_invariant_to_the_hidden_opponent(model):
    """Along random battles, at every decision with an unseen opponent slot: overwrite what the player cannot see
    (the set ids, the unseen party slots) in a copy of the state; the views are equal and so are the features
    (computed by separate featurizers with cold caches)."""
    n_compared = 0
    for seed, streak in ((8, 28), (11, 35), (5, 14), (21, 0)):
        b = _battle_start(seed, streak)
        ch = b.run_info().challenge_num
        fb, fc = TF.ThreatFeaturizer(model), TF.ThreatFeaturizer(model)
        rng = random.Random(seed)
        for _ in range(40):
            if b.phase not in (Phase.BATTLE, Phase.FORCED_SWITCH):
                break
            vb = b.view()
            if any(not m.seen for m in vb.enemy_party[:3]):
                c = _overwrite_hidden(b.clone())
                assert decode_party(b.game.read(S.addr("gEnemyParty"), 300)) != \
                    decode_party(c.game.read(S.addr("gEnemyParty"), 300))
                vc = c.view()
                assert vb == vc
                assert np.array_equal(fb(vb, ch), fc(vc, ch))
                n_compared += 1
            b.act(vb.legal_actions[rng.randrange(len(vb.legal_actions))])
    assert n_compared >= 10


def test_features_depend_only_on_the_view(model):
    """The featurizer's only inputs are the view, the challenge number and the model: the same view object gives
    the same features from a fresh featurizer and from a warm one (caches keyed by what the player knows)."""
    b = _battle_start()
    ch = b.run_info().challenge_num
    rng = random.Random(1)
    warm = TF.ThreatFeaturizer(model)
    for _ in range(25):
        if b.phase not in (Phase.BATTLE, Phase.FORCED_SWITCH):
            break
        v = b.view()
        f = warm(v, ch)
        assert f.shape == (6, TF.N_THREAT) and f.dtype == np.float32 and np.isfinite(f).all()
        assert np.array_equal(f, TF.ThreatFeaturizer(model)(v, ch))
        b.act(v.legal_actions[rng.randrange(len(v.legal_actions))])


def test_layout_and_ranges(model):
    b = _battle_start()
    v = b.view()
    f = TF.ThreatFeaturizer(model)(v, b.run_info().challenge_num)
    probs = [0, 2, 3, 5]
    for i in range(6):
        assert (f[i, probs] >= -1e-6).all() and (f[i, probs] <= 1 + 1e-6).all()
        assert (f[i, [1, 4]] <= TF.CAP + 1e-6).all() and (f[i, [1, 4]] >= 0).all()
    assert (f[:3, 2 * TF.PF:] == 0).all()                      # own tokens: no per-slot block
    for j, m in enumerate(v.enemy_party[:3]):
        assert (f[3 + j, 2 * TF.PF:2 * TF.PF + 4] <= 1 + 1e-6).all()
        if m.seen and not m.fainted:
            assert f[3 + j, 0:6].any()
    # own active vs active foe is the same pair in both tokens
    ia, ja = v.own_active.party_index, v.enemy_active.party_index
    assert np.array_equal(f[ia, :TF.PF], f[3 + ja, :TF.PF])


def test_revealed_move_sets_p_se(model):
    """A revealed super-effective move makes p_se exactly 1 for the target."""
    b = _battle_start()
    v = b.view()
    foe = v.enemy_party[v.enemy_active.party_index]
    me = v.own_party[v.own_active.party_index]
    from rl.gamedata import MOVES
    se = [m for m in range(1, len(MOVES)) if TF.M_DAMAGING[m] and TF.M_FIXED[m] == 0
          and TF.defender_mult(me.types, [me.ability])[1][TF.M_TYPE[m]] > 0 and TF.M_POW[m] > 0]
    foe.revealed_moves = [se[0]]
    f = TF.ThreatFeaturizer(model)(v, b.run_info().challenge_num)
    assert f[v.own_active.party_index, 0] == pytest.approx(1.0)


def test_empty_model_is_all_novel():
    b = _battle_start()
    v = b.view()
    f = TF.ThreatFeaturizer(OM.ExperienceModel())(v, b.run_info().challenge_num)
    for j, m in enumerate(v.enemy_party[:3]):
        if not (m.seen and m.fainted):
            assert f[3 + j, -1] == pytest.approx(1.0)               # p_novel


# ---- closed forms ---------------------------------------------------------------------------------------------------

def test_expected_max_and_p_none_match_enumeration():
    rng = np.random.default_rng(0)
    for n, k in ((7, 3), (5, 4), (6, 1), (9, 2), (4, 4)):
        vals = rng.random(n) * 2
        m0 = 0.7
        subsets = list(itertools.combinations(range(n), k))
        brute = np.mean([max(m0, max(vals[list(s)])) for s in subsets])
        srt = -np.sort(-vals)[None]
        got = TF.expected_max(np.concatenate([srt, -np.ones((1, 3))], 1), np.array([n]), np.array([k]),
                              np.array([m0]))[0]
        assert got == pytest.approx(brute)
        q = 3
        brute_none = np.mean([all(i >= q for i in s) for s in subsets])
        assert TF.p_none(n, q, k) == pytest.approx(brute_none)


def test_speed_table_matches_the_formula():
    gt, eq = TF.speed_table(100)
    sp = 6
    base = TF.BASE[sp, 3]
    speeds = [math.floor(math.floor((2 * base + iv + ev // 4) * 100 / 100 + 5) * 1.1)
              for iv in range(32) for ev in range(0, 253, 4)]
    x = 300
    assert gt[sp, 0, x] == pytest.approx(np.mean([s > x for s in speeds]), abs=1e-6)
    assert eq[sp, 0, x] == pytest.approx(np.mean([s == x for s in speeds]), abs=1e-6)


# ---- the network ----------------------------------------------------------------------------------------------------

def _batch(view, ch, feats):
    from rl.search import run_ctx  # noqa: F401
    ctx = {"streak": 7 * ch, "battle": 0, "challenge": ch, "rents": ch}
    o = encode.battle(view, ctx)
    x = encode.collate([o], "cpu")
    x["threat"] = torch.from_numpy(feats[None])
    return x


def test_threatnet_loads_a_factorynet_and_starts_identical(model):
    encode.set_version(5)
    torch.manual_seed(0)
    base = FactoryNet(share="embeddings", opt_feat=7).eval()
    net = TF.ThreatNet(share="embeddings", opt_feat=7).eval()
    net.load_state_dict(base.state_dict())
    b = _battle_start()
    v = b.view()
    ch = b.run_info().challenge_num
    x = _batch(v, ch, TF.ThreatFeaturizer(model)(v, ch))
    with torch.no_grad():
        l0, v0 = base.battler(x)
        l1, v1 = net.battler(x)
        assert torch.allclose(l0, l1) and torch.allclose(v0, v1)
        net.trunk.mon_enc[0].weight[:, -TF.N_THREAT:] = 0.5
        l2, _ = net.battler(x)
        assert not torch.allclose(l0, l2)
        x0 = dict(x)
        del x0["threat"]
        l3, _ = net.battler(x0)                                    # no features = zeros
        x0["threat"] = torch.zeros_like(x["threat"])
        assert torch.allclose(l3, net.battler(x0)[0])
    # a ThreatNet state_dict round-trips
    net2 = TF.ThreatNet(share="embeddings", opt_feat=7)
    net2.load_state_dict(net.state_dict())
    assert torch.equal(net2.trunk.mon_enc[0].weight, net.trunk.mon_enc[0].weight)


def test_threatnet_shared_trunk_tactician_unchanged():
    encode.set_version(5)
    torch.manual_seed(0)
    base = FactoryNet(share="all").eval()
    net = TF.ThreatNet(share="all").eval()
    net.load_state_dict(base.state_dict())
    from pybattle.backend import SimBackend
    be = SimBackend(max_turns=10 ** 9)
    be.reset(seed=3, win_streak=0, rents_count=0)
    assert be.phase == Phase.RENTAL
    ctx = {"streak": 0, "battle": 0, "challenge": 0, "rents": 0}
    x = encode.collate([encode.rental(be.view(), ctx)], "cpu")
    with torch.no_grad():
        a = base.rental_joint(x)
        c = net.rental_joint(x)
    assert all(torch.allclose(p, q) for p, q in zip(a, c))

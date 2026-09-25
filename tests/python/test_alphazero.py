"""Expert iteration (rl/alphazero.py) and the tactician's search by simulation (rl/tactician_search.py)."""

import inspect
import os
import random
import struct

import numpy as np
import pytest
import torch

from pybattle.backend import Phase, SimBackend
from pybattle.emu.decode import SYMBOLS as S, decode_party
from rl import alphazero as AZ
from rl import encode, search
from rl import tactician_search as TS
from rl.model import FactoryNet

TINY = ["--sims", "16", "--dets", "2", "--t-budget", "16", "--t-considered", "4", "--holdout", "0.3"]


@pytest.fixture(autouse=True)
def _keep_encode_version():
    v = encode.VERSION
    yield
    encode.set_version(v)


def _net(seed=0, version=4):
    torch.manual_seed(seed)
    encode.set_version(version)
    return FactoryNet(share="embeddings").eval()


def _valid_rental(v):
    import itertools
    return next(t for t in itertools.combinations(range(6), 3) if len({v.candidates[i].species for i in t}) == 3)


def _battle_to_swap(b, rng):
    """Play a battle with the first usable move (a random tactician choice before) until a SWAP decision."""
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


def _tactician_decision(kind, seed):
    """A SimBackend at a RENTAL (kind "rental") or SWAP decision, and its encoded observation."""
    rng = random.Random(seed)
    for s in range(seed, seed + 50):
        b = SimBackend(max_turns=10 ** 9)
        b.reset(seed=s, win_streak=7, rents_count=2)
        if kind == "swap" and not _battle_to_swap(b, rng):
            continue
        ctx = search.run_ctx(b)
        v = b.view()
        return b, (encode.rental(v, ctx) if kind == "rental" else encode.swap(v, ctx))
    raise RuntimeError("no swap decision reached")


# ---- network pieces --------------------------------------------------------------------------------------------------

def test_rental_joint_matches_rental():
    net = _net()
    b, obs = _tactician_decision("rental", 3)
    x = encode.collate([obs] * 6, "cpu")
    with torch.no_grad():
        ll, pl, v = net.rental_joint(x)
        for lead in range(6):
            l2, p2, _, v2 = net.rental(x, lead=torch.full((6,), lead, dtype=torch.long))
            assert torch.allclose(ll, l2, atol=1e-5)
            assert torch.allclose(pl[:, lead], p2, atol=1e-5)
            assert torch.allclose(v, v2, atol=1e-5)


def test_rental_loss_is_the_joint_cross_entropy():
    """-sum pi(l, p) [log q(l) + log q(p | l)]: lead marginal + pair conditional, as FactoryNet.rental factorizes."""
    net = _net(1)
    b, obs = _tactician_decision("rental", 5)
    x = encode.collate([obs], "cpu")
    opts = TS.options_of("rental", obs)
    rng = np.random.default_rng(0)
    w = rng.random(len(opts))
    pi = TS.policy_to_action_space("rental", opts, w / w.sum())
    with torch.no_grad():
        ce, kl, ent, _ = AZ.policy_terms("rental", net, x, torch.from_numpy(pi[None]))
        ref = 0.0
        for (lead, p), wi in zip(opts, w / w.sum()):
            ll, pl, _, _ = net.rental(x, lead=torch.tensor([lead]))
            ref -= wi * (torch.log_softmax(ll, -1)[0, lead] + torch.log_softmax(pl, -1)[0, p]).item()
    assert abs(ce.item() - ref) < 1e-4
    assert kl.item() >= -1e-5 and ent.item() > 0


def test_value_target_and_start_weights():
    z = np.array([1.0, 0.0, 3.0], np.float32)
    q = np.array([0.5, np.nan, 2.0], np.float32)
    assert np.allclose(AZ.value_target(z, q, 0.5), [0.75, 0.0, 2.5])
    w = AZ.start_weights_from_eval(None, 0.3)
    assert np.isclose(w[0], 0.3) and np.allclose(w[1:], 0.14) and np.isclose(w.sum(), 1)
    w = AZ.start_weights_from_eval({1: 0.9, 2: 0.8, 3: 0.4, 4: 0.8, 5: 0.2, 6: 0.0}, 0.3)
    hard = np.array([0.2, 0.6, 0.2, 0.8, 1.0])
    assert np.allclose(w[1:], 0.7 * hard / hard.sum()) and np.isclose(w.sum(), 1)
    env = AZ.CurriculumEnv(1, weights=[0, 0, 0, 0, 0, 1.0])
    for _ in range(3):
        env.reset()
        assert env.start_streak == 35 and env.backend.run_info().win_streak == 35


# ---- self-play targets ------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def selfplay_samples():
    torch.set_num_threads(2)
    v = encode.VERSION
    got = {k: [] for k in AZ.KINDS}
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(search.TRAINING_ENV, "1")         # the default opponent prior (factory_sets) is training-only
        cfg = AZ.worker_cfg(AZ.parse(TINY))
        assert cfg["opponent_prior"] == "factory_sets"
        sp = AZ.SelfPlayer(cfg, seed=3)
        for _ in range(40):
            out, st = sp.play(60)
            for k, d in out.items():
                if d is not None:
                    got[k].append(d)
            if all(got.values()):
                break
    encode.set_version(v)
    return {k: AZ.merge([{k: d} for d in v] + [{}])[k] if v else None for k, v in got.items()}


def test_targets_have_the_right_shapes_and_sum_to_one(selfplay_samples):
    for kind in AZ.KINDS:
        d = selfplay_samples[kind]
        assert d is not None, f"no {kind} sample"
        n = len(d["z"])
        assert d["pi"].shape == (n,) + AZ.PI_SHAPE[kind]
        assert d["q"].shape == d["z"].shape == d["held"].shape == (n,)
        assert np.allclose(d["pi"].reshape(n, -1).sum(1), 1.0, atol=1e-5)
        assert (d["pi"] >= 0).all()
        if kind == "battle":
            assert (d["pi"][~d["obs"]["mask"]] == 0).all()
            assert ((d["z"] >= 0) & (d["z"] <= 1)).all()
            q = d["q"][~np.isnan(d["q"])]
            assert len(q) and ((q >= 0) & (q <= 1)).all()
        elif kind == "rental":
            legal = d["obs"]["pair_mask"] & d["obs"]["lead_mask"][:, :, None]
            assert (d["pi"][~legal] == 0).all()
            assert (d["z"] >= 0).all()
        else:
            assert (d["pi"][~d["obs"]["mask"]] == 0).all()
            assert (d["z"] >= 0).all()
        for k, v in d["obs"].items():
            assert len(v) == n, k


def test_training_step_on_selfplay_samples(selfplay_samples):
    args = AZ.parse(TINY + ["--batch-b", "32", "--min-batch-t", "4"])
    net = _net(2)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    replay = AZ.Replay(2)
    replay.add(selfplay_samples)
    norms = {a: AZ.ValueNorm(0.3) for a in ("battler", "tactician")}
    counts = {k: int((~d["held"]).sum()) for k, d in selfplay_samples.items()}
    before = [p.detach().clone() for p in net.parameters()]
    out = AZ.train_iteration(net, opt, replay, counts, norms, args, torch.device("cpu"), np.random.default_rng(0))
    assert out["train/steps"] >= 1 and np.isfinite(out["train/loss"])
    for kind in AZ.KINDS:
        assert f"{kind}/policy_loss" in out and f"{kind}/explained_variance_train" in out
    assert any(not torch.equal(a, b) for a, b in zip(before, net.parameters()))
    assert norms["battler"].seen and norms["tactician"].seen
    AZ.heldout_stats(net, replay, norms, args, torch.device("cpu"))


# ---- the tactician's search never reads the real next opponent -------------------------------------------------------

def _swap_real_opponent(b):
    """A clone of `b` whose real next opponent (gFrontierTempParty: the Factory set ids generated before the
    decision) is replaced by other sets."""
    c = b.clone()
    addr = S.addr("gFrontierTempParty")
    ids = struct.unpack("<3H", c.game.read(addr, 6))
    c.game.write(addr, struct.pack("<3H", *[(i + 37 * (k + 1)) % 800 + 1 for k, i in enumerate(ids)]))
    return c


def _enemy_after_start(b, kind, option):
    c = b.clone()
    act = TS.decode_action(kind, option)
    ok = c.game.factory_rent(*act) if kind == "rental" else (
        c.game.factory_swap(-1) if act is None else c.game.factory_swap(*act))
    assert ok
    return [m.species for m in decode_party(c.game.read(S.addr("gEnemyParty"), 300))]


@pytest.mark.parametrize("kind", ["rental", "swap"])
def test_tactician_search_never_reads_the_real_next_opponent(kind):
    net = _net(4)
    ev = AZ.local_battler_evaluator(net, None)
    b, obs = _tactician_decision(kind, 11)
    b2 = _swap_real_opponent(b)
    opts = TS.options_of(kind, obs)
    # the swap took effect: without the search's replacement, the battles would be against other teams
    assert _enemy_after_start(b, kind, opts[0]) != _enemy_after_start(b2, kind, opts[0])
    # ... but a simulated battle starts from the same (sampled) team, and the whole search is identical
    s1 = TS.start_simulated_battle(b, kind, opts[0], random.Random(5))
    s2 = TS.start_simulated_battle(b2, kind, opts[0], random.Random(5))
    assert s1.game.read(S.addr("gEnemyParty"), 600) == s2.game.read(S.addr("gEnemyParty"), 600)
    assert s1.game.rng == s2.game.rng

    def values(kinds, obs_list):
        return np.full(len(obs_list), 1.5)

    logp = np.log(np.full(len(opts), 1.0 / len(opts)))
    res = []
    for backend in (b, b2):
        ts = TS.TacticianSearch(ev, values, budget=48, max_considered=6, seed=7)
        res.append(ts.search(backend, kind, obs, logp, noise=True))
    assert res[0]["visits"].sum() > 0
    assert np.array_equal(res[0]["visits"], res[1]["visits"])
    assert np.allclose(res[0]["q"], res[1]["q"], equal_nan=True)
    assert res[0]["action"] == res[1]["action"]


def test_simulated_opponent_is_sampled_from_player_knowledge():
    b, obs = _tactician_decision("rental", 2)
    opts = TS.options_of("rental", obs)
    rng_ref = random.Random(9)
    c = TS.start_simulated_battle(b, "rental", opts[0], random.Random(9))
    enemy = decode_party(c.game.read(S.addr("gEnemyParty"), 300))
    own_level = decode_party(c.game.read(S.addr("gPlayerParty"), 300))[0].level
    specs = TS.unseen_team_specs(own_level, rng_ref, True)
    assert [m.species for m in enemy] == [s[1] for s in specs]
    assert [m.held_item for m in enemy] == [s[3] for s in specs]
    assert all(m.level == own_level and m.hp == m.max_hp for m in enemy)
    assert c.game.determinized


# ---- perfect information is impossible in training -------------------------------------------------------------------

def test_perfect_mode_impossible_in_training(monkeypatch):
    assert "rl.alphazero" in search._TRAINING_MAINS
    src = inspect.getsource(AZ.main)
    assert "mark_training()" in src.split("parse(argv)")[0]
    assert "mode" not in inspect.signature(AZ.AZBattler).parameters
    monkeypatch.setenv(search.TRAINING_ENV, "1")

    class P:
        net, device, value_norm = None, "cpu", {}
    with pytest.raises(PermissionError):
        search.SearchBattler(P(), mode="perfect", allow_perfect=True)
    net = _net()
    battler = AZ.AZBattler(AZ.local_battler_evaluator(net, None), n_sims=8, n_determinizations=2)
    assert battler.mode == "legal"
    b = SimBackend(max_turns=10 ** 9)
    b.reset(seed=4)
    b.act(_valid_rental(b.view()))
    assert b.phase in (Phase.BATTLE, Phase.FORCED_SWITCH)
    r = battler.search_root(b, 0)                  # legal roots only: fine
    assert np.isclose(r["visits"].sum(), r["visits"][r["legal"]].sum())
    if search.NativeSearcher is not None:           # the C++ loop refuses a true-state root in training
        with pytest.raises(PermissionError):
            battler.searcher.search([(b.game.clone(), b._observer.fast_copy(), 0)], search.run_ctx(b),
                                    [1 / 7] * 7, [True] * 7, 0.5, battler.evaluate_batch)


# ---- one tiny iteration end to end --------------------------------------------------------------------------------------

def test_tiny_iteration_end_to_end(tmp_path, monkeypatch):
    monkeypatch.delenv(search.TRAINING_ENV, raising=False)
    run = str(tmp_path / "az")
    AZ.main(["--run-dir", run, "--iterations", "2", "--decisions-per-iter", "40", "--workers", "2",
             "--inference", "cpu", "--device", "cpu", "--eval-runs", "2", "--eval-rounds", "1",
             "--eval-workers", "1", "--eval-envs-per-worker", "2", "--batch-b", "16", "--min-batch-t", "4"] + TINY)
    assert os.environ.get(search.TRAINING_ENV) == "1"          # main marked the process as training
    files = sorted(os.listdir(run))
    assert "latest.pt" in files and "config.json" in files and len([f for f in files if f.startswith("ckpt_")]) == 2
    ck = torch.load(os.path.join(run, "latest.pt"), map_location="cpu")
    assert ck["args"]["algo"] == "alphazero" and ck["iteration"] == 1 and ck["battler_steps"] >= 80
    assert "opt" in ck and ck["value_norm"]["battler"] is not None
    from rl.policy import Policy
    pol = Policy(os.path.join(run, "latest.pt"), torch.device("cpu"))
    assert pol.algo == "alphazero"
    b, obs = _tactician_decision("rental", 1)
    a = pol.act("rental", encode.collate([obs], "cpu"))
    assert a.shape == (1, 2)


# ---- the GPU server's reload ---------------------------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_inference_server_reload(tmp_path):
    from rl import inference as I
    paths = []
    for seed in (0, 1):
        net = _net(seed)
        p = str(tmp_path / f"snap{seed}.pt")
        torch.save({"net": net.state_dict(), "args": {"algo": "alphazero", "encode_version": 4, "d_emb": 64, "d": 128,
                                                      "layers": 2, "heads": 4, "share": "embeddings"},
                    "value_norm": {"battler": {"mean": 0.3 + 0.2 * seed, "var": 0.04, "seen": True}}}, p)
        paths.append(p)
    rng = random.Random(0)
    obs = []
    for s in range(3):
        b = SimBackend(max_turns=10 ** 9)
        b.reset(seed=s)
        while len(obs) < 12 * (s + 1) and b.phase != Phase.RUN_OVER:
            v = b.view()
            if b.phase == Phase.RENTAL:
                b.act(_valid_rental(v))
                continue
            if b.phase == Phase.SWAP:
                b.act(None)
                continue
            obs.append(encode.battle(v, search.run_ctx(b)))
            if b.phase == Phase.FORCED_SWITCH:
                b.act(("switch", rng.choice(v.switch_targets)))
            else:
                moves = [i for i, ok in enumerate(v.usable_moves) if ok]
                b.act(("move", rng.choice(moves)) if moves else ("move", 0))
    batch = I.stack_obs(obs)
    with I.InferenceServer(paths[0], n_clients=1, max_batch=64, max_total=128) as server:
        c = server.client(0)
        for p in (paths[0], paths[1], paths[0]):
            if p != server.ckpt:
                server.reload(p)
            pri, val = c.evaluate(batch)
            rp, rv = I.make_evaluator(p, "cpu")(batch)
            assert np.abs(pri - rp).max() < 1e-4 and np.abs(val - rv).max() < 1e-4

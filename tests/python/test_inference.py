"""GPU inference server (rl/inference.py): same outputs as the CPU network, concurrency, robustness."""

import glob
import multiprocessing as mp
import os
import random
import signal
import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from rl import inference as I  # noqa: E402

CKPTS = sorted(glob.glob("runs/ppo_joint_v3/latest.pt"))
if not CKPTS:
    pytest.skip("no v3 checkpoint", allow_module_level=True)
CKPT = CKPTS[0]
TOL = 1e-4


def _real_observations(n=400, seeds=(3, 11, 29), streaks=(0, 14, 28)):
    """Battle observations (encode.battle) from Factory runs: tactician = the network, battler = random legal."""
    from rl.envs import NO_ACTION, FactoryEnv, decode_action
    from rl.policy import Policy
    from rl import encode
    policy = Policy(CKPT, torch.device("cpu"))
    rng = random.Random(0)
    out = []
    for i in range(10 ** 6):
        env = FactoryEnv(seeds[i % len(seeds)] + 1000 * i, win_streak=streaks[i % len(streaks)])
        ev = env._advance(NO_ACTION)
        for _ in range(400):
            kind = ev["kind"]
            if kind == "battle":
                out.append(ev["obs"])
                if len(out) >= n:
                    return out
                a = rng.choice(np.flatnonzero(ev["obs"]["mask"]).tolist())
            else:
                a = policy.act(kind, encode.collate([ev["obs"]], policy.device), greedy=True)[0]
            ev = env.step(decode_action(kind, a))
            if ev["truncate"] or ev["stats"].get("run") is not None:
                break
    return out


@pytest.fixture(scope="module")
def data():
    torch.set_num_threads(2)
    obs = _real_observations()
    batch = I.stack_obs(obs)
    cpu = I.make_evaluator(CKPT, device="cpu")
    pri, val = cpu(batch)
    return obs, batch, pri, val


@pytest.fixture(scope="module")
def server():
    with I.InferenceServer(CKPT, n_clients=20, max_batch=256, max_total=2048, deadline_ms=0.5, timeout=60) as s:
        yield s


def _take(batch, idx):
    return {k: v[idx] for k, v in batch.items()}


# ---- equivalence ------------------------------------------------------------------------------------------------

def test_same_outputs_as_the_cpu_search_path(data, server):
    """The server's (priors, values) equal SearchBattler.evaluate on the CPU network (fp32, 1e-4)."""
    from rl.policy import Policy
    from rl.search import SearchBattler
    obs, batch, pri, val = data
    sb = SearchBattler(Policy(CKPT, torch.device("cpu")), n_determinizations=1)
    p_ref, v_ref = sb.evaluate(obs)
    assert np.abs(p_ref - pri).max() < 1e-6 and np.abs(v_ref - val).max() < 1e-6    # LocalEvaluator == search
    c = server.client(0)
    p, v = c.evaluate(batch)                           # 400 > max_batch 256: split by the client
    assert p.shape == (len(obs), 7) and v.shape == (len(obs),)
    assert p.dtype == np.float32 and v.dtype == np.float32
    assert np.abs(p - p_ref).max() < TOL and np.abs(v - v_ref).max() < TOL
    assert (p[~batch["mask"]] == 0).all() and np.allclose(p.sum(1), 1, atol=1e-5)
    assert ((v >= 0) & (v <= 1)).all()
    for n in (1, 7, 32, 255, 256):
        idx = np.arange(n) * 3 % len(obs)
        p, v = c.evaluate(_take(batch, idx))
        assert np.abs(p - p_ref[idx]).max() < TOL and np.abs(v - v_ref[idx]).max() < TOL
    # the adapter used by the Python search
    p, v = I.obs_evaluator(c)(obs[:50])
    assert np.abs(p - p_ref[:50]).max() < TOL
    assert c.evaluate(_take(batch, np.arange(0)))[0].shape == (0, 7)


# ---- concurrency ------------------------------------------------------------------------------------------------

def _client_loop(server, slot, batch, pri, val, n_req, seed, q):
    try:
        c = server.client(slot)
        rng = np.random.default_rng(seed)
        worst = 0.0
        for _ in range(n_req):
            n = int(rng.integers(1, 257))
            idx = rng.integers(0, len(pri), n)
            p, v = c.evaluate(_take(batch, idx))
            worst = max(worst, float(np.abs(p - pri[idx]).max()), float(np.abs(v - val[idx]).max()))
        q.put((slot, "ok", worst))
    except BaseException as e:                          # pragma: no cover - reported to the parent
        q.put((slot, "error", repr(e)))


def test_many_concurrent_workers(data, server):
    _, batch, pri, val = data
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    before = server.stats()
    procs = [ctx.Process(target=_client_loop, args=(server, i, batch, pri, val, 40, i, q)) for i in range(18)]
    for p in procs:
        p.start()
    res = [q.get(timeout=120) for _ in procs]
    for p in procs:
        p.join(10)
    assert all(r[1] == "ok" for r in res), res
    assert max(r[2] for r in res) < TOL
    st = server.stats()
    assert st["requests"] - before["requests"] == 18 * 40
    assert st["batches"] - before["batches"] < 18 * 40          # requests were batched together


# ---- robustness ------------------------------------------------------------------------------------------------

def _die_mid_request(server, slot, batch):
    c = server.client(slot)
    s = c._slot                                           # submit a request, then die without reading the answer
    for k in I.KEYS:
        s.x[k][:10] = batch[k][:10]
    s.hdr[1] += 1
    s.hdr[2] = 10
    s.hdr[0] = I.PENDING
    c._req.release()
    os._exit(0)


def _loop_forever(server, slot, batch):
    c = server.client(slot)
    while True:
        c.evaluate(_take(batch, np.arange(64)))


def test_a_dead_worker_does_not_block_the_others(data, server):
    _, batch, pri, val = data
    ctx = mp.get_context("fork")
    dead = ctx.Process(target=_die_mid_request, args=(server, 18, batch))
    dead.start()
    dead.join(10)
    killed = ctx.Process(target=_loop_forever, args=(server, 19, batch))
    killed.start()
    time.sleep(0.5)
    os.kill(killed.pid, signal.SIGKILL)                   # killed while waiting / mid-request
    killed.join(10)
    q = ctx.Queue()
    procs = [ctx.Process(target=_client_loop, args=(server, i, batch, pri, val, 10, 100 + i, q)) for i in range(4)]
    t0 = time.time()
    for p in procs:
        p.start()
    res = [q.get(timeout=60) for _ in procs]
    for p in procs:
        p.join(10)
    assert all(r[1] == "ok" and r[2] < TOL for r in res), res
    assert time.time() - t0 < 30
    # a slot whose previous owner died is reusable (stale answers are skipped by sequence number)
    for slot in (18, 19):
        p, v = server.client(slot).evaluate(_take(batch, np.arange(5)))
        assert np.abs(p - pri[:5]).max() < TOL


def _shm_exists(name):
    return os.path.exists("/dev/shm/" + name.lstrip("/"))


def test_shutdown_unlinks_and_clients_raise(data):
    _, batch, pri, _ = data
    s = I.InferenceServer(CKPT, n_clients=2, max_batch=64, max_total=128, poll=0.1, timeout=10)
    names = [m.name for m in s._shms] + [s._ctl_shm.name]
    c = s.client(0)
    assert np.abs(c.evaluate(_take(batch, np.arange(8)))[0] - pri[:8]).max() < TOL
    assert all(_shm_exists(n) for n in names)
    s.close()
    assert not s.alive()
    assert not any(_shm_exists(n) for n in names)
    t0 = time.time()
    with pytest.raises(I.ServerDied):
        c.evaluate(_take(batch, np.arange(8)))
    assert time.time() - t0 < 5
    s.close()                                             # idempotent


def test_server_death_raises_in_clients(data):
    _, batch, _, _ = data
    with I.InferenceServer(CKPT, n_clients=1, max_batch=64, max_total=128, poll=0.1, timeout=30) as s:
        c = s.client(0)
        c.evaluate(_take(batch, np.arange(4)))
        s._proc.kill()                                    # our own server process
        s._proc.join(5)
        t0 = time.time()
        with pytest.raises(I.ServerDied):
            c.evaluate(_take(batch, np.arange(4)))
        assert time.time() - t0 < 5
        with pytest.raises(I.ServerDied):                 # the client stays unusable
            c.evaluate(_take(batch, np.arange(4)))


def test_timeout_when_the_server_stalls(data):
    _, batch, pri, _ = data
    with I.InferenceServer(CKPT, n_clients=2, max_batch=64, max_total=128, poll=0.1, timeout=1.0) as s:
        c0, c1 = s.client(0), s.client(1)
        c0.evaluate(_take(batch, np.arange(4)))
        os.kill(s.pid, signal.SIGSTOP)
        try:
            t0 = time.time()
            with pytest.raises(TimeoutError):
                c0.evaluate(_take(batch, np.arange(4)))
            assert 0.9 < time.time() - t0 < 5
        finally:
            os.kill(s.pid, signal.SIGCONT)
        assert np.abs(c1.evaluate(_take(batch, np.arange(4)))[0] - pri[:4]).max() < TOL   # the server recovers


def test_eval_rounds_with_the_gpu_server_plays_like_the_cpu_net():
    """--battler net: same runs through the server as with the workers' CPU networks (same seeds)."""
    from rl.eval_rounds import eval_round_inprocess
    kw = dict(workers=1, envs_per_worker=2, seed=5, procs=2)
    cpu = eval_round_inprocess(CKPT, 1, 2, "net", **kw)
    with I.InferenceServer(CKPT, n_clients=2) as s:
        gpu = eval_round_inprocess(CKPT, 1, 2, "net", server=s, **kw)
        assert s.stats()["requests"] == gpu["decisions"]
    assert (cpu["complete"], cpu["battle"], cpu["n"], cpu["decisions"]) == \
           (gpu["complete"], gpu["battle"], gpu["n"], gpu["decisions"])

"""PPO pieces (rl/ppo.py) and the environment's transitions and rewards (rl/envs.py)."""

import collections
import random

import numpy as np
import pytest
import torch

from rl import encode
from rl.envs import NO_ACTION, FactoryEnv, decode_action
from rl.model import FactoryNet
from rl.ppo import AgentBuffer, Transition, ValueNorm, act, ppo_update


def _gae_reference(rewards, values, next_values, dones, truncs, gamma, lam):
    """Textbook GAE over one environment's sequence; the chain restarts after terminal / truncated steps."""
    adv = np.zeros(len(rewards))
    last = 0.0
    for t in reversed(range(len(rewards))):
        boot = 0.0 if dones[t] else gamma * next_values[t]
        delta = rewards[t] + boot - values[t]
        end = dones[t] or truncs[t] or t == len(rewards) - 1
        last = delta if end else delta + gamma * lam * last
        adv[t] = last
    return adv


@pytest.mark.parametrize("gamma", [0.99, 1.0])
def test_gae_matches_the_reference(gamma):
    rng = np.random.default_rng(0)
    buf = AgentBuffer(gamma, 0.95)
    buf.ensure(3)
    ref = []
    for e in range(3):
        n = 40
        r, v = rng.normal(size=n), rng.normal(size=n)
        d = rng.random(n) < 0.1
        tr = ~d & (rng.random(n) < 0.05)
        nv = np.append(v[1:], rng.normal())
        nv[tr] = rng.normal(size=tr.sum())            # truncation: bootstrap with the value at the cut
        for t in range(n):
            buf.per_env[e].append(Transition("battle", {}, 0, 0.0, float(v[t]), float(r[t]), bool(d[t]), bool(tr[t]),
                                             float(nv[t]), True))
        buf.per_env[e].append(Transition("battle", {}, 0, 0.0, 0.0))     # pending: not taken
        ref.append((_gae_reference(r, v, nv, d, tr, gamma, 0.95), v))
    out, adv, ret = buf.take()
    assert len(out) == 120 and all(len(s) == 1 for s in buf.per_env)
    exp_adv = np.concatenate([a for a, _ in ref])
    assert np.allclose(adv, exp_adv, atol=1e-5)
    assert np.allclose(ret, exp_adv + np.concatenate([v for _, v in ref]), atol=1e-5)


def test_value_norm_state_roundtrip():
    a = ValueNorm()
    a.update(np.array([1.0, 3.0, 5.0]))
    a.update(np.array([10.0, 12.0]))
    b = ValueNorm()
    b.load(a.state())
    assert (b.mean, b.var, b.seen) == (a.mean, a.var, a.seen)
    assert b.denorm(0.5) == a.denorm(0.5)


def _rollout(net, n_steps, seed=0, **kw):
    """Run one FactoryEnv with the network's sampled actions, closing transitions like rl/train.py."""
    env = FactoryEnv(seed, **kw)
    ev = env._advance(NO_ACTION)
    bufs = {"battler": AgentBuffer(1.0, 0.95), "tactician": AgentBuffer(1.0, 0.95)}
    for b in bufs.values():
        b.ensure(1)
    last_t = 0.0
    log = collections.defaultdict(list)
    for _ in range(n_steps):
        kind = ev["kind"]
        a, lp, v = act(net, kind, encode.collate([ev["obs"]], "cpu"))
        a, lp, v = a[0], float(lp[0]), float(v[0])
        for agent, key in (("battler", "close_b"), ("tactician", "close_t")):
            if ev[key] is not None:
                r, done, trunc = ev[key]
                t = bufs[agent].per_env[0][-1]
                t.reward, t.done, t.trunc, t.complete = r, done, trunc, True
                if agent == "battler":
                    t.next_value = v if kind == "battle" else 0.0
                else:
                    t.next_value = v if kind != "battle" else max(last_t - r, 0.0)
                log[agent].append((r, done, trunc))
        for k in ("battle", "run"):
            if k in ev["stats"]:
                log[k].append(ev["stats"][k])
        if ev["truncate"]:
            ev = env.step("reset")
            continue
        agent = "battler" if kind == "battle" else "tactician"
        bufs[agent].per_env[0].append(Transition(kind, ev["obs"], a, lp, v))
        if agent == "tactician":
            last_t = v
        ev = env.step(decode_action(kind, a))
    return bufs, log


def test_tactician_rewards_are_the_wins_and_updates_run():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    encode.set_version(3)
    net = FactoryNet(16, 32, 1, 2, share="embeddings")
    bufs, log = _rollout(net, 1000, seed=3, max_decisions=40, gamma=1.0, beta=0.5)
    # the tactician's rewards add up to the battles won, per run
    won = sum(b["won"] is True for b in log["battle"])
    assert sum(r for r, _, _ in log["tactician"]) == won
    assert sum(done for _, done, _ in log["tactician"]) == len(log["run"])
    # every battle's battler episode ends once: terminal (a result) or truncated
    ends = [(done, trunc) for _, done, trunc in log["battler"] if done or trunc]
    assert len(ends) == len(log["battle"])
    # an update on each agent's batch runs, with the staleness diagnostics
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    for agent, refresh in (("battler", False), ("tactician", True)):
        tr, adv, ret = bufs[agent].take()
        assert len(tr) > 10
        s = ppo_update(net, opt, tr, adv, ret, "cpu", 2, 2, norm=ValueNorm(), target_kl=0.02, refresh_logp=refresh)
        assert np.isfinite(s["policy_loss"]) and s["stale_kl"] >= -1e-6
        if refresh:
            assert s["stale_kl"] < 1e-5              # no parameter changed since the tactician's actions


def test_shaping_telescopes_with_gamma_one():
    """With gamma 1 and a constant beta, a battle's shaped return is win - beta * Phi(first decision)."""
    torch.set_num_threads(1)
    torch.manual_seed(1)
    encode.set_version(3)
    net = FactoryNet(16, 32, 1, 2, share="embeddings")
    env = FactoryEnv(7, gamma=1.0, beta=0.5, max_decisions=200)
    ev = env._advance(NO_ACTION)
    phi0 = None
    n = 0
    for _ in range(2000):
        kind = ev["kind"]
        if kind == "battle" and env.decisions == 1:
            phi0 = env.b_phi
        if "battle" in ev["stats"] and phi0 is not None:
            st = ev["stats"]["battle"]
            if st["won"] is not None:
                assert st["shaped_return"] == pytest.approx(float(st["won"]) - 0.5 * phi0, abs=1e-6)
                n += 1
            phi0 = None
        if ev["truncate"]:
            ev = env.step("reset")
            continue
        a, _, _ = act(net, kind, encode.collate([ev["obs"]], "cpu"))
        ev = env.step(decode_action(kind, a[0]))
    assert n >= 5

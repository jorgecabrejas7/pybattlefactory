"""PPO for the two agents (docs/RL_DECISIONS.md §7).

Transitions are kept per environment and per agent in decision order. A transition is *complete* once the
agent's next decision in the same episode is seen (its value is the bootstrap target), or its episode ended
(terminal) or was truncated (bootstrap with the value at the truncation point).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import torch
from torch.distributions import Categorical

from .encode import collate


@dataclass
class Transition:
    kind: str                   # "battle" | "rental" | "swap"
    obs: Dict[str, np.ndarray]
    action: Any                 # int, or (lead, pair) for rentals
    logp: float
    value: float
    reward: float = 0.0
    done: bool = False          # terminal: no bootstrap
    trunc: bool = False         # truncated: bootstrap with next_value, but the episode ends here
    next_value: float = 0.0
    complete: bool = False


@dataclass
class AgentBuffer:
    gamma: float
    lam: float
    per_env: List[List[Transition]] = field(default_factory=list)

    def ensure(self, n_envs):
        while len(self.per_env) < n_envs:
            self.per_env.append([])

    def n_complete(self):
        return sum(sum(t.complete for t in seq) for seq in self.per_env)

    def take(self):
        """Pop every complete transition, with GAE advantages and returns."""
        out, adv, ret = [], [], []
        for e, seq in enumerate(self.per_env):
            k = 0
            while k < len(seq) and seq[k].complete:
                k += 1
            done_part, self.per_env[e] = seq[:k], seq[k:]
            a = np.zeros(k, np.float32)
            last = 0.0
            for t in reversed(range(k)):
                tr = done_part[t]
                boot = 0.0 if tr.done else self.gamma * tr.next_value
                delta = tr.reward + boot - tr.value
                end = tr.done or tr.trunc or t == k - 1
                last = delta if end else delta + self.gamma * self.lam * last
                a[t] = last
            out += done_part
            adv += list(a)
            ret += [a[t] + done_part[t].value for t in range(k)]
        return out, np.array(adv, np.float32), np.array(ret, np.float32)


# ---- policy evaluation for each decision kind ----------------------------------------------------------------

def evaluate(net, kind, x, actions):
    """log-prob, entropy and value of `actions` under the current network."""
    if kind == "battle":
        logits, v = net.battler(x)
        d = Categorical(logits=logits)
        return d.log_prob(actions), d.entropy(), v
    if kind == "swap":
        logits, v = net.swap(x)
        d = Categorical(logits=logits)
        return d.log_prob(actions), d.entropy(), v
    lead, pair = actions[:, 0], actions[:, 1]
    ll, pl, _, v = net.rental(x, lead=lead)
    d1, d2 = Categorical(logits=ll), Categorical(logits=pl)
    return d1.log_prob(lead) + d2.log_prob(pair), d1.entropy() + d2.entropy(), v


@torch.no_grad()
def act(net, kind, x, greedy=False):
    """Sample (or argmax) actions. Returns actions (np), log-probs (np), values (np)."""
    if kind == "rental":
        ll, _, _, _ = net.rental(x, lead=torch.zeros(len(x["mon_ids"]), dtype=torch.long, device=x["mon_ids"].device))
        lead = ll.argmax(-1) if greedy else Categorical(logits=ll).sample()
        ll, pl, _, v = net.rental(x, lead=lead)
        pair = pl.argmax(-1) if greedy else Categorical(logits=pl).sample()
        logp = Categorical(logits=ll).log_prob(lead) + Categorical(logits=pl).log_prob(pair)
        return torch.stack([lead, pair], 1).cpu().numpy(), logp.cpu().numpy(), v.cpu().numpy()
    logits, v = net.battler(x) if kind == "battle" else net.swap(x)
    a = logits.argmax(-1) if greedy else Categorical(logits=logits).sample()
    return a.cpu().numpy(), Categorical(logits=logits).log_prob(a).cpu().numpy(), v.cpu().numpy()


@torch.no_grad()
def values(net, kind, x):
    if kind == "battle":
        return net.battler(x)[1].cpu().numpy()
    if kind == "swap":
        return net.swap(x)[1].cpu().numpy()
    return net.rental(x, lead=torch.zeros(len(x["mon_ids"]), dtype=torch.long, device=x["mon_ids"].device))[3].cpu().numpy()


class ValueNorm:
    """Running mean / std of an agent's returns. The critic predicts normalized values, so both agents' value
    losses have a comparable scale (the tactician's returns are streaks, ~300x the battler's loss otherwise)."""

    def __init__(self, rate=0.05):
        self.mean, self.var, self.rate, self.seen = 0.0, 1.0, rate, False

    @property
    def std(self):
        return max(self.var, 1e-4) ** 0.5

    def update(self, x):
        m, v = float(np.mean(x)), float(np.var(x))
        if not self.seen:
            self.mean, self.var, self.seen = m, max(v, 1e-4), True
        else:
            self.mean += self.rate * (m - self.mean)
            self.var += self.rate * (v - self.var)

    def denorm(self, v):
        return v * self.std + self.mean

    def state(self):
        return {"mean": self.mean, "var": self.var, "seen": self.seen}


def ppo_update(net, opt, transitions, adv, ret, device, epochs, minibatches, clip=0.2, vf_coef=0.5, ent_coef=0.01,
               max_grad_norm=0.5, norm=None):
    """One PPO update over a batch that may mix decision kinds (the tactician's rentals and swaps).
    With `norm`, stored values are in return units and the critic is trained on normalized targets."""
    n = len(transitions)
    kinds = np.array([t.kind for t in transitions])
    old_logp = torch.tensor([t.logp for t in transitions], device=device)
    old_v = np.array([t.value for t in transitions], np.float32)
    adv_t = torch.tensor(adv, device=device)
    if norm is not None:
        norm.update(ret)
        ret_t = torch.tensor((ret - norm.mean) / norm.std, dtype=torch.float32, device=device)
    else:
        ret_t = torch.tensor(ret, device=device)
    full, pos, acts_full = {}, np.zeros(n, np.int64), {}
    for kind in np.unique(kinds):
        idx = np.nonzero(kinds == kind)[0]
        pos[idx] = np.arange(len(idx))
        full[kind] = collate([transitions[i].obs for i in idx], device)
        acts_full[kind] = torch.tensor(np.array([transitions[i].action for i in idx]), device=device)
    stats = {k: [] for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac", "grad_norm")}
    for _ in range(epochs):
        perm = np.random.permutation(n)
        for mb in np.array_split(perm, minibatches):
            logp_l, ent_l, v_l, idx_l = [], [], [], []
            for kind in np.unique(kinds[mb]):
                idx = mb[kinds[mb] == kind]
                p = torch.tensor(pos[idx], device=device)
                x = {k: t[p] for k, t in full[kind].items()}
                lp, ent, v = evaluate(net, kind, x, acts_full[kind][p])
                logp_l.append(lp); ent_l.append(ent); v_l.append(v); idx_l.append(torch.tensor(idx, device=device))
            idx = torch.cat(idx_l)
            logp, ent, v = torch.cat(logp_l), torch.cat(ent_l), torch.cat(v_l)
            a = adv_t[idx]
            a = (a - a.mean()) / (a.std() + 1e-8)
            ratio = torch.exp(logp - old_logp[idx])
            pl = -torch.min(ratio * a, torch.clamp(ratio, 1 - clip, 1 + clip) * a).mean()
            vl = 0.5 * ((v - ret_t[idx]) ** 2).mean()
            loss = pl + vf_coef * vl - ent_coef * ent.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            opt.step()
            with torch.no_grad():
                lr = logp - old_logp[idx]
                stats["approx_kl"].append(((torch.exp(lr) - 1) - lr).mean().item())
                stats["clip_frac"].append(((ratio - 1).abs() > clip).float().mean().item())
            stats["policy_loss"].append(pl.item()); stats["value_loss"].append(vl.item())
            stats["entropy"].append(ent.mean().item()); stats["grad_norm"].append(float(gn))
    out = {k: float(np.mean(v)) for k, v in stats.items()}
    var = np.var(ret)
    out["explained_variance"] = float(1 - np.var(ret - old_v) / var) if var > 0 else 0.0
    out["adv_mean"], out["return_mean"], out["value_mean"] = float(adv.mean()), float(ret.mean()), float(old_v.mean())
    return out

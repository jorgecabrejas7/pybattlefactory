"""Rainbow DQN for the two agents (docs/RL_DECISIONS.md §12).

The six parts: double Q-learning, prioritized replay, dueling heads, n-step returns, distributional (C51)
returns and noisy nets. The network reuses the PPO trunk (shared embeddings, Pokemon encoder, attention), so the
comparison with PPO isolates the algorithm.

Actions (one Q per complete action):
    battle  7   moves 0-3 of the active Pokemon, switch to own party slot 0-2
    rental  90  lead * 15 + pair (the 60 valid ones are unmasked: distinct slots, distinct species)
    swap    10  keep, or 1 + 3 * own_slot + enemy_slot
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import encode as E
from .model import Trunk, mlp

N_ACTIONS = {"battle": 7, "rental": 90, "swap": 10}


# ---- noisy linear layer (factorized Gaussian noise, Fortunato et al. 2018) ----------------------------------

class NoisyLinear(nn.Module):
    def __init__(self, i, o, sigma0=0.5):
        super().__init__()
        self.i, self.o = i, o
        self.w_mu = nn.Parameter(torch.empty(o, i).uniform_(-1 / math.sqrt(i), 1 / math.sqrt(i)))
        self.w_sigma = nn.Parameter(torch.full((o, i), sigma0 / math.sqrt(i)))
        self.b_mu = nn.Parameter(torch.empty(o).uniform_(-1 / math.sqrt(i), 1 / math.sqrt(i)))
        self.b_sigma = nn.Parameter(torch.full((o,), sigma0 / math.sqrt(i)))
        self.register_buffer("eps_in", torch.zeros(i))
        self.register_buffer("eps_out", torch.zeros(o))
        self.noise = True

    @staticmethod
    def _f(x):
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        self.eps_in.copy_(self._f(torch.randn(self.i, device=self.eps_in.device)))
        self.eps_out.copy_(self._f(torch.randn(self.o, device=self.eps_out.device)))

    def forward(self, x):
        if not self.noise:
            return F.linear(x, self.w_mu, self.b_mu)
        w = self.w_mu + self.w_sigma * torch.outer(self.eps_out, self.eps_in)
        b = self.b_mu + self.b_sigma * self.eps_out
        return F.linear(x, w, b)


def noisy_mlp(i, h, o):
    return nn.Sequential(NoisyLinear(i, h), nn.ReLU(), NoisyLinear(h, o))


# ---- network ----------------------------------------------------------------------------------------------

class RainbowNet(nn.Module):
    """Returns log-probabilities over atoms, [B, n_actions, atoms], for each decision kind."""

    def __init__(self, atoms=51, d_emb=64, d=128, layers=2, heads=4):
        super().__init__()
        self.atoms = atoms
        self.trunk = Trunk(d_emb, d, layers, heads)
        # action embeddings (deterministic) -> noisy dueling heads
        self.b_move = mlp(3 * d, d, d)
        self.b_switch = mlp(2 * d, d, d)
        self.t_rent = mlp(3 * d, d, d)
        self.t_keep = mlp(2 * d, d, d)
        self.t_swap = mlp(3 * d, d, d)
        self.b_adv, self.b_val = noisy_mlp(d, d, atoms), noisy_mlp(2 * d, d, atoms)
        self.t_adv, self.t_val = noisy_mlp(d, d, atoms), noisy_mlp(2 * d, d, atoms)
        self.register_buffer("pairs", torch.tensor(E.PAIRS, dtype=torch.long))

    def noisy_layers(self):
        return [m for m in self.modules() if isinstance(m, NoisyLinear)]

    def reset_noise(self):
        for m in self.noisy_layers():
            m.reset_noise()

    def set_noise(self, on: bool):
        for m in self.noisy_layers():
            m.noise = on

    @staticmethod
    def _dueling(adv, val, mask):
        """Q logits = V + A - mean over legal actions of A."""
        m = mask.unsqueeze(-1).float()
        mean = (adv * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp(min=1)
        return F.log_softmax(val.unsqueeze(1) + adv - mean, -1)

    def forward(self, kind, x):
        if kind == "battle":
            mons, ctx, moves = self.trunk(x, 0)
            active = x["active"]
            b = torch.arange(len(active), device=active.device)
            am, ah = moves[b, active], mons[b, active]
            mv = self.b_move(torch.cat([am, ah[:, None].expand(-1, 4, -1), ctx[:, None].expand(-1, 4, -1)], -1))
            sw = self.b_switch(torch.cat([mons[:, :3], ctx[:, None].expand(-1, 3, -1)], -1))
            h = torch.cat([mv, sw], 1)                                          # [B, 7, d]
            pool = torch.cat([mons.mean(1), ctx], -1)
            return self._dueling(self.b_adv(h), self.b_val(pool), x["mask"])
        if kind == "rental":
            mons, ctx, _ = self.trunk(x, 1)
            pair_h = mons[:, self.pairs[:, 0]] + mons[:, self.pairs[:, 1]]     # [B, 15, d]
            h = self.t_rent(torch.cat([mons[:, :, None].expand(-1, 6, 15, -1), pair_h[:, None].expand(-1, 6, -1, -1),
                                       ctx[:, None, None].expand(-1, 6, 15, -1)], -1)).flatten(1, 2)   # [B, 90, d]
            pool = torch.cat([mons.mean(1), ctx], -1)
            return self._dueling(self.t_adv(h), self.t_val(pool), x["pair_mask"].flatten(1))
        mons, ctx, _ = self.trunk(x, 2)
        own, foe = mons[:, :3], mons[:, 3:]
        grid = self.t_swap(torch.cat([own[:, :, None].expand(-1, 3, 3, -1), foe[:, None].expand(-1, 3, 3, -1),
                                      ctx[:, None, None].expand(-1, 3, 3, -1)], -1)).flatten(1, 2)
        pool = torch.cat([mons.mean(1), ctx], -1)
        h = torch.cat([self.t_keep(pool)[:, None], grid], 1)                    # [B, 10, d]
        return self._dueling(self.t_adv(h), self.t_val(pool), x["mask"])


def action_mask(kind, x):
    return x["pair_mask"].flatten(1) if kind == "rental" else x["mask"]


def expected_q(logp, support):
    return (logp.exp() * support).sum(-1)


def to_env_action(kind, a):
    """Rainbow action index -> the (network-style) action decode_action expects."""
    a = int(a)
    return (a // 15, a % 15) if kind == "rental" else a


@torch.no_grad()
def act(net, kind, x, support, greedy=False):
    """argmax of the expected Q over legal actions (exploration comes from the noisy layers)."""
    q = expected_q(net(kind, x), support).masked_fill(~action_mask(kind, x), -1e9)
    a = q.argmax(-1)
    return a.cpu().numpy(), q.gather(1, a[:, None])[:, 0].cpu().numpy()


# ---- prioritized replay -----------------------------------------------------------------------------------

class SumTree:
    def __init__(self, capacity):
        self.cap = 1
        while self.cap < capacity:
            self.cap *= 2
        self.tree = np.zeros(2 * self.cap, np.float64)

    def set(self, idx, p):
        i = np.asarray(idx) + self.cap
        self.tree[i] = p
        i = np.unique(i // 2)
        while i[0] >= 1:
            self.tree[i] = self.tree[2 * i] + self.tree[2 * i + 1]
            i = np.unique(i // 2)
            if i[0] == 0:
                break

    def total(self):
        return self.tree[1]

    def find(self, v):
        """Leaf indices for prefix sums v (vectorized)."""
        i = np.ones(len(v), np.int64)
        v = v.copy()
        while i[0] < self.cap:
            left = 2 * i
            go_right = v > self.tree[left]
            v = np.where(go_right, v - self.tree[left], v)
            i = np.where(go_right, left + 1, left)
        return i - self.cap


OBS_DTYPES = {"mon_ids": np.int16, "mon_num": np.float16, "move_num": np.float16, "ctx_ids": np.int16,
              "ctx_num": np.float16, "mask": bool, "active": np.int64, "lead_mask": bool, "pair_mask": bool}
KIND_ID = {"battle": 0, "rental": 1, "swap": 2}
KIND_NAME = {v: k for k, v in KIND_ID.items()}


class Replay:
    """Prioritized replay of n-step transitions (s, a, R, s', discount). Observations are stored in compact
    dtypes; the tactician's rental and swap observations share one buffer (masks of both kinds are kept)."""

    def __init__(self, capacity, example_obs, alpha=0.5):
        self.cap, self.alpha = capacity, alpha
        self.n = self.pos = 0
        self.tree = SumTree(capacity)
        self.max_p = 1.0
        self.obs, self.next_obs = {}, {}
        for k, dt in OBS_DTYPES.items():
            if k in example_obs:
                shape = example_obs[k].shape
                self.obs[k] = np.zeros((capacity, *shape), dt)
                self.next_obs[k] = np.zeros((capacity, *shape), dt)
        self.kind = np.zeros(capacity, np.int8)
        self.next_kind = np.zeros(capacity, np.int8)
        self.action = np.zeros(capacity, np.int64)
        self.ret = np.zeros(capacity, np.float32)
        self.disc = np.zeros(capacity, np.float32)       # gamma^n, or 0 when the episode terminated

    def nbytes(self):
        return sum(a.nbytes for a in self.obs.values()) * 2

    def add(self, kind, obs, action, ret, disc, next_kind, next_obs):
        i = self.pos
        for k in self.obs:
            if k in obs:
                self.obs[k][i] = obs[k]
            else:
                self.obs[k][i] = 0
            if next_obs is not None and k in next_obs:
                self.next_obs[k][i] = next_obs[k]
            else:
                self.next_obs[k][i] = 0
        self.kind[i], self.next_kind[i] = KIND_ID[kind], KIND_ID[next_kind]
        self.action[i], self.ret[i], self.disc[i] = action, ret, disc
        self.tree.set([i], self.max_p ** self.alpha)
        self.pos = (i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, batch, beta):
        total = self.tree.total()
        seg = total / batch
        v = (np.arange(batch) + np.random.rand(batch)) * seg
        idx = np.clip(self.tree.find(v), 0, self.n - 1)
        p = self.tree.tree[idx + self.tree.cap] / total
        w = (self.n * np.maximum(p, 1e-12)) ** (-beta)
        return idx, (w / w.max()).astype(np.float32)

    def update(self, idx, prio):
        prio = np.asarray(prio, np.float64) + 1e-6
        self.max_p = max(self.max_p, float(prio.max()))
        self.tree.set(idx, prio ** self.alpha)

    def batch_obs(self, store, idx, keys, device):
        out = {}
        for k in keys:
            a = store[k][idx]
            if a.dtype == np.float16:
                a = a.astype(np.float32)
            elif a.dtype == np.int16:
                a = a.astype(np.int64)
            out[k] = torch.from_numpy(a).to(device, non_blocking=True)
        return out


KEYS = {"battle": ("mon_ids", "mon_num", "move_num", "ctx_ids", "ctx_num", "mask", "active"),
        "rental": ("mon_ids", "mon_num", "move_num", "ctx_ids", "ctx_num", "lead_mask", "pair_mask"),
        "swap": ("mon_ids", "mon_num", "move_num", "ctx_ids", "ctx_num", "mask")}


def learn(net, target, opt, replay, support, batch, beta, device, max_grad_norm=10.0):
    """One distributional double-DQN step on a prioritized batch. Returns stats."""
    idx, w = replay.sample(batch, beta)
    atoms = support.numel()
    vmin, vmax = support[0].item(), support[-1].item()
    dz = (vmax - vmin) / (atoms - 1)
    ret = torch.from_numpy(replay.ret[idx]).to(device)
    disc = torch.from_numpy(replay.disc[idx]).to(device)
    act_t = torch.from_numpy(replay.action[idx]).to(device)
    logp = torch.zeros(batch, atoms, device=device)
    target_p = torch.zeros(batch, atoms, device=device)
    kinds, next_kinds = replay.kind[idx], replay.next_kind[idx]
    q_taken = torch.zeros(batch, device=device)
    for kid in np.unique(kinds):
        sel = np.nonzero(kinds == kid)[0]
        kind = KIND_NAME[int(kid)]
        x = replay.batch_obs(replay.obs, idx[sel], KEYS[kind], device)
        lp = net(kind, x)                                                   # [b, A, atoms]
        s = torch.from_numpy(sel).to(device)
        lpa = lp[torch.arange(len(sel), device=device), act_t[s]]
        logp[s] = lpa
        q_taken[s] = (lpa.exp() * support).sum(-1).detach()
    with torch.no_grad():
        for kid in np.unique(next_kinds):
            sel = np.nonzero(next_kinds == kid)[0]
            kind = KIND_NAME[int(kid)]
            x = replay.batch_obs(replay.next_obs, idx[sel], KEYS[kind], device)
            mask = action_mask(kind, x)
            if not mask.any(1).all():                  # terminal rows carry an all-zero next obs
                mask = mask | ~mask.any(1, keepdim=True)
            q_online = expected_q(net(kind, x), support).masked_fill(~mask, -1e9)
            a_star = q_online.argmax(-1)                                           # double Q: online selects
            p_next = target(kind, x).exp()[torch.arange(len(sel), device=device), a_star]   # target evaluates
            target_p[torch.from_numpy(sel).to(device)] = p_next
        # project R + disc * z onto the support
        tz = (ret[:, None] + disc[:, None] * support[None]).clamp(vmin, vmax)
        b = ((tz - vmin) / dz).clamp(0, atoms - 1)          # float error could push it past the last atom
        lo, hi = b.floor().long(), b.ceil().long()
        lo[(hi > 0) & (lo == hi)] -= 1
        hi[(lo < atoms - 1) & (lo == hi)] += 1
        m = torch.zeros(batch, atoms, device=device)
        m.scatter_add_(1, lo, target_p * (hi.float() - b))
        m.scatter_add_(1, hi, target_p * (b - lo.float()))
    loss_each = -(m * logp).sum(-1)
    loss = (torch.from_numpy(w).to(device) * loss_each).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
    opt.step()
    replay.update(idx, loss_each.detach().cpu().numpy())
    return {"loss": loss.item(), "q_taken": q_taken.mean().item(), "grad_norm": float(gn),
            "is_weight_mean": float(w.mean())}


class NStep:
    """Per-environment n-step accumulator for one agent. Transitions are pushed when the agent acts; rewards
    arrive when the transition closes; complete n-step transitions are emitted to the replay."""

    def __init__(self, n, gamma):
        self.n, self.gamma = n, gamma
        self.q = []                     # [kind, obs, action, reward or None]

    def push(self, kind, obs, action):
        self.q.append([kind, obs, action, None])

    def close(self, reward, done, trunc, next_kind, next_obs, emit):
        """Close the latest transition. `next_obs` is the agent's next observation (None if terminal)."""
        if not self.q:
            return
        self.q[-1][3] = reward
        if done or trunc:
            # flush everything: terminal -> no bootstrap; truncated -> bootstrap from next_obs
            while self.q:
                self._emit_first(len(self.q), 0.0 if done else None, next_kind, next_obs, emit)
            return
        if len(self.q) >= self.n:
            self._emit_first(self.n, None, next_kind, next_obs, emit)

    def _emit_first(self, k, disc_override, next_kind, next_obs, emit):
        R = sum(self.gamma ** j * self.q[j][3] for j in range(k))
        disc = self.gamma ** k if disc_override is None else disc_override
        kind, obs, action, _ = self.q.pop(0)
        emit(kind, obs, action, R, disc, next_kind, next_obs if disc > 0 else None)

    def clear(self):
        self.q = []

"""Periodic refresh of the team evaluator W during alphazero_v2 (docs/RL_DECISIONS.md §19.6).

W (rl/team_eval.py) is the probability that OUR battler wins one battle with a team; it was trained with the v1
battler, and the v2 battler changes. Once the v2 battler plays as well as v1 (RefreshSchedule), every
--w-refresh-every iterations the trainer:
    1. generates fresh simulated battles with the current battler snapshot in the self-play workers
       (generate_battles: rl/team_eval_data.Generator, the same team / opponent distribution as the original data),
       plus the real self-play battles of the last iterations (RealBattleLog: team, round, battle type, won);
    2. fine-tunes a copy of W from its current weights on them (finetune: the binomial log-likelihood of
       rl.team_eval.train, fewer epochs, lower learning rate);
    3. blends the parameters: new = alpha * fine-tuned + (1 - alpha) * old (blend_params). Parameters rather than
       outputs: both nets start from the same weights and the fine-tune is short, so their average is a network in
       the same basin whose outputs lie between the two (checked by the probe |dW| and the held-out metrics logged
       after blending); it keeps ONE network (the evaluator's cost does not grow) and repeated refreshes are an
       exponential moving average of the weights, whereas an output blend would have to keep every past network;
    4. saves runs/<name>/team_eval_<iter>.pt atomically; the workers and the evaluation reload it.
Metrics (w_refresh/*): data sizes and time, per-battle BCE / Brier / AUC on the fresh held-out teams before (old W),
after fine-tuning and after blending, and mean |dW| on a fixed probe set of teams (probe_teams).
"""

import math
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

TEAM_KEYS = ("round", "btype", "battle", "rents", "species", "moves", "item", "ability", "stats", "set_id",
             "swapped")


# ---- the trigger ------------------------------------------------------------------------------------------------

V1_REFERENCE = (66.0, 66.8, 41.8, 56.6, 21.1, 19.5)   # §18b, "AZ v1, red sola", P(complete round k) in %


def parse_reference(spec):
    """"66.0,66.8,..." (percent, rounds 1..) -> {round: probability}; "none" -> None (never); "0" -> {} (armed at
    the first evaluation)."""
    if spec is None or str(spec).lower() == "none":
        return None
    s = str(spec).strip()
    if s in ("", "0"):
        return {}
    vals = [float(x) for x in s.split(",")]
    return {k + 1: v / 100.0 for k, v in enumerate(vals)}


class RefreshSchedule:
    """Armed once the mean over the last `window` evaluations of mean_k P(complete round k) (the rounds evaluated
    that have a reference) reaches the reference's mean over the same rounds; then a refresh is due at the arming
    iteration and every `every` iterations after the last one."""

    def __init__(self, reference, every=20, window=3):
        self.reference, self.every, self.window = reference, every, window
        self.history = []                   # (iteration, mean P(complete), reference mean)
        self.armed_it = None
        self.last_it = None
        self.count = 0

    @property
    def enabled(self):
        return self.reference is not None and self.every > 0

    def observe(self, it, round_eval):
        if not self.enabled or not round_eval:
            return
        ks = [k for k in sorted(round_eval) if not self.reference or k in self.reference]
        if not ks:
            return
        ref = float(np.mean([self.reference[k] for k in ks])) if self.reference else 0.0
        self.history.append((int(it), float(np.mean([round_eval[k] for k in ks])), ref))
        if self.armed_it is None:
            last = self.history[-self.window:]
            if len(last) == self.window and np.mean([m for _, m, _ in last]) >= ref:
                self.armed_it = int(it)

    def running_mean(self):
        last = self.history[-self.window:]
        return float(np.mean([m for _, m, _ in last])) if last else float("nan")

    def due(self, it):
        if not self.enabled or self.armed_it is None:
            return False
        return self.last_it is None or it - self.last_it >= self.every

    def done(self, it):
        self.last_it = int(it)
        self.count += 1

    def state(self):
        return {"history": self.history, "armed_it": self.armed_it, "last_it": self.last_it, "count": self.count}

    def load(self, st):
        self.history = [tuple(h) for h in st.get("history", [])]
        self.armed_it, self.last_it, self.count = st.get("armed_it"), st.get("last_it"), st.get("count", 0)


# ---- data ----------------------------------------------------------------------------------------------------------

def generate_battles(evaluator, n_battles, seed, battles_per_team=16, concurrency=256, max_decisions=300,
                     p_swap=0.5):
    """At least n_battles simulated battles with `evaluator` (the battler network): rl.team_eval_data's generator,
    new teams started until the budget is reached, then every started team finished. -> to_arrays dict."""
    from .team_eval_data import Generator, to_arrays
    gen = Generator(evaluator, seed, battles_per_team, concurrency, max_decisions, p_swap)
    started = lambda: len(gen.teams) * battles_per_team          # noqa: E731
    while True:
        refill = started() < n_battles
        if not refill and not gen.active:
            break
        gen.step(refill)                  # (overshoots the budget by at most `concurrency` battles)
    teams, bat = gen.take()
    return to_arrays(teams, bat)


def merge_arrays(parts):
    """Concatenate to_arrays dicts (b_team re-indexed)."""
    parts = [p for p in parts if p is not None and len(p["round"])]
    if not parts:
        return None
    out = {k: np.concatenate([p[k] for p in parts]) for k in TEAM_KEYS}
    off, bt = 0, []
    for p in parts:
        bt.append(p["b_team"] + off)
        off += len(p["round"])
    out["b_team"] = np.concatenate(bt).astype(np.int32)
    for k in ("b_won", "b_trunc", "b_decisions"):
        out[k] = np.concatenate([p[k] for p in parts])
    return out


class RealBattleLog:
    """The team of each real self-play battle as the player knows it at the battle's start, and its outcome, in
    rl.team_eval_data's format (one battle per team). start() right after the tactician's action, end() when the
    battle is decided (won) or cut (truncated)."""

    def __init__(self):
        self.teams, self.battles = [], []
        self.cur = None
        self.swapped = [False] * 3

    def on_tactician(self, kind, action_index):
        if kind == "rental":
            self.swapped = [False] * 3
        elif action_index:
            self.swapped[(int(action_index) - 1) // 3] = True     # the traded-in Pokemon takes the slot

    def start(self, backend, ctx):
        from pybattle.emu.decode import SYMBOLS as S, decode_party
        from pybattle.view import OwnMon
        from . import team_eval as T
        from .search import own_set_ids
        g = backend.game
        mons = [OwnMon.from_party(m) for m in decode_party(g.read(S.addr("gPlayerParty"), 300))[:3]]
        r, k = int(ctx["challenge"]) + 1, int(ctx["battle"]) + 1
        btype = 0 if k < 7 else (2 if T.battle_types(r)[6] == T.NOLAND else 1)
        try:
            sets = [int(x) for x in own_set_ids(g)]
        except Exception:                                     # (set ids are not used by training)
            sets = [0, 0, 0]
        self.cur = {"round": r, "btype": btype, "battle": k, "rents": int(ctx["rents"]),
                    "species": [m.species for m in mons], "moves": [(list(m.moves) + [0] * 4)[:4] for m in mons],
                    "item": [m.item for m in mons], "ability": [m.ability for m in mons],
                    "stats": [[m.max_hp] + list(m.stats) for m in mons], "set_id": sets,
                    "swapped": list(self.swapped)}

    def end(self, won, trunc=False, decisions=0):
        if self.cur is None:
            return
        self.battles.append((len(self.teams), bool(won), bool(trunc), int(decisions)))
        self.teams.append(self.cur)
        self.cur = None

    def take(self):
        from .team_eval_data import to_arrays
        if not self.teams:
            return None
        out = to_arrays(self.teams, self.battles)
        self.teams, self.battles = [], []
        return out


# ---- fine-tuning ----------------------------------------------------------------------------------------------------

def _per_battle(ev, d, idx, rng):
    """Predicted W and outcome of every non-truncated battle of teams idx."""
    from .team_eval import hide_swapped
    ids, num = hide_swapped(d, idx, rng)
    W = np.concatenate([ev.w_features(ids[i:i + 8192], num[i:i + 8192], d["round"][idx][i:i + 8192])
                        for i in range(0, len(idx), 8192)]) if len(idx) else np.zeros((0, 3))
    pos = np.full(len(d["round"]), -1)
    pos[idx] = np.arange(len(idx))
    sel = (pos[d["b_team"]] >= 0) & ~d["b_trunc"]
    bt = d["b_team"][sel]
    return W[pos[bt], d["btype"][bt]], d["b_won"][sel].astype(np.float64)


def metrics(ev, d, idx, seed=123):
    from .team_eval import auc
    p, y = _per_battle(ev, d, idx, np.random.default_rng(seed))
    if not len(y):
        return {"brier": float("nan"), "bce": float("nan"), "auc": float("nan"), "n": 0}
    q = np.clip(p, 1e-6, 1 - 1e-6)
    return {"brier": float(np.mean((p - y) ** 2)), "bce": float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))),
            "auc": auc(y, p), "n": int(len(y))}


def finetune(ev, d, idx, epochs=2, lr=2e-4, batch=1024, weight_decay=1e-4, device="cpu", seed=0):
    """A fine-tuned copy of ev.net on teams idx (binomial log-likelihood of won / played, as rl.team_eval.train;
    swapped-in Pokemon hidden with prob 0.5). -> TeamEvaluator on the CPU."""
    import copy
    from .team_eval import TeamEvaluator, hide_swapped, round_index, team_targets
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = torch.device(device)
    net = copy.deepcopy(ev.net).to(dev).train()
    played, won = team_targets(d)
    idx = idx[played[idx] > 0]
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    steps = max(1, epochs * math.ceil(len(idx) / batch))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    losses = []
    for _ in range(epochs):
        order = rng.permutation(idx)
        for i in range(0, len(order), batch):
            b = np.sort(order[i:i + batch])
            ids, num = hide_swapped(d, b, rng)
            x = [torch.from_numpy(v).to(dev) for v in (ids, num, round_index(d["round"][b]))]
            typ = torch.from_numpy(d["btype"][b].astype(np.int64)).to(dev)
            pl = torch.from_numpy(played[b].astype(np.float32)).to(dev)
            wn = torch.from_numpy(won[b].astype(np.float32)).to(dev)
            logit = net(*x).gather(1, typ[:, None])[:, 0]
            loss = (pl * F.softplus(logit) - wn * logit).sum() / pl.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
    net.to("cpu").eval()
    out = TeamEvaluator(net, ev.meta, "cpu")
    out.train_loss = float(np.mean(losses)) if losses else float("nan")
    return out


def blend_params(old, new, alpha):
    """TeamEvaluator whose parameters are alpha * new + (1 - alpha) * old (buffers: new's)."""
    import copy
    from .team_eval import TeamEvaluator
    net = copy.deepcopy(new.net).cpu()
    so, sn = old.net.state_dict(), new.net.state_dict()
    params = {k for k, _ in net.named_parameters()}
    net.load_state_dict({k: (alpha * sn[k].float().cpu() + (1 - alpha) * so[k].float().cpu()).to(sn[k].dtype)
                         if k in params else sn[k] for k in sn})
    return TeamEvaluator(net.eval(), dict(new.meta), "cpu")


def probe_teams(n=1024, seed=12345):
    """A fixed set of teams (rl.team_eval_data.make_case, no battles) to measure how much W moves: (ids, num,
    round, btype)."""
    from .team_eval import hide_swapped
    from .team_eval_data import make_case, to_arrays
    rng = random.Random(seed)
    recs = []
    while len(recs) < n:
        try:
            recs.append(make_case(rng)[3])
        except RuntimeError:
            continue
    d = to_arrays(recs, [])
    ids, num = hide_swapped(d, np.arange(n), np.random.default_rng(seed))
    return {"ids": ids, "num": num, "round": d["round"].astype(np.int64), "btype": d["btype"].astype(np.int64)}


def probe_w(ev, probe):
    return ev.w_features(probe["ids"], probe["num"], probe["round"])


def refresh(ev, data, *, epochs=2, lr=2e-4, alpha=0.5, heldout=0.05, batch=1024, device="cpu", seed=0,
            probe=None):
    """Fine-tune + blend on `data` (to_arrays dict). -> (new TeamEvaluator on the CPU, metrics dict)."""
    t0 = time.time()
    rng = np.random.default_rng(seed)
    n = len(data["round"])
    perm = rng.permutation(n)
    n_held = max(1, int(round(heldout * n))) if n > 1 else 0
    held, tr = np.sort(perm[:n_held]), perm[n_held:]
    m = {"teams": n, "battles": int((~data["b_trunc"]).sum()), "train_teams": len(tr), "heldout_teams": n_held}
    before = metrics(ev, data, held)
    ft = finetune(ev, data, tr, epochs, lr, batch, device=device, seed=seed)
    after_ft = metrics(ft, data, held)
    new = blend_params(ev, ft, alpha)
    after = metrics(new, data, held)
    for name, r in (("before", before), ("finetuned", after_ft), ("after", after)):
        for k in ("brier", "bce", "auc"):
            m[f"{k}_{name}"] = r[k]
    m["heldout_battles"] = before["n"]
    m["train_loss"] = ft.train_loss
    if probe is not None:
        w0, w1 = probe_w(ev, probe), probe_w(new, probe)
        m["probe_abs_dw"] = float(np.abs(w1 - w0).mean())
        m["probe_abs_dw_own_type"] = float(np.abs(w1 - w0)[np.arange(len(w0)), probe["btype"]].mean())
        m["probe_w_mean_before"] = float(w0.mean())
        m["probe_w_mean_after"] = float(w1.mean())
    m["finetune_seconds"] = time.time() - t0
    return new, m

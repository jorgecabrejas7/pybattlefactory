"""Capacity test: can a bigger network fit the value functions better than the current one?

    python -m rl.capacity_test --ckpt runs/ppo_joint_v3/latest.pt --decisions 1500000

1. Collects battler and tactician decisions with the (stochastic) v3 policy, with their Monte Carlo targets
   (battler: 1 if that battle is won, 0 otherwise; with gamma = 1 and no shaping this is its return;
   tactician: battles won from that decision until the run ends).
2. Trains the critic *from scratch* on this fixed dataset (supervised, many epochs) with the current
   architecture (d=128) and a wider one (d=256, d_emb=128, 3 layers), and reports explained variance on the
   training data and on held-out data.
   - If the current net reaches the wider net's held-out EV, capacity is not the bottleneck.
   - If only the wider net climbs, the current net under-fits.
   Targets are noisy (one battle's outcome), so absolute EV is capped by the game's randomness: compare widths.
"""

import argparse
import collections
import json
import os
import time

import numpy as np
import torch

from . import encode
from .envs import VecEnv, decode_action
from .model import FactoryNet
from .policy import Policy

KEYS = ("mon_ids", "mon_num", "move_num", "ctx_ids", "ctx_num", "mask", "active", "lead_mask", "pair_mask")


def compact(o):
    out = {}
    for k, v in o.items():
        if v.dtype == np.int64:
            out[k] = v.astype(np.int16)
        elif v.dtype == np.float32 or v.dtype == np.float64:
            out[k] = v.astype(np.float16)
        else:
            out[k] = v
    return out


def collect(ckpt, n_decisions, workers=16, envs_per_worker=4, seed=4321):
    policy = Policy(ckpt, torch.device("cuda"))
    a = torch.load(ckpt, map_location="cpu")["args"]
    env = VecEnv(workers, envs_per_worker, seed=seed, gamma=1.0, beta=0.0, max_decisions=a["max_decisions"],
                 start_p0=0.5, start_max_round=5)
    n = env.n
    events = env.reset()
    battler, tact = [], []                       # (obs, env, battle_id) / (kind, obs, env, run_id, wins_at_decision)
    battle_id = [0] * n
    run_id = [0] * n
    wins = [0] * n
    battle_result = {}                           # (env, battle_id) -> won
    run_final = {}                               # (env, run_id) -> total wins in run
    t0 = time.time()
    while len(battler) < n_decisions:
        by = collections.defaultdict(list)
        for i, ev in enumerate(events):
            by[ev["kind"]].append(i)
        acts = [None] * n
        for kind, idx in by.items():
            aa = policy.act(kind, encode.collate([events[i]["obs"] for i in idx], policy.device), greedy=False)
            for j, i in enumerate(idx):
                acts[i] = aa[j]
        for i, ev in enumerate(events):
            st = ev["stats"]
            if "battle" in st and st["battle"]["won"] is not None:
                battle_result[(i, battle_id[i])] = bool(st["battle"]["won"])
                wins[i] += int(st["battle"]["won"])
                battle_id[i] += 1
            if "run" in st:
                run_final[(i, run_id[i])] = wins[i]
                run_id[i] += 1
                wins[i] = 0
            if ev["truncate"]:                   # truncated battle: drop its samples (no clean target)
                battle_result[(i, battle_id[i])] = None
                battle_id[i] += 1
                run_final[(i, run_id[i])] = None
                run_id[i] += 1
                wins[i] = 0
                acts[i] = "reset"
                continue
            kind = ev["kind"]
            if kind == "battle":
                battler.append((compact(ev["obs"]), i, battle_id[i]))
            else:
                tact.append((kind, compact(ev["obs"]), i, run_id[i], wins[i]))
            acts[i] = decode_action(kind, acts[i])
        events = env.step(acts)
        if len(battler) % 200_000 < n:
            print(f"collected {len(battler):,} battler / {len(tact):,} tactician decisions "
                  f"({time.time() - t0:.0f}s)", flush=True)
    env.close()
    B = [(o, float(battle_result[(e, b)]), (e, b)) for o, e, b in battler
         if battle_result.get((e, b)) is not None]
    T = [(k, o, float(run_final[(e, r)] - w), (e, r)) for k, o, e, r, w in tact if run_final.get((e, r)) is not None]
    return B, T


def stack(obs_list, keys):
    return {k: np.stack([o[k] for o in obs_list]) for k in keys if k in obs_list[0]}


def to_t(d, idx, device):
    out = {}
    for k, v in d.items():
        a = v[idx]
        a = a.astype(np.float32) if a.dtype == np.float16 else (a.astype(np.int64) if a.dtype == np.int16 else a)
        out[k] = torch.from_numpy(a).to(device)
    return out


def ev(pred, y):
    var = np.var(y)
    return float(1 - np.var(y - pred) / var) if var > 0 else 0.0


def fit(net_kw, data_by_kind, targets_by_kind, head, epochs, device, bs=1024, lr=3e-4, seed=0, groups_by_kind=None):
    """Supervised fit of one critic head. data_by_kind: {kind: dict of arrays}, targets: {kind: array}."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = FactoryNet(**net_kw).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    splits = {}
    for kind, y in targets_by_kind.items():
        # held-out = whole battles (battler) / whole runs (tactician): decisions of one battle share its target,
        # so a per-decision split would let the net score on battles it has already seen
        g = groups_by_kind[kind]
        uniq = np.unique(g)
        test_groups = set(rng.choice(uniq, size=max(1, len(uniq) // 10), replace=False).tolist())
        is_test = np.array([x in test_groups for x in g])
        splits[kind] = (rng.permutation(np.nonzero(~is_test)[0]), np.nonzero(is_test)[0])
    mu = np.mean(np.concatenate([y for y in targets_by_kind.values()]))
    sd = np.std(np.concatenate([y for y in targets_by_kind.values()])) + 1e-6

    def value(kind, x):
        if kind == "battle":
            return net.battler(x)[1]
        if kind == "swap":
            return net.swap(x)[1]
        return net.rental(x, lead=torch.zeros(len(x["mon_ids"]), dtype=torch.long, device=device))[3]

    def predict(kind, idx):
        out = []
        with torch.no_grad():
            for s in range(0, len(idx), 8192):
                out.append(value(kind, to_t(data_by_kind[kind], idx[s:s + 8192], device)).float().cpu().numpy())
        return np.concatenate(out) * sd + mu

    hist = []
    for ep in range(epochs):
        net.train()
        order = [(kind, i) for kind, (tr, _) in splits.items() for i in np.array_split(rng.permutation(tr),
                                                                                    max(1, len(tr) // bs))]
        rng.shuffle(order)
        for kind, idx in order:
            x = to_t(data_by_kind[kind], idx, device)
            y = torch.from_numpy(((targets_by_kind[kind][idx] - mu) / sd).astype(np.float32)).to(device)
            loss = ((value(kind, x) - y) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        tr_p = np.concatenate([predict(k, s[0][:200_000]) for k, s in splits.items()])
        tr_y = np.concatenate([targets_by_kind[k][s[0][:200_000]] for k, s in splits.items()])
        te_p = np.concatenate([predict(k, s[1]) for k, s in splits.items()])
        te_y = np.concatenate([targets_by_kind[k][s[1]] for k, s in splits.items()])
        hist.append({"epoch": ep + 1, "train_ev": ev(tr_p, tr_y), "heldout_ev": ev(te_p, te_y)})
        print(f"  [{head} {net_kw}] epoch {ep + 1}: train EV {hist[-1]['train_ev']:.3f}  "
              f"held-out EV {hist[-1]['heldout_ev']:.3f}", flush=True)
    return hist, sum(p.numel() for p in net.parameters())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--decisions", type=int, default=1_500_000)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--out", default="runs/capacity_test.json")
    args = p.parse_args()
    device = torch.device("cuda")
    t0 = time.time()
    Policy(args.ckpt, device)                    # sets the encode version (v3) before workers fork
    B, T = collect(args.ckpt, args.decisions)
    print(f"dataset: {len(B):,} battler, {len(T):,} tactician samples ({time.time() - t0:.0f}s)", flush=True)
    a = torch.load(args.ckpt, map_location="cpu")["args"]
    small = dict(d_emb=a["d_emb"], d=a["d"], layers=a["layers"], heads=a["heads"], share=a.get("share", "all"))
    wide = dict(d_emb=2 * a["d_emb"], d=2 * a["d"], layers=a["layers"] + 1, heads=a["heads"], share=small["share"])
    res = {"battler_samples": len(B), "tactician_samples": len(T)}
    bdata = {"battle": stack([o for o, _, _ in B], ("mon_ids", "mon_num", "move_num", "ctx_ids", "ctx_num", "mask",
                                                 "active"))}
    btarg = {"battle": np.array([y for _, y, _ in B], np.float32)}
    bgrp = {"battle": np.array([hash(g) for _, _, g in B])}
    tdata, ttarg, tgrp = {}, {}, {}
    for kind in ("rental", "swap"):
        rows = [(o, y, g) for k, o, y, g in T if k == kind]
        if rows:
            keys = ("mon_ids", "mon_num", "move_num", "ctx_ids", "ctx_num") + \
                   (("lead_mask", "pair_mask") if kind == "rental" else ("mask",))
            tdata[kind] = stack([o for o, _, _ in rows], keys)
            ttarg[kind] = np.array([y for _, y, _ in rows], np.float32)
            tgrp[kind] = np.array([hash(g) for _, _, g in rows])
    for name, kw in (("current", small), ("wide", wide)):
        hb, nb = fit(kw, bdata, btarg, "battler", args.epochs, device, groups_by_kind=bgrp)
        ht, nt = fit(kw, tdata, ttarg, "tactician", args.epochs, device, groups_by_kind=tgrp)
        res[name] = {"net": kw, "params": nb, "battler": hb, "tactician": ht}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print(json.dumps({k: (v if not isinstance(v, dict) else {
        "params": v["params"], "battler_final": v["battler"][-1], "tactician_final": v["tactician"][-1]})
        for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    main()

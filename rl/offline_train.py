"""Offline battler training on the v5 dataset (rl/offline_data.py, docs/RL_DECISIONS.md §27): a battler network from
scratch, distilled from alphazero_v4's recorded search targets, to measure how much capacity / compute / target
changes help BEFORE a long v5 run.

    python -m rl.offline_train --name control                 # v4's architecture (d 128, 2 layers), the recipe below
    python -m rl.offline_train --name L4 --layers 4
    python -m rl.offline_train --name w256 --d 256 --heads 8
    python -m rl.offline_train --name vz --value-mix 0        # value target: outcome z only (v4: 0.5 z + 0.5 root Q)
    python -m rl.offline_train --name s1024 --subset runs/v5_offline/subset_400k --targets runs/v5_offline/var_s1024
    python -m rl.offline_train --v4-reference                 # alphazero_v4's own held-out numbers on the dataset
    python -m rl.offline_train --speed                        # forward-pass cost of the architectures

Recipe: the battler loss of rl.alphazero.train_iteration (cross-entropy to the recorded policy target + vf_coef x
0.5 MSE of the value on ValueNorm-normalized targets; target = (1 - m) z + m root_Q, z where the decision had a single
legal action), AdamW (lr --lr, weight decay 1e-4, eps 1e-5), gradient clip 1.0, batch --batch, a linear warmup then a
cosine decay to 5 % over --epochs passes over the training rows. ValueNorm: the exact mean / variance of the training
targets (training uses a running estimate of the same thing). Held out: whole runs (offline_data.is_heldout, 5 %).

Held-out metrics, each epoch: KL(target || net) and cross-entropy, top-1 agreement with the target's argmax; value
explained variance against the training target and against z, and Brier score (mean (v - z)^2, untruncated battles).

The checkpoint (runs/v5_offline/models/<name>/final.pt and best.pt, best = lowest held-out KL + value loss) is an
alphazero checkpoint (v4's args with this network's size), so `python -m rl.agent_probe battler --battler
net:runs/v5_offline/models/<name>/final.pt` plays it on benchmark B. Only the battler is trained: the tactician's
heads stay at their initialization (and share the embedding tables), so the checkpoint is a battler, not an agent.
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import offline_data as OD

V4_CKPT = "runs/alphazero_v4/stopped_it1111.pt"
MODELS = "runs/v5_offline/models"


def explained_variance(pred, target):
    var = np.var(target)
    return float(1 - np.var(target - pred) / var) if var > 0 else 0.0


def value_target(z, q, mix):
    """(1 - mix) z + mix q; z where there was no search value (q NaN). rl.alphazero.value_target."""
    return np.where(np.isnan(q), z, (1.0 - mix) * z + mix * np.nan_to_num(q)).astype(np.float32)


class Data:
    """Rows of the dataset with their policy target / value target in RAM; inputs read from the memory map."""

    def __init__(self, data_dir, subset=None, targets=None, value_mix=0.5, limit=None, seed=0):
        self.ds = OD.Dataset(data_dir)
        n = len(self.ds)
        rows = np.arange(n) if subset is None else np.load(os.path.join(subset, "rows.npy"))
        pi = np.asarray(self.ds["target"]).astype(np.float32)
        q = np.asarray(self.ds["root_q"]).astype(np.float32)
        if targets is not None:                        # another search's targets for these rows (experiment 4)
            with np.load(os.path.join(targets, "targets.npz")) as t:
                tr = t["row"]
                pi[tr] = t["target"]
                q[tr] = t["root_q"]
                have = np.zeros(n, bool)
                have[tr] = True
            rows = rows[have[rows]]
        self.pi, self.q = pi, q
        self.pi_base = pi if targets is None else np.asarray(self.ds["target"]).astype(np.float32)
        self.z = np.asarray(self.ds["z"]).astype(np.float32)
        self.trunc = np.asarray(self.ds["trunc"])
        self.vt = value_target(self.z, q, value_mix)
        held = self.ds.held[rows]
        self.train, self.held = rows[~held], rows[held]
        if limit is not None and limit < len(self.train):    # a fraction of the training rows, by run
            run = np.asarray(self.ds["run"])[self.train]
            uniq = np.unique(run)
            keep = np.random.default_rng(seed).permutation(uniq)
            cnt = np.cumsum(np.bincount(np.searchsorted(uniq, run), minlength=len(uniq))[np.searchsorted(uniq, keep)])
            keep = keep[:np.searchsorted(cnt, limit) + 1]
            self.train = self.train[np.isin(run, keep)]
        self.v_mean = float(self.vt[self.train].mean())
        self.v_var = float(max(self.vt[self.train].var(), 1e-4))
        self.threat = None                  # --threat: rl.threat_features per row (set by train)

    def batch(self, idx, device):
        idx = np.sort(idx)
        x = self.ds.obs_rows(idx)
        if self.threat is not None:
            x["threat"] = np.asarray(self.threat[idx], np.float32)
        x = {k: torch.from_numpy(np.ascontiguousarray(v)).to(device, non_blocking=True) for k, v in x.items()}
        return (idx, x, torch.from_numpy(self.pi[idx]).to(device, non_blocking=True),
                torch.from_numpy(self.vt[idx]).to(device, non_blocking=True))


def iterate(data, idx_list, device, threads=4, depth=8):
    """Yield data.batch(idx, device) for each index array in order, loaded `depth` ahead by `threads` threads."""
    import collections
    from concurrent.futures import ThreadPoolExecutor
    it = iter(idx_list)
    with ThreadPoolExecutor(threads) as ex:
        futs = collections.deque(ex.submit(data.batch, ix, device) for _, ix in zip(range(depth), it))
        while futs:
            b = futs.popleft().result()
            nx = next(it, None)
            if nx is not None:
                futs.append(ex.submit(data.batch, nx, device))
            yield b


def build_net(a, opt_feat=7):
    from .model import FactoryNet
    if getattr(a, "threat", None):           # + the threat features (rl/threat_features.py, docs §27.x)
        from .threat_features import N_THREAT, ThreatNet
        return ThreatNet(a.d_emb, a.d, a.layers, a.heads, share="embeddings", opt_feat=opt_feat,
                         threat_dim=N_THREAT)
    return FactoryNet(a.d_emb, a.d, a.layers, a.heads, share="embeddings", opt_feat=opt_feat)


def battler_params(net):
    return sum(p.numel() for n, p in net.named_parameters() if n.startswith(("trunk.", "b_")))


def losses(net, x, pi, vt_n):
    logits, v = net.battler(x)
    logq = F.log_softmax(logits.float(), -1).masked_fill(~x["mask"], -1e9)
    ce = -(pi * logq).sum(-1)
    h = -(pi * torch.log(pi.clamp_min(1e-12))).sum(-1)
    vl = 0.5 * (v.float() - vt_n) ** 2
    return ce, ce - h, vl, logits, v


@torch.no_grad()
def evaluate(net, data, rows, device, v_mean, v_std, batch=4096):
    net.eval()
    rows = np.sort(rows)
    kl, ce, agree, vals, logqs = [], [], [], [], []
    for _, x, pi, _ in iterate(data, np.array_split(rows, max(1, math.ceil(len(rows) / batch))), device):
        c, k, _, logits, v = losses(net, x, pi, torch.zeros_like(pi[:, 0]))
        kl.append(k.cpu().numpy())
        ce.append(c.cpu().numpy())
        agree.append((logits.argmax(-1) == pi.argmax(-1)).cpu().numpy())
        vals.append((v.float() * v_std + v_mean).clamp(0, 1).cpu().numpy())
        logqs.append(F.log_softmax(logits.float(), -1).cpu().numpy())
    v = np.concatenate(vals)
    out = metrics(np.concatenate(kl), np.concatenate(ce), np.concatenate(agree), v, data, rows)
    if data.pi_base is not data.pi:              # experiment 4: also against the recorded (256, factory_sets) target
        out["kl_base"] = float(kl_to(data.pi_base[rows], np.concatenate(logqs)).mean())
    return out


def kl_to(pi, logq):
    """KL(pi || q) per row; logq: log-probabilities (masked actions have pi = 0)."""
    pi = pi.astype(np.float64)
    m = pi > 0
    return np.where(m, pi * (np.log(np.where(m, pi, 1.0)) - np.where(m, logq, 0.0)), 0.0).sum(1)


def metrics(kl, ce, agree, v, data, rows):
    vt, z, tr = data.vt[rows], data.z[rows], data.trunc[rows]
    rnd = np.asarray(data.ds["round"])[rows]
    nol = np.asarray(data.ds["noland"])[rows]
    out = {"kl": float(kl.mean()), "ce": float(ce.mean()), "top1": float(agree.mean()),
           "ev_target": explained_variance(v, vt), "ev_z": explained_variance(v[~tr], z[~tr]),
           "brier": float(((v[~tr] - z[~tr]) ** 2).mean()), "n": int(len(rows))}
    for name, m in (("r1_4", (rnd <= 4) & ~nol), ("r5_7", (rnd >= 5) & ~nol), ("noland", nol)):
        if m.any():
            out[f"kl_{name}"] = float(kl[m].mean())
            out[f"brier_{name}"] = float(((v[m & ~tr] - z[m & ~tr]) ** 2).mean()) if (m & ~tr).any() else None
    return out


def v4_reference(a):
    """alphazero_v4's own network on the held-out rows, from the recorded prior / value (no recomputation)."""
    data = Data(a.data, a.subset, a.targets, a.value_mix)
    rows = np.sort(data.held)
    pri = np.asarray(data.ds["prior"])[rows].astype(np.float64)
    mask = data.ds.obs_rows(rows)["mask"]
    pri = np.where(mask, pri, 0)
    pri /= pri.sum(1, keepdims=True)
    pi = data.pi[rows].astype(np.float64)
    logq = np.log(np.maximum(pri, 1e-30))
    ce = -(pi * np.where(pi > 0, logq, 0)).sum(1)
    h = -(pi * np.log(np.maximum(pi, 1e-12))).sum(1)
    agree = pri.argmax(1) == pi.argmax(1)
    v = np.asarray(data.ds["value_net"])[rows]
    m = metrics(ce - h, ce, agree, v, data, rows)
    # the noise floor of the target: one search's target vs the mean target is not available per row; report the
    # target's own entropy as context
    m["target_entropy"] = float(h.mean())
    print(json.dumps(m, indent=1))
    os.makedirs(MODELS, exist_ok=True)
    json.dump(m, open(os.path.join(MODELS, f"v4_reference{'_' + os.path.basename(a.targets) if a.targets else ''}"
                                           f"{'_sub' if a.subset else ''}.json"), "w"), indent=1)
    return m


def save_ckpt(net, a, data, path, epoch, extra):
    import torch as T
    v4 = T.load(V4_CKPT, map_location="cpu", weights_only=False)
    args = dict(v4["args"], d_emb=a.d_emb, d=a.d, layers=a.layers, heads=a.heads, variant="offline_battler",
                offline=dict(vars(a), **extra), value_mix_b=a.value_mix)
    # v4's opponent model: a search with this battler (agent_probe search:CKPT) draws opponents as v4's does
    T.save({"net": {k: v.detach().cpu() for k, v in net.state_dict().items()}, "args": args,
            "value_norm": {"battler": {"mean": data.v_mean, "var": data.v_var, "seen": True}, "tactician": None},
            "iteration": int(epoch), "battler_steps": 0, "opponent_model": v4.get("opponent_model")},
           path + ".tmp")
    os.replace(path + ".tmp", path)


def train(a):
    from .search import mark_training
    mark_training()
    torch.manual_seed(a.seed)
    device = torch.device(a.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from . import encode
    encode.set_version(OD.ENCODE_VERSION)
    out = os.path.join(MODELS, a.name)
    os.makedirs(out, exist_ok=True)
    data = Data(a.data, a.subset, a.targets, a.value_mix, a.limit, a.seed)
    if getattr(a, "threat", None):
        data.threat = np.load(os.path.join(a.threat, "threat.npy"), mmap_mode="r")
        assert len(data.threat) == len(data.ds) and np.load(os.path.join(a.threat, "done.npy")).all()
    net = build_net(a).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.weight_decay, eps=1e-5)
    v_mean, v_std = data.v_mean, data.v_var ** 0.5
    steps_per_epoch = len(data.train) // a.batch
    total = steps_per_epoch * a.epochs
    warm = min(a.warmup, total // 10)
    rng = np.random.default_rng(a.seed)
    info = {"train_rows": int(len(data.train)), "held_rows": int(len(data.held)),
            "battler_params": battler_params(net), "steps": total}
    print(json.dumps(info), flush=True)
    hist = []
    best = None
    step = 0
    t_start = time.time()
    for ep in range(a.epochs):
        net.train()
        perm = rng.permutation(data.train)
        idx_list = [perm[i * a.batch:(i + 1) * a.batch] for i in range(steps_per_epoch)]
        t0 = time.time()
        tl = []
        for _, x, pi, vt in iterate(data, idx_list, device, threads=a.loader_threads):
            lr = a.lr * (step + 1) / (warm + 1) if step < warm else \
                a.lr * (a.min_frac + (1 - a.min_frac) * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(total - warm, 1))))
            for g in opt.param_groups:
                g["lr"] = lr
            vt_n = (vt - v_mean) / v_std
            ce, kl, vl, _, _ = losses(net, x, pi, vt_n)
            loss = ce.mean() + a.vf_coef * vl.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), a.max_grad_norm)
            opt.step()
            step += 1
            if step % 200 == 0:
                tl.append((kl.mean().item(), vl.mean().item()))
        t_ep = time.time() - t0
        m = evaluate(net, data, data.held, device, v_mean, v_std)
        m.update(epoch=ep + 1, lr=lr, seconds=t_ep, train_kl=float(np.mean([t[0] for t in tl])) if tl else None,
                 train_vloss=float(np.mean([t[1] for t in tl])) if tl else None)
        hist.append(m)
        print(f"[{a.name} ep {ep + 1}/{a.epochs} {t_ep:.0f}s] held KL {m['kl']:.4f} top1 {m['top1']:.3f} "
              f"EV {m['ev_target']:.3f} EV_z {m['ev_z']:.3f} brier {m['brier']:.4f}  train KL {m['train_kl']}",
              flush=True)
        score = m["kl"] + m["brier"]
        if best is None or score < best[0]:
            best = (score, ep + 1)
            save_ckpt(net, a, data, os.path.join(out, "best.pt"), ep + 1, {"best_epoch": ep + 1})
        json.dump({"info": info, "args": vars(a), "history": hist, "best_epoch": best[1]},
                  open(os.path.join(out, "metrics.json"), "w"), indent=1)
    save_ckpt(net, a, data, os.path.join(out, "final.pt"), a.epochs, {"best_epoch": best[1]})
    info["minutes"] = (time.time() - t_start) / 60
    json.dump({"info": info, "args": vars(a), "history": hist, "best_epoch": best[1],
               "speed": speed_of(a)}, open(os.path.join(out, "metrics.json"), "w"), indent=1)
    return hist


@torch.no_grad()
def speed_of(a, batches=(64, 256, 1024), reps=30):
    """Forward-pass milliseconds of the battler head on the GPU (fp32, as the inference server) per batch size."""
    from . import encode
    encode.set_version(OD.ENCODE_VERSION)
    if not torch.cuda.is_available():
        return {}
    dev = torch.device("cuda")
    net = build_net(a).to(dev).eval()
    torch.backends.cuda.matmul.allow_tf32 = False
    ds = OD.Dataset(a.data)
    out = {}
    for b in batches:
        x = {k: torch.from_numpy(v).to(dev) for k, v in ds.obs_rows(np.arange(b) % len(ds)).items()}
        for _ in range(5):
            net.battler(x)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(reps):
            net.battler(x)
        torch.cuda.synchronize()
        out[str(b)] = (time.perf_counter() - t) / reps * 1000
    xc = {k: torch.from_numpy(v) for k, v in ds.obs_rows(np.arange(32) % len(ds)).items()}
    netc = build_net(a).eval()
    torch.set_num_threads(1)
    t = time.perf_counter()
    for _ in range(10):
        netc.battler(xc)
    out["cpu32_1thread"] = (time.perf_counter() - t) / 10 * 1000
    out["battler_params"] = battler_params(net)
    return out


def load_model(ck_path, device):
    """A models/<name>/*.pt checkpoint -> (net, v_mean, v_std, offline args namespace)."""
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    oa = argparse.Namespace(**ck["args"]["offline"])
    net = build_net(oa)
    net.load_state_dict(ck["net"])
    vn = ck["value_norm"]["battler"]
    return net.to(device).eval(), vn["mean"], max(vn["var"], 1e-4) ** 0.5, oa


def eval_targets(names, targets, data_dir=OD.DATA_DIR, device="cuda"):
    """Held-out KL of each model against another target set (e.g. the 1024-simulation targets: a less noisy
    reference), on the held-out rows that have it. -> models/<name>/eval_<targets>.json"""
    from . import encode
    encode.set_version(OD.ENCODE_VERSION)
    data = Data(data_dir, None, targets, 0.5)
    tag = os.path.basename(os.path.normpath(targets))
    res = {}
    for nm in names:
        net, vm, vs, oa = load_model(model_ckpt(nm), device)
        if getattr(oa, "threat", None):
            data.threat = np.load(os.path.join(oa.threat, "threat.npy"), mmap_mode="r")
        else:
            data.threat = None
        m = evaluate(net, data, data.held, device, vm, vs)
        res[nm] = {"kl": m["kl"], "top1": m["top1"], "n": m["n"], "kl_base": m.get("kl_base")}
        json.dump(res[nm], open(os.path.join(MODELS, nm, f"eval_{tag}.json"), "w"), indent=1)
        print(nm, json.dumps(res[nm]), flush=True)
    return res


# ---- benchmark B and the report ------------------------------------------------------------------------------------

def model_ckpt(name, which="final"):
    return name if name.endswith(".pt") else os.path.join(MODELS, name, f"{which}.pt")


def bench(names, procs=2, which="final"):
    """Play each model's network on benchmark B (rl.agent_probe battler, a separate non-training process)."""
    import subprocess
    import sys
    for nm in names:
        ck = model_ckpt(nm, which)
        subprocess.run(["nice", "-n", "10", sys.executable, "-m", "rl.agent_probe", "battler", "--battler",
                        f"net:{ck}", "--procs", str(procs)], check=True, stdout=subprocess.DEVNULL)


def boot_ci(d, B=2000, seed=0):
    """Mean and percentile-bootstrap 95 % interval of the mean over independent units (benchmark states)."""
    d = np.asarray(d, np.float64)
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def bench_summary(ck, ref=None, ref2=None):
    """Win % on B (total and by stratum) and paired differences (bootstrap over states) against ref / ref2."""
    from . import agent_probe as P
    s = P.summarize_battler(f"net:{ck}")
    bench_ = P.Bench()
    per = s["per_state"]
    out = {"n_states": len(per), "win": boot_ci(list(per.values()))}
    strata = sorted({bench_.by_id[i]["stratum"] for i in per})
    out["by_stratum"] = {st: boot_ci([v for i, v in per.items() if bench_.by_id[i]["stratum"] == st])
                         for st in strata}
    for tag, r in (("vs_ref", ref), ("vs_ref2", ref2)):
        if r is None:
            continue
        ids = sorted(set(per) & set(r))
        out[tag] = {"all": boot_ci([per[i] - r[i] for i in ids])}
        for st in strata:
            out[tag][st] = boot_ci([per[i] - r[i] for i in ids if bench_.by_id[i]["stratum"] == st])
    return out, per


def report(names, control="control", out_path="runs/v5_offline/report"):
    """Held-out metrics + benchmark B of every model, paired against the control and v4 it1111 (markdown + json)."""
    from . import agent_probe as P
    v4 = P.summarize_battler(f"net:{V4_CKPT}")["per_state"]
    ctrl = P.summarize_battler(f"net:{model_ckpt(control)}")["per_state"]
    rows = {}
    v4b, _ = bench_summary(V4_CKPT, None, None)
    rows["v4_it1111"] = {"bench": v4b, "heldout": _load_json(os.path.join(MODELS, "v4_reference.json"))}
    for nm in names:
        ck = model_ckpt(nm)
        if not os.path.exists(ck):
            continue
        b, _ = bench_summary(ck, ctrl, v4)
        mj = _load_json(os.path.join(MODELS, nm, "metrics.json")) or {}
        h = mj.get("history") or [{}]
        rows[nm] = {"bench": b, "heldout": h[-1], "best_epoch": mj.get("best_epoch"), "info": mj.get("info"),
                    "speed": mj.get("speed"), "args": mj.get("args")}
    json.dump(rows, open(out_path + ".json", "w"), indent=1)

    def pct(t):
        return f"{100 * t[0]:.1f}" if t else "-"

    def dci(t):
        return f"{100 * t[0]:+.1f} [{100 * t[1]:+.1f}, {100 * t[2]:+.1f}]" if t else "-"

    L = ["| model | params (battler) | held KL | top-1 | EV | EV z | Brier | B win % | R5 | R6 | R3 | Noland | "
         "Δ vs control [95 % IC] | Δ vs v4 [95 % IC] | fwd ms b256 (×ctrl) |", "|" + "---|" * 15]
    sp0 = (rows.get(control, {}).get("speed") or {}).get("256")
    for nm, r in rows.items():
        h, b, sp = r["heldout"] or {}, r["bench"], r.get("speed") or {}
        bs = b["by_stratum"]
        f = (lambda k: f"{h[k]:.3f}" if h.get(k) is not None else "-")
        ms = sp.get("256")
        L.append(f"| {nm} | {sp.get('battler_params', '-')} | {f('kl')} | {f('top1')} | {f('ev_target')} | "
                 f"{f('ev_z')} | {f('brier')} | {pct(b['win'])} | {pct(bs.get('r5'))} | {pct(bs.get('r6'))} | "
                 f"{pct(bs.get('r3'))} | {pct(bs.get('noland'))} | {dci((b.get('vs_ref') or {}).get('all'))} | "
                 f"{dci((b.get('vs_ref2') or {}).get('all'))} | "
                 f"{f'{ms:.2f}' if ms else '-'}{f' ({ms / sp0:.1f}×)' if ms and sp0 else ''} |")
    L += ["", "Paired difference by stratum (points, 95 % bootstrap IC):", "",
          "| model | vs | R5 | R6 | R3 | Noland |", "|---|---|---|---|---|---|"]
    for nm, r in rows.items():
        for tag, lab in (("vs_ref", "control"), ("vs_ref2", "v4")):
            d = r["bench"].get(tag)
            if d:
                L.append(f"| {nm} | {lab} | {dci(d.get('r5'))} | {dci(d.get('r6'))} | {dci(d.get('r3'))} | "
                         f"{dci(d.get('noland'))} |")
    open(out_path + ".md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def _load_json(p):
    return json.load(open(p)) if os.path.exists(p) else None


def report_groups(spec, out_path="runs/v5_offline/report"):
    """Groups of models (seeds of one variant): spec "name=m1+m2+m3,name2=...", the first group is the control.
    Per benchmark state the group's win rate is the mean over its models; paired differences (bootstrap over the
    states) against the control group and against v4 it1111; held-out metrics averaged over the group's models."""
    from . import agent_probe as P
    groups = [(g.split("=")[0], g.split("=")[1].split("+")) for g in spec.split(",")]
    bench_ = P.Bench()
    v4 = P.summarize_battler(f"net:{V4_CKPT}")["per_state"]
    per = {}
    for name, ms in groups:
        ps = [P.summarize_battler(f"net:{model_ckpt(m)}")["per_state"] for m in ms]
        ids = sorted(set.intersection(*[set(p) for p in ps]))
        per[name] = {i: float(np.mean([p[i] for p in ps])) for i in ids}
    ctrl = groups[0][0]
    strata = ("r5", "r6", "r3", "noland")
    rows = {}
    ref = _load_json(os.path.join(MODELS, "v4_reference.json")) or {}
    speeds = _load_json("runs/v5_offline/speed.json")

    def bench_of(p, r):
        ids = sorted(set(p) & set(r)) if r is not None else sorted(p)
        o = {"win": boot_ci([p[i] for i in ids]),
             "by": {s: boot_ci([p[i] for i in ids if bench_.by_id[i]["stratum"] == s]) for s in strata}}
        if r is not None:
            o["d"] = boot_ci([p[i] - r[i] for i in ids])
            o["d_by"] = {s: boot_ci([p[i] - r[i] for i in ids if bench_.by_id[i]["stratum"] == s]) for s in strata}
        return o

    rows["v4_it1111"] = {"n": 1, "heldout": ref, "b": bench_of(v4, None), "vs_v4": None, "vs_ctrl": None,
                         "speed": None}
    for name, ms in groups:
        mets = [(_load_json(os.path.join(MODELS, m, "metrics.json")) or {}) for m in ms]
        hs = [m["history"][-1] for m in mets if m.get("history")]
        held = {k: float(np.mean([h[k] for h in hs if h.get(k) is not None]))
                for k in ("kl", "top1", "ev_target", "ev_z", "brier") if hs}
        rows[name] = {"n": len(ms), "models": ms, "heldout": held, "b": bench_of(per[name], None),
                      "vs_ctrl": bench_of(per[name], per[ctrl]) if name != ctrl else None,
                      "vs_v4": bench_of(per[name], v4), "speed": next((m.get("speed") for m in mets if m.get("speed")),
                                                                      None)}
        arch = next((f"d{m['args']['d']}_L{m['args']['layers']}" for m in mets if m.get("args")), None)
        if speeds and arch in speeds:          # the interleaved measurement (speed_all) when there is one
            rows[name]["speed"] = speeds[arch]
    json.dump(rows, open(out_path + ".json", "w"), indent=1)

    def pct(t):
        return f"{100 * t[0]:.1f}" if t else "-"

    def dci(t):
        return f"{100 * t[0]:+.1f} [{100 * t[1]:+.1f}, {100 * t[2]:+.1f}]" if t else "-"

    sp0 = (rows[ctrl]["speed"] or {}).get("256")
    L = ["| variant | seeds | params (battler) | held KL | top-1 | EV | EV z | Brier | B win % | R5 | R6 | R3 | Noland "
         "| Δ vs control | Δ vs v4 | fwd ms b256 |", "|" + "---|" * 16]
    for name, r in rows.items():
        h, b, sp = r["heldout"] or {}, r["b"], r["speed"] or {}
        f = (lambda k: f"{h[k]:.3f}" if h.get(k) is not None else "-")
        ms = sp.get("256")
        L.append(f"| {name} | {r['n']} | {sp.get('battler_params', '-')} | {f('kl')} | {f('top1')} | {f('ev_target')} "
                 f"| {f('ev_z')} | {f('brier')} | {pct(b['win'])} | {pct(b['by']['r5'])} | {pct(b['by']['r6'])} | "
                 f"{pct(b['by']['r3'])} | {pct(b['by']['noland'])} | {dci((r['vs_ctrl'] or {}).get('d'))} | "
                 f"{dci((r['vs_v4'] or {}).get('d'))} | {f'{ms:.2f}' if ms else '-'}"
                 f"{f' ({ms / sp0:.2f}×)' if ms and sp0 else ''} |")
    L += ["", "Δ by stratum (points, 95 % bootstrap IC over states):", "",
          "| variant | vs | R5 | R6 | R3 | Noland |", "|---|---|---|---|---|---|"]
    for name, r in rows.items():
        for tag, lab in (("vs_ctrl", "control"), ("vs_v4", "v4")):
            d = r.get(tag)
            if d:
                L.append(f"| {name} | {lab} | " + " | ".join(dci(d["d_by"][s]) for s in strata) + " |")
    open(out_path + ".md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    return rows


SPEED_ARCHS = {"d128_L2": (128, 2, 4), "d128_L4": (128, 4, 4), "d128_L6": (128, 6, 4), "d256_L2": (256, 2, 8),
               "d256_L4": (256, 4, 8)}


def speed_all(a, rounds=5, path="runs/v5_offline/speed.json"):
    """Forward cost of every architecture, interleaved over `rounds` rounds (median), so that a busy GPU affects
    them alike. Also the battler parameters. -> runs/v5_offline/speed.json"""
    res = {k: [] for k in SPEED_ARCHS}
    for _ in range(rounds):
        for k, (d, layers, heads) in SPEED_ARCHS.items():
            b = argparse.Namespace(**dict(vars(a), d=d, layers=layers, heads=heads, threat=None))
            res[k].append(speed_of(b))
    out = {}
    for k, rs in res.items():
        out[k] = {key: float(np.median([r[key] for r in rs])) for key in rs[0]}
    base = out["d128_L2"]
    for k in out:
        for key in ("64", "256", "1024", "cpu32_1thread"):
            out[k][f"x_{key}"] = out[k][key] / base[key]
    json.dump(out, open(path, "w"), indent=1)
    print(json.dumps(out, indent=1))
    return out


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--name", default="control")
    p.add_argument("--data", default=OD.DATA_DIR)
    p.add_argument("--subset", default=None, help="a subset dir with rows.npy (offline_data subset)")
    p.add_argument("--targets", default=None, help="a re-search dir with targets.npz (offline_data research)")
    p.add_argument("--limit", type=int, default=None, help="use only this many training rows (whole runs)")
    p.add_argument("--value-mix", type=float, default=0.5, help="value target (1-m) z + m root_Q (v4: 0.5)")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-frac", type=float, default=0.05)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--vf-coef", type=float, default=1.0)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--d-emb", type=int, default=64)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--loader-threads", type=int, default=4)
    p.add_argument("--threat", default=None, help="a rl.threat_features compute dir: + the threat features")
    p.add_argument("--v4-reference", action="store_true")
    p.add_argument("--speed", action="store_true")
    p.add_argument("--bench", default=None, help="comma-separated model names: play them on benchmark B")
    p.add_argument("--report", default=None, help="comma-separated model names: the comparison table")
    p.add_argument("--which", default="final", help="--bench: final or best checkpoint")
    p.add_argument("--speed-all", action="store_true", help="forward cost of every architecture (speed.json)")
    p.add_argument("--eval-targets", default=None, help="with --models: held-out KL against this targets dir")
    p.add_argument("--models", default=None, help="comma-separated model names (--eval-targets)")
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.v4_reference:
        return v4_reference(a)
    if a.eval_targets:
        return eval_targets(a.models.split(","), a.eval_targets, a.data)
    if a.bench:
        return bench(a.bench.split(","), which=a.which)
    if a.speed_all:
        return speed_all(a)
    if a.report and "=" in a.report:
        return report_groups(a.report)
    if a.report:
        return report(a.report.split(","))
    if a.speed:
        r = speed_of(a)
        print(json.dumps(r))
        return r
    return train(a)


if __name__ == "__main__":
    main()

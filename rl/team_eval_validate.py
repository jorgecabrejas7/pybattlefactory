"""Does the team evaluator pick better than the tactician network? (docs/RL_DECISIONS.md §19.3's condition)

    python -m rl.team_eval_validate run --evaluator runs/team_eval_v1/model.pt --out runs/team_eval_v1/validate.pkl
    python -m rl.team_eval_validate analyze runs/team_eval_v1/validate.pkl

The same real-game test as runs/diag_az1 (tsearchB.py / tsearch_diag.py Q3):
- states from alphazero_v1's own play, battler greedy, tactician sampled: rentals = the first decision of runs
  from round 1; swaps = swap screens of rounds 3-5 (runs starting at round 3, 4 or 5), Noland's swap screens
  skipped; 7 + 7 states per process (126 + 126 with 18 processes);
- options scored by pi_eval = argmax Q_eval, Q_eval(option) = Phi of the state after it = E_round + P(complete) *
  v_next (rl.team_eval.phi_options), v_next = V_NEXT below; also by E_round alone; and by the network (log pi);
- real outcome of an option: a clone of the state, the game RNG reseeded with each of 64 paired seeds (the same
  seeds for every option), the option applied, then the alphazero_v1 battler greedy and its tactician greedy
  (later swaps) until the current round ends or the run is lost. wins = battles won until the round ends;
  real Phi = wins + v_next if the round was completed (the same units as Q_eval);
- options played: swaps: all legal ones; rentals: the union of the top 8 (--top) by the network, by Q_eval and by
  E_round, or every legal rental with --top 0 (the "best option" of a state is the best of those played).

V_NEXT: expected wins from the start of round r + 1 under alphazero_v1 greedy play (battler and tactician): the mean
Monte Carlo return at the rental decisions of streak 7r in runs/diag_az1/q1_az_allgreedy.npz (45,000 tactician
decisions; SE 0.12-0.26 for rounds 2-6, 0.40 for round 7).

alphazero_v3's planner (docs §20; `run --version 3`): the same test with the alphazero_v2 agents (its checkpoint:
battler and tactician network, whose per-option inputs come from the W it was trained with, --net-team-eval), the
hint-aware W_v3 (--evaluator) and v_next = the v2 checkpoint's table (table[r + 1]). Criteria, each an argmax over
the options (option_features_v3 with the attendant's hint of the next battle):
    q_plan1   Q_plan, planner depth 1          q_plan2   Q_plan, planner depth 2
    phi_h     Phi with the hint (depth 0)      e_round   E_round with the hint (no v_next)
    python -m rl.team_eval_validate run --version 3 --ckpt runs/alphazero_v2/latest.pt \
        --net-team-eval runs/alphazero_v2/team_eval_0188.pt --evaluator runs/team_eval_v3/model.pt \
        --out runs/team_eval_v3/validate.pkl
"""

import argparse
import math
import multiprocessing as mp
import os
import pickle
import random
import sys
import time

import numpy as np
import torch

CKPT = "runs/alphazero_v1/latest.pt"
V_NEXT = {1: 11.02, 2: 7.97, 3: 7.45, 4: 3.79, 5: 4.19, 6: 5.67}
TOP = 8


def v_next(round_idx):
    return V_NEXT[min(max(int(round_idx), 1), 6)]


def ctx_of(be):
    from rl.search import run_ctx
    return run_ctx(be)


def load_net(path):
    from rl import encode
    from rl.model import FactoryNet
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck["args"]
    encode.set_version(a.get("encode_version", 2))
    net = FactoryNet(a["d_emb"], a["d"], a["layers"], a["heads"], share=a.get("share", "embeddings"),
                     opt_feat=a.get("opt_feat", 0)).eval()
    net.load_state_dict(ck["net"])
    return net, (ck.get("value_norm") or {}).get("battler")


def net_action(net, kind, obs, greedy):
    from rl import encode, ppo
    x = encode.collate([obs], "cpu")
    with torch.no_grad():
        a, _, _ = ppo.act(net, kind, x, greedy=greedy)
    return a[0]


def t_logits(net, kind, obs):
    from rl import encode
    x = encode.collate([obs], "cpu")
    with torch.no_grad():
        if kind == "rental":
            ll, pl, _ = net.rental_joint(x)
            return ll[0].numpy(), pl[0].numpy()
        lg, _ = net.swap(x)
        return lg[0].numpy()


def rollouts(net, beval, backend, kind, options, seeds, max_decisions=300, obs_hook=None):
    """Each (option, seed) played in the real game with the battler greedy and the tactician greedy until the
    current round ends (runs/diag_az1/tsearch_diag.py rollouts). obs_hook: the tactician's per-option inputs
    (alphazero_v2's TeamFeatures). -> wins [n_opt, n_seeds], truncated count."""
    from pybattle.backend import Phase
    from rl import encode
    from rl.envs import decode_action
    streak0 = backend.run_info().win_streak
    target = (streak0 // 7 + 1) * 7
    R = []
    for oi, opt in enumerate(options):
        for si, s in enumerate(seeds):
            be = backend.clone()
            be.game.set_rng(s & 0xFFFFFFFF)
            be.act(decode_action(kind, opt))
            R.append({"be": be, "oi": oi, "si": si, "done": False, "ndec": 0, "trunc": False})
    wins = np.zeros((len(options), len(seeds)))
    ntrunc = 0
    active = R
    while active:
        batch, who = [], []
        for r in active:
            be = r["be"]
            while True:
                st = be.run_info().win_streak
                if be.phase == Phase.RUN_OVER or st >= target:
                    r["done"] = True
                    break
                if be.phase in (Phase.RENTAL, Phase.SWAP):
                    k = "rental" if be.phase == Phase.RENTAL else "swap"
                    view = be.view()
                    obs = encode.rental(view, ctx_of(be)) if k == "rental" else encode.swap(view, ctx_of(be))
                    if obs_hook is not None:
                        obs = obs_hook(be, k, obs, ctx_of(be))
                    be.act(decode_action(k, net_action(net, k, obs, greedy=True)))
                    r["ndec"] = 0
                    continue
                view = be.view()
                if be.phase == Phase.BATTLE and not any(view.usable_moves) and not view.switch_targets \
                        and r["ndec"] < max_decisions:
                    be.act(("move", 0))
                    r["ndec"] += 1
                    continue
                if r["ndec"] >= max_decisions:
                    r["done"] = r["trunc"] = True
                    break
                batch.append(encode.battle(view, ctx_of(be)))
                who.append(r)
                break
        if batch:
            pri, _ = beval({k: np.stack([x[k] for x in batch]) for k in batch[0]})
            for r, x, p in zip(who, batch, pri):
                a = int(np.argmax(np.where(x["mask"], p, -1.0)))
                prev = r["be"].run_info().win_streak
                r["be"].act(decode_action("battle", a))
                r["ndec"] += 1
                if r["be"].run_info().win_streak != prev or r["be"].phase not in (Phase.BATTLE, Phase.FORCED_SWITCH):
                    r["ndec"] = 0
        for r in R:
            if r["done"] and "w" not in r:
                r["w"] = min(r["be"].run_info().win_streak, target) - streak0
                ntrunc += r["trunc"]
                wins[r["oi"], r["si"]] = r["w"]
        active = [r for r in R if not r["done"]]
    return wins, ntrunc


def stream(net, env, rng, want):
    """alphazero_v1 play, battler greedy, tactician sampled; a fresh run after each accepted state."""
    from rl.envs import decode_action
    ev = env.reset()
    while True:
        if ev["truncate"]:
            ev = env.step("reset")
            continue
        kind = ev["kind"]
        if kind != "battle":
            streak = env.backend.run_info().win_streak
            if want(kind, streak, rng):
                yield kind, ev["obs"], streak
                ev = env.reset()
                continue
        a = net_action(net, kind, ev["obs"], greedy=(kind == "battle"))
        ev = env.step(decode_action(kind, a))


def worker(args):
    wid, per_proc, n_seeds, seed, ev_path, ckpt, out_dir, top = args
    torch.set_num_threads(1)
    from rl.alphazero import local_battler_evaluator
    from rl.envs import FactoryEnv
    from rl.tactician_search import option_logp, options_of
    from rl import team_eval as T
    net, bnorm = load_net(ckpt)
    beval = local_battler_evaluator(net, bnorm)
    ev = T.TeamEvaluator.load(ev_path)
    rng = random.Random(seed)
    out = []
    path = os.path.join(out_dir, f"validate_w{wid:02d}.pkl")
    for kind in ("rental", "swap"):
        if kind == "rental":
            env = FactoryEnv(seed, gamma=1.0, beta=0.0, max_decisions=300, start_p0=1.0)
            want = lambda k, s, r: k == "rental" and s == 0                                     # noqa: E731
        else:
            env = FactoryEnv(seed + 7, gamma=1.0, beta=0.0, max_decisions=300, win_streak=14)
            want = lambda k, s, r: (k == "swap" and 14 <= s < 35                                  # noqa: E731
                                    and not env.backend.run_info().noland and r.random() < 0.5)
        n = 0
        for k, obs, streak in stream(net, env, rng, want):
            if kind == "swap":
                env.win_streak = 7 * rng.choice([2, 3, 4])
            be = env.backend
            view, ctx = be.view(), ctx_of(be)
            r_idx = streak // 7 + 1
            opts = options_of(k, obs)
            logp = option_logp(k, t_logits(net, k, obs), opts)
            t0 = time.time()
            feats = T.option_features(ev, view, k, ctx, opts, v_next(r_idx))
            q_phi = T.phi_options(ev, view, k, ctx, opts, v_next(r_idx))
            t_eval = time.time() - t0
            q_e = feats[:, 0].astype(np.float64)
            if k == "swap" or top <= 0:
                sel = list(range(len(opts)))
            else:
                sel = set()
                for s in (logp, q_phi, q_e):
                    sel |= set(np.argsort(-np.asarray(s), kind="stable")[:top].tolist())
                sel = sorted(sel)
            seeds = [rng.getrandbits(32) for _ in range(n_seeds)]
            t1 = time.time()
            w, nt = rollouts(net, beval, be, k, [opts[i] for i in sel], seeds)
            rec = {"kind": k, "streak": streak, "n_options": len(opts), "sel": sel, "logp": logp, "q_phi": q_phi,
                   "q_e": q_e, "feats": feats, "wins": w, "trunc": nt, "v_next": v_next(r_idx),
                   "remaining": 7 - streak % 7, "eval_ms": 1000 * t_eval}
            out.append(rec)
            pickle.dump(out, open(path, "wb"))
            j = {i: n_ for n_, i in enumerate(sel)}
            a_n, a_p = int(np.argmax(logp)), int(np.argmax(q_phi))
            print(f"[w{wid}] {k} streak {streak} opts {len(opts)} played {len(sel)}: net {w[j[a_n]].mean():.2f} "
                  f"eval {w[j[a_p]].mean():.2f} best {w.mean(1).max():.2f} ({time.time() - t1:.0f}s)", flush=True)
            n += 1
            if n >= per_proc:
                break
    return out


def worker3(args):
    """alphazero_v3's validation (see the module docstring): v2 agents, W_v3 criteria."""
    (wid, per_proc, n_seeds, seed, ev_path, ckpt, net_te, out_dir, top, n_opp, n_opp2, beam) = args
    torch.set_num_threads(1)
    from rl.alphazero import local_battler_evaluator
    from rl.envs import FactoryEnv
    from rl.tactician_features import TeamFeatures, v_next_of
    from rl.tactician_search import option_logp, options_of
    from rl import team_eval as T
    net, bnorm = load_net(ckpt)
    beval = local_battler_evaluator(net, bnorm)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    table = list(ck["v_next"]["table"])
    hook = TeamFeatures(net_te, "rl.team_eval", v_next=table, shaping=False)
    ev = T.TeamEvaluator.load(ev_path)
    rng = random.Random(seed)
    out = []
    path = os.path.join(out_dir, f"validate3_w{wid:02d}.pkl")
    for kind in ("rental", "swap"):
        if kind == "rental":
            env = FactoryEnv(seed, gamma=1.0, beta=0.0, max_decisions=300, start_p0=1.0, obs_hook=hook)
            want = lambda k, s, r: k == "rental" and s == 0                                     # noqa: E731
        else:
            env = FactoryEnv(seed + 7, gamma=1.0, beta=0.0, max_decisions=300, win_streak=14, obs_hook=hook)
            want = lambda k, s, r: (k == "swap" and 14 <= s < 35                                  # noqa: E731
                                    and not env.backend.run_info().noland and r.random() < 0.5)
        n = 0
        for k, obs, streak in stream(net, env, rng, want):
            if kind == "swap":
                env.win_streak = 7 * rng.choice([2, 3, 4])
            be = env.backend
            view, ctx = be.view(), ctx_of(be)
            opts = options_of(k, obs)
            logp = option_logp(k, t_logits(net, k, obs), opts)
            hint = (int(view.hint_type), int(view.hint_style))
            vn = v_next_of(table, ctx)
            scores, ms = {}, {}
            for name, depth in (("q_plan1", 1), ("q_plan2", 2)):
                t0 = time.time()
                T._LAST_PLAN.clear()
                f = T.option_features_v3(ev, view, k, ctx, opts, table, hint,
                                         dict(depth=depth, n_opp=n_opp, n_opp2=n_opp2, beam=beam,
                                              seed=rng.getrandbits(32)))
                ms[name] = 1000 * (time.time() - t0)
                scores[name] = T.q_eval_v3(f)
                if depth == 1:
                    scores["phi_h"] = f[:, 6].astype(np.float64) * T.WIN_SCALE
                    scores["e_round"] = f[:, 0].astype(np.float64) * T.WIN_SCALE
                    feats = f
            if k == "swap" or top <= 0:
                sel = list(range(len(opts)))
            else:
                sel = set(np.argsort(-np.asarray(logp), kind="stable")[:top].tolist())
                for sc in scores.values():
                    sel |= set(np.argsort(-np.asarray(sc), kind="stable")[:top].tolist())
                sel = sorted(sel)
            seeds = [rng.getrandbits(32) for _ in range(n_seeds)]
            t1 = time.time()
            w, nt = rollouts(net, beval, be, k, [opts[i] for i in sel], seeds, obs_hook=hook)
            rec = {"kind": k, "streak": streak, "n_options": len(opts), "sel": sel, "logp": logp, "scores": scores,
                   "feats": feats, "wins": w, "trunc": nt, "v_next": vn, "hint": hint,
                   "remaining": 7 - streak % 7, "plan_ms": ms}
            out.append(rec)
            pickle.dump(out, open(path, "wb"))
            j = {i: n_ for n_, i in enumerate(sel)}
            a_n, a_p = int(np.argmax(logp)), int(np.argmax(scores["q_plan1"]))
            print(f"[w{wid}] {k} streak {streak} opts {len(opts)} played {len(sel)}: net {w[j[a_n]].mean():.2f} "
                  f"q_plan1 {w[j[a_p]].mean():.2f} best {w.mean(1).max():.2f} ({time.time() - t1:.0f}s; plan "
                  f"{ms['q_plan1']:.0f} / {ms['q_plan2']:.0f} ms)", flush=True)
            n += 1
            if n >= per_proc:
                break
    return out


# ---- analysis --------------------------------------------------------------------------------------------------------------

def _real(rec, units):
    w = rec["wins"]
    if units == "phi":
        return w + (w >= rec["remaining"]) * rec["v_next"]
    return w


def _spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    if np.std(ra) == 0 or np.std(rb) == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def _criteria(rec):
    """Old records: q_phi / q_e fields; v3 records: a "scores" dict."""
    if "scores" in rec:
        return rec["scores"]
    return {"q_phi": rec["q_phi"], "q_e": rec["q_e"]}


def analyze(recs, out=print):
    rows = {}
    for kind in ("rental", "swap"):
        R = [r for r in recs if r["kind"] == kind]
        if not R:
            continue
        out(f"== {kind}: {len(R)} states, rounds {np.bincount([r['streak'] // 7 + 1 for r in R]).tolist()}, "
            f"options {np.mean([r['n_options'] for r in R]):.1f} (played {np.mean([len(r['sel']) for r in R]):.1f}), "
            f"truncated rollouts {sum(r['trunc'] for r in R)}")
        for units in ("wins", "phi"):
            for crit in _criteria(R[0]):
                gains, claimed, allg, pb = [], [], [], []
                top3, top3_net, top1, top1_net, sp_eval, sp_net, regret_e, regret_n = [], [], [], [], [], [], [], []
                for r in R:
                    j = {i: n for n, i in enumerate(r["sel"])}
                    real = _real(r, units)                                  # [n_sel, seeds]
                    m = real.mean(1)
                    sc = _criteria(r)[crit]
                    a_n, a_e = int(np.argmax(r["logp"])), int(np.argmax(sc))
                    d = real[j[a_e]] - real[j[a_n]]
                    g = float(d.mean())
                    allg.append(g)
                    if a_e != a_n:
                        gains.append(g)
                        claimed.append(float(sc[a_e] - sc[a_n]))
                        pb.append(g > 0)
                    best = int(np.argmax(m))
                    sel = np.array(r["sel"])
                    qe, lp = np.asarray(sc)[sel], np.asarray(r["logp"])[sel]
                    top3.append(best in np.argsort(-qe, kind="stable")[:3])
                    top3_net.append(best in np.argsort(-lp, kind="stable")[:3])
                    top1.append(best == int(np.argmax(qe)))
                    top1_net.append(best == int(np.argmax(lp)))
                    regret_e.append(m.max() - m[j[a_e]])
                    regret_n.append(m.max() - m[j[a_n]])
                    if len(sel) > 2:
                        sp_eval.append(_spearman(qe, m))
                        sp_net.append(_spearman(lp, m))
                gains, allg = np.array(gains), np.array(allg)
                se = gains.std(ddof=1) / math.sqrt(len(gains)) if len(gains) > 1 else float("nan")
                se_all = allg.std(ddof=1) / math.sqrt(len(allg))
                corr = float(np.corrcoef(claimed, gains)[0, 1]) if len(gains) > 2 else float("nan")
                row = {"kind": kind, "units": units, "criterion": crit, "states": len(R), "differ": len(gains),
                       "gain_differ": float(gains.mean()) if len(gains) else 0.0, "se_differ": se,
                       "gain_all": float(allg.mean()), "se_all": float(se_all),
                       "p_better": float(np.mean(pb)) if pb else float("nan"), "corr_claimed_real": corr,
                       "top3_eval": float(np.mean(top3)), "top3_net": float(np.mean(top3_net)),
                       "top1_eval": float(np.mean(top1)), "top1_net": float(np.mean(top1_net)),
                       "spearman_eval": float(np.nanmean(sp_eval)) if sp_eval else float("nan"),
                       "spearman_net": float(np.nanmean(sp_net)) if sp_net else float("nan"),
                       "regret_eval": float(np.mean(regret_e)), "regret_net": float(np.mean(regret_n))}
                rows[(kind, units, crit)] = row
                out(f"  [{units:4s} | pi_eval = argmax {crit:7s}] differs {len(gains)}/{len(R)}: gain "
                    f"{row['gain_differ']:+.3f} +- {se:.3f} (all states {row['gain_all']:+.3f} +- {se_all:.3f}), "
                    f"P(better) {row['p_better']:.2f}, corr(claimed, real) {corr:+.2f} | best option in top-3: eval "
                    f"{row['top3_eval']:.2f} net {row['top3_net']:.2f}; top-1: eval {row['top1_eval']:.2f} net "
                    f"{row['top1_net']:.2f} | Spearman(score, real) eval {row['spearman_eval']:+.2f} net "
                    f"{row['spearman_net']:+.2f} | regret eval {row['regret_eval']:.2f} net {row['regret_net']:.2f}")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("run", "analyze"))
    p.add_argument("path", nargs="?")
    p.add_argument("--evaluator", default="runs/team_eval_v1/model.pt")
    p.add_argument("--ckpt", default=CKPT)
    p.add_argument("--procs", type=int, default=18)
    p.add_argument("--per-proc", type=int, default=7)
    p.add_argument("--seeds", type=int, default=64)
    p.add_argument("--seed", type=int, default=424242)
    p.add_argument("--out", default="runs/team_eval_v1/validate.pkl")
    p.add_argument("--top", type=int, default=TOP, help="rentals: play the union of the top-N options by each "
                   "score (0: every legal rental)")
    p.add_argument("--version", type=int, default=2, help="3: alphazero_v3's planner with the v2 agents")
    p.add_argument("--net-team-eval", default="runs/alphazero_v2/team_eval_0188.pt",
                   help="(--version 3) the W the tactician network was trained with (its per-option inputs)")
    p.add_argument("--n-opp", type=int, default=8)
    p.add_argument("--n-opp2", type=int, default=4)
    p.add_argument("--beam", type=int, default=3)
    a = p.parse_args()
    if a.mode == "analyze":
        analyze(pickle.load(open(a.path or a.out, "rb")))
        return
    t0 = time.time()
    out_dir = os.path.dirname(os.path.abspath(a.out))
    os.makedirs(out_dir, exist_ok=True)
    if a.version == 3:
        fn = worker3
        jobs = [(w, a.per_proc, a.seeds, a.seed + 101 * w, a.evaluator, a.ckpt, a.net_team_eval, out_dir, a.top,
                 a.n_opp, a.n_opp2, a.beam) for w in range(a.procs)]
    else:
        fn = worker
        jobs = [(w, a.per_proc, a.seeds, a.seed + 101 * w, a.evaluator, a.ckpt, out_dir, a.top)
                for w in range(a.procs)]
    with mp.get_context("spawn").Pool(a.procs) as pool:
        res = [r for chunk in pool.map(fn, jobs, chunksize=1) for r in chunk]
    pickle.dump(res, open(a.out, "wb"))
    print(f"saved {len(res)} records to {a.out} ({time.time() - t0:.0f}s)")
    analyze(res)


if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())
    main()

"""Forced-rule experiment: would multi-turn stall moves (Double Team, Toxic) help the trained battler?

    nice -n 10 python -m rl.forced_rules --ckpt runs/ppo_joint_v3/ckpt_X.pt [--runs 300] [--jobs 8]

On top of a trained policy (network battler and tactician, greedy), a rule overrides the battler at voluntary
battle decisions (not forced switches):
    force_evasion  our active has a usable EVASION_UP / EVASION_UP_2 move and its evasion stage < +2 -> use it
    force_toxic    our active has a usable Toxic, the foe's active has no major status, is neither Steel nor
                   Poison and has no Substitute -> use it
    force_both     both (evasion first)
Evaluation is per round, as in rl/eval_rounds.py: for k = 1..6, runs start at the beginning of round k (streak
7(k-1), random feasible rental counter, silver symbol for k >= 4) and stop when round k is completed or lost.
Environments are in-process; every run (env i, run j) is a fresh FactoryEnv with its own fixed seed, so it starts
from the same state in every variant and the first battle of each run is a paired comparison (identical team and
opponent at battle start).
"""

import argparse
import glob
import json
import multiprocessing as mp
import os
import time

import numpy as np
import torch

from . import encode
from .envs import NO_ACTION, FactoryEnv, decode_action
from .gamedata import MOVES
from .policy import Policy

EFFECT_EVASION_UP, EFFECT_EVASION_UP_2, EFFECT_TOXIC = 16, 56, 33
TYPE_POISON, TYPE_STEEL = 3, 8
EVA = 6                                          # index of evasion in ActiveState.stat_stages
VARIANTS = ("baseline", "force_evasion", "force_toxic", "force_both")


def _usable_slot(view, mask, effects):
    me = view.own_party[view.own_active.party_index]
    for k, mv in enumerate(me.moves):
        if mv and MOVES[mv]["effect"] in effects and view.usable_moves[k] and mask[k]:
            return k
    return None


def rule_evasion(view, mask):
    if view.own_active.stat_stages[EVA] >= 2:
        return None
    return _usable_slot(view, mask, (EFFECT_EVASION_UP, EFFECT_EVASION_UP_2))


def rule_toxic(view, mask):
    foe = view.enemy_active
    if view.enemy_party[foe.party_index].status != 0 or foe.substitute or \
            TYPE_POISON in foe.types or TYPE_STEEL in foe.types:
        return None
    return _usable_slot(view, mask, (EFFECT_TOXIC,))


def forced_move(view, mask, variant):
    """Move slot the rule forces at this voluntary battle decision, or None (use the network)."""
    rules = {"force_evasion": (rule_evasion,), "force_toxic": (rule_toxic,),
             "force_both": (rule_evasion, rule_toxic)}.get(variant, ())
    for r in rules:
        k = r(view, mask)
        if k is not None:
            return k
    return None


def play_round(ckpt, k, variant, n_runs, n_envs=50, seed=777, device="cpu"):
    """Runs starting at round k, stopped when round k is completed or lost. Returns one record per run:
    {"env", "run", "complete", "battles": [{"won", "decisions", "fired", "changed"}]} (won None = truncated)."""
    torch.set_num_threads(1)
    device = torch.device(device)
    policy = Policy(ckpt, device)
    base = (seed + 1000 * k) * 100_003
    target = int(np.ceil(n_runs / n_envs))
    runs_done = np.zeros(n_envs, int)

    def fresh(i):
        # a new environment per run: env.reset() keeps state across runs (the backend's BattleObserver survives a
        # reset that happens at a SWAP, i.e. after a completed round), so a run's start would depend on the
        # previous run's play. Run (i, j) gets its own seed and starts identically in every variant.
        e = FactoryEnv(base + i + 7919 * int(runs_done[i]), win_streak=7 * (k - 1))
        return e, e._advance(NO_ACTION)

    envs, events = map(list, zip(*[fresh(i) for i in range(n_envs)]))
    cur = [{"battles": []} for _ in envs]
    bat = [{"decisions": 0, "fired": 0, "changed": 0} for _ in envs]
    out = []
    while min(runs_done) < target:
        acts = [None] * n_envs
        by = {}
        for i, ev in enumerate(events):
            by.setdefault(ev["kind"], []).append(i)
        for kind, idx in by.items():
            a = policy.act(kind, encode.collate([events[i]["obs"] for i in idx], device), greedy=True)
            for j, i in enumerate(idx):
                acts[i] = decode_action(kind, a[j])
        for i, (e, ev) in enumerate(zip(envs, events)):
            b = ev["stats"].get("battle")
            finished = False
            if b is not None:
                rec = dict(bat[i], won=b["won"])
                cur[i]["battles"].append(rec)
                bat[i] = {"decisions": 0, "fired": 0, "changed": 0}
                finished = not b["won"] or sum(x["won"] is True for x in cur[i]["battles"]) == 7
            if ev["truncate"]:
                finished = True
            if finished:
                if runs_done[i] < target:
                    ws = sum(x["won"] is True for x in cur[i]["battles"])
                    out.append({"env": i, "run": int(runs_done[i]), "complete": ws == 7, "battles": cur[i]["battles"]})
                runs_done[i] += 1
                cur[i] = {"battles": []}
                acts[i] = "fresh"
                continue
            if ev["kind"] == "battle":
                view = e.backend.view()
                if not view.forced_switch:
                    bat[i]["decisions"] += 1
                    # the baseline only counts where force_both would fire (and the network disagrees)
                    f = forced_move(view, ev["obs"]["mask"], "force_both" if variant == "baseline" else variant)
                    if f is not None:
                        bat[i]["fired"] += 1
                        if acts[i] != ("move", f):
                            bat[i]["changed"] += 1
                            if variant != "baseline":
                                acts[i] = ("move", f)
        for i, x in enumerate(acts):
            if isinstance(x, str) and x == "fresh":
                envs[i], events[i] = fresh(i)
            else:
                events[i] = envs[i].step(x)
    return out


def _job(args):
    ckpt, k, variant, n_runs, n_envs, seed = args
    t = time.time()
    r = play_round(ckpt, k, variant, n_runs, n_envs, seed)
    print(f"round {k} {variant}: {len(r)} runs, {time.time() - t:.0f}s", flush=True)
    return (k, variant), r


def _boot_ci(f, arrays, n=2000, seed=0):
    """Percentile bootstrap 95% CI of f over runs resampled jointly (arrays share their first axis)."""
    rng = np.random.default_rng(seed)
    m = len(arrays[0])
    if m == 0:
        return [None, None]
    vals = []
    for _ in range(n):
        ix = rng.integers(0, m, m)
        v = f(*[a[ix] for a in arrays])
        if v == v:
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else [None, None]


def _rate(w, lost):
    return w.sum() / max(w.sum() + lost.sum(), 1)


def summarize(runs_by):
    """runs_by[(k, variant)] -> list of run records. Paired by (env, run)."""
    out = {}
    for k in sorted({k for k, _ in runs_by}):
        base = {(r["env"], r["run"]): r for r in runs_by[(k, "baseline")]}
        keys = sorted(base)
        rk = {}
        for v in VARIANTS:
            rv = {(r["env"], r["run"]): r for r in runs_by[(k, v)]}
            assert sorted(rv) == keys
            comp = np.array([rv[x]["complete"] for x in keys], float)
            wins = np.array([sum(b["won"] is True for b in rv[x]["battles"]) for x in keys], float)
            lost = 1 - comp
            bs = [b for x in keys for b in rv[x]["battles"]]
            dec = sum(b["decisions"] for b in bs)
            e = {"n_runs": len(keys), "complete": float(comp.mean()), "battle": float(_rate(wins, lost)),
                 "battles": len(bs), "decisions": dec,
                 "fired_per_decision": sum(b["fired"] for b in bs) / max(dec, 1),
                 "changed_per_decision": sum(b["changed"] for b in bs) / max(dec, 1),
                 "battles_fired": float(np.mean([b["fired"] > 0 for b in bs])),
                 "battles_changed": float(np.mean([b["changed"] > 0 for b in bs]))}
            if v != "baseline":
                c0 = np.array([base[x]["complete"] for x in keys], float)
                w0 = np.array([sum(b["won"] is True for b in base[x]["battles"]) for x in keys], float)
                e["diff_complete"] = float(comp.mean() - c0.mean())
                e["diff_complete_ci95"] = _boot_ci(lambda a, b: a.mean() - b.mean(), [comp, c0])
                e["diff_battle"] = float(_rate(wins, lost) - _rate(w0, 1 - c0))
                e["diff_battle_ci95"] = _boot_ci(lambda a, b, c, d: _rate(a, 1 - b) - _rate(c, 1 - d),
                                                 [wins, comp, w0, c0])
                # paired, battle 1 of each run: identical starting state in both
                b1v = [rv[x]["battles"][0] for x in keys]
                b1b = [base[x]["battles"][0] for x in keys]
                ch = np.array([b["changed"] > 0 for b in b1v])
                wv = np.array([b["won"] is True for b in b1v], float)[ch]
                wb = np.array([b["won"] is True for b in b1b], float)[ch]
                # sanity: where the rule never changed an action, the battle must replay identically
                same = np.mean([(a["won"] is True) == (b["won"] is True)
                                for a, b, c in zip(b1v, b1b, ch) if not c]) if (~ch).any() else None
                e["paired_battle1"] = {
                    "n": int(ch.sum()), "win_variant": float(wv.mean()) if ch.any() else None,
                    "win_baseline": float(wb.mean()) if ch.any() else None,
                    "diff": float(wv.mean() - wb.mean()) if ch.any() else None,
                    "diff_ci95": _boot_ci(lambda a, b: a.mean() - b.mean(), [wv, wb]),
                    "variant_better": int(((wv == 1) & (wb == 0)).sum()),
                    "baseline_better": int(((wv == 0) & (wb == 1)).sum()),
                    "unchanged_battles_identical": same}
                # all battles where the rule changed an action, matched with the baseline's battle at the same
                # (run, position) when the baseline reached it (starting states only approximately equal after 1)
                pv, pb = [], []
                for x in keys:
                    for j, b in enumerate(rv[x]["battles"]):
                        if b["changed"] > 0 and j < len(base[x]["battles"]):
                            pv.append(b["won"] is True)
                            pb.append(base[x]["battles"][j]["won"] is True)
                pv, pb = np.array(pv, float), np.array(pb, float)
                e["paired_all_positions"] = {
                    "n": len(pv), "win_variant": float(pv.mean()) if len(pv) else None,
                    "win_baseline": float(pb.mean()) if len(pb) else None,
                    "diff": float(pv.mean() - pb.mean()) if len(pv) else None,
                    "diff_ci95": _boot_ci(lambda a, b: a.mean() - b.mean(), [pv, pb])}
            rk[v] = e
        out[k] = rk
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None, help="default: newest runs/ppo_joint_v3/ckpt_*.pt")
    p.add_argument("--runs", type=int, default=300)
    p.add_argument("--envs", type=int, default=50)
    p.add_argument("--rounds", default="1,2,3,4,5,6")
    p.add_argument("--variants", default=",".join(VARIANTS))
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    ckpt = args.ckpt or sorted(glob.glob("runs/ppo_joint_v3/ckpt_*.pt"))[-1]
    rounds = [int(x) for x in args.rounds.split(",")]
    variants = args.variants.split(",")
    jobs = [(ckpt, k, v, args.runs, args.envs, args.seed) for k in rounds for v in variants]
    with mp.get_context("fork").Pool(args.jobs) as pool:
        runs_by = dict(pool.imap_unordered(_job, jobs))
    res = {"ckpt": ckpt, "runs_per_round": args.runs, "seed": args.seed, "rounds": summarize(runs_by),
           "raw": {f"{k}/{v}": r for (k, v), r in runs_by.items()}}
    path = args.out or os.path.join(os.path.dirname(ckpt), "analysis_forced_rules.json")
    json.dump(res, open(path, "w"), indent=1, default=float)
    for k, rk in res["rounds"].items():
        for v, e in rk.items():
            s = f"R{k} {v:14s} complete {e['complete']:.3f} battle {e['battle']:.3f} " \
                f"fired/dec {e['fired_per_decision']:.3f} battles_changed {e['battles_changed']:.3f}"
            if v != "baseline":
                pb = e["paired_battle1"]
                s += f" | dC {e['diff_complete']:+.3f} {e['diff_complete_ci95']} | b1 n={pb['n']} diff {pb['diff']}"
            print(s)


if __name__ == "__main__":
    main()

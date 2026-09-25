"""Where runs die inside a round: win rate by battle position, and what the lost battles looked like.

    python -m rl.analyze_rounds --ckpt runs/NAME/ckpt_X.pt --rounds 3,4,6 --runs 600

Runs start at the beginning of round k (random feasible rental counter) and stop when the round is completed or
lost. For every battle: its position in the round (1-7), whether it was against the Factory Head (Noland), the
outcome; for losses: the opponent's species, our team, and how many Pokemon each side still had.
"""

import argparse
import collections
import json
import os

import numpy as np
import torch

from . import encode
from .analyze import name
from .envs import NO_ACTION, FactoryEnv, decode_action
from .policy import Policy


def analyze_round(policy, device, k, n_runs, n_envs=64, seed=31337):
    envs = [FactoryEnv(seed + 1000 * k + i, win_streak=7 * (k - 1)) for i in range(n_envs)]
    events = [e._advance(NO_ACTION) for e in envs]
    wins = np.zeros(n_envs, int)
    snap = [None] * n_envs
    by_pos = collections.defaultdict(lambda: [0, 0])          # position -> [won, played]
    brain = collections.defaultdict(lambda: [0, 0])
    losses, runs = [], 0
    while runs < n_runs:
        acts = [None] * n_envs
        by = collections.defaultdict(list)
        for i, ev in enumerate(events):
            by[ev["kind"]].append(i)
        for kind, idx in by.items():
            a = policy.act(kind, encode.collate([events[i]["obs"] for i in idx], device), greedy=True)
            for j, i in enumerate(idx):
                acts[i] = decode_action(kind, a[j])
        for i, (e, ev) in enumerate(zip(envs, events)):
            b = ev["stats"].get("battle")
            finished = ev["truncate"]
            if b is not None and b["won"] is not None and snap[i] is not None:
                s = snap[i]
                by_pos[s["pos"]][0] += b["won"]; by_pos[s["pos"]][1] += 1
                brain[s["brain"]][0] += b["won"]; brain[s["brain"]][1] += 1
                if b["won"]:
                    wins[i] += 1
                    finished = finished or wins[i] == 7
                else:
                    losses.append(dict(s, decisions=b["decisions"]))
                    finished = True
                snap[i] = None
            if finished:
                runs += 1
                wins[i] = 0
                snap[i] = None
                acts[i] = "reset"
                continue
            if ev["kind"] == "battle":
                be = e.backend
                v = be.view()
                snap[i] = {"pos": be.run_info().battle_in_challenge + 1,
                           "brain": bool(be.game.factory_info.brain_status),
                           "foe": [m.species for m in v.enemy_party if m.seen],
                           "foe_left": sum(1 for m in v.enemy_party if not m.fainted),
                           "own": [m.species for m in v.own_party],
                           "own_left": sum(1 for m in v.own_party if m.hp > 0)}
        events = [e.step(a) for e, a in zip(envs, acts)]
    return by_pos, brain, losses


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--rounds", default="3,4,6")
    p.add_argument("--runs", type=int, default=600)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = Policy(args.ckpt, device)
    out = {}
    for k in map(int, args.rounds.split(",")):
        by_pos, brain, losses = analyze_round(policy, device, k, args.runs)
        noland = [l for l in losses if l["brain"]]
        rec = {
            "win_rate_by_position": {p: round(w / n, 3) for p, (w, n) in sorted(by_pos.items())},
            "played_by_position": {p: n for p, (w, n) in sorted(by_pos.items())},
            "losses_by_position": dict(sorted(collections.Counter(l["pos"] for l in losses).items())),
            "noland_win_rate": round(brain[True][0] / brain[True][1], 3) if brain[True][1] else None,
            "other_win_rate": round(brain[False][0] / brain[False][1], 3) if brain[False][1] else None,
            "noland_battles": brain[True][1],
        }
        if noland:
            rec["noland_species_seen_in_losses"] = [(name("species", s), c) for s, c in
                                                    collections.Counter(x for l in noland for x in l["foe"]).most_common(10)]
            rec["noland_losses_foe_left_mean"] = float(np.mean([l["foe_left"] for l in noland]))
            rec["noland_losses_decisions_mean"] = float(np.mean([l["decisions"] for l in noland]))
            rec["our_team_in_noland_losses"] = [(name("species", s), c) for s, c in
                                                collections.Counter(x for l in noland for x in l["own"]).most_common(10)]
        other = [l for l in losses if not l["brain"]]
        if other:
            rec["other_losses_foe_left_mean"] = float(np.mean([l["foe_left"] for l in other]))
            rec["foe_species_in_other_losses"] = [(name("species", s), c) for s, c in
                                                  collections.Counter(x for l in other for x in l["foe"]).most_common(10)]
        out[k] = rec
        print(f"round {k}:", json.dumps(rec, ensure_ascii=False), flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(args.ckpt), "analysis_rounds.json"), "w"), indent=1,
              ensure_ascii=False)


if __name__ == "__main__":
    main()

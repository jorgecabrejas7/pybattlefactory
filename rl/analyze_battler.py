"""Where the battler departs from the max-damage baseline, and what each departure is worth.

    python -m rl.analyze_battler --ckpt runs/NAME/latest.pt

1. Categorizes every voluntary decision where the policy does not pick the max-damage move.
2. Ablations on the same run seeds: the policy as is; its moves replaced by the max-damage move (switches kept);
   its voluntary switches replaced by its own best move (moves kept).
"""

import argparse
import collections
import json
import os

import numpy as np
import torch

from pybattle.view import NAMES
from . import encode
from .analyze import name
from .baselines import move_score
from .envs import NO_ACTION, FactoryEnv, decode_action
from .gamedata import MOVES, effectiveness
from .policy import Policy
from .ppo import act

ABSORB = {name("abilities", int(k)) for k, v in NAMES["abilities"].items()
          if v in ("ABILITY_LEVITATE", "ABILITY_WATER_ABSORB", "ABILITY_VOLT_ABSORB", "ABILITY_FLASH_FIRE",
                   "ABILITY_WONDER_GUARD")}


def maxdmg_choice(view):
    me = view.own_party[view.own_active.party_index]
    usable = [k for k, ok in enumerate(view.usable_moves) if ok]
    sc = {k: move_score(me.moves[k], me.types, view.enemy_active.types) for k in usable}
    return sc, (max(sc, key=sc.get) if sc else None)


def categorize(view, d, rec):
    me = view.own_party[view.own_active.party_index]
    sc, best = maxdmg_choice(view)
    foe = view.enemy_party[view.enemy_active.party_index]
    foe_hp = max(foe.hp_pixels, 0) / 48
    rec["decisions"] += 1
    f = {"own_hp": me.hp / me.max_hp,
         "best_eff": max((effectiveness(MOVES[me.moves[k]]["type"], view.enemy_active.types)
                          for k in sc if MOVES[me.moves[k]]["power"] > 1), default=0),
         "foe_revealed_threat": max((effectiveness(MOVES[m]["type"], me.types) for m in foe.revealed_moves
                                     if MOVES[m]["power"] > 1), default=1),
         "foe_boosted": sum(max(x, 0) for x in view.enemy_active.stat_stages[:5]) > 0,
         "own_status": me.status != 0,
         "own_dropped": sum(min(x, 0) for x in view.own_active.stat_stages[:5]) < 0}
    for k, v in f.items():
        rec["all_ctx"][k].append(v)
    if d[0] == "switch":
        rec["switch"] += 1
        for k, v in f.items():
            rec["switch_ctx"][k].append(v)
        return
    mv = me.moves[d[1]]
    if best is None or sc[d[1]] == sc[best]:
        rec["agree"] += 1
        return
    m, b = MOVES[mv], MOVES[me.moves[best]]
    abil = {name("abilities", a) for a in foe.possible_abilities}
    if m["power"] == 0:
        cat = "status: " + name("effects", m["effect"])
    elif m["power"] == 1:
        cat = "fixed/variable damage (" + name("moves", mv) + ")"
    elif m["priority"] > 0:
        cat = "priority move, foe HP<=25%" if foe_hp <= 0.25 else "priority move"
    elif abil & ABSORB and b["type"] in (4, 11, 13, 10) and effectiveness(b["type"], view.enemy_active.types) > 0:
        cat = "avoids a possible immunity ability (" + "/".join(sorted(abil & ABSORB)) + ")"
    elif m["accuracy"] > b["accuracy"] and b["accuracy"]:
        cat = "more accurate, weaker"
    elif foe_hp <= 0.25:
        cat = "weaker move, foe HP<=25%"
    elif m["secondary_chance"] > 0:
        cat = "weaker move with secondary effect (" + name("effects", m["effect"]) + ")"
    else:
        cat = "other weaker move"
    rec["cats"][cat] += 1
    rec["cat_moves"][cat][name("moves", mv)] += 1


def play(net, device, mode, n_envs=64, runs_per_env=6, seed=4242, collect=None):
    envs = [FactoryEnv(seed + i) for i in range(n_envs)]
    evs = [e._advance(NO_ACTION) for e in envs]
    res = [[] for _ in envs]
    while min(len(r) for r in res) < runs_per_env:
        by = collections.defaultdict(list)
        for i, ev in enumerate(evs):
            by[ev["kind"]].append(i)
        acts = [None] * n_envs
        for k, idx in by.items():
            aa, _, _ = act(net, k, encode.collate([evs[i]["obs"] for i in idx], device), greedy=True)
            for j, i in enumerate(idx):
                acts[i] = aa[j]
        for i, (e, ev) in enumerate(zip(envs, evs)):
            if "run" in ev["stats"]:
                res[i].append(ev["stats"]["run"]["streak"])
            if ev["truncate"]:
                res[i].append(ev["stats"]["wins"])
                acts[i] = "reset"
                continue
            d = decode_action(ev["kind"], acts[i])
            if ev["kind"] == "battle":
                view = e.backend.view()
                if not view.forced_switch:
                    if collect is not None:
                        categorize(view, d, collect)
                    if mode == "maxdamage_moves" and d[0] == "move":
                        _, best = maxdmg_choice(view)
                        d = ("move", best) if best is not None else d
                    elif mode == "no_switch" and d[0] == "switch":
                        logits = ev["obs"]["mask"].copy()
                        with torch.no_grad():
                            lg, _ = net.battler(encode.collate([ev["obs"]], device))
                        lg = lg[0, :4].cpu().numpy()
                        lg[~ev["obs"]["mask"][:4]] = -1e9
                        d = ("move", int(lg.argmax())) if ev["obs"]["mask"][:4].any() else d
            acts[i] = d
        evs = [e.step(x) for e, x in zip(envs, acts)]
    s = np.array([x for r in res for x in r[:runs_per_env]], float)
    boot = [np.random.default_rng(k).choice(s, len(s)).mean() for k in range(1000)]
    return {"streak_mean": s.mean(), "ci95": [np.percentile(boot, 2.5), np.percentile(boot, 97.5)], "n": len(s)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = Policy(args.ckpt, device)
    net = policy.net
    a = torch.load(args.ckpt, map_location="cpu")["args"]
    rec = {"decisions": 0, "agree": 0, "switch": 0, "cats": collections.Counter(),
           "cat_moves": collections.defaultdict(collections.Counter),
           "switch_ctx": collections.defaultdict(list), "all_ctx": collections.defaultdict(list)}
    out = {"policy": play(net, device, "policy", collect=rec),
           "policy_moves_replaced_by_maxdamage": play(net, device, "maxdamage_moves"),
           "policy_without_voluntary_switches": play(net, device, "no_switch")}
    n = rec["decisions"]
    out["decisions"] = n
    out["share"] = {"agree_maxdamage": rec["agree"] / n, "switch": rec["switch"] / n,
                    "other_move": 1 - (rec["agree"] + rec["switch"]) / n}
    out["deviating_moves"] = [(c, round(k / n, 4), rec["cat_moves"][c].most_common(4)) for c, k in rec["cats"].most_common(25)]
    ctx = rec["switch_ctx"]
    out["switch_context"] = {k: {"when_switching": float(np.mean(v)), "all_decisions": float(np.mean(rec["all_ctx"][k]))}
                             for k, v in ctx.items()}
    json.dump(out, open(os.path.join(os.path.dirname(args.ckpt), "analysis_battler.json"), "w"), indent=1,
              default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()

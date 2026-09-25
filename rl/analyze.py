"""What a trained policy does: which Pokemon it rents and swaps, and how it battles.

    python -m rl.analyze --ckpt runs/NAME/latest.pt --runs 400 [--sampled]

Plays full runs with the policy (greedy by default) and records every decision with the views behind it.
Writes <ckpt dir>/analysis.json and prints a summary.
"""

import argparse
import collections
import json
import os

import numpy as np
import torch

from pybattle.backend import Phase
from pybattle.view import NAMES
from . import encode
from .baselines import move_score
from .envs import NO_ACTION, FactoryEnv, decode_action
from .gamedata import MOVES, SPECIES
from .policy import Policy
from .ppo import act


def name(table, i):
    n = NAMES[table].get(str(i), str(i))
    for p in ("SPECIES_", "MOVE_", "ITEM_", "EFFECT_", "ABILITY_", "TYPE_"):
        n = n.replace(p, "")
    return n.title().replace("_", " ")


def bst(species):
    return sum(SPECIES[species]["base"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--runs", type=int, default=400)
    p.add_argument("--envs", type=int, default=64)
    p.add_argument("--sampled", action="store_true")
    p.add_argument("--seed", type=int, default=777)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = Policy(args.ckpt, device)
    net = policy.net
    a = torch.load(args.ckpt, map_location="cpu")["args"]
    greedy = not args.sampled

    envs = [FactoryEnv(args.seed * 1000 + i, max_decisions=a["max_decisions"]) for i in range(args.envs)]
    events = [e._advance(NO_ACTION) for e in envs]

    rent_offered, rent_chosen, rent_lead = collections.Counter(), collections.Counter(), collections.Counter()
    feat_chosen, feat_offered = collections.defaultdict(list), collections.defaultdict(list)
    swaps = []                      # per swap decision
    moves_used, effects_used = collections.Counter(), collections.Counter()
    agree_maxdmg = [0, 0]
    switch_decisions = [0, 0]       # voluntary switches, voluntary decisions
    turn1_effects = collections.Counter()
    streaks, runs = [], 0
    team_at_loss = collections.Counter()
    items_chosen = collections.Counter()

    while runs < args.runs:
        by_kind = collections.defaultdict(list)
        for i, ev in enumerate(events):
            by_kind[ev["kind"]].append(i)
        actions = [None] * len(envs)
        for kind, idx in by_kind.items():
            acts = policy.act(kind, encode.collate([events[i]["obs"] for i in idx], device), greedy=greedy)
            for j, i in enumerate(idx):
                actions[i] = acts[j]
        for i, (e, ev) in enumerate(zip(envs, events)):
            be = e.backend
            if "run" in ev["stats"]:
                streaks.append(ev["stats"]["run"]["streak"])
                runs += 1
            if ev["truncate"]:
                actions[i] = "reset"
                continue
            view = be.view()
            kind, a_ = ev["kind"], actions[i]
            if kind == "rental":
                lead, pair = int(a_[0]), int(a_[1])
                picks = [lead, *encode.PAIRS[pair]]
                for k, m in enumerate(view.candidates):
                    rent_offered[m.species] += 1
                    chosen = k in picks
                    f = {"bst": bst(m.species), "speed": m.stats[2], "atk": max(m.stats[0], m.stats[3]),
                         "bulk": m.max_hp * (m.stats[1] + m.stats[4]) / 2,
                         "status_moves": sum(MOVES[mv]["power"] == 0 for mv in m.moves if mv)}
                    for key, val in f.items():
                        feat_offered[key].append(val)
                        if chosen:
                            feat_chosen[key].append(val)
                    if chosen:
                        rent_chosen[m.species] += 1
                        items_chosen[m.item] += 1
                    if k == lead:
                        rent_lead[m.species] += 1
            elif kind == "swap":
                info = be.run_info()
                d = decode_action("swap", a_)
                rec = {"battle": info.battle_in_challenge, "challenge": info.challenge_num,
                       "rents": e._ctx()["rents"], "swap": d is not None,
                       "legal": int(ev["obs"]["mask"].sum()) - 1}
                if d is not None:
                    out_m, in_m = view.own_party[d[0]], view.enemy_party[d[1]]
                    rec.update(slot=d[0], out=out_m.species, inn=in_m.species, bst_out=bst(out_m.species),
                               bst_in=bst(in_m.species))
                rec["team_bst"] = [bst(m.species) for m in view.own_party]
                rec["foe_bst"] = [bst(m.species) for m in view.enemy_party]
                swaps.append(rec)
            else:
                d = decode_action("battle", a_)
                if not view.forced_switch:
                    switch_decisions[1] += 1
                    me = view.own_party[view.own_active.party_index]
                    if d[0] == "move":
                        mv = me.moves[d[1]]
                        moves_used[mv] += 1
                        effects_used[MOVES[mv]["effect"]] += 1
                        if view.last_turn.turn == -1:
                            turn1_effects[MOVES[mv]["effect"]] += 1
                        usable = [k for k, ok in enumerate(view.usable_moves) if ok]
                        if usable:
                            sc = {k: move_score(me.moves[k], me.types, view.enemy_active.types) for k in usable}
                            agree_maxdmg[1] += 1
                            agree_maxdmg[0] += sc[d[1]] == max(sc.values())
                    else:
                        switch_decisions[0] += 1
            actions[i] = decode_action(kind, a_)
        events = [e.step(x) for e, x in zip(envs, actions)]

    # ---- summary ------------------------------------------------------------------------------------------
    out = {"runs": runs, "greedy": greedy, "streak_mean": float(np.mean(streaks)), "streak_median": float(np.median(streaks))}
    sel = sorted(((rent_chosen[s] / rent_offered[s], rent_offered[s], s) for s in rent_offered if rent_offered[s] >= 15),
                 reverse=True)
    out["rental_top_by_rate"] = [(name("species", s), round(r, 2), n) for r, n, s in sel[:20]]
    out["rental_bottom_by_rate"] = [(name("species", s), round(r, 2), n) for r, n, s in sel[-15:]]
    out["rental_most_chosen"] = [(name("species", s), c) for s, c in rent_chosen.most_common(20)]
    out["lead_most"] = [(name("species", s), c, round(c / rent_chosen[s], 2)) for s, c in rent_lead.most_common(15)]
    out["items_chosen"] = [(name("items", s), c) for s, c in items_chosen.most_common(12)]
    out["rental_features"] = {k: {"chosen": float(np.mean(feat_chosen[k])), "offered": float(np.mean(feat_offered[k]))}
                              for k in feat_offered}
    sw = swaps
    out["swap_rate"] = float(np.mean([s["swap"] for s in sw]))
    out["swap_rate_by_battle"] = {b: round(float(np.mean([s["swap"] for s in sw if s["battle"] == b])), 3)
                                  for b in sorted({s["battle"] for s in sw})}
    out["swap_n_by_battle"] = {b: sum(s["battle"] == b for s in sw) for b in sorted({s["battle"] for s in sw})}
    out["swap_rate_by_challenge"] = {c: round(float(np.mean([s["swap"] for s in sw if s["challenge"] == c])), 3)
                                     for c in sorted({s["challenge"] for s in sw})}
    done = [s for s in sw if s["swap"]]
    out["swap_slot_given"] = dict(collections.Counter(s["slot"] for s in done))
    out["swap_bst_in_minus_out"] = float(np.mean([s["bst_in"] - s["bst_out"] for s in done])) if done else 0
    out["swap_frac_bst_gain"] = float(np.mean([s["bst_in"] > s["bst_out"] for s in done])) if done else 0
    kept = [s for s in sw if not s["swap"]]
    out["keep_when_max_foe_bst_gain"] = float(np.mean([max(s["foe_bst"]) - min(s["team_bst"]) for s in kept])) if kept else 0
    out["swap_when_max_foe_bst_gain"] = float(np.mean([max(s["foe_bst"]) - min(s["team_bst"]) for s in done])) if done else 0
    out["swapped_in_most"] = [(name("species", s), c) for s, c in collections.Counter(s["inn"] for s in done).most_common(12)]
    out["swapped_out_most"] = [(name("species", s), c) for s, c in collections.Counter(s["out"] for s in done).most_common(12)]
    tot = sum(moves_used.values())
    out["battler_switch_rate"] = switch_decisions[0] / max(switch_decisions[1], 1)
    out["battler_agrees_with_maxdamage"] = agree_maxdmg[0] / max(agree_maxdmg[1], 1)
    out["battler_status_move_share"] = sum(c for m, c in moves_used.items() if MOVES[m]["power"] == 0) / max(tot, 1)
    out["moves_top"] = [(name("moves", m), round(c / tot, 3)) for m, c in moves_used.most_common(20)]
    out["effects_top"] = [(name("effects", f), round(c / tot, 3)) for f, c in effects_used.most_common(15)]
    t1 = sum(turn1_effects.values())
    out["turn1_effects_top"] = [(name("effects", f), round(c / t1, 3)) for f, c in turn1_effects.most_common(10)]
    path = os.path.join(os.path.dirname(args.ckpt), "analysis_greedy.json" if greedy else "analysis_sampled.json")
    json.dump(out, open(path, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

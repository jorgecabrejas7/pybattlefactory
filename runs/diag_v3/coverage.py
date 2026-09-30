"""Type-coverage study of a checkpoint's tactician: chosen team vs all options (rentals: the 20 trios; swaps: keep + 9)."""
import collections, itertools, json, sys
import numpy as np, torch
from rl import encode
from rl.envs import NO_ACTION, FactoryEnv, decode_action
from rl.gamedata import SPECIES, MOVES, TYPE_EFFECTIVENESS, TYPE_MYSTERY, N_TYPES
from rl.policy import Policy
from rl.eval_rounds import features_hook

TYPES = [t for t in range(N_TYPES) if t != TYPE_MYSTERY]
def types(sp): return sorted(set(SPECIES[sp]["types"]))
def eff(atk, sp):
    m = 1.0
    for t in types(sp): m *= float(TYPE_EFFECTIVENESS[atk, t])
    return m
def metrics(team):   # team: list of (species, moves)
    ts = [types(sp) for sp, _ in team]
    dup = int(any(set(a) & set(b) for a, b in itertools.combinations(ts, 2)))
    shared_weak = sum(sum(eff(a, sp) > 1 for sp, _ in team) >= 2 for a in TYPES)
    no_resist = sum(all(eff(a, sp) >= 1 for sp, _ in team) for a in TYPES)   # attacking types nobody resists
    mtypes = {MOVES[m]["type"] for _, mv in team for m in mv if m and MOVES[m]["power"] > 1}
    off = sum(any(float(TYPE_EFFECTIVENESS[mt, d]) > 1 for mt in mtypes) for d in TYPES)
    return {"dup_type": dup, "shared_weak": shared_weak, "no_resist": no_resist, "off_cover": off}

def main(ckpt, runs=200, envs=6, seed=4242):
    dev = torch.device("cuda"); pol = Policy(ckpt, dev)
    class A: features = "auto"; team_eval = None
    hook = features_hook(ckpt, A)
    E = [FactoryEnv(seed * 1000 + i, obs_hook=hook) for i in range(envs)]
    evs = [e._advance(NO_ACTION) for e in E]
    rec = {"rental": [], "swap": []}; done = 0
    while done < runs:
        by = collections.defaultdict(list)
        for i, ev in enumerate(evs): by[ev["kind"]].append(i)
        acts = [None] * len(E)
        for k, idx in by.items():
            a = pol.act(k, encode.collate([evs[i]["obs"] for i in idx], dev), greedy=True)
            for j, i in enumerate(idx): acts[i] = a[j]
        for i, (e, ev) in enumerate(zip(E, evs)):
            if "run" in ev["stats"]: done += 1
            if ev["truncate"]: acts[i] = "reset"; continue
            k = ev["kind"]; v = e.backend.view()
            if k == "rental":
                cand = [(m.species, list(m.moves)) for m in v.candidates]
                lead, pair = int(acts[i][0]), int(acts[i][1]); pick = sorted([lead, *encode.PAIRS[pair]])
                opts = [metrics([cand[x] for x in c]) for c in itertools.combinations(range(6), 3)]
                ch = metrics([cand[x] for x in pick])
                rec["rental"].append({"chosen": ch, "opts": opts, "round": e.backend.run_info().challenge_num + 1})
            elif k == "swap":
                own = [(m.species, list(m.moves)) for m in v.own_party[:3]]
                foe = [(m.species, [x for x in m.revealed_moves if x]) for m in v.enemy_party[:3]]
                # moves of the defeated foes: only what was seen (player view)
                opts = [metrics(own)]
                for s in range(3):
                    for f in range(3):
                        t = list(own); t[s] = foe[f]; opts.append(metrics(t))
                d = decode_action("swap", acts[i])
                ch = opts[0] if d is None else opts[1 + 3 * d[0] + d[1]]
                rec["swap"].append({"chosen": ch, "opts": opts, "swap": d is not None,
                                    "round": e.backend.run_info().challenge_num + 1})
        evs = [e.step(a if isinstance(a, str) else decode_action(ev["kind"], a)) for e, ev, a in zip(E, evs, acts)]
    out = {}
    for kind, R in rec.items():
        s = {}
        for key in ["dup_type", "shared_weak", "no_resist", "off_cover"]:
            ch = np.array([r["chosen"][key] for r in R]); base = np.array([np.mean([o[key] for o in r["opts"]]) for r in R])
            best_low = key != "off_cover"
            pct = np.mean([np.mean([(o[key] > r["chosen"][key]) if best_low else (o[key] < r["chosen"][key]) for o in r["opts"]]) +
                           0.5 * np.mean([o[key] == r["chosen"][key] for o in r["opts"]]) for r in R])
            s[key] = {"chosen": round(float(ch.mean()), 3), "options_mean": round(float(base.mean()), 3), "percentile_better_than": round(float(pct), 3)}
        s["n"] = len(R)
        out[kind] = s
    print(json.dumps(out, indent=1)); json.dump({"summary": out}, open("runs/diag_v3/coverage.json", "w"), indent=1)

if __name__ == "__main__":
    main(sys.argv[1])

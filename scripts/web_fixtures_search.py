#!/usr/bin/env python3
"""Golden opponent determinizations for the browser port's parity tests (battle-arcade games/emerald-factory/tests/
determinize.test.ts against src/ai/determinize.ts).

Plays SimBackend runs (random legal actions) and, at battle decisions (BATTLE / FORCED_SWITCH), records what the
legal search's "experience" sampler sees and draws (rl/search.py SearchBattler._roots with opponent_prior=
"experience", the model of rl/real_agents.py V3Player):
    view, info              the web JSON (pybattle/web_schema.py)
    challenge, noland, hint the sampler's inputs (info.challenge_num, DET.battle_hint(view, info.noland))
    hidden                  DET.hidden_counters(view) (Gen3Game.determinize's 14 ints)
    post                    deterministic intermediates of the compiled model (rl/opponent_model.py _Compiled):
                              bucket, species_probs(b, seen species),
                              per slot: seen -> config_posterior with the draw's starting used items (all revealed
                              items: _draw_team) and with the other slots' revealed items (_hint_team's slot_table);
                              unseen -> config_posterior of the 5 likeliest species (used: the revealed items),
                              joint_w: _hint_team's joint table weights (seen species zeroed), or null
    samples                 N determinizations (model.pt as is: use_hint False, the hint is ignored), each 3 slots
                            [species (KEEP -1), m0, m1, m2, m3, item, nature, ability_bit, hp_fraction]
    samples_hint            the same with use_hint True (the _hint_team path; only when there is a hint) and
    hint_fallback           how many of them fell back (no team with the hint in max_hint_tries)

    web_fixtures_search.py [--decisions 40] [--n 1000] [--out dist/fixtures/determinize.json]
"""

import argparse
import json
import os
import random
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from pybattle.backend import Phase, SimBackend  # noqa: E402
from pybattle.web_schema import view_json  # noqa: E402
from rl import determinize as DET  # noqa: E402
from rl.opponent_model import ExperienceModel, bucket_of  # noqa: E402

MODEL = os.path.join(REPO, "runs", "opp_model_v3", "model.pt")


def compact(specs):
    return [[int(s[1]), *[int(x) for x in s[2]], int(s[3]), int(s[6]), int(s[7]), round(float(s[8]), 7)]
            for s in specs]


def posteriors(t, view, challenge, level):
    b = bucket_of(challenge)
    enemy = view.enemy_party[:3]
    seen_sp = {m.species for m in enemy if m.seen}
    rev = frozenset(m.revealed_item for m in enemy if m.seen and m.revealed_item)
    sp_p = t.species_probs(b, seen_sp)
    slots = []
    for m in enemy:
        if m.seen:
            R = frozenset(x for x in m.revealed_moves if x)
            I = int(m.revealed_item) if m.revealed_item else -1
            cs, pc, pn = t.config_posterior(b, int(m.species), R, I, rev)
            cs2, pc2, pn2 = t.config_posterior(b, int(m.species), R, I, rev - ({I} if I >= 0 else set()))
            slots.append({"seen": True, "species": int(m.species), "draw": [pc.tolist(), float(pn)],
                          "slot_table": [pc2.tolist(), float(pn2)]})
        else:
            top = [int(t.frontier[i]) for i in np.argsort(-sp_p, kind="stable")[:5]]
            slots.append({"seen": False, "species": top,
                          "configs": [[c.tolist() for c in t.config_posterior(b, s, frozenset(), -1, rev)[1:2]]
                                      + [float(t.config_posterior(b, s, frozenset(), -1, rev)[2])] for s in top]})
    joint = None
    if any(not m.seen for m in enemy):
        tab = t.joint_table(b, rev, level, sorted(seen_sp))
        joint = np.diff(np.concatenate([[0.0], tab["cum"]])).tolist()
    return {"bucket": b, "species_probs": sp_p.tolist(), "slots": slots, "joint_w": joint}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", type=int, default=40)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", default=os.path.join(REPO, "dist", "fixtures", "determinize.json"))
    args = ap.parse_args()
    model = ExperienceModel.load(MODEL)
    model_hint = ExperienceModel.load(MODEL, cfg={"use_hint": True})
    t = model.compile()
    model_hint._compiled = t                       # (same counts: cfg use_hint does not change the tables)
    rng = random.Random(3)
    out = {"model": os.path.relpath(MODEL, REPO), "n": args.n, "decisions": []}
    want_noland = 4
    want = args.decisions - want_noland
    seed = 0
    while (want > 0 or want_noland > 0) and seed < 5000:
        noland_run = want_noland > 0 and seed % 5 == 4
        sim = SimBackend(max_turns=100)
        sim.reset(seed, 20 if noland_run else rng.choice([0, 6, 13, 20, 27, 34, 41, 48]))
        seed += 1
        per_battle = {}                            # (at most 2 decisions per battle: spread over many battles)
        while sim.phase != Phase.RUN_OVER:
            v, info = sim.view(), sim.run_info()
            if sim.phase in (Phase.BATTLE, Phase.FORCED_SWITCH):
                key = (info.win_streak, info.battle_in_challenge)
                take = (rng.random() < min(0.04 + 0.02 * v.turn, 0.3)    # (later turns: more seen and revealed)
                        and per_battle.get(key, 0) < 2)
                if info.noland and want_noland > 0 and take:
                    want_noland -= 1
                elif not info.noland and want > 0 and take and not noland_run:
                    want -= 1
                else:
                    take = False
                if take:
                    per_battle[key] = per_battle.get(key, 0) + 1
                    hint = DET.battle_hint(v, info.noland)
                    level = DET.team_level(v)
                    d = {"view": view_json(v), "info": view_json(info), "challenge": info.challenge_num,
                         "noland": bool(info.noland), "hint": list(hint) if hint else None,
                         "hidden": [int(x) for x in DET.hidden_counters(v)],
                         "post": posteriors(t, v, info.challenge_num, level)}
                    r = random.Random(rng.getrandbits(32))
                    d["samples"] = [compact(DET.determinization(v, None, r, True, "experience", hint=hint, model=model,
                                                                challenge=info.challenge_num))
                                    for _ in range(args.n)]
                    d["samples_hint"], d["hint_fallback"] = None, 0
                    if hint is not None:
                        f0 = DET.hint_stats["fallback"]
                        d["samples_hint"] = [compact(DET.determinization(v, None, r, True, "experience", hint=hint,
                                                                         model=model_hint,
                                                                         challenge=info.challenge_num))
                                             for _ in range(args.n)]
                        d["hint_fallback"] = DET.hint_stats["fallback"] - f0
                    out["decisions"].append(d)
                    print(f"{len(out['decisions'])}: seed {seed - 1} challenge {info.challenge_num} noland "
                          f"{info.noland} seen {sum(m.seen for m in v.enemy_party)} fallback {d['hint_fallback']}",
                          flush=True)
            if sim.phase == Phase.RENTAL:
                while True:
                    act = tuple(rng.sample(range(6), 3))
                    if len({v.candidates[i].species for i in act}) == 3:
                        break
            elif sim.phase == Phase.SWAP:
                act = None
            else:
                act = rng.choice([x for x in v.legal_actions if x[0] != "forfeit"])
            sim.act(act)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"{args.out}: {len(out['decisions'])} decisions, {os.path.getsize(args.out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()

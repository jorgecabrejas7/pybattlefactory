#!/usr/bin/env python3
"""The "experience" opponent model's tables for the browser port (battle-arcade games/emerald-factory/src/ai/
determinize.ts), dumped from the Python objects themselves: the compiled ExperienceModel (rl/opponent_model.py
_Compiled: species and configuration tables after EM, nature counts) and the rl/determinize.py game knowledge the
experience sampler falls back on (learnsets, plausible items, base HP / abilities, the hint rule).

    export_opponent_model.py [--model runs/opp_model_v3/model.pt] [--out dist/data/opponent_model.json]

Layout (compact JSON, arrays):
    cfg                 the model's cfg (a_nat, use_hint, max_hint_tries, level, ...)
    n_buckets           N_BUCKETS (challenge buckets)
    frontier            _Compiled.frontier: the Frontier species (p_species columns)
    p_species           _Compiled.p_species [n_buckets][len(frontier)]
    configs             per species with known configurations: [species, [[moves (sorted), item], ...] (Python's
                        sorted order), table [n_buckets][n_cfg + 1] (last column: novel), natures [n_cfg] (25 counts
                        or null)]
    learn               per species: [[move, min level], ...] sorted by move: learnable(s, level) is the moves whose
                        min level <= level (0: TM / tutor / egg; Smeargle: every move but Struggle at level 0),
                        checked here against DET.learnable at levels 50 and 100
    items_general       DET._GENERAL_ITEMS; items_specific: {species: [items]} (plausible_items = general + specific)
    species_base_hp     SPECIES[s]["base"][0]; species_abilities: SPECIES[s]["abilities"] (2 per species)
    hint                the hint rule (same layout as planner.json "hint")
    consts              KEEP, NUM_NATURES, MAX_EVS, MAX_EV_PER_STAT, HP_BAR_PIXELS, SHEDINJA, INTIMIDATE, batches
"""

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from rl import determinize as DET  # noqa: E402
from rl import opponent_model as OM  # noqa: E402


def learn_table():
    """{species: [[move, min level]]} reproducing DET.learnable for every species (asserted at levels 50 / 100)."""
    out = {}
    for s in range(1, len(DET.SPECIES)):
        if s == DET.SPECIES_SMEARGLE:
            rows = [[m, 0] for m in DET.learnable(s, 100)]
        else:
            ls = DET._LEARNSETS.get(s, {"level": [], "other": []})
            lv = {}
            for level, m in ls["level"]:
                lv[m] = min(lv.get(m, 10 ** 9), level)
            for m in ls["other"]:
                lv[m] = 0
            rows = [[m, lv[m]] for m in sorted(lv)]
        for level in (50, 100):
            assert tuple(m for m, l in rows if l <= level) == DET.learnable(s, level), (s, level)
        if rows:
            out[str(s)] = rows
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(REPO, "runs", "opp_model_v3", "model.pt"))
    ap.add_argument("--out", default=os.path.join(REPO, "dist", "data", "opponent_model.json"))
    args = ap.parse_args()
    model = OM.ExperienceModel.load(args.model)
    t = model.compile()
    configs = []
    for s in sorted(t.cfgs):
        cs = t.cfgs[s]
        nat = [t.natures[(s, c)].tolist() if (s, c) in t.natures else None for c in cs]
        configs.append([int(s), [[list(c[0]), int(c[1])] for c in cs], t.tables[s].tolist(), nat])
    assert not t.p_novel_unknown or all(s not in t.cfgs for s in t.p_novel_unknown)
    out = {
        "source": os.path.relpath(args.model, REPO),
        "cfg": dict(model.cfg),
        "n_buckets": OM.N_BUCKETS,
        "frontier": t.frontier.tolist(),
        "p_species": t.p_species.tolist(),
        "configs": configs,
        "learn": learn_table(),
        "items_general": list(DET._GENERAL_ITEMS),
        "items_specific": {str(k): list(v) for k, v in DET._SPECIFIC_ITEMS.items()},
        "species_base_hp": [int(s["base"][0]) for s in DET.SPECIES],
        "species_abilities": [[int(x) for x in s["abilities"]] for s in DET.SPECIES],
        "hint": {"move_style": DET.MOVE_STYLE.tolist(), "style_required": list(DET.STYLE_REQUIRED),
                 "species_types": DET.SPECIES_TYPES.tolist(), "type_none": DET.HINT_TYPE_NONE,
                 "style_many": DET.HINT_STYLE_MANY},
        "consts": {"KEEP": DET.KEEP, "NUM_NATURES": DET.NUM_NATURES, "MAX_EVS": DET.MAX_EVS,
                   "MAX_EV_PER_STAT": DET.MAX_EV_PER_STAT, "HP_BAR_PIXELS": DET.HP_BAR_PIXELS,
                   "SPECIES_SHEDINJA": DET.SPECIES_SHEDINJA, "ABILITY_INTIMIDATE": DET.ABILITY_INTIMIDATE,
                   "hint_batches": list(DET._batches(model.cfg["max_hint_tries"]))},
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"{args.out}: {os.path.getsize(args.out) / 1e3:.0f} kB ({len(configs)} species with configurations, "
          f"{sum(len(c[1]) for c in configs)} configurations)")


if __name__ == "__main__":
    main()

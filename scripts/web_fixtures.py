#!/usr/bin/env python3
"""Golden tactician decisions for the browser port's parity tests (battle-arcade games/emerald-factory/tests).

Plays SimBackend runs (random legal actions) and, at every rental / swap decision, records what the Python
tactician computes (rl/alphazero_v3.py TeamFeaturesV3 + the network, as rl/real_agents.py V3Player does):
    view, run_info          the web JSON (pybattle/web_schema.py)
    ctx, hint, v_next       the planner's inputs
    obs                     rl/encode.py rental / swap encodings
    options                 rl/tactician_search.py options_of
    draws                   the planner's random draws, in call order: sample_pool_teams results and
                            seen_set_mon results (the TS planner replays them instead of drawing)
    feats, phi              option_features_v3 [n_opt, 7] (WIN_COLUMNS / WIN_SCALE), phi_v3 (wins)
    opt_feat                feats scattered into the action space (the network input)
    net                     the network outputs on obs + opt_feat (rental: lead_logp, pair_logp, critic; swap:
                            probs, critic), denormalized like scripts/export_onnx.py
    action                  the greedy action

    web_fixtures.py [--decisions 40] [--out dist/fixtures/tactician.json]
"""

import argparse
import json
import os
import random
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from pybattle.backend import Phase, SimBackend  # noqa: E402
from pybattle.web_schema import view_json  # noqa: E402
from rl import determinize as DET, encode, team_eval as TE  # noqa: E402
from rl.alphazero_v3 import features_hook  # noqa: E402
from rl.policy import Policy  # noqa: E402
from rl.real_agents import ctx_of  # noqa: E402
from rl.tactician_features import to_action_space  # noqa: E402
from rl.tactician_search import options_of  # noqa: E402


def _plain(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    return x


class Recorder:
    """Wraps the planner's random draws (rl.team_eval.plan_values calls DET.sample_pool_teams and seen_set_mon)."""

    def __init__(self):
        self.draws = []
        self._pool, self._seen = DET.sample_pool_teams, TE.seen_set_mon

    def __enter__(self):
        def pool_teams(*a, **kw):
            out = self._pool(*a, **kw)
            self.draws.append({"pool_teams": out.tolist()})
            return out

        def seen(*a, **kw):
            m = self._seen(*a, **kw)
            self.draws.append({"seen": [m.species, list(m.moves), m.item, m.ability]})
            return m

        DET.sample_pool_teams, TE.seen_set_mon = pool_teams, seen
        return self

    def __exit__(self, *exc):
        DET.sample_pool_teams, TE.seen_set_mon = self._pool, self._seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(REPO, "runs", "alphazero_v3", "snapshot.pt"))
    ap.add_argument("--decisions", type=int, default=40)
    ap.add_argument("--out", default=os.path.join(REPO, "dist", "fixtures", "tactician.json"))
    args = ap.parse_args()
    pol = Policy(args.ckpt, "cpu")
    hook = features_hook(args.ckpt)
    hook.planner_cfg["seed"] = 0
    norm = pol.value_norm.get("tactician")
    std = max(norm["var"], 1e-4) ** 0.5 if norm else 1.0
    mean = norm["mean"] if norm else 0.0
    rng = random.Random(1)
    out = {"ckpt": os.path.relpath(args.ckpt, REPO), "v_next": hook.table().tolist(), "decisions": []}
    want = {"rental": args.decisions // 2, "swap": args.decisions - args.decisions // 2}
    seed = 0
    while (want["rental"] > 0 or want["swap"] > 0) and seed < 5000:
        sim = SimBackend(max_turns=100)
        sim.reset(seed, rng.choice([0, 6, 13, 20, 27, 34, 41, 48]))
        seed += 1
        while sim.phase != Phase.RUN_OVER:
            v, info = sim.view(), sim.run_info()
            if sim.phase in (Phase.RENTAL, Phase.SWAP):
                kind = "rental" if sim.phase == Phase.RENTAL else "swap"
                if want[kind] > 0:
                    want[kind] -= 1
                    ctx = ctx_of(info)
                    obs = encode.rental(v, ctx) if kind == "rental" else encode.swap(v, ctx)
                    hook.planner_cfg["seed"] = rng.getrandbits(31)
                    with Recorder() as rec:
                        r = hook.compute(sim, kind, obs, ctx)
                    hook2 = dict(hook.planner_cfg)
                    with Recorder() as rec_phi:
                        phi = TE.phi_v3(hook._load(), v, kind, ctx, hook.table(), (v.hint_type, v.hint_style), hook2)
                    x = encode.collate([dict(obs, opt_feat=r["opt_feat"])], "cpu")
                    with torch.no_grad():
                        if kind == "rental":
                            ll, pl, crit = pol.net.rental_joint(x)
                            net = {"lead_logp": torch.log_softmax(ll, -1)[0].numpy(),
                                   "pair_logp": torch.log_softmax(pl, -1)[0].numpy(),
                                   "critic": float(crit[0]) * std + mean}
                        else:
                            lg, crit = pol.net.swap(x)
                            net = {"probs": torch.softmax(lg, -1)[0].numpy(), "critic": float(crit[0]) * std + mean}
                    a = pol.act(kind, x, greedy=True)[0]
                    out["decisions"].append(_plain({
                        "kind": kind, "view": view_json(v), "info": view_json(info), "ctx": ctx,
                        "hint": [v.hint_type, v.hint_style], "obs": {k: np.asarray(val) for k, val in obs.items()},
                        "options": [list(o) if isinstance(o, tuple) else o for o in r["options"]],
                        "draws": rec.draws, "draws_phi": rec_phi.draws, "feats": r["feats"], "phi": phi,
                        "opt_feat": r["opt_feat"], "net": net, "action": a}))
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
        json.dump(out, f)
    print(f"{args.out}: {len(out['decisions'])} decisions, {os.path.getsize(args.out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()

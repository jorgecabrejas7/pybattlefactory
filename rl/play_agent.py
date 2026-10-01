"""The alphazero_v3 agent as it plays (docs/RL_DECISIONS.md §20.6, §22.6), with what rl/playlog.py logs at each
decision: its policy, its per-option features and the counterfactual probes (docs §23).

    agent = PlayAgent("runs/alphazero_v3/latest.pt", battler="net" | "search")
    action, policy, extras = agent.tactician(backend, kind, obs, ctx)      # kind rental | swap
    action, policy, extras = agent.battler(backend, obs, ctx, decisions)   # BATTLE / FORCED_SWITCH
    logger.decision(kind, backend.view(), backend.run_info(), policy, action, extras)

backend: anything with view() / run_info() (SimBackend; the emulator backend for the network battler; the search
needs a SimBackend). obs: rl.encode's observation of the decision (FactoryEnv's event "obs"); ctx: FactoryEnv._ctx().

Tactician (greedy, exactly as rl.eval_rounds plays it): its network with the per-option features of its team
evaluator (W and the depth-1 planner, rl.alphazero_v3.features_hook: the checkpoint's W, v_next table and
planner_cfg). The planner's opponent draws use a seed per decision (`planner_seed`) so that the probes below see the
same draws (common random numbers); the agent's play is the same in distribution as with fresh draws.
Probes (network only; logged in extras["probes"], never acted on):
    hint_none_input   the hint replaced by "none" (type 18, style 0) in the network's input only
    hint_none_full    ... and in the features too (W and the planner without a hint: the "unknown" symbol)
    hint_<t>_input / hint_<t>_full   the hint type replaced by type t (style kept), for n_hint_probes types t drawn
                      among the 17 others (not the true one, not the mystery type 9)
    feat_zero         the per-option features (W / planner) zeroed
    each: {"choice": option index (in policy["legal"] order), "probs": [...], "value", "trio_probs" (rental)}
Battler: the network alone (greedy argmax, battler="net") or the legal search (512 simulations, 8 determinizations,
the "experience" opponent sampler of rl/opponent_model.py with runs/opp_model_v3/model.pt, battler="search").
Logged: the network's probabilities and value (both modes), the search's visits / Q (search), the damage estimates of
rl/damage.py (what a player can compute: per own move [min, max, can-KO] of the opponent's max HP; the max-damage
move), and the experience model's belief about the opponent's active Pokemon (top configurations).
"""

import dataclasses
import random
from itertools import combinations

import numpy as np
import torch
import torch.nn.functional as F

from . import encode
from .damage import battle_estimates
from .envs import decode_action
from .policy import Policy

TRIOS = list(combinations(range(6), 3))           # the 20 unordered teams of a rental screen
NO_HINT = (18, 0)
TYPE_MYSTERY = 9
FEATURE_NAMES = ("e_round", "w_normal", "w_last", "delta", "q_plan", "q_plan_centered", "phi")
WIN_SCALE = 10.0
DEFAULT_OPP_MODEL = "runs/opp_model_v3/model.pt"


def _r(x, d=4):
    return np.round(np.asarray(x, np.float64), d).tolist()


def rental_trio(option):
    """(lead, pair index) -> the sorted trio of rental slots."""
    lead, p = option
    j, k = encode.PAIRS[p]
    return tuple(sorted((lead, j, k)))


def trio_probs(options, probs):
    """P(team = trio) for the 20 trios (summing over the leads)."""
    out = np.zeros(len(TRIOS))
    idx = {t: i for i, t in enumerate(TRIOS)}
    for o, p in zip(options, probs):
        out[idx[rental_trio(o)]] += p
    return out


class PlayAgent:
    def __init__(self, ckpt, battler="net", sims=512, dets=8, opponent_model=DEFAULT_OPP_MODEL, team_eval=None,
                 probes=True, n_hint_probes=3, full_hint_probes=True, seed=0, device="cpu", beliefs=True,
                 evaluator=None):
        from .alphazero_v3 import features_hook
        self.policy = Policy(ckpt, torch.device(device))
        self.net = self.policy.net
        self.device = self.policy.device
        self.features = features_hook(ckpt, team_eval)
        if self.features is None:
            raise ValueError(f"{ckpt} is not an alphazero_v3 checkpoint with tactician features")
        vn = self.policy.value_norm or {}
        self._tnorm = vn.get("tactician")
        self._bnorm = vn.get("battler")
        self.battler_kind = battler
        self.probes, self.n_hint_probes, self.full_hint_probes = probes, n_hint_probes, full_hint_probes
        self.rng = random.Random(seed)
        self.seed = seed
        self.n_decisions = 0
        self.opp_model = None
        if opponent_model and (beliefs or battler == "search"):
            from .opponent_model import ExperienceModel
            self.opp_model = ExperienceModel.load(opponent_model)
            self.opp_model.compile()
        self.beliefs = beliefs and self.opp_model is not None
        self.search = None
        if battler == "search":
            from .search import SearchBattler
            kw = {"evaluator": evaluator} if evaluator is not None else {}
            self.search = SearchBattler(self.policy, n_sims=sims, n_determinizations=dets, mode="legal", seed=seed,
                                        opponent_prior="experience", opponent_model=self.opp_model, **kw)
        elif battler != "net":
            raise ValueError(f"battler must be 'net' or 'search', not {battler!r}")
        self.meta = {"ckpt": ckpt, "battler": battler, "sims": sims if battler == "search" else 0,
                     "dets": dets if battler == "search" else 0, "opponent_prior": "experience",
                     "opponent_model": opponent_model, "team_eval": self.features.path,
                     "planner_cfg": self.features.planner_cfg, "probes": probes, "n_hint_probes": n_hint_probes,
                     "full_hint_probes": full_hint_probes, "feature_names": list(FEATURE_NAMES)}

    # ---- tactician -------------------------------------------------------------------------------------------------

    def _denorm(self, v, norm):
        return float(v) * max(norm["var"], 1e-4) ** 0.5 + norm["mean"] if norm else float(v)

    @torch.no_grad()
    def _tactician_eval(self, kind, obs_list, options):
        """Per observation: (greedy choice index in options, option probabilities, value, extra)."""
        x = encode.collate(obs_list, self.device)
        out = []
        if kind == "rental":
            ll, pl, v = self.net.rental_joint(x)
            llp = F.log_softmax(ll, -1).cpu().numpy()
            plp = F.log_softmax(pl, -1).cpu().numpy()
            for b in range(len(obs_list)):
                lead = int(np.argmax(llp[b]))
                pair = int(np.argmax(plp[b, lead]))
                probs = np.array([np.exp(llp[b, l] + plp[b, l, p]) for l, p in options])
                choice = options.index((lead, pair))
                out.append((choice, probs, self._denorm(v[b], self._tnorm),
                            {"lead_probs": _r(np.exp(llp[b])), "trio_probs": _r(trio_probs(options, probs))}))
        else:
            logits, v = self.net.swap(x)
            lp = F.log_softmax(logits, -1).cpu().numpy()
            for b in range(len(obs_list)):
                probs = np.exp(lp[b][options])
                choice = int(np.argmax(probs))      # = argmax over the 10 logits (illegal ones are masked)
                out.append((choice, probs, self._denorm(v[b], self._tnorm), {}))
        return out

    def _features(self, view, kind, ctx, options, hint, seed):
        from . import team_eval as T
        ev = self.features._load()
        cfg = dict(self.features.planner_cfg, seed=seed)
        return np.asarray(T.option_features_v3(ev, view, kind, ctx, options, self.features.v_next, hint, cfg),
                          np.float32)

    def _encode(self, kind, view, ctx):
        return encode.rental(view, ctx) if kind == "rental" else encode.swap(view, ctx)

    def tactician(self, backend, kind, obs, ctx):
        """-> (backend action, policy dict, extras). obs: the decision's observation without "opt_feat"."""
        from .tactician_search import options_of
        from .tactician_features import to_action_space
        view = backend.view()
        info = backend.run_info()
        options = options_of(kind, obs)
        noland = bool(info.noland)
        hint = None if noland else (int(view.hint_type), int(view.hint_style))
        pseed = self.rng.getrandbits(31)
        nf = len(FEATURE_NAMES)
        feats = self._features(view, kind, ctx, options, hint, pseed)

        def with_feat(o, f):
            o = dict(o)
            o["opt_feat"] = to_action_space(kind, options, f, width=nf)
            return o

        variants = [("base", with_feat(obs, feats))]
        if self.probes:
            if hint is not None and hint != NO_HINT:
                v0 = dataclasses.replace(view, hint_type=NO_HINT[0], hint_style=NO_HINT[1])
                o0 = self._encode(kind, v0, ctx)
                variants.append(("hint_none_input", with_feat(o0, feats)))
                if self.full_hint_probes:
                    variants.append(("hint_none_full", with_feat(o0, self._features(v0, kind, ctx, options, None,
                                                                                    pseed))))
            if hint is not None:
                others = [t for t in range(18) if t not in (TYPE_MYSTERY, hint[0])]
                for t in self.rng.sample(others, min(self.n_hint_probes, len(others))):
                    vt = dataclasses.replace(view, hint_type=t)
                    ot = self._encode(kind, vt, ctx)
                    variants.append((f"hint_{t}_input", with_feat(ot, feats)))
                    if self.full_hint_probes:
                        variants.append((f"hint_{t}_full", with_feat(ot, self._features(vt, kind, ctx, options,
                                                                                        (t, hint[1]), pseed))))
            variants.append(("feat_zero", with_feat(obs, np.zeros_like(feats))))
        res = self._tactician_eval(kind, [o for _, o in variants], options)
        choice, probs, value, extra = res[0]
        option = options[choice]
        action = decode_action(kind, option if kind == "rental" else option)
        legal = [list(o) for o in options] if kind == "rental" else list(options)
        policy = {"probs": _r(probs, 5), "value": value, "legal": legal, "choice": choice, **extra}
        probes = {}
        for (name, _), (c, p, v, ex) in zip(variants[1:], res[1:]):
            probes[name] = {"choice": c, "probs": _r(p, 4), "value": round(v, 4)}
            if "trio_probs" in ex:
                probes[name]["trio_probs"] = ex["trio_probs"]
        q_plan = feats[:, FEATURE_NAMES.index("q_plan")] * WIN_SCALE
        extras = {"options": legal, "features": _r(feats, 4), "q_plan": _r(q_plan, 4), "hint": hint,
                  "noland": noland, "planner_seed": pseed, "probes": probes, "ctx": ctx}
        self.n_decisions += 1
        return action, policy, extras

    # ---- battler ---------------------------------------------------------------------------------------------------

    @torch.no_grad()
    def _battle_net(self, obs):
        logits, v = self.net.battler(encode.collate([obs], self.device))
        p = torch.softmax(logits, -1)[0].cpu().numpy()
        p = np.where(obs["mask"], p, 0.0)
        return p, min(max(self._denorm(v[0], self._bnorm), 0.0), 1.0)

    def _estimates(self, view):
        """rl/damage.py's player-side estimates for the active Pokemon (fractions of the opponent's max HP)."""
        try:
            own_moves, foe_moves, threats, speed = battle_estimates(view, version=4)
        except Exception as e:                                   # pragma: no cover - unexpected view states
            return {"error": repr(e)}
        a = view.own_active.party_index
        rows = own_moves[a]
        usable = list(view.usable_moves)
        mean = [((lo + hi) / 2 if usable[j] and view.own_party[a].moves[j] else -1.0) for j, (lo, hi, _) in
                enumerate(rows)]
        best = int(np.argmax(mean)) if max(mean) > 0 else -1
        return {"own_moves": _r(rows, 3), "max_damage_move": best, "foe_moves": _r(foe_moves, 3),
                "threats": _r(threats, 3), "speed": _r(speed, 1), "all_own_moves": _r(own_moves, 3)}

    def _beliefs(self, view, ctx):
        """The experience model's belief about the opponent's active slot: top configurations (moves, item, p)."""
        if not self.beliefs:
            return None
        slot = view.enemy_active.party_index
        try:
            post = self.opp_model.slot_posterior(view, slot, ctx["challenge"])
            sp = view.enemy_party[slot].species
            cs, pc, pn = post["config"](sp)
        except Exception as e:                                   # pragma: no cover
            return {"error": repr(e)}
        order = np.argsort(-np.asarray(pc))[:3] if len(pc) else []
        return {"slot": slot, "species": sp, "p_novel": round(float(pn), 4), "n_known": len(cs),
                "top": [{"moves": list(cs[i][0]), "item": int(cs[i][1]), "p": round(float(pc[i]), 4)} for i in order]}

    def battler(self, backend, obs, ctx, decisions=0):
        """-> (backend action, policy dict, extras)."""
        view = backend.view()
        p, v = self._battle_net(obs)
        legal = [int(i) for i in np.flatnonzero(obs["mask"])]
        extras = {"estimates": self._estimates(view), "ctx": ctx}
        policy = {"probs": _r(p, 5), "value": round(v, 5), "legal": legal}
        if self.search is not None:
            action = self.search.act_on(backend, decisions=decisions)
            st = self.search.last_stats
            a = int(st["action"])
            extras["search"] = {"visits": _r(st["visits"], 1), "q": _r(st["q"], 4), "searched": st["searched"],
                                "ms": round(st["ms"], 1)}
        else:
            a = int(np.argmax(np.where(obs["mask"], p, -1.0)))
            action = decode_action("battle", a)
        policy["action_index"] = a
        b = self._beliefs(view, ctx)
        if b is not None:
            extras["beliefs"] = b
        self.n_decisions += 1
        return action, policy, extras

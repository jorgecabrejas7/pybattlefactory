"""alphazero_v3 as a player of the real game (scripts/play_real.py, scripts/watch.py).

    player = V3Player("runs/alphazero_v3/latest.pt", mode="search")     # or mode="net"
    choice = player.act(backend)            # backend: EmuBackend (or SimBackend) at a decision -> Choice

Exactly the agents of the evaluation (rl/eval_rounds.py, docs §20–§22):
- tactician: the checkpoint's network, greedy, with its per-option features (rl.alphazero_v3.features_hook: the
  W it trained with, its V_siguiente table and planner config; the attendant's hint from the view, none before
  Noland). The planner draws its 8 opponents with a seed per decision (logged), so a decision is reproducible;
- battler, mode "net": the network alone, greedy (argmax of its policy);
- battler, mode "search": rl.search.SearchBattler, legal mode, 512 simulations over 8 determinizations, c_puct 1.5,
  batch 32, opponents drawn by the "experience" model (rl/opponent_model.py, runs/opp_model_v3/model.pt).
  The search steps a native Gen3Game: on the emulator it runs on the battle mirrored in the simulator
  (pybattle.emu.mirror: the same battle code, started from the battle's start snapshot and given the same actions,
  checked byte for byte at every decision). If the mirror ever disagrees, that battle falls back to the network
  (details["search_fallback"]).

Every Choice carries `details` for the watch window (probabilities, P(win), search visits / Q, damage estimates,
the tactician's per-option W / E_round / Q_plan / Δ, the hint, the opponent model's beliefs) and
details["log"] = (policy, extras) for rl.playlog.PlayLogger.decision.
"""

import os
import random
import time

import numpy as np
import torch

from pybattle.agent import Choice
from pybattle.backend import Phase
from pybattle.view import NAMES, NO_HINT_TYPE
from . import damage, encode
from .envs import decode_action
from .policy import Policy

DEFAULT_CKPT = "runs/alphazero_v3/latest.pt"
DEFAULT_OPP_MODEL = "runs/opp_model_v3/model.pt"
STYLES = ["none", "preparation", "slow & steady", "endurance", "high risk", "weakening", "unpredictable", "weather",
          "several"]


def _name(table, i):
    n = NAMES[table].get(str(int(i)), str(i)).split("_", 1)[-1]
    return n.title().replace("_", " ")


def ctx_of(info):
    """The context the networks see (FactoryEnv._ctx / rl.search.run_ctx)."""
    return {"streak": info.win_streak, "battle": info.battle_in_challenge, "challenge": info.challenge_num,
            "rents": info.rents}


def hint_text(view, noland):
    if noland:
        return "Noland next: no hint"
    if view.hint_type == NO_HINT_TYPE and not view.hint_style:
        return "hint: no common type, no style"
    ty = _name("types", view.hint_type) if view.hint_type < 18 else "no common type"
    st = STYLES[view.hint_style] if view.hint_style < len(STYLES) else str(view.hint_style)
    return f"hint: {ty} · {st}"


def beliefs(model, view, challenge, k=3):
    """The opponent model's beliefs about each opposing Pokemon given what has been revealed: per slot the k most
    likely (species, moves, item) with their probabilities, and the probability of a set it has never seen."""
    out = []
    enemy = view.enemy_party[:3]
    for slot, m in enumerate(enemy):
        d = {"slot": slot, "seen": bool(m.seen), "fainted": bool(m.fainted),
             "species": int(m.species) if m.seen else 0, "revealed_moves": [int(x) for x in m.revealed_moves if x],
             "revealed_item": int(m.revealed_item or 0), "sets": [], "p_new": 0.0}
        if m.seen and m.fainted:
            out.append(d)
            continue
        post = model.slot_posterior(view, slot, challenge)
        scored, p_new = [], 0.0
        if post["species"] is None:
            sp_list = [(int(m.species), 1.0)]
        else:
            sp_list = [(int(s), float(p)) for s, p in zip(post["frontier"], post["species"]) if p > 0]
        for s, ps in sp_list:
            cs, pc, pn = post["config"](s)
            scored += [(ps * float(p), s, c) for c, p in zip(cs, pc) if p > 0]
            p_new += ps * float(pn)
        scored.sort(key=lambda z: -z[0])
        d["sets"] = [{"p": p, "species": s, "moves": [int(x) for x in c[0]], "item": int(c[1])}
                     for p, s, c in scored[:k]]
        d["p_new"] = p_new
        if post["species"] is not None:            # unseen: also the most likely species
            top = np.argsort(-post["species"])[:k]
            d["species_probs"] = [{"species": int(post["frontier"][i]), "p": float(post["species"][i])} for i in top]
        out.append(d)
    return out


class V3Player:
    def __init__(self, ckpt=DEFAULT_CKPT, mode="net", sims=512, dets=8, c_puct=1.5, batch=32,
                 opponent_model=DEFAULT_OPP_MODEL, seed=0, device="cpu", threads=1, team_eval=None,
                 show_beliefs=True, planner_seed=True):
        if mode not in ("net", "search"):
            raise ValueError("mode is 'net' or 'search'")
        if threads:
            torch.set_num_threads(threads)
        self.ckpt, self.mode, self.seed = ckpt, mode, seed
        self.device = torch.device(device)
        self.policy = Policy(ckpt, self.device)
        from .alphazero_v3 import features_hook
        self.hook = features_hook(ckpt, team_eval)              # TeamFeaturesV3 (None for other checkpoints)
        self.planner_seed = planner_seed
        self.rng = random.Random(seed)
        self.opp_model = None
        if opponent_model and os.path.exists(opponent_model) and (show_beliefs or mode == "search"):
            from .opponent_model import ExperienceModel
            self.opp_model = ExperienceModel.load(opponent_model)
            self.opp_model.compile()
        self.show_beliefs = show_beliefs and self.opp_model is not None
        self.search = None
        self.search_kw = {}
        if mode == "search":
            from .search import SearchBattler
            self.search_kw = {"n_sims": sims, "n_determinizations": dets, "c_puct": c_puct, "batch": batch,
                              "mode": "legal", "seed": seed, "opponent_prior": "experience"}
            self.search = SearchBattler(self.policy, opponent_model=self.opp_model,
                                        **self.search_kw)
        from pybattle.emu.mirror import SimMirror
        self.mirror = SimMirror()
        self.battle_decisions = 0
        self.n_decisions = 0
        self.timing = {"tactician_ms": 0.0, "battler_ms": 0.0, "tactician": 0, "battler": 0}

    def describe(self):
        return {"ckpt": self.ckpt, "mode": self.mode, "seed": self.seed, "search": self.search_kw,
                "team_eval": getattr(self.hook, "path", None),
                "planner_cfg": getattr(self.hook, "planner_cfg", None), "opponent_model": DEFAULT_OPP_MODEL
                if self.opp_model is not None else None}

    # --- the decision ----------------------------------------------------------------------------------------

    def act(self, backend, phase=None, view=None, info=None) -> Choice:
        phase = phase or backend.phase
        view = view if view is not None else backend.view()
        info = info if info is not None else backend.run_info()
        t0 = time.perf_counter()
        if phase in (Phase.RENTAL, Phase.SWAP):
            ch = self._tactician(backend, phase, view, info)
            self.timing["tactician_ms"] += (time.perf_counter() - t0) * 1000
            self.timing["tactician"] += 1
            self.battle_decisions = 0
        else:
            ch = self._battler(backend, phase, view, info)
            self.timing["battler_ms"] += (time.perf_counter() - t0) * 1000
            self.timing["battler"] += 1
            self.battle_decisions += 1
            self.mirror.apply(ch.action)
        ch.details["ms"] = (time.perf_counter() - t0) * 1000
        ch.details["mode"] = self.mode
        self.n_decisions += 1
        return ch

    # --- tactician -----------------------------------------------------------------------------------------------

    def tactician_obs(self, backend, kind, view, info):
        """The observation the network gets, with the per-option features, as FactoryEnv + features_hook build it.
        -> (obs, hook output or None, planner seed)"""
        ctx = ctx_of(info)
        obs = encode.rental(view, ctx) if kind == "rental" else encode.swap(view, ctx)
        out, pseed = None, None
        if self.hook is not None:
            if self.planner_seed:
                pseed = self.rng.getrandbits(31)
                self.hook.planner_cfg["seed"] = pseed
            out = self.hook.compute(backend, kind, obs, ctx)
            obs = dict(obs)
            obs["opt_feat"] = out["opt_feat"]
        return obs, out, pseed

    @torch.no_grad()
    def _tactician(self, backend, phase, view, info):
        from .team_eval import WIN_COLUMNS, WIN_SCALE
        p = self.policy
        kind = "rental" if phase == Phase.RENTAL else "swap"
        obs, out, pseed = self.tactician_obs(backend, kind, view, info)
        x = encode.collate([obs], p.device)
        a = p.act(kind, x, greedy=True)[0]
        action = decode_action(kind, a)
        feats = out["feats"] if out is not None else None
        options = out["options"] if out is not None else None
        if feats is not None:
            feats = np.asarray(feats, np.float64).copy()
            feats[:, list(WIN_COLUMNS)] *= WIN_SCALE                     # back to wins
        norm = p.value_norm.get("tactician")
        hint = hint_text(view, info.noland)
        rows = []
        if kind == "rental":
            lead_logits, pair_all, v = p.net.rental_joint(x)
            lp = torch.log_softmax(lead_logits[0], -1).cpu().numpy()
            pp = torch.log_softmax(pair_all[0], -1).cpu().numpy()
            sp = [m.species for m in view.candidates]
            from .tactician_search import options_of
            opts = options or options_of(kind, obs)
            for n, (lead, pair) in enumerate(opts):
                j, k = encode.PAIRS[pair]
                prob = float(np.exp(lp[lead] + pp[lead, pair]))
                r = {"label": f"{_name('species', sp[lead])} + {_name('species', sp[j])}, {_name('species', sp[k])}",
                     "p": prob, "action": (lead, j, k), "option": [int(lead), int(pair)]}
                rows.append(r)
            chosen = next((n for n, r in enumerate(rows) if r["option"] == [int(a[0]), int(a[1])]), None)
            lead_probs = np.exp(lp)
        else:
            logits, v = p.net.swap(x)
            pr = torch.softmax(logits[0], -1).cpu().numpy()
            from .tactician_search import options_of
            opts = options or options_of(kind, obs)
            for q in opts:
                act = decode_action("swap", q)
                label = "Keep the team" if act is None else \
                    f"{_name('species', view.own_party[act[0]].species)} → {_name('species', view.enemy_party[act[1]].species)}"
                rows.append({"label": label, "p": float(pr[q]), "action": act, "option": int(q)})
            chosen = next((n for n, r in enumerate(rows) if r["option"] == int(a)), None)
            lead_probs = None
        if feats is not None:
            for r, f in zip(rows, feats):
                r.update({"e_round": float(f[0]), "w_normal": float(f[1]), "w_last": float(f[2]), "delta": float(f[3]),
                          "q_plan": float(f[4]), "phi": float(f[6])})
        critic = float(v.item())
        if norm:
            critic = critic * max(norm["var"], 1e-4) ** 0.5 + norm["mean"]
        # the critic learned the shaped return, wins ahead - Phi(s) (alphazero_v2 §19): expected wins = critic + Phi,
        # Phi = phi_v3 = the mean Q_plan of the legal rentals / the Q_plan of keeping the team
        phi = 0.0
        if feats is not None and len(rows):
            phi = float(np.mean([r["q_plan"] for r in rows])) if kind == "rental" else \
                float(next((r["q_plan"] for r in rows if r["option"] == 0), rows[0]["q_plan"]))
        ew = critic + phi
        best_q = max(range(len(rows)), key=lambda n: rows[n].get("q_plan", -1e9)) if rows and feats is not None else None
        ch_row = rows[chosen] if chosen is not None else {}
        reason = f"{ch_row.get('label', action)} ({ch_row.get('p', 0):.0%})"
        if "q_plan" in ch_row:
            reason += f" · Q_plan {ch_row['q_plan']:.2f} wins"
        reason += f" · expected wins ahead ≈ {ew:.1f} · {hint}"
        details = {"kind": kind, "tactician_options": rows, "chosen_option": chosen, "best_q_plan": best_q,
                   "expected_wins": ew, "critic": critic, "phi": phi, "hint": hint, "planner_seed": pseed,
                   "team": list(action) if kind == "rental" else None}
        details["actions"], details["chosen"] = self._tactician_bars(rows, chosen, kind)
        if kind == "rental":
            details["lead_probs"] = [float(x) for x in lead_probs]
        policy = {"probs": {r["label"]: r["p"] for r in rows}, "value": ew, "critic": critic, "phi": phi}
        extras = {"options": [{k: r.get(k) for k in ("label", "p", "e_round", "w_normal", "w_last", "delta", "q_plan",
                                                      "phi", "option")} for r in rows],
                  "q_plan": [r.get("q_plan") for r in rows], "hint": hint,
                  "hint_raw": [int(view.hint_type), int(view.hint_style)], "noland": bool(info.noland),
                  "planner_seed": pseed, "chosen_option": chosen}
        details["log"] = (policy, extras)
        return Choice(action, reason, details)

    @staticmethod
    def _tactician_bars(rows, chosen, kind):
        """The bar list for the generic decision panel (rentals: the 8 most likely teams + the chosen one)."""
        order = sorted(range(len(rows)), key=lambda n: -rows[n]["p"])
        if kind == "rental":
            keep = order[:8]
            if chosen is not None and chosen not in keep:
                keep.append(chosen)
        else:
            keep = order
        bars = [dict(rows[n]) for n in keep]
        return bars, (keep.index(chosen) if chosen in keep else None)

    # --- battler -------------------------------------------------------------------------------------------------

    def _mirror_backend(self, backend, phase):
        """A backend the search can step (SimBackend: itself; EmuBackend: the mirrored battle) or (None, why)."""
        if hasattr(backend, "game"):
            return backend, None
        emu = backend.d.emu
        take = getattr(emu, "take_start", None)
        if take is None:
            return None, "the emulator is not a WatchedMachine (no battle start snapshot)"
        start = take()
        if start is not None:
            self.mirror.begin(start)
        elif self.mirror.game is None:
            return None, "no battle start seen"
        if not self.mirror.at_decision(backend.d, phase == Phase.FORCED_SWITCH):
            return None, self.mirror.error or "mirror out of sync"
        from pybattle.emu.mirror import MirroredBackend
        return MirroredBackend(backend, self.mirror), None

    @torch.no_grad()
    def _battler(self, backend, phase, view, info):
        p = self.policy
        ctx = ctx_of(info)
        obs = encode.battle(view, ctx)
        x = encode.collate([obs], p.device)
        logits, v = p.net.battler(x)
        pr = torch.softmax(logits, -1)[0].cpu().numpy()
        norm = p.value_norm.get("battler")
        win = float(v.item())
        if norm:
            win = win * max(norm["var"], 1e-4) ** 0.5 + norm["mean"]
        win = min(max(win, 0.0), 1.0)
        mask = obs["mask"].astype(bool)
        net_a = int(np.argmax(pr)) if mask.any() else 0
        a, search = net_a, None
        fallback = None
        mb = None
        if self.search is not None:
            mb, fallback = self._mirror_backend(backend, phase)
            if mb is not None and mask.sum() > 1:
                act = self.search.act_on(mb, decisions=max(self.battle_decisions, 0))
                a = (act[1] if act[0] == "move" else 4 + act[1])
                search = dict(self.search.last_stats)
        elif not hasattr(backend, "game"):
            pass
        me = view.own_party[view.own_active.party_index]
        labels = [_name("moves", me.moves[i]) if i < 4 and me.moves[i] else
                  ("→ " + _name("species", view.own_party[i - 4].species) if i >= 4 and i - 4 < len(view.own_party)
                   else "—") for i in range(7)]
        try:
            own_est, foe_est, threats, _ = damage.battle_estimates(view, ctx)
        except Exception:
            own_est, foe_est, threats = None, None, None
        acts = []
        for i in range(7):
            if not mask[i]:
                continue
            d = {"label": labels[i], "p": float(pr[i]), "action": decode_action("battle", i), "index": i}
            if i < 4:
                d["move"] = int(me.moves[i])
                if own_est is not None:
                    lo, hi, ko = own_est[view.own_active.party_index][i]
                    d["damage"] = [float(lo), float(hi), bool(ko)]
            else:
                d["species"] = int(view.own_party[i - 4].species)
                if threats is not None:
                    d["threat"] = [float(threats[i - 4][0]), bool(threats[i - 4][1])]
            if search is not None:
                d["visits"] = float(search["visits"][i])
                d["q"] = float(search["q"][i])
            acts.append(d)
        action = decode_action("battle", a)
        chosen = next((n for n, d in enumerate(acts) if d["index"] == a), None)
        if not mask.any():
            action, chosen = ("move", 0), None
        top = ", ".join(f"{labels[i]} {pr[i]:.0%}" for i in np.argsort(-pr)[:3] if pr[i] > 0.005)
        reason = f"net: {top} | P(win) ≈ {win:.0%}"
        if search is not None:
            tot = max(sum(search["visits"]), 1.0)
            reason = (f"search {int(tot)} visits: {labels[a]} {search['visits'][a] / tot:.0%} (Q {search['q'][a]:.2f})"
                      f" | net: {labels[net_a]} {pr[net_a]:.0%} | P(win) ≈ {win:.0%}")
        elif self.search is not None and fallback:
            reason += f" | search off this battle: {fallback}"
        details = {"kind": "battle", "actions": acts, "p_win": win, "chosen": chosen, "net_action": net_a,
                   "search": None, "search_fallback": fallback}
        if search is not None:
            details["search"] = {"visits": search["visits"], "q": search["q"], "ms": search["ms"],
                                 "leaves": search["leaves"], "value": search["value"],
                                 "q_chosen": float(search["q"][a])}
        if foe_est is not None:
            details["foe_damage"] = [[float(z) for z in e[:2]] + [bool(e[2])] for e in foe_est] if foe_est else []
        bel = None
        if self.show_beliefs:
            try:
                bel = beliefs(self.opp_model, view, info.challenge_num)
            except Exception as e:                                 # display extra: never fail a decision
                bel = None
                details["beliefs_error"] = f"{type(e).__name__}: {e}"
        details["beliefs"] = bel
        policy = {"probs": {labels[i]: float(pr[i]) for i in range(7) if mask[i]}, "value": win}
        extras = {"net_action": net_a, "search": details["search"], "search_fallback": fallback,
                  "beliefs": bel, "mirror_synced": bool(self.mirror.synced) if self.search is not None else None,
                  "damage": {d["label"]: d.get("damage") for d in acts if "damage" in d},
                  "search_seed": self.seed if self.search is not None else None,
                  "battle_decision": self.battle_decisions}
        details["log"] = (policy, extras)
        return Choice(action, reason, details)


# --- watch.py adapters (--agents rl.real_agents:Tactician,rl.real_agents:Battler) ------------------------------------

_PLAYER = None


def shared_player():
    """One V3Player for both adapters: PYB_CKPT (default runs/alphazero_v3/latest.pt), PYB_MODE net|search."""
    global _PLAYER
    if _PLAYER is None:
        _PLAYER = V3Player(os.environ.get("PYB_CKPT", DEFAULT_CKPT), os.environ.get("PYB_MODE", "net"),
                           sims=int(os.environ.get("PYB_SIMS", 512)), threads=int(os.environ.get("PYB_THREADS", 2)))
    return _PLAYER


class _Adapter:
    """watch.py agent: act(phase, view, info) on the backend given by bind()."""

    def __init__(self):
        self.backend = None

    def bind(self, backend):
        self.backend = backend

    def act(self, phase, view, info):
        return shared_player().act(self.backend, phase, view, info)


class Tactician(_Adapter):
    pass


class Battler(_Adapter):
    pass

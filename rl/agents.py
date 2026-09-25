"""A trained checkpoint as the two agents of the watch window (scripts/watch.py).

    PYB_CKPT=runs/ppo_joint_v3/latest.pt python scripts/watch.py --rom ROM --save SAV \
        --agents rl.agents:Tactician,rl.agents:Battler

The panel's "reason" line shows what the network thinks: the top actions with their probabilities and, for the
battler, the critic's estimate of the chance of winning the battle (the battler's return is 1 for a win, 0 for a
loss, gamma = 1, so its value is a win probability).
"""

import glob
import os

import numpy as np
import torch

from pybattle.agent import Choice
from pybattle.backend import Phase
from pybattle.view import NAMES
from . import damage, encode
from .envs import decode_action
from .policy import Policy

_P = None


def policy():
    global _P
    if _P is None:
        path = os.environ.get("PYB_CKPT") or max(glob.glob("runs/ppo_joint_v3/ckpt_*.pt"))
        _P = Policy(path, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        ck = torch.load(path, map_location="cpu")
        _P.norm = ck.get("value_norm") or {}
        _P.path = path
    return _P


def _ctx(info):
    return {"streak": info.win_streak, "battle": info.battle_in_challenge, "challenge": info.challenge_num,
            "rents": getattr(info, "rents", 0)}


def _name(table, i):
    n = NAMES[table].get(str(i), str(i)).split("_", 1)[-1]
    return n.title().replace("_", " ")


def _value(kind_agent, v):
    n = policy().norm.get(kind_agent)
    if n:
        v = v * max(n["var"], 1e-4) ** 0.5 + n["mean"]
    return v


class Tactician:
    def act(self, phase, view, info):
        p = policy()
        kind = "rental" if phase == Phase.RENTAL else "swap"
        obs = encode.rental(view, _ctx(info)) if kind == "rental" else encode.swap(view, _ctx(info))
        x = encode.collate([obs], p.device)
        a = p.act(kind, x, greedy=True)[0]
        action = decode_action(kind, a)
        with torch.no_grad():
            if kind == "rental":
                ll, pl, _, v = p.net.rental(x, lead=torch.tensor([int(a[0])], device=p.device))
                pr = torch.softmax(ll, -1)[0].cpu().numpy()
                pp = torch.softmax(pl, -1)[0].cpu().numpy()
                sp = [m.species for m in view.candidates]
                lead_txt = ", ".join(f"{_name('species', sp[i])} {pr[i]:.0%}" for i in np.argsort(-pr)[:3])
                ew = _value('tactician', v.item())
                reason = (f"lead: {lead_txt} | team {', '.join(_name('species', sp[i]) for i in action)}"
                          f" | expected wins ahead ≈ {ew:.1f}")
                legal = [i for i in range(6) if obs["lead_mask"][i]]
                pairs = [{"label": f"{_name('species', sp[j])} + {_name('species', sp[k])}", "p": float(pp[q]),
                          "slots": [j, k]}
                         for q, (j, k) in enumerate(encode.PAIRS) if obs["pair_mask"][action[0], q]]
                details = {"kind": "rental", "lead_probs": [float(v_) for v_ in pr], "team": list(action),
                           "actions": [{"label": _name("species", sp[i]), "p": float(pr[i]), "species": sp[i],
                                        "slot": i} for i in legal],
                           "chosen": legal.index(action[0]) if action[0] in legal else None,
                           "pairs": pairs, "chosen_pair": next((n for n, q in enumerate(pairs)
                                                               if set(q["slots"]) == set(action[1:])), None),
                           "expected_wins": ew}
            else:
                logits, v = p.net.swap(x)
                pr = torch.softmax(logits, -1)[0].cpu().numpy()
                ew = _value('tactician', v.item())
                reason = f"keep {pr[0]:.0%}"
                if action is not None:
                    reason += (f" | trade {_name('species', view.own_party[action[0]].species)} → "
                               f"{_name('species', view.enemy_party[action[1]].species)} {pr[int(a)]:.0%}")
                reason += f" | expected wins ahead ≈ {ew:.1f}"
                acts = []
                for q in range(10):
                    if not obs["mask"][q]:
                        continue
                    act = decode_action("swap", q)
                    label = "Keep the team" if act is None else \
                        f"{_name('species', view.own_party[act[0]].species)} → {_name('species', view.enemy_party[act[1]].species)}"
                    acts.append({"label": label, "p": float(pr[q]), "action": act, "index": q})
                details = {"kind": "swap", "actions": acts, "expected_wins": ew,
                           "chosen": next((n for n, d in enumerate(acts) if d["index"] == int(a)), None)}
        return Choice(action, reason, details)


class Battler:
    def act(self, phase, view, info):
        p = policy()
        ctx = _ctx(info)
        obs = encode.battle(view, ctx)
        if not obs["mask"].any():                      # nothing to choose: the game uses Struggle
            return Choice(("move", 0), "no choice (Struggle)", {"kind": "battle", "actions": []})
        x = encode.collate([obs], p.device)
        with torch.no_grad():
            logits, v = p.net.battler(x)
        pr = torch.softmax(logits, -1)[0].cpu().numpy()
        a = int(np.argmax(pr))
        me = view.own_party[view.own_active.party_index]
        labels = [_name("moves", me.moves[i]) if i < 4 else "→ " + _name("species", view.own_party[i - 4].species)
                  for i in range(7)]
        top = ", ".join(f"{labels[i]} {pr[i]:.0%}" for i in np.argsort(-pr)[:3] if pr[i] > 0.005)
        win = min(max(_value("battler", v.item()), 0.0), 1.0)
        try:
            own_est, _, threats, _ = damage.battle_estimates(view, ctx)
        except Exception:                                # estimates are a display extra: never fail a decision
            own_est, threats = None, None
        acts = []
        for i in range(7):
            if not obs["mask"][i]:
                continue
            d = {"label": labels[i], "p": float(pr[i]), "action": decode_action("battle", i)}
            if i < 4:
                d["move"] = me.moves[i]
                if own_est is not None:
                    lo, hi, ko = own_est[view.own_active.party_index][i]
                    d["damage"] = [float(lo), float(hi), bool(ko)]
            else:
                d["species"] = view.own_party[i - 4].species
                if threats is not None:
                    d["threat"] = [float(threats[i - 4][0]), bool(threats[i - 4][1])]
            acts.append(d)
        details = {"kind": "battle", "actions": acts, "p_win": win,
                   "chosen": next((n for n, d in enumerate(acts) if d["action"] == decode_action("battle", a)), None)}
        return Choice(decode_action("battle", a), f"{top} | P(win) ≈ {win:.0%}", details)

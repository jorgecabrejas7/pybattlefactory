"""How the agent plays: metrics from rl/playlog.py logs (emulator or simulator replica), per round and across
rounds, with bootstrap confidence intervals; a Spanish markdown report, a JSON with every number, and a few plots.
docs/RL_DECISIONS.md §23 defines every metric.

    python -m rl.playstats --logs runs/playstudy_v3/net --out runs/playstudy_v3/report_net [--title "..."]
    python -m rl.playstats --logs a.jsonl.gz b.jsonl.gz --out DIR [--no-late] [--boot 1000]

Pipeline: (1) extract(): one pass over the logs, turning each run into compact rows (rental decisions, swap
decisions, battles, battle decisions, Pokemon tenures); views are read from the player's side only (the record's
"truth" field is read only by truth_section(), flagged in the report). (2) analyze(): the metrics of §23 A-H on
those rows. (3) write_report(): report.md, stats.json, plots/*.png.

Rounds: "1".."6" = the round (challenge_num + 1) of the battle (a tactician decision belongs to the battle it
precedes), "7+" = rounds 7 and later, "N" = battles against Noland (the Factory Head: battle 21 and 42, ...), kept
apart. Runs that started at round 5 / 6 (rl/play_sim.py --late-runs) are included only with --late (default: off
for the run-level numbers, on for the per-round metrics, where they only add samples of rounds 5-6 and later).

Confidence intervals: 95 % percentile bootstrap over RUNS (all the decisions of a run are resampled together:
decisions of the same run are not independent), Poisson weights, --boot replicates (1000 by default).
"""

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np

from . import playknow as K
from .playlog import iter_runs

ROUND_ORDER = ["1", "2", "3", "4", "5", "6", "7+", "N"]
NO_HINT_TYPE = 18
TRIOS = list(combinations(range(6), 3))
PAIRS = list(combinations(range(6), 2))
MOVE_ACTION, SWITCH_ACTION = 1, 2          # TurnEvents actions (pybattle.view.ACTIONS)
WEATHER = {"rain": 1, "sun": 2, "sandstorm": 3, "hail": 4}


def rlabel(challenge, noland):
    if noland:
        return "N"
    return str(challenge + 1) if challenge < 6 else "7+"


def entropy(p):
    p = np.asarray([x for x in p if x > 0], np.float64)
    if not len(p):
        return 0.0
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())


# ---- bootstrap ------------------------------------------------------------------------------------------------------

class Boot:
    """Cluster (run) bootstrap with Poisson(1) weights per run and replicate."""

    def __init__(self, n_clusters, B=1000, seed=0):
        self.C, self.B = max(int(n_clusters), 1), B
        self.W = np.random.default_rng(seed).poisson(1.0, (B, self.C)).astype(np.float64)

    def _sums(self, x, cl):
        x = np.asarray(x, np.float64)
        cl = np.asarray(cl, np.int64)
        return np.bincount(cl, weights=x, minlength=self.C), np.bincount(cl, minlength=self.C).astype(np.float64)

    def mean(self, x, cl):
        """{"m": mean, "lo", "hi": 95 % interval, "n": observations}; None values of x are dropped."""
        x, cl = _clean(x, cl)
        n = len(x)
        if n == 0:
            return {"m": None, "lo": None, "hi": None, "n": 0}
        S, N = self._sums(x, cl)
        with np.errstate(invalid="ignore", divide="ignore"):
            rep = (self.W @ S) / (self.W @ N)
        rep = rep[np.isfinite(rep)]
        lo, hi = (np.percentile(rep, [2.5, 97.5]) if len(rep) else (np.nan, np.nan))
        return {"m": float(np.mean(x)), "lo": _f(lo), "hi": _f(hi), "n": n}

    def diff(self, x, cl, g, strata=None):
        """mean(x | g) - mean(x | not g); with strata: sum_s w_s (mean(x | g, s) - mean(x | not g, s)) over the
        strata with both groups, w_s = observations of s / those of all such strata (a comparison "controlled by"
        the stratum). Bootstrap over runs as mean()."""
        keep = [i for i, v in enumerate(x) if v is not None and g[i] is not None]
        x = np.asarray([float(x[i]) for i in keep], np.float64)
        cl = np.asarray([cl[i] for i in keep], np.int64)
        g = np.asarray([bool(g[i]) for i in keep], bool)
        st = np.asarray([strata[i] for i in keep]) if strata is not None else np.zeros(len(keep), int)
        out_strata, w, reps, pts = [], [], [], []
        for s in sorted(set(st.tolist()), key=str):
            m = st == s
            if (m & g).sum() == 0 or (m & ~g).sum() == 0:
                continue
            S1, N1 = self._sums(x[m & g], cl[m & g])
            S0, N0 = self._sums(x[m & ~g], cl[m & ~g])
            with np.errstate(invalid="ignore", divide="ignore"):
                reps.append((self.W @ S1) / (self.W @ N1) - (self.W @ S0) / (self.W @ N0))
            pts.append(x[m & g].mean() - x[m & ~g].mean())
            w.append(m.sum())
            out_strata.append(s)
        if not w:
            return {"m": None, "lo": None, "hi": None, "n1": int(g.sum()), "n0": int((~g).sum())}
        w = np.asarray(w, np.float64) / sum(w)
        rep = np.sum([wi * r for wi, r in zip(w, reps)], axis=0)
        rep = rep[np.isfinite(rep)]
        lo, hi = np.percentile(rep, [2.5, 97.5]) if len(rep) else (np.nan, np.nan)
        return {"m": float(np.dot(w, pts)), "lo": _f(lo), "hi": _f(hi), "n1": int(g.sum()), "n0": int((~g).sum())}


def _f(x):
    return None if x is None or not np.isfinite(x) else float(x)


def _clean(x, cl):
    keep = [i for i, v in enumerate(x) if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return np.asarray([float(x[i]) for i in keep]), np.asarray([cl[i] for i in keep], np.int64)


# ---- extraction -----------------------------------------------------------------------------------------------------

class Data:
    def __init__(self):
        self.runs, self.rentals, self.swaps, self.battles, self.decs, self.tenure = [], [], [], [], [], []
        self.opp_species = defaultdict(Counter)        # round label -> opponent species seen (player view)
        self.meta = None
        self.n_clusters = 0
        self.truth_battles = []                        # analysis only (true opponent team)


def _team_from_options(kind, cand, own, enemy, option):
    """The team an option leads to (as the player sees it at the decision)."""
    if kind == "rental":
        lead, p = option
        j, k = PAIRS[p]
        return [cand[lead], cand[j], cand[k]], tuple(sorted((lead, j, k)))
    a = int(option)
    if a == 0:
        return list(own), None
    i, j = (a - 1) // 3, (a - 1) % 3
    t = list(own)
    t[i] = enemy[j]
    return t, (i, j)


def _act_tuple(a):
    return tuple(a) if isinstance(a, list) else a


def extract(paths, include_late=True):
    """Logs -> Data (compact rows). A run must have its run_end (complete) and no error."""
    D = Data()
    for run in iter_runs(paths):
        if D.meta is None and run["meta"] is not None:
            D.meta = run["meta"]
        info = run["end"]["info"]
        if info.get("error"):
            continue
        late = bool(info.get("late_start"))
        if late and not include_late:
            continue
        cl = D.n_clusters
        D.n_clusters += 1
        D.runs.append({"cl": cl, "late": late, "start_round": info.get("start_round", 1), "streak": run["end"]["streak"],
                       "wins": info.get("wins"), "battles": info.get("battles"), "end": info.get("end"),
                       "start_streak": info.get("start_streak", 0)})
        _extract_run(D, run, cl, late)
    return D


def _battle_outcomes(run):
    """Per battle index: won, and whether the run then won every remaining battle of that round."""
    bs = run["battles"]
    won = [bool(b["end"] and b["end"]["won"]) for b in bs]
    pos = []
    for b in bs:
        d = (b["decisions"] or b["pre"] or [None])[0]
        pos.append(d["run_info"]["battle_in_challenge"] if d else 0)
    comp = []
    for i in range(len(bs)):
        ok, j = True, i
        while j < len(bs):
            if not won[j]:
                ok = False
                break
            if pos[j] == 6:
                break
            j += 1
        else:
            ok = None                       # the run was capped before the end of the round
        comp.append(ok)
    return won, comp


def _extract_run(D, run, cl, late):
    won, comp = _battle_outcomes(run)
    tenure_open = {}                        # mon key -> {"start", "origin", "species", "role"}
    for bi, b in enumerate(run["battles"]):
        decs = b["decisions"]
        first = decs[0] if decs else None
        for pre in b["pre"]:
            ri = pre["run_info"]
            row = {"cl": cl, "late": late, "round": rlabel(ri["challenge_num"], ri["noland"]),
                   "rnum": ri["challenge_num"] + 1, "noland": ri["noland"], "streak": ri["win_streak"],
                   "pos": ri["battle_in_challenge"] + 1, "next_won": won[bi], "round_completed": comp[bi]}
            if pre["kind"] == "rental":
                _rental_row(D, pre, row, first)
            else:
                _swap_row(D, pre, row, first)
        if first is None:
            continue
        ri = first["run_info"]
        rl = rlabel(ri["challenge_num"], ri["noland"])
        team = [K.own_mon(m) for m in first["view"]["own_party"][:3]]
        # tenure: the team of this battle (own sets as keys)
        keys = [K.mon_key(m) for m in team]
        for k, m in zip(keys, team):
            if k not in tenure_open:
                origin = "rental" if any(p["kind"] == "rental" for p in b["pre"]) else "swap"
                tenure_open[k] = {"cl": cl, "species": m["species"], "role": K.role_of(m), "origin": origin,
                                  "start_round": rl, "battles": 0}
        for k in list(tenure_open):
            if k in keys:
                tenure_open[k]["battles"] += 1
            else:
                t = tenure_open.pop(k)
                t["censored"] = False
                D.tenure.append(t)
        _battle_rows(D, b, cl, late, rl, team, won[bi])
    for t in tenure_open.values():
        t["censored"] = True
        D.tenure.append(t)


def _rental_row(D, rec, row, first):
    v, pol, ex = rec["view"], rec["policy"], rec["extras"]
    cand = [K.own_mon(m) for m in v["candidates"]]
    opts = [tuple(o) for o in pol["legal"]]
    lead, j, k = rec["action"]
    trio = tuple(sorted((lead, j, k)))
    tp = pol.get("trio_probs")
    if not tp:                              # (a log without them: from the option probabilities)
        tp = np.zeros(len(TRIOS))
        for (l, pi), pr in zip(opts, pol["probs"]):
            tp[TRIOS.index(tuple(sorted((l,) + PAIRS[pi])))] += pr
    tp = np.asarray(tp, np.float64)
    p_in = [float(sum(tp[t] for t, tr in enumerate(TRIOS) if i in tr)) for i in range(6)]
    probs = pol["probs"]
    hint = ex.get("hint")
    row.update({"kind": "rental", "cand": cand, "trio": trio, "lead": lead, "trio_probs": tp.tolist(),
                "lead_probs": pol.get("lead_probs"), "p_in": p_in, "value": pol.get("value"),
                "entropy": entropy(probs), "maxprob": max(probs) if probs else None, "hint": hint,
                "options": opts, "choice": pol.get("choice"), "q_plan": ex.get("q_plan"),
                "features": ex.get("features"), "probes": ex.get("probes") or {}, "frontier_ids": v.get("frontier_ids")})
    D.rentals.append(row)


def _swap_row(D, rec, row, first):
    v, pol, ex = rec["view"], rec["policy"], rec["extras"]
    own = [K.own_mon(m) for m in v["own_party"][:3]]
    enemy = [K.seen_mon(m) for m in v["enemy_party"][:3]]
    opts = [int(o) for o in pol["legal"]]
    a = rec["action"]
    chosen = 0 if a is None else 1 + 3 * a[0] + a[1]
    probs = pol["probs"]
    post = None
    if a is not None and first is not None:
        post = K.own_mon(first["view"]["own_party"][a[0]])
    row.update({"kind": "swap", "own": own, "enemy": enemy, "options": opts, "chosen": chosen,
                "chosen_idx": opts.index(chosen), "probs": probs, "value": pol.get("value"), "entropy": entropy(probs),
                "maxprob": max(probs), "hint": ex.get("hint"), "q_plan": ex.get("q_plan"),
                "features": ex.get("features"), "probes": ex.get("probes") or {}, "post": post})
    D.swaps.append(row)


def _foe_frac(view):
    f = view["enemy_party"][view["enemy_active"]["party_index"]]
    return max(f["hp_pixels"], 0) / 48.0


def _threat_types(view):
    """Attacking types the opponent's active Pokemon is known or likely to use: its revealed typed attacking moves
    and its own types (STAB)."""
    f = view["enemy_party"][view["enemy_active"]["party_index"]]
    ts = {K.MOVES[m]["type"] for m in f["revealed_moves"] if m and K.is_typed_attack(m)}
    ts |= set(view["enemy_active"]["types"])
    return sorted(t for t in ts if t != K.TYPE_MYSTERY and t < 18)


def _resist_score(mon, types):
    """Fraction of the threat types the Pokemon resists (x <= 0.5)."""
    if not types:
        return None
    return sum(K.effectiveness(t, mon["types"], mon.get("ability", 0)) < 1 for t in types) / len(types)


_BOOST_STATS = {}
for _n, _s in (("ATTACK_UP", [0]), ("ATTACK_UP_2", [0]), ("DEFENSE_UP", [1]), ("DEFENSE_UP_2", [1]),
               ("SPEED_UP", [2]), ("SPEED_UP_2", [2]), ("SPECIAL_ATTACK_UP", [3]), ("SPECIAL_ATTACK_UP_2", [3]),
               ("SPECIAL_DEFENSE_UP", [4]), ("SPECIAL_DEFENSE_UP_2", [4]), ("ACCURACY_UP", [5]),
               ("ACCURACY_UP_2", [5]), ("EVASION_UP", [6]), ("EVASION_UP_2", [6]), ("MINIMIZE", [6]),
               ("CALM_MIND", [3, 4]), ("DRAGON_DANCE", [0, 2]), ("BULK_UP", [0, 1]), ("COSMIC_POWER", [1, 4]),
               ("CURSE", [0, 1]), ("BELLY_DRUM", [0]), ("DEFENSE_CURL", [1])):
    _BOOST_STATS[K.eff_id(_n)] = _s


def _battle_rows(D, b, cl, late, rl, team, won):
    decs = b["decisions"]
    end = b["end"]["info"] if b["end"] else {}
    hint = decs[0]["view"].get("hint_type", NO_HINT_TYPE)
    ri = decs[0]["run_info"]
    flags = Counter()
    seen = set()
    maxprobs, values = [], []
    prev_enemy_idx, prev_spikes = None, 0
    for i, rec in enumerate(decs):
        nxt = decs[i + 1] if i + 1 < len(decs) else None
        row = _decision_row(rec, nxt, cl, late, rl, won, b["end"])
        D.decs.append(row)
        v = rec["view"]
        for m in v["enemy_party"]:
            if m["seen"]:
                seen.add(m["species"])
        maxprobs.append(row["maxprob"])
        values.append(row["value"])
        # strategy flags of the battle (what our side chose)
        mv = row.get("move")
        weather = v["weather"]
        if mv:
            e = K.MOVES[mv]["effect"]
            g = K.status_group(mv)
            if g:
                flags["status_" + g] += 1
                flags["status_any"] += 1
            if e in K.BOOST:
                flags["boost_any"] += 1
                flags["boost_off" if e in K.OFFENSIVE_BOOST else "boost_def"] += 1
            if e == K.EXPLOSION_EFFECT:
                flags["explosion"] += 1
            if e == K.OHKO_EFFECT:
                flags["ohko"] += 1
            if e == K.eff_id("RAIN_DANCE"):
                flags["rain_used"] += 1
            if e == K.eff_id("SUNNY_DAY"):
                flags["sun_used"] += 1
            if e == K.eff_id("SANDSTORM"):
                flags["sand_used"] += 1
            if e == K.eff_id("HAIL"):
                flags["hail_used"] += 1
            mt = row.get("move_type")
            if weather == WEATHER["rain"] and (mt == K.WATER or e == K.eff_id("THUNDER")) and K.is_attack(mv):
                flags["rain_payoff"] += 1
            if weather == WEATHER["sun"] and (mt == K.FIRE or e == K.eff_id("SOLAR_BEAM") or mv == K.WEATHER_BALL) \
                    and K.is_attack(mv):
                flags["sun_payoff"] += 1
            if e == K.BATON_PASS_EFFECT:
                flags["baton_pass_used"] += 1
                st = v["own_active"]
                if sum(x for x in st["stat_stages"] if x > 0) > 0 or st["substitute"]:
                    flags["baton_pass_payoff"] += 1
            if e in (K.eff_id("REFLECT"), K.eff_id("LIGHT_SCREEN")):
                flags["screens_used"] += 1
            if e == K.eff_id("SPIKES"):
                flags["spikes_used"] += 1
            if e == K.eff_id("HEAL_BELL"):
                flags["cleric_used"] += 1
                if any(m["status"] for m in v["own_party"][:3] if m["hp"] > 0):
                    flags["cleric_payoff"] += 1
        if v["own_side"]["reflect_turns"] or v["own_side"]["light_screen_turns"]:
            flags["screens_up"] += 1
        a_own = v["own_active"]["party_index"]
        if weather == WEATHER["sandstorm"]:
            if team[a_own]["ability"] == K.ability_id("SAND_STREAM"):
                flags["sand_used"] += 1
            if not any(t in (K.ROCK, K.GROUND, K.STEEL) for t in v["enemy_active"]["types"]):
                flags["sand_payoff"] += 1
        if weather == WEATHER["hail"] and K.ICE not in v["enemy_active"]["types"]:
            flags["hail_payoff"] += 1
        if weather == WEATHER["rain"] and team[a_own]["ability"] == K.ability_id("SWIFT_SWIM"):
            flags["rain_payoff"] += 1
        eidx = v["enemy_active"]["party_index"]
        if prev_enemy_idx is not None and eidx != prev_enemy_idx and prev_spikes > 0:
            flags["spikes_payoff"] += 1
        prev_enemy_idx, prev_spikes = eidx, v["enemy_side"]["spikes"]
        if row["kind"] == "battle" and row["action_kind"] == "switch":
            flags["switch"] += 1
    D.opp_species[rl].update(seen)
    hp = end.get("own_hp") or []
    hp_frac = sum(h for h, _ in hp) / max(sum(m for _, m in hp), 1) if hp and won else 0.0
    prof = K.team_profile(team)
    row = {"cl": cl, "late": late, "round": rl, "rnum": ri["challenge_num"] + 1, "pos": ri["battle_in_challenge"] + 1,
           "noland": ri["noland"], "streak": ri["win_streak"], "won": won, "turns": end.get("turns"),
           "decisions": end.get("decisions", len(decs)), "hp_frac": hp_frac, "events": end.get("events", {}),
           "luck": end.get("luck_balance", 0), "truncated": end.get("truncated", False), "hint": hint,
           "team": [m["species"] for m in team], "lead": team[decs[0]["view"]["own_active"]["party_index"]]["species"],
           "value0": values[0], "maxprob_mean": float(np.mean([x for x in maxprobs if x is not None])),
           "flags": dict(flags), "profile": prof}
    D.battles.append(row)
    truth = (b["end"] or {}).get("truth") or {}
    if truth.get("opp_team"):
        D.truth_battles.append({"cl": cl, "round": rl, "won": won, "opp": [m["species"] for m in truth["opp_team"]],
                                "opp_sets": [tuple(m.get("sets") or []) for m in truth["opp_team"]]})


def _decision_row(rec, nxt, cl, late, rl, won, bend):
    v, pol, ex = rec["view"], rec["policy"], rec["extras"]
    probs = pol["probs"]
    legal = pol["legal"]
    pl = [probs[i] for i in legal]
    a = rec["action"]
    ak = a[0]
    a_own = v["own_active"]["party_index"]
    me = K.own_mon(v["own_party"][a_own])
    foe_d = v["enemy_party"][v["enemy_active"]["party_index"]]
    foe_types = v["enemy_active"]["types"]
    foe_ab = K.known_ability_of(foe_d["species"], foe_d.get("revealed_ability", 0)) if foe_d["species"] else 0
    own_hp = v["own_party"][a_own]["hp"] / max(v["own_party"][a_own]["max_hp"], 1)
    foe_frac = _foe_frac(v)
    est = ex.get("estimates") or {}
    rows = est.get("own_moves") or []
    threats = est.get("threats") or []
    speed = est.get("speed") or []
    row = {"cl": cl, "late": late, "round": rl, "won": won, "kind": rec["kind"], "turn": v["turn"],
           "entropy": entropy(pl), "maxprob": max(pl) if pl else None, "value": pol.get("value"),
           "n_legal": len(legal), "action_kind": ak, "own_hp": own_hp, "foe_hp": foe_frac,
           "own_status": v["own_party"][a_own]["status"], "foe_status": foe_d["status"],
           "foe_boost": sum(x for x in v["enemy_active"]["stat_stages"][:5] if x > 0),
           "own_boost": sum(x for x in v["own_active"]["stat_stages"][:5] if x > 0),
           "threat": threats[a_own][0] if a_own < len(threats) else None,
           "threat_ko": threats[a_own][1] if a_own < len(threats) else None,
           "sure_faster": speed[a_own][0] if a_own < len(speed) else None,
           "alive": sum(m["hp"] > 0 for m in v["own_party"][:3]),
           "switch_avail": bool(v["switch_targets"]) and rec["kind"] == "battle"}
    if "search" in ex:
        net_a = int(np.argmax([probs[i] if i in legal else -1 for i in range(len(probs))]))
        row["search_agree"] = net_a == pol.get("action_index")
        vis = ex["search"]["visits"]
        tot = sum(vis)
        row["search_maxshare"] = max(vis) / tot if tot else None
    # what happened next (player view): our Pokemon fainted this turn / the opponent's status changed
    nv = nxt["view"] if nxt is not None else None
    row["fainted_next"] = (bool(nv["last_turn"]["own_fainted"]) if nv is not None else (not won and
                           bend is not None))
    th = _threat_types(v)
    row["disadvantage"] = any(K.effectiveness(t, me["types"], me.get("ability", 0)) > 1 for t in th)
    if ak == "move":
        slot = a[1]
        mv = v["own_party"][a_own]["moves"][slot]
        row["move"] = mv
        if mv:
            mi = K.MOVES[mv]
            mt = K.move_type(mv, me)
            row.update({"move_type": mt, "power": mi["power"], "acc": mi["accuracy"], "prio": mi["priority"],
                        "eff": K.effectiveness(mt, foe_types, foe_ab) if K.is_typed_attack(mv) else None,
                        "stab": mt in me["types"], "attack": K.is_attack(mv), "status": K.is_status_move(mv),
                        "effect": mi["effect"], "sec": mi["secondary_chance"] > 0 or mi["effect"] == K.HIGH_CRIT_EFFECT,
                        "contact": bool(mi["flags"] & K.FLAG_CONTACT)})
            g = K.status_group(mv)
            if g:
                row["status_group"] = g
                row["status_fail"] = K.status_failure(mv, foe_types, foe_d["status"], foe_ab,
                                                      v["enemy_active"]["confused"], v["enemy_active"]["leech_seeded"],
                                                      v["enemy_active"]["substitute"],
                                                      v["enemy_side"]["safeguard_turns"] > 0)
                if nv is not None:
                    nf = nv["enemy_party"][v["enemy_active"]["party_index"]]
                    na = nv["enemy_active"]
                    if g in K.MAJOR_STATUS_GROUPS:
                        row["status_landed"] = bool(nf["status"]) and not foe_d["status"]
                    elif g == "confusion":
                        row["status_landed"] = bool(na["confused"]) and not v["enemy_active"]["confused"]
                    elif g == "leech_seed":
                        row["status_landed"] = bool(na["leech_seeded"])
                    elif g == "yawn":
                        row["status_landed"] = bool(na["yawn"])
            if mi["effect"] in _BOOST_STATS:
                st = [v["own_active"]["stat_stages"][i] for i in _BOOST_STATS[mi["effect"]]]
                row["boost_capped"] = all(x >= 6 for x in st)
    # the alternatives among our usable moves (damage estimates: fractions of the opponent's max HP)
    usable = v["usable_moves"]
    moves = v["own_party"][a_own]["moves"]
    alts = []
    for j in range(4):
        m = moves[j] if j < len(moves) else 0
        if not m or not usable[j] or rec["kind"] != "battle":
            continue
        lo, hi, _ = rows[j] if j < len(rows) else (0, 0, 0)
        mi = K.MOVES[m]
        mt = K.move_type(m, me)
        alts.append({"slot": j, "move": m, "lo": lo, "hi": hi, "mean": (lo + hi) / 2, "acc": mi["accuracy"],
                     "attack": K.is_attack(m), "typed": K.is_typed_attack(m),
                     "eff": K.effectiveness(mt, foe_types, foe_ab) if K.is_typed_attack(m) else None,
                     "stab": mt in me["types"], "prio": mi["priority"], "effect": mi["effect"],
                     "sec": mi["secondary_chance"] > 0 or mi["effect"] == K.HIGH_CRIT_EFFECT,
                     "contact": bool(mi["flags"] & K.FLAG_CONTACT)})
    row["alts"] = alts
    row["max_damage_move"] = est.get("max_damage_move", -1)
    row["chosen_slot"] = a[1] if ak == "move" else None
    row["foe_contact_punish"] = bool(foe_d["species"]) and any(ab in K.CONTACT_PUNISH for ab in
                                                               (foe_d.get("possible_abilities") or []))
    row["foe_species"] = foe_d["species"]
    # switches: whom it brings in
    if ak == "switch" or rec["kind"] == "forced_switch":
        targets = [t for t in v["switch_targets"]]
        tgt = a[1]
        row["switch_target"] = tgt
        row["threat_types"] = th
        row["resist_chosen"] = _resist_score(K.own_mon(v["own_party"][tgt]), th)
        others = [_resist_score(K.own_mon(v["own_party"][t]), th) for t in targets if t != tgt]
        others = [x for x in others if x is not None]
        row["resist_others"] = float(np.mean(others)) if others else None
        row["resist_best"] = (row["resist_chosen"] is not None and (not others or row["resist_chosen"] >= max(others)))
        row["n_targets"] = len(targets)
    return row


# ---- analysis -------------------------------------------------------------------------------------------------------

def per_round(boot, rows, fn, rounds=ROUND_ORDER, extra_all=True):
    """{round label: CI of mean(fn(row))} + "all" (fn returning None drops the row)."""
    out = {}
    for r in rounds:
        rr = [x for x in rows if x["round"] == r]
        vals = [fn(x) for x in rr]
        if any(v is not None for v in vals):
            out[r] = boot.mean(vals, [x["cl"] for x in rr])
    if extra_all:
        vals = [fn(x) for x in rows]
        out["all"] = boot.mean(vals, [x["cl"] for x in rows])
    return out


def _bool(x):
    return None if x is None else float(bool(x))


def analyze(D, B=1000, min_n=30):
    boot = Boot(D.n_clusters, B)
    R = {"meta": D.meta.get("meta") if D.meta else {}, "counts": {
        "runs": len(D.runs), "late_runs": sum(r["late"] for r in D.runs), "battles": len(D.battles),
        "rentals": len(D.rentals), "swaps": len(D.swaps), "battle_decisions": len(D.decs)}}
    R["H_outcomes"] = section_outcomes(D, boot)
    R["A_rental"] = section_rental(D, boot, min_n)
    R["B_swaps"] = section_swaps(D, boot, min_n)
    R["C_synergies"] = section_synergies(D, boot)
    R["D_influence"] = section_influence(D, boot)
    R["E_status_boost"] = section_status(D, boot, min_n)
    R["F_moves"] = section_moves(D, boot)
    R["G_switching"] = section_switching(D, boot)
    R["H_confidence"] = section_confidence(D, boot)
    R["truth"] = truth_section(D, boot, min_n)
    return R


# ---- H (outcomes first: the reference numbers) ---------------------------------------------------------------------

def section_outcomes(D, boot):
    out = {}
    B = D.battles
    out["battle_win"] = per_round(boot, B, lambda b: float(b["won"]))
    out["battle_win_by_pos"] = {str(p): boot.mean([float(b["won"]) for b in B if b["pos"] == p and b["round"] != "N"],
                                                  [b["cl"] for b in B if b["pos"] == p and b["round"] != "N"])
                                for p in range(1, 8)}
    runs1 = [r for r in D.runs if not r["late"] and r["start_round"] == 1]
    out["streak"] = boot.mean([r["streak"] for r in runs1], [r["cl"] for r in runs1])
    out["p_complete_round"] = {}
    for k in range(1, 8):
        reach = [r for r in runs1 if r["streak"] >= 7 * (k - 1)]
        out["p_complete_round"][str(k)] = boot.mean([float(r["streak"] >= 7 * k) for r in reach],
                                                    [r["cl"] for r in reach])
    out["p_reach_round"] = {str(k): float(np.mean([r["streak"] >= 7 * (k - 1) for r in runs1])) if runs1 else None
                            for k in range(1, 9)}
    out["streak_hist"] = Counter(min(r["streak"] // 7, 7) for r in runs1)
    out["turns"] = per_round(boot, B, lambda b: b["turns"])
    out["decisions"] = per_round(boot, B, lambda b: b["decisions"])
    out["hp_margin_wins"] = per_round(boot, [b for b in B if b["won"]], lambda b: b["hp_frac"])
    bad = K_BAD = _bad_events()  # noqa: N806
    out["loss_with_bad_luck"] = per_round(boot, [b for b in B if not b["won"]],
                                          lambda b: float(any(k in bad for k in b["events"])))
    out["win_with_bad_luck"] = per_round(boot, [b for b in B if b["won"]],
                                         lambda b: float(any(k in bad for k in b["events"])))
    out["luck_balance_losses"] = per_round(boot, [b for b in B if not b["won"]], lambda b: b["luck"])
    out["luck_balance_wins"] = per_round(boot, [b for b in B if b["won"]], lambda b: b["luck"])
    ev = defaultdict(lambda: [0, 0])
    nl, nw = sum(not b["won"] for b in B), sum(b["won"] for b in B)
    for b in B:
        for k, n in b["events"].items():
            ev[k][0 if not b["won"] else 1] += n
    out["event_rates"] = {k: {"per_loss": v[0] / max(nl, 1), "per_win": v[1] / max(nw, 1), "bad": k in K_BAD}
                          for k, v in sorted(ev.items(), key=lambda kv: -kv[1][0])}
    return out


def _bad_events():
    from .autopsy import BAD_FOR_US
    return set(BAD_FOR_US)


# ---- A rental ------------------------------------------------------------------------------------------------------

def section_rental(D, boot, min_n):
    out = {}
    rows = D.rentals
    # (1) confidence per Pokemon
    sp = defaultdict(lambda: {"offered": 0, "picked": 0, "p_in": [], "led": 0, "p_lead": [], "cert": [], "cl": [],
                              "picked_v": []})
    for r in rows:
        for i, m in enumerate(r["cand"]):
            s = sp[m["species"]]
            s["offered"] += 1
            picked = i in r["trio"]
            s["picked"] += picked
            s["picked_v"].append(float(picked))
            s["p_in"].append(r["p_in"][i])
            s["cert"].append(abs(2 * r["p_in"][i] - 1))
            s["cl"].append(r["cl"])
            if picked:
                s["led"] += r["lead"] == i
            if r["lead_probs"]:
                s["p_lead"].append(r["lead_probs"][i])
    table = []
    for s, d in sp.items():
        if d["offered"] < min_n:
            continue
        table.append({"species": s, "name": K.species_name(s), "offered": d["offered"],
                      "pick_rate": boot.mean(d["picked_v"], d["cl"]), "p_in": float(np.mean(d["p_in"])),
                      "certainty": float(np.mean(d["cert"])), "picked": d["picked"],
                      "lead_given_picked": d["led"] / d["picked"] if d["picked"] else None,
                      "p_lead": float(np.mean(d["p_lead"])) if d["p_lead"] else None, "role_mode": None})
    table.sort(key=lambda x: -x["pick_rate"]["m"])
    out["species"] = table
    out["never_leads"] = [x for x in table if x["picked"] >= 20 and x["lead_given_picked"] == 0]
    out["mean_p_chosen_trio"] = per_round(boot, rows, lambda r: r["trio_probs"][TRIOS.index(r["trio"])])
    out["entropy"] = per_round(boot, rows, lambda r: r["entropy"])
    # (2) lead vs the hint of the first opponent
    hint_rows = [r for r in rows if r["hint"] and r["hint"][0] != NO_HINT_TYPE]
    def lead_res(r):
        t = r["hint"][0]
        return float(K.effectiveness(t, r["cand"][r["lead"]]["types"], r["cand"][r["lead"]]["ability"]) < 1)
    def team_res(r):
        t = r["hint"][0]
        return float(np.mean([K.effectiveness(t, r["cand"][i]["types"], r["cand"][i]["ability"]) < 1
                              for i in r["trio"]]))
    def lead_se(r):
        return float(K.se_count([r["cand"][r["lead"]]], r["hint"][0]) > 0)
    def team_se(r):
        return float(np.mean([K.se_count([r["cand"][i]], r["hint"][0]) > 0 for i in r["trio"]]))
    out["lead_vs_hint"] = {
        "n": len(hint_rows),
        "lead_resists": boot.mean([lead_res(r) for r in hint_rows], [r["cl"] for r in hint_rows]),
        "team_member_resists": boot.mean([team_res(r) for r in hint_rows], [r["cl"] for r in hint_rows]),
        "lead_minus_team_resists": boot.mean([lead_res(r) - team_res(r) for r in hint_rows],
                                             [r["cl"] for r in hint_rows]),
        "lead_se": boot.mean([lead_se(r) for r in hint_rows], [r["cl"] for r in hint_rows]),
        "team_member_se": boot.mean([team_se(r) for r in hint_rows], [r["cl"] for r in hint_rows]),
        "lead_minus_team_se": boot.mean([lead_se(r) - team_se(r) for r in hint_rows], [r["cl"] for r in hint_rows]),
    }
    out["lead_speed_minus_team"] = per_round(boot, rows, lambda r: r["cand"][r["lead"]]["base"][3] - np.mean(
        [r["cand"][i]["base"][3] for i in r["trio"]]))
    # (3) chosen vs offered
    attrs = {
        "bst": lambda m: K.bst(m), "base_hp": lambda m: m["base"][0], "base_atk": lambda m: m["base"][1],
        "base_def": lambda m: m["base"][2], "base_spe": lambda m: m["base"][3], "base_spa": lambda m: m["base"][4],
        "base_spd": lambda m: m["base"][5], "speed_stat": lambda m: m["stats"][2],
        "phys_bulk_stat": lambda m: K.stat_bulk(m)[0], "spec_bulk_stat": lambda m: K.stat_bulk(m)[1],
        "status_moves": lambda m: K.n_status_moves(m), "attack_moves": lambda m: len(K.attack_types(m)),
    }
    out["chosen_vs_offered"] = {}
    for name, fn in attrs.items():
        out["chosen_vs_offered"][name] = {
            "chosen": per_round(boot, rows, lambda r: np.mean([fn(r["cand"][i]) for i in r["trio"]])),
            "offered": per_round(boot, rows, lambda r: np.mean([fn(m) for m in r["cand"]])),
            "diff": per_round(boot, rows, lambda r: np.mean([fn(r["cand"][i]) for i in r["trio"]])
                              - np.mean([fn(m) for m in r["cand"]]))}
    items = defaultdict(lambda: [0, 0])
    for r in rows:
        for i, m in enumerate(r["cand"]):
            items[m["item"]][0] += 1
            items[m["item"]][1] += i in r["trio"]
    out["items"] = sorted([{"item": it, "name": K.item_name(it), "offered": n, "pick_rate": p / n}
                           for it, (n, p) in items.items() if n >= min_n], key=lambda x: -x["pick_rate"])
    # roles
    role_ch, role_of = Counter(), Counter()
    comp = Counter()
    for r in rows:
        roles = [K.role_of(m) for m in r["cand"]]
        role_of.update(roles)
        ch = [roles[i] for i in r["trio"]]
        role_ch.update(ch)
        comp[tuple(sorted(ch))] += 1
    n_ch, n_of = sum(role_ch.values()), sum(role_of.values())
    out["roles"] = {ro: {"chosen_share": role_ch[ro] / max(n_ch, 1), "offered_share": role_of[ro] / max(n_of, 1),
                         "lift": (role_ch[ro] / max(n_ch, 1)) / (role_of[ro] / n_of) if role_of[ro] else None}
                    for ro in K.ROLES}
    out["roles_by_round"] = {ro: per_round(boot, rows, lambda r, ro=ro: sum(K.role_of(r["cand"][i]) == ro
                                                                            for i in r["trio"]) / 3)
                             for ro in K.ROLES}
    out["role_compositions"] = [{"roles": list(k), "share": n / max(len(rows), 1)} for k, n in comp.most_common(10)]
    return out


# ---- B swaps --------------------------------------------------------------------------------------------------------

COVERAGE_KEYS = ("resisted", "unresisted", "shared_weak", "resisted_phys", "resisted_spec", "unresisted_phys",
                 "unresisted_spec", "shared_weak_phys", "shared_weak_spec", "se", "se_phys", "se_spec",
                 "phys_bulk", "spec_bulk")
LOWER_IS_BETTER = {"unresisted", "shared_weak", "unresisted_phys", "unresisted_spec", "shared_weak_phys",
                   "shared_weak_spec"}


def _option_profiles(r):
    """Team profiles of every option of a tactician decision (as the player sees them), and the chosen one's index."""
    if r["kind"] == "rental":
        profs, keys = {}, []
        for t in TRIOS:
            if len({r["cand"][i]["species"] for i in t}) == 3:
                profs[t] = K.team_profile([r["cand"][i] for i in t])
        return profs, r["trio"]
    profs = {}
    for o in r["options"]:
        team, _ = _team_from_options("swap", None, r["own"], r["enemy"], o)
        profs[o] = K.team_profile(team)
    return profs, r["chosen"]


def _percentile(chosen, vals, lower_better):
    """Share of the options the chosen one beats (ties count half)."""
    vals = [v for v in vals if v is not None]
    if chosen is None or not vals:
        return None
    if lower_better:
        return float(np.mean([(v > chosen) + 0.5 * (v == chosen) for v in vals]))
    return float(np.mean([(v < chosen) + 0.5 * (v == chosen) for v in vals]))


def _profile_cache(rows):
    for r in rows:
        if "_profs" not in r:
            r["_profs"] = _option_profiles(r)
    return rows


def section_swaps(D, boot, min_n):
    out = {}
    S = _profile_cache(D.swaps)
    _profile_cache(D.rentals)
    out["swap_rate"] = per_round(boot, S, lambda r: float(r["chosen"] != 0))
    out["entropy"] = per_round(boot, S, lambda r: r["entropy"])
    # (1) who goes out / in
    out_c, in_c, on_team, offered = Counter(), Counter(), Counter(), Counter()
    trans = Counter()
    for r in S:
        for m in r["own"]:
            on_team[m["species"]] += 1
        for m in r["enemy"]:
            offered[m["species"]] += 1
        if r["chosen"]:
            i, j = (r["chosen"] - 1) // 3, (r["chosen"] - 1) % 3
            out_c[r["own"][i]["species"]] += 1
            in_c[r["enemy"][j]["species"]] += 1
            inc = r["post"] or r["enemy"][j]
            trans[(K.role_of(r["own"][i]), K.role_of(inc))] += 1
    out["swapped_out"] = sorted([{"species": s, "name": K.species_name(s), "n": n, "on_team": on_team[s],
                                  "rate": n / on_team[s]} for s, n in out_c.items() if on_team[s] >= min_n],
                                key=lambda x: -x["rate"])
    out["swapped_in"] = sorted([{"species": s, "name": K.species_name(s), "n": n, "offered": offered[s],
                                 "rate": n / offered[s]} for s, n in in_c.items() if offered[s] >= min_n],
                               key=lambda x: -x["rate"])
    out["most_swapped_out"] = sorted(out["swapped_out"], key=lambda x: -x["n"])[:15]
    out["most_swapped_in"] = sorted(out["swapped_in"], key=lambda x: -x["n"])[:15]
    tot = sum(trans.values())
    out["role_transitions"] = {f"{a}->{b}": n / max(tot, 1) for (a, b), n in trans.most_common()}
    out["same_role_share"] = sum(n for (a, b), n in trans.items() if a == b) / max(tot, 1)
    ten = defaultdict(list)
    for t in D.tenure:
        ten[t["species"]].append(t)
    out["tenure_by_origin"] = {o: {"mean_battles": float(np.mean([t["battles"] for t in D.tenure if t["origin"] == o]))
                                   if any(t["origin"] == o for t in D.tenure) else None,
                                   "n": sum(t["origin"] == o for t in D.tenure),
                                   "censored_share": float(np.mean([t["censored"] for t in D.tenure
                                                                    if t["origin"] == o]))
                                   if any(t["origin"] == o for t in D.tenure) else None}
                               for o in ("rental", "swap")}
    out["tenure_by_role"] = {ro: float(np.mean([t["battles"] for t in D.tenure if t["role"] == ro]))
                             for ro in K.ROLES if any(t["role"] == ro for t in D.tenure)}
    tl = [{"species": s, "name": K.species_name(s), "n": len(v), "mean_battles": float(np.mean([t["battles"] for t in v])),
           "uncensored_mean": float(np.mean([t["battles"] for t in v if not t["censored"]]))
           if any(not t["censored"] for t in v) else None} for s, v in ten.items() if len(v) >= min_n]
    out["tenure_species"] = sorted(tl, key=lambda x: -x["mean_battles"])
    # (2-3) coverage: before / after each swap, chosen vs the alternatives (rentals too)
    out["coverage"] = {}
    for kind, rows in (("rental", D.rentals), ("swap", S)):
        cov = {}
        for key in COVERAGE_KEYS:
            lb = key in LOWER_IS_BETTER
            def chosen(r, key=key):
                profs, c = r["_profs"]
                return profs[c][key]
            def mean_opt(r, key=key):
                profs, _ = r["_profs"]
                return float(np.mean([p[key] for p in profs.values()]))
            def best_opt(r, key=key, lb=lb):
                profs, _ = r["_profs"]
                vals = [p[key] for p in profs.values()]
                return min(vals) if lb else max(vals)
            def pct(r, key=key, lb=lb):
                profs, c = r["_profs"]
                return _percentile(profs[c][key], [p[key] for o, p in profs.items() if o != c], lb)
            cov[key] = {"chosen": per_round(boot, rows, chosen), "options_mean": per_round(boot, rows, mean_opt),
                        "best_option": per_round(boot, rows, best_opt), "percentile": per_round(boot, rows, pct)}
        out["coverage"][kind] = cov
    sw = [r for r in S if r["chosen"]]
    ba = {}
    for key in COVERAGE_KEYS:
        def delta(r, key=key):
            i = (r["chosen"] - 1) // 3
            after = list(r["own"])
            after[i] = r["post"] or r["enemy"][(r["chosen"] - 1) % 3]
            return K.team_profile(after)[key] - K.team_profile(r["own"])[key]
        ba[key] = per_round(boot, sw, delta)
    out["before_after_swap"] = ba
    # (4) physical / special balance and stat-move fit; (5) speed
    bal = {}
    for kind, rows in (("rental", D.rentals), ("swap", S)):
        d = {}
        for key in ("phys_share", "fit", "misfits", "base_speed", "max_base_speed", "min_base_speed", "base_offense",
                    "base_hp", "status_moves"):
            def ch(r, key=key):
                profs, c = r["_profs"]
                return profs[c][key]
            def mo(r, key=key):
                profs, _ = r["_profs"]
                vals = [p[key] for p in profs.values() if p[key] is not None]
                return float(np.mean(vals)) if vals else None
            d[key] = {"chosen": per_round(boot, rows, ch), "options_mean": per_round(boot, rows, mo)}
        arch_ch, arch_of = Counter(), Counter()
        for r in rows:
            profs, c = r["_profs"]
            arch_ch[profs[c]["archetype"]] += 1
            for p in profs.values():
                arch_of[p["archetype"]] += 1 / len(profs)
        n = max(len(rows), 1)
        d["archetype"] = {a: {"chosen": arch_ch[a] / n, "options": arch_of[a] / n} for a in
                          ("fast_frail", "mixed", "bulky_slow")}
        bal[kind] = d
    out["balance_speed"] = bal
    out["speed_tiers"] = _speed_tiers(D, boot)
    out["profile_correlations"] = _profile_correlations(D)
    # (6) does not swap although W / planner / stats say a swap is better
    out["keep_vs_better"] = _keep_vs_better(S, boot)
    # (7) quality by outcome: swap vs keep in comparable states
    out["swap_outcome"] = _swap_outcome(S, boot)
    return out


def _speed_tiers(D, boot):
    """Per chosen rental team: mean over its members of the share of the round's opponents (the species seen in this
    log's battles of that round, weighted by how often) that the member surely outspeeds (its Speed stat above the
    opponent's maximum possible Speed: any IV / EV / nature at level 100); vs the 6 offered."""
    from .damage import stat_range_any_iv
    tiers = {}
    for r, c in D.opp_species.items():
        tot = sum(c.values())
        tiers[r] = [(stat_range_any_iv(K.SPECIES[s]["base"][3])[1], n / tot) for s, n in c.items() if s]

    def outspeed(m, r):
        t = tiers.get(r) or []
        return sum(w for mx, w in t if m["stats"][2] > mx)
    rows = [r for r in D.rentals if r["round"] in tiers]
    return {"chosen": per_round(boot, rows, lambda r: np.mean([outspeed(r["cand"][i], r["round"]) for i in r["trio"]])),
            "offered": per_round(boot, rows, lambda r: np.mean([outspeed(m, r["round"]) for m in r["cand"]])),
            "chosen_max": per_round(boot, rows, lambda r: max(outspeed(r["cand"][i], r["round"]) for i in r["trio"])),
            "offered_best3_max": per_round(boot, rows, lambda r: max(outspeed(m, r["round"]) for m in r["cand"]))}


def _profile_correlations(D):
    """Spearman correlations between team means (speed, physical / special bulk, offense, HP) over battle teams."""
    keys = ("base_speed", "phys_bulk", "spec_bulk", "base_offense", "base_hp")
    X = np.array([[b["profile"][k] for k in keys] for b in D.battles], np.float64)
    if len(X) < 3:
        return {}
    Rk = np.argsort(np.argsort(X, 0), 0).astype(np.float64)
    C = np.corrcoef(Rk.T)
    return {f"{a}~{b}": float(C[i, j]) for i, a in enumerate(keys) for j, b in enumerate(keys) if i < j}


def _swap_signals(r):
    """Planner / W / base stats: (best swap minus keep) for Q_plan, W(normal | hint), and BST(in) - BST(out)."""
    opts = r["options"]
    q = r["q_plan"] or []
    f = r["features"] or []
    k = opts.index(0)
    swaps = [i for i, o in enumerate(opts) if o != 0]
    if not swaps:
        return None
    dq = max(q[i] for i in swaps) - q[k] if q else None
    dw = max(f[i][1] for i in swaps) - f[k][1] if f else None
    dbst = max(K.bst(r["enemy"][(opts[i] - 1) % 3]) - K.bst(r["own"][(opts[i] - 1) // 3]) for i in swaps)
    ch = r["chosen_idx"]
    dq_ch = (q[ch] - q[k]) if q else None
    return {"dq": dq, "dw": dw, "dbst": dbst, "dq_chosen": dq_ch, "planner_argmax": int(np.argmax(q)) == ch if q else None}


def _keep_vs_better(S, boot, q_margin=0.5, w_margin=0.02, bst_margin=40):
    for r in S:
        r["_sig"] = _swap_signals(r)
    rows = [r for r in S if r["_sig"]]
    keeps = [r for r in rows if r["chosen"] == 0]
    out = {"thresholds": {"q_margin": q_margin, "w_margin": w_margin, "bst_margin": bst_margin}}
    out["keep_share"] = per_round(boot, rows, lambda r: float(r["chosen"] == 0))
    out["keep_though_planner"] = per_round(boot, keeps, lambda r: _bool(r["_sig"]["dq"] > q_margin))
    out["keep_though_w"] = per_round(boot, keeps, lambda r: _bool(r["_sig"]["dw"] > w_margin))
    out["keep_though_bst"] = per_round(boot, keeps, lambda r: _bool(r["_sig"]["dbst"] >= bst_margin))
    out["swap_though_planner_keep"] = per_round(boot, [r for r in rows if r["chosen"]],
                                                lambda r: _bool(r["_sig"]["dq_chosen"] < -q_margin))
    out["agree_planner_argmax"] = per_round(boot, rows, lambda r: _bool(r["_sig"]["planner_argmax"]))
    # outcome where the planner said "swap": kept vs swapped (controlled by round)
    ps = [r for r in rows if r["_sig"]["dq"] > q_margin]
    out["planner_says_swap_outcome"] = boot.diff([float(r["next_won"]) for r in ps], [r["cl"] for r in ps],
                                                 [r["chosen"] == 0 for r in ps], [r["round"] for r in ps])
    out["planner_says_swap_n"] = len(ps)
    return out


def _swap_outcome(S, boot):
    rows = [r for r in S if r["q_plan"]]
    if not rows:
        return {}
    qk = np.array([r["q_plan"][r["options"].index(0)] for r in rows])
    rounds = np.array([r["round"] for r in rows])
    cuts = {rd: np.percentile(qk[rounds == rd], [33.3, 66.7]) for rd in set(rounds.tolist())}
    strata = [f"{r['round']}|{int(np.searchsorted(cuts[r['round']], q))}" for r, q in zip(rows, qk)]
    g = [r["chosen"] != 0 for r in rows]
    return {"next_battle": boot.diff([float(r["next_won"]) for r in rows], [r["cl"] for r in rows], g, strata),
            "round_completed": boot.diff([None if r["round_completed"] is None else float(r["round_completed"])
                                          for r in rows], [r["cl"] for r in rows], g, strata),
            "next_battle_by_round": {rd: boot.diff([float(r["next_won"]) for r in rows if r["round"] == rd],
                                                   [r["cl"] for r in rows if r["round"] == rd],
                                                   [r["chosen"] != 0 for r in rows if r["round"] == rd],
                                                   [s for s, r in zip(strata, rows) if r["round"] == rd])
                                     for rd in ROUND_ORDER if any(r["round"] == rd for r in rows)}}


# ---- C synergies ----------------------------------------------------------------------------------------------------

def section_synergies(D, boot):
    out = {"presence": {}, "usage": {}}
    for kind, rows in (("rental", _profile_cache(D.rentals)), ("swap", _profile_cache(D.swaps))):
        pres = {}
        for s in K.SYNERGIES:
            key = "syn_" + s
            def ch(r, key=key):
                profs, c = r["_profs"]
                return float(profs[c][key])
            def base(r, key=key):
                profs, _ = r["_profs"]
                return float(np.mean([p[key] for p in profs.values()]))
            avail = [r for r in rows if base(r) > 0]
            c = boot.mean([ch(r) for r in avail], [r["cl"] for r in avail])
            b = boot.mean([base(r) for r in avail], [r["cl"] for r in avail])
            pres[s] = {"available_share": len(avail) / max(len(rows), 1), "chosen": c, "baseline": b,
                       "lift": (c["m"] / b["m"]) if c["m"] is not None and b["m"] else None,
                       "chosen_minus_baseline": boot.mean([ch(r) - base(r) for r in avail], [r["cl"] for r in avail]),
                       "by_round": per_round(boot, avail, lambda r: ch(r) - base(r))}
        out["presence"][kind] = pres
    B = D.battles
    for s in ("rain", "sun", "sand", "hail", "baton_pass", "screens", "spikes", "cleric"):
        have = [b for b in B if b["profile"]["syn_" + s]]
        if not have:
            continue
        out["usage"][s] = {"battles_with": len(have),
                           "used": boot.mean([float(b["flags"].get(s + "_used", 0) > 0) for b in have],
                                             [b["cl"] for b in have]),
                           "payoff": boot.mean([float(b["flags"].get(s + "_payoff", 0) > 0) for b in have],
                                               [b["cl"] for b in have]) if s != "screens" else
                           boot.mean([float(b["flags"].get("screens_up", 0) > 0) for b in have],
                                     [b["cl"] for b in have])}
    return out


# ---- D hint / features influence --------------------------------------------------------------------------------------

def _probe_team(r, choice):
    if r["kind"] == "rental":
        team, trio = _team_from_options("rental", r["cand"], None, None, r["options"][choice])
        return team, trio
    o = r["options"][choice]
    team, _ = _team_from_options("swap", None, r["own"], r["enemy"], o)
    return team, o


def section_influence(D, boot):
    out = {}
    for kind, rows in (("rental", D.rentals), ("swap", D.swaps)):
        base_choice = (lambda r: r["choice"]) if kind == "rental" else (lambda r: r["chosen_idx"])
        res = {}
        for probe in ("hint_none_input", "hint_none_full", "hint_t_input", "hint_t_full", "feat_zero"):
            ch, team_ch, d_res, d_se, d_q, cl, rnd = [], [], [], [], [], [], []
            for r in rows:
                pr = r["probes"]
                names = [n for n in pr if (n == probe) or (probe.startswith("hint_t") and n.startswith("hint_")
                                                           and not n.startswith("hint_none")
                                                           and n.endswith(probe.split("_")[-1]))]
                for n in names:
                    c = pr[n]["choice"]
                    b = base_choice(r)
                    tb, kb = _probe_team(r, b)
                    tp, kp = _probe_team(r, c)
                    ch.append(float(c != b))
                    team_ch.append(float(kb != kp))
                    cl.append(r["cl"])
                    rnd.append(r["round"])
                    if probe.startswith("hint_none"):
                        t = r["hint"][0] if r["hint"] else NO_HINT_TYPE
                        if t != NO_HINT_TYPE:
                            d_res.append(K.resist_count(tb, t) - K.resist_count(tp, t))
                            d_se.append(K.se_count(tb, t) - K.se_count(tp, t))
                        else:
                            d_res.append(None)
                            d_se.append(None)
                    elif probe.startswith("hint_t"):
                        t = int(n.split("_")[1])
                        d_res.append(K.resist_count(tp, t) - K.resist_count(tb, t))
                        d_se.append(K.se_count(tp, t) - K.se_count(tb, t))
                    else:
                        d_res.append(None)
                        d_se.append(None)
                    if r["q_plan"]:
                        d_q.append(r["q_plan"][b] - r["q_plan"][c])
                    else:
                        d_q.append(None)
            if not ch:
                continue
            rr = [{"round": x, "cl": c_, "v": v} for x, c_, v in zip(rnd, cl, ch)]
            res[probe] = {"n": len(ch), "changed": boot.mean(ch, cl), "team_changed": boot.mean(team_ch, cl),
                          "changed_by_round": per_round(boot, rr, lambda x: x["v"]),
                          "delta_resist": boot.mean(d_res, cl), "delta_se": boot.mean(d_se, cl),
                          "delta_resist_changed": boot.mean([d for d, c_ in zip(d_res, ch) if c_],
                                                            [x for x, c_ in zip(cl, ch) if c_]),
                          "delta_qplan": boot.mean(d_q, cl),
                          "delta_qplan_changed": boot.mean([d for d, c_ in zip(d_q, ch) if c_],
                                                           [x for x, c_ in zip(cl, ch) if c_])}
        # agreement of the policy with the planner's argmax (with and without the features)
        agree = [float(base_choice(r) == int(np.argmax(r["q_plan"]))) for r in rows if r["q_plan"]]
        agree0 = [float(r["probes"]["feat_zero"]["choice"] == int(np.argmax(r["q_plan"])))
                  for r in rows if r["q_plan"] and "feat_zero" in r["probes"]]
        cl1 = [r["cl"] for r in rows if r["q_plan"]]
        cl0 = [r["cl"] for r in rows if r["q_plan"] and "feat_zero" in r["probes"]]
        res["planner_argmax_agreement"] = boot.mean(agree, cl1)
        res["planner_argmax_agreement_feat_zero"] = boot.mean(agree0, cl0)
        out[kind] = res
    return out


# ---- E status / boost -----------------------------------------------------------------------------------------------

def section_status(D, boot, min_n):
    out = {}
    decs = [d for d in D.decs if d["kind"] == "battle"]
    groups = list(K.STATUS_GROUPS)
    for g in groups:
        avail = [d for d in decs if any(K.status_group(a["move"]) == g for a in d["alts"])]
        used = [d for d in avail if d.get("status_group") == g]
        if not used:
            out[g] = {"available": len(avail), "used": 0}
            continue
        o = {"available": len(avail), "used": len(used),
             "use_rate": per_round(boot, avail, lambda d: float(d.get("status_group") == g)),
             "when_own_hp": boot.mean([d["own_hp"] for d in used], [d["cl"] for d in used]),
             "avail_own_hp": boot.mean([d["own_hp"] for d in avail], [d["cl"] for d in avail]),
             "when_foe_hp": boot.mean([d["foe_hp"] for d in used], [d["cl"] for d in used]),
             "avail_foe_hp": boot.mean([d["foe_hp"] for d in avail], [d["cl"] for d in avail]),
             "when_threat_ko": boot.mean([d["threat_ko"] for d in used], [d["cl"] for d in used]),
             "avail_threat_ko": boot.mean([d["threat_ko"] for d in avail], [d["cl"] for d in avail]),
             "when_foe_statused": boot.mean([float(bool(d["foe_status"])) for d in used], [d["cl"] for d in used]),
             "when_turn": boot.mean([d["turn"] for d in used], [d["cl"] for d in used]),
             "error": boot.mean([float(d.get("status_fail") is not None) for d in used], [d["cl"] for d in used]),
             "error_reasons": dict(Counter(d.get("status_fail") for d in used if d.get("status_fail"))),
             "landed": boot.mean([_bool(d.get("status_landed")) for d in used], [d["cl"] for d in used]),
             "moves": dict(Counter(K.move_name(d["move"]) for d in used).most_common(8))}
        out[g] = o
    # payoff: battles with vs without each strategy (controlled by round)
    out["payoff"] = _battle_feature_payoff(D, boot)
    # boosts
    bo = {}
    for name, effs in (("offensive", K.OFFENSIVE_BOOST), ("defensive", K.DEFENSIVE_BOOST)):
        avail = [d for d in decs if any(a["effect"] in effs for a in d["alts"])]
        used = [d for d in avail if d.get("effect") in effs]
        if not used:
            continue
        bo[name] = {"available": len(avail), "used": len(used),
                    "use_rate": per_round(boot, avail, lambda d: float(d.get("effect") in effs)),
                    "when_own_hp": boot.mean([d["own_hp"] for d in used], [d["cl"] for d in used]),
                    "avail_own_hp": boot.mean([d["own_hp"] for d in avail], [d["cl"] for d in avail]),
                    "when_threat_ko": boot.mean([d["threat_ko"] for d in used], [d["cl"] for d in used]),
                    "avail_threat_ko": boot.mean([d["threat_ko"] for d in avail], [d["cl"] for d in avail]),
                    "when_own_boost": boot.mean([d["own_boost"] for d in used], [d["cl"] for d in used]),
                    "fainted_next": boot.mean([float(d["fainted_next"]) for d in used], [d["cl"] for d in used]),
                    "capped_error": boot.mean([_bool(d.get("boost_capped")) for d in used], [d["cl"] for d in used]),
                    "moves": dict(Counter(K.move_name(d["move"]) for d in used).most_common(10))}
    out["boosts"] = bo
    per_move = Counter()
    per_move_avail = Counter()
    for d in decs:
        for a in d["alts"]:
            if a["effect"] in K.BOOST or K.status_group(a["move"]):
                per_move_avail[a["move"]] += 1
        if d.get("move") and (d.get("effect") in K.BOOST or d.get("status_group")):
            per_move[d["move"]] += 1
    out["per_move"] = sorted([{"move": K.move_name(m), "used": per_move[m], "available": n, "rate": per_move[m] / n}
                              for m, n in per_move_avail.items() if n >= min_n], key=lambda x: -x["rate"])
    return out


def _battle_feature_payoff(D, boot):
    """Win rate of battles with vs without each strategy feature, controlled by round (strata = round)."""
    B = D.battles
    feats = {"status_any": lambda b: b["flags"].get("status_any", 0) > 0,
             "boost_off": lambda b: b["flags"].get("boost_off", 0) > 0,
             "boost_def": lambda b: b["flags"].get("boost_def", 0) > 0,
             "explosion": lambda b: b["flags"].get("explosion", 0) > 0,
             "voluntary_switch": lambda b: b["flags"].get("switch", 0) > 0}
    for g in K.STATUS_GROUPS:
        feats["status_" + g] = lambda b, g=g: b["flags"].get("status_" + g, 0) > 0
    for s in K.SYNERGIES:
        feats["team_" + s] = lambda b, s=s: bool(b["profile"]["syn_" + s])
    for ro in K.ROLES:
        feats["team_has_" + ro] = lambda b, ro=ro: b["profile"]["role_" + ro] > 0
    for a in ("fast_frail", "mixed", "bulky_slow"):
        feats["archetype_" + a] = lambda b, a=a: b["profile"]["archetype"] == a
    out = {}
    for name, fn in feats.items():
        g = [fn(b) for b in B]
        if sum(g) < 10:
            continue
        out[name] = boot.diff([float(b["won"]) for b in B], [b["cl"] for b in B], g, [b["round"] for b in B])
        out[name]["share"] = float(np.mean(g))
    return out


# ---- F move choice -----------------------------------------------------------------------------------------------------

def section_moves(D, boot):
    out = {}
    decs = [d for d in D.decs if d["kind"] == "battle" and d["action_kind"] == "move" and d.get("move")]
    att = [d for d in decs if d.get("attack") and d.get("eff") is not None]
    typed = lambda d: [a for a in d["alts"] if a["typed"]]
    out["move_share"] = per_round(boot, [d for d in D.decs if d["kind"] == "battle"],
                                  lambda d: float(d["action_kind"] == "move"))
    out["attack_share_of_moves"] = per_round(boot, decs, lambda d: float(bool(d.get("attack"))))
    multi = [d for d in att if len(typed(d)) >= 2]
    out["eff_chosen"] = per_round(boot, att, lambda d: d["eff"])
    out["eff_is_best"] = per_round(boot, multi, lambda d: float(d["eff"] >= max(a["eff"] for a in typed(d))))
    out["eff_best_available"] = per_round(boot, multi, lambda d: max(a["eff"] for a in typed(d)))
    out["resisted_when_neutral_available"] = per_round(
        boot, [d for d in multi if max(a["eff"] for a in typed(d)) >= 1],
        lambda d: float(d["eff"] < 1))
    dm = [d for d in decs if sum(a["attack"] for a in d["alts"]) >= 2 and d["max_damage_move"] >= 0]
    out["max_damage_chosen"] = per_round(boot, dm, lambda d: float(d["chosen_slot"] == d["max_damage_move"]))
    # KO: a "sure KO" move = minimum estimated damage >= the opponent's HP bar fraction
    def sure(d):
        return [a for a in d["alts"] if a["attack"] and d["foe_hp"] > 0 and a["lo"] >= d["foe_hp"]]
    ko = [d for d in D.decs if d["kind"] == "battle" and sure(d)]
    out["sure_ko_available_share"] = len(ko) / max(sum(d["kind"] == "battle" for d in D.decs), 1)
    out["sure_ko_taken"] = per_round(boot, ko, lambda d: float(d["action_kind"] == "move" and d["chosen_slot"] in
                                                               [a["slot"] for a in sure(d)]))
    ko2 = [d for d in ko if len({a["acc"] for a in sure(d)}) >= 2 and d["action_kind"] == "move"
           and d["chosen_slot"] in [a["slot"] for a in sure(d)]]
    out["sure_ko_most_accurate"] = per_round(boot, ko2, lambda d: float(
        next(a["acc"] for a in d["alts"] if a["slot"] == d["chosen_slot"]) >= max(a["acc"] for a in sure(d))))
    # accuracy risk: a safer alternative with similar damage (>= 85 % of the chosen move's mean estimate)
    def risky(d):
        c = next((a for a in d["alts"] if a["slot"] == d["chosen_slot"]), None)
        if c is None or not c["attack"] or c["mean"] <= 0:
            return None
        safer = [a for a in d["alts"] if a["attack"] and a["slot"] != c["slot"] and a["acc"] > c["acc"]
                 and a["mean"] >= 0.85 * c["mean"]]
        return float(bool(safer))
    out["unneeded_accuracy_risk"] = per_round(boot, decs, risky)
    out["chosen_accuracy"] = per_round(boot, att, lambda d: d["acc"] if d["acc"] else 100)
    # OHKO / Explosion
    for name, eff in (("ohko", K.OHKO_EFFECT), ("explosion", K.EXPLOSION_EFFECT)):
        avail = [d for d in D.decs if d["kind"] == "battle" and any(a["effect"] == eff for a in d["alts"])]
        used = [d for d in avail if d.get("effect") == eff and d["action_kind"] == "move"]
        out[name] = {"available": len(avail), "used": len(used),
                     "use_rate": per_round(boot, avail, lambda d: float(d.get("effect") == eff
                                                                        and d["action_kind"] == "move")),
                     "own_hp": boot.mean([d["own_hp"] for d in used], [d["cl"] for d in used]),
                     "foe_hp": boot.mean([d["foe_hp"] for d in used], [d["cl"] for d in used]),
                     "alive": boot.mean([d["alive"] for d in used], [d["cl"] for d in used]),
                     "turn": boot.mean([d["turn"] for d in used], [d["cl"] for d in used]),
                     "threat_ko": boot.mean([d["threat_ko"] for d in used], [d["cl"] for d in used])}
    # secondary effects (incl. high critical-hit ratio) among similar-damage alternatives
    def sim(d):
        at = [a for a in d["alts"] if a["attack"] and a["mean"] > 0]
        if len(at) < 2:
            return None
        mx = max(a["mean"] for a in at)
        s = [a for a in at if a["mean"] >= 0.85 * mx]
        if len(s) < 2 or len({a["sec"] for a in s}) < 2:
            return None
        return s
    sec = [(d, sim(d)) for d in decs]
    sec = [(d, s) for d, s in sec if s and d["chosen_slot"] in [a["slot"] for a in s]]
    out["secondary_chosen"] = boot.mean([float(d["sec"]) for d, _ in sec], [d["cl"] for d, _ in sec])
    out["secondary_baseline"] = boot.mean([float(np.mean([a["sec"] for a in s])) for _, s in sec],
                                          [d["cl"] for d, _ in sec])
    # STAB vs coverage
    out["stab_share"] = per_round(boot, att, lambda d: float(d["stab"]))
    nonstab = [d for d in att if not d["stab"] and any(a["stab"] and a["typed"] for a in d["alts"])]
    out["nonstab_more_effective"] = per_round(boot, nonstab, lambda d: float(
        d["eff"] > max(a["eff"] for a in d["alts"] if a["stab"] and a["typed"])))
    # priority when the opponent is low
    low = [d for d in D.decs if d["kind"] == "battle" and 0 < d["foe_hp"] <= 0.25
           and any(a["attack"] and a["prio"] > 0 for a in d["alts"])]
    out["priority_when_foe_low"] = {
        "all": per_round(boot, low, lambda d: float(d.get("prio", 0) > 0 and d["action_kind"] == "move")),
        "not_sure_faster": boot.mean([float(d.get("prio", 0) > 0 and d["action_kind"] == "move") for d in low
                                      if not d["sure_faster"]], [d["cl"] for d in low if not d["sure_faster"]]),
        "sure_faster": boot.mean([float(d.get("prio", 0) > 0 and d["action_kind"] == "move") for d in low
                                  if d["sure_faster"]], [d["cl"] for d in low if d["sure_faster"]])}
    # contact moves against abilities that punish contact
    both = [d for d in att if len({a["contact"] for a in d["alts"] if a["attack"]}) == 2]
    out["contact_vs_punish"] = boot.diff([float(d["contact"]) for d in both], [d["cl"] for d in both],
                                         [d["foe_contact_punish"] for d in both], [d["round"] for d in both])
    out["top_moves"] = Counter(K.move_name(d["move"]) for d in decs).most_common(25)
    return out


# ---- G switching --------------------------------------------------------------------------------------------------------

def section_switching(D, boot):
    out = {}
    decs = [d for d in D.decs if d["kind"] == "battle" and d["switch_avail"]]
    out["switch_rate"] = per_round(boot, decs, lambda d: float(d["action_kind"] == "switch"))
    conds = {"disadvantage": lambda d: d["disadvantage"], "low_hp": lambda d: d["own_hp"] < 1 / 3,
             "foe_boosted": lambda d: d["foe_boost"] >= 2, "own_statused": lambda d: bool(d["own_status"]),
             "threat_ko": lambda d: bool(d["threat_ko"])}
    out["switch_rate_given"] = {}
    for n, fn in conds.items():
        g = [fn(d) for d in decs]
        out["switch_rate_given"][n] = {
            "yes": boot.mean([float(d["action_kind"] == "switch") for d, x in zip(decs, g) if x],
                             [d["cl"] for d, x in zip(decs, g) if x]),
            "no": boot.mean([float(d["action_kind"] == "switch") for d, x in zip(decs, g) if not x],
                            [d["cl"] for d, x in zip(decs, g) if not x]),
            "diff_by_round": boot.diff([float(d["action_kind"] == "switch") for d in decs], [d["cl"] for d in decs],
                                       g, [d["round"] for d in decs])}
    for name, rows in (("voluntary", [d for d in decs if d["action_kind"] == "switch"]),
                       ("forced", [d for d in D.decs if d["kind"] == "forced_switch"])):
        rows = [d for d in rows if d.get("resist_chosen") is not None]
        multi = [d for d in rows if d["n_targets"] >= 2]
        out[name] = {"n": len(rows),
                     "resist_chosen": boot.mean([d["resist_chosen"] for d in rows], [d["cl"] for d in rows]),
                     "resist_others": boot.mean([d["resist_others"] for d in multi], [d["cl"] for d in multi]),
                     "chose_best_resist": boot.mean([float(d["resist_best"]) for d in multi], [d["cl"] for d in multi]),
                     "by_round": per_round(boot, multi, lambda d: float(d["resist_best"]))}
    thr = [d for d in decs if d["threat_ko"]]
    out["threatened"] = {
        "n": len(thr),
        "stayed_and_fainted": boot.mean([float(d["action_kind"] == "move" and d["fainted_next"]) for d in thr],
                                        [d["cl"] for d in thr]),
        "switched": boot.mean([float(d["action_kind"] == "switch") for d in thr], [d["cl"] for d in thr]),
        "stayed_and_fainted_by_round": per_round(boot, thr, lambda d: float(d["action_kind"] == "move"
                                                                            and d["fainted_next"]))}
    return out


# ---- H confidence ---------------------------------------------------------------------------------------------------------

def section_confidence(D, boot):
    out = {}
    for kind, rows in (("rental", D.rentals), ("swap", D.swaps),
                       ("battle", [d for d in D.decs if d["kind"] == "battle" and d["n_legal"] >= 2]),
                       ("forced_switch", [d for d in D.decs if d["kind"] == "forced_switch" and d["n_legal"] >= 2])):
        out[kind] = {"entropy": per_round(boot, rows, lambda r: r["entropy"]),
                     "maxprob": per_round(boot, rows, lambda r: r["maxprob"])}
    # accuracy vs confidence
    for kind, rows, y in (("rental", D.rentals, "next_won"), ("swap", D.swaps, "next_won")):
        mp = np.array([r["maxprob"] for r in rows])
        if len(mp) < 30:
            continue
        cuts = np.percentile(mp, [33.3, 66.7])
        out[kind]["win_by_maxprob_tercile"] = {
            str(t): boot.mean([float(r[y]) for r, m in zip(rows, mp) if np.searchsorted(cuts, m) == t],
                              [r["cl"] for r, m in zip(rows, mp) if np.searchsorted(cuts, m) == t]) for t in range(3)}
        out[kind]["maxprob_terciles"] = cuts.tolist()
    B = D.battles
    mp = np.array([b["maxprob_mean"] for b in B])
    if len(B) >= 30:
        cuts = np.percentile(mp, [33.3, 66.7])
        out["battle"]["win_by_mean_maxprob_tercile"] = {
            str(t): boot.diff([float(b["won"]) for b in B], [b["cl"] for b in B],
                              [np.searchsorted(cuts, m) == t for m in mp], [b["round"] for b in B]) for t in range(3)}
        out["battle"]["mean_maxprob_terciles"] = cuts.tolist()
    # calibration of the battler's value at the first decision of each battle
    cal = []
    v = np.array([b["value0"] if b["value0"] is not None else np.nan for b in B])
    for lo in np.arange(0, 1.0, 0.1):
        m = (v >= lo) & (v < lo + 0.1 + (1e-9 if lo >= 0.9 else 0))
        if m.sum() >= 10:
            cal.append({"bin": [round(lo, 1), round(lo + 0.1, 1)], "n": int(m.sum()), "pred": float(v[m].mean()),
                        "won": float(np.mean([B[i]["won"] for i in np.flatnonzero(m)]))})
    out["value_calibration"] = cal
    sa = [d for d in D.decs if "search_agree" in d and d["n_legal"] >= 2]
    if sa:
        out["search_agreement"] = per_round(boot, sa, lambda d: float(d["search_agree"]))
        mp = np.array([d["maxprob"] for d in sa])
        out["search_agreement_by_net_maxprob"] = {
            f"{lo:.1f}-{lo + 0.2:.1f}": boot.mean([float(d["search_agree"]) for d, m in zip(sa, mp)
                                                   if lo <= m < lo + 0.2 + 1e-9],
                                                  [d["cl"] for d, m in zip(sa, mp) if lo <= m < lo + 0.2 + 1e-9])
            for lo in (0.0, 0.2, 0.4, 0.6, 0.8)}
    return out


# ---- truth (analysis only) -------------------------------------------------------------------------------------------------

def truth_section(D, boot, min_n):
    """FLAGGED: uses the opponent's true team (the record's "truth" field), which the player never sees."""
    T = D.truth_battles
    if not T:
        return {}
    by = defaultdict(lambda: [0, 0, []])
    for t in T:
        for s in set(t["opp"]):
            by[s][0] += 1
            by[s][1] += not t["won"]
    rows = [{"species": s, "name": K.species_name(s), "faced": n, "loss_rate": l / n} for s, (n, l, _) in by.items()
            if n >= min_n]
    rows.sort(key=lambda x: -x["loss_rate"])
    return {"hardest_opponents": rows[:20], "easiest_opponents": rows[-10:]}


# ---- report -------------------------------------------------------------------------------------------------------------

def _ci(d, pct=False, nd=3):
    if not d or d.get("m") is None:
        return "–"
    k = 100 if pct else 1
    f = (lambda x: f"{x * k:.1f}") if pct else (lambda x: f"{x:.{nd}f}")
    if d.get("lo") is None:
        return f(d["m"])
    return f"{f(d['m'])} [{f(d['lo'])}, {f(d['hi'])}]"


def _n(d):
    if not d:
        return ""
    return d.get("n", d.get("n1", ""))


def _round_table(rows, pct=False, nd=3, label="métrica"):
    """rows: [(name, per_round dict)] -> markdown table with the rounds as columns."""
    cols = [r for r in ROUND_ORDER + ["all"] if any(r in (d or {}) for _, d in rows)]
    hdr = "| " + label + " | " + " | ".join("todas" if c == "all" else ("Noland" if c == "N" else f"R{c}")
                                            for c in cols) + " |"
    out = [hdr, "|" + "---|" * (len(cols) + 1)]
    for name, d in rows:
        out.append(f"| {name} | " + " | ".join(_ci((d or {}).get(c), pct, nd) for c in cols) + " |")
    return "\n".join(out)


def _diff(d, pct=False):
    if not d or d.get("m") is None:
        return "–"
    return _ci(d, pct) + f" (n {d.get('n1', '?')} / {d.get('n0', '?')})"


def write_report(R, D, out_dir, title):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(R, f, indent=1, default=lambda o: o if not isinstance(o, (np.floating, np.integer)) else o.item())
    plots = make_plots(R, D, os.path.join(out_dir, "plots"))
    md = report_md(R, title, plots)
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write(md)
    return os.path.join(out_dir, "report.md")


def make_plots(R, D, d):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:                       # pragma: no cover
        return []
    os.makedirs(d, exist_ok=True)
    out = []
    # 1 battle win rate per round
    w = R["H_outcomes"]["battle_win"]
    cols = [c for c in ROUND_ORDER if c in w]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    m = [w[c]["m"] for c in cols]
    err = [[w[c]["m"] - w[c]["lo"] for c in cols], [w[c]["hi"] - w[c]["m"] for c in cols]]
    ax.bar(cols, m, yerr=err, color="#4C72B0", capsize=3)
    ax.set_ylim(min(m) - 0.08, 1.0)
    ax.set_ylabel("victorias por combate")
    ax.set_xlabel("ronda (N = Noland)")
    fig.tight_layout()
    p = os.path.join(d, "win_by_round.png")
    fig.savefig(p, dpi=110)
    plt.close(fig)
    out.append(("Victorias por combate en cada ronda (IC 95 %)", p))
    # 2 pick rate per species
    sp = R["A_rental"]["species"]
    if sp:
        sel = sp[:15] + sp[-15:] if len(sp) > 30 else sp
        fig, ax = plt.subplots(figsize=(7, max(3, 0.22 * len(sel))))
        names = [x["name"] for x in sel][::-1]
        ax.barh(names, [x["pick_rate"]["m"] for x in sel][::-1], color="#55A868", label="tasa de elección")
        ax.plot([x["p_in"] for x in sel][::-1], names, "k|", markersize=8, label="prob. media de la política")
        ax.set_xlabel("P(elegido | ofrecido)")
        ax.legend(loc="lower right", fontsize=7)
        fig.tight_layout()
        p = os.path.join(d, "pick_rate_species.png")
        fig.savefig(p, dpi=110)
        plt.close(fig)
        out.append(("Tasa de elección en el alquiler (15 más y 15 menos elegidos)", p))
    # 3 entropy by round
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for kind in ("rental", "swap", "battle"):
        e = R["H_confidence"][kind]["entropy"]
        cols = [c for c in ROUND_ORDER if c in e]
        ax.plot(cols, [e[c]["m"] for c in cols], marker="o", label=kind)
    ax.set_ylabel("entropía (nats)")
    ax.set_xlabel("ronda")
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = os.path.join(d, "entropy_by_round.png")
    fig.savefig(p, dpi=110)
    plt.close(fig)
    out.append(("Entropía de la política por ronda", p))
    # 4 value calibration
    cal = R["H_confidence"].get("value_calibration") or []
    if cal:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.plot([c["pred"] for c in cal], [c["won"] for c in cal], "o-")
        ax.set_xlabel("valor del combatiente (1.ª decisión)")
        ax.set_ylabel("fracción ganada")
        fig.tight_layout()
        p = os.path.join(d, "value_calibration.png")
        fig.savefig(p, dpi=110)
        plt.close(fig)
        out.append(("Calibración del valor del combatiente", p))
    return out


def report_md(R, title, plots):
    from .playreport import render
    return render(R, title, plots)


# ---- CLI ---------------------------------------------------------------------------------------------------------------

def _paths(items):
    out = []
    for it in items:
        if os.path.isdir(it):
            out += sorted(os.path.join(it, f) for f in os.listdir(it) if f.endswith(".jsonl") or f.endswith(".jsonl.gz"))
        else:
            out.append(it)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--logs", nargs="+", required=True, help="log files or folders of *.jsonl[.gz]")
    p.add_argument("--out", required=True)
    p.add_argument("--title", default=None)
    p.add_argument("--boot", type=int, default=1000)
    p.add_argument("--min-n", type=int, default=30, help="minimum observations for per-species / per-move rows")
    p.add_argument("--no-late", action="store_true", help="leave out the runs that started at rounds 5-6")
    a = p.parse_args(argv)
    paths = _paths(a.logs)
    D = extract(paths, include_late=not a.no_late)
    R = analyze(D, B=a.boot, min_n=a.min_n)
    R["files"] = paths
    title = a.title or f"Cómo juega el agente ({(R['meta'] or {}).get('battler', '?')})"
    path = write_report(R, D, a.out, title)
    print(path)
    return R


if __name__ == "__main__":
    main()

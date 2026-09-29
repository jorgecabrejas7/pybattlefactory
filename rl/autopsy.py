"""Loss autopsy of an alphazero_v3 checkpoint (docs/RL_DECISIONS.md §21): why do the battles of rounds 5-6 get lost?

    python -m rl.autopsy --ckpt runs/alphazero_v3/latest.pt --name v3 [--losses 500] [--rounds 5,6] \\
        [--battler net|search] [--workers 18] [--inference gpu] [budgets: see --help]
    python -m rl.autopsy --name v3 --report-only          # rebuild summary / CSV / report from saved results

Diagnosis only, never training: the strong battler below uses perfect-information search, which rl.search allows
outside training processes only (this entry point is not one; it refuses to run inside one).

Pipeline (outputs in runs/autopsy_<name>/):
1. Collection (collect.jsonl): runs that start at the beginning of round k (k in --rounds, alternating; the
   protocol of rl.eval_rounds: a run stops when round k is completed or lost) played by the checkpoint's agents as
   evaluated: the tactician = its network, greedy, with its per-option features (W, planner: rl.alphazero_v3
   features_hook); the battler = its network, greedy (--battler net) or the legal search of the evaluation
   (--battler search: 512 simulations, 8 determinizations). Every battle is recorded (wins too: base rates):
   our team and the opponent's true team (analysis only), the attendant's hint, Noland or not, and the battle's
   messages (Gen3Game.log_messages: every CONTROLLER_PRINTSTRING of the game), turned into events (crits, misses,
   freeze / sleep / paralysis, OHKO, confusion self-hits, flinches, Explosion, Perish Song...). A lost battle also
   keeps its replay recipe: the run's environment seed and every backend action since the run began (the
   simulator is deterministic: replaying them rebuilds every state exactly, checked by a state hash), the log
   index of the tactician decision before the battle and of each of our battle decisions, and the turn log.
2. Analysis of each loss (analysis.jsonl), from replayed states:
   (a) LUCK: P0 = P(win | battle start) with the strong battler (--strong-sims, perfect information by default),
       over --luck-seeds battle RNG seeds: from the tactician decision before the battle, the same option is
       applied, the true opponent team written back, and the battle RNG reseeded before its first frame.
   (b) BATTLER ERROR: rewinding to our decisions from the last one backwards, a strong search (legal and perfect)
       says what it would play; where it differs from our move, P_d = P(win) with strong play from there
       (--mistake-seeds seeds, the turn redrawn: Gen3Game.redraw_turn), and P_ours = P(our move, then strong
       play) on the same seeds; the latest decision with P_d >= 0.5 and P_d - P_ours >= --mistake-delta is "the
       decisive mistake" (without the delta condition: "p_found", also reported).
   (c) TEAM: the same start with each alternative option of that tactician decision (swap: the 10 options;
       rental, battle 1 of the round: every legal rental), the same true opponent and the same seeds (common
       random numbers). Stage 1 screens every option with the network battler (--team-screen-seeds); stage 2 plays
       the --team-top best alternatives with the strong battler on the first --team-seeds luck seeds (ours: (a) on
       the same seeds). Also W's and the planner's predictions (Q_plan) for each option.
   Label (precedence): battler_error > team > luck > mixed > bad_matchup (see label()).
3. Report: summary.json (label fractions with bootstrap CIs, overall / Noland / by round; patterns with
   lift = P(feature | loss) / P(feature | win)), losses.csv (one row per loss), report.md (short, in Spanish).
"""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import struct
import sys
import time
from collections import Counter, defaultdict

import numpy as np

from pybattle.backend import Phase
from pybattle.emu.decode import SYMBOLS as S, decode_battle_mon, decode_party
from pybattle.view import NAMES

# ---- battle messages -> events ------------------------------------------------------------------------------------

# STRINGID_* of the decomp (third_party/pokeemerald/include/constants/battle_string_ids.h; a test checks them)
STRINGIDS = {
    "USEDMOVE": 4, "ATTACKMISSED": 23, "ATTACKERFAINTED": 28, "TARGETFAINTED": 29, "PKMNFELLASLEEP": 35,
    "PKMNWASFROZEN": 49, "PKMNISFROZEN": 51, "PKMNWASPARALYZED": 55, "PKMNWASPARALYZEDBY": 56,
    "PKMNISPARALYZED": 57, "PKMNWASCONFUSED": 67, "PKMNIMMOBILIZEDBYLOVE": 71, "PKMNFLINCHED": 74,
    "PKMNFASTASLEEP": 107, "PKMNWOKEUP": 108, "CRITICALHIT": 217, "ONEHITKO": 218, "ITHURTCONFUSION": 230,
    "PKMNHUNGONWITHX": 303,
}
_SID = STRINGIDS
# event -> (string ids, whose event it is: the battler field of the message the text names)
MESSAGE_EVENTS = {
    "crit": ((_SID["CRITICALHIT"],), "attacker"),                  # A critical hit! (by the attacker)
    "miss": ((_SID["ATTACKMISSED"],), "attacker"),                 # X's attack missed! (accuracy roll failed)
    "ohko_hit": ((_SID["ONEHITKO"],), "attacker"),                 # It's a one-hit KO!
    "confusion_hit": ((_SID["ITHURTCONFUSION"],), "attacker"),     # It hurt itself in its confusion!
    "flinch": ((_SID["PKMNFLINCHED"],), "attacker"),               # X flinched!
    "freeze": ((_SID["PKMNWASFROZEN"],), "effect"),                # X was frozen solid! (inflicted on X)
    "frozen_turn": ((_SID["PKMNISFROZEN"],), "attacker"),          # X is frozen solid! (turn lost)
    "paralysis": ((_SID["PKMNWASPARALYZED"], _SID["PKMNWASPARALYZEDBY"]), "effect"),
    "full_para": ((_SID["PKMNISPARALYZED"],), "attacker"),         # X is paralyzed! It can't move!
    "sleep": ((_SID["PKMNFELLASLEEP"],), "effect"),                # X fell asleep!
    "asleep_turn": ((_SID["PKMNFASTASLEEP"],), "attacker"),        # X is fast asleep. (turn lost)
    "wake": ((_SID["PKMNWOKEUP"],), "attacker"),
    "confused": ((_SID["PKMNWASCONFUSED"],), "effect"),
    "love_turn": ((_SID["PKMNIMMOBILIZEDBYLOVE"],), "attacker"),
    "focus_band": ((_SID["PKMNHUNGONWITHX"],), "target"),          # X hung on using its Focus Band!
}
_EVENT_OF_SID = {sid: (ev, who) for ev, (sids, who) in MESSAGE_EVENTS.items() for sid in sids}
_MOVE_ID = {v: int(k) for k, v in NAMES["moves"].items()}
MOVE_EVENTS = {                                                    # "X used Y!" with these moves
    "explosion": {_MOVE_ID["MOVE_EXPLOSION"], _MOVE_ID["MOVE_SELF_DESTRUCT"]},
    "ohko_move": {_MOVE_ID[m] for m in ("MOVE_FISSURE", "MOVE_HORN_DRILL", "MOVE_GUILLOTINE", "MOVE_SHEER_COLD")},
    "perish_song": {_MOVE_ID["MOVE_PERISH_SONG"]},
}
# derived: a crit whose target fainted from that same move ("crit_ko": a KO the crit made possible or hastened)
EVENT_NAMES = tuple(MESSAGE_EVENTS) + tuple(MOVE_EVENTS) + ("crit_ko",)
# events that hurt us when they are about side 0 ("own") / side 1 ("opp"): the luck balance of a battle
BAD_FOR_US = ({f"opp_{e}" for e in ("crit", "ohko_hit", "crit_ko", "focus_band")}
              | {f"own_{e}" for e in ("miss", "confusion_hit", "flinch", "freeze", "frozen_turn", "paralysis",
                                      "full_para", "sleep", "asleep_turn", "confused", "love_turn")})
GOOD_FOR_US = {("opp_" + e[4:]) if e.startswith("own_") else ("own_" + e[4:]) for e in BAD_FOR_US}


def parse_messages(msgs):
    """Gen3Game.take_messages() tuples (string_id, active, attacker, target, effect, move) -> events
    [{"ev": name, "side": 0 us / 1 opponent, "move": id}]; side = the battler the event is about (the attacker for
    crit / miss / flinch / a lost turn..., the Pokemon that got the status for freeze / sleep / paralysis /
    confusion, the defender for Focus Band). In singles battler 0 is ours and 1 the opponent's."""
    out = []
    cur = None                  # the move being executed: {"side", "move", "crit", "ohko"}
    for sid, active, attacker, target, effect, move in msgs:
        if sid == _SID["USEDMOVE"]:
            cur = {"side": attacker & 1, "move": move, "crit": False}
            for ev, moves in MOVE_EVENTS.items():
                if move in moves:
                    out.append({"ev": ev, "side": attacker & 1, "move": move})
            continue
        hit = _EVENT_OF_SID.get(sid)
        if hit is not None:
            ev, who = hit
            b = {"attacker": attacker, "effect": effect, "target": target}[who]
            out.append({"ev": ev, "side": b & 1, "move": move})
            if ev == "crit" and cur is not None:
                cur["crit"] = True
            continue
        if sid == _SID["TARGETFAINTED"] and cur is not None and cur["crit"] and (target & 1) != cur["side"]:
            out.append({"ev": "crit_ko", "side": cur["side"], "move": cur["move"]})
            cur["crit"] = False
    return out


def event_keys(events):
    """{"own_crit", "opp_miss", ...}: which (side, event) pairs happened at least once."""
    return {("own_" if e["side"] == 0 else "opp_") + e["ev"] for e in events}


def luck_balance(counts):
    """Events that hurt us minus events that helped us (a rough luck score of a battle)."""
    return int(sum(n for k, n in counts.items() if k in BAD_FOR_US) - sum(n for k, n in counts.items()
                                                                            if k in GOOD_FOR_US))


# ---- names, sets ------------------------------------------------------------------------------------------------

def species_name(s):
    return NAMES["species"].get(str(s), str(s)).replace("SPECIES_", "")


def move_name(m):
    return NAMES["moves"].get(str(m), str(m)).replace("MOVE_", "")


_SET_INDEX = None


def set_ids_of(species, moves, item):
    """The Battle Frontier set ids (gBattleFrontierMons) with this species, move set and held item."""
    global _SET_INDEX
    if _SET_INDEX is None:
        from .determinize import SET_ITEM, SET_MOVES, SET_SPECIES
        _SET_INDEX = defaultdict(list)
        for i in range(len(SET_SPECIES)):
            _SET_INDEX[(int(SET_SPECIES[i]), int(SET_ITEM[i]), SET_MOVES[i])].append(i)
    return list(_SET_INDEX.get((int(species), int(item), frozenset(x for x in moves if x)), []))


def team_of(party):
    """decode_party mons -> JSON-able team: species, moves, item, set ids."""
    out = []
    for m in party[:3]:
        out.append({"species": m.species, "name": species_name(m.species), "moves": list(m.moves),
                    "item": m.held_item, "sets": set_ids_of(m.species, m.moves, m.held_item)})
    return out


_HASHED = (("gPlayerParty", 600), ("gEnemyParty", 600), ("gBattleMons", 0x58 * 2), ("gRngValue", 4),
           ("gBattleResults", 0x26), ("gBattlerPartyIndexes", 4))


def state_hash(game):
    """A hash of the battle state that is the same in every process: parties, battlers, RNG, results, frame count
    (the raw state holds function / heap pointers, which differ between processes)."""
    h = hashlib.sha1()
    for sym, n in _HASHED:
        h.update(game.read(S.addr(sym), n))
    h.update(struct.pack("<I", game.frames))
    return h.hexdigest()[:16]


def stable_seed(*parts):
    """A 32-bit seed from any parts, the same in every process (Python's hash() is salted per process)."""
    return int.from_bytes(hashlib.blake2b(repr(parts).encode(), digest_size=4).digest(), "little")


# ---- actions (JSON) ------------------------------------------------------------------------------------------------

def enc_action(a):
    if a is None:
        return ["keep"]
    if isinstance(a, tuple) and a and isinstance(a[0], str):
        return list(a)
    return ["opt"] + [int(x) for x in a]


def dec_action(a):
    if a[0] == "keep":
        return None
    if a[0] == "opt":
        return tuple(a[1:])
    return tuple(a)


# ---- recording a run -----------------------------------------------------------------------------------------------

_BATTLE = (Phase.BATTLE, Phase.FORCED_SWITCH)


def _actives(game):
    mons = [decode_battle_mon(game.read(S.addr("gBattleMons") + b * 0x58, 0x58)) for b in range(2)]
    return [{"species": m.species, "hp": m.hp, "max_hp": m.max_hp} for m in mons]


class Recorder:
    """Wraps a SimBackend's act(): logs every backend action since the run's reset (the replay recipe) and builds a
    record per battle: teams, hint, Noland, the turn log (our decisions with the true active Pokemon / HP and the
    events of the messages the game printed until the next decision), and the state hash at the battle's end."""

    def __init__(self, backend):
        self.be = backend
        self.log = []
        self.cur = None
        self.finished = []
        backend.game.log_messages(True)
        self._orig = backend.act
        backend.act = self.act

    def act(self, action):
        be = self.be
        phase = be.phase
        li = len(self.log)
        pre = None
        if phase in _BATTLE:
            v = be.view()
            n_legal = len([a for a in v.legal_actions if a[0] != "forfeit"])
            pre = {"li": li, "turn": int(v.turn), "forced": phase == Phase.FORCED_SWITCH,
                   "action": enc_action(action), "n_choices": n_legal, "act": _actives(be.game)}
        elif phase in (Phase.RENTAL, Phase.SWAP):
            info = be.run_info()
            pre = {"li": li, "kind": "rental" if phase == Phase.RENTAL else "swap", "action": enc_action(action),
                   "streak": info.win_streak, "battle_in_challenge": info.battle_in_challenge}
        self.log.append(enc_action(action))
        self._orig(action)
        msgs = be.game.take_messages()
        if phase in (Phase.RENTAL, Phase.SWAP):
            self._start_battle(pre, msgs)
        elif phase in _BATTLE and self.cur is not None:
            ev = parse_messages(msgs)
            pre["events"] = [[e["ev"], e["side"]] for e in ev]
            self.cur["turns"].append(pre)
            self.cur["_events"] += ev
            if be.phase not in _BATTLE:
                self._end_battle()

    def _start_battle(self, pre, msgs):
        be, g = self.be, self.be.game
        if be.phase not in _BATTLE:            # (the battle ended without a decision of ours)
            return
        info = g.factory_info
        v = be.view()
        own = decode_party(g.read(S.addr("gPlayerParty"), 300))
        foe = decode_party(g.read(S.addr("gEnemyParty"), 300))
        streak = pre["streak"]
        self.cur = {"pre_index": pre["li"], "pre_kind": pre["kind"], "pre_action": pre["action"],
                    "streak": streak, "round": streak // 7 + 1, "battle": streak % 7 + 1,
                    "noland": info.brain_status != 0, "hint": [int(v.hint_type), int(v.hint_style)],
                    "own": team_of(own), "opp": team_of(foe), "turns": [], "_events": parse_messages(msgs)}

    def _end_battle(self):
        be, b = self.be, self.cur
        ev = b.pop("_events")
        counts = Counter(("own_" if e["side"] == 0 else "opp_") + e["ev"] for e in ev)
        b["won"] = bool(be.last_battle_won)
        b["events"] = dict(counts)
        b["luck_balance"] = luck_balance(counts)
        b["n_decisions"] = len(b["turns"])
        b["end_index"] = len(self.log)
        b["end_hash"] = state_hash(be.game)
        self.finished.append(b)
        self.cur = None


# ---- worker state -------------------------------------------------------------------------------------------------

_W = None           # per-process state (built by _init_worker)
_SERVER = None      # the GPU inference server, set before forking


class Worker:
    """What a worker process needs: the policy (tactician, and the net battler without a server), the battler
    evaluator (a GPU server client or the local network), the tactician features hook, and cached battlers."""

    def __init__(self, cfg, slot=0):
        import torch
        from .inference import make_evaluator
        from .policy import Policy
        torch.set_num_threads(cfg.get("threads", 1))
        self.cfg = cfg
        self.policy = Policy(cfg["ckpt"], torch.device("cpu"))
        if _SERVER is not None:
            self.evaluator = _SERVER.client(slot).evaluate
        else:
            self.evaluator = make_evaluator(self.policy)
        self.hook = None
        if cfg.get("features", True):
            from .alphazero_v3 import features_hook
            self.hook = features_hook(cfg["ckpt"], cfg.get("team_eval"))
        self._search = {}

    # battlers: f(backend, decisions_so_far) -> backend action
    def net_act(self, be, ndec=0):
        from . import encode
        from .inference import stack_obs
        from .search import run_ctx, to_backend_action
        x = encode.battle(be.view(), run_ctx(be))
        pri, _ = self.evaluator(stack_obs([x]))
        p = np.where(x["mask"], pri[0], -1.0)
        return to_backend_action(int(np.argmax(p)))

    def search(self, mode, sims, dets):
        key = (mode, sims, dets)
        if key not in self._search:
            from .search import SearchBattler
            c = self.cfg
            self._search[key] = SearchBattler(self.policy, n_sims=sims, n_determinizations=dets,
                                              c_puct=c.get("c_puct", 1.5), batch=c.get("search_batch", 32),
                                              mode=mode, allow_perfect=mode == "perfect", seed=0,
                                              evaluator=self.evaluator)
        return self._search[key]

    def search_act(self, mode, sims, dets, seed):
        sb = self.search(mode, sims, dets)

        def act(be, ndec=0):
            sb.rng = random.Random(stable_seed(seed, ndec))
            return sb.act_on(be, decisions=ndec)
        act.search = sb
        return act

    def strong(self, mode, seed):
        c = self.cfg
        return self.search_act(mode, c["strong_sims"], c["strong_dets"], seed)


def _init_worker(cfg, slots):
    global _W
    slot = slots.get() if slots is not None else 0
    _W = Worker(cfg, slot)


# ---- collection -------------------------------------------------------------------------------------------------

def run_seed(base, k, i):
    return stable_seed("autopsy-run", base, k, i) & 0x7FFFFFFF


def _tactician_act(W, kind, obs):
    from . import encode
    from .envs import decode_action
    a = W.policy.act(kind, encode.collate([obs], W.policy.device), greedy=True)[0]
    return decode_action(kind, a)


def play_run(W, k, env_seed, battler):
    """One run from the start of round k (the eval_rounds protocol: stops when round k is completed or lost).
    -> (battle records, completed). A lost battle's record carries its replay recipe (log up to its end)."""
    from .envs import NO_ACTION, FactoryEnv
    env = FactoryEnv(env_seed, win_streak=7 * (k - 1), obs_hook=W.hook)
    rec = Recorder(env.backend)
    ev = env._advance(NO_ACTION)
    wins, out = 0, []
    while True:
        for b in rec.finished:
            b.update({"env_seed": env_seed, "start_round": k})
            if not b["won"]:
                b["log"] = list(rec.log[:b["end_index"]])
            out.append(b)
        rec.finished = []
        st = ev["stats"].get("battle")
        if st is not None and st["won"] is not None:
            if not st["won"]:
                return out, False
            wins += 1
            if wins == 7:
                return out, True
        if ev["truncate"]:
            return out, False
        kind = ev["kind"]
        if kind == "battle":
            action = battler(env.backend, env.decisions - 1)
        else:
            action = _tactician_act(W, kind, ev["obs"])
        ev = env.step(action)


def _collect_job(job):
    W = _W
    p, n_jobs, target, max_runs = job["p"], job["n_jobs"], job["target"], job["max_runs"]
    cfg = W.cfg
    if cfg["battler"] == "net":
        battler = W.net_act
    else:
        battler = W.search_act("legal", cfg["battler_sims"], cfg["battler_dets"], stable_seed("battler", p))
    rounds = cfg["rounds"]
    losses, wins, runs, t0 = [], [], 0, time.time()
    i = p
    while len(losses) < target and runs < max_runs:
        k = rounds[(i // n_jobs) % len(rounds)]
        seed = run_seed(cfg["seed"], k, i)
        battles, _ = play_run(W, k, seed, battler)
        for b in battles:
            b["uid"] = f"{seed}-{b['streak']}"
            (wins if b["won"] else losses).append(b)
        runs += 1
        i += n_jobs
    return {"losses": losses, "wins": wins, "runs": runs, "seconds": time.time() - t0}


# ---- replay and battle starts ----------------------------------------------------------------------------------

def replay(rec, upto):
    """The SimBackend of the loss's run after its first `upto` logged actions (deterministic)."""
    from .envs import FactoryEnv
    env = FactoryEnv(rec["env_seed"], win_streak=7 * (rec["start_round"] - 1))
    be = env.backend
    for a in rec["log"][:upto]:
        be.act(dec_action(a))
    return be


def replay_battle(rec):
    """-> (backend at the tactician decision before the battle, [(turn entry, backend clone at that decision)]).
    Checks that the replay ends the battle as recorded (a loss, same state hash)."""
    pre = replay(rec, rec["pre_index"])
    be = pre.clone()
    li = rec["pre_index"]
    by_li = {t["li"]: t for t in rec["turns"]}
    states = []
    while li < rec["end_index"]:
        if li in by_li:
            states.append((by_li[li], be.clone()))
        be.act(dec_action(rec["log"][li]))
        li += 1
    if be.last_battle_won or state_hash(be.game) != rec["end_hash"]:
        raise RuntimeError(f"replay of {rec['uid']} diverged from the recorded battle")
    return pre, states


def option_action(kind, option):
    from .envs import decode_action
    return decode_action(kind, option)


def true_enemy_bytes(pre, action):
    """gEnemyParty right after the tactician's actual option started the battle (before any frame): the true
    opponent (Noland's team is drawn there, and depends on our team)."""
    be = pre.clone()
    _apply_option(be, action)
    return be.game.read(S.addr("gEnemyParty"), 600)      # the whole party (Gen3_WriteParty clears the rest)


def _apply_option(be, action):
    g = be.game
    if be.phase == Phase.RENTAL:
        ok = g.factory_rent(*action)
    elif be.phase == Phase.SWAP:
        ok = g.factory_swap(-1) if action is None else g.factory_swap(action[0], action[1])
    else:
        raise RuntimeError(f"not at a tactician decision: {be.phase}")
    if not ok:
        raise ValueError(f"option {action} refused by the game")


def start_battle(pre, action, enemy_bytes, seed):
    """From the tactician decision `pre` (a SimBackend at RENTAL / SWAP): apply `action`, write the true opponent
    team back (enemy_bytes; None: keep what the game generated), reseed the battle RNG before its first frame (seed
    None: keep the game's own RNG, which replays the real battle) and run to our first decision.
    Mirrors SimBackend.act (observer, opponent_knowledge)."""
    be = pre.clone()
    be.opponent_knowledge = be.opponent_knowledge_for(action)
    rental = be.phase == Phase.RENTAL
    if rental:
        be._observer = be._new_observer()
    _apply_option(be, action)
    if not rental:
        be._observer = be._new_observer()
    if enemy_bytes is not None:
        be.game.write_party(1, enemy_bytes)
    if seed is not None:
        be.game.set_rng(seed & 0xFFFFFFFF)
    be.phase = Phase.BATTLE
    be._sync()
    return be


def play_out(be, battler, ndec=0, max_decisions=300):
    """Play the current battle of `be` to its end. -> 1.0 win, 0.0 loss or cut at max_decisions (the FactoryEnv
    limit); decisions without a choice are played as FactoryEnv plays them (move 0)."""
    n = 0
    while be.phase in _BATTLE:
        if ndec >= max_decisions:
            return 0.0, n
        if be.phase == Phase.BATTLE:
            v = be.view()
            if not any(v.usable_moves) and not v.switch_targets:
                be.act(("move", 0))
                ndec += 1
                continue
        be.act(battler(be, ndec))
        ndec += 1
        n += 1
    return float(bool(be.last_battle_won)), n


# ---- analysis of one loss ------------------------------------------------------------------------------------------

class Counters:
    def __init__(self):
        self.decisions = Counter()      # strong-search decisions per phase
        self.seconds = Counter()

    def add(self, phase, n, s):
        self.decisions[phase] += n
        self.seconds[phase] += s


def p_start(W, pre, action, enemy, seeds, mode, cnt, phase):
    """P(win | battle start with this option) with the strong battler (mode) over the seeds, and the outcomes."""
    t0, n, out = time.time(), 0, []
    for s in seeds:
        be = start_battle(pre, action, enemy, s)
        z, k = play_out(be, W.strong(mode, s) if mode != "net" else W.net_act)
        out.append(z)
        n += k
    cnt.add(phase, n, time.time() - t0)
    return float(np.mean(out)), out


def p_from(W, be_d, ndec, seeds, mode, first_action=None, cnt=None, phase="mistake", stop_below=None):
    """P(win) from our decision state be_d with strong play, the turn's randomness redrawn per seed
    (Gen3Game.redraw_turn); first_action: play it at be_d first (our move), strong play afterwards.
    stop_below: stop as soon as the mean over all the seeds can no longer reach it (the estimate is then the mean
    of the seeds played, below the threshold either way). -> (P, seeds played)."""
    from .search import redraw_turn
    t0, n, out = time.time(), 0, []
    for s in seeds:
        if stop_below is not None and (sum(out) + len(seeds) - len(out)) / len(seeds) < stop_below:
            break
        be = be_d.clone()
        redraw_turn(be.game, s)
        d = ndec
        if first_action is not None:
            be.act(first_action)
            d += 1
        z, k = play_out(be, W.strong(mode, s), d)
        out.append(z)
        n += k
    if cnt is not None:
        cnt.add(phase, n, time.time() - t0)
    return float(np.mean(out)), len(out)


def find_mistake(W, rec, states, mode, cnt):
    """Scan our decisions from the last backwards: the strong search (mode) at the real state; where its choice
    differs from ours, P_d = P(win | strong play from d) and, if P_d >= mistake_p, P_ours = P(our move, then strong
    play) with the same seeds. The decisive mistake is the latest such decision with P_d - P_ours >= mistake_delta
    ("found"); "p_found" marks the latest with P_d >= mistake_p whatever P_ours (the bare definition). Stops after
    cfg max_rewinds differing decisions. -> {"found", "p_found", "scanned", "compared", "agree", "rewinds", ...}."""
    cfg = W.cfg
    K = cfg["mistake_seeds"]
    ndec_of = {}
    n = 0
    for t, _ in states:                     # decisions played before each one (FactoryEnv's counter)
        ndec_of[t["li"]] = n
        n += 1
    res = {"found": False, "p_found": False, "scanned": 0, "compared": 0, "agree": 0, "rewinds": []}
    choice_states = [(t, b) for t, b in states if t["n_choices"] >= 2]
    for t, b in reversed(choice_states):
        if res["scanned"] >= cfg["max_rewinds"]:
            break
        t0 = time.time()
        strong = W.strong(mode, stable_seed(rec["uid"], "choice", mode, t["li"]))
        choice = strong(b.clone(), ndec_of[t["li"]])
        st = strong.search.last_stats
        cnt.add(f"choice_{mode}", 1, time.time() - t0)
        ours = dec_action(t["action"])
        res["compared"] += 1
        if tuple(choice) == tuple(ours):
            res["agree"] += 1
            continue
        res["scanned"] += 1
        seeds = [stable_seed(rec["uid"], "rewind", t["li"], j) for j in range(K)]
        p_d, n_played = p_from(W, b, ndec_of[t["li"]], seeds, mode, cnt=cnt, phase=f"mistake_{mode}",
                               stop_below=cfg["mistake_p"])
        a_idx = lambda a: a[1] if a[0] == "move" else 4 + a[1]
        q = st["q"] if st else [0.0] * 7
        entry = {"li": t["li"], "turn": t["turn"], "decision": ndec_of[t["li"]], "ours": list(ours),
                 "strong": list(choice), "p_strong": p_d, "seeds": n_played,
                 "q_gap": float(q[a_idx(choice)] - q[a_idx(ours)]),
                 "forced": t["forced"]}
        res["rewinds"].append(entry)
        if p_d >= cfg["mistake_p"]:
            entry["p_ours"], _ = p_from(W, b, ndec_of[t["li"]], seeds, mode, first_action=ours, cnt=cnt,
                                        phase=f"mistake_{mode}")
            entry["delta"] = p_d - entry["p_ours"]
            if not res["p_found"]:
                res.update(p_found=True, p_found_turn=t["turn"], p_found_delta=entry["delta"])
            if entry["delta"] < cfg["mistake_delta"]:
                continue                            # our move did about as well: not the mistake, look earlier
            res.update(found=True, **{k: entry[k] for k in ("turn", "decision", "ours", "strong", "p_strong",
                                                             "p_ours", "delta", "q_gap", "forced")})
            res["from_end"] = len(states) - 1 - entry["decision"]
            res["move_kind"] = _kind_of(ours, rec, t)
            res["strong_kind"] = _kind_of(choice, rec, t)
            break
    return res


def _kind_of(action, rec, t):
    """A short description of a battle action: 'move:<NAME>' / 'switch' (our team's moves from the record)."""
    if action[0] == "switch":
        return "switch"
    own = rec["own"]
    sp = t["act"][0]["species"]
    mon = next((m for m in own if m["species"] == sp), None)
    if mon is None:
        return "move"
    return "move:" + move_name(mon["moves"][action[1]])


def team_analysis(W, rec, pre, p0_outcomes, seeds, cnt):
    """(c) every option of the tactician decision before the battle against the true opponent, common seeds.
    Stage 2 uses the first cfg["team_seeds"] luck seeds; our P(win) is (a)'s on the same seeds (common random
    numbers: the gain compares options on the same battle RNG streams)."""
    from . import encode
    from .search import run_ctx
    from .tactician_search import options_of
    cfg = W.cfg
    kind = rec["pre_kind"]
    view = pre.view()
    ctx = run_ctx(pre)
    obs = encode.rental(view, ctx) if kind == "rental" else encode.swap(view, ctx)
    options = options_of(kind, obs)
    actions = [option_action(kind, o) for o in options]
    ours = dec_action(rec["pre_action"])
    try:
        our_i = actions.index(tuple(ours) if ours is not None else None)
    except ValueError:
        our_i = -1
    enemy = true_enemy_bytes(pre, ours)
    feats = None
    if W.hook is not None:
        try:
            feats = W.hook.compute(pre, kind, obs, ctx)["feats"]
        except Exception as e:                        # (the evaluator is optional for the autopsy)
            print(f"autopsy: features failed for {rec['uid']}: {e}", file=sys.stderr)
    last_or_noland = rec["battle"] == 7 or rec["noland"]
    from .team_eval import WIN_SCALE
    w_col = 2 if last_or_noland else 1

    def pred(i):
        if feats is None or i < 0:
            return {}
        return {"w": float(feats[i][w_col]), "q_plan": float(feats[i][4] * WIN_SCALE)}

    # stage 1: every option, network battler
    screen_seeds = [stable_seed(rec["uid"], "screen", j) for j in range(cfg["team_screen_seeds"])]
    p_net = []
    for i, a in enumerate(actions):
        try:
            p, _ = p_start(W, pre, a, enemy, screen_seeds, "net", cnt, "team_screen")
        except ValueError:
            p = float("nan")
        p_net.append(p)
    # stage 2: ours (P0 on these seeds, from (a)) and the top alternatives, strong battler
    n2 = min(cfg["team_seeds"], len(seeds))
    tseeds = seeds[:n2]
    P0 = float(np.mean(p0_outcomes[:n2]))
    order = [int(i) for i in np.argsort([-x if not math.isnan(x) else 1 for x in p_net], kind="stable")
             if i != our_i]
    top = order[:cfg["team_top"]]
    alts = []
    for i in top:
        p, _ = p_start(W, pre, actions[i], enemy, tseeds, cfg["strong_mode"], cnt, "team_strong")
        alts.append({"option": i, "action": enc_action(actions[i]), "team": _team_after(pre, kind, actions[i]),
                     "p_strong": p, "p_net": p_net[i], **pred(i)})
    best = max(alts, key=lambda x: x["p_strong"]) if alts else None
    res = {"kind": kind, "n_options": len(options), "our_option": our_i, "p0_team_seeds": P0, "team_seeds": n2,
           "p_net_ours": p_net[our_i] if our_i >= 0 else None,
           "p_net_best": float(np.nanmax(p_net)) if p_net else None, "alternatives": alts,
           **{"ours_" + k: v for k, v in pred(our_i).items()}}
    if best is not None:
        res.update(best_p=best["p_strong"], gain=best["p_strong"] - P0, best_team=best["team"])
        if feats is not None and our_i >= 0:
            q = feats[:, 4]
            w = feats[:, w_col]
            b = best["option"]
            res.update(planner_saw=bool(q[b] > q[our_i]), w_saw=bool(w[b] > w[our_i]),
                       planner_rank_best=int((q > q[b]).sum()) + 1, planner_rank_ours=int((q > q[our_i]).sum()) + 1)
    return res


def _team_after(pre, kind, action):
    """Species names of our team after an option (party order: lead first)."""
    if kind == "rental":
        v = pre.view()
        return [species_name(v.candidates[i].species) for i in action]
    own = decode_party(pre.game.read(S.addr("gPlayerParty"), 300))[:3]
    names = [species_name(m.species) for m in own]
    if action is not None:
        foe = pre.view().enemy_party
        names[action[0]] = species_name(foe[action[1]].species)
    return names


def analyze_loss(W, rec):
    cfg = W.cfg
    cnt = Counters()
    t0 = time.time()
    pre, states = replay_battle(rec)
    ours = dec_action(rec["pre_action"])
    enemy = true_enemy_bytes(pre, ours)
    seeds = [stable_seed(rec["uid"], "luck", m) for m in range(cfg["luck_seeds"])]
    out = {"uid": rec["uid"]}
    # (a) luck
    if cfg["luck_seeds"] > 0:
        P0, zs = p_start(W, pre, ours, enemy, seeds, cfg["strong_mode"], cnt, "luck")
        out["p0"] = P0
        out["p0_outcomes"] = zs
        if cfg.get("p0_net_seeds", 0):            # the same start with the network battler (same first seeds)
            net_seeds = [stable_seed(rec["uid"], "luck", m) for m in range(cfg["p0_net_seeds"])]
            out["p0_net"], _ = p_start(W, pre, ours, enemy, net_seeds, "net", cnt, "luck_net")
    else:
        P0 = float("nan")
    # (b) battler error
    out["mistake"] = {}
    for mode in cfg["mistake_modes"]:
        out["mistake"][mode] = find_mistake(W, rec, states, mode, cnt)
    # (c) team
    if cfg["team_top"] > 0 and cfg["luck_seeds"] > 0:
        out["team"] = team_analysis(W, rec, pre, zs, seeds, cnt)
    out["cost"] = {"decisions": dict(cnt.decisions), "seconds": {k: round(v, 2) for k, v in cnt.seconds.items()},
                   "total_seconds": round(time.time() - t0, 2)}
    out["label"], out["flags"] = label(out, cfg)
    return out


def _analyze_job(rec):
    try:
        return analyze_loss(_W, rec)
    except Exception as e:                           # one bad loss must not stop the run
        import traceback
        traceback.print_exc()
        return {"uid": rec["uid"], "error": f"{type(e).__name__}: {e}"}


# ---- labels -------------------------------------------------------------------------------------------------------

LABELS = ("battler_error", "team", "luck", "mixed", "bad_matchup")


def label(a, cfg):
    """One label per loss, by precedence, and the non-exclusive flags behind it:
        battler_error  a decisive mistake was found (mode cfg["label_mistake_mode"]; "either": any mode): a
                       differing decision with P(win | strong play) >= mistake_p and
                       delta = P(strong move) - P(our move) >= cfg["mistake_delta"];
        team           some alternative option had P(win) >= P0 + cfg["team_gain"] with the strong battler;
        luck           P0 >= cfg["luck_p"]: strong play wins this start most of the time, and no clear mistake;
        mixed          cfg["bad_p"] <= P0 < cfg["luck_p"], no mistake, no better team;
        bad_matchup    P0 < cfg["bad_p"] and no alternative team is clearly better.
    The battler comes first (the actual game shows a winnable position thrown away), then the team."""
    m = a.get("mistake", {})
    modes = list(m) if cfg["label_mistake_mode"] == "either" else [cfg["label_mistake_mode"]]
    mistake = any(m.get(md, {}).get("found") and m[md].get("delta", 1.0) >= cfg["mistake_delta"] for md in modes)
    t = a.get("team") or {}
    team = t.get("gain") is not None and t["gain"] >= cfg["team_gain"]
    p0 = a.get("p0", float("nan"))
    flags = {"mistake": bool(mistake), "team": bool(team), "p0_high": bool(p0 >= cfg["luck_p"]),
             "p0_low": bool(p0 < cfg["bad_p"])}
    if mistake:
        return "battler_error", flags
    if team:
        return "team", flags
    if p0 >= cfg["luck_p"]:
        return "luck", flags
    if p0 >= cfg["bad_p"]:
        return "mixed", flags
    return "bad_matchup", flags


# ---- statistics -----------------------------------------------------------------------------------------------------

def bootstrap_fractions(labels, cats, B=2000, seed=0):
    """Fraction of each category with a 95 % percentile bootstrap CI (resampling the losses)."""
    labels = np.asarray(labels)
    n = len(labels)
    out = {}
    if n == 0:
        return {c: {"frac": None, "lo": None, "hi": None, "n": 0} for c in cats}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    for c in cats:
        x = (labels == c).astype(float)
        bs = x[idx].mean(1)
        out[c] = {"frac": float(x.mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)),
                  "n": int(x.sum())}
    return out


def battle_features(b):
    """The pattern features of a battle record (loss or win): sets of 'family:value' strings."""
    f = set()
    for m in b["opp"]:
        f.add("opp_species:" + m["name"])
        for s in m["sets"][:1]:
            f.add(f"opp_set:{s}:{m['name']}")
    for m in b["own"]:
        f.add("own_species:" + m["name"])
    ht, hs = b["hint"]
    f.add("hint_style:" + ("noland" if b["noland"] else str(hs)))
    f.add("hint_type:" + ("noland" if b["noland"] else str(ht)))
    for k in b["events"]:
        f.add("event:" + k)
    f.add("round:" + str(b["round"]))
    f.add("battle:" + str(b["battle"]))
    f.add("noland:" + str(bool(b["noland"])))
    lb = b["luck_balance"]
    f.add("luck_balance:" + ("<=-1" if lb <= -1 else ">=2" if lb >= 2 else str(lb)))
    return f


def lift_table(losses, wins, min_losses=5):
    """Per feature: n / fraction among losses and wins, lift = P(f | loss) / P(f | win) (add-0.5 smoothing of the
    win count), and a 95 % CI of the log lift (Katz)."""
    nl, nw = len(losses), len(wins)
    cl, cw = Counter(), Counter()
    for b in losses:
        cl.update(battle_features(b))
    for b in wins:
        cw.update(battle_features(b))
    rows = []
    for f in set(cl) | set(cw):
        a, c = cl[f], cw[f]
        pl = a / max(nl, 1)
        pw = (c + 0.5) / (nw + 0.5)
        lift = pl / pw if pw > 0 else float("inf")
        se = math.sqrt(max(1 / a - 1 / max(nl, 1), 0) + 1 / (c + 0.5) - 1 / (nw + 0.5)) if a > 0 else float("nan")
        rows.append({"feature": f, "family": f.split(":")[0], "n_loss": a, "n_win": c, "p_loss": pl,
                     "p_win": c / max(nw, 1), "lift": lift,
                     "lift_lo": lift * math.exp(-1.96 * se) if a > 0 else 0.0,
                     "lift_hi": lift * math.exp(1.96 * se) if a > 0 else 0.0})
    rows.sort(key=lambda r: (r["family"], -r["lift"]))
    return [r for r in rows if r["n_loss"] >= min_losses or r["family"] in ("event", "hint_style", "round",
                                                                              "battle", "noland")]


def summarize(cfg, losses, wins, analyses, collect_stats):
    by_uid = {a["uid"]: a for a in analyses if "error" not in a}
    rows = [(l, by_uid[l["uid"]]) for l in losses if l["uid"] in by_uid]
    for _, a in rows:                       # (re)labelled with these thresholds (--report-only can change them)
        a["label"], a["flags"] = label(a, cfg)
    labs = [a["label"] for _, a in rows]
    out = {"config": cfg, "collection": collect_stats, "n_losses": len(losses), "n_wins": len(wins),
           "n_analyzed": len(rows), "errors": [a for a in analyses if "error" in a],
           "labels": bootstrap_fractions(labs, LABELS, seed=cfg["seed"])}
    for name, sel in (("noland", lambda l: l["noland"]), ("not_noland", lambda l: not l["noland"])):
        out["labels_" + name] = bootstrap_fractions([a["label"] for l, a in rows if sel(l)], LABELS, seed=cfg["seed"])
    for k in sorted({l["round"] for l, _ in rows}):
        out[f"labels_round_{k}"] = bootstrap_fractions([a["label"] for l, a in rows if l["round"] == k], LABELS,
                                                       seed=cfg["seed"])
    flags = defaultdict(int)
    for _, a in rows:
        flags[tuple(sorted(f for f, v in a["flags"].items() if v))] += 1
    out["flag_combinations"] = {"+".join(k) or "none": v for k, v in sorted(flags.items(), key=lambda x: -x[1])}
    p0 = np.array([a.get("p0", np.nan) for _, a in rows], float)
    out["p0"] = {"mean": float(np.nanmean(p0)) if len(p0) else None,
                 "hist": np.histogram(p0[~np.isnan(p0)], bins=[0, .2, .4, .5, .6, .8, 1.0001])[0].tolist()
                 if len(p0) else []}
    # mistakes
    mk = {}
    for mode in cfg["mistake_modes"]:
        found = [a["mistake"][mode] for _, a in rows if a["mistake"].get(mode, {}).get("found")]
        comp = sum(a["mistake"].get(mode, {}).get("compared", 0) for _, a in rows)
        agr = sum(a["mistake"].get(mode, {}).get("agree", 0) for _, a in rows)
        mk[mode] = {"frac": bootstrap_fractions([bool(a["mistake"].get(mode, {}).get("found")) for _, a in rows],
                                                [True], seed=cfg["seed"])[True],
                    "p_found_frac": bootstrap_fractions([bool(a["mistake"].get(mode, {}).get("p_found"))
                                                         for _, a in rows], [True], seed=cfg["seed"])[True],
                    "agreement_scanned": agr / comp if comp else None,
                    "turn_median": float(np.median([f["turn"] for f in found])) if found else None,
                    "from_end_median": float(np.median([f["from_end"] for f in found])) if found else None,
                    "delta_mean": float(np.mean([f["delta"] for f in found])) if found else None,
                    "our_kind": Counter(f["move_kind"] for f in found).most_common(15),
                    "strong_kind": Counter(f["strong_kind"] for f in found).most_common(15),
                    "switch_vs_move": Counter(f"{f['ours'][0]}->{f['strong'][0]}" for f in found).most_common()}
    out["mistakes"] = mk
    # team
    tm = [a["team"] for _, a in rows if a.get("team")]
    out["team"] = {"n": len(tm), "gain_mean": float(np.mean([t["gain"] for t in tm if "gain" in t])) if tm else None,
                   "gain_ge": {str(g): float(np.mean([t.get("gain", -1) >= g for t in tm])) if tm else None
                               for g in (0.1, 0.2, 0.3, 0.5)},
                   "planner_saw": _mean([t["planner_saw"] for t in tm if t.get("gain", 0) >= cfg["team_gain"]
                                         and "planner_saw" in t]),
                   "w_saw": _mean([t["w_saw"] for t in tm if t.get("gain", 0) >= cfg["team_gain"] and "w_saw" in t]),
                   "by_kind": dict(Counter(t["kind"] for t in tm))}
    out["patterns"] = lift_table(losses, wins, cfg["min_pattern_losses"])
    out["events_per_battle"] = {"loss": _event_rates(losses), "win": _event_rates(wins)}
    # losses are longer (more turns, more chances for any event): the same per 10 of our decisions
    out["events_per_10_decisions"] = {"loss": _event_density(losses), "win": _event_density(wins)}
    out["decisions_per_battle"] = {"loss": _mean([b["n_decisions"] for b in losses]),
                                   "win": _mean([b["n_decisions"] for b in wins])}
    out["luck_balance"] = {"loss": _mean([l["luck_balance"] for l in losses]),
                           "win": _mean([w["luck_balance"] for w in wins])}
    costs = [a["cost"] for _, a in rows]
    dec, sec = Counter(), Counter()
    for c in costs:
        dec.update(c["decisions"])
        sec.update(c["seconds"])
    out["cost"] = {"decisions_per_loss": {k: v / max(len(costs), 1) for k, v in dec.items()},
                   "seconds_per_loss": {k: v / max(len(costs), 1) for k, v in sec.items()},
                   "ms_per_decision": {k: 1000 * sec[k] / dec[k] for k in dec if dec[k]},
                   "wall_seconds_per_loss": _mean([c["total_seconds"] for c in costs])}
    return out


def _mean(x):
    x = list(x)
    return float(np.mean(x)) if x else None


def _event_density(battles):
    c = Counter()
    for b in battles:
        c.update(b["events"])
    n = max(sum(b["n_decisions"] for b in battles), 1)
    return {k: 10 * v / n for k, v in sorted(c.items())}


def _event_rates(battles):
    c = Counter()
    for b in battles:
        c.update(set(b["events"]))
    n = max(len(battles), 1)
    return {k: v / n for k, v in sorted(c.items())}


# ---- outputs --------------------------------------------------------------------------------------------------------

CSV_FIELDS = ("uid", "round", "battle", "noland", "streak", "hint_type", "hint_style", "own", "opp", "n_decisions",
              "luck_balance", "events", "label", "p0", "p0_net", "mistake_legal", "mistake_legal_turn",
              "mistake_legal_delta", "mistake_legal_ours", "mistake_legal_strong", "mistake_perfect",
              "mistake_perfect_turn", "mistake_perfect_delta", "team_kind", "team_gain", "team_best",
              "team_planner_saw", "team_w_saw", "ours_w", "ours_q_plan")


def csv_row(l, a):
    m = a.get("mistake", {})
    t = a.get("team") or {}
    ml, mp = m.get("legal", {}), m.get("perfect", {})
    return {"uid": l["uid"], "round": l["round"], "battle": l["battle"], "noland": l["noland"], "streak": l["streak"],
            "hint_type": l["hint"][0], "hint_style": l["hint"][1], "own": "/".join(x["name"] for x in l["own"]),
            "opp": "/".join(x["name"] for x in l["opp"]), "n_decisions": l["n_decisions"],
            "luck_balance": l["luck_balance"], "events": " ".join(f"{k}={v}" for k, v in sorted(l["events"].items())),
            "label": a.get("label"), "p0": a.get("p0"), "p0_net": a.get("p0_net"),
            "mistake_legal": ml.get("found"), "mistake_legal_turn": ml.get("turn"),
            "mistake_legal_delta": ml.get("delta"), "mistake_legal_ours": ml.get("move_kind"),
            "mistake_legal_strong": ml.get("strong_kind"), "mistake_perfect": mp.get("found"),
            "mistake_perfect_turn": mp.get("turn"), "mistake_perfect_delta": mp.get("delta"),
            "team_kind": t.get("kind"), "team_gain": t.get("gain"), "team_best": "/".join(t.get("best_team") or []),
            "team_planner_saw": t.get("planner_saw"), "team_w_saw": t.get("w_saw"), "ours_w": t.get("ours_w"),
            "ours_q_plan": t.get("ours_q_plan")}


def _f(x, d=2):
    return "—" if x is None else f"{x:.{d}f}"


def _pct(x):
    return "—" if x is None else f"{100 * x:.1f} %"


def _ci(d):
    if d["frac"] is None:
        return "—"
    return f"{100 * d['frac']:.1f} % [{100 * d['lo']:.1f}, {100 * d['hi']:.1f}] (n = {d['n']})"


def report_md(summary):
    s = summary
    c = s["config"]
    L = [f"# Autopsia de derrotas: `{c['ckpt']}`", "",
         f"Derrotas analizadas: {s['n_analyzed']} de {s['n_losses']} (victorias de referencia: {s['n_wins']}); "
         f"rondas {c['rounds']}; combatiente de la recogida: {c['battler']}; combatiente fuerte: búsqueda "
         f"{c['strong_mode']} de {c['strong_sims']} simulaciones; semillas: suerte {c['luck_seeds']}, "
         f"error {c['mistake_seeds']}, equipo {c['team_screen_seeds']} (criba) + {c.get('team_seeds')} (fuerte).", "",
         "Definiciones: **P0** = probabilidad de ganar el mismo combate (mismos equipos, mismo rival verdadero) "
         "desde su inicio con el combatiente fuerte, con otras semillas del RNG del combate. **Error decisivo** = "
         "la última decisión nuestra en la que la búsqueda fuerte habría jugado otra cosa y, jugando fuerte desde "
         f"allí, P(ganar) ≥ {c['mistake_p']} y, con las mismas semillas, jugar nuestra jugada y luego fuerte gana "
         f"al menos {c['mistake_delta']} menos. **Ganancia de equipo** = mejor P(ganar) de una opción alternativa "
         "del táctico − P0 (las mismas semillas). "
         "Intervalos: bootstrap al 95 % remuestreando derrotas.", "",
         "## Etiquetas (precedencia: error del combatiente > equipo > suerte > mixto > mal emparejamiento)", "",
         "| Etiqueta | Todas | Noland | Sin Noland |", "|---|---|---|---|"]
    for lab in LABELS:
        L.append(f"| {lab} | {_ci(s['labels'][lab])} | {_ci(s['labels_noland'][lab])} | "
                 f"{_ci(s['labels_not_noland'][lab])} |")
    L += ["", "Combinaciones de indicadores (no exclusivos): " + ", ".join(f"{k}: {v}" for k, v in
                                                                         s["flag_combinations"].items()), "",
          f"P0 media {_pct(s['p0']['mean'])}; histograma (0–0,2–0,4–0,5–0,6–0,8–1): {s['p0']['hist']}.", "",
          "## Error del combatiente", ""]
    for mode, m in s["mistakes"].items():
        L.append(f"- **{mode}**: con error decisivo {_ci(m['frac'])} (sin exigir P(fuerte) − P(nuestra) ≥ "
                 f"{c['mistake_delta']}: {_ci(m['p_found_frac'])}); acuerdo con la búsqueda fuerte en las "
                 f"decisiones revisadas {_pct(m['agreement_scanned'])}; turno mediano {m['turn_median']}, "
                 f"decisiones antes del final {m['from_end_median']}; tamaño medio (P fuerte − P nuestra) "
                 f"{_f(m['delta_mean'])}; "
                 f"nuestra jugada: {m['our_kind'][:6]}; la fuerte: {m['strong_kind'][:6]}; {m['switch_vs_move']}.")
    t = s["team"]
    L += ["", "## Equipo", "",
          f"Derrotas con análisis de equipo: {t['n']} ({t['by_kind']}); ganancia media {_f(t['gain_mean'])}; "
          f"fracción con ganancia ≥ 0,1 / 0,2 / 0,3 / 0,5: "
          + " / ".join(_pct(v) for v in t["gain_ge"].values()) + f". Cuando la ganancia ≥ {c['team_gain']}: "
          f"el planificador prefería la alternativa {_pct(t['planner_saw'])}, W {_pct(t['w_saw'])}.", "",
          "## Patrones (lift = P(rasgo | derrota) / P(rasgo | victoria))", ""]
    fam_rows = defaultdict(list)
    for r in s["patterns"]:
        fam_rows[r["family"]].append(r)
    for fam in ("event", "opp_species", "opp_set", "hint_style", "hint_type", "own_species", "noland", "round",
                "battle", "luck_balance"):
        rows = sorted(fam_rows.get(fam, []), key=lambda r: -r["lift"])[:12]
        if not rows:
            continue
        L += [f"**{fam}**", "", "| rasgo | derrotas | victorias | P(·|derrota) | P(·|victoria) | lift [IC 95 %] |",
              "|---|---|---|---|---|---|"]
        for r in rows:
            L.append(f"| {r['feature'].split(':', 1)[1]} | {r['n_loss']} | {r['n_win']} | {r['p_loss']:.3f} | "
                     f"{r['p_win']:.3f} | {r['lift']:.2f} [{r['lift_lo']:.2f}, {r['lift_hi']:.2f}] |")
        L.append("")
    lb = s["luck_balance"]
    dl, dw = s["events_per_10_decisions"]["loss"], s["events_per_10_decisions"]["win"]
    top = sorted(dl, key=lambda k: -dl[k] / max(dw.get(k, 0.0), 1e-3))[:10]
    L += [f"Las derrotas son más largas ({_f(s['decisions_per_battle']['loss'], 1)} decisiones nuestras de media "
          f"frente a {_f(s['decisions_per_battle']['win'], 1)}): parte del lift de los eventos es solo más tiempo para que ocurran. "
          "Por cada 10 decisiones nuestras (derrota / victoria): "
          + ", ".join(f"{k} {dl[k]:.2f} / {dw.get(k, 0.0):.2f}" for k in top) + ".", "",
          f"Balance de suerte medio (eventos en contra − a favor): derrotas {_f(lb['loss'])}, victorias {_f(lb['win'])}.", "",
          "## Coste", "", f"Por derrota: {s['cost']['wall_seconds_per_loss']} s de reloj en un proceso; decisiones "
          f"fuertes por fase {s['cost']['decisions_per_loss']}; ms por decisión {s['cost']['ms_per_decision']}.", ""]
    return "\n".join(L)


def write_outputs(out_dir, cfg, losses, wins, analyses, collect_stats):
    summary = summarize(cfg, losses, wins, analyses, collect_stats)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=_json_default)
    by_uid = {a["uid"]: a for a in analyses if "error" not in a}
    with open(os.path.join(out_dir, "losses.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for l in losses:
            if l["uid"] in by_uid:
                w.writerow(csv_row(l, by_uid[l["uid"]]))
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write(report_md(summary))
    return summary


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _dump_jsonl(path, items, mode="w"):
    with open(path, mode) as f:
        for x in items:
            f.write(json.dumps(x, default=_json_default) + "\n")


def _load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ---- cost estimate --------------------------------------------------------------------------------------------------

# Calibration (docs §21.4): 14 losses of rounds 5-6 analyzed with these seed budgets, and runs/diag_az1 (search
# evaluation of 2048 simulations, 18 workers + GPU server, free machine: ~215 ms per decision per worker).
CALIBRATION = {
    "strong_per_battle_start": 20.0,   # strong decisions to play a lost battle's start to its end
    "strong_per_battle_alt": 20.0,     # the same with an alternative team
    "compared_per_mode": 12.0,         # our decisions compared with a strong choice (when no mistake is found)
    "remaining_after_rewind": 8.0,     # strong decisions from a rewound decision to the end
    "net_per_battle": 15.0,            # network decisions per simulated battle (screen)
    "rental_frac": 0.2, "rental_options": 55, "swap_options": 10,
    "strong_ms_2048": 215.0,           # per decision per worker, 18 workers sharing the GPU server
    "net_ms": 1.3,
}


def estimate(cfg, n_losses, workers=None, cal=None):
    """Expected cost of the analysis of n_losses losses with cfg's budgets, in hours (collection excluded: a few
    minutes). Strong decisions scale linearly with the simulations (§15b: 512 -> 53 ms, 2048 -> 215 ms)."""
    c = dict(CALIBRATION, **(cal or {}))
    workers = workers or cfg["workers"]
    n_modes = len(cfg["mistake_modes"])
    R = c["remaining_after_rewind"]
    K = cfg["mistake_seeds"]
    strong = {
        "luck": cfg["luck_seeds"] * c["strong_per_battle_start"],
        "choice": n_modes * c["compared_per_mode"],
        # rewinds that differ: most are lost positions, stopped early (~K/2 + 1 seeds); a found mistake costs K
        # more seeds (our move)
        "mistake": n_modes * (cfg["max_rewinds"] * (K / 2 + 1) * R + 0.5 * K * R),
        "team_strong": cfg["team_top"] * min(cfg["team_seeds"], cfg["luck_seeds"]) * c["strong_per_battle_alt"],
    }
    opts = c["rental_frac"] * c["rental_options"] + (1 - c["rental_frac"]) * c["swap_options"]
    net = {"luck_net": cfg.get("p0_net_seeds", 0) * c["strong_per_battle_start"],
           "team_screen": opts * cfg["team_screen_seeds"] * c["net_per_battle"]}
    ms = c["strong_ms_2048"] * cfg["strong_sims"] / 2048
    per_loss_s = (sum(strong.values()) * ms + sum(net.values()) * c["net_ms"]) / 1000
    return {"strong_decisions_per_loss": {k: round(v) for k, v in strong.items()},
            "net_decisions_per_loss": {k: round(v) for k, v in net.items()},
            "strong_ms_per_decision_per_worker": ms, "workers": workers,
            "seconds_per_loss_per_worker": round(per_loss_s, 1),
            "hours": round(n_losses * per_loss_s / workers / 3600, 2)}


# ---- main ---------------------------------------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ckpt", default="runs/alphazero_v3/latest.pt")
    p.add_argument("--team-eval", default=None, help="the tactician's W (default: the checkpoint's)")
    p.add_argument("--name", default=None, help="output: runs/autopsy_<name>/")
    p.add_argument("--out-dir", default=None, help="output directory (overrides --name)")
    p.add_argument("--losses", type=int, default=500, help="lost battles to collect")
    p.add_argument("--rounds", default="5,6", help="rounds whose runs are played (e.g. 3,5,6)")
    p.add_argument("--max-runs", type=int, default=100000, help="per worker: stop collecting after this many runs")
    p.add_argument("--battler", choices=("net", "search"), default="net",
                   help="the battler during collection: its network, greedy, or the evaluation's legal search")
    p.add_argument("--battler-sims", type=int, default=512)
    p.add_argument("--battler-dets", type=int, default=8)
    p.add_argument("--strong-sims", type=int, default=1024,
                   help="the strong battler's simulations per decision (2048: ~2x the time, see --estimate)")
    p.add_argument("--strong-dets", type=int, default=8)
    p.add_argument("--strong-mode", choices=("perfect", "legal"), default="perfect",
                   help="the strong battler of (a) and (c)")
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--search-batch", type=int, default=32)
    p.add_argument("--luck-seeds", type=int, default=32, help="(a) and (c) stage 2: battle RNG seeds (0: skip)")
    p.add_argument("--p0-net-seeds", type=int, default=32, help="(a) also with the network battler (0: skip)")
    p.add_argument("--mistake-modes", default="legal,perfect", help="(b) strong search modes ('' : skip)")
    p.add_argument("--mistake-seeds", type=int, default=8, help="(b) seeds per rewound decision")
    p.add_argument("--max-rewinds", type=int, default=4, help="(b) per mode: differing decisions evaluated")
    p.add_argument("--mistake-p", type=float, default=0.5, help="(b) P(win) under strong play at the mistake")
    p.add_argument("--mistake-delta", type=float, default=0.25,
                   help="(b) a decisive mistake needs P(strong move) - P(our move) >= this (same seeds)")
    p.add_argument("--label-mistake-mode", default="legal", choices=("legal", "perfect", "either"))
    p.add_argument("--team-screen-seeds", type=int, default=16, help="(c) stage 1, network battler, every option")
    p.add_argument("--team-top", type=int, default=2, help="(c) stage 2: alternatives played by the strong battler")
    p.add_argument("--team-seeds", type=int, default=16,
                   help="(c) stage 2: seeds (the first of the luck seeds; ours is (a) on the same ones)")
    p.add_argument("--team-gain", type=float, default=0.3)
    p.add_argument("--luck-p", type=float, default=0.8)
    p.add_argument("--bad-p", type=float, default=0.5)
    p.add_argument("--min-pattern-losses", type=int, default=5)
    p.add_argument("--workers", type=int, default=18)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--inference", choices=("gpu", "cpu"), default="gpu")
    p.add_argument("--deadline-ms", type=float, default=0.75)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--no-features", action="store_true", help="the tactician without its per-option features")
    p.add_argument("--collect-only", action="store_true")
    p.add_argument("--report-only", action="store_true")
    p.add_argument("--estimate", action="store_true", help="print the cost estimate of these settings and exit")
    return p


def config_of(args):
    return {"ckpt": args.ckpt, "team_eval": args.team_eval, "rounds": [int(x) for x in args.rounds.split(",")],
            "battler": args.battler, "battler_sims": args.battler_sims, "battler_dets": args.battler_dets,
            "strong_sims": args.strong_sims, "strong_dets": args.strong_dets, "strong_mode": args.strong_mode,
            "c_puct": args.c_puct, "search_batch": args.search_batch, "luck_seeds": args.luck_seeds,
            "p0_net_seeds": args.p0_net_seeds,
            "mistake_modes": [m for m in args.mistake_modes.split(",") if m], "mistake_seeds": args.mistake_seeds,
            "max_rewinds": args.max_rewinds, "mistake_p": args.mistake_p, "mistake_delta": args.mistake_delta,
            "label_mistake_mode": args.label_mistake_mode, "team_screen_seeds": args.team_screen_seeds,
            "team_top": args.team_top, "team_seeds": args.team_seeds, "team_gain": args.team_gain,
            "luck_p": args.luck_p, "bad_p": args.bad_p,
            "min_pattern_losses": args.min_pattern_losses, "workers": args.workers, "threads": args.threads,
            "seed": args.seed, "features": not args.no_features, "losses": args.losses}


def _pool(cfg, workers, fn, items, server):
    """Run fn over items: in this process (workers == 1) or in forked workers, each with its own inference slot."""
    global _W, _SERVER
    import multiprocessing as mp
    _SERVER = server
    try:
        if workers == 1:
            _W = _W if (_W is not None and _W.cfg is cfg) else Worker(cfg, 0)
            for x in items:
                yield fn(x)
            return
        ctx = mp.get_context("fork")
        slots = ctx.Queue()
        for i in range(workers):
            slots.put(i)
        with ctx.Pool(workers, initializer=_init_worker, initargs=(cfg, slots)) as pool:
            for r in pool.imap_unordered(fn, items):
                yield r
    finally:
        _SERVER = None


def collect(cfg, workers, server=None):
    target = int(math.ceil(cfg["losses"] / workers))
    jobs = [{"p": p, "n_jobs": workers, "target": target, "max_runs": cfg.get("max_runs", 100000)}
            for p in range(workers)]
    losses, wins, runs, t0 = [], [], 0, time.time()
    for r in _pool(cfg, workers, _collect_job, jobs, server):
        losses += r["losses"]
        wins += r["wins"]
        runs += r["runs"]
    losses.sort(key=lambda b: b["uid"])
    losses = losses[:cfg["losses"]] if len(losses) > cfg["losses"] else losses
    stats = {"runs": runs, "battles": len(losses) + len(wins), "seconds": time.time() - t0,
             "battle_win_rate": len(wins) / max(len(wins) + len(losses), 1)}
    return losses, wins, stats


def analyze(cfg, losses, workers, server=None, out_path=None):
    done = {a["uid"]: a for a in _load_jsonl(out_path)} if out_path else {}
    todo = [l for l in losses if l["uid"] not in done]
    t0 = time.time()
    for i, a in enumerate(_pool(cfg, min(workers, max(len(todo), 1)), _analyze_job, todo, server)):
        done[a["uid"]] = a
        if out_path:
            _dump_jsonl(out_path, [a], "a")
        print(f"autopsy: {i + 1}/{len(todo)} losses analyzed ({time.time() - t0:.0f} s): {a['uid']} "
              f"{a.get('label', a.get('error'))}", flush=True)
    return [done[l["uid"]] for l in losses if l["uid"] in done]


def main(argv=None):
    args = build_parser().parse_args(argv)
    from .search import in_training
    if in_training():
        # this process is marked as training (PYB_TRAINING): the perfect-information battler cannot exist here
        raise SystemExit("rl.autopsy is a diagnostic entry point: it cannot run inside a training process")
    cfg = config_of(args)
    cfg["max_runs"] = args.max_runs
    if args.estimate:
        print(json.dumps(estimate(cfg, cfg["losses"]), indent=1))
        return
    out_dir = args.out_dir or os.path.join("runs", f"autopsy_{args.name or time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)
    col_path, ana_path = os.path.join(out_dir, "collect.jsonl"), os.path.join(out_dir, "analysis.jsonl")
    meta_path = os.path.join(out_dir, "collect_meta.json")
    server = None
    try:
        if os.path.exists(col_path):
            recs = _load_jsonl(col_path)
            losses = [r for r in recs if not r["won"]]
            wins = [r for r in recs if r["won"]]
            collect_stats = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
            print(f"autopsy: {len(losses)} losses / {len(wins)} wins from {col_path}", flush=True)
        else:
            if args.report_only:
                raise SystemExit(f"nothing collected in {out_dir}")
            if args.inference == "gpu" and args.workers > 1:
                from .inference import InferenceServer
                server = InferenceServer(args.ckpt, n_clients=args.workers, deadline_ms=args.deadline_ms)
            losses, wins, collect_stats = collect(cfg, args.workers, server)
            _dump_jsonl(col_path, losses + wins)
            json.dump(collect_stats, open(meta_path, "w"), indent=1)
            print(f"autopsy: collected {len(losses)} losses, {len(wins)} wins in {collect_stats['runs']} runs "
                  f"({collect_stats['seconds']:.0f} s; battle win rate {collect_stats['battle_win_rate']:.3f})",
                  flush=True)
        if args.collect_only:
            return
        if args.report_only:
            analyses = _load_jsonl(ana_path)
        else:
            if server is None and args.inference == "gpu" and args.workers > 1:
                from .inference import InferenceServer
                server = InferenceServer(args.ckpt, n_clients=args.workers, deadline_ms=args.deadline_ms)
            analyses = analyze(cfg, losses, args.workers, server, ana_path)
    finally:
        if server is not None:
            server.close()
    summary = write_outputs(out_dir, cfg, losses, wins, analyses, collect_stats)
    print(json.dumps({k: {c: round(v["frac"], 3) if v["frac"] is not None else None for c, v in summary[k].items()}
                      for k in ("labels", "labels_noland")}, indent=1))
    print(f"autopsy: outputs in {out_dir}")
    return summary


if __name__ == "__main__":
    main()

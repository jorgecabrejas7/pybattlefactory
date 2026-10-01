"""Simulator replica of the play study (docs/RL_DECISIONS.md §23): the alphazero_v3 agents play whole runs in the
simulator, as in the real game (from round 1 until the first loss), and every decision is logged with
rl/playlog.py's PlayLogger, exactly as the emulator runner logs the real game.

    python -m rl.play_sim --mode net --battles 20000 --procs 3 --out runs/playstudy_v3
    python -m rl.play_sim --mode search --hours 3 --procs 3 --out runs/playstudy_v3
    python -m rl.play_sim --mode net --late-runs 500 ...      # + runs starting at rounds 5-6 (flagged)
    python -m rl.play_sim --mode search --procs 1 --run-offset 50000 --suffix _b ...   # a second launch, other runs

Runs: run i (start round k) has the environment seed stable_seed("play_sim", seed, k, i): the same runs for both
modes (paired). A run starts at the rental screen of round k (k = 1, or 5 / 6 for --late-runs, with a random
feasible rental counter as rl.envs.FactoryEnv draws it) and ends at its first lost battle (or a battle that
reaches 300 decisions, logged as a loss with "truncated"), or after --max-battles battles (then "capped").
Each process writes its own file <out>/<mode>/<mode>_p<j>.jsonl.gz; runs are dealt round-robin to processes.
Agents: rl/play_agent.py (tactician = network with W / planner features, greedy; battler = network alone, greedy,
or the legal search with 512 simulations, 8 determinizations and the "experience" opponent sampler).

Logged beyond the agent's own policy (rl/play_agent.py): per battle decision the events (rl/autopsy.py's parse of
the game's messages) since our previous decision ("events_prev"); per battle end: won, turns, decisions, our HP left
(fraction of our team's max HP, our own Pokemon: player knowledge), the event counts and the luck balance; under
"truth" (analysis only): the opponent's true team (battle end) and the true HP / set of its active Pokemon (each
battle decision).
"""

import argparse
import json
import multiprocessing as mp
import os
import time
from collections import Counter

import numpy as np

from .autopsy import luck_balance, parse_messages, stable_seed, team_of
from .playlog import PlayLogger

_BATTLE_PHASES = None


def _phases():
    from pybattle.backend import Phase
    return (Phase.BATTLE, Phase.FORCED_SWITCH)


class _MessageTap:
    """Collects the game's battle messages after every backend action (Gen3Game.log_messages)."""

    def __init__(self, backend):
        self.be = backend
        self.events = []
        backend.game.log_messages(True)
        self._orig = backend.act
        backend.act = self.act

    def act(self, action):
        self._orig(action)
        self.events += parse_messages(self.be.game.take_messages())

    def take(self):
        out, self.events = self.events, []
        return out


def _own_hp(game):
    from pybattle.emu.decode import SYMBOLS as S, decode_party
    party = decode_party(game.read(S.addr("gPlayerParty"), 300))[:3]
    return [[m.hp, m.max_hp] for m in party]


def _true_enemy_active(game):
    import struct
    from pybattle.emu.decode import SYMBOLS as S, decode_battle_mon
    m = decode_battle_mon(game.read(S.addr("gBattleMons") + 0x58, 0x58))
    idx = struct.unpack("<2H", game.read(S.addr("gBattlerPartyIndexes"), 4))[1]
    return {"slot": int(idx), "species": m.species, "hp": m.hp, "max_hp": m.max_hp, "moves": list(m.moves),
            "item": m.item}


def _true_enemy_team(game):
    from pybattle.emu.decode import SYMBOLS as S, decode_party
    return team_of(decode_party(game.read(S.addr("gEnemyParty"), 300)))


def _ev_list(events):
    return [[e["ev"], e["side"], e["move"]] for e in events]


def play_run(agent, logger, env_seed, start_round, max_battles=200, late=False):
    """One run (until the first loss, or max_battles battles) logged decision by decision. -> summary dict."""
    from pybattle.backend import Phase
    from .envs import NO_ACTION, FactoryEnv
    env = FactoryEnv(env_seed, win_streak=7 * (start_round - 1))
    tap = _MessageTap(env.backend)
    be = env.backend
    ev = env._advance(NO_ACTION)
    battles = wins = 0
    battle_events = []
    opp_team = None
    last_turn = -1
    decisions_in_battle = 0
    t0 = time.time()
    end = None
    while True:
        st = ev["stats"].get("battle")
        if st is not None and st["won"] is not None or ev["truncate"]:
            won = bool(st and st["won"])
            truncated = bool(ev["truncate"])
            battle_events += tap.take()
            counts = Counter(("own_" if e["side"] == 0 else "opp_") + e["ev"] for e in battle_events)
            hp = _own_hp(be.game)
            info = {"turns": last_turn + 1, "decisions": decisions_in_battle, "events": dict(counts),
                    "luck_balance": luck_balance(counts), "own_hp": hp, "truncated": truncated,
                    "truth": {"opp_team": opp_team}}
            logger.battle_end(won, info)
            battles += 1
            wins += int(won)
            battle_events, opp_team, last_turn, decisions_in_battle = [], None, -1, 0
            if not won:
                end = "loss" if not truncated else "truncated"
                break
            if battles >= max_battles:
                end = "capped"
                break
        kind = ev["kind"]
        ctx = env._ctx()
        if kind in ("rental", "swap"):
            pre = tap.take()                    # (nothing in practice: no battle runs between the screens)
            action, policy, extras = agent.tactician(be, kind, ev["obs"], ctx)
            if pre:
                extras["events_prev"] = _ev_list(pre)
            logger.decision(kind, be.view(), be.run_info(), policy, action, extras)
        else:
            if opp_team is None:
                opp_team = _true_enemy_team(be.game)
            evs = tap.take()
            battle_events += evs
            view = be.view()
            action, policy, extras = agent.battler(be, ev["obs"], ctx, env.decisions - 1)
            extras["events_prev"] = _ev_list(evs)
            extras["truth"] = {"enemy_active": _true_enemy_active(be.game)}
            k = "forced_switch" if be.phase == Phase.FORCED_SWITCH else "battle"
            logger.decision(k, view, be.run_info(), policy, action, extras)
            last_turn = max(last_turn, int(view.turn))
            decisions_in_battle += 1
        ev = env.step(action)
    info = be.run_info()
    logger.run_end(7 * (start_round - 1) + wins, {"start_round": start_round, "start_streak": 7 * (start_round - 1),
                                                   "wins": wins, "battles": battles, "end": end, "late_start": late,
                                                   "env_seed": env_seed, "seconds": round(time.time() - t0, 2),
                                                   "rents": info.rents})
    return {"battles": battles, "wins": wins, "end": end}


# ---- workers -----------------------------------------------------------------------------------------------------

def _worker(job):
    import torch
    torch.set_num_threads(1)
    try:
        os.nice(job.get("nice", 0))
    except OSError:
        pass
    from .play_agent import PlayAgent
    agent = PlayAgent(job["ckpt"], battler=job["mode"], sims=job["sims"], dets=job["dets"],
                      opponent_model=job["opponent_model"], seed=stable_seed("agent", job["seed"], job["p"]),
                      probes=job["probes"], n_hint_probes=job["n_hint_probes"],
                      full_hint_probes=job["full_hint_probes"])
    meta = dict(agent.meta, source="sim", process=job["p"], seed=job["seed"], mode=job["mode"])
    logger = PlayLogger(job["path"], meta)
    t0 = time.time()
    out = {"runs": 0, "battles": 0, "wins": 0, "late_runs": 0, "decisions": 0}
    try:
        for (k, i, late) in job["runs"]:
            if out["battles"] >= job["battle_target"] or (job["deadline"] and time.time() > job["deadline"]):
                break
            seed = stable_seed("play_sim", job["seed"], k, i) & 0x7FFFFFFF
            try:
                r = play_run(agent, logger, seed, k, job["max_battles"], late)
            except Exception as e:                  # keep going; the run is closed and flagged (analysis skips it)
                import traceback
                logger.run_end(-1, {"error": repr(e), "traceback": traceback.format_exc(), "start_round": k,
                                    "late_start": late, "env_seed": seed})
                out["errors"] = out.get("errors", 0) + 1
                continue
            out["runs"] += 1
            out["late_runs"] += int(late)
            out["battles"] += r["battles"]
            out["wins"] += r["wins"]
    finally:
        logger.close()
    out["decisions"] = agent.n_decisions
    out["seconds"] = time.time() - t0
    return out


def run(args):
    out_dir = os.path.join(args.out, args.mode)
    os.makedirs(out_dir, exist_ok=True)
    o = args.run_offset
    runs = [(1, o + i, False) for i in range(args.max_runs)]
    late = [(5 + (i % 2), o + i, True) for i in range(args.late_runs)]
    procs = max(1, args.procs)
    deadline = time.time() + args.hours * 3600 if args.hours else 0
    jobs = []
    for p in range(procs):
        mine = late[p::procs] + runs[p::procs]           # the late-start runs first (few, then the round-1 runs)
        jobs.append({"p": p, "ckpt": args.ckpt, "mode": args.mode, "sims": args.sims, "dets": args.dets,
                     "opponent_model": args.opponent_model, "seed": args.seed, "probes": not args.no_probes,
                     "n_hint_probes": args.hint_probes, "full_hint_probes": not args.input_only_probes,
                     "path": os.path.join(out_dir, f"{args.mode}_p{p}{args.suffix}.jsonl.gz"), "runs": mine,
                     "battle_target": int(np.ceil(args.battles / procs)),
                     "deadline": deadline, "max_battles": args.max_battles, "nice": args.nice})
    t0 = time.time()
    if procs == 1:
        results = [_worker(jobs[0])]
    else:
        with mp.get_context("fork").Pool(procs) as pool:
            results = pool.map(_worker, jobs)
    secs = time.time() - t0
    tot = {k: sum(r.get(k, 0) for r in results) for k in ("runs", "battles", "wins", "late_runs", "decisions", "errors")}
    tot.update(seconds=round(secs, 1), procs=procs, battles_per_hour=round(tot["battles"] / max(secs, 1e-9) * 3600, 1),
               mode=args.mode, files=[j["path"] for j in jobs])
    with open(os.path.join(out_dir, f"throughput{args.suffix}.json"), "w") as f:
        json.dump(tot, f, indent=1)
    print(json.dumps(tot, indent=1), flush=True)
    return tot


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ckpt", default="runs/alphazero_v3/latest.pt")
    p.add_argument("--mode", choices=("net", "search"), default="net")
    p.add_argument("--sims", type=int, default=512)
    p.add_argument("--dets", type=int, default=8)
    p.add_argument("--opponent-model", default="runs/opp_model_v3/model.pt")
    p.add_argument("--battles", type=int, default=20000, help="stop starting runs after this many battles in total")
    p.add_argument("--hours", type=float, default=0.0, help="also stop starting runs after this many hours (0: no)")
    p.add_argument("--max-runs", type=int, default=100000, help="round-1 runs available")
    p.add_argument("--late-runs", type=int, default=0, help="extra runs starting at rounds 5 / 6 (flagged), first")
    p.add_argument("--max-battles", type=int, default=200, help="a run stops after this many battles (capped)")
    p.add_argument("--procs", type=int, default=3)
    p.add_argument("--nice", type=int, default=10)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--hint-probes", type=int, default=3, help="other hint types probed per tactician decision")
    p.add_argument("--input-only-probes", action="store_true", help="hint probes on the network input only")
    p.add_argument("--no-probes", action="store_true")
    p.add_argument("--out", default="runs/playstudy_v3")
    p.add_argument("--suffix", default="", help="file name suffix (several launches into the same folder)")
    p.add_argument("--run-offset", type=int, default=0,
                   help="first run index (a second launch into the same folder takes other runs: e.g. 50000)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    main()

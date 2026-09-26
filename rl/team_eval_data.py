"""Training data for the team evaluator (docs/RL_DECISIONS.md §19.2): (team, round, battle type) -> won / lost.

    python -m rl.team_eval_data --battler-ckpt runs/alphazero_v1/latest.pt --out runs/team_eval_v1/data \
        --workers 18 --minutes 110

A sample ("case") is one team in one round and one battle type, played in `--battles` (default 16) independent
simulated battles against fresh opponents:
    1. round r uniform in 1..6; type: normal (battle b uniform in 1..6) or last (battle 7); in rounds 3 and 6 the
       7th battle is Noland's, so the types there are normal / Noland;
    2. SimBackend.reset at streak 7(r-1)+b-1 with a random feasible rental counter [r-1, 7(r-1)] (as FactoryEnv):
       the game generates the round's 6 rentals (its pool, rents rank, no repeated species); a random legal trio
       and lead is rented (Gen3Game.factory_rent: party [lead, j, k]); the battle-number field of the save is set
       to b-1 so that the battler's streak context is the one of that battle;
    3. teams after swaps: with prob --p-swap one (or, 30 % of those, two) of our Pokemon is replaced by a set from
       the round's opponent pool (the Pokemon a swap brings: the set built as the game built the opponent: its EVs
       and nature, IV 3; species not already on the team); flagged "swapped" so that training can hide what the
       swap screen would not show;
    4. each battle: a clone of the game, the opponent team replaced by one drawn with rl/determinize.py
       sample_factory_sets (the round's pool, the exclusions of its generation, the set's EVs / nature, the game's
       IV for that battle: 3, 6 in battle 7, Noland's row), the battle RNG reseeded; the battler network plays it,
       greedy, without search (rl/tactician_search.BattleSimulator mechanics: moves without a choice are move 0;
       cut at --max-decisions and flagged truncated).
All battles of a worker advance in lockstep (one batched evaluator call per step, new cases are started as old
ones finish); the evaluator is the GPU inference server (--inference gpu) or a CPU copy.

Output: one .npz per worker chunk: per team round, btype (0 normal, 1 last, 2 Noland), battle (1..7), rents,
species / moves / item / ability / stats [3, ...] (party order, slot 0 leads), set_id, swapped; per battle b_team,
b_won, b_trunc, b_decisions. meta.json: battler checkpoint, settings, throughput.
"""

import argparse
import json
import multiprocessing as mp
import os
import random
import time

import numpy as np
import torch

from pybattle.backend import OpponentKnowledge, Phase, SimBackend
from pybattle.emu.decode import SB2_FRONTIER_CUR_BATTLE_NUM, SYMBOLS as S, decode_party
from pybattle.view import OwnMon, possible_abilities
from . import determinize as DET
from . import encode

_PLAYER = S.addr("gPlayerParty")
_ENEMY = S.addr("gEnemyParty")
MON_BYTES = 100


def _round_type(rng):
    r = rng.randint(1, 6)
    if r in (3, 6):
        t = 2 if rng.random() < 0.4 else 0
    else:
        t = 1 if rng.random() < 0.3 else 0
    return r, t


def _pick_rental(view, rng):
    while True:
        pick = rng.sample(range(6), 3)
        if len({view.candidates[i].species for i in pick}) == 3:
            return tuple(pick)


def build_swapped(game, own, slot, set_id, rng):
    """Our party bytes with `slot` replaced by Battle Frontier set `set_id` built as the game builds an opponent
    (set EVs / nature, IV 3: what a swap brings), from our own party as the template (level, OT, friendship 0)."""
    game.write_party(1, own)
    bit = rng.randrange(2) if len(possible_abilities(int(DET.SET_SPECIES[set_id]))) > 1 else 0
    game.determinize([DET.set_spec(slot, set_id, DET.fixed_iv(0, False), bit, 1.0)])
    mon = game.read(_ENEMY + slot * MON_BYTES, MON_BYTES)
    return own[:slot * MON_BYTES] + mon + own[(slot + 1) * MON_BYTES:]


def make_case(rng, p_swap=0.5, open_level=True, round_type=None):
    """A team ready to battle. -> (backend at the battle's start (opponent not yet replaced), our party bytes,
    OpponentKnowledge, team record dict)."""
    r, t = round_type or _round_type(rng)
    b = 6 if t else rng.randrange(6)                    # 0-based battle index in the round
    k = r - 1
    be = SimBackend(open_level=open_level, max_turns=10 ** 9)
    be.reset(seed=rng.getrandbits(32), win_streak=7 * k + b, rents_count=rng.randint(k, 7 * k))
    if be.phase != Phase.RENTAL:
        raise RuntimeError("expected a rental screen after reset")
    info = be.run_info()
    if info.noland != (t == 2):
        raise RuntimeError(f"round {r} battle {b + 1}: noland {info.noland}, type {t}")
    view = be.view()
    lead, j, kk = _pick_rental(view, rng)
    g = be.game
    if not g.factory_rent(lead, j, kk):
        raise RuntimeError("rental refused")
    own = bytes(g.read(_PLAYER, 3 * MON_BYTES))
    set_ids = [view.frontier_ids[i] for i in (lead, j, kk)]
    swapped = [False] * 3
    pool = DET.pool(k, open_level)
    n_sw = 0 if rng.random() >= p_swap else (2 if rng.random() < 0.3 else 1)
    replaced = []
    for slot in rng.sample(range(3), n_sw):
        team_sp = {int(m.species) for m in decode_party(own)[:3]}
        cand = pool[~np.isin(DET.SET_SPECIES[pool], sorted(team_sp))]
        sid = int(cand[rng.randrange(len(cand))])
        replaced.append(int(decode_party(own)[slot].species))
        own = build_swapped(g, own, slot, sid, rng)
        set_ids[slot], swapped[slot] = sid, True
    g.write_party(0, own)
    g.write_party(1, own)
    g.write_saveblock2(SB2_FRONTIER_CUR_BATTLE_NUM, int(b).to_bytes(2, "little"))
    mons = decode_party(own)[:3]
    own_sp = [int(m.species) for m in mons]
    if t == 2:
        # Noland excludes sets: our team after the swap and the defeated team (3 sets of the pool)
        others = [int(x) for x in rng.sample(list(pool), 3)]
        know = OpponentKnowledge(k, True, set_ids=frozenset(set_ids + others), battle=b)
    elif n_sw == 0:
        know = OpponentKnowledge(k, False, frozenset(int(m.species) for m in view.candidates), battle=b)
    else:
        # generated at the swap screen: our team before the swap + the defeated team (which held the traded-in ones)
        defeated = [own_sp[i] for i in range(3) if swapped[i]]
        while len(defeated) < 3:
            s = int(DET.SET_SPECIES[pool[rng.randrange(len(pool))]])
            if s not in defeated and s not in own_sp:
                defeated.append(s)
        before = [own_sp[i] for i in range(3) if not swapped[i]] + replaced
        know = OpponentKnowledge(k, False, frozenset(before + defeated), battle=b)
    om = [OwnMon.from_party(m) for m in mons]
    rec = {"round": r, "btype": t, "battle": b + 1, "rents": info.rents,
           "species": [m.species for m in om], "moves": [(list(m.moves) + [0] * 4)[:4] for m in om],
           "item": [m.item for m in om], "ability": [m.ability for m in om],
           "stats": [[m.max_hp] + list(m.stats) for m in om], "set_id": set_ids, "swapped": swapped}
    return be, own, know, rec


def start_battle(be, own, know, rng, open_level=True):
    """A clone of the case's backend with a fresh opponent drawn from the Factory's sets and a reseeded RNG."""
    from .tactician_search import _unseen_view
    c = be.clone()
    g = c.game
    g.write_party(1, own)
    level = decode_party(own)[0].level
    specs = DET.sample_factory_sets(_unseen_view(level), know, rng, open_level)
    if not g.determinize(specs):
        raise RuntimeError("the opponent team was not fully rebuilt")
    g.set_rng(rng.getrandbits(32))
    c.opponent_knowledge = know
    c.phase = Phase.BATTLE
    return c


class Generator:
    """Keeps `concurrency` simulated battles alive, stepping them in lockstep with one evaluator call."""

    def __init__(self, evaluator, seed, battles_per_team=16, concurrency=256, max_decisions=300, p_swap=0.5):
        from .tactician_search import BattleSimulator
        self.sim = BattleSimulator(evaluator, max_decisions)
        self.rng = random.Random(seed)
        self.bpt, self.conc, self.p_swap = battles_per_team, concurrency, p_swap
        self.teams, self.battles = [], []                    # finished records
        self.active = []
        self.n_teams = 0

    def _add_case(self):
        from .tactician_search import _Sim
        be, own, know, rec = make_case(self.rng, self.p_swap)
        tid = len(self.teams)
        self.teams.append(rec)
        for _ in range(self.bpt):
            s = _Sim(start_battle(be, own, know, self.rng), tid, self.sim.cpp)
            self.sim._run(s)
            self.active.append(s)

    def step(self, refill=True):
        while refill and len(self.active) + self.bpt <= self.conc:
            self._add_case()
        sim = self.sim
        batch, who = [], []
        for s in self.active:
            x = sim._to_choice(s)
            if x is not None:
                batch.append(x)
                who.append(s)
        if batch:
            pri, val = sim.evaluator({k: np.stack([x[k] for x in batch]) for k in batch[0]})
            sim.net_calls += 1
            sim.steps += len(batch)
            for s, x, p in zip(who, batch, pri):
                if s.trunc:
                    s.done, s.won = True, False
                    continue
                a = int(np.argmax(np.where(x["mask"], p, -1.0)))
                try:
                    if a < 4:
                        s.g.choose_move(a)
                    else:
                        s.g.choose_switch(a - 4)
                except (RuntimeError, ValueError, IndexError):
                    s.done, s.won = True, False
                    sim.errors += 1
                    continue
                s.ndec += 1
                sim._run(s)
        still = []
        for s in self.active:
            if s.done:
                self.battles.append((s.tag, s.won, s.trunc, s.ndec))
            else:
                still.append(s)
        self.active = still
        return len(batch)

    def take(self):
        """Finished teams (every battle done) and their battles, re-indexed; unfinished ones stay."""
        open_ids = {s.tag for s in self.active}
        done_ids = [i for i in range(len(self.teams)) if i not in open_ids and self.teams[i] is not None]
        remap = {t: n for n, t in enumerate(done_ids)}
        teams = [self.teams[i] for i in done_ids]
        bat = [b for b in self.battles if b[0] in remap]
        self.battles = [b for b in self.battles if b[0] not in remap]
        for i in done_ids:
            self.teams[i] = None
        return teams, [(remap[t], w, tr, nd) for t, w, tr, nd in bat]


def to_arrays(teams, battles):
    out = {k: np.array([t[k] for t in teams]) for k in ("round", "btype", "battle", "rents")}
    for k in ("species", "item", "ability", "set_id"):
        out[k] = np.array([t[k] for t in teams], np.int32)
    out["moves"] = np.array([t["moves"] for t in teams], np.int32)
    out["stats"] = np.array([t["stats"] for t in teams], np.int16)
    out["swapped"] = np.array([t["swapped"] for t in teams], bool)
    for k in ("round", "btype", "battle"):
        out[k] = out[k].astype(np.int8)
    out["rents"] = out["rents"].astype(np.int16)
    b = np.array(battles, np.int64).reshape(-1, 4)
    out["b_team"] = b[:, 0].astype(np.int32)
    out["b_won"] = b[:, 1].astype(bool)
    out["b_trunc"] = b[:, 2].astype(bool)
    out["b_decisions"] = b[:, 3].astype(np.int16)
    return out


def _worker(wid, cfg, server, out_dir, t_end, q):
    from .search import mark_training
    mark_training()
    torch.set_num_threads(1)
    encode.set_version(cfg["encode_version"])
    if server is not None:
        ev = server.client(wid).evaluate
    else:
        from .alphazero import local_battler_evaluator
        from .model import FactoryNet
        ck = torch.load(cfg["ckpt"], map_location="cpu", weights_only=False)
        a = ck["args"]
        net = FactoryNet(a["d_emb"], a["d"], a["layers"], a["heads"], share=a.get("share", "embeddings")).eval()
        net.load_state_dict(ck["net"])
        ev = local_battler_evaluator(net, (ck.get("value_norm") or {}).get("battler"))
    gen = Generator(ev, cfg["seed"] * 1_000_003 + 7919 * wid, cfg["battles"], cfg["concurrency"],
                    cfg["max_decisions"], cfg["p_swap"])
    chunk, last_save = 0, time.time()
    n_b = 0

    def save():
        nonlocal chunk
        teams, bat = gen.take()
        if teams:
            np.savez_compressed(os.path.join(out_dir, f"w{wid:02d}_{chunk:04d}.npz"), **to_arrays(teams, bat))
            chunk += 1
        return len(teams), len(bat)

    while True:
        refill = time.time() < t_end
        if not refill and not gen.active:
            break
        gen.step(refill)
        if time.time() - last_save > cfg["save_every"]:
            nt, nb = save()
            n_b += nb
            q.put((wid, nt, nb, gen.sim.steps, gen.sim.errors))
            last_save = time.time()
    nt, nb = save()
    q.put((wid, nt, nb, gen.sim.steps, gen.sim.errors))
    q.put((wid, None, None, None, None))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--battler-ckpt", default="runs/alphazero_v1/latest.pt")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=18)
    p.add_argument("--minutes", type=float, default=110.0)
    p.add_argument("--battles", type=int, default=16, help="battles per team (independent opponents)")
    p.add_argument("--concurrency", type=int, default=256, help="simulated battles alive per worker")
    p.add_argument("--max-decisions", type=int, default=300)
    p.add_argument("--p-swap", type=float, default=0.5)
    p.add_argument("--inference", choices=("gpu", "cpu"), default="gpu")
    p.add_argument("--save-every", type=float, default=300.0)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    from .search import mark_training
    mark_training()                    # factory_sets opponents: a training-data process
    os.makedirs(a.out, exist_ok=True)
    ck = torch.load(a.battler_ckpt, map_location="cpu", weights_only=False)
    cfg = dict(ckpt=a.battler_ckpt, encode_version=ck["args"].get("encode_version", 2), battles=a.battles,
               concurrency=a.concurrency, max_decisions=a.max_decisions, p_swap=a.p_swap, seed=a.seed,
               save_every=a.save_every)
    encode.set_version(cfg["encode_version"])
    server = None
    if a.inference == "gpu":
        from .inference import InferenceServer
        server = InferenceServer(a.battler_ckpt, n_clients=a.workers)
    t0 = time.time()
    t_end = t0 + 60 * a.minutes
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(w, cfg, server, a.out, t_end, q), daemon=True)
             for w in range(a.workers)]
    for pr in procs:
        pr.start()
    teams = battles = 0
    steps = {}
    alive = a.workers
    last = t0
    while alive:
        wid, nt, nb, st, err = q.get()
        if nt is None:
            alive -= 1
            continue
        teams += nt
        battles += nb
        steps[wid] = (st, err)
        if (time.time() - last > 60 and len(steps) == a.workers) or alive < a.workers:
            el = time.time() - t0
            print(f"[{el / 60:.1f} min] {teams:,} teams, {battles:,} battles ({battles / el:.0f} battles/s, "
                  f"{sum(s for s, _ in steps.values()) / el:.0f} decisions/s), errors "
                  f"{sum(e for _, e in steps.values())}", flush=True)
            last = time.time()
    for pr in procs:
        pr.join()
    el = time.time() - t0
    meta = dict(vars(a), encode_version=cfg["encode_version"], teams=teams, battles=battles, seconds=el,
                battles_per_s=battles / el, decisions_per_s=sum(s for s, _ in steps.values()) / el)
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(json.dumps(meta, indent=1))
    if server is not None:
        server.close()


if __name__ == "__main__":
    main()

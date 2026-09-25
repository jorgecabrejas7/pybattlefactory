"""Timing of the search core: sim_step, determinize, clone, MctsTree select+backup.

    python scripts/bench_search_core.py
"""

import itertools
import os
import random
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pybattle.backend import SimBackend  # noqa: E402
from pybattle.emu.decode import SYMBOLS as S, decode_party  # noqa: E402
from pybattle.pybattle_native import Gen3Game, MctsTree  # noqa: E402

D = Gen3Game.Decision


def start(seed):
    b = SimBackend()
    b.reset(seed=seed, win_streak=35)
    species = [m.species for m in b.view().candidates]
    b.act(next(t for t in itertools.combinations(range(6), 3) if len({species[i] for i in t}) == 3))
    return b.game


def bench_sim_step(n_battles=300):
    rng = random.Random(0)
    steps, t_step, t_clone = 0, 0.0, 0.0
    for seed in range(n_battles):
        base = start(seed)
        t0 = time.perf_counter()
        g = base.clone()
        t_clone += time.perf_counter() - t0
        decision, n = D.ACTION, 0
        while decision in (D.ACTION, D.SWITCH) and n < 300:   # random play can stall forever
            n += 1
            if decision == D.SWITCH:
                party = decode_party(g.read(S.addr("gPlayerParty"), 300))
                active = struct.unpack("<H", g.read(S.addr("gBattlerPartyIndexes"), 2))[0]
                bench = [i for i, m in enumerate(party[:3]) if i != active and m.hp > 0]
                t0 = time.perf_counter()
                decision, _ = g.sim_step(1, rng.choice(bench))
                t_step += time.perf_counter() - t0
                steps += 1
                continue
            unusable = g.unusable_moves(0)
            moves = [i for i in range(4) if not unusable & (1 << i)] or [0]
            t0 = time.perf_counter()
            decision, _ = g.sim_step(0, rng.choice(moves))
            t_step += time.perf_counter() - t0
            steps += 1
    print(f"sim_step:    {t_step / steps * 1e6:7.1f} us  ({steps} steps)")
    print(f"clone:       {t_clone / n_battles * 1e6:7.1f} us")


def bench_determinize(n=5000):
    base = start(1)
    rng = random.Random(0)
    specs = [[(s, rng.randrange(372, 882), 3, rng.randrange(2), rng.choice([-1.0, 0.5])) for s in range(3)]
             for _ in range(64)]
    games = [base.clone() for _ in range(64)]
    t0 = time.perf_counter()
    for i in range(n):
        games[i % 64].determinize(specs[i % 64], i)
    dt = time.perf_counter() - t0
    print(f"determinize: {dt / n * 1e6:7.1f} us  (3 slots, incl. game swap-in)")
    g = base.clone()
    t0 = time.perf_counter()
    for i in range(n):
        g.determinize(specs[i % 64], i)
    print(f"determinize: {(time.perf_counter() - t0) / n * 1e6:7.1f} us  (3 slots, same game)")


def bench_tree(nodes=1000, batch=8):
    rng = random.Random(0)
    tree = MctsTree()
    priors = [1.0] * 7
    legal = [True] * 7
    ops, t = 0, 0.0
    total_t0 = time.perf_counter()
    while tree.node_count() < nodes:
        t0 = time.perf_counter()
        leaves = tree.select(batch)
        t += time.perf_counter() - t0
        for leaf, _, _ in leaves:
            p = [rng.random() for _ in priors]
            tree.expand(leaf, p, legal)
            t0 = time.perf_counter()
            tree.backup(leaf, rng.random())
            t += time.perf_counter() - t0
            ops += 1
    # steady state at ~1000 nodes: select+backup on a tree that keeps growing slowly
    t, ops = 0.0, 0
    for _ in range(2000):
        t0 = time.perf_counter()
        leaves = tree.select(1)
        leaf = leaves[0][0]
        tree.backup(leaf, rng.random())
        t += time.perf_counter() - t0
        tree.expand(leaf, priors, legal)
        ops += 1
    print(f"select(1)+backup at {nodes}-{tree.node_count()} nodes: {t / ops * 1e6:5.2f} us")
    t0 = time.perf_counter()
    leaves = tree.select(batch)
    for leaf, _, _ in leaves:
        tree.backup(leaf, 0.5)
    print(f"select({batch})+{len(leaves)} backups: {(time.perf_counter() - t0) * 1e6:5.2f} us")


if __name__ == "__main__":
    bench_sim_step()
    bench_determinize()
    bench_tree()

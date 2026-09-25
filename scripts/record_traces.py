#!/usr/bin/env python3
"""Record Battle Factory battle traces from the real game (headless mGBA).

Each trace starts a fresh Open Level singles challenge from the lobby save, waits a random
number of frames first (the overworld RNG advances every frame, so rentals and opponents
differ), rents three random mons, and plays the first battle with a random policy.

Usage: record_traces.py --rom R.gba --save R.sav --out tests/diff/traces -n 20 [--seed 0]
"""

import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pybattle.emu.driver import FactoryDriver  # noqa: E402
from pybattle.emu.trace import random_policy, record_battle, save_trace  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rom", required=True)
    ap.add_argument("--save", required=True)
    ap.add_argument("--out", default="tests/diff/traces")
    ap.add_argument("-n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--streaks", default="0",
                    help="comma list of win streaks to sample from, e.g. 0,7,14,20,28,41 (20/41: Factory Head)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    d = FactoryDriver(args.rom, args.save)
    d.boot_to_overworld()
    lobby = d.save_state()
    rng = random.Random(args.seed)
    for i in range(args.n):
        t = time.time()
        d.load_state(lobby)
        d.emu.run_frames(rng.randrange(1, 2000))
        streak = rng.choice([int(x) for x in args.streaks.split(",")])
        d.set_factory_streak(streak, rents=rng.randrange(0, 50))
        d.start_challenge(open_level=True)
        d.pick_rentals(rng.sample(range(6), 3))
        trace = record_battle(d, random_policy(rng))
        path = os.path.join(args.out, f"trace_s{args.seed}_{i:04d}_streak{streak}.json")
        save_trace(trace, path)
        print(f"{path}: streak {streak} trainer {trace['start']['trainer_id']} ai {trace['start'].get('ai_flags', 0):#x}, "
              f"{len(trace['decisions'])} decisions, outcome {trace['end']['outcome']}, "
              f"{time.time() - t:.1f}s", flush=True)


if __name__ == "__main__":
    main()

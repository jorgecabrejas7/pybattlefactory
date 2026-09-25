"""Benchmark decision-time search: ms per decision of rl/search.py SearchBattler, impl="python" vs impl="cpp"
(C++ loop, with the C++ ObsMemory observer when built and with the Python observer called from C++), same
decisions, same seeds, the PPO network on the CPU. Also the share of the time spent in the network.

    python scripts/bench_search.py [--sims 256] [--k 8] [--batch 32] [--decisions 30] [--threads 1]
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rl import search                      # noqa: E402
from rl.policy import Policy               # noqa: E402
from test_search import _decisions         # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=(sorted(glob.glob("runs/ppo_joint_v3/latest.pt")) or [None])[0])
    ap.add_argument("--sims", type=int, default=256)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--decisions", type=int, default=30)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=31)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    pol = Policy(args.ckpt, torch.device("cpu"))
    configs = [("python", "python")]
    if search.NativeSearcher is not None:
        if search.NativeObsMemory is not None:
            configs.append(("cpp", "cpp"))
        configs.append(("cpp", "python"))
    kw = dict(n_sims=args.sims, n_determinizations=args.k, batch=args.batch, seed=args.seed)
    battlers = {c: search.SearchBattler(pol, impl=c[0], observer=c[1], **kw) for c in configs}
    tot = {c: [0.0, 0.0, 0.0, 0] for c in configs}     # ms, ms in the network, ms in C++, leaves
    n = same = 0
    for b, v, tracker in _decisions(40, seed=args.seed, streaks=(0, 14, 28, 35)):
        x = search.encode.battle(v, search.run_ctx(b))
        if x["mask"].sum() <= 1:
            continue
        acts = []
        for c, sb in battlers.items():
            sb.act_on(b, tracker)
            st = sb.last_stats
            t = tot[c]
            t[0] += st["ms"]; t[1] += st["ms_eval"]; t[2] += st["ms_cpp"] or 0.0; t[3] += st["leaves"]
            acts.append((tuple(st["visits"]), st["action"]))
        same += len(set(acts)) == 1
        n += 1
        if n >= args.decisions:
            break
    print(f"{n} searched decisions, {args.sims} sims, K={args.k}, batch {args.batch}, torch threads {args.threads}; "
          f"identical visits and action in all configurations: {same}/{n}")
    base = tot[configs[0]][0] / n
    for c in configs:
        ms, ev, cpp, leaves = tot[c]
        print(f"  impl={c[0]:6s} observer={c[1]:6s}: {ms / n:7.1f} ms/decision (x{base / (ms / n):.2f}), "
              f"network {100 * ev / ms:4.1f}% ({ev / n:.1f} ms), C++ loop excl. network {cpp / n:.1f} ms, "
              f"{leaves / n:.0f} leaves")


if __name__ == "__main__":
    main()

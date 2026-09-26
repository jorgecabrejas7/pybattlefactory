"""Self-play throughput of rl.alphazero (battler decisions / s) with N workers and the GPU inference server.

    python -m rl.az_bench --workers 18 --decisions 200 [--sims 128 --t-budget 256] [--ckpt runs/X/latest.pt]
    python -m rl.az_bench --v2 --workers 18 --decisions 200 [--sims 256]      # rl.alphazero_v2's self-play

Uses a freshly initialized network unless --ckpt is given (an alphazero or PPO checkpoint of the same layout).
"""

import argparse
import os
import tempfile
import time

import numpy as np
import torch

from . import alphazero as AZ
from . import encode
from .model import FactoryNet


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=18)
    p.add_argument("--decisions", type=int, default=200, help="battler decisions per worker (after a warm-up)")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--inference", default="gpu", choices=("gpu", "cpu"))
    p.add_argument("--v2", action="store_true", help="alphazero_v2's self-play (Gumbel battler, no tactician search)")
    args, rest = p.parse_known_args()
    from .search import mark_training
    mark_training()
    if args.v2:
        from . import alphazero_v2 as V2
        a = V2.parse(rest)
        cfg, player, extra = V2.worker_cfg(a), V2.SelfPlayerV2, {"opt_feat": V2.N_FEAT, "version": 2}
    else:
        a = AZ.parse(rest)
        cfg, player, extra = AZ.worker_cfg(a), None, {}
    encode.set_version(a.encode_version)
    tmp = tempfile.mkdtemp(prefix="az_bench_")
    snap = os.path.join(tmp, "snapshot.pt")
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location="cpu")
        torch.save({"net": ck["net"], "args": dict(ck["args"], algo="alphazero", share="embeddings", **extra),
                    "value_norm": ck.get("value_norm")}, snap)
    else:
        net = FactoryNet(a.d_emb, a.d, a.layers, a.heads, share="embeddings", opt_feat=extra.get("opt_feat", 0))
        torch.save({"net": net.state_dict(), "args": dict(vars(a), algo="alphazero", share="embeddings", **extra),
                    "value_norm": None}, snap)
    server = None
    if args.inference == "gpu":
        from .inference import InferenceServer
        server = InferenceServer(snap, n_clients=args.workers)
    pool = AZ.WorkerPool(args.workers, cfg, server, seed=1, player=player)
    try:
        pool.broadcast("load", snap)
        pool.broadcast("play", args.warmup)
        t = time.time()
        res = pool.broadcast("play", args.decisions)
        dt = time.time() - t
    finally:
        pool.close()
        if server is not None:
            print("server:", server.stats())
            server.close()
    st = [s for _, s in res]
    tot = {k: sum(s.get(k, 0) for s in st) for k in ("b_decisions", "b_ms", "t_decisions", "t_ms", "t_sims",
                                                    "t_net_calls", "b_leaves", "b_searched", "battles")}
    n = tot["b_decisions"]
    print(f"workers {args.workers}  sims {a.sims}  dets {a.dets}  t_budget {getattr(a, 't_budget', None)}"
          + (f"  v2: Gumbel root, non-root {a.non_root}" if args.v2 else ""))
    print(f"battler decisions/s: {n / dt:.1f}  ({n} in {dt:.1f} s)")
    print(f"battler search: {tot['b_ms'] / max(n, 1):.1f} ms/decision, {tot['b_leaves'] / max(tot['b_searched'], 1):.0f} "
          f"leaves/searched decision")
    print(f"tactician: {tot['t_decisions']:.0f} decisions, {tot['t_ms'] / max(tot['t_decisions'], 1):.0f} ms/decision, "
          f"{tot['t_net_calls'] / max(tot['t_decisions'], 1):.0f} net calls/decision")
    wall = dt * args.workers * 1000
    print(f"share of worker time: battler {tot['b_ms'] / wall:.2f}, tactician {tot['t_ms'] / wall:.2f}, battles {tot['battles']:.0f}")
    return n / dt


if __name__ == "__main__":
    main()

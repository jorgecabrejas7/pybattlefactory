"""Offline battler dataset (docs/RL_DECISIONS.md §27): every battler decision of alphazero_v4 self-play, with the
training search's target, and a replay key from which the exact decision state can be rebuilt.

    python -m rl.offline_data record --ckpt runs/alphazero_v4/stopped_it1111.pt --out runs/v5_offline/main \
        --workers 12 --hours 3                                    # self-play exactly as v4's (Gumbel 256, factory_sets)
    python -m rl.offline_data consolidate --out runs/v5_offline/main     # shards -> one .npy per field
    python -m rl.offline_data subset --data runs/v5_offline/main --positions 400000 --out runs/v5_offline/subset_400k
    python -m rl.offline_data research --data runs/v5_offline/main --subset runs/v5_offline/subset_400k \
        --sims 1024 --prior factory_sets --out runs/v5_offline/var_s1024     # same positions, another search target
    python -m rl.offline_data check --data runs/v5_offline/main --n 200   # replay keys -> the recorded inputs, exactly

Recording (`record`): the alphazero_v4 self-play worker (rl.alphazero_v4.SelfPlayerV4, through rl.alphazero_v3's
pieces) with the checkpoint's network in the GPU inference server, its tactician (sampled, on-policy, with the
checkpoint's W / planner features and v_next table), its curriculum over the starting round, and the training search
(Gumbel, --sims 256, 8 determinizations, top-16, opponent_prior factory_sets: the training-only sampler; this module
calls mark_training(), so no perfect-information search can exist in it). Nothing is trained: the network is fixed.
For every battler decision it stores the encoded input, the legal mask, the policy target (r["target"], what
rl.alphazero.train_iteration trains on), the network's prior and value, visits, raw and completed Q, the root value
(visit-weighted Q, the value target's q), the played action, the battle outcome z (the network value at a
truncation, as in training), round / battle / Noland / streak, and the replay key.

Replay key (runs/v5_offline/CONTRACT.md): (run, li). `run` is an int64 id of one Factory run, whose record in
runs.jsonl.gz holds the SimBackend.reset arguments (seed, win_streak, rents_count) and the list of EVERY action given
to SimBackend.act since that reset (tactician and battler actions, and the automatic move-0 of decisions without a
choice), in rl.autopsy.enc_action's JSON form. `li` = how many of those actions precede the decision. The simulator is
deterministic, so SimBackend(max_turns=10**9).reset(**reset) + act(log[0..li-1]) rebuilds the exact pre-decision
state, observer memory included (replay_positions).
"""

import argparse
import glob
import gzip
import hashlib
import json
import math
import os
import time
from collections import defaultdict, namedtuple

import numpy as np

from pybattle.backend import Phase, SimBackend

DATA_DIR = "runs/v5_offline/main"
ENCODE_VERSION = 5
OBS_KEYS = ("mon_ids", "mon_num", "move_num", "ctx_ids", "ctx_num", "mask", "active")
OBS_STORE = {"mon_ids": np.int32, "ctx_ids": np.int32, "active": np.int8, "mask": np.bool_,
             "mon_num": np.float32, "move_num": np.float32, "ctx_num": np.float32}
OBS_LOAD = {"mon_ids": np.int64, "ctx_ids": np.int64, "active": np.int64}      # what rl.encode.battle returns
META_FIELDS = {"run": np.int64, "li": np.int32, "ndec": np.int16, "round": np.int8, "battle": np.int8,
               "noland": np.bool_, "streak": np.int16, "target": np.float32, "prior": np.float32,
               "visits": np.float32, "q": np.float32, "cq": np.float32, "value_net": np.float32,
               "root_q": np.float32, "action": np.int8, "searched": np.bool_, "z": np.float32, "trunc": np.bool_}
HOLDOUT_PCT = 5
_BATTLE = (Phase.BATTLE, Phase.FORCED_SWITCH)


def stable_u32(*parts):
    return int.from_bytes(hashlib.blake2b(repr(parts).encode(), digest_size=4).digest(), "little")


def is_heldout(run, pct=HOLDOUT_PCT):
    """Hold-out by run (5 %): every position of a held-out run is held out."""
    return stable_u32("v5-offline-holdout", int(run)) % 100 < pct


# ---- recording ------------------------------------------------------------------------------------------------------

class RecBackend(SimBackend):
    """SimBackend that keeps, per run, the reset arguments and every action given to act() (the replay recipe).
    clone() returns a plain SimBackend (SimBackend.clone builds one explicitly), so the searches' clones never
    record."""

    def setup_recording(self, prefix):
        self._rec_prefix, self._rec_n = int(prefix), 0
        self.runs = {}
        self.run_uid, self.log = None, []

    def reset(self, seed=0, win_streak=0, rents_count=0):
        super().reset(seed=seed, win_streak=win_streak, rents_count=rents_count)
        if not hasattr(self, "_rec_prefix"):              # SimBackend.__init__ paths before setup_recording
            return
        self.run_uid = self._rec_prefix * 1_000_000 + self._rec_n
        self._rec_n += 1
        self.log = []
        self.runs[self.run_uid] = {"run": self.run_uid,
                                   "reset": {"seed": int(seed) & 0xFFFFFFFF, "win_streak": int(win_streak),
                                             "rents_count": int(rents_count)}, "log": self.log}

    def act(self, action):
        from .autopsy import enc_action
        if hasattr(self, "_rec_prefix"):
            self.log.append(enc_action(action))
        return super().act(action)


def _player_class():
    from . import alphazero_v4 as V4
    from .gumbel import completed_q, normalized_prior

    class RecordingPlayer(V4.SelfPlayerV4):
        """alphazero_v4's self-play worker that writes every battler decision (with its replay key) to shards."""

        def __init__(self, cfg, seed, evaluator=None):
            super().__init__(cfg, seed, evaluator)
            self.rec = RecBackend(max_turns=10 ** 9)
            self.rec.setup_recording(seed)
            self.env.backend = self.rec                  # play() resets the env first: the first run is recorded
            self._search = self.battler.search_root
            self.battler.search_root = self._search_rec
            self.pending_meta, self.out_meta = [], []
            self._trunc = False
            self.chunk = 0
            self.out_dir = cfg["rec_out"]

        def _search_rec(self, backend, decisions=0, root_x=None, noise=True):
            r = self._search(backend, decisions, root_x, noise)
            info = backend.run_info()
            legal = r["legal"]
            if r["searched"]:
                cq = completed_q(r["visits"], r["q"], normalized_prior(r["prior"], legal), legal, r["value"])
            else:
                cq = np.zeros(7)
            self.pending_meta.append({
                "run": self.rec.run_uid, "li": len(self.rec.log), "ndec": int(decisions),
                "round": int(info.challenge_num) + 1, "battle": int(info.battle_in_challenge) + 1,
                "noland": bool(info.noland), "streak": int(info.win_streak), "target": r["target"],
                "prior": r["prior"], "visits": r["visits"], "q": r["q"], "cq": cq, "value_net": r["value"],
                "root_q": r["root_q"], "action": int(r["action"]), "searched": bool(r["searched"])})
            return r

        def _new_battle(self):
            super()._new_battle()
            self.pending_meta = []

        def _after(self, ev):
            self._trunc = bool(ev["truncate"])
            super()._after(ev)

        def _finish_battle(self, z):
            if len(self.pending_meta) != len(self.pending_b):
                raise RuntimeError("recording out of step with the self-play samples")
            for m in self.pending_meta:
                self.out_meta.append(dict(m, z=float(z), trunc=self._trunc))
            super()._finish_battle(z)

        def play(self, n_decisions):
            out, st = super().play(n_decisions)
            b = out.get("battle")
            n = 0 if b is None else len(b["pi"])
            if n != len(self.out_meta):
                raise RuntimeError(f"{n} samples, {len(self.out_meta)} records")
            if n:
                if not np.array_equal(b["pi"], np.stack([m["target"] for m in self.out_meta]).astype(np.float32)):
                    raise RuntimeError("recorded targets differ from the self-play samples")
                write_shard(self.out_dir, f"w{self.rec._rec_prefix}_c{self.chunk:05d}", b["obs"], self.out_meta,
                            self.rec.runs)
            # keep only the run in progress (its log keeps growing; it is written again with later shards)
            self.rec.runs = {self.rec.run_uid: self.rec.runs[self.rec.run_uid]}
            self.out_meta = []
            self.chunk += 1
            small = {k: v for k, v in st.items() if isinstance(v, (int, float))}
            small["positions"] = n
            return {}, small

    return RecordingPlayer


def write_shard(out_dir, name, obs, meta, runs):
    """One shard: <name>.npz (obs + meta arrays) and <name>.runs.jsonl.gz (the logs of the runs it references, as
    long as they are now; a later shard may hold a longer log of the same run: consolidate keeps the longest)."""
    os.makedirs(out_dir, exist_ok=True)
    arrays = {f"obs_{k}": np.asarray(obs[k]).astype(OBS_STORE[k]) for k in OBS_KEYS}
    for f, dt in META_FIELDS.items():
        arrays[f] = np.asarray([m[f] for m in meta]).astype(dt)
    used = sorted({m["run"] for m in meta})
    tmp = os.path.join(out_dir, f".{name}.tmp")
    with gzip.open(tmp + ".runs", "wt") as fh:
        for u in used:
            r = runs[u]
            fh.write(json.dumps({"run": r["run"], "reset": r["reset"], "log": list(r["log"])},
                                separators=(",", ":")) + "\n")
    np.savez(tmp + ".npz", **arrays)
    os.replace(tmp + ".runs", os.path.join(out_dir, f"{name}.runs.jsonl.gz"))
    os.replace(tmp + ".npz", os.path.join(out_dir, f"{name}.npz"))       # the .npz last: it marks the shard done


def record(a):
    import torch
    from . import alphazero as AZ
    from . import alphazero_v3 as V3
    from . import alphazero_v4 as V4
    from . import encode
    from .inference import InferenceServer
    from .search import mark_training
    mark_training()                     # training data: no perfect-information search can exist in this process
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    args = argparse.Namespace(**ck["args"])
    if args.encode_version != ENCODE_VERSION or ck["args"].get("variant") != "alphazero_v4":
        raise ValueError(f"{a.ckpt}: expected an alphazero_v4 checkpoint (encoding 5)")
    args.sims = a.sims
    args.opponent_prior = a.prior
    encode.set_version(args.encode_version)
    shard_dir = os.path.join(a.out, "shards")
    os.makedirs(shard_dir, exist_ok=True)
    cfg = V3.worker_cfg(args, ck["args"]["opt_feat"])
    cfg.update(opponent_model=a.ckpt, rec_out=shard_dir)
    w_path = ck["team_eval"]
    weights = AZ.start_weights_from_eval({int(k): v for k, v in ck["round_eval"].items()}, args.start_p0)
    info = {"ckpt": a.ckpt, "iteration": ck["iteration"], "sims": a.sims, "prior": a.prior, "dets": args.dets,
            "max_considered": args.max_considered, "team_eval": w_path, "seed": a.seed, "workers": a.workers,
            "start_weights": weights.tolist(), "v_next": ck["v_next"]["table"], "encode_version": ENCODE_VERSION,
            "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(info, open(os.path.join(a.out, f"record_seed{a.seed}.json"), "w"), indent=1)
    del ck
    server = InferenceServer(a.ckpt, n_clients=a.workers)
    pool = AZ.WorkerPool(a.workers, cfg, server, a.seed, player=_player_class())
    log = open(os.path.join(a.out, f"record_seed{a.seed}.log"), "a")
    t0, total, it = time.time(), 0, 0
    try:
        pool.broadcast("load", a.ckpt)              # the tactician's CPU copy and the value norms
        pool.broadcast("weights", weights.tolist())
        pool.broadcast("call", ("set_v_next", (info["v_next"],)))
        if w_path != args.team_eval:
            pool.broadcast("call", ("set_team_eval", (w_path,)))
        while True:
            t1 = time.time()
            res = pool.broadcast("play", a.chunk)
            n = sum(s["positions"] for _, s in res)
            dec = sum(s["battler_decisions"] for _, s in res)
            total += n
            it += 1
            el = time.time() - t0
            msg = (f"[{el / 60:7.1f} min] chunk {it}: {dec} decisions in {time.time() - t1:.0f} s "
                   f"({dec / (time.time() - t1):.0f}/s), {total:,} positions written")
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()
            if total >= a.positions or el >= a.hours * 3600 or os.path.exists(os.path.join(a.out, "STOP")):
                break
    finally:
        pool.close()
        server.close()
    return total


# ---- consolidation ----------------------------------------------------------------------------------------------------

def consolidate(a):
    """shards/*.npz -> <field>.npy (memory-mappable), runs.jsonl.gz (longest log per run), meta.json."""
    shard_dir = os.path.join(a.out, "shards")
    names = sorted(p[:-4] for p in glob.glob(os.path.join(shard_dir, "*.npz")))
    if not names:
        raise FileNotFoundError(f"no shards in {shard_dir}")
    sizes = []
    for nm in names:
        with np.load(nm + ".npz") as z:
            sizes.append(len(z["run"]))
    n = int(sum(sizes))
    with np.load(names[0] + ".npz") as z:
        specs = {k: (z[k].shape[1:], z[k].dtype) for k in z.files}
    outs = {k: np.lib.format.open_memmap(os.path.join(a.out, f"{k}.npy.tmp"), mode="w+", dtype=dt,
                                         shape=(n,) + sh) for k, (sh, dt) in specs.items()}
    runs = {}
    i = 0
    for nm, s in zip(names, sizes):
        with np.load(nm + ".npz") as z:
            for k in outs:
                outs[k][i:i + s] = z[k]
        with gzip.open(nm + ".runs.jsonl.gz", "rt") as fh:
            for line in fh:
                r = json.loads(line)
                if r["run"] not in runs or len(r["log"]) > len(runs[r["run"]]["log"]):
                    runs[r["run"]] = r
        i += s
    for k, m in outs.items():
        m.flush()
        del m
        os.replace(os.path.join(a.out, f"{k}.npy.tmp"), os.path.join(a.out, f"{k}.npy"))
    run = np.load(os.path.join(a.out, "run.npy"), mmap_mode="r")
    li = np.load(os.path.join(a.out, "li.npy"), mmap_mode="r")
    held = np.array([is_heldout(u) for u in run], bool)
    np.save(os.path.join(a.out, "held.npy"), held)
    for u, l in zip(run[:: max(1, n // 100000)], li[:: max(1, n // 100000)]):
        if len(runs[int(u)]["log"]) < int(l):
            raise RuntimeError(f"run {u}: log shorter than a recorded position")
    with gzip.open(os.path.join(a.out, "runs.jsonl.gz"), "wt") as fh:
        for u in sorted(runs):
            fh.write(json.dumps(runs[u], separators=(",", ":")) + "\n")
    rnd = np.load(os.path.join(a.out, "round.npy"))
    nol = np.load(os.path.join(a.out, "noland.npy"))
    meta = {"positions": n, "runs": len(runs), "heldout_positions": int(held.sum()), "shards": len(names),
            "by_round": {int(r): int((rnd == r).sum()) for r in np.unique(rnd)}, "noland": int(nol.sum()),
            "fields": {k: [list(sh), str(dt)] for k, (sh, dt) in specs.items()},
            "record": [json.load(open(p)) for p in sorted(glob.glob(os.path.join(a.out, "record_seed*.json")))],
            "consolidated": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k not in ("fields", "record")}, indent=1))


# ---- loading -----------------------------------------------------------------------------------------------------------

class Dataset:
    """The consolidated dataset, memory-mapped: d.obs[k] (stored dtypes), d[field], d.held, len(d)."""

    def __init__(self, data_dir=DATA_DIR, mmap=True):
        self.dir = data_dir
        self.meta = json.load(open(os.path.join(data_dir, "meta.json")))
        mode = "r" if mmap else None
        self.obs = {k: np.load(os.path.join(data_dir, f"obs_{k}.npy"), mmap_mode=mode) for k in OBS_KEYS}
        self._f = {}
        self._mode = mode
        self.held = np.load(os.path.join(data_dir, "held.npy"))
        self._runs = None

    def __len__(self):
        return len(self.held)

    def __getitem__(self, field):
        if field not in self._f:
            self._f[field] = np.load(os.path.join(self.dir, f"{field}.npy"), mmap_mode=self._mode)
        return self._f[field]

    def obs_rows(self, idx):
        """Encoded inputs of rows idx with rl.encode.battle's dtypes."""
        return {k: np.asarray(self.obs[k][idx]).astype(OBS_LOAD.get(k, self.obs[k].dtype)) for k in OBS_KEYS}

    @property
    def runs(self):
        if self._runs is None:
            self._runs = load_runs(self.dir)
        return self._runs

    def keys(self, idx):
        return [(int(self["run"][i]), int(self["li"][i])) for i in np.atleast_1d(idx)]

    def rows_of(self, keys):
        """(run, li) keys -> row indices (-1 if absent)."""
        lut = {(int(r), int(l)): i for i, (r, l) in enumerate(zip(self["run"], self["li"]))}
        return np.array([lut.get((int(r), int(l)), -1) for r, l in keys], np.int64)


def load_runs(data_dir=DATA_DIR):
    with gzip.open(os.path.join(data_dir, "runs.jsonl.gz"), "rt") as fh:
        return {r["run"]: r for r in map(json.loads, fh)}


# ---- replay (the contract: runs/v5_offline/CONTRACT.md) ---------------------------------------------------------------

Replayed = namedtuple("Replayed", "key backend view observer ctx obs")


def replay_positions(keys, data_dir=DATA_DIR, runs=None, clone=True, check=False):
    """Rebuild recorded battler decisions. keys: iterable of (run, li) tuples, or of int row indices of the dataset
    in data_dir. Yields Replayed(key, backend, view, observer, ctx, obs), GROUPED BY RUN and in increasing li (not
    in the input order; each run is replayed once):
        backend   a SimBackend at the decision (phase BATTLE or FORCED_SWITCH); a private clone when clone=True
                  (default), so the caller may act on it / search from it;
        view      backend.view(): the player's BattleView (own party, the opponent as seen, observer memory);
        observer  backend._observer: the player's BattleObserver (what was revealed so far in this battle);
        ctx       rl.search.run_ctx(backend): the streak context the network sees;
        obs       rl.encode.battle(view, ctx) with encoding version 5: the network input, equal to the recorded one.
    check=True (needs row-index keys or the dataset in data_dir): raise if obs differs from the recorded input.
    Only the backend's public state is used; nothing hidden goes into obs."""
    from . import encode
    from .autopsy import dec_action
    from .search import run_ctx
    encode.set_version(ENCODE_VERSION)
    keys = list(keys)
    ds = None
    if keys and not isinstance(keys[0], (tuple, list)):
        ds = Dataset(data_dir)
        rows = np.asarray(keys, np.int64)
        pairs = ds.keys(rows)
    else:
        rows = None
        pairs = [(int(r), int(l)) for r, l in keys]
        if check:
            ds = Dataset(data_dir)
            rows = ds.rows_of(pairs)
    if runs is None:
        runs = ds.runs if ds is not None else load_runs(data_dir)
    by_run = defaultdict(list)
    for j, (r, l) in enumerate(pairs):
        by_run[r].append((l, j))
    for r in sorted(by_run):
        rec = runs[r]
        be = SimBackend(max_turns=10 ** 9)
        be.reset(**rec["reset"])
        li = 0
        for l, j in sorted(by_run[r]):
            while li < l:
                be.act(dec_action(rec["log"][li]))
                li += 1
            if be.phase not in _BATTLE:
                raise RuntimeError(f"replay of {(r, l)}: not at a battle decision ({be.phase})")
            b = be.clone() if clone else be
            view = b.view()
            ctx = run_ctx(b)
            obs = encode.battle(view, ctx)
            key = int(rows[j]) if rows is not None and not isinstance(keys[0], (tuple, list)) else (r, l)
            if check:
                ref = ds.obs_rows([rows[j]])
                for k in OBS_KEYS:
                    if not np.array_equal(np.asarray(obs[k]).astype(ref[k].dtype), ref[k][0]):
                        raise RuntimeError(f"replay of {(r, l)}: encoded input {k} differs from the recording")
            yield Replayed(key, b, view, b._observer, ctx, obs)


def check(a):
    ds = Dataset(a.data)
    rng = np.random.default_rng(a.seed)
    rows = np.sort(rng.choice(len(ds), min(a.n, len(ds)), replace=False))
    t0 = time.time()
    n = sum(1 for _ in replay_positions(rows, a.data, check=True))
    print(f"{n} positions replayed, every encoded input identical ({time.time() - t0:.1f} s)")


# ---- subsets and re-search (experiment 4: other targets on the same positions) ---------------------------------------

def subset(a):
    """Whole runs drawn at random until --positions rows: rows.npy (sorted indices into the dataset)."""
    ds = Dataset(a.data)
    run = np.asarray(ds["run"])
    uniq, inv, counts = np.unique(run, return_inverse=True, return_counts=True)
    order = np.random.default_rng(a.seed).permutation(len(uniq))
    take = order[np.searchsorted(np.cumsum(counts[order]), a.positions) + 1:]
    keep = np.ones(len(uniq), bool)
    keep[take] = False
    rows = np.flatnonzero(keep[inv])
    os.makedirs(a.out, exist_ok=True)
    np.save(os.path.join(a.out, "rows.npy"), rows)
    json.dump({"data": a.data, "positions": int(len(rows)), "runs": int(keep.sum()),
               "heldout": int(ds.held[rows].sum()), "seed": a.seed}, open(os.path.join(a.out, "meta.json"), "w"),
              indent=1)
    print(f"{len(rows):,} positions from {int(keep.sum()):,} runs ({int(ds.held[rows].sum()):,} held out)")


class Researcher:
    """A worker (rl.alphazero.WorkerPool player) that replays recorded positions and searches them again."""

    def __init__(self, cfg, seed, evaluator=None):
        import torch
        from . import encode
        from .gumbel import GumbelBattler
        torch.set_num_threads(1)
        encode.set_version(ENCODE_VERSION)
        self.cfg = cfg
        self.gb = GumbelBattler(evaluator, cfg["sims"], cfg["dets"], cfg["search_batch"], seed=seed,
                                max_considered=cfg["max_considered"], c_visit=cfg["c_visit"], c_scale=cfg["c_scale"],
                                rescale=bool(cfg["gumbel_rescale"]), non_root=cfg["non_root"], c_puct=cfg["c_puct"],
                                opponent_prior=cfg["prior"],
                                opponent_model=cfg["ckpt"] if cfg["prior"] == "experience" else None)
        self.ds = Dataset(cfg["data"])

    def load(self, path):
        pass

    def run(self, rows, out_path):
        from .gumbel import completed_q, normalized_prior
        rows = np.asarray(rows, np.int64)
        ndec = np.asarray(self.ds["ndec"])
        res = {k: [] for k in ("row", "target", "visits", "q", "cq", "prior", "value_net", "root_q", "action")}
        t0 = time.time()
        for rp in replay_positions(rows, self.cfg["data"], runs=self.ds.runs, check=True):
            i = rp.key
            r = self.gb.search_root(rp.backend, int(ndec[i]), rp.obs, noise=True)
            legal = r["legal"]
            cq = completed_q(r["visits"], r["q"], normalized_prior(r["prior"], legal), legal, r["value"]) \
                if r["searched"] else np.zeros(7)
            for k, v in (("row", i), ("target", r["target"]), ("visits", r["visits"]), ("q", r["q"]), ("cq", cq),
                         ("prior", r["prior"]), ("value_net", r["value"]), ("root_q", r["root_q"]),
                         ("action", r["action"])):
                res[k].append(v)
        arr = {k: np.asarray(v) for k, v in res.items()}
        order = np.argsort(arr["row"])
        np.savez(out_path, **{k: v[order] for k, v in arr.items()})
        return {"positions": len(rows), "seconds": time.time() - t0}


def research(a):
    import torch
    from . import alphazero as AZ
    from .inference import InferenceServer
    from .search import mark_training
    mark_training()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    ca = ck["args"]
    del ck
    rows = np.load(os.path.join(a.subset, "rows.npy"))
    out_parts = os.path.join(a.out, "parts")
    os.makedirs(out_parts, exist_ok=True)
    ds = Dataset(a.data)
    run = np.asarray(ds["run"])[rows]
    # chunks of whole runs (each run replayed once), dealt to the workers
    uniq = np.unique(run)
    groups = np.array_split(np.random.default_rng(0).permutation(uniq), max(1, math.ceil(len(rows) / a.chunk)))
    jobs = []
    for gi, g in enumerate(groups):
        p = os.path.join(out_parts, f"g{gi:05d}.npz")
        if not os.path.exists(p):
            jobs.append((rows[np.isin(run, g)], p))
    cfg = {k: ca[k] for k in ("dets", "search_batch", "max_considered", "c_visit", "c_scale", "gumbel_rescale",
                              "non_root", "c_puct")}
    cfg.update(sims=a.sims, prior=a.prior, ckpt=a.ckpt, data=a.data)
    json.dump(dict(cfg, subset=a.subset, started=time.strftime("%Y-%m-%d %H:%M:%S")),
              open(os.path.join(a.out, "research.json"), "w"), indent=1)
    print(f"{len(rows):,} positions, {len(jobs)} chunks to do", flush=True)
    server = InferenceServer(a.ckpt, n_clients=a.workers)
    pool = AZ.WorkerPool(a.workers, cfg, server, a.seed, player=Researcher)
    t0, done = time.time(), 0
    try:
        for k in range(0, len(jobs), a.workers):
            batch = jobs[k:k + a.workers]
            cmds = [("run", (r, p)) for r, p in batch] + [("load", (None,))] * (a.workers - len(batch))
            out = pool.call("call", cmds)
            done += sum(o["positions"] for o in out if o)
            print(f"[{(time.time() - t0) / 60:6.1f} min] {done:,} positions ({done / (time.time() - t0):.0f}/s)",
                  flush=True)
            if os.path.exists(os.path.join(a.out, "STOP")):
                break
    finally:
        pool.close()
        server.close()
    merge_research(a.out)


def merge_research(out):
    parts = sorted(glob.glob(os.path.join(out, "parts", "*.npz")))
    arr = defaultdict(list)
    for p in parts:
        with np.load(p) as z:
            for k in z.files:
                arr[k].append(z[k])
    arr = {k: np.concatenate(v) for k, v in arr.items()}
    order = np.argsort(arr["row"])
    np.savez(os.path.join(out, "targets.npz"), **{k: v[order] for k, v in arr.items()})
    print(f"{out}/targets.npz: {len(order):,} positions")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--ckpt", default="runs/alphazero_v4/stopped_it1111.pt")
    r.add_argument("--out", default=DATA_DIR)
    r.add_argument("--workers", type=int, default=12)
    r.add_argument("--hours", type=float, default=3.0)
    r.add_argument("--positions", type=int, default=4_000_000)
    r.add_argument("--chunk", type=int, default=2000, help="battler decisions per worker between shard writes")
    r.add_argument("--sims", type=int, default=256)
    r.add_argument("--prior", default="factory_sets", choices=("factory_sets", "experience"))
    r.add_argument("--seed", type=int, default=1)
    c = sub.add_parser("consolidate")
    c.add_argument("--out", default=DATA_DIR)
    k = sub.add_parser("check")
    k.add_argument("--data", default=DATA_DIR)
    k.add_argument("--n", type=int, default=200)
    k.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("subset")
    s.add_argument("--data", default=DATA_DIR)
    s.add_argument("--positions", type=int, default=400_000)
    s.add_argument("--out", required=True)
    s.add_argument("--seed", type=int, default=0)
    e = sub.add_parser("research")
    e.add_argument("--data", default=DATA_DIR)
    e.add_argument("--subset", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--ckpt", default="runs/alphazero_v4/stopped_it1111.pt")
    e.add_argument("--sims", type=int, default=1024)
    e.add_argument("--prior", default="factory_sets", choices=("factory_sets", "experience"))
    e.add_argument("--workers", type=int, default=12)
    e.add_argument("--chunk", type=int, default=2000, help="positions per job (whole runs)")
    e.add_argument("--seed", type=int, default=3)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    {"record": record, "consolidate": consolidate, "check": check, "subset": subset, "research": research}[a.cmd](a)


if __name__ == "__main__":
    main()

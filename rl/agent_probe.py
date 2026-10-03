"""Per-agent probes (docs/RL_DECISIONS.md §25): how much has each agent (battler, tactician) learned, measured apart,
so that a plateau of the whole system is not misread.

    python -m rl.agent_probe generate [--runs 500]                       # once: runs/probe/battles_v1/
    python -m rl.agent_probe battler --battler net:CKPT [--subset N --seeds K]      # B
    python -m rl.agent_probe tactician --tactician net:CKPT | planner:CKPT          # C(1)
    python -m rl.agent_probe outcomes                                    # C(2): the option outcome table
    python -m rl.agent_probe crossplay --tactician net:X --battler net:Y [--runs 256]   # A
    python -m rl.agent_probe curve --ckpts A,B,... [--procs 2]          # the learning-curve driver + report
    python -m rl.agent_probe report                                     # rebuild the report from the caches

Diagnosis only (never training). Everything is cached under runs/probe/ and every stage is resumable: results are
appended line by line (JSONL, one file per agent and process) and finished keys are skipped on a relaunch.

The benchmark (generate, once): runs played by alphazero_v3 final (tactician = its network, greedy, with its W /
planner features; battler = its network, greedy) from the start of rounds 5 / 6 / 3 (40 / 40 / 20 %), until the
round is completed or lost. Each run is kept as its replay recipe (rl/autopsy.py: environment seed + every backend
action); every tactician decision of a run is a state (the run replayed up to it, checked with a state hash and the
hash of the tactician's observation). From those states:
    B   battle starts: the tactician decision + v3's option (our team) + the opponent team the game generated with it
        (the true one, checked by a hash), the battle RNG reseeded before its first frame with K fixed seeds.
    C1  ~300 rental + ~600 swap decisions; C2 a subset (~50 + ~100) for the real-outcome regret.
Agents (specs): battler  net:CKPT (greedy) | search:CKPT[:SIMS] (legal search, the experience opponent model: the
checkpoint's own, else runs/opp_model_v3/model.pt) | perfect:CKPT[:SIMS] (perfect information: a ceiling);
tactician  net:CKPT (greedy, with the checkpoint's own W / planner features) | planner:CKPT (argmax of the
checkpoint's own Q_plan).
"""

import argparse
import glob
import gzip
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np

from pybattle.backend import Phase

PROBE_DIR = os.environ.get("PYBATTLE_PROBE_DIR", "runs/probe")
BENCH_NAME = "battles_v1"
GEN_CKPT = "runs/alphazero_v3/latest.pt"
OPP_MODEL = "runs/opp_model_v3/model.pt"
MAX_DECISIONS = 300
WIN_SCALE = 10.0
Q_COL = 4                                   # q_plan in option_features_v3 (in wins / WIN_SCALE)
_BATTLE = (Phase.BATTLE, Phase.FORCED_SWITCH)


# ---- small utilities ----------------------------------------------------------------------------------------------

def stable_seed(*parts):
    """A 31-bit seed from any parts, the same in every process."""
    return int.from_bytes(hashlib.blake2b(repr(parts).encode(), digest_size=4).digest(), "little") & 0x7FFFFFFF


def feat_seed(state_id):
    """The planner's seed at a benchmark state (the same for every checkpoint: common random numbers)."""
    return stable_seed("probe-feat", state_id)


def obs_hash(obs):
    h = hashlib.sha1()
    for k in sorted(obs):
        if k == "opt_feat":
            continue
        h.update(k.encode())
        h.update(np.ascontiguousarray(np.asarray(obs[k])).tobytes())
    return h.hexdigest()[:16]


def write_jsonl(path, rows, mode="a"):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    op = gzip.open if path.endswith(".gz") else open
    with op(path, mode + "t") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")


def read_jsonl(path):
    op = gzip.open if path.endswith(".gz") else open
    out = []
    try:
        with op(path, "rt") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:          # a line cut by an interrupted process
                    break
    except (EOFError, OSError):
        pass
    return out


def read_shards(prefix):
    """All rows of <prefix>*.jsonl (one file per process)."""
    rows = []
    for p in sorted(glob.glob(prefix + "*.jsonl")):
        rows += read_jsonl(p)
    return rows


_TAGS = {}


def ckpt_info(ckpt):
    """-> {"tag": "<run>_it<iteration>", "iteration", "run", "opponent_model"} (cached)."""
    if ckpt not in _TAGS:
        import torch
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        run = os.path.basename(os.path.dirname(os.path.abspath(ckpt)))
        it = int(ck.get("iteration", -1))
        _TAGS[ckpt] = {"tag": f"{run}_it{it:04d}", "iteration": it, "run": run,
                       "opponent_model": bool(ck.get("opponent_model")),
                       "encode_version": int(ck["args"].get("encode_version", 2))}
    return _TAGS[ckpt]


def parse_spec(spec):
    """'kind:path[:sims]' -> (kind, path, sims)."""
    kind, rest = spec.split(":", 1)
    sims = None
    if kind in ("search", "perfect") and ":" in rest and rest.rsplit(":", 1)[1].isdigit():
        rest, s = rest.rsplit(":", 1)
        sims = int(s)
    if kind not in ("net", "planner", "search", "perfect"):
        raise ValueError(f"unknown agent kind {kind!r} in {spec!r}")
    if sims is None and kind in ("search", "perfect"):
        sims = 512 if kind == "search" else 2048
    return kind, rest, sims


def spec_tag(spec):
    kind, path, sims = parse_spec(spec)
    t = ckpt_info(path)["tag"]
    return f"{kind}{sims if sims else ''}-{t}"


def mean_ci(x, z=1.96):
    """(mean, half-width of the normal 95 % interval) over independent units."""
    x = np.asarray([v for v in x if v is not None and not (isinstance(v, float) and math.isnan(v))], np.float64)
    if len(x) == 0:
        return float("nan"), float("nan"), 0
    if len(x) == 1:
        return float(x[0]), float("nan"), 1
    return float(x.mean()), float(z * x.std(ddof=1) / math.sqrt(len(x))), len(x)


# ---- agents (one cache per process) ------------------------------------------------------------------------------

class Agents:
    """Networks, battler evaluators and W / planner hooks of the checkpoints, loaded once per process."""

    def __init__(self, device="cpu", threads=1):
        import torch
        torch.set_num_threads(threads)
        self.device = device
        self._pol, self._ev, self._hook, self._search = {}, {}, {}, {}

    def policy(self, ckpt):
        if ckpt not in self._pol:
            import torch
            from .policy import Policy
            self._pol[ckpt] = Policy(ckpt, torch.device(self.device))
        return self._pol[ckpt]

    def evaluator(self, ckpt):
        if ckpt not in self._ev:
            from .inference import make_evaluator
            self._ev[ckpt] = make_evaluator(self.policy(ckpt))
        return self._ev[ckpt]

    def hook(self, ckpt):
        if ckpt not in self._hook:
            from .alphazero_v3 import features_hook
            h = features_hook(ckpt)
            if h is None:
                raise ValueError(f"{ckpt}: not an alphazero_v3 / v4 checkpoint with tactician features")
            self._hook[ckpt] = h
        return self._hook[ckpt]

    def search(self, ckpt, mode, sims, dets=8):
        key = (ckpt, mode, sims, dets)
        if key not in self._search:
            from .search import SearchBattler
            kw = {}
            if mode == "legal":
                model = ckpt if ckpt_info(ckpt)["opponent_model"] else OPP_MODEL
                kw = {"opponent_prior": "experience", "opponent_model": model}
            self._search[key] = SearchBattler(self.policy(ckpt), n_sims=sims, n_determinizations=dets, mode=mode,
                                              allow_perfect=mode == "perfect", seed=0,
                                              evaluator=self.evaluator(ckpt), **kw)
        return self._search[key]


def option_features(agents, ckpt, be, kind, obs, ctx, seed):
    """(options, feats [n_opt, 7]) of a tactician decision with the checkpoint's W / planner (rl.alphazero_v3
    features_hook, exactly as the evaluation computes them), the planner's opponent draws seeded."""
    from .tactician_search import options_of
    hook = agents.hook(ckpt)
    options = options_of(kind, obs)
    view = be.view()
    hint = (int(view.hint_type), int(view.hint_style))
    ev = hook._load()
    cfg = dict(hook.planner_cfg, seed=int(seed))
    feats = np.asarray(hook._mod.option_features_v3(ev, view, kind, ctx, options, hook.table(), hint, cfg),
                       np.float32)
    return options, feats


def tactician_obs(be):
    from . import encode
    from .search import run_ctx
    kind = "rental" if be.phase == Phase.RENTAL else "swap"
    ctx = run_ctx(be)
    view = be.view()
    obs = encode.rental(view, ctx) if kind == "rental" else encode.swap(view, ctx)
    return kind, obs, ctx


def tactician_decide(agents, spec, be, seed, feats_cache=None):
    """A tactician decision at a RENTAL / SWAP backend. -> (option index, options, feats of the spec's checkpoint).
    net: the network, greedy (rental: lead, then pair), with its own features; planner: argmax of its own Q_plan."""
    from . import encode
    from .tactician_features import to_action_space
    kind_t, path, _ = parse_spec(spec)
    kind, obs, ctx = tactician_obs(be)
    if feats_cache is not None and path in feats_cache:
        options, feats = feats_cache[path]
    else:
        options, feats = option_features(agents, path, be, kind, obs, ctx, seed)
        if feats_cache is not None:
            feats_cache[path] = (options, feats)
    if kind_t == "planner":
        return int(np.argmax(feats[:, Q_COL])), options, feats
    if kind_t != "net":
        raise ValueError(f"tactician spec must be net: or planner:, not {spec!r}")
    pol = agents.policy(path)
    o = dict(obs)
    o["opt_feat"] = to_action_space(kind, options, feats)
    a = pol.act(kind, encode.collate([o], pol.device), greedy=True)[0]
    opt = (int(a[0]), int(a[1])) if kind == "rental" else int(a)
    return options.index(opt), options, feats


def option_to_action(kind, option):
    from .envs import decode_action
    return decode_action(kind, option)


# ---- battlers (batched over games) --------------------------------------------------------------------------------

class Battler:
    """act_many(games) -> backend actions. net: one batched forward pass of the checkpoint's network (greedy argmax
    over the legal actions); search / perfect: rl.search.SearchBattler, game by game, its RNG seeded per (game,
    decision) so that a result is reproducible."""

    def __init__(self, agents, spec, dets=8):
        self.kind, self.ckpt, self.sims = parse_spec(spec)
        if self.kind == "planner":
            raise ValueError("planner: is a tactician spec")
        self.agents = agents
        self.dets = dets
        if self.kind == "net":
            self.ev = agents.evaluator(self.ckpt)
        else:
            self.sb = agents.search(self.ckpt, "legal" if self.kind == "search" else "perfect", self.sims, dets)

    def act_many(self, games):
        from . import encode
        from .inference import stack_obs
        from .search import run_ctx, to_backend_action
        if not games:
            return []
        if self.kind == "net":
            obs = [encode.battle(g.be.view(), run_ctx(g.be)) for g in games]
            pri, _ = self.ev(stack_obs(obs))
            out = []
            for o, p in zip(obs, pri):
                out.append(to_backend_action(int(np.argmax(np.where(o["mask"], p, -1.0)))))
            return out
        out = []
        for g in games:
            self.sb.rng = random.Random(stable_seed(g.key, g.ndec, "search"))
            out.append(self.sb.act_on(g.be, decisions=g.ndec))
        return out


class Game:
    """One game in flight: a backend at a battle start (or at a tactician decision), played to the end of the
    battle (until="battle") or of the round (until="round": the tactician continues between battles)."""
    __slots__ = ("key", "be", "until", "ndec", "battles", "wins", "first_won", "done", "tdec", "meta", "decisions",
                 "truncated")

    def __init__(self, key, be, until="battle", meta=None):
        self.key, self.be, self.until, self.meta = key, be, until, meta
        self.ndec = 0
        self.battles = self.wins = self.tdec = self.decisions = 0
        self.first_won = None
        self.done = False
        self.truncated = False


def _battle_over(g, won):
    g.battles += 1
    g.wins += int(won)
    if g.first_won is None:
        g.first_won = bool(won)
    g.ndec = 0


def advance(g, tactician=None):
    """Run g to its next battle decision that has a choice, or to its end. tactician(g) -> backend action, called
    at RENTAL / SWAP decisions of an until="round" game (the rental only before its first battle)."""
    be = g.be
    while True:
        ph = be.phase
        if ph in _BATTLE:
            if g.ndec >= MAX_DECISIONS:                     # FactoryEnv's truncation: counted as a loss
                g.truncated = True
                _battle_over(g, False)
                g.done = True
                return
            if ph == Phase.BATTLE:
                v = be.view()
                if not any(v.usable_moves) and not v.switch_targets:
                    be.act(("move", 0))
                    g.ndec += 1
                    if be.phase not in _BATTLE:
                        won = bool(be.last_battle_won)
                        _battle_over(g, won)
                        if g.until == "battle" or not won:
                            g.done = True
                            return
                    continue
            return
        if ph == Phase.RUN_OVER or g.until == "battle":
            g.done = True
            return
        if ph == Phase.RENTAL and g.battles > 0:            # the round is over
            g.done = True
            return
        if tactician is None:
            g.done = True
            return
        be.act(tactician(g))
        g.tdec += 1


def play_lockstep(feeder, battler, on_done, width=256, tactician=None):
    """Play every game the feeder yields, `width` at a time, battle decisions batched across games.
    on_done(game) is called when a game ends."""
    active, it, exhausted = [], iter(feeder), False
    while True:
        while not exhausted and len(active) < width:
            try:
                g = next(it)
            except StopIteration:
                exhausted = True
                break
            active.append(g)
        if not active:
            return
        need = []
        for g in active:
            advance(g, tactician)
            if not g.done:
                need.append(g)
        acts = battler.act_many(need)
        for g, a in zip(need, acts):
            g.be.act(a)
            g.ndec += 1
            g.decisions += 1
            if g.be.phase not in _BATTLE:
                won = bool(g.be.last_battle_won)
                _battle_over(g, won)
                if g.until == "battle" or not won:
                    g.done = True
        for g in active:
            if g.done:
                on_done(g)
        active = [g for g in active if not g.done]


# ---- the benchmark: generation ----------------------------------------------------------------------------------

def bench_dir(name=BENCH_NAME):
    return os.path.join(PROBE_DIR, name)


def gen_round(i, pattern=(5, 6, 5, 6, 3)):
    return pattern[i % len(pattern)]


def generate_run(agents, ckpt, i, seed=0):
    """One run of v3 final from the start of round gen_round(i), until the round is completed or lost.
    -> run record: environment seed, start round, log (every backend action), tactician states and battles."""
    from .autopsy import enc_action, state_hash, team_of
    from .envs import FactoryEnv
    from pybattle.emu.decode import SYMBOLS as S, decode_party
    k = gen_round(i)
    env_seed = stable_seed("probe-gen", seed, i)
    env = FactoryEnv(env_seed, win_streak=7 * (k - 1))
    be = env.backend
    battler = Battler(agents, f"net:{ckpt}")
    log, states, battles = [], [], []
    ndec, wins = 0, 0
    cur = None
    while be.phase != Phase.RUN_OVER:
        ph = be.phase
        if ph in (Phase.RENTAL, Phase.SWAP):
            if ph == Phase.RENTAL and battles:
                break                                    # round completed
            kind, obs, ctx = tactician_obs(be)
            info = be.run_info()
            st = {"li": len(log), "kind": kind, "hash": state_hash(be.game), "obs_hash": obs_hash(obs),
                  "round": info.challenge_num + 1, "battle": info.battle_in_challenge + 1,
                  "noland": bool(info.noland), "hint": [int(be.view().hint_type), int(be.view().hint_style)],
                  "streak": info.win_streak}
            pseed = feat_seed(f"{i}-{len(states)}")      # = C1's: v3's C1 choice is the generation's option
            c, options, feats = tactician_decide(agents, f"net:{ckpt}", be, pseed)
            action = option_to_action(kind, options[c])
            pre = be.clone()
            pre.act(action)
            enemy = pre.game.read(S.addr("gEnemyParty"), 600) if pre.phase in _BATTLE else b""
            st.update(option=c, action=enc_action(action), n_options=len(options),
                      enemy_sha=hashlib.sha1(enemy).hexdigest()[:16],
                      own=team_of(decode_party(pre.game.read(S.addr("gPlayerParty"), 300))),
                      opp=team_of(decode_party(enemy)) if enemy else None)
            states.append(st)
            log.append(enc_action(action))
            be.act(action)
            cur = st
            ndec = 0
            continue
        if ndec >= MAX_DECISIONS:
            cur["won"] = False
            cur["truncated"] = True
            break
        if ph == Phase.BATTLE:
            v = be.view()
            if not any(v.usable_moves) and not v.switch_targets:
                a = ("move", 0)
            else:
                a = battler.act_many([_G(be, ndec)])[0]
        else:
            a = battler.act_many([_G(be, ndec)])[0]
        log.append(enc_action(a))
        be.act(a)
        ndec += 1
        if be.phase not in _BATTLE:
            won = bool(be.last_battle_won)
            cur["won"] = won
            cur["decisions"] = ndec
            battles.append(cur["li"])
            if not won:
                break
            wins += 1
            if wins == 7:
                break
    return {"run": i, "env_seed": env_seed, "start_round": k, "log": log, "states": states}


class _G:
    __slots__ = ("be", "ndec", "key")

    def __init__(self, be, ndec, key="gen"):
        self.be, self.ndec, self.key = be, ndec, key


def _gen_job(args):
    ckpt, idx, seed, device, out = args
    agents = Agents(device)
    done = {r["run"] for r in read_jsonl(out)}
    t0 = time.time()
    for i in idx:
        if i in done:
            continue
        write_jsonl(out, [generate_run(agents, ckpt, i, seed)])
    return time.time() - t0


def select_states(runs, quota_b, n_rental, n_swap, n_c2_rental, n_c2_swap, seed=0):
    """Sample the B / C1 / C2 states (stratified, deterministic)."""
    rng = random.Random(seed)
    allst = []
    for r in runs:
        for j, s in enumerate(r["states"]):
            s = dict(s)
            s["id"] = f"{r['run']}-{j}"
            s["run"] = r["run"]
            s["stratum"] = "noland" if s["noland"] else f"r{s['round']}"
            allst.append(s)
    played = [s for s in allst if s.get("won") is not None and not s.get("truncated")]
    b = []
    for stratum, q in quota_b.items():
        pool = [s for s in played if s["stratum"] == stratum]
        rng.shuffle(pool)
        b += pool[:q]
    rentals = [s for s in allst if s["kind"] == "rental"]
    swaps = [s for s in allst if s["kind"] == "swap"]
    rng.shuffle(rentals)
    rng.shuffle(swaps)
    c1 = rentals[:n_rental] + swaps[:n_swap]

    def strat(pool, n):
        by = defaultdict(list)
        for s in pool:
            by[s["stratum"]].append(s)
        out, keys = [], sorted(by)
        while len(out) < n and any(by[k] for k in keys):
            for k in keys:
                if by[k] and len(out) < n:
                    out.append(by[k].pop())
        return out

    c2 = strat([s for s in c1 if s["kind"] == "rental"], n_c2_rental) + \
        strat([s for s in c1 if s["kind"] == "swap"], n_c2_swap)
    return b, c1, c2


def generate(a):
    import multiprocessing as mp
    out_dir = bench_dir(a.bench)
    os.makedirs(out_dir, exist_ok=True)
    meta_path = os.path.join(out_dir, "meta.json")
    if os.path.exists(meta_path) and not a.force:
        print(f"{meta_path} exists: the benchmark is fixed (--force to rebuild the selection)")
        return
    t0 = time.time()
    procs = max(1, a.procs)
    jobs = [(a.gen_ckpt, list(range(p, a.runs, procs)), a.seed, a.device, os.path.join(out_dir, f"runs.p{p}.jsonl"))
            for p in range(procs)]
    if procs == 1:
        [_gen_job(j) for j in jobs]
    else:
        with mp.get_context("spawn").Pool(procs) as pool:
            pool.map(_gen_job, jobs)
    runs = sorted(read_shards(os.path.join(out_dir, "runs.p")), key=lambda r: r["run"])
    runs = [r for r in runs if r["run"] < a.runs]
    write_jsonl(os.path.join(out_dir, "runs.jsonl.gz"), runs, mode="w")
    quota = dict((k, int(v)) for k, v in (x.split(":") for x in a.b_quota.split(",")))
    b, c1, c2 = select_states(runs, quota, a.c1_rental, a.c1_swap, a.c2_rental, a.c2_swap, a.seed)
    b, c1, c2 = ([dict(s) for s in x] for x in (b, c1, c2))     # (a state can be in several sets: own seeds)
    for s in b:
        s["seeds"] = [stable_seed("probe-B", s["id"], j) for j in range(a.b_seeds)]
    for s in c2:
        s["seeds"] = [stable_seed("probe-C2", s["id"], j) for j in range(a.c2_seeds)]
    for name, rows in (("battles", b), ("tactician", c1), ("outcomes", c2)):
        json.dump(rows, open(os.path.join(out_dir, f"{name}.json"), "w"))
    n_battles = sum(1 for r in runs for s in r["states"] if s.get("won") is not None)
    meta = {"generator": a.gen_ckpt, "generator_tag": ckpt_info(a.gen_ckpt)["tag"], "runs": len(runs),
            "encode_version": ckpt_info(a.gen_ckpt)["encode_version"],
            "seed": a.seed, "battles_played": n_battles,
            "b": {"states": len(b), "seeds": a.b_seeds, "by_stratum": _count(b, "stratum")},
            "c1": {"states": len(c1), "by_kind": _count(c1, "kind"), "by_stratum": _count(c1, "stratum")},
            "c2": {"states": len(c2), "seeds": a.c2_seeds, "by_kind": _count(c2, "kind"),
                   "by_stratum": _count(c2, "stratum")},
            "seconds": round(time.time() - t0, 1), "created": time.strftime("%Y-%m-%d %H:%M")}
    json.dump(meta, open(meta_path, "w"), indent=1)
    print(json.dumps(meta, indent=1))


def _count(rows, key):
    out = defaultdict(int)
    for r in rows:
        out[r[key]] += 1
    return dict(sorted(out.items()))


# ---- the benchmark: loading and replay ----------------------------------------------------------------------------

class Bench:
    def __init__(self, name=BENCH_NAME):
        d = bench_dir(name)
        if not os.path.exists(os.path.join(d, "meta.json")):
            raise FileNotFoundError(f"{d}: no benchmark (python -m rl.agent_probe generate)")
        self.dir = d
        self.meta = json.load(open(os.path.join(d, "meta.json")))
        from . import encode
        encode.set_version(self.meta.get("encode_version", 5))   # the observation hashes of the generation
        self.runs = {r["run"]: r for r in read_jsonl(os.path.join(d, "runs.jsonl.gz"))}
        self.battles = json.load(open(os.path.join(d, "battles.json")))
        self.tactician = json.load(open(os.path.join(d, "tactician.json")))
        self.outcomes = json.load(open(os.path.join(d, "outcomes.json")))
        self.by_id = {s["id"]: s for s in self.tactician + self.battles + self.outcomes}   # kind, stratum...
        self.b_by_id = {s["id"]: s for s in self.battles}
        self.c2_by_id = {s["id"]: s for s in self.outcomes}


def replay_states(bench, states, check=True):
    """Yield (state, backend at its tactician decision), replaying each run once (states sorted by run, index).
    check: the state hash and the observation hash must match the generation."""
    from .autopsy import dec_action, state_hash
    from .envs import FactoryEnv
    by_run = defaultdict(list)
    for s in states:
        by_run[s["run"]].append(s)
    for run_id in sorted(by_run):
        r = bench.runs[run_id]
        env = FactoryEnv(r["env_seed"], win_streak=7 * (r["start_round"] - 1))
        be = env.backend
        li = 0
        for s in sorted(by_run[run_id], key=lambda s: s["li"]):
            while li < s["li"]:
                be.act(dec_action(r["log"][li]))
                li += 1
            if check:
                _, obs, _ = tactician_obs(be)
                if state_hash(be.game) != s["hash"] or obs_hash(obs) != s["obs_hash"]:
                    raise RuntimeError(f"replay of state {s['id']} diverged from the generation")
            yield s, be.clone()


def battle_start(pre, action, seed):
    """The battle after `action` at the tactician decision `pre`, its RNG reseeded before the first frame (the
    opponent: the one the game generates with that option, as in the real game; rl.autopsy.start_battle)."""
    from .autopsy import start_battle
    return start_battle(pre, action, None, seed)


def _enemy_sha(be):
    from pybattle.emu.decode import SYMBOLS as S
    return hashlib.sha1(be.game.read(S.addr("gEnemyParty"), 600)).hexdigest()[:16]


# ---- sharded execution --------------------------------------------------------------------------------------------

def run_sharded(fn, job_common, procs):
    """fn(job_common, p, procs) in `procs` spawned processes (each loads its own networks)."""
    import multiprocessing as mp
    if procs <= 1:
        return [fn(job_common, 0, 1)]
    with mp.get_context("spawn").Pool(procs) as pool:
        return pool.starmap(fn, [(job_common, p, procs) for p in range(procs)])


def _shard(items, p, procs, key):
    return [x for x in items if stable_seed(key(x)) % procs == p]


# ---- B: battler benchmark -----------------------------------------------------------------------------------------

def results_dir(*parts):
    return os.path.join(PROBE_DIR, "results", *parts)


def battler_prefix(spec, bench_name=BENCH_NAME):
    return os.path.join(PROBE_DIR, "results", bench_name, "battler", spec_tag(spec))


def b_states(bench, subset=None, seed_subset=None):
    """The B states (the first `subset` of a fixed shuffled order, stratified by the generation's sampling)."""
    st = list(bench.battles)
    if subset:
        order = sorted(st, key=lambda s: stable_seed("probe-B-subset", s["id"]))
        st = order[:subset]
    return st


def _battler_job(c, p, procs):
    agents = Agents(c["device"])
    bench = Bench(c["bench"])
    battler = Battler(agents, c["spec"])
    states = _shard(b_states(bench, c["subset"]), p, procs, lambda s: s["run"])
    out = f"{c['prefix']}.p{p}.jsonl"
    done = {(r["id"], r["seed"]) for r in read_shards(c["prefix"])}
    k = c["seeds"]
    todo = [s for s in states if any((s["id"], sd) not in done for sd in s["seeds"][:k])]
    t0, n_dec, buf = time.time(), [0], []

    def feeder():
        for s, pre in replay_states(bench, todo):
            action = _dec(s["action"])
            check_opponent(s, pre, action)
            for sd in s["seeds"][:k]:
                if (s["id"], sd) in done:
                    continue
                be = battle_start(pre, action, sd)
                yield Game((s["id"], sd), be, "battle", {"id": s["id"], "seed": sd})

    def on_done(g):
        n_dec[0] += g.decisions
        buf.append({"id": g.meta["id"], "seed": g.meta["seed"], "won": int(g.wins), "dec": g.decisions,
                    "trunc": int(g.truncated)})
        if len(buf) >= 64:
            write_jsonl(out, buf)
            buf.clear()

    play_lockstep(feeder(), battler, on_done, width=c["width"] if battler.kind == "net" else 1)
    write_jsonl(out, buf)
    return {"seconds": time.time() - t0, "decisions": n_dec[0]}


def check_opponent(s, pre, action):
    """The option applied at the replayed decision must give the opponent of the generation (same gEnemyParty)."""
    b = pre.clone()
    b.act(action)
    if _enemy_sha(b) != s["enemy_sha"]:
        raise RuntimeError(f"state {s['id']}: the opponent differs from the generation")


def _dec(a):
    from .autopsy import dec_action
    return dec_action(a)


def run_battler(spec, procs=2, subset=None, seeds=4, device="cuda", width=256, bench_name=BENCH_NAME):
    prefix = battler_prefix(spec, bench_name)
    c = {"spec": spec, "subset": subset, "seeds": seeds, "device": device, "width": width, "bench": bench_name,
         "prefix": prefix}
    t0 = time.time()
    stats = run_sharded(_battler_job, c, procs)
    dec = sum(s["decisions"] for s in stats)
    el = time.time() - t0
    timing = {"seconds": round(el, 1), "decisions": dec, "procs": procs}
    _timing_log(f"battler {spec_tag(spec)} subset={subset} seeds={seeds}", timing)
    return summarize_battler(spec, bench_name)


def summarize_battler(spec, bench_name=BENCH_NAME, ids=None, seeds=None):
    """Per state: mean over its seeds. -> {"overall": (mean, ci, n), "by_stratum": {...}, "per_state": {id: mean}}."""
    bench = Bench(bench_name)
    rows = read_shards(battler_prefix(spec, bench_name))
    per = defaultdict(list)
    for r in rows:
        if seeds is not None and r["seed"] not in seeds.get(r["id"], ()):
            continue
        per[r["id"]].append(r["won"])
    per = {k: float(np.mean(v)) for k, v in per.items() if ids is None or k in ids}
    strata = defaultdict(list)
    for k, v in per.items():
        strata[bench.by_id[k]["stratum"]].append(v)
    return {"overall": mean_ci(list(per.values())), "by_stratum": {k: mean_ci(v) for k, v in sorted(strata.items())},
            "per_state": per}


def paired_diff(per_a, per_b, bench=None, stratum=None):
    ids = sorted(set(per_a) & set(per_b))
    if stratum is not None:
        ids = [i for i in ids if bench.by_id[i]["stratum"] == stratum]
    return mean_ci([per_a[i] - per_b[i] for i in ids])


# ---- C1: tactician choices and Q_plan regret --------------------------------------------------------------------

def tactician_prefix(spec, bench_name=BENCH_NAME):
    return os.path.join(PROBE_DIR, "results", bench_name, "tactician", spec_tag(spec))


def _tactician_job(c, p, procs):
    agents = Agents("cpu")
    bench = Bench(c["bench"])
    states = _shard(bench.tactician, p, procs, lambda s: s["run"])
    done = {r["id"] for r in read_shards(c["prefix"])}
    todo = [s for s in states if s["id"] not in done]
    out = f"{c['prefix']}.p{p}.jsonl"
    t0, buf = time.time(), []
    for s, pre in replay_states(bench, todo):
        cache = {}
        rows = {}
        for spec in c["specs"]:
            ch, options, feats = tactician_decide(agents, spec, pre, feat_seed(s["id"]), cache)
            rows[spec] = {"id": s["id"], "choice": ch, "n": len(options),
                          "q": np.round(feats[:, Q_COL] * WIN_SCALE, 4).tolist()}
        for spec, r in rows.items():
            buf.append((spec, r))
        if len(buf) >= 32:
            _flush_t(buf, c, p)
    _flush_t(buf, c, p)
    return {"seconds": time.time() - t0, "states": len(todo)}


def _flush_t(buf, c, p):
    by = defaultdict(list)
    for spec, r in buf:
        by[spec].append(r)
    for spec, rows in by.items():
        write_jsonl(f"{tactician_prefix(spec, c['bench'])}.p{p}.jsonl", rows)
    buf.clear()


def run_tactician(ckpt, procs=2, bench_name=BENCH_NAME):
    """C1 for a checkpoint: its network's choice and the planner's (argmax of its own Q_plan), same features."""
    specs = [f"net:{ckpt}", f"planner:{ckpt}"]
    c = {"specs": specs, "bench": bench_name, "prefix": tactician_prefix(specs[0], bench_name)}
    t0 = time.time()
    stats = run_sharded(_tactician_job, c, procs)
    _timing_log(f"tactician {spec_tag(specs[0])}", {"seconds": round(time.time() - t0, 1),
                                                   "states": sum(s["states"] for s in stats), "procs": procs})


def load_choices(spec, bench_name=BENCH_NAME):
    return {r["id"]: r for r in read_shards(tactician_prefix(spec, bench_name))}


def summarize_c1(spec, ref_spec, bench_name=BENCH_NAME):
    """Q_plan regret (wins) of the spec's choices under the reference planner's Q_plan (v3 final's W, same seeds),
    and agreement with the reference argmax, by kind."""
    bench = Bench(bench_name)
    ch, ref = load_choices(spec, bench_name), load_choices(ref_spec, bench_name)
    out = {}
    for kind in ("rental", "swap"):
        reg, agree = [], []
        for i, r in ch.items():
            if i not in ref or bench.by_id[i]["kind"] != kind:
                continue
            q = np.asarray(ref[i]["q"])
            reg.append(float(q.max() - q[r["choice"]]))
            agree.append(float(r["choice"] == int(np.argmax(q))))
        out[kind] = {"regret": mean_ci(reg), "agree": mean_ci(agree)}
    return out


# ---- C2: real-outcome regret --------------------------------------------------------------------------------------

def outcomes_prefix(bench_name=BENCH_NAME):
    return os.path.join(PROBE_DIR, "results", bench_name, "outcomes", "table")


def c2_options(bench, tactician_specs, top_k=8, bench_name=BENCH_NAME):
    """Per C2 state, the options to play: every legal swap option; for a rental the top_k of the reference Q_plan
    (v3 final's) plus every option some evaluated tactician chose."""
    ref = load_choices(f"net:{GEN_CKPT}", bench_name)
    chosen = defaultdict(set)
    for spec in tactician_specs:
        for i, r in load_choices(spec, bench_name).items():
            chosen[i].add(r["choice"])
    out = {}
    for s in bench.outcomes:
        if s["id"] not in ref:
            continue
        n = ref[s["id"]]["n"]
        if s["kind"] == "swap":
            opts = set(range(n))
        else:
            q = np.asarray(ref[s["id"]]["q"])
            opts = set(int(x) for x in np.argsort(-q, kind="stable")[:top_k]) | {s["option"]}
        out[s["id"]] = sorted(opts | chosen[s["id"]])
    return out


def _outcomes_job(c, p, procs):
    agents = Agents(c["device"])
    bench = Bench(c["bench"])
    battler = Battler(agents, f"net:{c['battler']}")
    cont = f"net:{c['tactician']}"
    want = c["want"]
    states = _shard([s for s in bench.outcomes if s["id"] in want], p, procs, lambda s: s["run"])
    done = {(r["id"], r["opt"], r["seed"]) for r in read_shards(c["prefix"])}
    out = f"{c['prefix']}.p{p}.jsonl"
    nseeds = c["seeds"]
    t0, buf, n_dec = time.time(), [], [0]

    def todo_of(s):
        return [(o, sd) for o in want[s["id"]] for sd in s["seeds"][:nseeds] if (s["id"], o, sd) not in done]

    todo = [s for s in states if todo_of(s)]

    def feeder():
        from .tactician_search import options_of
        for s, pre in replay_states(bench, todo):
            kind, obs, _ = tactician_obs(pre)
            options = options_of(kind, obs)
            for o, sd in todo_of(s):
                be = battle_start(pre, option_to_action(kind, options[o]), sd)
                yield Game((s["id"], o, sd), be, "round", {"id": s["id"], "opt": o, "seed": sd})

    def tact(g):
        seed = stable_seed("probe-C2-cont", g.key, g.tdec)
        ch, options, _ = tactician_decide(agents, cont, g.be, seed)
        kind = "rental" if g.be.phase == Phase.RENTAL else "swap"
        return option_to_action(kind, options[ch])

    def on_done(g):
        n_dec[0] += g.decisions
        buf.append({"id": g.meta["id"], "opt": g.meta["opt"], "seed": g.meta["seed"], "wins": g.wins,
                    "first": int(bool(g.first_won)), "trunc": int(g.truncated)})
        if len(buf) >= 64:
            write_jsonl(out, buf)
            buf.clear()

    play_lockstep(feeder(), battler, on_done, width=c["width"], tactician=tact)
    write_jsonl(out, buf)
    return {"seconds": time.time() - t0, "decisions": n_dec[0]}


def run_outcomes(tactician_specs, procs=2, seeds=32, device="cuda", width=256, top_k=8, bench_name=BENCH_NAME):
    """Extend the outcome table to every option needed by these tacticians (cached: computed once per option)."""
    bench = Bench(bench_name)
    want = c2_options(bench, tactician_specs, top_k, bench_name)
    c = {"bench": bench_name, "battler": GEN_CKPT, "tactician": GEN_CKPT, "want": want, "seeds": seeds,
         "device": device, "width": width, "prefix": outcomes_prefix(bench_name)}
    t0 = time.time()
    stats = run_sharded(_outcomes_job, c, procs)
    _timing_log("outcomes C2", {"seconds": round(time.time() - t0, 1), "decisions": sum(s["decisions"] for s in stats),
                                "procs": procs, "options": sum(len(v) for v in want.values())})


def outcome_table(bench_name=BENCH_NAME, seeds=None):
    """{state id: {option: np.array of wins per seed (ordered as the state's seeds)}}."""
    bench = Bench(bench_name)
    tab = defaultdict(lambda: defaultdict(dict))
    for r in read_shards(outcomes_prefix(bench_name)):
        tab[r["id"]][r["opt"]][r["seed"]] = r["wins"]
    out = {}
    for i, opts in tab.items():
        sd = bench.c2_by_id[i]["seeds"][:seeds] if seeds else bench.c2_by_id[i]["seeds"]
        full = {o: np.array([v[s] for s in sd], np.float64) for o, v in opts.items() if all(s in v for s in sd)}
        if full:
            out[i] = full
    return out


def c2_regret(table, choices):
    """Per state with the chosen option in the table: raw regret = max_o mean(o) - mean(chosen) (biased up: the max of
    noisy means) and cross-validated regret (best option picked on one half of the seeds, measured on the other,
    both halves averaged: biased down). -> lists (raw, cv, ids)."""
    raw, cv, ids = [], [], []
    for i, opts in table.items():
        if i not in choices or choices[i] not in opts:
            continue
        c = choices[i]
        m = {o: v.mean() for o, v in opts.items()}
        raw.append(max(m.values()) - m[c])
        n = len(opts[c])
        h = n // 2
        r2 = []
        for a_, b_ in ((slice(0, h), slice(h, n)), (slice(h, n), slice(0, h))):
            best = max(opts, key=lambda o: (opts[o][a_].mean(), -o))
            r2.append(opts[best][b_].mean() - opts[c][b_].mean())
        cv.append(float(np.mean(r2)))
        ids.append(i)
    return raw, cv, ids


def summarize_c2(spec, bench_name=BENCH_NAME, table=None):
    bench = Bench(bench_name)
    table = table if table is not None else outcome_table(bench_name)
    ch = {i: r["choice"] for i, r in load_choices(spec, bench_name).items()}
    out = {}
    for kind in ("all", "rental", "swap"):
        t = {i: v for i, v in table.items() if kind == "all" or bench.by_id[i]["kind"] == kind}
        raw, cv, ids = c2_regret(t, ch)
        val = [t[i][ch[i]].mean() for i in ids]
        out[kind] = {"raw": mean_ci(raw), "cv": mean_ci(cv), "value": mean_ci(val)}
    return out


# ---- A: cross-play ----------------------------------------------------------------------------------------------

def crossplay_path(t_spec, b_spec, runs, seed=777):
    return os.path.join(PROBE_DIR, "results", "crossplay", f"T_{spec_tag(t_spec)}__B_{spec_tag(b_spec)}__n{runs}"
                        f"_s{seed}")


def _crossplay_job(c, p, procs):
    from .envs import FactoryEnv
    agents = Agents(c["device"])
    battler = Battler(agents, c["battler"])
    prefix = c["prefix"]
    done = {(r["round"], r["i"]) for r in read_shards(prefix)}
    keys = [(k, i) for k in c["rounds"] for i in range(c["runs"])]
    keys = [x for x in _shard(keys, p, procs, lambda x: x) if x not in done]
    out = f"{prefix}.p{p}.jsonl"
    t0, buf, n_dec, n_t = time.time(), [], [0], [0]

    def feeder():
        for k, i in keys:
            env_seed = (c["seed"] + 1000 * k) * 100_003 + i
            env = FactoryEnv(env_seed, win_streak=7 * (k - 1))
            yield Game((k, i), env.backend, "round", {"round": k, "i": i})

    def tact(g):
        t1 = time.time()
        ch, options, _ = tactician_decide(agents, c["tactician"], g.be, stable_seed("probe-A", g.key, g.tdec))
        kind = "rental" if g.be.phase == Phase.RENTAL else "swap"
        n_t[0] += time.time() - t1
        return option_to_action(kind, options[ch])

    def on_done(g):
        n_dec[0] += g.decisions
        buf.append({"round": g.meta["round"], "i": g.meta["i"], "wins": g.wins, "complete": int(g.wins == 7),
                    "trunc": int(g.truncated)})
        if len(buf) >= 32:
            write_jsonl(out, buf)
            buf.clear()

    play_lockstep(feeder(), battler, on_done, width=c["width"], tactician=tact)
    write_jsonl(out, buf)
    return {"seconds": time.time() - t0, "decisions": n_dec[0], "tactician_seconds": n_t[0]}


def run_crossplay(t_spec, b_spec, runs=256, rounds=(1, 2, 3, 4, 5, 6), procs=2, device="cuda", width=128, seed=777):
    prefix = crossplay_path(t_spec, b_spec, runs, seed)
    c = {"tactician": t_spec, "battler": b_spec, "runs": runs, "rounds": list(rounds), "device": device,
         "width": width, "seed": seed, "prefix": prefix}
    t0 = time.time()
    stats = run_sharded(_crossplay_job, c, procs)
    _timing_log(f"crossplay T={spec_tag(t_spec)} B={spec_tag(b_spec)} n={runs}",
                {"seconds": round(time.time() - t0, 1), "decisions": sum(s["decisions"] for s in stats),
                 "tactician_seconds": round(sum(s["tactician_seconds"] for s in stats), 1), "procs": procs})
    return summarize_crossplay(t_spec, b_spec, runs, seed)


def summarize_crossplay(t_spec, b_spec, runs=256, seed=777, B=2000):
    """Per round: P(complete) and wins per battle; P(6 rounds) = product of the rounds' P(complete), with a
    bootstrap interval (runs resampled within each round)."""
    rows = read_shards(crossplay_path(t_spec, b_spec, runs, seed))
    by = defaultdict(dict)
    for r in rows:
        by[r["round"]][r["i"]] = r
    out = {"rounds": {}, "per_run": {}}
    rng = np.random.default_rng(0)
    boots = np.ones(B)
    p6 = 1.0
    for k in sorted(by):
        rr = [by[k][i] for i in sorted(by[k])]
        c = np.array([r["complete"] for r in rr], np.float64)
        w = np.array([r["wins"] for r in rr], np.float64)
        losses = (1 - c).sum()
        out["rounds"][k] = {"complete": mean_ci(c), "battle": float(w.sum() / max(w.sum() + losses, 1)),
                            "n": len(rr)}
        out["per_run"][k] = {i: by[k][i]["complete"] for i in by[k]}
        p6 *= c.mean()
        boots *= c[rng.integers(0, len(c), (B, len(c)))].mean(1)
    out["p6"] = (float(p6), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    return out


# ---- timing log ----------------------------------------------------------------------------------------------

def _timing_log(what, d):
    d = dict(d, what=what, at=time.strftime("%Y-%m-%d %H:%M:%S"))
    write_jsonl(os.path.join(PROBE_DIR, "timing.jsonl"), [d])
    print(f"[probe] {what}: {json.dumps(d)}", flush=True)


# ---- the learning-curve driver -----------------------------------------------------------------------------------

def v4_ckpt(it, run="runs/alphazero_v4", steps_per_it=40014):
    return os.path.join(run, f"ckpt_{steps_per_it * (it + 1):011d}.pt")


def curve(a):
    ckpts = [c for c in a.ckpts.split(",") if c]
    for c in ckpts:
        if not os.path.exists(c):
            raise FileNotFoundError(c)
    if GEN_CKPT not in ckpts:
        ckpts = [GEN_CKPT] + ckpts
    latest = a.latest or ckpts[-1]
    if not os.path.exists(os.path.join(bench_dir(a.bench), "meta.json")):
        generate(a)
    # B: every checkpoint's network battler, the full benchmark
    for c in ckpts:
        if not a.skip_b:
            run_battler(f"net:{c}", a.procs, None, a.b_seeds, a.device, a.width, a.bench)
    # C1: every checkpoint (network and planner)
    for c in ckpts:
        run_tactician(c, a.procs, a.bench)
    # C2: the outcome table for every option chosen
    if not a.skip_c2:
        specs = [f"{k}:{c}" for c in ckpts for k in ("net", "planner")]
        run_outcomes(specs, a.procs, a.c2_seeds, a.device, a.width, a.c2_top, a.bench)
    # A: the cross-play grid
    if not a.skip_a:
        v3 = GEN_CKPT
        grid = [(f"net:{v3}", f"net:{v3}"), (f"net:{latest}", f"net:{latest}"), (f"net:{v3}", f"net:{latest}"),
                (f"net:{latest}", f"net:{v3}"), (f"planner:{latest}", f"net:{latest}"),
                (f"planner:{v3}", f"net:{v3}")]
        for t, b in grid:
            run_crossplay(t, b, a.a_runs, procs=a.procs, device=a.device, width=a.width)
    report(a, ckpts, latest)
    # B with search on a subset (optional, the costliest: last)
    specs = [s for s in (a.search or "").split(",") if s]
    for spec in specs:
        run_battler(spec, a.procs, a.search_subset, a.search_seeds, a.device, a.width, a.bench)
    if specs:
        report(a, ckpts, latest)


# ---- report ---------------------------------------------------------------------------------------------------

def _fmt(m, ci=None, pct=True, d=1):
    if m is None or (isinstance(m, float) and math.isnan(m)):
        return "–"
    f = 100.0 if pct else 1.0
    s = f"{m * f:.{d}f}"
    if ci is not None and not (isinstance(ci, float) and math.isnan(ci)):
        s += f" ± {ci * f:.{d}f}"
    return s


def _label(c):
    i = ckpt_info(c)
    return ("v3 final" if i["run"] == "alphazero_v3" else f"v4 it {i['iteration']}")


def report(a, ckpts=None, latest=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir = os.path.join(PROBE_DIR, "report")
    os.makedirs(out_dir, exist_ok=True)
    bench = Bench(a.bench)
    if ckpts is None:
        ckpts = [c for c in a.ckpts.split(",") if c]
        if GEN_CKPT not in ckpts:
            ckpts = [GEN_CKPT] + ckpts
    latest = latest or a.latest or ckpts[-1]
    v4 = sorted([c for c in ckpts if ckpt_info(c)["run"] != "alphazero_v3"], key=lambda c: ckpt_info(c)["iteration"])
    summary = {"bench": bench.meta, "battler": {}, "c1": {}, "c2": {}, "crossplay": {}, "search": {}}
    L = [f"# Sondas por agente: combatiente y táctico por separado ({time.strftime('%Y-%m-%d')})", ""]
    m = bench.meta
    L += [f"Banco fijo `{bench.dir}`: {m['runs']} rachas de {m['generator_tag']} ({m['battles_played']} combates). "
          f"B: {m['b']['states']} inicios de combate × {m['b']['seeds']} semillas ({m['b']['by_stratum']}). "
          f"C1: {m['c1']['states']} decisiones ({m['c1']['by_kind']}). C2: {m['c2']['states']} decisiones "
          f"× {m['c2']['seeds']} semillas ({m['c2']['by_kind']}).", ""]

    # ---- B
    ref_b = summarize_battler(f"net:{GEN_CKPT}", a.bench)
    L += ["## B. Combatiente solo (red voraz) en el banco fijo", "",
          "Victorias por combate (%), media sobre estados ± IC 95 %; Δ = diferencia emparejada con v3 final "
          "(mismos estados y semillas).", "",
          "| combatiente | total | R3 | R5 | R6 | Noland | Δ total vs v3 |", "|---|---|---|---|---|---|---|"]
    for c in ckpts:
        s = summarize_battler(f"net:{c}", a.bench)
        if not s["per_state"]:
            continue
        d = paired_diff(s["per_state"], ref_b["per_state"])
        summary["battler"][_label(c)] = {"overall": s["overall"], "by_stratum": s["by_stratum"], "diff_v3": d,
                                         "iteration": ckpt_info(c)["iteration"]}
        bs = s["by_stratum"]
        L.append(f"| {_label(c)} | {_fmt(*s['overall'][:2])} | " + " | ".join(
            _fmt(*bs.get(k, (float('nan'), float('nan')))[:2]) for k in ("r3", "r5", "r6", "noland"))
            + f" | {_fmt(*d[:2]) if c != GEN_CKPT else '–'} |")
    L.append("")
    # ceilings on a subset
    srows = []
    for spec in (a.search or "").split(","):
        if not spec:
            continue
        s = summarize_battler(spec, a.bench)
        if not s["per_state"]:
            continue
        ids = set(s["per_state"])
        seeds = {i: set(bench.b_by_id[i]["seeds"][:a.search_seeds]) for i in ids}
        kind, path, _ = parse_spec(spec)
        net = summarize_battler(f"net:{path}", a.bench, ids=ids, seeds=seeds)
        d = paired_diff(s["per_state"], net["per_state"])
        summary["search"][spec_tag(spec)] = {"overall": s["overall"], "net_same": net["overall"], "diff": d}
        srows.append(f"| {kind} {parse_spec(spec)[2]} ({_label(path)}) | {len(ids)} × {a.search_seeds} | "
                     f"{_fmt(*s['overall'][:2])} | {_fmt(*net['overall'][:2])} | {_fmt(*d[:2])} |")
    if srows:
        L += ["Búsqueda sobre un subconjunto (mismos estados y semillas que la red):", "",
              "| combatiente | estados × semillas | victorias | red sola, mismos | Δ |", "|---|---|---|---|---|"] + srows
        L.append("")

    # ---- C1
    ref_t = f"net:{GEN_CKPT}"
    L += ["## C1. Táctico: regret según el planificador de referencia (W de v3 final)", "",
          "Regret_Q = máx Q_plan − Q_plan(elegida), en victorias de la ronda (Q_plan de v3 final, mismas semillas "
          "del planificador); acuerdo = elige el máximo.", "",
          "| táctico | regret alquiler | regret cambio | acuerdo alquiler | acuerdo cambio |", "|---|---|---|---|---|"]
    for c in ckpts:
        for k in ("net", "planner"):
            spec = f"{k}:{c}"
            if not load_choices(spec, a.bench):
                continue
            s = summarize_c1(spec, ref_t, a.bench)
            name = f"{'red' if k == 'net' else 'planificador'} {_label(c)}"
            summary["c1"][name] = s
            L.append(f"| {name} | {_fmt(*s['rental']['regret'][:2], pct=False, d=3)} | "
                     f"{_fmt(*s['swap']['regret'][:2], pct=False, d=3)} | {_fmt(*s['rental']['agree'][:2])} | "
                     f"{_fmt(*s['swap']['agree'][:2])} |")
    L.append("")

    # ---- C2
    table = outcome_table(a.bench)
    if table:
        n_opt = sum(len(v) for v in table.values())
        L += ["## C2. Táctico: regret por el resultado real (combatiente fijo: red de v3 final)", "",
              f"{len(table)} decisiones, {n_opt} opciones jugadas × {m['c2']['seeds']} semillas hasta el final de la "
              "ronda (victorias que quedan en la ronda). Regret bruto (sesgado hacia arriba) y validado en mitades "
              "de las semillas (sesgado hacia abajo); valor = victorias medias con la opción elegida.", "",
              "| táctico | regret bruto | regret validado | alquiler (validado) | cambio (validado) | valor |",
              "|---|---|---|---|---|---|"]
        for c in ckpts:
            for k in ("net", "planner"):
                spec = f"{k}:{c}"
                if not load_choices(spec, a.bench):
                    continue
                s = summarize_c2(spec, a.bench, table)
                name = f"{'red' if k == 'net' else 'planificador'} {_label(c)}"
                summary["c2"][name] = s
                L.append(f"| {name} | {_fmt(*s['all']['raw'][:2], pct=False, d=3)} | "
                         f"{_fmt(*s['all']['cv'][:2], pct=False, d=3)} | {_fmt(*s['rental']['cv'][:2], pct=False, d=3)}"
                         f" | {_fmt(*s['swap']['cv'][:2], pct=False, d=3)} | {_fmt(*s['all']['value'][:2], pct=False, d=2)} |")
        # the best option itself (oracle, per half) for reference
        best = [max(v.mean() for v in opts.values()) for opts in table.values()]
        L += ["", f"Valor del mejor de las opciones jugadas (con el sesgo del máximo): {_fmt(*mean_ci(best)[:2], pct=False, d=2)}"
              " victorias.", ""]

    # ---- A
    v3 = GEN_CKPT
    grid = [(f"net:{v3}", f"net:{v3}"), (f"net:{latest}", f"net:{latest}"), (f"net:{v3}", f"net:{latest}"),
            (f"net:{latest}", f"net:{v3}"), (f"planner:{latest}", f"net:{latest}"), (f"planner:{v3}", f"net:{v3}")]
    arows = []
    per_run = {}
    for t, b in grid:
        if not read_shards(crossplay_path(t, b, a.a_runs)):
            continue
        s = summarize_crossplay(t, b, a.a_runs)
        tn = f"{'red' if t.startswith('net') else 'planificador'} {_label(parse_spec(t)[1])}"
        bn = _label(parse_spec(b)[1])
        summary["crossplay"][f"{tn} | {bn}"] = s
        per_run[(t, b)] = s["per_run"]
        r = s["rounds"]
        arows.append(f"| {tn} | {bn} | " + " | ".join(_fmt(r[k]['complete'][0]) if k in r else "–" for k in range(1, 7))
                     + f" | {s['p6'][0] * 100:.2f} [{s['p6'][1] * 100:.2f}, {s['p6'][2] * 100:.2f}] |")
    if arows:
        L += ["## A. Juego cruzado (táctico de X con combatiente de Y; red sola, semilla 777)", "",
              f"P(completar la ronda) por ronda (%), {a.a_runs} rachas por ronda; P(6) = producto, IC bootstrap.", "",
              "| táctico | combatiente | R1 | R2 | R3 | R4 | R5 | R6 | P(6) % |", "|---|---|---|---|---|---|---|---|---|"]
        L += arows
        # paired round-by-round differences of interest
        pairs = [((f"net:{latest}", f"net:{latest}"), (f"net:{v3}", f"net:{latest}"), "táctico v4 − táctico v3 (combatiente v4)"),
                 ((f"net:{latest}", f"net:{v3}"), (f"net:{v3}", f"net:{v3}"), "táctico v4 − táctico v3 (combatiente v3)"),
                 ((f"net:{v3}", f"net:{latest}"), (f"net:{v3}", f"net:{v3}"), "combatiente v4 − combatiente v3 (táctico v3)"),
                 ((f"net:{latest}", f"net:{latest}"), (f"net:{latest}", f"net:{v3}"), "combatiente v4 − combatiente v3 (táctico v4)"),
                 ((f"planner:{latest}", f"net:{latest}"), (f"net:{latest}", f"net:{latest}"), "planificador − red v4 como táctico (combatiente v4)")]
        prow = []
        for x, y, name in pairs:
            if x in per_run and y in per_run:
                d = []
                for k in (5, 6):
                    px, py = per_run[x].get(k, {}), per_run[y].get(k, {})
                    ids = sorted(set(px) & set(py))
                    d.append(mean_ci([px[i] - py[i] for i in ids]))
                allk = []
                for k in range(1, 7):
                    px, py = per_run[x].get(k, {}), per_run[y].get(k, {})
                    allk += [px[i] - py[i] for i in sorted(set(px) & set(py))]
                prow.append(f"| {name} | {_fmt(*mean_ci(allk)[:2])} | {_fmt(*d[0][:2])} | {_fmt(*d[1][:2])} |")
                summary["crossplay"]["diff: " + name] = {"all": mean_ci(allk), "r5": d[0], "r6": d[1]}
        if prow:
            L += ["", "Diferencias emparejadas de P(completar) (puntos, mismas rachas):", "",
                  "| comparación | R1–R6 (media) | R5 | R6 |", "|---|---|---|---|"] + prow
        L.append("")

    # ---- plots
    if v4:
        xs = [ckpt_info(c)["iteration"] for c in v4]
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        for key, lab in (("overall", "total"), ("r5", "R5"), ("r6", "R6"), ("noland", "Noland"), ("r3", "R3")):
            ys, es = [], []
            for c in v4:
                s = summary["battler"].get(_label(c))
                v = s["overall"] if key == "overall" else (s["by_stratum"].get(key) if s else None) if s else None
                ys.append(v[0] * 100 if v else np.nan)
                es.append(v[1] * 100 if v else np.nan)
            ax[0].errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=lab)
            ref = summary["battler"].get("v3 final")
            if ref and key == "overall":
                ax[0].axhline(ref["overall"][0] * 100, color="k", ls="--", lw=1, label="v3 final (total)")
        ax[0].set_title("B: combatiente (red) — victorias por combate %")
        ax[0].set_xlabel("iteración de v4")
        ax[0].legend(fontsize=7)
        for j, kind in enumerate(("rental", "swap")):
            for who, st in (("red", "-o"), ("planificador", "--s")):
                ys, es = [], []
                for c in v4:
                    s = summary["c1"].get(f"{who} {_label(c)}")
                    ys.append(s[kind]["regret"][0] if s else np.nan)
                    es.append(s[kind]["regret"][1] if s else np.nan)
                ax[1].errorbar(xs, ys, yerr=es, fmt=st, capsize=3, label=f"{who}, {'alquiler' if kind == 'rental' else 'cambio'}")
        ax[1].set_title("C1: regret según Q_plan de v3 (victorias)")
        ax[1].set_xlabel("iteración de v4")
        ax[1].legend(fontsize=7)
        for who, st in (("red", "-o"), ("planificador", "--s")):
            ys, es = [], []
            for c in v4:
                s = summary["c2"].get(f"{who} {_label(c)}")
                ys.append(s["all"]["cv"][0] if s else np.nan)
                es.append(s["all"]["cv"][1] if s else np.nan)
            ax[2].errorbar(xs, ys, yerr=es, fmt=st, capsize=3, label=f"{who} (validado)")
            r3 = summary["c2"].get(f"{who} v3 final")
            if r3:
                ax[2].axhline(r3["all"]["cv"][0], ls=":", lw=1, color="k" if who == "red" else "gray",
                              label=f"{who} v3 final")
        ax[2].set_title("C2: regret por resultado real (victorias)")
        ax[2].set_xlabel("iteración de v4")
        ax[2].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "curves.png"), dpi=110)
        plt.close(fig)
        L += ["## Curvas", "", "![curvas](curves.png)", ""]

    # ---- timing
    tl = read_jsonl(os.path.join(PROBE_DIR, "timing.jsonl"))
    if tl:
        L += ["## Coste (cada ejecución)", "", "| qué | segundos | procesos |", "|---|---|---|"]
        for t in tl[-60:]:
            L.append(f"| {t['what']} | {t.get('seconds')} | {t.get('procs')} |")
        L.append("")
    open(os.path.join(out_dir, "report.md"), "w").write("\n".join(L))
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=1, default=str)
    print("\n".join(L))
    return summary


# ---- command line ----------------------------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("cmd", choices=("generate", "battler", "tactician", "outcomes", "crossplay", "curve", "report"))
    p.add_argument("--bench", default=BENCH_NAME)
    p.add_argument("--procs", type=int, default=2)
    p.add_argument("--device", default="cuda", help="the battler networks' device (tactician: CPU)")
    p.add_argument("--width", type=int, default=256, help="games in flight per process (batched battle decisions)")
    # generate
    p.add_argument("--gen-ckpt", default=GEN_CKPT)
    p.add_argument("--runs", type=int, default=500, help="generate: runs (rounds 5 / 6 / 3: 40 / 40 / 20 %%)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true")
    p.add_argument("--b-quota", default="r5:650,r6:650,r3:300,noland:250")
    p.add_argument("--b-seeds", type=int, default=4)
    p.add_argument("--c1-rental", type=int, default=300)
    p.add_argument("--c1-swap", type=int, default=600)
    p.add_argument("--c2-rental", type=int, default=50)
    p.add_argument("--c2-swap", type=int, default=100)
    p.add_argument("--c2-seeds", type=int, default=32)
    p.add_argument("--c2-top", type=int, default=8, help="C2 rentals: the reference planner's top-k options")
    # agents
    p.add_argument("--battler", default=None, help="battler spec: net:CKPT | search:CKPT[:SIMS] | perfect:CKPT[:SIMS]")
    p.add_argument("--tactician", default=None, help="tactician spec: net:CKPT | planner:CKPT")
    p.add_argument("--subset", type=int, default=None, help="battler: the first N states of a fixed order")
    p.add_argument("--seeds", type=int, default=None, help="battler: seeds per state (default --b-seeds)")
    p.add_argument("--a-runs", type=int, default=256, help="crossplay: runs per round")
    p.add_argument("--rounds", default="1-6")
    # curve
    p.add_argument("--ckpts", default="", help="curve / report: comma-separated checkpoints (v3 final is added)")
    p.add_argument("--latest", default=None, help="curve: the checkpoint of the cross-play grid (default: the last)")
    p.add_argument("--search", default="", help="curve: battler specs played on --search-subset states")
    p.add_argument("--search-subset", type=int, default=300)
    p.add_argument("--search-seeds", type=int, default=2)
    p.add_argument("--skip-b", action="store_true")
    p.add_argument("--skip-a", action="store_true")
    p.add_argument("--skip-c2", action="store_true")
    return p


def _rounds(spec):
    if "-" in spec:
        x, y = spec.split("-")
        return list(range(int(x), int(y) + 1))
    return [int(v) for v in spec.split(",")]


def main(argv=None):
    a = build_parser().parse_args(argv)
    from .search import in_training
    if in_training():
        raise RuntimeError("rl.agent_probe is a diagnosis tool: never inside a training process")
    if a.cmd == "generate":
        return generate(a)
    if a.cmd == "battler":
        s = run_battler(a.battler, a.procs, a.subset, a.seeds or a.b_seeds, a.device, a.width, a.bench)
        print(json.dumps({k: v for k, v in s.items() if k != "per_state"}, indent=1))
        return s
    if a.cmd == "tactician":
        kind, path, _ = parse_spec(a.tactician)
        return run_tactician(path, a.procs, a.bench)
    if a.cmd == "outcomes":
        specs = [a.tactician] if a.tactician else []
        return run_outcomes(specs, a.procs, a.c2_seeds, a.device, a.width, a.c2_top, a.bench)
    if a.cmd == "crossplay":
        s = run_crossplay(a.tactician, a.battler, a.a_runs, _rounds(a.rounds), a.procs, a.device, min(a.width, 128))
        print(json.dumps({k: v for k, v in s.items() if k != "per_run"}, indent=1, default=str))
        return s
    if a.cmd == "curve":
        return curve(a)
    if a.cmd == "report":
        return report(a)


if __name__ == "__main__":
    sys.exit(main() and 0)

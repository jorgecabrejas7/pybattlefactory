"""alphazero_v3 (docs/RL_DECISIONS.md §20): alphazero_v2 with a warmup-stable-decay learning rate, the attendant's
hints in the tactician's input, and the planner's per-option features, from scratch.

    python -m rl.alphazero_v3 --name alphazero_v3 --team-eval runs/team_eval_v3/model.pt
    tensorboard --logdir runs

A separate entry point on top of rl/alphazero_v2.py (itself on rl/alphazero.py), so `python -m rl.alphazero` and
`python -m rl.alphazero_v2` stay exactly as they were. What changes against v2 (everything else is v2: Gumbel
battler with 256 simulations, on-policy tactician PPO before the battler update, shaping, v_next table, W refresh
every 20 iterations once armed, evaluation network-only every iteration and with the 512-simulation search every 5):

  1. Learning rate "warmup-stable-decay" (WSD, class WSD): --lr-warmup iterations rising linearly to --lr, constant,
     then a decay (--lr-decay cosine | linear) to --lr * --lr-min-frac over the last --decay-frac of --iterations.
     The plan depends only on --iterations until the decay starts: resuming (--init-from) with a larger --iterations
     before the decay moves the decay to the new end. Once the decay has started, a resume keeps the checkpoint's
     plan (to lengthen a finished run, resume from a ckpt_<steps>.pt written before its decay). Battler and tactician
     share the optimizer and so the schedule.
  2. Encoding version 5 (rl/encode.py): v4 + the hints (next opponent's most common type, 19 one-hot with 18 = none;
     style, 8 one-hot) in the rental and swap context. The battle layout is v4's (the C++ encoder runs as v4).
  3. Per-option features from the v3 evaluator module (resolve_module): option_features_v3(ev, view, kind, ctx,
     options, v_next_table, hint, planner_cfg) -> [n_opt, N_OPT_FEATURES] (at least [E_round, W(normal | hint),
     W(last or Noland), Delta, Q_plan, Q_plan - mean]); the network's opt_feat width is N_OPT_FEATURES, read from
     the evaluator (attribute of the loaded evaluator if it has one, else of the module). Shaping potential
     phi_v3(...), imitation target pi_eval = softmax(q_eval_v3(features) / --t-eval-temp) behind --t-imitation
     (0 by default). planner_cfg: --planner-cfg (JSON, passed through; {} = the module's defaults).
  4. The tactician's PPO signal: --t-ppo-epochs (2) epochs, then extra epochs while the approximate KL to the
     behaviour policy over all the iteration's samples is below --t-kl-target (0.01; 0 = off), up to --t-max-epochs
     (4) in total; --t-round-weight alpha (0 = v2) weights each sample's policy and imitation terms by
     (1 - alpha) + alpha * hard_r / mean(hard), hard_r = 1 - P(complete round r) of the last network-only
     evaluation (the curriculum's difficulty), normalized to mean 1 over the iteration's samples.

Checkpoints: args["algo"] = "alphazero", args["version"] = 3, args["opt_feat"] = N_OPT_FEATURES,
args["encode_version"] = 5, args["team_eval_module"] = the resolved module; plus "lr_schedule" (the WSD plan).
rl.policy.Policy loads them; rl.eval_rounds adds their tactician features (features_hook) by default.
"""

import collections
import importlib
import json
import math
import os
import time

import numpy as np
import torch

from . import alphazero as AZ
from . import alphazero_v2 as V2
from . import encode
from . import team_eval_refresh as REF
from .model import FactoryNet
from .ppo import ValueNorm
from .tactician_features import V_NEXT_ROUNDS, load_evaluator, to_action_space

VERSION = 3
ENCODE_VERSION = 5
V3_API = ("N_OPT_FEATURES", "option_features_v3", "phi_v3", "q_eval_v3")
DEFAULT_MODULES = ("rl.team_eval_v3", "rl.team_eval")
STUB_MODULE = "rl.az3_stub"
T_KINDS = V2.T_KINDS


# ---- the learning rate: warmup-stable-decay ------------------------------------------------------------------------

class WSD:
    """lr(it): it < warmup: lr (it + 1) / (warmup + 1); constant lr until decay_start; then down to lr * min_frac at
    the last iteration (it = iterations - 1): frac = (it - decay_start + 1) / (iterations - decay_start), cosine
    lo + (lr - lo)(1 + cos(pi frac)) / 2 or linear lr - (lr - lo) frac. Past `iterations`: lo."""

    def __init__(self, lr, iterations, warmup=3, decay_frac=0.15, shape="cosine", min_frac=0.05, decay_start=None):
        if shape not in ("cosine", "linear"):
            raise ValueError(f"decay shape must be cosine or linear, not {shape!r}")
        self.base, self.iterations, self.warmup = float(lr), int(iterations), int(warmup)
        self.decay_frac, self.shape, self.min_frac = float(decay_frac), shape, float(min_frac)
        n_decay = int(round(self.decay_frac * self.iterations)) if self.decay_frac > 0 else 0
        self.decay_start = int(decay_start) if decay_start is not None else self.iterations - n_decay

    def lr(self, it):
        lo = self.base * self.min_frac
        if it >= self.decay_start:
            n = max(self.iterations - self.decay_start, 1)
            frac = min((it - self.decay_start + 1) / n, 1.0)
            if self.shape == "cosine":
                return lo + 0.5 * (self.base - lo) * (1 + math.cos(math.pi * frac))
            return self.base - (self.base - lo) * frac
        if it < self.warmup:
            return self.base * (it + 1) / (self.warmup + 1)
        return self.base

    def started(self, it0):
        """Did the decay start before iteration it0 (was iteration decay_start already trained)?"""
        return it0 > self.decay_start

    def state(self):
        return {"lr": self.base, "iterations": self.iterations, "warmup": self.warmup, "decay_frac": self.decay_frac,
                "shape": self.shape, "min_frac": self.min_frac, "decay_start": self.decay_start}

    @classmethod
    def from_state(cls, st):
        return cls(st["lr"], st["iterations"], st["warmup"], st["decay_frac"], st["shape"], st["min_frac"],
                   st["decay_start"])

    @classmethod
    def plan(cls, args, prev=None, it0=0):
        """The schedule for a run that trains iterations it0.. of args.iterations. prev: the checkpoint's
        state(). -> (WSD, note or None)."""
        new = cls(args.lr, args.iterations, args.lr_warmup, args.decay_frac, args.lr_decay, args.lr_min_frac)
        if prev:
            old = cls.from_state(prev)
            if old.started(it0):
                note = None
                if old.state() != new.state():
                    note = (f"the learning-rate decay started at iteration {old.decay_start} (before {it0}): keeping "
                            f"the checkpoint's plan (iterations {old.iterations}); to lengthen the run, resume from "
                            f"a checkpoint before its decay")
                return old, note
        if new.decay_start < it0:
            # a shorter plan whose decay would already be under way: start the decay now, over what remains
            new.decay_start = it0
            return new, f"decay moved to start at the resumed iteration {it0}"
        return new, None


# ---- the v3 evaluator module and its features ----------------------------------------------------------------------

def _has_api(mod):
    return all(hasattr(mod, k) for k in V3_API)


def resolve_module(name=None, path=None):
    """The module providing the v3 evaluator API: `name` if given; the stub for a "stub..." path; else the first of
    DEFAULT_MODULES that has it; the stub when there is no evaluator at all (only its width is used). -> name."""
    if name:
        if not _has_api(importlib.import_module(name)):
            raise ImportError(f"{name} lacks the alphazero_v3 evaluator API {V3_API}")
        return name
    if path is not None and str(path).startswith("stub"):
        return STUB_MODULE
    for n in DEFAULT_MODULES:
        try:
            if _has_api(importlib.import_module(n)):
                return n
        except ImportError:
            pass
    if path is None:
        return STUB_MODULE
    raise ImportError(f"no module with the alphazero_v3 evaluator API {V3_API} among {DEFAULT_MODULES}: pass "
                      f"--team-eval-module")


def n_opt_features(mod, ev=None):
    """N_OPT_FEATURES: the loaded evaluator's (from its checkpoint) if it has one, else the module's."""
    for k in ("N_OPT_FEATURES", "n_opt_features"):
        n = getattr(ev, k, None) if ev is not None else None
        if n:
            return int(n)
    return int(mod.N_OPT_FEATURES)


class TeamFeaturesV3:
    """rl.tactician_features.TeamFeatures for the v3 API: per-option features with the hint, the shaping potential
    and Q_eval. Picklable (the evaluator is loaded lazily in the process that uses it), so it is also the
    FactoryEnv obs_hook of the evaluation workers (__call__ adds "opt_feat").

    compute(backend, kind, obs, ctx) -> {"options", "opt_feat" (action space x n_feat), "phi", "q_eval", "feats",
    "hint"}; without an evaluator (path None): zeros."""

    def __init__(self, path, module, n_feat, device="cpu", v_next=None, shaping=True, planner_cfg=None):
        self.path, self.module, self.n_feat, self.device, self.shaping = path, module, int(n_feat), device, shaping
        self.planner_cfg = dict(planner_cfg or {})
        self.v_next = None if v_next is None else list(map(float, v_next))
        self._ev = self._mod = None
        self._pid = os.getpid()

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_ev"] = d["_mod"] = None
        return d

    @property
    def enabled(self):
        return self.path is not None

    def _load(self):
        if self._ev is None and self.enabled:
            if os.getpid() != self._pid:
                torch.set_num_threads(1)          # a forked process: see rl.tactician_features.TeamFeatures._load
            self._mod = importlib.import_module(self.module)
            self._ev = load_evaluator(self._mod, self.path, self.device)
            n = n_opt_features(self._mod, self._ev)
            if n != self.n_feat:
                raise ValueError(f"{self.path}: {n} option features, the network takes {self.n_feat}")
        return self._ev

    def set_v_next(self, table):
        self.v_next = None if table is None else list(map(float, table))

    def table(self):
        return np.asarray(self.v_next if self.v_next is not None else np.zeros(V_NEXT_ROUNDS), np.float64)

    def compute(self, backend, kind, obs, ctx):
        from .tactician_search import options_of
        options = options_of(kind, obs)
        view = backend.view()
        hint = (int(view.hint_type), int(view.hint_style))
        shape = ((6, 15) if kind == "rental" else (10,)) + (self.n_feat,)
        if not self.enabled:
            return {"options": options, "opt_feat": np.zeros(shape, np.float32), "phi": 0.0, "hint": hint,
                    "q_eval": np.zeros(len(options)), "feats": np.zeros((len(options), self.n_feat), np.float32)}
        ev = self._load()
        m, tab = self._mod, self.table()
        feats = np.asarray(m.option_features_v3(ev, view, kind, ctx, options, tab, hint, self.planner_cfg), np.float32)
        if feats.shape != (len(options), self.n_feat):
            raise ValueError(f"option_features_v3 returned {feats.shape}, expected {(len(options), self.n_feat)}")
        phi = float(m.phi_v3(ev, view, kind, ctx, tab, hint, self.planner_cfg)) if self.shaping else 0.0
        q = np.asarray(m.q_eval_v3(feats), np.float64).reshape(-1)
        return {"options": options, "opt_feat": to_action_space(kind, options, feats), "phi": phi, "q_eval": q,
                "feats": feats, "hint": hint}

    def __call__(self, backend, kind, obs, ctx):
        out = dict(obs)
        out["opt_feat"] = self.compute(backend, kind, obs, ctx)["opt_feat"]
        return out


def features_hook(ckpt, team_eval=None, device="cpu"):
    """The evaluation obs_hook of an alphazero_v3 checkpoint (path or loaded dict): its features with the W it used
    (team_eval overrides), its v_next table, no shaping. None for other checkpoints."""
    ck = torch.load(ckpt, map_location="cpu", weights_only=False) if isinstance(ckpt, str) else ckpt
    a = ck.get("args", {})
    if a.get("version") != VERSION or not a.get("opt_feat"):
        return None
    vn = (ck.get("v_next") or {}).get("table")
    return TeamFeaturesV3(team_eval or ck.get("team_eval") or a.get("team_eval"), a.get("team_eval_module"),
                          a["opt_feat"], device, v_next=vn, shaping=False, planner_cfg=a.get("planner_cfg"))


# ---- arguments ----------------------------------------------------------------------------------------------------

def build_parser():
    """v3's argument parser (rl.alphazero_v4 extends it)."""
    p = V2.build_parser()
    p.description = __doc__.split("\n")[0]
    p.set_defaults(name=time.strftime("alphazero_v3_%Y%m%d_%H%M%S"), iterations=600, encode_version=ENCODE_VERSION)
    for act in p._actions:
        if act.dest == "iterations":
            act.help = "planned iterations (WSD: the decay takes the last --decay-frac of them)"
        elif act.dest == "team_eval":
            act.help = "v3 team evaluator checkpoint (e.g. runs/team_eval_v3/model.pt; 'stub' / 'stub:varied': " \
                       "rl.az3_stub). None: no features (zeros), no shaping, no imitation"
        elif act.dest == "team_eval_module":
            act.help = f"module with the v3 evaluator API {V3_API} (default: the first of {DEFAULT_MODULES} " \
                       f"that has it)"
        elif act.dest == "lr_min_frac":
            act.help = "the decay's final learning rate, as a fraction of --lr"
    p.add_argument("--lr-warmup", type=int, default=3, help="WSD: iterations of linear warmup")
    p.add_argument("--decay-frac", type=float, default=0.15, help="WSD: fraction of --iterations that decays")
    p.add_argument("--lr-decay", choices=("cosine", "linear"), default="cosine", help="WSD: the decay's shape")
    p.add_argument("--planner-cfg", default="{}", help="JSON dict passed to the evaluator as planner_cfg")
    p.add_argument("--t-kl-target", type=float, default=0.01,
                   help="extra tactician PPO epochs while the approximate KL is below this (0: exactly "
                        "--t-ppo-epochs)")
    p.add_argument("--t-max-epochs", type=int, default=4, help="tactician PPO epochs at most (with --t-kl-target)")
    p.add_argument("--t-round-weight", type=float, default=0.0,
                   help="alpha: weight of the tactician's samples toward hard rounds (0: uniform, as v2)")
    return p


def finish_args(a):
    a.value_mix_t, a.min_batch_t = 0.0, 1
    a.planner_cfg = json.loads(a.planner_cfg) if isinstance(a.planner_cfg, str) else dict(a.planner_cfg or {})
    return a


def parse(argv=None):
    return finish_args(build_parser().parse_args(argv))


# ---- self-play ----------------------------------------------------------------------------------------------------

class SelfPlayerV3(V2.SelfPlayerV2):
    """alphazero_v2's self-play worker with the v3 features (width cfg["opt_feat"]) and the sample's round."""

    def _make_net(self, cfg):
        return FactoryNet(cfg["d_emb"], cfg["d"], cfg["layers"], cfg["heads"], share="embeddings",
                          opt_feat=cfg["opt_feat"]).eval()

    def _make_features(self, path, v_next=None):
        c = self.cfg
        return TeamFeaturesV3(path, c["team_eval_module"], c["opt_feat"], v_next=v_next, shaping=bool(c["t_shaping"]),
                              planner_cfg=c["planner_cfg"])

    def gen_team_data(self, n_battles, seed):
        """Simulated battles for a W refresh: the evaluator module's generate_battles if it has one, else v2's."""
        c = self.cfg
        gen = getattr(importlib.import_module(c["team_eval_module"]), "generate_battles", REF.generate_battles)
        return gen(self.evaluator, n_battles, seed, c["w_refresh_battles_per_team"], c["w_refresh_concurrency"],
                   c["max_decisions"])

    def _tactician(self, kind, obs):
        ctx = self.env._ctx()
        action = super()._tactician(kind, obs)
        t = self.traj[-1]
        t["round"] = int(ctx["challenge"]) + 1
        view = self.env.backend.view()
        self.stats["t_hint_none"] += float(int(view.hint_type) >= encode.N_HINT_TYPES - 1)
        return action


def worker_cfg(args, n_feat):
    return dict(V2.worker_cfg(args), opt_feat=int(n_feat), planner_cfg=dict(args.planner_cfg))


def round_weights(rounds, complete, alpha):
    """Per-sample weights toward hard rounds: (1 - alpha) + alpha hard_r / mean(hard), hard_r = 1 - P(complete
    round min(r, 6)) (1 for a round not evaluated yet), mean over these samples; all 1 when alpha = 0 or nothing is
    hard."""
    rounds = np.asarray(rounds)
    if alpha <= 0 or not len(rounds):
        return np.ones(len(rounds), np.float32)
    hard = np.array([1.0 - float(complete[min(int(r), 6)]) if complete and min(int(r), 6) in complete else 1.0
                     for r in rounds])
    hard = np.clip(hard, 0.0, 1.0)
    if hard.mean() <= 0:
        return np.ones(len(rounds), np.float32)
    return ((1 - alpha) + alpha * hard / hard.mean()).astype(np.float32)


def add_round_weights(tdata, complete, alpha):
    parts = [k for k in T_KINDS if tdata.get(k) is not None]
    if alpha <= 0 or not parts:
        return
    rounds = np.concatenate([tdata[k]["round"] for k in parts])
    w = round_weights(rounds, complete, alpha)
    i = 0
    for k in parts:
        n = len(tdata[k]["round"])
        tdata[k]["weight"] = w[i:i + n]
        i += n


def refresh_enabled(module, team_eval):
    return team_eval is not None and hasattr(importlib.import_module(module), "TeamNet")


# ---- main -----------------------------------------------------------------------------------------------------------

def main(argv=None, ext=None):
    """ext: an extension (rl.alphazero_v4.OpponentModelExt) with hooks parse, config_extra, load, worker_cfg,
    player, before_play, after_play, ckpt_extra, eval_search_kw; None = exactly alphazero_v3."""
    from .search import mark_training
    mark_training()     # no perfect-information search can exist in a training process (rl/search.py)
    args = ext.parse(argv) if ext is not None else parse(argv)
    if args.encode_version < 5:
        raise ValueError("alphazero_v3 needs --encode-version 5 (the tactician's hints)")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    run_dir = args.run_dir or os.path.join("runs", args.name)
    os.makedirs(run_dir, exist_ok=True)

    ck = torch.load(args.init_from, map_location=device, weights_only=False) if args.init_from else None
    resume = ck is not None and ck.get("args", {}).get("version") == VERSION
    w_path = ck["team_eval"] if resume and ck.get("team_eval") else args.team_eval     # the W in use
    args.team_eval_module = resolve_module(args.team_eval_module or (ck["args"].get("team_eval_module")
                                                                     if resume else None), w_path)
    mod = importlib.import_module(args.team_eval_module)
    n_feat = n_opt_features(mod, load_evaluator(mod, w_path, "cpu") if w_path is not None else None)
    if resume and ck["args"].get("opt_feat") != n_feat:
        raise ValueError(f"{args.init_from} has {ck['args'].get('opt_feat')} option features, the evaluator {n_feat}")

    config = dict(vars(args), algo=AZ.ALGO, version=VERSION, opt_feat=n_feat, share="embeddings", gamma_b=1.0,
                  beta=0.0)
    if ext is not None:
        config.update(ext.config_extra(args))
    json.dump(config, open(os.path.join(run_dir, "config.json"), "w"), indent=2)
    from torch.utils.tensorboard import SummaryWriter
    tb = SummaryWriter(run_dir)
    tb.add_text("config", "```\n" + json.dumps(config, indent=2) + "\n```")

    encode.set_version(args.encode_version)
    net = FactoryNet(args.d_emb, args.d, args.layers, args.heads, share="embeddings", opt_feat=n_feat).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=1e-5)
    norms = {a: ValueNorm(args.value_norm_rate) for a in ("battler", "tactician")}
    vnext = V2.VNext(args.v_next_rate)
    schedule = REF.RefreshSchedule(REF.parse_reference(args.w_refresh_after), args.w_refresh_every,
                                   args.w_refresh_window)
    if schedule.enabled and not refresh_enabled(args.team_eval_module, args.team_eval):
        print("W refresh disabled: it needs --team-eval with a trainable evaluator (TeamNet)", flush=True)
        schedule.every = 0
    real_parts = collections.deque(maxlen=max(args.w_refresh_real_iters, 1))
    probe = None
    it0, b_steps, t_steps = 0, 0, 0
    round_eval = {}
    lr_prev = None
    if ck is not None:
        net.load_state_dict(ck["net"])
        if resume:
            if "opt" in ck:
                opt.load_state_dict(ck["opt"])
            for a, st in (ck.get("value_norm") or {}).items():
                if st and a in norms:
                    norms[a].load(st)
            if ck.get("v_next"):
                vnext.load(ck["v_next"])
            if ck.get("w_refresh"):
                schedule.load(ck["w_refresh"])
            it0 = ck.get("iteration", -1) + 1
            b_steps, t_steps = ck.get("battler_steps", 0), ck.get("tactician_steps", 0)
            round_eval = {int(k): v for k, v in (ck.get("round_eval") or {}).items()}
            lr_prev = ck.get("lr_schedule")
            print(f"resumed from {args.init_from}: iteration {it0}, {b_steps:,} battler steps", flush=True)
        else:
            print(f"initialized the network from {args.init_from} (version {ck.get('args', {}).get('version')}); "
                  f"a new run from iteration 0", flush=True)
    if ext is not None:
        ext.load(ck if resume else None, args, run_dir)
    del ck
    lr_sched, note = WSD.plan(args, lr_prev, it0)
    if note:
        print("learning rate: " + note, flush=True)
    config["lr_schedule"] = lr_sched.state()
    tb.add_scalar("model/parameters", sum(p.numel() for p in net.parameters()), b_steps)
    net.eval()
    snap_path = os.path.join(run_dir, "snapshot.pt")

    def ckpt_dict(it):
        d = {"net": net.state_dict(), "opt": opt.state_dict(), "args": config, "battler_steps": b_steps,
             "tactician_steps": t_steps, "iteration": it, "round_eval": round_eval, "v_next": vnext.state(),
             "team_eval": w_path, "w_refresh": schedule.state(), "lr_schedule": lr_sched.state(),
             "value_norm": {k: (v.state() if v.seen else None) for k, v in norms.items()}}
        if ext is not None:
            d.update(ext.ckpt_extra())
        return d

    def write_snapshot():
        AZ.save_atomic({"net": {k: v.detach().cpu() for k, v in net.state_dict().items()}, "args": config,
                        "value_norm": {k: (v.state() if v.seen else None) for k, v in norms.items()},
                        "v_next": vnext.state(), "team_eval": w_path}, snap_path)

    write_snapshot()
    server = None
    eval_slots = args.eval_search_procs if args.eval_search_every > 0 and args.eval_search_sims > 0 else 0
    if args.inference == "gpu":
        from .inference import InferenceServer
        server = InferenceServer(snap_path, n_clients=args.workers + eval_slots)
    cfg_w = worker_cfg(args, n_feat)
    if ext is not None:
        cfg_w = ext.worker_cfg(cfg_w)
    pool = AZ.WorkerPool(args.workers, cfg_w, server, args.seed,
                         player=ext.player if ext is not None else SelfPlayerV3)
    if w_path != args.team_eval:                      # resumed after a refresh
        pool.broadcast("call", ("set_team_eval", (w_path,)))
    replay = AZ.Replay(args.window)
    rng = np.random.default_rng(args.seed + 17)
    t_start = time.time()
    try:
        for it in range(it0, args.iterations):
            t_it = time.time()
            lr = lr_sched.lr(it)
            for g in opt.param_groups:
                g["lr"] = lr
            weights = AZ.start_weights_from_eval(round_eval, args.start_p0)
            pool.broadcast("load", snap_path)
            pool.broadcast("weights", weights.tolist())
            pool.broadcast("call", ("set_v_next", (vnext.table.tolist(),)))
            if ext is not None:
                ext.before_play(pool, it)

            # ---- 1. self-play ---------------------------------------------------------------------------------
            per_worker = math.ceil(args.decisions_per_iter / args.workers)
            srv0 = server.stats() if server is not None else None
            t0 = time.time()
            results = pool.broadcast("play", per_worker)
            t_play = time.time() - t0
            stats = [s for _, s in results]
            made = sum(s["battler_decisions"] for s in stats)
            if server is not None:
                srv1 = server.stats()
                nb = max(srv1["batches"] - srv0["batches"], 1)
                tb.add_scalar("perf/server_mean_batch", (srv1["samples"] - srv0["samples"]) / nb, b_steps + made)
                tb.add_scalar("perf/server_forward_ms", (srv1["forward_ms"] * srv1["batches"]
                                                         - srv0["forward_ms"] * srv0["batches"]) / nb, b_steps + made)
            bdata = AZ.merge([{"battle": r["battle"]} for r, _ in results])
            tdata = V2.merge_t([r for r, _ in results])
            v_samples = [x for r, _ in results for x in r["v_next"]]
            real_parts.append(REF.merge_arrays([r.get("real") for r, _ in results]))
            new_counts = {"battle": int((~bdata["battle"]["held"]).sum()) if bdata["battle"] is not None else 0}
            b_steps += made
            t_steps += int(sum(s.get("t_decisions", 0) for s in stats))
            replay.add(bdata)
            vcount = vnext.update(v_samples)
            V2.log_selfplay_v2(tb, stats, bdata, tdata, made, t_play, weights, b_steps, vnext, vcount)
            nt = max(sum(s.get("t_decisions", 0) for s in stats), 1)
            tb.add_scalar("tactician/hint_none_frac", sum(s.get("t_hint_none", 0) for s in stats) / nt, b_steps)
            if ext is not None:
                ext.after_play(results, tb, b_steps, it)

            # ---- 2. training: the tactician's PPO first (docs §19.7), then the battler ----------------------------
            t0 = time.time()
            add_round_weights(tdata, round_eval, args.t_round_weight)
            tr = V2.tactician_ppo(net, opt, tdata, norms["tactician"], args, device, rng)
            tr.update(V2.tactician_heldout(net, tdata, norms["tactician"], device))
            tr.update(AZ.train_iteration(net, opt, replay, new_counts, norms, args, device, rng))
            tr.update(AZ.heldout_stats(net, replay, norms, args, device))
            t_train = time.time() - t0
            for k, v in tr.items():
                tb.add_scalar(k.replace("battle/", "battler/"), v, b_steps)
            tb.add_scalar("train/lr", lr, b_steps)
            tb.add_scalar("train/iteration", it, b_steps)
            for a, nm in norms.items():
                tb.add_scalar(f"{a}/value_norm_mean", nm.mean, b_steps)
                tb.add_scalar(f"{a}/value_norm_std", nm.std, b_steps)

            # ---- 3. checkpoint, new snapshot, evaluation -----------------------------------------------------
            write_snapshot()
            if server is not None:
                server.reload(snap_path)
            AZ.save_atomic(ckpt_dict(it), os.path.join(run_dir, f"ckpt_{b_steps:011d}.pt"))
            t0 = time.time()
            if args.eval_every and (it + 1) % args.eval_every == 0 and args.eval_runs > 0:
                with_search = bool(args.eval_search_every and (it + 1) % args.eval_search_every == 0
                                   and args.eval_search_sims > 0 and server is not None)
                hook = TeamFeaturesV3(w_path, args.team_eval_module, n_feat, v_next=vnext.table.tolist(),
                                      shaping=False, planner_cfg=args.planner_cfg)
                round_eval = V2.evaluate_rounds_v2(net, device, args, tb, b_steps, server, snap_path,
                                                   vnext.table.tolist(), with_search, w_path, hook=hook,
                                                   search_kw=ext.eval_search_kw() if ext is not None else None)
                schedule.observe(it, round_eval)
            t_eval = time.time() - t0
            # ---- 4. W refresh (self-play is paused between iterations) -----------------------------------------
            if schedule.enabled:
                tb.add_scalar("w_refresh/trigger_mean_complete", schedule.running_mean(), b_steps)
                tb.add_scalar("w_refresh/armed", float(schedule.armed_it is not None), b_steps)
            if schedule.due(it):
                if probe is None:
                    probe = getattr(mod, "probe_teams", REF.probe_teams)(args.w_refresh_probe, seed=12345)
                new_path = os.path.join(run_dir, f"team_eval_{it:04d}.pt")
                m = V2.run_refresh(pool, args.workers, w_path, new_path, real_parts, args, it, device, probe, tb,
                                   b_steps)
                w_path = new_path
                schedule.done(it)
                tb.add_scalar("w_refresh/count", schedule.count, b_steps)
                print(f"[it {it:4d}] W refreshed -> {new_path}: {m['battles']:,} battles, brier "
                      f"{m['brier_before']:.4f} -> {m['brier_after']:.4f}, probe |dW| "
                      f"{m.get('probe_abs_dw', float('nan')):.4f} ({m['seconds']:.0f} s)", flush=True)
            AZ.save_atomic(ckpt_dict(it), os.path.join(run_dir, "latest.pt"))
            tb.add_scalar("perf/selfplay_seconds", t_play, b_steps)
            tb.add_scalar("perf/train_seconds", t_train, b_steps)
            tb.add_scalar("perf/eval_seconds", t_eval, b_steps)
            tb.add_scalar("perf/iteration_seconds", time.time() - t_it, b_steps)
            tb.add_scalar("perf/hours", (time.time() - t_start) / 3600, b_steps)
            if torch.cuda.is_available() and device.type == "cuda":
                tb.add_scalar("perf/gpu_mem_peak_mb", torch.cuda.max_memory_allocated() / 2 ** 20, b_steps)
            tb.flush()
            won = sum(s.get("battles_won", 0) for s in stats) / max(sum(s.get("battles", 0) for s in stats), 1)
            print(f"[it {it:4d} | {(time.time() - t_start) / 60:7.1f} min] {b_steps:,} battler steps  "
                  f"{made / max(t_play, 1e-9):6.0f} dec/s  lr {lr:.2e}  win {won:.3f}  "
                  f"CE b {tr.get('battle/policy_loss', float('nan')):.3f}  "
                  f"EV b {tr.get('battle/explained_variance_train', float('nan')):.2f}/"
                  f"{tr.get('battle/explained_variance_heldout', float('nan')):.2f}  "
                  f"t kl {tr.get('tactician/approx_kl', float('nan')):.4f} "
                  f"ep {tr.get('tactician/epochs', args.t_ppo_epochs):.0f} "
                  f"EV t {tr.get('tactician/explained_variance_train', float('nan')):.2f}  "
                  f"(play {t_play:.0f}s train {t_train:.0f}s eval {t_eval:.0f}s)"
                  + ("  rounds " + " ".join(f"{round_eval[k]:.2f}" for k in sorted(round_eval)) if round_eval else ""),
                  flush=True)
    finally:
        pool.close()
        if server is not None:
            server.close()
        tb.close()
    return run_dir


if __name__ == "__main__":
    main()

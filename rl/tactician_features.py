"""The team evaluator's per-option features, the shaping potential and the expert policy of the alphazero_v2
tactician (docs/RL_DECISIONS.md §19.2-19.3).

    tf = TeamFeatures("runs/team_eval/latest.pt")      # None: no evaluator (features 0, potential 0)
    tf.set_v_next(table)                               # table[r]: mean real wins from the start of round r
    out = tf.compute(backend, kind, obs, ctx)          # at a RENTAL / SWAP decision of a SimBackend
    out["opt_feat"]   [6, 15, 4] (rental, by (lead, pair)) / [10, 4] (swap): the network's extra input
    out["phi"]        Phi(s) = E_round + P(complete) * v_next of the current state (shaping potential)
    out["q_eval"]     Q_eval of each legal option (Phi of the state it leads to), in options_of order
    out["options"]    options_of(kind, obs)

The options are enumerated exactly as the policy's action space (rl.tactician_search.options_of): a rental option is
a (lead, pair) of the factorized rental head, so its 4 features go to cell [lead, pair] of opt_feat (masked cells
stay 0); a swap option is its action index 0..9.

Q_eval from the features: with k = ctx["battle"] + 1 the next battle of the round and W_n, W_l the option's
W(normal) and W(last or Noland), P(complete) = W_n^(7 - k) W_l, so Q_eval = E_round + W_n^(7 - k) W_l v_next.
v_next = table[round + 1], round = ctx["challenge"] + 1.

TeamFeatures is picklable (module name + path; the evaluator is loaded lazily in the process that uses it), so it
is also the FactoryEnv obs_hook of the evaluation workers (__call__ adds "opt_feat" to the observation).
"""

import importlib
import inspect

import numpy as np

N_FEAT = 4
V_NEXT_ROUNDS = 12                     # table[r] for r = 0..11 (rounds past it use the last entry)


def team_eval_module(name=None):
    """rl.team_eval when it exists (the real evaluator), else the stub; or the module `name`."""
    if name:
        return importlib.import_module(name)
    try:
        return importlib.import_module("rl.team_eval")
    except ImportError:
        return importlib.import_module("rl.team_eval_stub")


def load_evaluator(mod, path, device="cpu"):
    """TeamEvaluator.load(path, device) as a classmethod / staticmethod returning the evaluator, or as an instance
    method loading into a new instance."""
    cls = mod.TeamEvaluator
    ld = inspect.getattr_static(cls, "load")
    if isinstance(ld, (classmethod, staticmethod)):
        ev = cls.load(path, device)
        if ev is not None:
            return ev
    ev = cls()
    out = ev.load(path, device)
    return out if isinstance(out, cls) else ev


def v_next_of(table, ctx):
    if table is None:
        return 0.0
    r = int(ctx["challenge"]) + 2                      # the next round, 1-based
    return float(table[min(r, len(table) - 1)])


def q_eval_from_features(feat, ctx, v_next):
    """Phi of the state each option leads to: E_round + W_n^(7 - k) W_l v_next."""
    feat = np.asarray(feat, np.float64)
    if feat.size == 0:
        return np.zeros(0)
    k = int(ctx["battle"]) + 1
    p_complete = feat[:, 1] ** max(7 - k, 0) * feat[:, 2]
    return feat[:, 0] + p_complete * v_next


def softmax(z, temp):
    z = np.asarray(z, np.float64) / max(temp, 1e-6)
    z = np.exp(z - z.max())
    return z / z.sum()


def to_action_space(kind, options, values, fill=0.0, width=None):
    """Per-option values -> the action space: rental [6, 15, ...], swap [10, ...]."""
    values = np.asarray(values, np.float32)
    tail = values.shape[1:] if width is None else (width,)
    out = np.full(((6, 15) if kind == "rental" else (10,)) + tuple(tail), fill, np.float32)
    for o, v in zip(options, values):
        out[tuple(o) if kind == "rental" else o] = v
    return out


class TeamFeatures:
    def __init__(self, path=None, module=None, device="cpu", v_next=None, shaping=True):
        self.path, self.module, self.device, self.shaping = path, module, device, shaping
        self.v_next = None if v_next is None else list(map(float, v_next))
        self._ev = self._mod = None

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_ev"] = d["_mod"] = None
        return d

    @property
    def enabled(self):
        return self.path is not None

    def _load(self):
        if self._ev is None and self.enabled:
            self._mod = team_eval_module(self.module)
            self._ev = load_evaluator(self._mod, self.path, self.device)
        return self._ev

    def set_v_next(self, table):
        self.v_next = None if table is None else list(map(float, table))

    def compute(self, backend, kind, obs, ctx):
        from .tactician_search import options_of
        options = options_of(kind, obs)
        shape = ((6, 15) if kind == "rental" else (10,)) + (N_FEAT,)
        if not self.enabled:
            return {"options": options, "opt_feat": np.zeros(shape, np.float32), "phi": 0.0,
                    "q_eval": np.zeros(len(options)), "feats": np.zeros((len(options), N_FEAT), np.float32)}
        ev = self._load()
        view = backend.view()
        vn = v_next_of(self.v_next, ctx)
        feats = np.asarray(self._mod.option_features(ev, view, kind, ctx, options, vn), np.float32)
        if feats.shape != (len(options), N_FEAT):
            raise ValueError(f"option_features returned {feats.shape}, expected {(len(options), N_FEAT)}")
        phi = float(self._mod.phi(ev, view, kind, ctx, vn)) if self.shaping else 0.0
        return {"options": options, "opt_feat": to_action_space(kind, options, feats), "phi": phi,
                "q_eval": q_eval_from_features(feats, ctx, vn), "feats": feats}

    def __call__(self, backend, kind, obs, ctx):
        """FactoryEnv obs_hook: the observation with "opt_feat"."""
        out = dict(obs)
        out["opt_feat"] = self.compute(backend, kind, obs, ctx)["opt_feat"]
        return out

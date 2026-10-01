"""Raw play logger: one JSON line per decision / battle end / run end, shared by the emulator (real game) and the
simulator replica (rl/play_sim.py). docs/RL_DECISIONS.md §23.

    log = PlayLogger("runs/playlog_x/net_0.jsonl.gz", meta={"ckpt": ..., "battler": "net", "source": "sim"})
    log.decision(kind, view, run_info, policy, action, extras)   # before the action is given to the game
    log.battle_end(won, info)
    log.run_end(streak, info)
    log.close()

    for rec in read_log(path): ...                               # the records back, as dicts
    for run in iter_runs(paths): ...                             # grouped: run -> battles -> decisions

Interface (fixed: the emulator runner codes against it):
    kind      "rental" | "swap" | "battle" | "forced_switch"
    view      the player's view object (pybattle.view RentalView / SwapView / BattleView: dataclasses)
    run_info  pybattle.view.RunInfo (or a dict with its fields)
    policy    {"probs": [...], "value": float, "legal": [...]} (+ any other key: e.g. "lead_probs", "trio_probs")
    action    the backend action (rental (lead, j, k); swap None | (own slot, enemy slot); battle ("move", i) /
              ("switch", i)); stored as JSON lists, None for "keep"
    extras    anything (W features per option, Q_plan, hint, probes, search visits / Q, opponent-model beliefs,
              battle events...): whatever is JSON-able is kept (dataclasses -> dicts, numpy -> lists, sets -> sorted
              lists, tuples -> lists, NaN / inf -> None); anything else becomes {"__repr__": repr(x)}.

Player view vs truth. Every record keeps what the player could see (the view, run_info, the agent's own policy and
extras) in its normal fields. Information the player does NOT have (the opponent's true team, its exact HP, the true
configuration of a slot...) is only allowed under the key "truth": a top-level "truth" key of `extras` (decision) or
`info` (battle_end / run_end) is moved to the record's own "truth" field, which every analysis that describes the
player's behaviour must ignore (rl/playstats.py only reads it in sections flagged "análisis con información
verdadera"). The meta line lists both groups of fields (meta["player_view_fields"], meta["truth_fields"]).

Records (one per line; gzip when the path ends in .gz):
    {"t": "meta", "version": 1, "meta": {...}, "player_view_fields": [...], "truth_fields": ["truth"]}
    {"t": "decision", "i": seq, "run": r, "battle": b, "kind": ..., "run_info": {...}, "view": {...},
     "policy": {...}, "action": ..., "extras": {...}, ["truth": {...}]}
    {"t": "battle_end", "i": seq, "run": r, "battle": b, "won": bool, "info": {...}, ["truth": {...}]}
    {"t": "run_end", "i": seq, "run": r, "streak": int, "battles": n, "info": {...}, ["truth": {...}]}
"run" counts the runs of this file (0, 1, ...; a new run starts after each run_end) and "battle" the battles of the
run (the battle a decision belongs to, or that the tactician decision precedes).
"""

import dataclasses
import enum
import gzip
import json
import math
import os
import time

import numpy as np

LOG_VERSION = 1
TRUTH_KEY = "truth"
PLAYER_VIEW_FIELDS = ("kind", "run_info", "view", "policy", "action", "extras", "won", "info", "streak")
TRUTH_FIELDS = (TRUTH_KEY,)
KINDS = ("rental", "swap", "battle", "forced_switch")


def to_jsonable(x, _depth=0):
    """A JSON-able copy of x: dataclasses -> dicts, numpy -> lists / numbers, sets -> sorted lists, tuples -> lists,
    enums -> their value, NaN / inf -> None; anything else -> {"__repr__": repr(x)}."""
    if _depth > 50:
        return {"__repr__": "<too deep>"}
    if x is None or isinstance(x, (bool, str)):
        return x
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        f = float(x)
        return f if math.isfinite(f) else None
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return to_jsonable(x.tolist(), _depth + 1)
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return {f.name: to_jsonable(getattr(x, f.name), _depth + 1) for f in dataclasses.fields(x)}
    if isinstance(x, enum.Enum):
        return to_jsonable(x.value, _depth + 1)
    if isinstance(x, dict):
        return {str(k): to_jsonable(v, _depth + 1) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v, _depth + 1) for v in x]
    if isinstance(x, (set, frozenset)):
        items = [to_jsonable(v, _depth + 1) for v in x]
        try:
            return sorted(items)
        except TypeError:
            return items
    if hasattr(x, "__dict__") and type(x).__module__ == "types":      # SimpleNamespace
        return to_jsonable(vars(x), _depth + 1)
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return to_jsonable(x.detach().cpu().numpy(), _depth + 1)
    except ImportError:                                                # pragma: no cover
        pass
    return {"__repr__": repr(x)}


def action_to_json(action):
    """The backend action as JSON: None (keep) stays None; tuples -> lists."""
    return to_jsonable(action)


def _split_truth(d):
    """(d without its "truth" key, the truth or None). d may be None."""
    if not d:
        return {}, None
    d = dict(d)
    truth = d.pop(TRUTH_KEY, None)
    return d, truth


class PlayLogger:
    """Appends JSON lines to `path` (gzip when it ends in .gz). Safe to call from one thread; flushes on close and
    every `flush_every` records (a crashed run keeps what was flushed)."""

    def __init__(self, path, meta: dict):
        d = os.path.dirname(os.path.abspath(path))
        os.makedirs(d, exist_ok=True)
        self.path = path
        self._f = gzip.open(path, "wt", compresslevel=6) if path.endswith(".gz") else open(path, "w")
        self.i = 0
        self.run = 0
        self.battle = 0
        self.flush_every = 200
        self._write({"t": "meta", "version": LOG_VERSION, "time": time.time(), "meta": to_jsonable(meta or {}),
                     "player_view_fields": list(PLAYER_VIEW_FIELDS), "truth_fields": list(TRUTH_FIELDS)})

    def _write(self, rec):
        self._f.write(json.dumps(rec, separators=(",", ":"), allow_nan=False))
        self._f.write("\n")
        self.i += 1
        if self.i % self.flush_every == 0:
            self._f.flush()

    def decision(self, kind: str, view, run_info, policy: dict, action, extras: dict) -> None:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, not {kind!r}")
        extras, truth = _split_truth(extras)
        rec = {"t": "decision", "i": self.i, "run": self.run, "battle": self.battle, "kind": kind,
               "run_info": to_jsonable(run_info), "view": to_jsonable(view), "policy": to_jsonable(policy or {}),
               "action": action_to_json(action), "extras": to_jsonable(extras)}
        if truth is not None:
            rec[TRUTH_KEY] = to_jsonable(truth)
        self._write(rec)

    def battle_end(self, won: bool, info: dict) -> None:
        info, truth = _split_truth(info)
        rec = {"t": "battle_end", "i": self.i, "run": self.run, "battle": self.battle, "won": bool(won),
               "info": to_jsonable(info)}
        if truth is not None:
            rec[TRUTH_KEY] = to_jsonable(truth)
        self._write(rec)
        self.battle += 1

    def run_end(self, streak: int, info: dict) -> None:
        info, truth = _split_truth(info)
        rec = {"t": "run_end", "i": self.i, "run": self.run, "streak": int(streak), "battles": self.battle,
               "info": to_jsonable(info)}
        if truth is not None:
            rec[TRUTH_KEY] = to_jsonable(truth)
        self._write(rec)
        self.run += 1
        self.battle = 0

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ---- reading --------------------------------------------------------------------------------------------------------

def read_log(path):
    """Yield the records of one log file (a truncated gzip tail, e.g. a killed process, ends the iteration)."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        try:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    return
        except (EOFError, OSError):
            return


def strip_truth(rec):
    """The record as the player saw it (no "truth" field)."""
    return {k: v for k, v in rec.items() if k != TRUTH_KEY}


def iter_runs(paths, complete_only=True):
    """Group the records of each file into runs:
        {"file", "meta", "run", "battles": [{"decisions": [...], "end": battle_end record or None,
         "pre": [tactician decisions before it]}], "end": run_end record}
    A battle's "pre" holds the rental / swap decisions just before it; its "decisions" the battle / forced_switch
    ones. complete_only: skip a run without its run_end (the tail of an interrupted file)."""
    if isinstance(paths, str):
        paths = [paths]
    for path in paths:
        meta = None
        cur = None
        for rec in read_log(path):
            t = rec.get("t")
            if t == "meta":
                meta = rec
                continue
            if cur is None or rec["run"] != cur["run"]:
                if cur is not None and not complete_only:
                    yield cur
                cur = {"file": path, "meta": meta, "run": rec["run"], "battles": [], "end": None}
            b = rec.get("battle", 0)
            while len(cur["battles"]) <= b:
                cur["battles"].append({"pre": [], "decisions": [], "end": None})
            if t == "decision":
                (cur["battles"][b]["pre"] if rec["kind"] in ("rental", "swap") else
                 cur["battles"][b]["decisions"]).append(rec)
            elif t == "battle_end":
                cur["battles"][b]["end"] = rec
            elif t == "run_end":
                # the last "battle" slot may hold only a trailing tactician decision (none in practice)
                cur["battles"] = [x for x in cur["battles"] if x["end"] is not None or x["decisions"]]
                cur["end"] = rec
                yield cur
                cur = None
        if cur is not None and not complete_only:
            yield cur

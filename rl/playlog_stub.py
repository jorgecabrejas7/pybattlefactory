"""Stand-in for rl.playlog.PlayLogger until it is merged (same interface; play_real uses the real one when it
exists). Writes one JSON line per event to `path`.

    log = PlayLogger(path, meta)
    log.decision(kind, view, run_info, policy, action, extras)   # policy: {"probs": {...}, "value": ...}
    log.battle_end(won, info)
    log.run_end(streak, info)
    log.close()
"""

import dataclasses
import json
import time


def _plain(x, depth=0):
    """JSON-friendly copy (dataclasses, numpy, tuples); views become dicts."""
    if depth > 8:
        return repr(x)
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return {f.name: _plain(getattr(x, f.name), depth + 1) for f in dataclasses.fields(x)}
    if isinstance(x, dict):
        return {str(k): _plain(v, depth + 1) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v, depth + 1) for v in x]
    if hasattr(x, "tolist"):
        return x.tolist()
    if hasattr(x, "value") and hasattr(x, "name"):          # enums
        return x.value
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if hasattr(x, "__dict__"):
        return {k: _plain(v, depth + 1) for k, v in vars(x).items() if not k.startswith("_")}
    return repr(x)


class PlayLogger:
    def __init__(self, path, meta=None):
        self.path = path
        self.f = open(path, "a")
        self._write({"event": "meta", "meta": meta or {}})

    def _write(self, rec):
        rec.setdefault("time", time.time())
        self.f.write(json.dumps(_plain(rec)) + "\n")
        self.f.flush()

    def decision(self, kind, view, run_info, policy, action, extras=None):
        self._write({"event": "decision", "kind": kind, "view": view, "run_info": run_info, "policy": policy,
                     "action": action, "extras": extras or {}})

    def battle_end(self, won, info=None):
        self._write({"event": "battle_end", "won": won, "info": info or {}})

    def run_end(self, streak, info=None):
        self._write({"event": "run_end", "streak": streak, "info": info or {}})

    def close(self):
        if self.f:
            self.f.close()
            self.f = None


def play_logger_class():
    """rl.playlog.PlayLogger once it exists, else this stub."""
    try:
        from .playlog import PlayLogger as Real
        return Real
    except ImportError:
        return PlayLogger

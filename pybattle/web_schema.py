"""The JSON the browser build's FactorySession (src/web/factory_session.cpp) writes for views, from the Python
views: RentalView / SwapView / BattleView / RunInfo as plain dicts with the dataclass field names (tuples become
lists). tests/python/test_web_session.py checks both sides write the same thing."""

import dataclasses


def _plain(x):
    if dataclasses.is_dataclass(x):
        return {f.name: _plain(getattr(x, f.name)) for f in dataclasses.fields(x)}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, bool) or x is None:
        return x
    if isinstance(x, int):
        return int(x)
    return x


def view_json(view):
    """A view (or RunInfo) as the web build writes it; None stays None (run over)."""
    return None if view is None else _plain(view)

"""The gen3 core must reproduce real-game battles byte for byte.

Fixtures are traces recorded from the unmodified ROM on mGBA (scripts/record_traces.py):
both parties, the RNG seed at battle start and the player's actions. Replaying them must
reproduce every recorded snapshot -- gRngValue, battle RAM, both parties -- and the outcome.
"""

import glob
import gzip
import json
import os

import pytest

from pybattle.diff import replay_trace

FIXTURES = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "diff", "fixtures", "*.json.gz")))


@pytest.mark.parametrize("path", FIXTURES, ids=[os.path.basename(p) for p in FIXTURES])
def test_replay_matches_real_game(path):
    with gzip.open(path, "rt") as f:
        trace = json.load(f)
    result = replay_trace(trace)
    assert result.ok, "\n".join(f"decision {m.decision}: {m.what} {m.expected} | {m.got}" for m in result.mismatches[:5])
    assert result.decisions_matched == result.total_decisions


def test_clone_is_independent():
    with gzip.open(FIXTURES[0], "rt") as f:
        trace = json.load(f)
    from pybattle.diff import setup_battle
    a = setup_battle(trace["start"])
    a.run()
    b = a.clone()
    a.choose_move(trace["decisions"][0]["action"].get("move", 0))
    a.run()
    # b is still at the first decision, with the pre-turn RNG
    assert b.rng == trace["decisions"][0]["state"]["rng"]
    assert a.rng != b.rng

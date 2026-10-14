"""The browser's resumable search (FactorySession.search_begin / prepare / submit, run one network round at a time)
grows the same trees as the native runSearch on the same roots and evaluator; its roots are SearchBattler._roots'
(determinized with player-knowledge specs, redrawn when what the player can do would change)."""

import random
import struct

import numpy as np
import pytest

from pybattle.backend import Phase, SimBackend
from pybattle.pybattle_native import FactorySession
from rl import determinize as DET

PHASES = [Phase.RENTAL, Phase.SWAP, Phase.BATTLE, Phase.FORCED_SWITCH, Phase.RUN_OVER]


def _pack(specs_list):
    out = []
    for specs in specs_list:
        by_slot = {s[0]: s for s in specs}
        for slot in range(3):
            s = by_slot.get(slot, (slot, 0, [0] * 4, 0, [0] * 6, [0] * 6, 0, 0, 1.0))
            ps, sp, mv, it, iv, ev, nat, ab, hp = s
            out += [ps, sp, *mv, it, *iv, *ev, nat, ab, struct.unpack("<i", struct.pack("<f", hp))[0]]
    return np.asarray(out, np.int32)


def _evaluator(batch):
    """Deterministic stand-in for the network: priors and values from the observation's numbers."""
    n = len(batch["mask"])
    s = batch["mon_num"].reshape(n, -1).sum(1) + batch["ctx_num"].sum(1) * 0.37 + batch["move_num"].reshape(n, -1).sum(1) * 0.11
    logits = np.sin(s[:, None] * np.arange(1, 8)[None, :] * 0.013) * 3
    logits = np.where(batch["mask"], logits, -1e9)
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    v = 1 / (1 + np.exp(-np.cos(s * 0.007) * 2))
    return p.astype(np.float32), v.astype(np.float32)


def _to_battle(seed):
    rng = random.Random(seed)
    sim, fs = SimBackend(max_turns=150), FactorySession(True, 150)
    sim.reset(seed, 0)
    fs.reset(seed, 0, 0)
    turns = rng.randrange(1, 8)
    while True:
        ph = sim.phase
        if ph == Phase.RUN_OVER:
            return None
        if ph == Phase.RENTAL:
            sim.act((0, 1, 2)); fs.act_rental(0, 1, 2)
        elif ph == Phase.SWAP:
            sim.act(None); fs.act_swap(-1, 0)
        else:
            view = sim.view()
            legal = [a for a in view.legal_actions if a[0] != "forfeit"]
            if ph == Phase.BATTLE and len(legal) > 1:
                turns -= 1
                if turns == 0:
                    return sim, fs, rng
            a = rng.choice(legal)
            sim.act(a)
            fs.act_move(a[1]) if a[0] == "move" else fs.act_switch(a[1])


@pytest.mark.parametrize("seed", range(12))
def test_resumable_search_equals_native(seed):
    got = _to_battle(seed)
    if got is None:
        pytest.skip("run over before a battle decision")
    sim, fs, rng = got
    view = sim.view()
    K, R = 3, 3
    det_rng = random.Random(seed + 1000)
    attempts = [DET.determinization(view, None, det_rng, True, "strict") for _ in range(K * R)]
    att = _pack(attempts)
    hid = np.asarray(DET.hidden_counters(view), np.int32)
    seeds = [det_rng.getrandbits(32) for _ in range(2 * K)]
    root_p, root_v = _evaluator({k: np.asarray(v)[None] for k, v in fs.encode_battle().items()})
    args = (att, R, hid, seeds, list(root_p[0]), float(root_v[0]), 96, 16, 1.5)

    before = fs.view_json(), fs.encode_battle()["mon_num"].copy()
    native = fs.search_run_native(*args, _evaluator)
    fs.search_begin(*args)
    rounds = 0
    while (n := fs.search_prepare()) >= 0:
        obs = fs.search_observations(n)
        p, v = _evaluator(obs) if n else (np.zeros((0, 7), np.float32), np.zeros(0, np.float32))
        fs.search_submit(p, v)
        rounds += 1
    step = fs.search_stats()
    fs.search_end()
    assert rounds > 1
    assert step["visits"] == native["visits"]
    assert np.allclose(step["q"], native["q"], atol=1e-9)
    assert (step["leaves"], step["nodes"], step["errors"]) == (native["leaves"], native["nodes"], native["errors"])
    # the session is untouched by the search
    assert fs.view_json() == before[0] and np.array_equal(fs.encode_battle()["mon_num"], before[1])
    assert sum(native["visits"]) >= 0.9 * 96

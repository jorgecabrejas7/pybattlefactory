"""The browser build (build-wasm/gen3_web.mjs, scripts/build_wasm.sh gen3_web) gives exactly what the native
FactorySession gives: same views, run info, battle log, estimates and encodings for the same seeds and actions
(tests/web/transcript.mjs drives the wasm side)."""

import json
import os
import shutil
import subprocess

import pytest

from pybattle.pybattle_native import FactorySession

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WASM = os.path.join(REPO, "build-wasm", "gen3_web.mjs")

pytestmark = pytest.mark.skipif(not (os.path.exists(WASM) and shutil.which("node")),
                                reason="needs node and the wasm build (scripts/build_wasm.sh gen3_web)")

LAYOUT = (("mon_ids", False), ("mon_num", True), ("move_num", True), ("ctx_ids", False), ("ctx_num", True),
          ("mask", False))


class Lcg:
    def __init__(self, s):
        self.s = s

    def __call__(self, n):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        return (self.s >> 33) % n


def native_transcript(runs, seed0):
    fs = FactorySession(True, 150)
    for r in range(runs):
        rnd = Lcg(seed0 * 7919 + r)
        fs.reset(seed0 * 1000003 + r, (r * 9) % 49, 0)
        for d in range(3000):
            phase = fs.phase
            view = json.loads(fs.view_json())
            line = {"r": r, "d": d, "phase": phase, "view": view, "log": json.loads(fs.take_log_json())}
            if phase == 4:
                yield line
                break
            line["info"] = json.loads(fs.run_info_json())
            if phase in (2, 3):
                line["est"] = json.loads(fs.estimates_json())
                e = fs.encode_battle()
                line["enc"] = {k: [float(x) for x in e[k].ravel()] if f else
                               [int(x) for x in e[k].ravel()] for k, f in LAYOUT}
                line["enc"]["active"] = int(e["active"])
            yield line
            if phase == 0:
                while True:
                    a, b, c = rnd(6), rnd(6), rnd(6)
                    try:
                        fs.act_rental(a, b, c)
                        break
                    except Exception:
                        pass
            elif phase == 1:
                keep, own, foe = rnd(2), rnd(3), rnd(3)
                try:
                    if keep == 0:
                        raise ValueError
                    fs.act_swap(own, foe)
                except Exception:
                    fs.act_swap(-1, 0)
            else:
                legal = [a for a in view["legal_actions"] if a[0] != "forfeit"]
                a = legal[rnd(len(legal))]
                fs.act_move(a[1]) if a[0] == "move" else fs.act_switch(a[1])


def test_wasm_matches_native():
    runs, seed = 12, 3
    out = subprocess.run(["node", os.path.join(REPO, "tests", "web", "transcript.mjs"), str(runs), str(seed)],
                         capture_output=True, text=True, check=True, timeout=600).stdout.splitlines()
    native = list(native_transcript(runs, seed))
    assert len(out) == len(native)
    for got, want in zip(out, native):
        got = json.loads(got)
        if "enc" in want:
            assert got["enc"] == want["enc"], (want["r"], want["d"])
        for k in ("view", "log", "info", "est", "phase"):
            assert got.get(k) == want.get(k), (k, want["r"], want["d"])

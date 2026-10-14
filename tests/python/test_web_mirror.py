"""The browser build's mirror mode (FactorySession.mirror_*: the real game in an emulator, its battles replayed by the
simulator) sees exactly what the Python emulator backend sees: same battle views, encodings and swap views, with the
player's action found by the simulator itself. Needs the ROM and save (skipped otherwise)."""

import json
import os
import random

import numpy as np
import pytest

from pybattle.backend import Phase
from pybattle.emu.decode import SB2_RENTAL_MONS, SYMBOLS as S
from pybattle.pybattle_native import FactorySession
from pybattle.view import BattleObserver
from pybattle.web_schema import view_json
from rl import encode
from rl.real_agents import ctx_of

ROM = os.path.expanduser("~/Desktop/Dev/pokemon-battle-factory/roms/Pokemon - Emerald Version (USA, Europe).gba")
SAV = ROM[:-4] + ".sav"
REGIONS = ["gBattleMons", "gBattlerPartyIndexes", "gDisableStructs", "gSideStatuses", "gSideTimers", "gStatuses3",
           "gBattleWeather", "gWishFutureKnock", "gProtectStructs", "gLastMoves", "gLastLandedMoves", "gLastHitBy",
           "gChosenMoveByBattler", "gRandomTurnNumber", "gBattleResults"]

pytestmark = pytest.mark.skipif(not os.path.exists(ROM), reason="needs the ROM")


def snapshot(d):
    emu = d.emu
    out = emu.read32(S.addr("gRngValue")).to_bytes(4, "little")
    for n in REGIONS:
        out += emu.read(S.addr(n), S.size(n))
    return out + d.player_party_raw()[:300] + emu.read(S.addr("gEnemyParty"), 300)


def test_mirror_matches_emulator_views():
    from pybattle.emu.mirror import WatchedMachine, _battle_start
    from pybattle.emu_backend import EmuBackend
    old = encode.VERSION
    encode.set_version(5)
    b = EmuBackend(ROM, SAV)
    wm = WatchedMachine(b.d.emu)
    b.d.emu = wm
    fs = FactorySession(True, 500)
    fs.mirror_regions([(S.addr(n), S.size(n)) for n in REGIONS])
    finished = []
    orig_finish = BattleObserver.finish

    def finish(self, ram):
        orig_finish(self, ram)
        finished.append(fs.mirror_finish(snapshot(b.d)))

    BattleObserver.finish = finish
    rng = random.Random(3)
    checked = 0
    try:
        for run in range(2):
            b.reset(seed=11 + run, win_streak=0)
            hint = (18, 0)
            while b.phase != Phase.RUN_OVER and checked < 150:
                v, info = b.view(), b.run_info()
                fs.set_context(info.win_streak, info.battle_in_challenge, info.challenge_num, info.rents, info.noland, 0)
                if b.phase == Phase.RENTAL:
                    cands = b.d.rental_candidates()
                    raw = b"".join(c.frontier_mon_id.to_bytes(2, "little") + b.d.emu.read(
                        b.d._select_screen() + 12 + i * 108 + 8, 100) for i, c in enumerate(cands))
                    assert json.loads(fs.rental_view_json(raw, v.hint_type, v.hint_style)) == view_json(v)
                    hint = (v.hint_type, v.hint_style)
                    act = (0, 1, 2) if len({v.candidates[i].species for i in (0, 1, 2)}) == 3 else (0, 1, 3)
                elif b.phase == Phase.SWAP:
                    own = b.d.emu.read(S.addr("gPlayerParty"), 300)
                    foe = b.d.emu.read(S.addr("gEnemyParty"), 300)
                    assert finished and finished[-1], fs.mirror_error
                    assert json.loads(fs.swap_view_json(own, foe, v.hint_type, v.hint_style)) == view_json(v)
                    hint = (v.hint_type, v.hint_style)
                    act = None
                else:
                    start = wm.take_start()
                    if start is not None:
                        start = dict(start, player_party=bytes.fromhex(start["player_party"]),
                                     enemy_party=bytes.fromhex(start["enemy_party"]),
                                     rental_mons=bytes.fromhex(start["rental_mons"]))
                        fs.mirror_begin(start, *hint)
                    ok = fs.mirror_sync(b.phase == Phase.FORCED_SWITCH, snapshot(b.d))
                    assert ok, fs.mirror_error
                    web = json.loads(fs.view_json())
                    assert web == view_json(v)
                    e, p = fs.encode_battle(), encode.battle(v, ctx_of(info))
                    for k in ("mon_ids", "mon_num", "move_num", "ctx_num", "mask"):
                        assert np.allclose(np.asarray(e[k], np.float64), np.asarray(p[k], np.float64), atol=1e-6), k
                    checked += 1
                    if checked % 5 == 3:
                        # a savestate of the page resumes the mirrored battle (a fresh session, the saved state)
                        state = fs.save_state()
                        fs = FactorySession(True, 500)
                        fs.mirror_regions([(S.addr(n), S.size(n)) for n in REGIONS])
                        fs.load_state(state)
                        assert fs.mirror_sync(b.phase == Phase.FORCED_SWITCH, snapshot(b.d)), fs.mirror_error
                        assert json.loads(fs.view_json()) == web
                    legal = [a for a in v.legal_actions if a[0] != "forfeit"]
                    act = rng.choice(legal)
                b.act(act)
    finally:
        BattleObserver.finish = orig_finish
        encode.set_version(old)
    assert checked > 20

"""alphazero_v3 on the real game (rl/real_agents.py, rl/real_play.py, pybattle/emu/options.py, pybattle/emu/mirror.py,
the viewer's panels; docs/RL_DECISIONS.md §24).

- options bits: Text Speed / Battle Scene written in SaveBlock2 and read back, other bits untouched;
- replay determinism: a short run played headless (search mode, Battle Scene Off) and replayed from its savestate
  with the logged actions gives the same decision kinds, gRngValue and frame at every decision and the same
  outcomes; the simulator mirror the search runs on stayed byte-identical to the emulator;
- UI: the watch window draws every panel (battler with search visits / Q, tactician table, opponent model, header)
  with SDL's dummy video driver;
- agent parity: V3Player takes the same action as rl.eval_rounds' agents (the network with the features hook) on
  the same states (SimBackend; the planner's draws seeded the same).

Needs the ROM / save for the emulator tests and runs/alphazero_v3/latest.pt (+ its team evaluator,
runs/opp_model_v3/model.pt) for the agent tests; skipped otherwise.
"""

import os

import numpy as np
import pytest

_ROMS = "/home/apollo/Desktop/Dev/pokemon-battle-factory/roms"
ROM = os.environ.get("PYBATTLE_ROM", os.path.join(_ROMS, "Pokemon - Emerald Version (USA, Europe).gba"))
SAVE = os.environ.get("PYBATTLE_SAVE", os.path.join(_ROMS, "Pokemon - Emerald Version (USA, Europe).sav"))
CKPT = "runs/alphazero_v3/latest.pt"
OPP = "runs/opp_model_v3/model.pt"

have_rom = os.path.exists(ROM) and os.path.exists(SAVE)
have_ckpt = os.path.exists(CKPT) and os.path.exists(OPP) and os.path.exists("runs/alphazero_v3/team_eval_0580.pt")
needs_rom = pytest.mark.skipif(not have_rom, reason="ROM/save not available")
needs_ckpt = pytest.mark.skipif(not have_ckpt, reason="alphazero_v3 checkpoint / opponent model not available")


def _emu_new():
    from pybattle import pybattle_emu as native
    e = native.Emulator(ROM, SAVE)
    if not hasattr(e, "set_frameskip"):
        pytest.skip("pybattle_emu built before frameskip / clock / save memory support")
    return e


@needs_rom
def test_options_bits():
    from pybattle.emu.decode import SYMBOLS as S
    from pybattle.emu.driver import FactoryDriver
    from pybattle.emu.options import read_options, write_options
    d = FactoryDriver(machine=_emu_new())
    d.boot_to_overworld()
    sb2 = d.emu.read32(S.addr("gSaveBlock2Ptr"))
    before = d.emu.read16(sb2 + 0x14)
    o = write_options(d.emu, text_speed="fast", battle_scene=False)
    v = d.emu.read16(sb2 + 0x14)
    assert o["text_speed"] == "fast" and o["battle_scene"] is False
    assert v & 7 == 2 and (v >> 10) & 1 == 1
    assert v & ~(7 | (1 << 10)) == before & ~(7 | (1 << 10))       # frame type, sound, battle style untouched
    o = write_options(d.emu, text_speed="slow", battle_scene=True)
    v = d.emu.read16(sb2 + 0x14)
    assert o == read_options(d.emu) and v & 7 == 0 and not (v >> 10) & 1


@needs_rom
def test_frameskip_and_clock_do_not_change_emulation():
    """No rendering (frameskip) and the emulated-time clock: same RAM after the same frames as drawing every frame."""
    from pybattle.emu.decode import SYMBOLS as S
    out = []
    for fs in (0, 1 << 30):
        e = _emu_new()
        e.set_frameskip(fs)
        e.set_rtc_fake_epoch(1767268800000)
        e.run_frames(2500)
        out.append((e.read32(S.addr("gRngValue")), e.read(S.addr("gSaveBlock2Ptr"), 4), e.frame))
    assert out[0] == out[1]


@needs_rom
@needs_ckpt
def test_replay_is_deterministic(tmp_path):
    from rl import real_play as RP
    stats = RP.play_worker(0, str(tmp_path), ROM, SAVE, mode="search", max_runs=1, max_battles=2, resume=False,
                           log=lambda *a: None)
    assert stats["battles"] == 2 and stats["errors"] == 0
    assert stats["desyncs"] == 0 and stats["search_fallbacks"] == 0      # the mirror matched the emulator
    run = os.path.join(str(tmp_path), "w0", "runs", "run_00000")
    meta, recs = RP.load_run(run)
    assert meta["options"]["battle_scene"] is False and meta["options"]["text_speed"] == "fast"
    logged = [r["won"] for r in recs if r.get("event") == "battle_end"]
    rep = RP.replay_run(run, ROM, SAVE, strict=False)
    assert rep["ok"], rep["mismatches"][:3]
    assert rep["outcomes"] == logged and rep["battles"] == 2 and rep["same_frames"]
    assert rep["decisions"] == sum(1 for r in recs if "event" not in r)
    # the play log got every decision
    lines = open(os.path.join(str(tmp_path), "w0", "playlog.jsonl")).read().count('"event": "decision"')
    assert lines == rep["decisions"]


def _scenes():
    """Scenes for every panel from a SimBackend run: a rental, a battle decision (search), a swap."""
    from pybattle.backend import Phase, SimBackend
    from rl.real_agents import V3Player
    from pybattle.emu.viewer import Scene
    be = SimBackend()
    be.reset(seed=5, win_streak=14)
    pl = V3Player(CKPT, mode="search", sims=64, dets=4, threads=1)
    out = {}
    while len(out) < 3:
        ph, view, info = be.phase, be.view(), be.run_info()
        ch = pl.act(be, ph, view, info)
        key = {Phase.RENTAL: "rental", Phase.BATTLE: "battle", Phase.SWAP: "swap"}.get(ph)
        if key and key not in out and (key != "battle" or ch.details.get("search")):
            out[key] = Scene(ph.value, view, info, ch, False, None)
        be.act(ch.action)
        if be.phase == Phase.RUN_OVER:
            be.reset(seed=6, win_streak=14)
    return out


@needs_ckpt
def test_ui_renders_all_panels(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    import pygame
    from pybattle.emu import ui
    from pybattle.emu.viewer import Viewer
    scenes = _scenes()
    v = Viewer(speed=2, size=(1920, 1080), beliefs_panel=True, mode_label="search 512")
    assert v.r_beliefs is not None and v.r_beliefs.h > 100
    v._frame = np.zeros((160, 240, 3), np.uint8)
    for key, sc in scenes.items():
        v.set_scene(sc)
        v.info = sc.info
        v.draw()
        d = sc.details
        if key == "battle":
            assert d["search"] is not None and len(d["search"]["visits"]) == 7
            assert d["beliefs"] and len(d["beliefs"]) == 3
            assert all("q" in a and "visits" in a for a in d["actions"])
            surf = pygame.Surface((900, 1200))
            p = ui.Painter(surf, v.fonts, 20, ui.THEMES["dark"])
            assert ui.draw_beliefs(p, 0, 0, 880, sc) > 100
        else:
            rows = d["tactician_options"]
            assert rows and all(k in rows[0] for k in ("p", "w_normal", "w_last", "e_round", "q_plan", "delta"))
            assert d["hint"] and d["expected_wins"] is not None
        # the decision panel really drew something (not just the background)
        arr = pygame.surfarray.array3d(v.screen.subsurface(v.r_right))
        assert arr.std() > 10
    # the beliefs panel keeps the battle's beliefs while the tactician decides
    v.set_scene(scenes["swap"])
    v.draw()
    assert getattr(v, "_belief_scene", None) is scenes["battle"]
    pygame.quit()


@needs_ckpt
def test_agent_parity_with_eval():
    """Same states, same actions as rl.eval_rounds (network + features hook for the tactician, net_battler)."""
    import torch
    from rl import encode
    from rl.alphazero_v3 import features_hook
    from rl.envs import NO_ACTION, FactoryEnv, decode_action
    from rl.eval_rounds import net_battler
    from rl.policy import Policy
    from rl.real_agents import V3Player
    torch.set_num_threads(1)
    policy = Policy(CKPT, torch.device("cpu"))
    hook = features_hook(CKPT)
    pl = V3Player(CKPT, mode="net", threads=1, planner_seed=False, show_beliefs=False)
    battler = net_battler(policy)
    env = FactoryEnv(1234, win_streak=14, obs_hook=hook)
    seen = {"rental": 0, "swap": 0, "battle": 0}
    seed = 1000
    hook.planner_cfg["seed"] = seed            # the env computes the next tactician observation when it steps
    ev = env._advance(NO_ACTION)
    for step in range(400):
        kind = ev["kind"]
        if kind == "battle":
            want = battler(env, ev)
        else:
            want = decode_action(kind, policy.act(kind, encode.collate([ev["obs"]], policy.device), greedy=True)[0])
        pl.hook.planner_cfg["seed"] = seed     # the same planner draws on both sides
        got = pl.act(env.backend).action
        assert got == want, (step, kind, got, want)
        seen[kind] += 1
        seed += 1
        hook.planner_cfg["seed"] = seed
        ev = env.step(want)
    assert seen["rental"] >= 1 and seen["swap"] >= 1 and seen["battle"] >= 50

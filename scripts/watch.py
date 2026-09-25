#!/usr/bin/env python3
"""Watch agents play the Battle Factory on the real game — or play it yourself and see what the
agents would do.

    python scripts/watch.py --rom Emerald.gba --save Emerald.sav [--speed 2] [--step] [--human]

The save loads, the player walks to the singles attendant, and runs are played one after another
(after a loss the game sends you back to the lobby) until you close the window. Agents default to
the two random dummies in examples/random_agents.py; pass --agents module:Tactician,module:Battler
to use your own (classes with act(phase, view, run_info) -> pybattle.agent.Choice).

The window can be resized, maximized or made fullscreen; everything scales with it. Controls are
buttons along the bottom: Pause/Resume, Next (step mode: do the shown action), Step on/off, speed
- / +, Human on/off, text size A-/A+, Theme, Fullscreen, Quit. The keyboard plays the GBA only in
human mode: arrows, Z = A, X = B, Enter = Start, Backspace = Select, A/S = L/R.
Decisions are also written to watch_log.jsonl.
"""

import argparse
import importlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pybattle.backend import Phase  # noqa: E402
from pybattle.emu.decode import SYMBOLS as S  # noqa: E402
from pybattle.emu.driver import Decision  # noqa: E402
from pybattle.emu.viewer import Scene, Viewer, ViewedMachine  # noqa: E402
from pybattle.emu_backend import EmuBackend  # noqa: E402
from pybattle.view import BattleObserver, BattleView  # noqa: E402

try:                                    # damage estimates for the panels (pure Python; optional)
    from rl.damage import battle_estimates
except Exception:                       # pragma: no cover
    battle_estimates = None


def estimates(view, info):
    """rl.damage's per-move damage estimates for a battle view, or None."""
    if battle_estimates is None or not isinstance(view, BattleView):
        return None
    try:
        return battle_estimates(view, {"challenge": info.challenge_num, "battle": info.battle_in_challenge})
    except Exception:
        return None


def scene(phase, view, info, choice=None, suggestion=False):
    return Scene(phase.value, view, info, choice, suggestion, estimates(view, info))


def load_agents(spec):
    if not spec:
        from examples.random_agents import RandomBattler, RandomTactician
        return RandomTactician(), RandomBattler()
    out = []
    for part in spec.split(","):
        mod, cls = part.split(":")
        out.append(getattr(importlib.import_module(mod), cls)())
    return out


PHASE_OF = {Decision.RENTAL_SELECT: Phase.RENTAL, Decision.BATTLE_ACTION: Phase.BATTLE,
            Decision.PARTY_MENU: Phase.FORCED_SWITCH, Decision.SWAP_QUESTION: Phase.SWAP}


def human_play(backend, viewer, tactician, battler):
    """You drive the game; at each decision the panel shows what the agents would do."""
    d, m = backend.d, backend.d.emu
    last_key = None
    viewer.add_log("— human mode —", "human")
    while viewer.human:
        m.set_keys(viewer.human_keys())
        m.run_frames(1)
        dec = d.decision()
        if dec == Decision.NONE and d._in_pre_battle_yes_no():
            dec = Decision.SWAP_QUESTION
        phase = PHASE_OF.get(dec)
        if phase is None:
            continue
        sb2 = d.saveblock2()
        # one suggestion per decision: phase, battle number, battle turn, and the rentals on offer
        key = (phase, d.emu.read16(sb2 + 0xCB2), d.emu.read8(S.addr("gBattleResults") + 0x13),
               d.emu.read(sb2 + 0xE70, 72) if phase == Phase.RENTAL else b"")
        if key == last_key:
            continue
        last_key = key
        backend.phase = phase
        try:
            view = backend.view()
        except Exception:                           # e.g. a menu still opening
            last_key = None
            continue
        agent = tactician if phase in (Phase.RENTAL, Phase.SWAP) else battler
        info = backend.run_info()
        viewer.set_scene(scene(phase, view, info, None, suggestion=True))
        choice = agent.act(phase, view, info)
        if phase in (Phase.RENTAL, Phase.SWAP):
            backend._observer = BattleObserver(*d.hints())     # a new battle follows: fresh memory
        viewer.set_scene(scene(phase, view, info, choice, suggestion=True))
        viewer.add_log(f"[you] {phase.value}: agent would {choice.reason}", "human")
    m.set_keys(0)
    viewer.add_log("— agents take over at the next decision —", "human")
    backend._sync()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rom", required=True)
    ap.add_argument("--save", required=True)
    ap.add_argument("--speed", type=int, default=1, choices=[0, 1, 2, 4, 8], help="0 = max")
    ap.add_argument("--step", action="store_true", help="start in step mode (click Next before each agent action)")
    ap.add_argument("--human", action="store_true", help="start in human mode")
    ap.add_argument("--agents", default="", help="module:Tactician,module:Battler (default: random dummies)")
    ap.add_argument("--log", default="watch_log.jsonl")
    ap.add_argument("--open-level", action=argparse.BooleanOptionalAction, default=True,
                    help="Open Level (level 100); --no-open-level for Level 50")
    ap.add_argument("--seconds", type=float, default=0, help="stop after this long (0 = until closed)")
    ap.add_argument("--snapshot", default="", help="save the window to this PNG when stopping")
    ap.add_argument("--size", default="", help="window size WxH (default: fits the screen)")
    ap.add_argument("--theme", default="dark", choices=["dark", "light"])
    args = ap.parse_args()

    tactician, battler = load_agents(args.agents)
    size = tuple(int(v) for v in args.size.lower().split("x")) if args.size else None
    viewer = Viewer(speed=args.speed, step=args.step, human=args.human, size=size, theme=args.theme)
    if args.seconds:
        viewer.deadline = time.time() + args.seconds
    log = open(args.log, "a")

    def game():
        """Runs in a worker thread; the window (main thread) stays responsive meanwhile."""
        viewer.set_status("booting the save…")
        machine = ViewedMachine(args.rom, args.save, viewer)
        backend = EmuBackend(machine=machine, boot=True, open_level=args.open_level)
        run = 0
        while True:
            run += 1
            viewer.set_run(run)
            viewer.add_log(f"=== run {run} ===", "run")
            if viewer.human:
                human_play(backend, viewer, tactician, battler)
            if backend.phase == Phase.RUN_OVER:             # in the lobby: walk over and start a challenge
                viewer.set_status("walking to the attendant…")
                backend.reset(win_streak=None, rewind=False)
            # else: a human handed over in the middle of a challenge; the agents carry on from there
            viewer.set_status("")
            while backend.phase != Phase.RUN_OVER:
                viewer.check_quit()
                if viewer.human:
                    human_play(backend, viewer, tactician, battler)
                    continue
                phase, view = backend.phase, backend.view()
                info = backend.run_info()
                agent = tactician if phase in (Phase.RENTAL, Phase.SWAP) else battler
                viewer.set_scene(scene(phase, view, info))           # "thinking…"
                choice = agent.act(phase, view, info)
                viewer.check_quit()
                viewer.set_scene(scene(phase, view, info, choice))
                viewer.add_log(f"{phase.value}: {choice.reason}")
                log.write(json.dumps({"time": time.time(), "run": run, "phase": phase.value,
                                      "action": choice.action, "reason": choice.reason}) + "\n")
                log.flush()
                viewer.wait_enter()
                backend.act(choice.action)
                if backend.last_battle_won is not None and phase in (Phase.BATTLE, Phase.FORCED_SWITCH) \
                        and backend.phase not in (Phase.BATTLE, Phase.FORCED_SWITCH):
                    won = backend.last_battle_won
                    viewer.add_log("battle " + ("WON" if won else "LOST"), "won" if won else "lost")
            viewer.add_log(f"run {run} over: {backend.wins} wins", "run")

    try:
        viewer.run(game)
    finally:
        log.close()
        if args.snapshot:
            import pygame
            pygame.image.save(viewer.screen, args.snapshot)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""alphazero_v3 playing the real game (emulator) headless at full speed: logs, savestates, exact replays and the
video of the first full win (docs/RL_DECISIONS.md §24).

    # 2 workers with the search, 1 with the network alone (3 emulators), until stopped; records the first full win
    python scripts/play_real.py play --modes search,search,net --out runs/real_play [--hours 0] [--resume]
    python scripts/play_real.py status --out runs/real_play        # throughput, streaks, full wins, video
    touch runs/real_play/STOP                                      # stop every worker after its current decision

    # one run again, from its savestate with its logged actions
    python scripts/play_real.py verify RUN_DIR [--scene same|on]   # headless, checks RNG / frames / outcomes
    python scripts/play_real.py replay RUN_DIR [--speed 2]         # in the watch window
    python scripts/play_real.py record RUN_DIR --video out.mp4 [--speed 2] [--until-wins 42] [--size 1920x1080]

    python scripts/play_real.py bench --mode net --battles 20 [--scene on] [--render]   # battles / hour

Every worker plays like a player: Text Speed Fast and Battle Scene Off (default; --scene on keeps the animations),
no rendering, uncapped. Per run it keeps a savestate of the lobby and the raw action log (rl/real_play.py); per
decision the agent's details go to <out>/w<k>/playlog.jsonl (rl.playlog.PlayLogger). The first run that reaches 42
wins (gold Noland) is replayed with the full UI at x2 and recorded to <out>/videos/ (one worker pauses meanwhile,
so at most --max-emulators emulators run at once).
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

ROM = "/home/apollo/Desktop/Dev/pokemon-battle-factory/roms/Pokemon - Emerald Version (USA, Europe).gba"
SAV = ROM[:-4] + ".sav"


def _log_to(path):
    f = open(path, "a", buffering=1)

    def log(msg):
        line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
        print(line, flush=True)
        f.write(line + "\n")
    return log


def _nice(n):
    try:
        os.nice(n)
    except OSError:
        pass


# ---- commands -------------------------------------------------------------------------------------------------------

def cmd_worker(a):
    _nice(a.nice)
    from rl.real_play import FAST_OPTIONS, play_worker
    opts = dict(FAST_OPTIONS, battle_scene=a.scene == "on")
    os.makedirs(os.path.join(a.out, f"w{a.wid}"), exist_ok=True)
    log = _log_to(os.path.join(a.out, f"w{a.wid}", "worker.log"))
    log(f"worker {a.wid}: mode {a.mode}, options {opts}, pid {os.getpid()}")
    play_worker(a.wid, a.out, a.rom, a.save, mode=a.mode, seed=a.seed, hours=a.hours, sims=a.sims,
                threads=a.threads, options=opts, resume=a.resume, max_runs=a.max_runs, log=log)
    log(f"worker {a.wid}: stopped")


def _spawn(cmd, logpath, nice):
    out = open(logpath, "a")
    return subprocess.Popen(["nice", "-n", str(nice), sys.executable] + cmd, stdout=out, stderr=subprocess.STDOUT,
                            cwd=ROOT, start_new_session=True)


def cmd_play(a):
    """The supervisor: workers + automatic recording of the first full win."""
    os.makedirs(a.out, exist_ok=True)
    log = _log_to(os.path.join(a.out, "supervisor.log"))
    modes = a.modes.split(",")
    scenes = a.scene.split(",")
    scenes = (scenes * len(modes))[:len(modes)] if len(scenes) == 1 else scenes
    if len(scenes) != len(modes) or any(x not in ("on", "off") for x in scenes):
        raise SystemExit("--scene: on | off, or one per worker (e.g. on,on,off)")
    if len(modes) > a.max_emulators:
        raise SystemExit(f"{len(modes)} workers > --max-emulators {a.max_emulators}")
    try:
        os.remove(os.path.join(a.out, "STOP"))
    except FileNotFoundError:
        pass
    json.dump({"pid": os.getpid(), "modes": modes, "started": time.time(), "args": vars(a)},
              open(os.path.join(a.out, "supervisor.json"), "w"), indent=1)

    def worker_cmd(k, resume):
        return [os.path.join(ROOT, "scripts", "play_real.py"), "worker", "--wid", str(k), "--mode", modes[k],
                "--out", a.out, "--rom", a.rom, "--save", a.save, "--seed", str(a.seed), "--sims", str(a.sims),
                "--threads", str(a.threads), "--scene", scenes[k], "--hours", str(a.hours), "--nice", "0"] + \
            (["--resume"] if resume else [])

    procs = {}
    for k in range(len(modes)):
        procs[k] = _spawn(worker_cmd(k, a.resume), os.path.join(a.out, f"w{k}.out"), a.nice)
        log(f"started worker {k} ({modes[k]}, battle scene {scenes[k]}), pid {procs[k].pid}")
    recorded = _recorded(a.out)
    recorder = None
    paused = None
    while True:
        time.sleep(10)
        alive = {k: p for k, p in procs.items() if p.poll() is None}
        if os.path.exists(os.path.join(a.out, "STOP")) and not alive and recorder is None:
            log("all workers stopped")
            break
        # the first full win not recorded yet -> pause one worker, record, resume it
        if a.record and recorder is None:
            todo = [w for w in _full_wins(a.out) if w["dir"] not in recorded]
            if todo:
                win = todo[0]
                if len(alive) + 1 > a.max_emulators:
                    if paused is None:                          # the network worker if any, else the last one
                        paused = next((k for k in reversed(range(len(modes))) if modes[k] == "net" and k in alive),
                                      max(alive))
                        open(os.path.join(a.out, f"w{paused}", "STOP"), "w").close()
                        log(f"full win {win['run']} (worker {win['worker']}): pausing worker {paused} to record it")
                    continue
                vdir = os.path.join(a.out, "videos")
                os.makedirs(vdir, exist_ok=True)
                name = f"w{win['worker']}_{win['run']}_x{a.record_speed}"
                video = os.path.join(vdir, name + ".mp4")
                recorder = (_spawn([os.path.join(ROOT, "scripts", "play_real.py"), "record", win["dir"], "--video",
                                    video, "--speed", str(a.record_speed), "--until-wins", "42", "--rom", a.rom,
                                    "--save", a.save, "--size", a.record_size, "--scene", a.record_scene,
                                    "--report", os.path.join(vdir, name + ".json")],
                                   os.path.join(vdir, name + ".log"), a.nice), win, video)
                log(f"recording {win['run']} -> {video} (pid {recorder[0].pid})")
        if recorder is not None and recorder[0].poll() is not None:
            code = recorder[0].returncode
            win, video = recorder[1], recorder[2]
            log(f"recording of {win['run']} finished (exit {code}): {video}")
            recorded.add(win["dir"])
            with open(os.path.join(a.out, "recorded.jsonl"), "a") as f:
                f.write(json.dumps({"dir": win["dir"], "video": video, "exit": code, "time": time.time()}) + "\n")
            recorder = None
            if code == 0:
                a.record = a.record_all                         # by default only the first full win
            if paused is not None:
                try:
                    os.remove(os.path.join(a.out, f"w{paused}", "STOP"))
                except FileNotFoundError:
                    pass
        for k in range(len(modes)):                             # restart workers that stopped (pause over / crash)
            p = procs[k]
            if p.poll() is None or os.path.exists(os.path.join(a.out, "STOP")) \
                    or os.path.exists(os.path.join(a.out, f"w{k}", "STOP")):
                continue
            running = sum(q.poll() is None for q in procs.values()) + (recorder is not None)
            if running >= a.max_emulators:
                continue
            log(f"worker {k} exited ({p.returncode}); restarting with --resume")
            procs[k] = _spawn(worker_cmd(k, True), os.path.join(a.out, f"w{k}.out"), a.nice)
            if k == paused:
                paused = None


def _full_wins(out):
    p = os.path.join(out, "full_wins.jsonl")
    if not os.path.exists(p):
        return []
    return [json.loads(line) for line in open(p) if line.strip()]


def _recorded(out):
    p = os.path.join(out, "recorded.jsonl")
    if not os.path.exists(p):
        return set()
    return {json.loads(line)["dir"] for line in open(p) if line.strip() and json.loads(line).get("exit") == 0}


def cmd_status(a):
    tot = {"battles": 0, "runs": 0, "seconds": 0.0, "full_wins": 0, "battles_per_hour": 0.0}
    print(f"{'worker':7s} {'mode':7s} {'runs':>6s} {'battles':>8s} {'win%':>6s} {'bat/h':>7s} {'best':>5s} "
          f"{'full':>5s} {'ms/battler':>10s}  current")
    for sp in sorted(glob.glob(os.path.join(a.out, "w*", "stats.json"))):
        s = json.load(open(sp))
        wdir = os.path.dirname(sp)
        cur = ""
        runs = sorted(glob.glob(os.path.join(wdir, "runs", "run_*")))
        if runs and not os.path.exists(os.path.join(runs[-1], "result.json")):
            n = sum(1 for line in open(os.path.join(runs[-1], "actions.jsonl")) if '"battle_end"' in line and
                    '"won": true' in line)
            cur = f"{os.path.basename(runs[-1])}: {n} wins so far"
        t = s.get("timing", {})
        msb = t.get("battler_ms", 0) / max(t.get("battler", 0), 1)
        winp = s["wins"] / max(s["battles"], 1)
        age = time.time() - s.get("updated", 0)
        print(f"w{s['worker']:<6d} {s['mode']:7s} {s['runs']:6d} {s['battles']:8d} {winp:6.1%} "
              f"{s.get('battles_per_hour', 0):7.0f} {s['best']:5d} {s['full_wins']:5d} {msb:10.0f}  {cur}"
              f"{'' if age < 3600 else f'  (stats {age / 3600:.1f} h old)'}")
        for k in tot:
            tot[k] += s.get(k, 0)
    print(f"total: {tot['runs']} runs, {tot['battles']} battles, {tot['full_wins']} full wins, "
          f"{tot['battles_per_hour']:.0f} battles/h (sum over workers; counted at the end of each run)")
    for w in _full_wins(a.out):
        print("FULL WIN:", w["run"], "worker", w["worker"], w["mode"], w["dir"])
    p = os.path.join(a.out, "recorded.jsonl")
    if os.path.exists(p):
        for line in open(p):
            r = json.loads(line)
            print("video:", r["video"], "(exit", r["exit"], ")")


def cmd_verify(a):
    from rl.real_play import replay_run
    t = time.time()
    rep = replay_run(a.run, a.rom, a.save, scene=a.scene, strict=False,
                     until_wins=a.until_wins if a.until_wins else None)
    rep["seconds"] = time.time() - t
    print(json.dumps({k: rep[k] for k in ("ok", "decisions", "battles", "wins", "outcomes", "same_frames", "scene",
                                          "seconds")}))
    for m in rep["mismatches"][:10]:
        print("MISMATCH", m)
    return 0 if rep["ok"] else 1


def cmd_record(a):
    _nice(a.nice)
    from rl.real_play import record_run
    size = tuple(int(v) for v in a.size.lower().split("x"))
    rep = record_run(a.run, a.rom, a.save, a.video, until_wins=a.until_wins or None, game_speed=a.speed, fps=a.fps,
                     size=size, hold=a.hold, scene=a.scene, codec=a.codec, headless=not a.show,
                     from_wins=a.from_wins)
    if a.report:
        json.dump(rep, open(a.report, "w"), indent=1)
    print(json.dumps({k: rep[k] for k in ("ok", "battles", "wins", "video", "video_seconds", "wall_seconds", "codec")}))
    import shutil
    shutil.copy(os.path.join(a.run, "actions.jsonl"), os.path.splitext(a.video)[0] + ".actions.jsonl")
    shutil.copy(os.path.join(a.run, "run.json"), os.path.splitext(a.video)[0] + ".run.json")
    return 0 if rep["ok"] else 1


def cmd_replay(a):
    """The logged run in the watch window (live, at --speed)."""
    from pybattle.emu.viewer import Scene, Viewer, ViewedMachine
    from rl.real_play import load_run, replay_run
    meta, _ = load_run(a.run)
    viewer = Viewer(speed=a.speed, beliefs_panel=True, mode_label=f"REPLAY · {meta.get('mode', '')}")

    class _C:
        def __init__(self, action, reason, details):
            self.action, self.reason, self.details = action, reason, details

    def game():
        def wrap(emu):
            m = ViewedMachine.__new__(ViewedMachine)
            m.emu, m.viewer = emu, viewer
            return m

        def on_decision(be, rec, action):
            view, info = be.view(), be.run_info()
            viewer.set_scene(Scene(be.phase.value, view, info, _C(action, rec.get("reason", ""), rec.get("details"))))
            viewer.add_log(f"{be.phase.value}: {rec.get('reason', '')}")
            viewer.wait_enter()

        rep = replay_run(a.run, a.rom, a.save, machine_wrap=wrap, scene=a.scene, on_decision=on_decision,
                         on_battle_end=lambda won, be: viewer.add_log("battle " + ("WON" if won else "LOST"),
                                                                      "won" if won else "lost"),
                         on_status=viewer.set_status, until_wins=a.until_wins or None)
        viewer.add_log(f"replay over: {'identical' if rep['ok'] else 'DIVERGED'}", "run")

    viewer.run(game)


def cmd_bench(a):
    _nice(a.nice)
    from pybattle.backend import Phase
    from rl.real_agents import V3Player
    from rl.real_play import FAST_OPTIONS, NO_RENDER, make_backend
    opts = dict(FAST_OPTIONS, battle_scene=a.scene == "on")
    be = make_backend(a.rom, a.save, options=opts, frameskip=0 if a.render else NO_RENDER)
    pl = V3Player(mode=a.mode, sims=a.sims, threads=a.threads, seed=a.seed)
    be.reset(win_streak=0, rewind=False)
    emu = be.d.emu
    t0, f0, battles, wins = time.time(), emu.frame, 0, 0
    while battles < a.battles:
        if be.phase == Phase.RUN_OVER:
            be.reset(win_streak=None, rewind=False)
        ph = be.phase
        be.act(pl.act(be).action)
        if ph in (Phase.BATTLE, Phase.FORCED_SWITCH) and be.phase not in (Phase.BATTLE, Phase.FORCED_SWITCH):
            battles += 1
            wins += bool(be.last_battle_won)
    dt = time.time() - t0
    t = pl.timing
    res = {"mode": a.mode, "scene": a.scene, "render": a.render, "battles": battles, "wins": wins, "seconds": dt,
           "battles_per_hour": battles / dt * 3600, "frames_per_battle": (emu.frame - f0) / battles,
           "emulated_fps": (emu.frame - f0) / dt, "battler_ms": t["battler_ms"] / max(t["battler"], 1),
           "tactician_ms": t["tactician_ms"] / max(t["tactician"], 1),
           "agent_share": (t["battler_ms"] + t["tactician_ms"]) / 1000 / dt}
    print(json.dumps(res))
    if a.report:
        with open(a.report, "a") as f:
            f.write(json.dumps(res) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--rom", default=ROM)
        p.add_argument("--save", default=SAV)
        p.add_argument("--nice", type=int, default=5)

    p = sub.add_parser("play")
    common(p)
    p.add_argument("--modes", default="search,search,net", help="one battler mode per worker (net | search)")
    p.add_argument("--out", default="runs/real_play")
    p.add_argument("--hours", type=float, default=0.0, help="0: until STOP")
    p.add_argument("--resume", action="store_true", help="workers continue from their last run's savestate")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sims", type=int, default=512)
    p.add_argument("--threads", type=int, default=1, help="torch threads per worker")
    p.add_argument("--scene", default="off",
                   help="in-game Battle Scene of the runs: off (fastest) | on (animations, ~20%% slower), or one per "
                        "worker, e.g. on,on,off. A recording replays a run with the options it was played with")
    p.add_argument("--max-emulators", type=int, default=3)
    p.add_argument("--record", action=argparse.BooleanOptionalAction, default=True,
                   help="record the first full win (replay with the UI -> mp4)")
    p.add_argument("--record-all", action="store_true", help="record every full win, not only the first")
    p.add_argument("--record-speed", type=int, default=2)
    p.add_argument("--record-size", default="1920x1080")
    p.add_argument("--record-scene", choices=["same", "on", "off"], default="same")

    p = sub.add_parser("worker")
    common(p)
    p.add_argument("--wid", type=int, required=True)
    p.add_argument("--mode", choices=["net", "search"], default="search")
    p.add_argument("--out", default="runs/real_play")
    p.add_argument("--hours", type=float, default=0.0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sims", type=int, default=512)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--scene", choices=["off", "on"], default="off")
    p.add_argument("--max-runs", type=int, default=0)

    p = sub.add_parser("status")
    p.add_argument("--out", default="runs/real_play")

    for name in ("verify", "record", "replay"):
        p = sub.add_parser(name)
        common(p)
        p.add_argument("run", help="a run directory (<out>/w<k>/runs/run_NNNNN)")
        p.add_argument("--scene", choices=["same", "on", "off"], default="same",
                       help="same: the run's own options (exact); on/off: battle scene forced (checked on gRngValue)")
        p.add_argument("--until-wins", type=int, default=0, help="stop after that many wins (0: the whole log)")
        if name != "verify":
            p.add_argument("--speed", type=int, default=2, help="game speed (x real time)")
        if name == "record":
            p.add_argument("--video", required=True)
            p.add_argument("--from-wins", type=int, default=0,
                           help="fast-forward until that many wins, then record (35: round 6 only)")
            p.add_argument("--fps", type=int, default=30)
            p.add_argument("--size", default="1920x1080")
            p.add_argument("--hold", type=float, default=0.5, help="seconds of video the panels stay at each decision")
            p.add_argument("--codec", default="auto", help="auto (NVENC if available) | libx264 | h264_nvenc")
            p.add_argument("--report", default="")
            p.add_argument("--show", action="store_true", help="also show the window (default: headless)")

    p = sub.add_parser("bench")
    common(p)
    p.add_argument("--mode", choices=["net", "search"], default="net")
    p.add_argument("--battles", type=int, default=20)
    p.add_argument("--scene", choices=["off", "on"], default="off")
    p.add_argument("--render", action="store_true", help="draw every frame (as with a window)")
    p.add_argument("--sims", type=int, default=512)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--report", default="")

    a = ap.parse_args()
    if getattr(a, "out", None) and not os.path.isabs(a.out):
        a.out = os.path.join(ROOT, a.out)
    os.chdir(ROOT)
    return {"play": cmd_play, "worker": cmd_worker, "status": cmd_status, "verify": cmd_verify,
            "record": cmd_record, "replay": cmd_replay, "bench": cmd_bench}[a.cmd](a) or 0


if __name__ == "__main__":
    sys.exit(main())

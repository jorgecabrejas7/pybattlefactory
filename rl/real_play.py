"""alphazero_v3 playing the real game headless at full speed, with logs, savestates and exact replays
(scripts/play_real.py is the command line; docs/RL_DECISIONS.md §24).

A **run** = one Factory challenge streak, from the lobby until the streak is lost. Each worker process owns one
emulator (no rendering, uncapped, in-game options Text Speed = Fast, Battle Scene = Off by default, a cartridge
clock that follows the emulated frames) and plays runs one after another like a player: after a loss the game
puts the player back in the lobby and the next run starts there.

Per run, in <out>/w<k>/runs/<run id>/:
    start.state.z    the emulator's savestate in the lobby, just before the run starts (zlib)
    save.z           the cartridge's save memory at that moment (a savestate does not hold it)
    run.json         how to restart it exactly: reset arguments, options, clock, driver flags, agent and seeds
    actions.jsonl    every decision: phase, the action taken, gRngValue and frame when it was taken, streak,
                     battle number, and the agent's details (probabilities, search visits / Q, features, beliefs)
                     + one line per battle end; the raw action log the replay feeds back
    result.json      wins, why it ended
A run that reaches 42 wins (round 6 completed, gold Noland beaten) also gets FULL_WIN.json and a line in
<out>/full_wins.jsonl.

Replay (`replay_run`): a fresh emulator gets the savestate, the save memory, the clock and the same options;
EmuBackend.reset is called with the logged arguments and then fed the logged actions, never recomputed. Since
the emulator and the driver are deterministic, the game goes through exactly the same frames: the replay checks
the decision kind, gRngValue and the frame counter at every decision and each battle's outcome against the log.
"""

import json
import os
import random
import signal
import time
import zlib
from typing import Optional

from pybattle.backend import Phase

FULL_WIN = 42                       # 6 rounds x 7 battles: the gold Noland symbol is the 42nd win
RTC_EPOCH_MS = 1767268800000        # 2026-01-01 12:00 UTC: the cartridge clock at boot (then emulated time)
FAST_OPTIONS = {"text_speed": "fast", "battle_scene": False}
NO_RENDER = 1 << 30                 # frameskip: never draw (headless)


def plain(x):
    """JSON-friendly copy of an agent's details (tuples -> lists, numpy -> python)."""
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items() if k != "log"}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if hasattr(x, "tolist"):
        return x.tolist()
    if isinstance(x, float):
        return round(x, 6)
    return x


def action_from_json(phase: str, a):
    """A logged action back to the backend's form."""
    if a is None:
        return None
    if phase in ("battle", "forced_switch"):
        return (a[0], int(a[1])) if len(a) > 1 else (a[0],)
    return tuple(int(v) for v in a)


def _write_z(path, data: bytes):
    with open(path, "wb") as f:
        f.write(zlib.compress(data, 6))


def _read_z(path) -> bytes:
    with open(path, "rb") as f:
        return zlib.decompress(f.read())


def rng_frame(backend):
    from pybattle.emu.decode import SYMBOLS as S
    emu = backend.d.emu
    return emu.read32(S.addr("gRngValue")), int(emu.frame)


# ---- one run on disk -------------------------------------------------------------------------------------------

class RunWriter:
    def __init__(self, run_dir, backend, reset_kw, meta):
        os.makedirs(run_dir, exist_ok=True)
        self.dir = run_dir
        emu = backend.d.emu
        _write_z(os.path.join(run_dir, "start.state.z"), emu.save_state())
        _write_z(os.path.join(run_dir, "save.z"), emu.save_data())
        from pybattle.emu.options import read_options
        rng, frame = rng_frame(backend)
        self.meta = dict(meta, reset=reset_kw, options=read_options(emu), rng=rng, frame=frame,
                         outcome_reported=bool(getattr(backend.d, "_outcome_reported", False)),
                         started=time.time())
        json.dump(self.meta, open(os.path.join(run_dir, "run.json"), "w"), indent=1)
        self.f = open(os.path.join(run_dir, "actions.jsonl"), "w")
        self.n = 0

    def decision(self, backend, phase, info, choice, rng, frame):
        rec = {"i": self.n, "phase": phase.value, "action": plain(choice.action), "rng": rng, "frame": frame,
               "streak": info.win_streak, "battle": info.battle_in_challenge, "challenge": info.challenge_num,
               "noland": bool(info.noland), "wins": backend.wins, "reason": choice.reason,
               "details": plain(choice.details or {})}
        self.f.write(json.dumps(rec) + "\n")
        self.n += 1

    def battle_end(self, won, backend, rng, frame):
        self.f.write(json.dumps({"event": "battle_end", "won": bool(won), "wins": backend.wins, "rng": rng,
                                 "frame": frame}) + "\n")
        self.f.flush()

    def finish(self, result):
        self.f.flush()
        self.f.close()
        json.dump(dict(result, finished=time.time()), open(os.path.join(self.dir, "result.json"), "w"), indent=1)

    def mark(self, name, data):
        json.dump(data, open(os.path.join(self.dir, name), "w"), indent=1)


# ---- a worker: runs one after another ----------------------------------------------------------------------------

class Stop(Exception):
    pass


def make_backend(rom, save, rtc_epoch_ms=RTC_EPOCH_MS, options=FAST_OPTIONS, frameskip=NO_RENDER, state=None,
                 save_data=None, machine_wrap=None):
    """A headless emulator backend: booted from the save with `options`, or (state) restored in the lobby."""
    from pybattle import pybattle_emu as native
    from pybattle.emu.mirror import WatchedMachine
    from pybattle.emu_backend import EmuBackend
    emu = native.Emulator(rom, save)
    if not hasattr(emu, "set_frameskip") or not hasattr(emu, "save_data"):
        raise RuntimeError("pybattle_emu is too old for play_real (rebuild: cmake --build build --target pybattle_emu)")
    emu.set_frameskip(frameskip)
    emu.set_rtc_fake_epoch(int(rtc_epoch_ms))
    if state is not None:
        emu.load_state(state)
        if save_data is not None:
            emu.load_save_data(save_data)
    inner = machine_wrap(emu) if machine_wrap else emu
    m = WatchedMachine(inner)
    return EmuBackend(machine=m, boot=state is None, options=options if state is None else None)


def play_worker(wid, out, rom, save, mode="search", seed=0, hours=0.0, sims=512, threads=1, options=None,
                resume=True, max_runs=0, max_battles=0, log=print):
    """Play runs until `hours` elapse, <out>/STOP or <out>/w<wid>/STOP appears, or SIGTERM."""
    from rl.playlog_stub import play_logger_class
    from rl.real_agents import V3Player
    options = dict(FAST_OPTIONS if options is None else options)
    wdir = os.path.join(out, f"w{wid}")
    os.makedirs(os.path.join(wdir, "runs"), exist_ok=True)
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *a: stop.update(flag=True))
    stats_path = os.path.join(wdir, "stats.json")
    stats = json.load(open(stats_path)) if (resume and os.path.exists(stats_path)) else {
        "worker": wid, "mode": mode, "runs": 0, "battles": 0, "wins": 0, "full_wins": 0, "best": 0,
        "streaks": [], "seconds": 0.0, "errors": 0, "decisions": 0, "desyncs": 0, "search_fallbacks": 0}
    stats["mode"] = mode
    last = _last_run(wdir) if resume else None
    if last is not None:
        log(f"[w{wid}] resuming from the start of {os.path.basename(last)} (that run is played again)")
        be = make_backend(rom, save, json.load(open(os.path.join(last, "run.json")))["rtc_epoch_ms"],
                          state=_read_z(os.path.join(last, "start.state.z")),
                          save_data=_read_z(os.path.join(last, "save.z")))
        be.d._outcome_reported = json.load(open(os.path.join(last, "run.json")))["outcome_reported"]
        if not os.path.exists(os.path.join(last, "result.json")):
            json.dump({"abandoned": True}, open(os.path.join(last, "result.json"), "w"))
        first_reset = json.load(open(os.path.join(last, "run.json")))["reset"]
    else:
        be = make_backend(rom, save, options=options)
        # workers booted from the same save would start with the same overworld RNG: offset each by some frames
        be.d.emu.run_frames(1 + 977 * wid + 13 * seed)
        first_reset = {"win_streak": 0, "rents_count": 0}       # the first run starts at streak 0 whatever the save had
    player_seed = seed * 1000 + wid * 7919 + stats["runs"]
    player = V3Player(mode=mode, sims=sims, threads=threads, seed=player_seed)
    logger = play_logger_class()(os.path.join(wdir, "playlog.jsonl"),
                                 dict(player.describe(), worker=wid, options=options, rtc_epoch_ms=RTC_EPOCH_MS))
    t_start = time.time()
    sec0 = stats["seconds"]
    reset_kw = first_reset
    try:
        while True:
            if stop["flag"] or os.path.exists(os.path.join(out, "STOP")) or os.path.exists(os.path.join(wdir, "STOP")):
                break
            if hours and time.time() - t_start > hours * 3600:
                break
            if max_runs and stats["runs"] >= max_runs:
                break
            existing = [int(d[4:]) for d in os.listdir(os.path.join(wdir, "runs")) if d.startswith("run_")]
            run_id = f"run_{max(existing + [-1]) + 1:05d}"              # never overwrites (resumes, errors)
            run_dir = os.path.join(wdir, "runs", run_id)
            meta = {"worker": wid, "run": run_id, "mode": mode, "agent": player.describe(),
                    "rtc_epoch_ms": RTC_EPOCH_MS, "frameskip": NO_RENDER,
                    "search_rng_state": list(player.search.rng.getstate()[1][:4]) if player.search else None,
                    "planner_rng_seed": player_seed}
            # the reset arguments go to the replay as they are
            w = RunWriter(run_dir, be, {"win_streak": reset_kw.get("win_streak"),
                                        "rents_count": reset_kw.get("rents_count", 0)}, meta)
            t0 = time.time()
            try:
                result = play_run(be, player, w, logger, stop, wid, run_id, out, log, max_battles)
            except Stop:
                w.finish({"wins": be.wins, "end": "stopped"})
                break
            except Exception as e:                                   # driver timeouts etc.: restart from the lobby
                import traceback
                stats["errors"] += 1
                w.finish({"wins": be.wins, "end": "error", "error": f"{type(e).__name__}: {e}",
                          "traceback": traceback.format_exc()[-3000:]})
                log(f"[w{wid}] {run_id} error: {type(e).__name__}: {e}; restarting from this run's start")
                be = make_backend(rom, save, RTC_EPOCH_MS, state=_read_z(os.path.join(run_dir, "start.state.z")),
                                  save_data=_read_z(os.path.join(run_dir, "save.z")))
                # play a different run from here: let the overworld RNG move on
                be.d.emu.run_frames(1 + stats["errors"] % 61)
                stats["runs"] += 1
                reset_kw = w.meta["reset"]
                continue
            stats["runs"] += 1
            stats["battles"] += result["battles"]
            stats["wins"] += result["wins"]
            stats["decisions"] += result["decisions"]
            stats["streaks"].append(result["wins"])
            stats["best"] = max(stats["best"], result["wins"])
            stats["full_wins"] += result["wins"] >= FULL_WIN
            stats["desyncs"] = player.mirror.desyncs
            stats["search_fallbacks"] += result["search_fallbacks"]
            stats["seconds"] = sec0 + time.time() - t_start
            stats["battles_per_hour"] = stats["battles"] / max(stats["seconds"], 1) * 3600
            stats["timing"] = player.timing
            stats["updated"] = time.time()
            json.dump(stats, open(stats_path, "w"), indent=1)
            log(f"[w{wid}] {run_id}: {result['wins']} wins ({result['battles']} battles, {time.time() - t0:.0f} s) | "
                f"runs {stats['runs']} best {stats['best']} full wins {stats['full_wins']} | "
                f"{stats['battles_per_hour']:.0f} battles/h")
            _prune(wdir, keep=40)
            if result["end"] == "cut":                               # mid-challenge: not in the lobby any more
                break
            reset_kw = {"win_streak": None}                          # the game reset the streak itself
    finally:
        stats["seconds"] = sec0 + time.time() - t_start
        json.dump(stats, open(stats_path, "w"), indent=1)
        logger.close()
    return stats


def play_run(be, player, w, logger, stop, wid, run_id, out, log, max_battles=0):
    """One run from the lobby; -> result dict. max_battles: cut the run after that many battles (tests)."""
    kw = w.meta["reset"]
    be.reset(win_streak=kw["win_streak"], rents_count=kw.get("rents_count", 0), rewind=False)
    battles = decisions = fallbacks = 0
    full = False
    cut = False
    while be.phase != Phase.RUN_OVER:
        if max_battles and battles >= max_battles:
            cut = True
            break
        if stop["flag"] or os.path.exists(os.path.join(out, "STOP")) or \
                os.path.exists(os.path.join(out, f"w{wid}", "STOP")):
            raise Stop()
        phase = be.phase
        view, info = be.view(), be.run_info()
        choice = player.act(be, phase, view, info)
        rng, frame = rng_frame(be)
        w.decision(be, phase, info, choice, rng, frame)
        policy, extras = (choice.details or {}).get("log", ({}, {}))
        logger.decision(phase.value, view, info, policy, choice.action, dict(extras, run=run_id, worker=wid))
        fallbacks += bool((choice.details or {}).get("search_fallback")) and phase != Phase.RENTAL
        decisions += 1
        be.act(choice.action)
        if phase in (Phase.BATTLE, Phase.FORCED_SWITCH) and be.phase not in (Phase.BATTLE, Phase.FORCED_SWITCH):
            won = bool(be.last_battle_won)
            battles += 1
            rng, frame = rng_frame(be)
            w.battle_end(won, be, rng, frame)
            logger.battle_end(won, {"run": run_id, "wins": be.wins, "streak": be.run_info().win_streak,
                                    "worker": wid})
            if won and be.wins >= FULL_WIN and not full:
                full = True
                rec = {"worker": wid, "run": run_id, "dir": w.dir, "time": time.time(), "decisions": w.n,
                       "mode": player.mode}
                w.mark("FULL_WIN.json", rec)
                with open(os.path.join(out, "full_wins.jsonl"), "a") as f:
                    f.write(json.dumps(rec) + "\n")
                log(f"[w{wid}] *** FULL WIN: {run_id} reached {FULL_WIN} wins ***")
    logger.run_end(be.wins, {"run": run_id, "worker": wid, "battles": battles})
    res = {"wins": be.wins, "battles": battles, "decisions": decisions, "end": "cut" if cut else "lost", "full_win": full,
           "search_fallbacks": fallbacks}
    w.finish(res)
    return res


def _last_run(wdir):
    runs = sorted(d for d in os.listdir(os.path.join(wdir, "runs")) if d.startswith("run_"))
    for r in reversed(runs):
        p = os.path.join(wdir, "runs", r)
        if os.path.exists(os.path.join(p, "start.state.z")) and os.path.exists(os.path.join(p, "run.json")):
            return p
    return None


def _prune(wdir, keep=40):
    """Savestates take ~0.15 MB per run: keep them for the last `keep` runs, runs with >= 14 wins and full wins."""
    runs = sorted(d for d in os.listdir(os.path.join(wdir, "runs")) if d.startswith("run_"))
    for r in runs[:-keep]:
        p = os.path.join(wdir, "runs", r)
        try:
            res = json.load(open(os.path.join(p, "result.json")))
        except Exception:
            continue
        if res.get("wins", 0) >= 14 or os.path.exists(os.path.join(p, "FULL_WIN.json")):
            continue
        for f in ("start.state.z", "save.z"):
            try:
                os.remove(os.path.join(p, f))
            except FileNotFoundError:
                pass


# ---- replay --------------------------------------------------------------------------------------------------------

def load_run(run_dir):
    meta = json.load(open(os.path.join(run_dir, "run.json")))
    recs = [json.loads(line) for line in open(os.path.join(run_dir, "actions.jsonl")) if line.strip()]
    return meta, recs


def replay_run(run_dir, rom, save, machine_wrap=None, until_wins: Optional[int] = None, scene: str = "same",
               on_decision=None, on_battle_end=None, on_status=None, strict=True):
    """Replay a logged run from its savestate with the logged actions. machine_wrap(emu) -> the machine the backend
    drives (e.g. a ViewedMachine / RecordingMachine for the UI). until_wins: stop after that many wins (a full win's
    recording stops at 42). scene: "same" (the run's own options) or "on" / "off" (battle scene forced; the checks
    then compare gRngValue only, the frame counter changes). -> report dict (ok, mismatches, battles, ...)."""
    meta, recs = load_run(run_dir)
    state = _read_z(os.path.join(run_dir, "start.state.z"))
    sdata = _read_z(os.path.join(run_dir, "save.z"))
    be = make_backend(rom, save, meta["rtc_epoch_ms"], state=state, save_data=sdata, frameskip=0,
                      machine_wrap=machine_wrap)
    be.d._outcome_reported = meta.get("outcome_reported", False)
    same_frames = True
    if scene in ("on", "off"):
        from pybattle.emu.options import write_options
        want = scene == "on"
        if want != meta["options"]["battle_scene"]:
            write_options(be.d.emu, battle_scene=want)
            same_frames = False
    kw = meta["reset"]
    if on_status:
        on_status("walking to the attendant…")
    be.reset(win_streak=kw["win_streak"], rents_count=kw.get("rents_count", 0), rewind=False)
    if on_status:
        on_status("")
    report = {"run": run_dir, "decisions": 0, "battles": 0, "wins": 0, "mismatches": [], "scene": scene,
              "same_frames": same_frames, "outcomes": []}

    def mismatch(msg):
        report["mismatches"].append(msg)
        if strict:
            raise RuntimeError("replay diverged: " + msg)

    for rec in recs:
        if "event" in rec:
            if rec["event"] == "battle_end":
                won = bool(be.last_battle_won)
                report["battles"] += 1
                report["outcomes"].append(won)
                rng, frame = rng_frame(be)
                if won != rec["won"]:
                    mismatch(f"battle {report['battles']}: outcome {won} != logged {rec['won']}")
                if same_frames and (rng, frame) != (rec["rng"], rec["frame"]):
                    mismatch(f"battle {report['battles']} end: rng/frame {rng:08x}/{frame} != {rec['rng']:08x}/{rec['frame']}")
                if on_battle_end:
                    on_battle_end(won, be)
                if won:
                    report["wins"] += 1
                if until_wins is not None and report["wins"] >= until_wins:
                    break
            continue
        if be.phase.value != rec["phase"]:
            mismatch(f"decision {rec['i']}: phase {be.phase.value} != logged {rec['phase']}")
            break
        rng, frame = rng_frame(be)
        if rng != rec["rng"] or (same_frames and frame != rec["frame"]):
            mismatch(f"decision {rec['i']}: rng/frame {rng:08x}/{frame} != logged {rec['rng']:08x}/{rec['frame']}")
        action = action_from_json(rec["phase"], rec["action"])
        if on_decision:
            on_decision(be, rec, action)
        be.act(action)
        report["decisions"] += 1
    report["ok"] = not report["mismatches"]
    report["final_frame"] = int(be.d.emu.frame)
    return report


# ---- the window for replays / recordings -------------------------------------------------------------------------

class RecordingMachine:
    """The emulator drawn into a Viewer and written to a video at `game_speed` x real time, frame-exact (one video
    frame every 60 * game_speed / fps emulated frames, whatever the host speed). Same interface as Emulator."""

    def __init__(self, emu, viewer, writer=None, game_speed=2, fps=30):
        self.emu, self.viewer, self.writer = emu, viewer, writer
        self.every = max(1, round(60 * game_speed / fps))
        self.fps = fps
        self.n = 0
        self.video_frames = 0

    def run_frames(self, n: int = 1):
        for _ in range(n):
            self.emu.run_frames(1)
            self.n += 1
            if self.n % self.every == 0:
                self.viewer._capture(self.emu, 0.0)
                self.show()

    def show(self, repeat=1):
        v = self.viewer
        v.handle_events()
        v.check_quit()
        v.draw()
        if self.writer is not None:
            data = __import__("pygame").image.tobytes(v.screen, "RGB")
            for _ in range(repeat):
                self.writer.write(data)
                self.video_frames += 1

    def hold(self, seconds):
        """Keep the current picture for `seconds` of video (decision panels)."""
        k = int(round(seconds * self.fps))
        if k > 0:
            self.viewer._capture(self.emu, 0.0)                    # the picture at the decision itself
            self.show(k)

    def __getattr__(self, name):
        return getattr(self.emu, name)


class VideoWriter:
    """ffmpeg reading raw RGB frames on stdin: h264 (NVENC when the GPU has it, else libx264)."""

    def __init__(self, path, size, fps=30, codec="auto"):
        import subprocess
        w, h = size
        if codec == "auto":
            codec = "h264_nvenc" if _nvenc_ok() else "libx264"
        enc = (["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "24", "-b:v", "0"] if codec == "h264_nvenc"
               else ["-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-threads", "2"])
        self.cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                    "-r", str(fps), "-i", "-"] + enc + ["-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
        self.codec = codec
        self.p = subprocess.Popen(self.cmd, stdin=subprocess.PIPE)

    def write(self, data: bytes):
        self.p.stdin.write(data)

    def close(self):
        self.p.stdin.close()
        return self.p.wait()


def _nvenc_ok():
    import subprocess
    try:
        r = subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "color=black:s=256x144:d=0.2",
                            "-c:v", "h264_nvenc", "-f", "null", "-"], capture_output=True, timeout=30)
        return r.returncode == 0
    except Exception:
        return False


def record_run(run_dir, rom, save, video_path, until_wins=FULL_WIN, game_speed=2, fps=30, size=(1920, 1080),
               hold=0.5, scene="same", theme="dark", codec="auto", headless=True, log=print):
    """Replay a run with the full UI and write it to `video_path` (mp4). -> replay report (plus video info)."""
    if headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    from pybattle.emu.viewer import Scene, Viewer
    meta, _ = load_run(run_dir)
    mode = meta.get("mode", "")
    label = {"search": f"search {meta.get('agent', {}).get('search', {}).get('n_sims', 512)}",
             "net": "network"}.get(mode, mode)
    viewer = Viewer(speed=game_speed, size=size, theme=theme, beliefs_panel=True, mode_label=f"REPLAY · {label}")
    writer = VideoWriter(video_path, viewer.screen.get_size(), fps=fps, codec=codec)
    holder = {}

    def wrap(emu):
        holder["m"] = RecordingMachine(emu, viewer, writer, game_speed, fps)
        return holder["m"]

    class _Choice:
        def __init__(self, action, reason, details):
            self.action, self.reason, self.details = action, reason, details

    def on_decision(be, rec, action):
        from pybattle.emu_backend import Phase as P
        phase = be.phase
        view, info = be.view(), be.run_info()
        est = None
        try:
            from rl.damage import battle_estimates
            from pybattle.view import BattleView
            if isinstance(view, BattleView):
                est = battle_estimates(view, {"challenge": info.challenge_num, "battle": info.battle_in_challenge})
        except Exception:
            est = None
        viewer.set_scene(Scene(phase.value, view, info, _Choice(action, rec.get("reason", ""), rec.get("details")),
                               False, est))
        viewer.add_log(f"{phase.value}: {rec.get('reason', '')}")
        holder["m"].hold(hold)

    def on_battle_end(won, be):
        viewer.add_log("battle " + ("WON" if won else "LOST") + f" — run wins {be.wins}", "won" if won else "lost")

    viewer.set_run(1)
    viewer.add_log(f"=== replay of {os.path.basename(run_dir)} ({mode}) ===", "run")
    t0 = time.time()
    try:
        rep = replay_run(run_dir, rom, save, machine_wrap=wrap, until_wins=until_wins, scene=scene,
                         on_decision=on_decision, on_battle_end=on_battle_end, on_status=viewer.set_status)
        holder["m"].hold(3.0)
    finally:
        code = writer.close()
    rep.update(video=video_path, video_frames=holder["m"].video_frames if "m" in holder else 0,
               video_seconds=(holder["m"].video_frames if "m" in holder else 0) / fps, codec=writer.codec,
               ffmpeg_exit=code, wall_seconds=time.time() - t0, size=list(viewer.screen.get_size()))
    log(f"recorded {video_path}: {rep['video_seconds'] / 60:.1f} min of video in {rep['wall_seconds'] / 60:.1f} min")
    return rep

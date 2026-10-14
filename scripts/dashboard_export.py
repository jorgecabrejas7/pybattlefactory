#!/usr/bin/env python3
"""Export TensorBoard scalars of training runs to a compact metrics.json and push it to the
dashboard on the VPS (see docs/DASHBOARD.md).

    python scripts/dashboard_export.py [--runs alphazero_v2,alphazero_v1] [--no-upload]

Default runs: every directory in runs/ holding event files, newest first. Each series is
downsampled to at most --max-points points. The page (dashboard/index.html) is uploaded too, so
editing it locally is enough to deploy it.
Event files, configs and logs of all runs (not checkpoints) are mirrored to <remote-dir>/runs for
the VPS TensorBoard and file browser. Uploads are atomic: rsync to a temporary name, then
`ssh <host> mv`. The VPS never connects to this machine.
"""

import argparse
import glob
import json
import math
import os
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_PATTERN = "[r]l.alphazero"


def find_runs(runs_dir):
    dirs = [d for d in glob.glob(os.path.join(runs_dir, "*"))
            if os.path.isdir(d) and glob.glob(os.path.join(d, "events.out.tfevents*"))]
    dirs.sort(key=lambda d: max(os.path.getmtime(f) for f in glob.glob(os.path.join(d, "events.out.tfevents*"))),
              reverse=True)
    return [os.path.basename(d) for d in dirs]


def downsample(points, n):
    if len(points) <= n:
        return points
    stride = (len(points) - 1) / (n - 1)
    return [points[round(i * stride)] for i in range(n)]


def num(v):
    if math.isnan(v) or math.isinf(v):
        return None
    return float(f"{v:.5g}")


def training_processes():
    out = subprocess.run(["pgrep", "-af", PROC_PATTERN], capture_output=True, text=True).stdout
    return [line.split(" ", 1)[1] for line in out.splitlines() if " " in line]


def log_tail(runs_dir, run, n):
    for name in (f"{run}.log", f"{run}.train.log"):
        path = os.path.join(runs_dir, name)
        if os.path.exists(path):
            with open(path, "rb") as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - 16384))
                lines = f.read().decode("utf-8", "replace").splitlines()
            return lines[-n:]
    return []


def read_scalars(path, max_points, cache_dir):
    """All scalars of one run, from a per-run cache while its event files are unchanged."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    files = sorted(glob.glob(os.path.join(path, "events.out.tfevents*")))
    sig = [[os.path.basename(f), os.path.getsize(f), os.path.getmtime(f)] for f in files] + [max_points]
    cache = os.path.join(cache_dir, os.path.basename(path) + ".json")
    if os.path.exists(cache):
        with open(cache) as f:
            cached = json.load(f)
        if cached["sig"] == sig:
            return cached["scalars"], cached["last_wall"], cached["first_wall"]

    ea = EventAccumulator(path, size_guidance={"scalars": 0, "tensors": 0, "histograms": 1,
                                               "images": 1, "audio": 1, "compressedHistograms": 1})
    ea.Reload()
    scalars, last_wall, first_wall = {}, 0.0, None
    for tag in ea.Tags()["scalars"]:
        events = ea.Scalars(tag)
        pts = downsample([[e.step, num(e.value)] for e in events], max_points)
        scalars[tag] = pts
        last_wall = max(last_wall, events[-1].wall_time)
        first_wall = events[0].wall_time if first_wall is None else min(first_wall, events[0].wall_time)
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache, "w") as f:
        json.dump({"sig": sig, "scalars": scalars, "last_wall": last_wall, "first_wall": first_wall}, f)
    return scalars, last_wall, first_wall


def export_run(runs_dir, run, procs, is_newest, max_points, log_lines, cache_dir):
    path = os.path.join(runs_dir, run)
    scalars, last_wall, first_wall = read_scalars(path, max_points, cache_dir)
    config = {}
    if os.path.exists(os.path.join(path, "config.json")):
        with open(os.path.join(path, "config.json")) as f:
            config = json.load(f)

    # Alive: a training process whose command line names this run; if none names any run
    # (default --name is a timestamp), a running process is attributed to the newest run.
    alive = any(run in cmd for cmd in procs) or (is_newest and bool(procs) and not any(
        f"--name" in cmd for cmd in procs))
    it = scalars.get("train/iteration")
    return {
        "name": run,
        "alive": alive,
        "last_update": last_wall or None,
        "first_update": first_wall,
        "iteration": it[-1][1] if it else None,
        "iterations_planned": config.get("iterations"),
        "config": {k: config[k] for k in ("iterations", "decisions_per_iter", "sims", "dets", "lr",
                                          "workers", "opponent_prior") if k in config},
        "log_tail": log_tail(runs_dir, run, log_lines),
        "scalars": scalars,
    }


def upload(host, remote_dir, local_path, remote_name):
    tmp = f"{remote_dir}/.{remote_name}.{os.getpid()}.tmp"
    subprocess.run(["rsync", "-q", "--chmod=F644", local_path, f"{host}:{tmp}"], check=True, timeout=120)
    subprocess.run(["ssh", host, f"mv -f {tmp} {remote_dir}/{remote_name}"], check=True, timeout=60)


def sync_runs(host, remote_dir, runs_dir):
    """Mirror event files, configs and logs (no checkpoints) for the VPS TensorBoard and file browser."""
    subprocess.run(["rsync", "-rtq", "--delete", "--prune-empty-dirs", "--chmod=D755,F644",
                    "--include=*/", "--include=events.out.tfevents*", "--include=*.json", "--include=*.log",
                    "--exclude=*", runs_dir.rstrip("/") + "/", f"{host}:{remote_dir}/runs/"],
                   check=True, timeout=600)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default=None, help="comma-separated run names (default: all, newest first)")
    p.add_argument("--runs-dir", default=os.path.join(ROOT, "runs"))
    p.add_argument("--max-points", type=int, default=500)
    p.add_argument("--log-lines", type=int, default=15)
    p.add_argument("--out", default=os.path.join(ROOT, "dashboard", "metrics.json"))
    p.add_argument("--page", default=os.path.join(ROOT, "dashboard", "index.html"))
    p.add_argument("--host", default="contabo")
    p.add_argument("--remote-dir", default="/srv/rl-dashboard")
    p.add_argument("--no-upload", action="store_true")
    p.add_argument("--no-sync-runs", action="store_true", help="don't mirror event files and logs")
    args = p.parse_args()

    runs = args.runs.split(",") if args.runs else find_runs(args.runs_dir)
    procs = training_processes()
    data = {"generated": time.time(), "processes": len(procs), "runs": []}
    for i, run in enumerate(runs):
        try:
            data["runs"].append(export_run(args.runs_dir, run, procs, i == 0, args.max_points, args.log_lines,
                                            os.path.join(os.path.dirname(args.out), ".cache")))
        except Exception as e:  # one broken run must not stop the others
            print(f"skipping {run}: {e!r}", file=sys.stderr)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(args.out), suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    os.replace(tmp, args.out)
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1024:.0f} KiB, {len(data['runs'])} runs)")

    if not args.no_upload:
        upload(args.host, args.remote_dir, args.page, "index.html")
        upload(args.host, args.remote_dir, args.out, "metrics.json")
        if not args.no_sync_runs:
            sync_runs(args.host, args.remote_dir, args.runs_dir)
        print(f"uploaded to {args.host}:{args.remote_dir}")


if __name__ == "__main__":
    main()

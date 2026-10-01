#!/usr/bin/env bash
# Start alphazero_v3 on the real game in the background (detached; survives the terminal): 2 workers with the search
# (Battle Scene On: the video of a full win shows the animations) + 1 with the network alone (Battle Scene Off,
# fastest). Extra arguments go to `play_real.py play` (e.g. --resume). Stop: touch $OUT/STOP. Status:
# python scripts/play_real.py status --out $OUT
set -e
cd "$(dirname "$0")/.."
OUT=${OUT:-runs/real_play}
mkdir -p "$OUT"
setsid nohup nice -n 5 python scripts/play_real.py play --modes search,search,net --scene on,on,off --out "$OUT" "$@" \
    >> "$OUT/supervisor.out" 2>&1 < /dev/null &
echo "supervisor pid $! -> $OUT"

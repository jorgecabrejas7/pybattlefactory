#!/usr/bin/env bash
# Resume alphazero_v4 after a shutdown: training (from runs/alphazero_v4/latest.pt, same options) and TensorBoard,
# in the tmux session "factory". Safe to run twice: it does nothing if the training is already running.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=runs/alphazero_v4
if pgrep -f "[r]l.alphazero_v4 --name alphazero_v4" >/dev/null; then
    echo "alphazero_v4 is already running"; exit 0
fi
[ -f "$RUN/latest.pt" ] || { echo "no $RUN/latest.pt"; exit 1; }
PY=/home/apollo/anaconda3/bin/python
CMD="$PY -m rl.alphazero_v4 --name alphazero_v4 --team-eval runs/team_eval_v3/model.pt --t-imitation 0.1 \
--eval-every 3 --eval-search-every 15 --iterations 1500 --init-from $RUN/latest.pt"
tmux has-session -t factory 2>/dev/null || tmux new-session -d -s factory -n shell -c "$PWD"
tmux new-window -d -t factory -n az_v4 -c "$PWD" "$CMD 2>&1 | tee -a runs/alphazero_v4.log; exec zsh"
if ! pgrep -f "[t]ensorboard --logdir runs" >/dev/null; then
    tmux new-window -d -t factory -n tensorboard -c "$PWD" "$PY -m tensorboard.main --logdir runs --port 6006 --bind_all; exec zsh"
fi
echo "resumed alphazero_v4 from $RUN/latest.pt (tmux: factory:az_v4)"

#!/usr/bin/env bash
# At login: resume the pybattlefactory Claude Code conversation with Remote Control on, inside tmux session "claude"
# (attach with: tmux attach -t claude). Does nothing if that tmux session already exists.
SESSION_ID=6bf5ba15-32f2-4353-a976-fc1ddcd77605
cd /home/apollo/Dev/pybattlefactory || exit 1
sleep 20                                    # let the network come up
tmux has-session -t claude 2>/dev/null && exit 0
tmux new-session -d -s claude -c "$PWD" \
    "/home/apollo/.local/bin/claude --resume $SESSION_ID --remote-control pybattlefactory; exec zsh"

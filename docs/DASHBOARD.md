# Training dashboard

Live view of training runs: https://169-58-71-242.sslip.io — Google login, only the account in
`contabo:~/rl-oauth/emails.txt` gets in (30-day session).

- `/` — the dashboard (tabs: Overview, Rounds, Training, Search, Scalars = every TensorBoard scalar,
  grouped, with run overlay). Tap ⤢ on a chart for full screen.
- `/tb/` — TensorBoard, running on the VPS over the mirrored event files.
- `/files/` — browsable logs, `config.json` files and event files.

## How it works

Push only: the VPS never connects to this machine; the local TensorBoard is never exposed.

1. `scripts/dashboard_export.py` (this machine) reads the TensorBoard event files of every run in
   `runs/` (newest first; `--runs a,b` to choose), downsamples each scalar to ≤ 500 points, adds the
   iteration, last-update time, the tail of `runs/<run>.log` and whether a training process is alive
   (`pgrep -f "[r]l.alphazero"`), and writes `dashboard/metrics.json`. Unchanged runs come from a
   cache in `dashboard/.cache/`. It then uploads `dashboard/index.html` and `metrics.json` to
   `contabo:/srv/rl-dashboard` atomically (rsync to a temp name, then `ssh contabo mv`). Editing
   `dashboard/index.html` locally is therefore enough to deploy it.
2. A systemd **user** timer runs the exporter every 2 minutes at nice 19 / idle I/O.
3. On the VPS, Caddy (official apt repo) serves `/srv/rl-dashboard` over HTTPS (Let's Encrypt, via
   sslip.io). Every request passes `forward_auth` to oauth2-proxy (Google, `127.0.0.1:4180`,
   service `rl-oauth2-proxy`, binary + config in `~/rl-oauth/`). TensorBoard (service
   `rl-tensorboard`, venv `~/rl-tb`, `127.0.0.1:6006`, ≤ 50 % CPU / 1 GB) reads
   `/srv/rl-dashboard/runs`, which the exporter mirrors (event files, configs, logs; no checkpoints).
   ufw allows 80/443. Setup: `scripts/dashboard_vps_setup.sh`, then `scripts/dashboard_vps_oauth.sh`.
   The Google OAuth client (project `rl-dashboard`, Testing mode) lives in your Google Cloud console.
4. The page polls `metrics.json` every 60 s. "Last update" turns red after 10 minutes without new
   events.

## Timer

```
systemctl --user status rl-dashboard-export.timer     # state, next run
journalctl --user -u rl-dashboard-export -n 20        # exporter output
systemctl --user start rl-dashboard-export.service    # export now
systemctl --user disable --now rl-dashboard-export.timer   # stop
systemctl --user enable --now rl-dashboard-export.timer    # start again
```

Units: `~/.config/systemd/user/rl-dashboard-export.{service,timer}`. They run while you are logged
in; `loginctl enable-linger apollo` keeps them running after logout.

To show only some runs, add `--runs name1,name2` to `ExecStart` in the service, then
`systemctl --user daemon-reload`.

## Who can log in

Edit `contabo:~/rl-oauth/emails.txt` (one Google address per line; reloaded automatically) and add
the address as a test user in the Google OAuth consent screen.

## Remove everything

This machine:
```
systemctl --user disable --now rl-dashboard-export.timer
rm ~/.config/systemd/user/rl-dashboard-export.{service,timer}
systemctl --user daemon-reload
rm -rf dashboard/metrics.json dashboard/.cache
```

VPS (Minecraft and its rules are not affected):
```
ssh -t contabo 'sudo ufw delete allow 80/tcp; sudo ufw delete allow 443/tcp;
  sudo systemctl disable --now caddy rl-tensorboard rl-oauth2-proxy; sudo apt-get purge -y caddy;
  sudo rm -f /etc/systemd/system/rl-tensorboard.service /etc/systemd/system/rl-oauth2-proxy.service;
  sudo systemctl daemon-reload; rm -rf ~/rl-tb ~/rl-oauth ~/rl-dashboard-oauth.sh;
  sudo rm -rf /etc/caddy /var/lib/caddy /srv/rl-dashboard
    /etc/apt/sources.list.d/caddy-stable.list /usr/share/keyrings/caddy-stable-archive-keyring.gpg;
  sudo apt-get update; rm -f ~/rl-dashboard-setup.sh'
```
Then delete the OAuth client / project in the Google Cloud console. (`~/.local/bin/uv` was also
installed for the TensorBoard venv; `rm ~/.local/bin/uv ~/.local/bin/uvx` if unwanted.)

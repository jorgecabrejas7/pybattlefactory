#!/usr/bin/env bash
# Second VPS setup step for the training dashboard (see docs/DASHBOARD.md): replaces basic auth by
# Google login (oauth2-proxy, one allowed account) and adds TensorBoard (/tb/) and a file browser of
# the mirrored runs (/files/). Run on the VPS as root, after scripts/dashboard_vps_setup.sh:
#     sudo bash dashboard_vps_oauth.sh <hostname> <owner-user>
# Expects, owned by <owner-user>: ~/rl-oauth/oauth2-proxy (binary), ~/rl-oauth/oauth2-proxy.cfg,
# ~/rl-oauth/emails.txt, ~/rl-tb (venv with tensorboard). Both services listen on 127.0.0.1 only.
set -euo pipefail
HOST=$1 OWNER=$2
HOME_DIR=$(getent passwd "$OWNER" | cut -d: -f6)
WEBROOT=/srv/rl-dashboard

test -x "$HOME_DIR/rl-oauth/oauth2-proxy" && test -s "$HOME_DIR/rl-oauth/oauth2-proxy.cfg"
test -x "$HOME_DIR/rl-tb/bin/tensorboard"

cat > /etc/systemd/system/rl-tensorboard.service <<EOF
[Unit]
Description=TensorBoard for the training dashboard (localhost only)
After=network.target

[Service]
User=$OWNER
ExecStart=$HOME_DIR/rl-tb/bin/tensorboard --logdir $WEBROOT/runs --host 127.0.0.1 --port 6006 --path_prefix /tb --reload_interval 120
Nice=10
CPUQuota=50%
MemoryMax=1G
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF

cat > /etc/systemd/system/rl-oauth2-proxy.service <<EOF
[Unit]
Description=Google login for the training dashboard (localhost only)
After=network.target

[Service]
User=$OWNER
ExecStart=$HOME_DIR/rl-oauth/oauth2-proxy --config $HOME_DIR/rl-oauth/oauth2-proxy.cfg
MemoryMax=200M
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF

NEW=$(mktemp)
cat > "$NEW" <<EOF
# Training dashboard: Google login via oauth2-proxy, then the static page (pushed by
# scripts/dashboard_export.py), TensorBoard at /tb/ and the mirrored runs at /files/.
$HOST {
	encode zstd gzip
	header X-Robots-Tag "noindex, nofollow"

	handle /oauth2/* {
		reverse_proxy 127.0.0.1:4180
	}

	handle {
		forward_auth 127.0.0.1:4180 {
			uri /oauth2/auth
			@error status 401
			handle_response @error {
				redir * /oauth2/sign_in?rd={scheme}://{host}{uri}
			}
		}

		redir /tb /tb/
		redir /files /files/
		handle /tb/* {
			reverse_proxy 127.0.0.1:6006
		}
		handle_path /files/* {
			root * $WEBROOT/runs
			@log path *.log
			header @log Content-Type "text/plain; charset=utf-8"
			file_server browse
		}
		handle {
			root * $WEBROOT
			header /metrics.json Cache-Control "no-store"
			file_server
		}
	}
}
EOF
echo "== Caddyfile diff"; diff -u /etc/caddy/Caddyfile "$NEW" || true
caddy validate --adapter caddyfile --config "$NEW"

systemctl daemon-reload
systemctl enable --now rl-tensorboard rl-oauth2-proxy
sleep 3
systemctl is-active rl-tensorboard rl-oauth2-proxy
install -m 644 "$NEW" /etc/caddy/Caddyfile; rm -f "$NEW"
systemctl reload caddy

echo "== listening sockets"; ss -tlnp

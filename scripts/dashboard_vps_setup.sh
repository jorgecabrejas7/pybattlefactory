#!/usr/bin/env bash
# One-time VPS setup for the training dashboard (see docs/DASHBOARD.md). Run on the VPS as root:
#     sudo bash dashboard_vps_setup.sh <hostname> <owner-user> <bcrypt-hash-file>
# Installs Caddy from its official apt repository, serves /srv/rl-dashboard at https://<hostname>
# behind HTTP basic auth (user "rl"), and allows 80/443 in ufw if ufw is active. It touches nothing
# else: no other service, port, ufw rule or package.
set -euo pipefail
HOST=$1 OWNER=$2 HASH=$(cat "$3")
WEBROOT=/srv/rl-dashboard

echo "== before: listening sockets"; ss -tlnp
echo "== before: ufw"; ufw status verbose || true

if ! command -v caddy >/dev/null; then
  apt-get install -y debian-keyring debian-archive-keyring apt-transport-https curl gnupg
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
    | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
    > /etc/apt/sources.list.d/caddy-stable.list
  chmod o+r /usr/share/keyrings/caddy-stable-archive-keyring.gpg /etc/apt/sources.list.d/caddy-stable.list
  apt-get update
  apt-get install -y caddy
fi

install -d -o "$OWNER" -g "$OWNER" -m 755 "$WEBROOT"

NEW=$(mktemp)
cat > "$NEW" <<EOF
# Training dashboard: static files pushed by scripts/dashboard_export.py
$HOST {
	root * $WEBROOT
	basic_auth {
		rl $HASH
	}
	encode zstd gzip
	header X-Robots-Tag "noindex, nofollow"
	header /metrics.json Cache-Control "no-store"
	file_server
}
EOF
echo "== Caddyfile diff"; diff -u /etc/caddy/Caddyfile "$NEW" || true
caddy validate --adapter caddyfile --config "$NEW"
install -m 644 "$NEW" /etc/caddy/Caddyfile; rm -f "$NEW"
systemctl enable caddy
systemctl reload caddy || systemctl restart caddy

if ufw status | grep -q "Status: active"; then
  ufw allow 80/tcp comment 'rl-dashboard (caddy)'
  ufw allow 443/tcp comment 'rl-dashboard (caddy)'
fi

echo "== after: ufw"; ufw status verbose || true
echo "== after: listening sockets"; ss -tlnp
systemctl --no-pager status caddy | head -5

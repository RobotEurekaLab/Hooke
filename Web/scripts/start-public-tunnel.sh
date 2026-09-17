#!/usr/bin/env bash
# Expose the local Hooke webui (Hooke/webui/server.py, normally
# http://localhost:8080) to the public internet via a Cloudflare quick
# tunnel, so the "Simulation" card on the Robot Scientists landing page
# (Web/index.html) has something real to link to.
#
# This is a QUICK tunnel, not a permanent one: it needs no Cloudflare
# account, but every time it (re)starts it gets a brand-new random
# "https://<random-words>.trycloudflare.com" hostname. There is no way
# to get a stable, human-chosen public hostname without binding some
# account of yours (Cloudflare + a domain, ngrok's free static domain,
# Tailscale Funnel, ...) -- see the discussion in the repo history around
# this script's own commit if you want to set one of those up instead.
#
# Usage:
#   Web/scripts/start-public-tunnel.sh [port]
#
# Prints the tunnel URL once it's up. After it prints, update the
# Simulation card's href in Web/index.html to match, then commit+push
# so the live landing page points at the new link.

set -euo pipefail

PORT="${1:-8080}"
LOG_FILE="$(mktemp)"

if ! curl -s -o /dev/null "http://localhost:${PORT}/"; then
  echo "Nothing is listening on localhost:${PORT} yet." >&2
  echo "Start the webui first, e.g. from the Hooke/ simulator directory:" >&2
  echo "  export MUJOCO_GL=egl && python -m webui.server --host 0.0.0.0 --port ${PORT}" >&2
  exit 1
fi

if ! command -v cloudflared >/dev/null 2>&1; then
  echo "cloudflared is not installed. See https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/" >&2
  exit 1
fi

echo "Starting a Cloudflare quick tunnel to http://localhost:${PORT} ..."
nohup cloudflared tunnel --url "http://localhost:${PORT}" >"${LOG_FILE}" 2>&1 &
TUNNEL_PID=$!

echo "Tunnel process PID: ${TUNNEL_PID} (log: ${LOG_FILE})"
echo "Waiting for the public URL..."

for _ in $(seq 1 30); do
  URL="$(grep -o 'https://[a-zA-Z0-9.-]*\.trycloudflare\.com' "${LOG_FILE}" 2>/dev/null | head -1 || true)"
  if [ -n "${URL}" ]; then
    echo ""
    echo "Public URL: ${URL}"
    echo ""
    echo "Update Web/index.html's Simulation card href to this URL, then commit+push."
    exit 0
  fi
  sleep 1
done

echo "Timed out waiting for the tunnel URL -- check ${LOG_FILE}." >&2
exit 1

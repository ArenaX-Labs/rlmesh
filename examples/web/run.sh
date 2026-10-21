#!/usr/bin/env bash
# Runs the browser example: the example robot (a session host, the arm7 robot and its model policy)
# and rlmesh-gateway serving this directory's build. `mise run demo:web` builds the page first.
#
# Env: ROBOT_BASE (per-class TCP base, default 17600; uses 4 ports), HTTP_PORT (default 18080),
# CAM=jpeg|raw, POLICY=on|off, DURATION (s, default 3600), BIND (default 127.0.0.1; anything else
# needs RLMESH_GATEWAY_TOKEN, then open http://HOST:PORT/#token=...), SKIP_BUILD=1.
# Logs go to ${XDG_CACHE_HOME:-~/.cache}/rlmesh-web/.
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -f examples/web/dist/index.html ] || { echo "examples/web/dist missing: run 'mise run build:js --demo'" >&2; exit 1; }
[ "${SKIP_BUILD:-}" = 1 ] || cargo build --release -p rlmesh-session --bin rlmesh-gateway --example web_robot
target="$(cargo metadata --format-version 1 --no-deps | python3 -c 'import json, sys; print(json.load(sys.stdin)["target_directory"])')/release"
logs="${XDG_CACHE_HOME:-$HOME/.cache}/rlmesh-web"
mkdir -p "$logs"
rb=${ROBOT_BASE:-17600}; hp=${HTTP_PORT:-18080}; dur=${DURATION:-3600}
rm -f "$logs/session.id"
timeout "$dur" "$target/examples/web_robot" --listen-base "127.0.0.1:$rb" --session-file "$logs/session.id" \
  --cam "${CAM:-jpeg}" --policy "${POLICY:-on}" >"$logs/robot.log" 2>&1 &
robot=$!
gateway=
trap 'kill $robot $gateway 2>/dev/null || true' EXIT INT TERM
for _ in $(seq 100); do [ -s "$logs/session.id" ] && break; sleep 0.1; done
[ -s "$logs/session.id" ] || { cat "$logs/robot.log" >&2; exit 1; }
timeout "$dur" "$target/rlmesh-gateway" --connect "127.0.0.1:$rb" --session-file "$logs/session.id" --max-priority operator \
  --bind "${BIND:-127.0.0.1}" --port "$hp" --static examples/web/dist >"$logs/gateway.log" 2>&1 &
gateway=$!
echo "open http://127.0.0.1:$hp/   (logs in $logs; Ctrl-C to stop)"
wait

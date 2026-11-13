#!/usr/bin/env python3
"""Client for the in-editor RLMesh demo control server.

Suite apps/gyms call :func:`ensure_demo` so Unreal loads the matching scenario
before ``RemoteEnv`` connects. The CLI remains for debugging::

  python integrations/unreal/demos/common/demo_client.py list
  python integrations/unreal/demos/common/demo_client.py status
  python integrations/unreal/demos/common/demo_client.py ensure chase_point
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
import time
import urllib.error
import urllib.request
from typing import Any

DEFAULT_BASE = "http://127.0.0.1:8878"
_DEFAULT_SERVE_HOST = "127.0.0.1"
_DEFAULT_SERVE_PORT = 7447
_SERVE_ADDR_RE = re.compile(
    r"^(?:rlmesh://)?(?P<host>\[[^\]]+\]|[^:#/]+)?(?::(?P<port>\d+))?",
    re.IGNORECASE,
)

_CONTROL_HINT = (
    "Open the Unreal editor with the RLMesh plugin (demo control starts "
    "automatically on Play), or start the demo control server from the "
    "project's Scripts folder in the Output Log."
)


class DemoControlError(RuntimeError):
    """Control server unreachable or returned an error."""


def request(
    method: str,
    path: str,
    *,
    base: str = DEFAULT_BASE,
    timeout: float = 130.0,
) -> dict[str, Any]:
    """HTTP JSON request to the demo control server. Raises DemoControlError."""
    url = f"{base.rstrip('/')}{path}"
    req = urllib.request.Request(url, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            payload = {"error": body}
        payload["_http_status"] = exc.code
        if payload.get("ok") is False or exc.code >= 400:
            err = payload.get("error") or payload.get("status", {}).get("error") or body
            raise DemoControlError(f"demo control {method} {path} failed: {err}") from exc
        return payload
    except urllib.error.URLError as exc:
        raise DemoControlError(
            f"cannot reach demo control at {url}\n  {exc}\n{_CONTROL_HINT}"
        ) from exc


def get_status(*, base: str = DEFAULT_BASE) -> dict[str, Any]:
    return request("GET", "/v1/status", base=base, timeout=10.0)


def load_demo(demo_id: str, *, base: str = DEFAULT_BASE) -> dict[str, Any]:
    return request("POST", f"/v1/demos/{demo_id}/load", base=base)


def play(*, base: str = DEFAULT_BASE) -> dict[str, Any]:
    return request("POST", "/v1/play", base=base)


def stop(*, base: str = DEFAULT_BASE) -> dict[str, Any]:
    return request("POST", "/v1/stop", base=base)


def serve_endpoint(
    address: str | None = None,
) -> tuple[str, int]:
    """Host/port Unreal EnvServer should listen on (from addr or defaults)."""
    raw = (address or os.environ.get("RLMESH_UNREAL_ADDR", "")).strip()
    if not raw:
        return _DEFAULT_SERVE_HOST, _DEFAULT_SERVE_PORT
    # Strip invite key / participant: host:port#keys/env → host:port
    base = raw.split("#", 1)[0].split("/", 1)[0]
    match = _SERVE_ADDR_RE.match(base)
    if not match:
        return _DEFAULT_SERVE_HOST, _DEFAULT_SERVE_PORT
    host = (match.group("host") or _DEFAULT_SERVE_HOST).strip("[]") or _DEFAULT_SERVE_HOST
    port_s = match.group("port")
    port = int(port_s) if port_s else _DEFAULT_SERVE_PORT
    return host, port


def serving_reachable(
    *,
    address: str | None = None,
    timeout: float = 0.4,
) -> bool:
    """True when something accepts TCP on the EnvServer port (usually Serving)."""
    host, port = serve_endpoint(address)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def wait_for_serving(
    *,
    address: str | None = None,
    timeout_sec: float = 30.0,
    poll_sec: float = 0.5,
) -> bool:
    """Poll until EnvServer accepts connections, or ``timeout_sec`` elapses."""
    deadline = time.monotonic() + max(0.0, timeout_sec)
    while True:
        if serving_reachable(address=address):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(max(0.05, poll_sec))


def ensure_demo(
    demo_id: str,
    *,
    base: str = DEFAULT_BASE,
    wait_playing_sec: float = 30.0,
    poll_sec: float = 0.5,
    force_replay: bool = False,
) -> dict[str, Any]:
    """Load ``demo_id`` and begin Play if needed; return final status.

    Skips load/play only when this demo is already playing **and** the EnvServer
    port accepts TCP. After Ctrl-C or a failed ``Serving`` start, status can stay
    ``playing`` with nothing on :7447 — that case stops and Plays again.
    """
    demo_id = demo_id.strip()
    if not demo_id:
        raise ValueError("demo_id is required")

    # Suite .env may already be loaded; still allow addr from the process env.
    status = get_status(base=base)
    already = status.get("demo_id") == demo_id and status.get("state") == "playing"
    if already and not force_replay and serving_reachable():
        return status

    if status.get("state") == "playing" or force_replay:
        # Bounce PIE so EnvServer can bind again (stale Play after interrupt /
        # "Could not serve … session host" leaves control state lying).
        try:
            stop(base=base)
        except DemoControlError:
            pass
        # Let EndPlay finish releasing :7447 before load/play.
        dead_deadline = time.monotonic() + min(10.0, wait_playing_sec)
        while serving_reachable() and time.monotonic() < dead_deadline:
            time.sleep(poll_sec)

    load_result = load_demo(demo_id, base=base)
    if load_result.get("ok") is False:
        raise DemoControlError(
            f"load {demo_id!r} failed: {load_result.get('error') or load_result}"
        )

    play_result = play(base=base)
    if play_result.get("ok") is False:
        raise DemoControlError(
            f"play after load {demo_id!r} failed: {play_result.get('error') or play_result}"
        )

    # PostPIEStarted reloads the control server and can reset /v1/status; wait
    # until EnvServer actually accepts TCP (LogRLMesh Serving …).
    if not wait_for_serving(timeout_sec=wait_playing_sec, poll_sec=poll_sec):
        host, port = serve_endpoint()
        raise DemoControlError(
            f"Play started for {demo_id!r} but nothing accepted TCP on "
            f"{host}:{port} within {wait_playing_sec:.0f}s "
            "(check LogRLMesh for Serving / Could not serve)"
        )

    last: dict[str, Any] = play_result.get("status") or {}
    try:
        polled = get_status(base=base)
        if polled.get("demo_id") == demo_id:
            last = polled
    except DemoControlError:
        pass
    if last.get("demo_id") != demo_id:
        last = {
            "server": "ok",
            "state": "playing",
            "demo_id": demo_id,
            "env_id": last.get("env_id"),
            "message": f"play requested for {demo_id}",
            "error": None,
        }
    return last


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        default=DEFAULT_BASE,
        help=f"Control server base URL (default {DEFAULT_BASE})",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("health", help="GET /v1/health")
    sub.add_parser("list", help="GET /v1/demos")
    sub.add_parser("status", help="GET /v1/status")
    load_p = sub.add_parser("load", help="POST /v1/demos/<id>/load")
    load_p.add_argument("demo_id", help="reach_target | chase_point | …")
    sub.add_parser("play", help="POST /v1/play")
    sub.add_parser("stop", help="POST /v1/stop")
    ensure_p = sub.add_parser("ensure", help="load + play for demo_id if needed")
    ensure_p.add_argument("demo_id", help="reach_target | chase_point | …")

    args = parser.parse_args(argv)
    base = args.base.rstrip("/")

    try:
        if args.cmd == "health":
            data = request("GET", "/v1/health", base=base, timeout=10.0)
        elif args.cmd == "list":
            data = request("GET", "/v1/demos", base=base, timeout=10.0)
        elif args.cmd == "status":
            data = get_status(base=base)
        elif args.cmd == "load":
            data = load_demo(args.demo_id, base=base)
        elif args.cmd == "play":
            data = play(base=base)
        elif args.cmd == "stop":
            data = stop(base=base)
        elif args.cmd == "ensure":
            data = ensure_demo(args.demo_id, base=base)
        else:
            parser.error(f"unknown command {args.cmd}")
            return 2
    except DemoControlError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(data, indent=2))
    if data.get("ok") is False or data.get("_http_status", 200) >= 400:
        return 1
    if args.cmd == "load":
        status = data.get("status") or {}
        print(
            f"\nLoaded demo={status.get('demo_id')} env_id={status.get('env_id')}. "
            "Suite clients call ensure_demo automatically; or: demo_client.py play",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

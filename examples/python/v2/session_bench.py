"""Python in the loop: the bench2 T1/T2 shape through ``rlmesh.v2``.

The Rust ``session_bench`` (``crates/rlmesh-edge/examples``) on the same
shape, with Python on both sides: an embodiment process (the session host,
a Python robot publishing force/torque at 1 kHz and an optional 640x480 raw
camera at 20 Hz, ticking its edge every 1 ms) and a controller process (a
Python policy replanning every 10 ms with chunks of 1 ms actions, through
``Policy.run``). Every timestamp is ``CLOCK_MONOTONIC``, shared by both
processes. Its Rust baseline is ``session_bench both --controller bound``,
the same bound policy with a Rust predictor and a Rust robot.

    python examples/python/v2/session_bench.py both --camera raw --duration-s 30

The result is one JSON object:

- ``control_per_s``, ``miss_pct``: Driving ticks per second at the robot,
  and ticks not Driving, in the measurement window;
- ``chunk_age_ms``: chunk receipt at the edge minus the capture of the
  newest ft sample it was computed from (p50/p99/p99.9/max), split into
  ``snapshot_age_ms`` (that sample's age at the policy's snapshot: the
  contract's sampling) and ``transit_ms`` (snapshot to receipt: inference,
  packing and the link);
- ``tick_work_us``: the robot's Python work per tick (publish + due);
- ``late_ticks_over_1ms``, ``missed_periods``: robot loop jitter;
- ``controller``: the policy's run counters.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import rlmesh.v2 as rlmesh

ROBOT = "ur10"
RING = 1 << 16


def robot_map(args: argparse.Namespace) -> dict[str, Any]:
    parts: dict[str, Any] = {
        "cmd": {"kind": "x/wrench_cmd", "hz": 1000, "fields": {"wrench": ["float32", [6]]},
                "command": "wrench", "priority": "control", "hold_max_ms": 100, "safe": "stop"},
        "ft": {"kind": "x/ft", "hz": 1000, "fields": {"wrench": ["float32", [18]]}, "priority": "realtime"},
    }  # fmt: skip
    if args.camera == "raw":
        parts["camera"] = {"kind": "camera", "hz": args.camera_hz, "width": 640, "height": 480,
                           "max_age_ms": 100, "send": f"latest(hz={args.camera_hz})"}  # fmt: skip
    return parts


def policy_map(args: argparse.Namespace) -> dict[str, Any]:
    inputs: dict[str, Any] = {
        "ft": {"kind": "x/ft", "at": "ft", "fields": {"wrench": ["float32", [18]]}}
    }
    if args.camera == "raw":
        inputs["camera"] = {"kind": "camera", "at": "camera"}
    return {
        "hz": 1000 / args.replan_ms,
        "inputs": inputs,
        "actions": {"cmd": {"kind": "x/wrench_cmd", "at": "cmd", "command": "wrench",
                            "fields": {"wrench": ["float32", [6]]}, "hz": 1000, "horizon": args.horizon}},
    }  # fmt: skip


def percentiles(ns: list[int]) -> dict[str, float] | None:
    if not ns:
        return None
    ordered = np.sort(np.asarray(ns, dtype=np.int64))

    def at(q: float) -> float:
        rank = min(max(int(np.ceil(q * len(ordered))), 1), len(ordered))
        return float(ordered[rank - 1]) / 1e6

    return {
        "n": len(ordered),
        "p50": at(0.5),
        "p99": at(0.99),
        "p999": at(0.999),
        "max": at(1.0),
    }


def embodiment(args: argparse.Namespace, host: rlmesh.Host | str) -> dict[str, Any]:
    robot = rlmesh.Robot(ROBOT, robot_map(args), session=host)
    robot.arm()
    ft_capture = np.zeros(RING, dtype=np.int64)
    stop = threading.Event()

    def cameras() -> None:
        frames = [np.full((480, 640, 3), t * 16, dtype=np.uint8) for t in range(16)]
        rate = rlmesh.Rate(args.camera_hz)
        i = 0
        while not stop.is_set():
            robot.publish("camera", frames[i % len(frames)])
            i += 1
            rate.sleep()

    camera = threading.Thread(target=cameras) if args.camera == "raw" else None
    if camera is not None:
        camera.start()

    ticks: list[
        tuple[int, int, str, int | None]
    ] = []  # capture, lateness, mode, ft seq
    chunks: dict[
        tuple[int, int], tuple[int, int, int]
    ] = {}  # (token, seq) -> capture, receipt, t0
    work: list[int] = []
    value = np.zeros(18, dtype=np.float32)
    rate = rlmesh.Rate(1000)
    start = time.monotonic_ns()
    end = start + int((args.warmup_s + args.duration_s) * 1e9) + 200_000_000
    late = rate.sleep()
    while True:
        capture = time.monotonic_ns()
        if capture >= end:
            break
        value[0] += 1.0
        seq = robot.publish("ft", value, capture_ns=capture)
        if seq is not None:
            ft_capture[seq % RING] = capture
        due = robot.due("cmd")
        work.append(time.monotonic_ns() - capture)
        based = None
        mode = "idle"
        if due is not None:
            mode = due.mode
            ft = next((r for r in due.based_on if r[0] == "ft"), None)
            if mode == "driving" and ft is not None:
                based = ft[2]
                key = (due.lease_token or 0, due.chunk_seq or 0)
                if (
                    key not in chunks
                    and due.received_ns is not None
                    and due.slot_ns is not None
                ):
                    t0 = due.slot_ns - (due.index or 0) * 1_000_000
                    chunks[key] = (int(ft_capture[based % RING]), due.received_ns, t0)
        ticks.append((capture, late, mode, based))
        late = rate.sleep()
    stop.set()
    if camera is not None:
        camera.join()
    edge = robot.edge_stats("cmd")
    time.sleep(1.0)  # let the controller see the stream end
    robot.close()

    anchor = ticks[0][0]
    win_start = anchor + int(args.warmup_s * 1e9)
    win_end = win_start + int(args.duration_s * 1e9)
    window = [t for t in ticks if win_start <= t[0] < win_end]
    in_window = [i for i, t in enumerate(ticks) if win_start <= t[0] < win_end]
    driving = sum(1 for t in window if t[2] == "driving")
    modes: dict[str, int] = {}
    for t in window:
        modes[t[2]] = modes.get(t[2], 0) + 1
    chunk_rows = [c for c in chunks.values() if win_start <= c[0] < win_end]
    return {
        "config": {
            key: getattr(args, key)
            for key in ("camera", "camera_hz", "horizon", "replan_ms", "duration_s")
        },
        "loadavg": Path("/proc/loadavg").read_text().strip(),
        "ticks": len(window),
        "control_per_s": driving / args.duration_s,
        "miss_pct": 100.0 * (len(window) - driving) / max(len(window), 1),
        "mode_ticks": modes,
        "loop_ms": percentiles(
            [t[0] - int(ft_capture[t[3] % RING]) for t in window if t[3] is not None]
        ),
        "chunk_age_ms": percentiles(
            [receipt - capture for capture, receipt, _ in chunk_rows]
        ),
        "snapshot_age_ms": percentiles([t0 - capture for capture, _, t0 in chunk_rows]),
        "transit_ms": percentiles([receipt - t0 for _, receipt, t0 in chunk_rows]),
        "tick_work_us": {
            key: value * 1e3
            for key, value in (percentiles([work[i] for i in in_window]) or {}).items()
            if key != "n"
        },
        "late_ticks_over_1ms": sum(1 for t in window if t[1] > 1_000_000),
        "missed_periods": rate.missed,
        "edge": edge,
    }


def controller(args: argparse.Namespace) -> dict[str, Any]:
    horizon = args.horizon

    def predict(obs: rlmesh.Observation) -> dict[str, Any]:
        # The "model": hold the first six floats of the wrench.
        return {"cmd": np.broadcast_to(obs["ft"][:6], (horizon, 6))}

    with rlmesh.Policy(policy_map(args), session=args.address, name="policy") as policy:
        policy.bind(ROBOT)
        stats = policy.run(predict, duration=args.warmup_s + args.duration_s + 2.0)
        return {"run": stats, "events": policy.events()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("role", choices=["both", "embodiment", "controller"])
    parser.add_argument("--camera", choices=["off", "raw"], default="off")
    parser.add_argument("--camera-hz", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--replan-ms", type=float, default=10.0)
    parser.add_argument("--warmup-s", type=float, default=3.0)
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument(
        "--address", help="the session (embodiment: join it; controller: required)"
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    if args.role == "controller":
        result = controller(args)
    elif args.role == "embodiment" and args.address:
        result = embodiment(args, args.address)
    else:
        with rlmesh.Host() as host:
            child = None
            out = Path(tempfile.mkdtemp()) / "controller.json"
            if args.role == "both":
                forwarded = [
                    f"--{k.replace('_', '-')}={v}"
                    for k, v in vars(args).items()
                    if k not in ("role", "address", "out")
                ]
                child = subprocess.Popen(
                    [
                        sys.executable,
                        __file__,
                        "controller",
                        *forwarded,
                        "--address",
                        host.address,
                        "--out",
                        str(out),
                    ],
                    env=os.environ.copy(),
                )
            else:
                print(f"session: {host.address}", file=sys.stderr)
            result = embodiment(args, host)
            if child is not None:
                result["controller"] = (
                    json.loads(out.read_text())
                    if child.wait() == 0
                    else {"error": child.returncode}
                )
    text = json.dumps(result, indent=2)
    if args.out is not None:
        args.out.write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main()

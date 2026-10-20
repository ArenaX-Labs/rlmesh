"""The Python binding's costs on loopback, the twin of bench/session.mjs (same scenario, same output).

The robot's loops run on threads, as test_v2_session.py's FakeUR10e does; the
policy runs on the main thread.
"""

from __future__ import annotations

import json
import os
import resource
import threading
import time
from pathlib import Path

import numpy as np
import rlmesh.v2 as rlmesh

SECONDS = float(os.environ.get("BENCH_SECONDS", "10"))
SPEC = Path(__file__).resolve().parents[3] / "crates" / "rlmesh-spec" / "tests" / "fixtures"
UR10E = json.loads((SPEC / "robots" / "ur10e.json").read_text())
JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
POLICY = {
    "hz": 50,
    "inputs": {
        "q": {"kind": "joints", "at": "arm", "fields": ["pos"], "names": JOINTS},
        "ft": {"kind": "wrench", "at": "wrist/ft", "frame": "tool0"},
        "wrist": {"kind": "camera", "at": "wrist/cam", "size": (96, 96)},
        "prompt": {"kind": "text", "at": "/task"},
    },
    "actions": {
        "arm": {
            "kind": "joints",
            "at": "arm",
            "command": "position",
            "names": JOINTS,
            "hz": 100,
            "horizon": 10,
        }
    },
    "allow": ["resize", "crop"],
}
HOME = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0], dtype=np.float32)


def quantiles(xs: list[float]) -> dict[str, float]:
    s = sorted(xs)

    def at(q: float) -> float:
        return round(s[min(len(s) - 1, int(q * len(s)))], 1)

    return {"p50": at(0.5), "p99": at(0.99), "max": at(1.0)}


def main() -> None:
    cpu0 = resource.getrusage(resource.RUSAGE_SELF)
    with rlmesh.Host() as host, rlmesh.Robot("ur10e", UR10E, session=host) as robot:
        robot.arm()
        stop = threading.Event()
        work: list[float] = []
        late: list[float] = []
        missed: list[int] = []

        def control() -> None:
            rate = rlmesh.Rate(1000)
            q = HOME
            zeros = np.zeros(6, dtype=np.float32)
            force = np.array([0, 0, 9.8], dtype=np.float32)
            torque = np.zeros(3, dtype=np.float32)
            gripper = np.ones(1, dtype=np.float32)
            while not stop.is_set():
                t0 = time.perf_counter_ns()
                robot.publish("arm", pos=q, vel=zeros, effort=zeros)
                robot.publish("wrist/ft", force=force, torque=torque)
                robot.publish("gripper", gripper)
                due = robot.due("arm")
                if due is not None and isinstance(due.value, np.ndarray):
                    q = due.value
                robot.due("gripper")
                work.append((time.perf_counter_ns() - t0) / 1e3)
                late.append(rate.sleep() / 1e3)
            missed.append(rate.missed)

        def cameras() -> None:
            wrist = np.zeros((480, 640, 3), dtype=np.uint8)
            scene = np.zeros((720, 1280, 3), dtype=np.uint8)
            rate = rlmesh.Rate(30)
            while not stop.is_set():
                robot.publish("wrist/cam", wrist)
                robot.publish("scene/cam", scene)
                robot.publish("task", "hold the arm")
                rate.sleep()

        threads = [threading.Thread(target=f, daemon=True) for f in (control, cameras)]
        for t in threads:
            t.start()
        with rlmesh.Policy(POLICY, session=host, name="bench") as policy:
            policy.bind("ur10e")
            hold = np.zeros((10, 6), dtype=np.float32)

            def predict(obs: rlmesh.Observation) -> dict[str, np.ndarray]:
                hold[:] = obs["q"]
                return {"arm": hold}

            stats = policy.run(predict, duration=SECONDS)
            stop.set()
            for t in threads:
                t.join()

            policy.release()
            n = 100_000
            zeros = np.zeros(6, dtype=np.float32)
            t0 = time.perf_counter_ns()
            for _ in range(n):
                robot.publish("arm", pos=HOME, vel=zeros, effort=zeros)
            publish_us = (time.perf_counter_ns() - t0) / 1e3 / n

    cpu = resource.getrusage(resource.RUSAGE_SELF)
    print(
        json.dumps(
            {
                "binding": "python",
                "seconds": SECONDS,
                "loop_ticks": len(work),
                "loop_work_us": quantiles(work),
                "loop_late_us": {**quantiles(late), "missed": missed[0]},
                "predict_ms": {
                    "completions": stats["completions"],
                    "p50": stats["latency_p50_ms"],
                    "p99": stats["latency_p99_ms"],
                    "max": stats["latency_max_ms"],
                },
                "skipped": stats["skipped"],
                "publish_us": round(publish_us, 2),
                "cpu_s": round(
                    cpu.ru_utime + cpu.ru_stime - cpu0.ru_utime - cpu0.ru_stime, 2
                ),
            }
        )
    )


if __name__ == "__main__":
    main()

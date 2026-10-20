"""A Python robot and a Python policy in one real-time session (protocol v2).

The robot is a UR10e arm without physics (commanded joints become measured
ones). It publishes its joints at 500 Hz and asks its edge, on every control
tick, which command is due. The policy binds to it by kind and path, then
replans a small sine around the measured pose at 20 Hz: chunks of ten
10 ms actions that the edge plays between replans.

    uv run python examples/python/v2/loopback.py --seconds 3
"""

from __future__ import annotations

import argparse
import threading
from typing import Any

import numpy as np
import rlmesh.v2 as rlmesh

UR_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

UR10E = {
    "arm": {
        "kind": "joints", "names": UR_JOINTS, "fields": ["pos", "vel"], "hz": 500,
        "command": "position", "safe": "stop", "hold_max_ms": 100, "max_action_age_ms": 1000,
    },
    "wrist/ft": {"kind": "wrench", "frame": "tool0", "hz": 500},
    "task": {"kind": "text"},
}  # fmt: skip

SINE = {
    "hz": 20,
    "inputs": {
        "q": {"kind": "joints", "at": "arm", "fields": ["pos"], "names": UR_JOINTS},
        "prompt": "text",
    },
    "actions": {
        "arm": {"kind": "joints", "at": "arm", "command": "position", "names": UR_JOINTS,
                "hz": 100, "horizon": 10},
    },
}  # fmt: skip


def robot_loop(
    robot: rlmesh.Robot, stop: threading.Event, applied: list[float]
) -> None:
    q = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])
    rate = rlmesh.Rate(500)
    robot.publish("task", "wave the first joint")
    while not stop.is_set():
        robot.publish("arm", pos=q, vel=np.zeros(6))
        robot.publish("wrist/ft", force=[0.0, 0.0, 9.8], torque=[0.0, 0.0, 0.0])
        due = robot.due("arm")
        if due is not None and isinstance(due.value, np.ndarray):
            q = due.value.astype(np.float64)  # a real robot sends it to the drives here
            applied.append(float(q[0]))
        rate.sleep()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=3.0)
    args = parser.parse_args()

    with rlmesh.Host() as host:
        robot = rlmesh.Robot("ur10e", UR10E, session=host)
        robot.arm()
        stop, applied = threading.Event(), list[float]()
        loop = threading.Thread(target=robot_loop, args=(robot, stop, applied))
        loop.start()

        def predict(obs: rlmesh.Observation) -> dict[str, Any]:
            t = obs.meta.time_ns / 1e9 + np.arange(10) * 0.01
            arm = np.repeat(obs["q"][None], 10, axis=0)
            arm[:, 0] = 0.2 * np.sin(2 * np.pi * 0.5 * t)
            return {"arm": arm}

        with rlmesh.Policy(SINE, session=host, name="sine") as policy:
            print(policy.bind("ur10e"))
            stats = policy.run(predict, duration=args.seconds)
        stop.set()
        loop.join()
        edge = robot.edge_stats("arm")
        robot.close()

    print(
        f"policy: {stats['completions']} replans, {stats['chunks_sent']} chunks, skipped {stats['skipped']}"
    )
    print(f"robot: {len(applied)} commands applied, edge modes {edge['modes']}")
    print(f"first joint swept {min(applied):+.3f} .. {max(applied):+.3f} rad")


if __name__ == "__main__":
    main()

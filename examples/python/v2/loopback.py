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
from rlmesh import realtime
from rlmesh.realtime import actions, inputs, parts

UR_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

UR10E = {
    "arm": parts.joints(
        UR_JOINTS,
        fields=["pos", "vel"],
        hz=500,
        command="position",
        safe=parts.Safe.STOP,
        hold_max_ms=100,
        max_action_age_ms=1000,
    ),
    "wrist/ft": parts.wrench(frame="tool0", hz=500),
    "task": parts.text(),
}


def robot_loop(
    robot: realtime.Robot, stop: threading.Event, applied: list[float]
) -> None:
    q = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])
    rate = realtime.Rate(500)
    robot.publish("task", "wave the first joint")
    while not stop.is_set():
        robot.publish("arm", pos=q, vel=np.zeros(6))
        robot.publish("wrist/ft", force=[0.0, 0.0, 9.8], torque=[0.0, 0.0, 0.0])
        target = robot.command("arm")
        if isinstance(target, np.ndarray):
            q = target.astype(np.float64)  # a real robot sends it to the drives here
            applied.append(float(q[0]))
        rate.sleep()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=3.0)
    args = parser.parse_args()

    with realtime.Host() as host:
        robot = realtime.Robot("ur10e", UR10E, session=host)
        robot.arm()
        stop, applied = threading.Event(), list[float]()
        loop = threading.Thread(target=robot_loop, args=(robot, stop, applied))
        loop.start()

        def predict(obs: realtime.Observation) -> dict[str, Any]:
            t = obs.meta.time_ns / 1e9 + np.arange(10) * 0.01
            arm = np.repeat(obs["q"][None], 10, axis=0)
            arm[:, 0] = 0.2 * np.sin(2 * np.pi * 0.5 * t)
            return {"arm": arm}

        # Each builder returns the plain dict the policy map takes: a dict
        # written by hand works the same.
        with realtime.Policy(
            hz=20,
            inputs={
                "q": inputs.joints("arm", fields=["pos"], names=UR_JOINTS),
                "prompt": inputs.text(),
            },
            actions={
                "arm": actions.joints(
                    "arm", command="position", names=UR_JOINTS, hz=100, chunk=10
                )
            },
            session=host,
            name="sine",
        ) as policy:
            print(policy.bind("ur10e").binding)
            stats = policy.run(predict, duration=args.seconds)
        stop.set()
        loop.join()
        edge = robot.edge_stats("arm")
        robot.close()

    assert stats["clock"] == "wall"  # a robot of this loop's own
    print(
        f"policy: {stats['completions']} replans, {stats['chunks_sent']} chunks, skipped {stats['skipped']}"
    )
    print(f"robot: {len(applied)} commands applied, edge modes {edge['modes']}")
    print(f"first joint swept {min(applied):+.3f} .. {max(applied):+.3f} rad")


if __name__ == "__main__":
    main()

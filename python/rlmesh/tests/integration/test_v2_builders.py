"""The builders' dicts as the host takes them, and the conveniences around a run.

- Every builder's output binds against a robot with a part of every kind,
  in an in-process session: the real normaliser and binder take it.
- A policy declared by keyword drives a robot through its part handles;
  ``predict(obs, policy)`` publishes, ``obs.age_ms`` reads its inputs' ages,
  and a second policy reads what the first publishes with ``inputs.of``.
- Every result a session returns has its ``TypedDict``'s keys and types.
- ``Policy(timeout=None)`` waits for a late robot and stays interruptible;
  ``bind(timeout=...)`` overrides it.
- ``Robot.serve`` serves a lockstep robot, and ``serve_lockstep`` still does.
"""

from __future__ import annotations

import _thread
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pytest
from rlmesh import realtime
from rlmesh.realtime import actions, inputs, parts, results
from typed_results import check

WAIT = 10.0

BOXES = parts.custom(
    "x/boxes",
    hz=10,
    fields={"xy": parts.array(np.float32, 2)},
    command="place",
    command_fields={"xy": parts.array(np.float32, 2)},
    safe=parts.Safe.HOLD,
)

#: A part of every kind, every commandable one commanded.
BENCH = {
    "cam": parts.camera(64, 48, hz=10, frame="cam_optical"),
    "depth": parts.depth(64, 48, hz=10, frame="cam_optical"),
    "arm": parts.joints(["a", "b"], hz=50, command="position", safe=parts.Safe.HOLD),
    "hand": parts.hand(["f1", "f2"], hz=50, command="position", safe=parts.Safe.HOLD),
    "ft": parts.wrench(frame="tool0", hz=50, command="wrench", safe=parts.Safe.STOP),
    "imu": parts.imu(frame="base", hz=50, fields=["orientation", "ang_vel"]),
    "tcp": parts.pose(
        frame="base", child="tool0", hz=50, command="absolute", safe=parts.Safe.HOLD
    ),
    "gripper": parts.gripper(hz=50, command="position", safe=parts.Safe.HOLD),
    "relay": parts.switch(hz=10, command="set", safe=parts.Safe.STOP),
    "dimmer": parts.value(
        hz=10, unit="percent", range=(0, 100), command="set", safe=parts.Safe.HOLD
    ),
    "base": parts.velocity(
        frame="base", hz=50, command="velocity", safe=parts.Safe.STOP
    ),
    "skin": parts.tactile([4, 4], hz=10),
    "task": parts.text(),
    "reward": parts.reward(),
    "success": parts.success(),
    "boxes": BOXES,
}


def wait_until(what: str, done: Callable[[], bool]) -> None:
    deadline = time.monotonic() + WAIT
    while not done():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        time.sleep(0.01)


@pytest.fixture(scope="module")
def bench() -> Iterator[tuple[realtime.Host, realtime.Robot]]:
    with realtime.Host() as host, realtime.Robot("bench", BENCH, session=host) as robot:
        yield host, robot


def binds(host: realtime.Host, name: str, **policy: Any) -> realtime.BindingTable:
    with realtime.Policy(**policy, session=host, name=name) as bound:
        return bound.bind("bench")


def test_every_builder_with_its_defaults_binds(
    bench: tuple[realtime.Host, realtime.Robot],
) -> None:
    host, _ = bench
    table = binds(
        host,
        "defaults",
        hz=10,
        inputs={
            "cam": inputs.camera("bench:cam"),
            "depth": inputs.depth("bench:depth"),
            "arm": inputs.joints("bench:arm"),
            "hand": inputs.hand("bench:hand"),
            "ft": inputs.wrench("bench:ft"),
            "imu": inputs.imu("bench:imu"),
            "tcp": inputs.pose("bench:tcp"),
            "gripper": inputs.gripper("bench:gripper"),
            "relay": inputs.switch("bench:relay"),
            "dimmer": inputs.value("bench:dimmer"),
            "base": inputs.velocity("bench:base"),
            "skin": inputs.tactile("bench:skin"),
            "task": inputs.text("bench:task"),
            "reward": inputs.reward("bench:reward"),
            "success": inputs.success("bench:success"),
            "boxes": inputs.custom(
                "bench:boxes", kind="x/boxes", fields={"xy": parts.array("float32", 2)}
            ),
            "boxes_of": inputs.of(BOXES, at="bench:boxes"),
        },
        actions={
            "arm": actions.joints("bench:arm"),
            "hand": actions.hand("bench:hand"),
            # A part that can't hold its target needs chunks that outlast a replan.
            "ft": actions.wrench("bench:ft", chunk=3),
            "tcp": actions.pose("bench:tcp"),
            "gripper": actions.gripper("bench:gripper"),
            "relay": actions.switch("bench:relay"),
            "dimmer": actions.value("bench:dimmer"),
            "base": actions.velocity("bench:base", chunk=3),
            "boxes": actions.custom(
                "bench:boxes",
                kind="x/boxes",
                fields={"xy": parts.array("float32", 2)},
                chunk=3,
            ),
        },
    )
    assert [i["entry"] for i in table.json["inputs"]] == [
        "cam", "depth", "arm", "hand", "ft", "imu", "tcp", "gripper", "relay",
        "dimmer", "base", "skin", "task", "reward", "success", "boxes", "boxes_of",
    ]  # fmt: skip
    assert len(table.json["actions"]) == 9


#: One entry per builder with its options set, each bound alone.
OPTIONS: list[tuple[str, dict[str, Any]]] = [
    (
        "camera",
        {
            "inputs": {
                "cam": inputs.camera(
                    "bench:cam",
                    size=32,
                    layout="chw",
                    format="bgr8",
                    resample="area",
                    crop=0.9,
                    upside_down=True,
                    dtype="float32",
                    range=(0, 1),
                    frame="cam_optical",
                    max_age_ms=500,
                    allow=["resize", "crop", "bgr", "rotate180"],
                )
            }
        },
    ),
    (
        "depth",
        {
            "inputs": {
                "depth": inputs.depth(
                    "bench:depth",
                    size=(24, 32),
                    units={"depth": "mm"},
                    frame="cam_optical",
                    allow=["resize", "units"],
                )
            }
        },
    ),
    (
        "joints",
        {
            "inputs": {
                "q": inputs.joints(
                    "bench:arm",
                    names=["b", "a"],
                    fields=["pos"],
                    dtype=np.float64,
                    units={"pos": "deg"},
                    history=3,
                    history_hz=50,
                    source="applied",
                )
            }
        },
    ),
    (
        "joints-positional",
        {"inputs": {"q": inputs.joints("bench:arm", names="positional", count=2)}},
    ),
    (
        "hand",
        {"inputs": {"h": inputs.hand("bench:hand", names=["f2"], allow=["select"])}},
    ),
    (
        "wrench",
        {"inputs": {"ft": inputs.wrench("bench:ft", frame="tool0", fields=["force"])}},
    ),
    (
        "imu",
        {
            "inputs": {
                "imu": inputs.imu(
                    "bench:imu",
                    frame="base",
                    fields=["orientation"],
                    rotation="quat_wxyz",
                )
            }
        },
    ),
    (
        "pose",
        {
            "inputs": {
                "tcp": inputs.pose(
                    "bench:tcp",
                    frame="base",
                    child="tool0",
                    rotation="matrix",
                    allow=["rotation"],
                )
            }
        },
    ),
    (
        "gripper",
        {
            "inputs": {
                "g": inputs.gripper("bench:gripper", range=(0, 255), allow=["rescale"])
            }
        },
    ),
    ("switch", {"inputs": {"relay": inputs.switch("bench:relay", source="applied")}}),
    ("value", {"inputs": {"dimmer": inputs.value("bench:dimmer", scale=0.01)}}),
    (
        "velocity",
        {
            "inputs": {
                "base": inputs.velocity("bench:base", frame="base", fields=["linear"])
            }
        },
    ),
    ("tactile", {"inputs": {"skin": inputs.tactile("bench:skin", shape=[4, 4])}}),
    (
        "events",
        {
            "inputs": {
                "task": inputs.text("bench:task", max_age_ms=1000),
                "reward": inputs.reward("bench:reward"),
                "note": inputs.text("bench:note", optional=True, allow=["blank"]),
            }
        },
    ),
    (
        "joints-action",
        {
            "actions": {
                "arm": actions.joints(
                    "bench:arm",
                    command="position",
                    hz=100,
                    chunk=10,
                    names=["b", "a"],
                    t0="send",
                    ttl_ms=200,
                )
            }
        },
    ),
    (
        "hand-action",
        {"actions": {"h": actions.hand("bench:hand", command="position", count=2)}},
    ),
    (
        "wrench-action",
        {
            "actions": {
                "ft": actions.wrench(
                    "bench:ft", command="wrench", frame="tool0", chunk=3
                )
            }
        },
    ),
    (
        "pose-action",
        {
            "actions": {
                "tcp": actions.pose(
                    "bench:tcp",
                    command="absolute",
                    frame="base",
                    child="tool0",
                    rotation="rot6d",
                    allow=["rotation"],
                )
            }
        },
    ),
    (
        "gripper-action",
        {
            "actions": {
                "g": actions.gripper(
                    "bench:gripper",
                    command="position",
                    open=0,
                    closed=1,
                    allow=["rescale"],
                )
            }
        },
    ),
    (
        "switch-action",
        {"actions": {"r": actions.switch("bench:relay", command="set", chunk=2)}},
    ),
    (
        "value-action",
        {
            "actions": {
                "d": actions.value(
                    "bench:dimmer", command="set", range=(0, 1), allow=["rescale"]
                )
            }
        },
    ),
    (
        "velocity-action",
        {
            "actions": {
                "b": actions.velocity(
                    "bench:base", command="velocity", frame="base", chunk=3
                )
            }
        },
    ),
    (
        "custom-action",
        {
            "actions": {
                "boxes": actions.custom(
                    "bench:boxes",
                    kind="x/boxes",
                    command="place",
                    fields={"xy": parts.array("float32", 2)},
                    hz=10,
                    chunk=3,
                )
            }
        },
    ),
    (
        "optional-action",
        {
            "inputs": {"task": inputs.text("bench:task")},
            "actions": {
                "x": actions.joints("bench:missing", optional=True, allow=["drop"])
            },
        },
    ),
]


@pytest.mark.parametrize(("name", "policy"), OPTIONS, ids=[name for name, _ in OPTIONS])
def test_every_builder_with_its_options_binds(
    bench: tuple[realtime.Host, realtime.Robot], name: str, policy: dict[str, Any]
) -> None:
    host, _ = bench
    table = binds(host, name, hz=10, **policy)
    assert table.warnings == [] or all(isinstance(w, str) for w in table.warnings)


def test_a_hand_written_dict_and_a_builder_bind_alike(
    bench: tuple[realtime.Host, realtime.Robot],
) -> None:
    host, _ = bench
    built = binds(
        host,
        "alike",
        hz=10,
        inputs={"q": inputs.joints("bench:arm", fields=["pos"])},
        actions={"arm": actions.joints("bench:arm", command="position", chunk=4)},
    )
    with realtime.Policy(
        {
            "hz": 10,
            "inputs": {"q": {"kind": "joints", "at": "bench:arm", "fields": ["pos"]}},
            "actions": {
                "arm": {
                    "kind": "joints",
                    "at": "bench:arm",
                    "command": "position",
                    "chunk": 4,
                }
            },
        },
        session=host,
        name="alike",
    ) as written:
        assert written.bind("bench").fingerprint == built.fingerprint


GIMBAL = {
    "servos": parts.joints(
        ["pan", "tilt"],
        hz=200,
        fields=["pos"],
        command="position",
        safe=parts.Safe.HOLD,
    ),
    "task": parts.text(),
}
AIM = {"aim": parts.custom("x/aim", hz=20, fields={"xy": parts.array("float32", 2)})}


def test_a_keyword_policy_drives_a_robot_through_its_part_handles() -> None:
    with (
        realtime.Host(lease_grace_ms=1000) as host,
        realtime.Robot("gimbal", GIMBAL, session=host) as robot,
    ):
        servos = robot.part("servos")
        assert (servos.path, servos.kind, servos.commanded) == (
            "servos",
            "joints",
            True,
        )
        assert repr(servos) == "RobotPart(gimbal:servos, joints)"
        task = robot.part("task")
        with pytest.raises(
            ValueError, match=r"gimbal has no part 'eyes' \(its parts: servos, task\)"
        ):
            robot.part("eyes")
        with pytest.raises(TypeError, match="task takes no commands: due"):
            task.due()
        with pytest.raises(TypeError, match="not a camera declared encoding='jpeg'"):
            servos.publish_jpeg(b"")
        robot.arm()
        task.publish("look around")
        stop = threading.Event()
        applied: list[float] = []

        def loop() -> None:
            q = np.zeros(2, np.float32)
            rate = realtime.Rate(200)
            while not stop.is_set():
                servos.publish(pos=q)
                due = servos.due()
                if due is not None and isinstance(due.value, np.ndarray):
                    q = due.value.astype(np.float32)
                    applied.append(float(q[0]))
                rate.sleep()

        ticker = threading.Thread(target=loop, daemon=True)
        ticker.start()
        ages: list[float | None] = []
        seen: list[realtime.Policy] = []

        def predict(
            obs: realtime.Observation, policy: realtime.Policy
        ) -> dict[str, Any]:
            ages.append(obs.age_ms("q"))
            seen.append(policy)
            policy.publish("aim", xy=np.array([0.25, 0.5], np.float32))
            return {"servos": np.full((4, 2), 0.3, np.float32)}

        try:
            with (
                realtime.Policy(
                    hz=20,
                    inputs={"q": inputs.joints("gimbal:servos", fields=["pos"])},
                    actions={
                        "servos": actions.joints(
                            "gimbal:servos", command="position", hz=100, chunk=4
                        )
                    },
                    session=host,
                    name="tracker",
                    publishes=AIM,
                ) as tracker,
                realtime.Policy(
                    hz=20,
                    inputs={"aim": inputs.of(AIM["aim"], at="tracker:aim")},
                    session=host,
                    name="viewer",
                ) as viewer,
            ):
                tracker.bind("gimbal")
                running: list[object] = []
                thread = threading.Thread(
                    target=lambda: running.append(tracker.run(predict, duration=30.0)),
                    daemon=True,
                )
                thread.start()
                wait_until(
                    "the servos driven",
                    lambda: bool(applied) and abs(applied[-1] - 0.3) < 1e-6,
                )
                assert seen and all(policy is tracker for policy in seen)
                assert ages and all(age is not None and age >= 0.0 for age in ages)

                viewer.bind()
                observed: list[realtime.Observation] = []

                def aim_seen() -> bool:
                    try:
                        obs = viewer.observe()
                    except realtime.StartSkipped:
                        return False
                    observed.append(obs)
                    return obs.meta.ages_ns["aim"] is not None

                wait_until("the viewer reading the tracker's aim", aim_seen)
                obs = observed[-1]
                assert obs["aim"].tolist() == [0.25, 0.5]
                age = obs.age_ms("aim")
                assert age is not None and age == obs.meta.ages_ns["aim"] / 1e6
                with pytest.raises(
                    KeyError, match="no input 'cam' \\(the inputs: aim\\)"
                ):
                    obs.age_ms("cam")

                # Every result, key for key and type for type.
                check(tracker.stats(), results.PolicyStats, "stats()")
                check(tracker.health(), results.Health, "Policy.health()")
                check(robot.health(), results.Health, "Robot.health()")
                check(robot.edge_stats("servos"), results.EdgeStats, "edge_stats()")
                check(servos.edge_stats(), results.EdgeStats, "RobotPart.edge_stats()")
                check(tracker.held, list[results.Held], "held")
                check(robot.held, list[results.Held], "held")

                def reported() -> bool:
                    link = tracker.health(wait=True)["links"].get("tracker")
                    return link is not None and link["run"] is not None

                wait_until("the tracker's run in the host's health", reported)
                check(tracker.health(), results.Health, "Policy.health() with a run")
                wait_until(
                    "the host's own rows",
                    lambda: any(r["chunks_accepted"] > 0 for r in host.health()),
                )
                check(host.health(), list[results.HostLink], "Host.health()")
                tracker.stop()
                thread.join(WAIT)
                [stats] = running
                check(stats, results.PolicyStats, "run()")
                check(tracker.events(), list[results.Event], "events()")
        finally:
            stop.set()
            ticker.join(WAIT)


def test_a_policy_without_a_timeout_waits_for_a_late_robot() -> None:
    with realtime.Host() as host:
        policy = realtime.Policy(
            hz=10,
            inputs={"q": inputs.joints("late:servos")},
            session=host,
            name="patient",
            timeout=None,
        )
        bound: list[object] = []
        thread = threading.Thread(
            target=lambda: bound.append(policy.bind("late")), daemon=True
        )
        thread.start()
        time.sleep(1.5)  # past nothing in particular: the wait has no limit
        assert thread.is_alive() and not bound
        with realtime.Robot("late", GIMBAL, session=host):
            thread.join(WAIT)
            assert isinstance(bound[0], realtime.BindingTable)
        policy.close()


def test_bind_overrides_the_timeout_of_its_policy() -> None:
    with realtime.Host() as host:
        policy = realtime.Policy(
            hz=10,
            inputs={"q": inputs.joints("absent:servos")},
            session=host,
            name="hurried",
            timeout=None,
        )
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="absent"):
            policy.bind("absent", timeout=0.3)
        assert time.monotonic() - started < 5.0
        with pytest.raises(ValueError, match='"policy"'):
            policy.bind("absent", timeout="later")  # pyright: ignore[reportArgumentType]
        policy.close()


def test_bind_takes_the_policy_timeout_by_name() -> None:
    with realtime.Host() as host:
        policy = realtime.Policy(
            hz=10,
            inputs={"q": inputs.joints("absent:servos")},
            session=host,
            name="named",
            timeout=0.3,
        )
        with pytest.raises(TimeoutError, match="absent"):
            policy.bind("absent", timeout="policy")
        policy.close()


def test_a_wait_without_a_timeout_stops_on_ctrl_c() -> None:
    with realtime.Host() as host:
        policy = realtime.Policy(
            hz=10,
            inputs={"q": inputs.joints("absent:servos")},
            session=host,
            name="interrupted",
            timeout=None,
        )
        timer = threading.Timer(0.5, _thread.interrupt_main)
        timer.start()
        try:
            with pytest.raises(KeyboardInterrupt):
                policy.bind("absent")
        finally:
            timer.cancel()
            policy.close()


#: A lockstep robot's map: no event parts, which a lockstep robot cannot declare yet.
SERVOS = {"servos": GIMBAL["servos"]}


@pytest.mark.parametrize("serve", ["serve", "serve_lockstep"])
def test_a_lockstep_robot_is_served_by_either_name(serve: str) -> None:
    gym = pytest.importorskip("gymnasium")
    import rlmesh.numpy

    with realtime.Host() as host:
        robot = realtime.Robot(
            "arm1", SERVOS, session=host, clock=realtime.Clock.LOCKSTEP
        )
        q = np.zeros(2, np.float32)

        def step(commands: dict[str, Any]) -> Any:
            nonlocal q
            q = commands["servos"].astype(np.float32)
            robot.publish("servos", pos=q)
            return 1.0, False, False

        def reset(seed: int | None, options: Any) -> None:
            robot.publish("servos", pos=q)

        servos = robot.part("servos")
        with pytest.raises(
            TypeError, match="a lockstep robot gets its commands in step"
        ):
            servos.due()
        getattr(robot, serve)(step, reset, background=True)
        env = rlmesh.numpy.RemoteEnv(f"{host.address}/arm1")
        try:
            assert isinstance(env, gym.Env)
            env.reset(seed=0)
            obs, reward, *_ = env.step({"servos": np.array([0.5, -0.5], np.float32)})
            assert obs["servos"].tolist() == [0.5, -0.5] and reward == 1.0
        finally:
            env.close()
            robot.close()


def test_a_wall_clock_robot_is_not_served() -> None:
    with (
        realtime.Host() as host,
        realtime.Robot("wall", GIMBAL, session=host, clock="wall") as robot,
    ):
        with pytest.raises(RuntimeError, match=r"clock=Clock\.LOCKSTEP"):
            robot.serve(lambda commands: None)

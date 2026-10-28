"""rlmesh.v2 on loopback: a Python robot and Python policies in one session.

The robot is the §14.1 UR10e (the conformance fixture), faked: its control
loop publishes joints, wrench, gripper and two JPEG cameras, and applies
whatever its edge says is due. The policies bind to it by kind and path,
lease its parts and stream timed chunks.
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rlmesh.v2 as rlmesh

REPO_ROOT = Path(__file__).resolve().parents[4]
SPEC = REPO_ROOT / "crates" / "rlmesh-session-core" / "tests"
WAIT = 10.0

UR_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
HOME = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0], dtype=np.float32)

#: A small sine around the measured pose on the arm, the gripper half open;
#: reads the joints, the wrench, the wrist camera (cropped to 96x96) and the
#: task.
SINE = {
    "hz": 50,
    "inputs": {
        "q": {"kind": "joints", "at": "arm", "fields": ["pos"], "names": UR_JOINTS},
        "ft": {"kind": "wrench", "at": "wrist/ft", "frame": "tool0"},
        "wrist": {"kind": "camera", "at": "wrist/cam", "size": (96, 96)},
        "prompt": {"kind": "text", "at": "/task"},
    },
    "actions": {
        "arm": {
            "kind": "joints",
            "at": "arm",
            "command": "position",
            "names": UR_JOINTS,
            "hz": 100,
            "horizon": 10,
        },
        "gripper": {
            "kind": "gripper",
            "at": "gripper",
            "command": "position",
            "hz": 100,
            "horizon": 10,
        },
    },
    "allow": ["resize", "crop"],
}

#: Holds one target on the arm (the operator's and the model's, told apart
#: by value).
HOLD = {
    "hz": 50,
    "inputs": {
        "q": {"kind": "joints", "at": "arm", "fields": ["pos"], "names": UR_JOINTS}
    },
    "actions": {
        "arm": {
            "kind": "joints",
            "at": "arm",
            "command": "position",
            "names": UR_JOINTS,
            "hz": 100,
            "horizon": 10,
        }
    },
}


def fixture(kind: str, name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(
        (SPEC / "fixtures" / kind / f"{name}.json").read_text()
    )
    return data


def golden(name: str) -> str:
    return (SPEC / "golden" / f"{name}.txt").read_text().rstrip("\n")


def wait_until(
    what: str, done: Callable[[], bool], context: Callable[[], object] = lambda: ""
) -> None:
    """Poll ``done`` every 10 ms; ``context()`` goes into the timeout's message."""
    deadline = time.monotonic() + WAIT
    while not done():
        assert time.monotonic() < deadline, f"timed out waiting for {what} {context()}"
        time.sleep(0.01)


class FakeUR10e:
    """The §14.1 UR10e without physics: commanded joints become measured ones."""

    def __init__(self, host: rlmesh.Host, name: str = "ur10e") -> None:
        self.robot = rlmesh.Robot(name, fixture("robots", "ur10e"), session=host)
        self.q = HOME.copy()
        self.gripper = 1.0
        self.log: list[rlmesh.Due] = []
        self._stop = threading.Event()
        self._threads = [
            threading.Thread(target=self._control, daemon=True),
            threading.Thread(target=self._cameras, daemon=True),
        ]

    def __enter__(self) -> FakeUR10e:
        self.robot.arm()
        self.robot.publish("task", "wave the arm")
        for thread in self._threads:
            thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join()
        self.robot.close()

    def _control(self) -> None:
        rate = rlmesh.Rate(500)
        while not self._stop.is_set():
            self.robot.publish("arm", pos=self.q, vel=np.zeros(6), effort=np.zeros(6))
            self.robot.publish("wrist/ft", force=[0.0, 0.0, 9.8], torque=np.zeros(3))
            self.robot.publish("gripper", [self.gripper])
            due = self.robot.due("arm")
            if due is not None:
                if isinstance(due.value, np.ndarray):
                    self.q = due.value.astype(np.float32)
                self.log.append(due)
            gripper = self.robot.due("gripper")
            if gripper is not None and isinstance(gripper.value, np.ndarray):
                self.gripper = float(gripper.value[0])
            rate.sleep()

    def _cameras(self) -> None:
        wrist = np.zeros((480, 640, 3), dtype=np.uint8)
        scene = np.zeros((720, 1280, 3), dtype=np.uint8)
        rate = rlmesh.Rate(10)
        frame = 0
        while not self._stop.is_set():
            wrist[:, :, 0] = frame % 255
            self.robot.publish("wrist/cam", wrist)
            self.robot.publish("scene/cam", scene)
            frame += 1
            rate.sleep()

    def driven(self, predicate: Callable[[rlmesh.Due], bool], n: int = 5) -> bool:
        recent = self.log[-n:]
        return len(recent) == n and all(
            d.mode == "driving" and predicate(d) for d in recent
        )


@pytest.fixture
def host() -> Iterator[rlmesh.Host]:
    # A loaded CI runner can delay a heartbeat past the default 250 ms grace
    # and lapse a lease mid-test; the tests are about authority, not timing.
    with rlmesh.Host(lease_grace_ms=1000) as host:
        yield host


def in_thread(target: Callable[[], object]) -> tuple[threading.Thread, list[object]]:
    out: list[object] = []

    def run() -> None:
        try:
            out.append(target())
        except BaseException as exc:
            out.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, out


def raised(out: list[object]) -> bool:
    """Re-raises what an ``in_thread`` target raised, so a dead loop fails the
    wait at once with its own error instead of a timeout; False otherwise."""
    if out and isinstance(out[0], BaseException):
        raise out[0]
    return False


def test_a_python_policy_drives_a_python_robot_on_loopback(host: rlmesh.Host) -> None:
    seen: list[rlmesh.Observation] = []

    def predict(obs: rlmesh.Observation) -> dict[str, np.ndarray[Any, Any]]:
        seen.append(obs)
        t = obs.meta.time_ns / 1e9 + np.arange(10) * 0.01
        arm = HOME[None] + 0.1 * np.sin(2 * np.pi * t)[:, None]
        return {"arm": arm, "gripper": np.full((10, 1), 0.5)}

    with (
        FakeUR10e(host) as ur,
        rlmesh.Policy(SINE, session=host, name="sine") as policy,
    ):
        binding = policy.bind("ur10e")
        assert re.search(
            r"wrist +in +wrist/cam +bound +decode\(jpeg\) → crop\(640x480 → 96x96, bilinear\)",
            str(binding),
        )
        assert binding.json["actions"][0]["horizon"] == 10
        assert len(binding.fingerprint) == 16
        thread, result = in_thread(lambda: policy.run(predict, duration=30.0))
        wait_until(
            "the arm driven by the sine",
            lambda: raised(result)
            or ur.driven(lambda d: abs(float(np.asarray(d.value)[0])) <= 0.1001),
            lambda: {"stats": policy.stats(), "holds": policy.holds},
        )
        wait_until("the gripper commanded", lambda: abs(ur.gripper - 0.5) < 1e-6)
        policy.stop()
        thread.join(WAIT)
        stats = result[0]
        assert isinstance(stats, dict), stats
        assert stats["completions"] > 0 and stats["chunks_sent"] > 0
        assert stats["send_errors"] == 0
        assert not policy.holds, "a stopped run releases its leases"

        # predict saw what the binding table promises.
        obs = seen[-1]
        assert set(obs) == {"q", "ft", "wrist", "prompt"}
        assert obs["q"].shape == (6,) and obs["q"].dtype == np.float32
        assert obs["ft"].shape == (6,)
        assert obs["wrist"].shape == (96, 96, 3) and obs["wrist"].dtype == np.uint8
        assert not obs["wrist"].flags.writeable, "observation arrays are read-only"
        assert obs["prompt"] == "wave the arm"
        assert set(obs.meta.ages_ns) == {"q", "ft", "wrist", "prompt"}
        assert not obs.meta.stale

        # The robot applied the policy's chunks, and knows what they came from.
        driving = [d for d in ur.log if d.mode == "driving"]
        assert driving and all(d.path == "arm" for d in driving)
        assert any(path == "arm" for path, _, _ in driving[-1].based_on)
        edge = ur.robot.edge_stats("arm")
        assert edge["accepted"] > 0 and edge["rejected"] == 0, edge["rejected_by"]
        assert edge["modes"]["safe"] == (0, 0), "a released lease never latches safe"


def test_the_binding_table_and_refusals_match_the_conformance_goldens(
    host: rlmesh.Host,
) -> None:
    robot = rlmesh.Robot("ur10e", fixture("robots", "ur10e"), session=host)
    rtde = rlmesh.Robot("ur10e_rtde", fixture("robots", "ur10e_rtde"), session=host)
    try:
        with rlmesh.Policy(
            fixture("policies", "pi0_single"), session=host, name="pi0_single"
        ) as pi0:
            binding = pi0.bind("ur10e")
            assert binding.table.rstrip("\n") == golden("pi0_single__ur10e")
            assert binding.warnings and "gripper" in binding.warnings[0]

        refused = rlmesh.Policy(
            fixture("policies", "force_policy"), session=host, name="force_policy"
        )
        with pytest.raises(rlmesh.BindingRefused) as caught:
            refused.bind("ur10e_rtde")
        assert str(caught.value).rstrip("\n") == golden(
            "force_policy__ur10e_rtde__refused"
        )
        assert caught.value.problems == [("incompatible", "input", "ft")]
        # The fix the message suggests is a different robot: the same policy
        # binds where the wrench is in tool0.
        assert "wrist/ft" in str(refused.bind("ur10e"))
        refused.close()

        with pytest.raises(rlmesh.BindingRefused, match="unknown conversion"):
            rlmesh.Policy(HOLD, session=host, name="typo").bind(
                "ur10e", allow=["resise"]
            )
    finally:
        rtde.close()
        robot.close()


def test_an_operator_preempts_the_model_and_hands_back_under_a_fresh_token(
    host: rlmesh.Host,
) -> None:
    def hold(target: float) -> Callable[[rlmesh.Observation], dict[str, Any]]:
        return lambda obs: {"arm": np.full((10, 6), target, dtype=np.float32)}

    with (
        FakeUR10e(host) as ur,
        rlmesh.Policy(HOLD, session=host, name="model") as model,
        rlmesh.Policy(
            HOLD, session=host, name="teleop", priority="operator"
        ) as operator,
    ):
        model.bind("ur10e")
        operator.bind("ur10e")
        thread, result = in_thread(lambda: model.run(hold(0.25), duration=30.0))
        wait_until(
            "the model driving",
            lambda: raised(result)
            or ur.driven(lambda d: float(np.asarray(d.value)[0]) == 0.25),
            lambda: {"stats": model.stats(), "holds": model.holds},
        )
        model_token = ur.log[-1].lease_token
        assert model_token is not None

        # Taking over: the operator class preempts the model class.
        tokens = operator.acquire()
        assert tokens["arm"] > model_token
        wait_until("the model's lease gone", lambda: not model.holds)
        streaming = threading.Event()

        def teleop() -> None:
            while not streaming.is_set():
                try:
                    obs = operator.observe()
                except rlmesh.StartSkipped:
                    # Under load an input may not be fresh yet: try next tick.
                    time.sleep(0.01)
                    continue
                sent = operator.send({"arm": np.full((10, 6), 0.75)}, obs)
                if sent["arm"] is None:
                    raise AssertionError(
                        f"the operator lost its lease (holds={operator.holds})"
                    )
                time.sleep(0.01)

        operator_thread, operator_result = in_thread(teleop)
        wait_until(
            "the operator driving",
            lambda: raised(operator_result)
            or ur.driven(lambda d: float(np.asarray(d.value)[0]) == 0.75),
            lambda: {
                "teleop": operator_result,
                "rejected_by": ur.robot.edge_stats("arm")["rejected_by"],
                "recent": [(d.mode, d.lease_token) for d in ur.log[-5:]],
                "operator_holds": operator.holds,
            },
        )
        assert ur.log[-1].lease_token == tokens["arm"]
        assert model.stats()["skipped"]["no_lease"] > 0, (
            "a preempted model stays warm, skips its starts"
        )

        # A model cannot take it back while the operator holds it.
        with pytest.raises(rlmesh.SessionRefused) as busy:
            model.acquire()
        assert busy.value.status == "LEASE_BUSY"

        # Handing back: the model re-requests and resumes under a fresh token.
        streaming.set()
        operator_thread.join(WAIT)
        assert operator_result == [None]
        operator.release()
        wait_until(
            "the model driving again",
            lambda: raised(result)
            or ur.driven(lambda d: float(np.asarray(d.value)[0]) == 0.25),
            lambda: {"stats": model.stats(), "holds": model.holds},
        )
        resumed = ur.log[-1].lease_token
        assert resumed is not None and resumed > tokens["arm"]
        model.stop()
        thread.join(WAIT)
        assert isinstance(result[0], dict)


def test_a_recorder_writes_the_session_to_mcap_and_reads_it_back(
    host: rlmesh.Host, tmp_path: Path
) -> None:
    path = tmp_path / "session.mcap"
    with rlmesh.Recorder(path, session=host) as recorder:
        with (
            FakeUR10e(host) as ur,
            rlmesh.Policy(HOLD, session=host, name="model") as model,
        ):
            model.bind("ur10e")
            thread, result = in_thread(
                lambda: model.run(
                    lambda obs: {"arm": np.full((10, 6), 0.25, dtype=np.float32)},
                    duration=30.0,
                )
            )
            wait_until("the model driving", lambda: ur.driven(lambda d: True, n=50))
            model.stop()
            thread.join(WAIT)
            assert isinstance(result[0], dict)
    stats = recorder.stats
    assert stats is not None and stats["messages"] > 0
    assert recorder.close() == stats, "closing twice keeps the stats"

    summary = rlmesh.read_recording(path)
    assert summary["messages"] == stats["messages"]
    topics = summary["topics"]
    for topic in (
        "/ur10e/arm",
        "/ur10e/arm/_applied",
        "/ur10e/arm/_chunks",
        "/ur10e/wrist/cam",
        "/_session/authority",
        "/_session/bindings",
    ):
        row = topics[topic]
        assert row["messages"] > 0, topic
        assert row["undecodable"] == 0, topic
        assert row["log_time_monotonic"] and row["publish_time_monotonic"], topic
    assert topics["/ur10e/arm/_chunks"]["schema"] == "rlmesh.session.v2.ActionChunk"
    assert "rlmesh.robot.ur10e" in summary["metadata"]
    arm = [lease for lease in summary["leases"] if lease["part"] == "ur10e/arm"]
    assert arm[0]["holder"] == "model" and arm[0]["priority"] == "model"

    # The same recording as a collected episode, resampled for LeRobot.
    from rlmesh.v2 import lerobot

    take = tmp_path / "collected" / "default" / "1"
    take.mkdir(parents=True)
    path.rename(take / path.name)
    arm_topic = topics["/ur10e/arm"]
    (take / "episode.json").write_text(
        json.dumps(
            {
                "task": "hold",
                "start_ns": arm_topic["first_log_ns"],
                "end_ns": arm_topic["last_log_ns"],
                "success": True,
            }
        )
    )
    [episode] = lerobot._takes(take.parent, include_failed=False)
    frames = list(lerobot._frames(episode, fps=30))
    assert len(frames) > 1
    states, actions, cameras = lerobot._layout(frames[-1], None)
    assert "ur10e/arm" in states and actions == ["ur10e/arm/_applied"]
    assert cameras == {
        "observation.images.scene_cam": "ur10e/scene/cam",
        "observation.images.wrist_cam": "ur10e/wrist/cam",
    }
    np.testing.assert_allclose(lerobot._numbers(frames[-1]["ur10e/arm/_applied"]), 0.25)


def test_a_raising_predict_releases_the_leases_and_propagates(
    host: rlmesh.Host,
) -> None:
    calls = 0

    def predict(obs: rlmesh.Observation) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("the model crashed")
        return {"arm": np.repeat(obs["q"][None], 10, axis=0)}

    with FakeUR10e(host), rlmesh.Policy(HOLD, session=host, name="crashy") as policy:
        policy.bind("ur10e")
        with pytest.raises(RuntimeError, match="the model crashed"):
            policy.run(predict, duration=WAIT)
        assert not policy.holds
        # A wrong shape is refused before anything is sent.
        with pytest.raises(ValueError, match=r"expected \[10, 6\]"):
            policy.send({"arm": np.zeros(6)}, policy.observe())
        with pytest.raises(ValueError, match="not an action"):
            policy.send({"legs": np.zeros((10, 6))}, policy.observe())


def test_publish_checks_fields_dtypes_and_sizes(host: rlmesh.Host) -> None:
    with rlmesh.Robot("ur10e", fixture("robots", "ur10e"), session=host) as robot:
        assert robot.parts["wrist/cam"] == "camera"
        robot.publish("arm", {"pos": HOME, "vel": np.zeros(6), "effort": np.zeros(6)})
        assert robot.publish("wrist/ft", force=[1, 2, 3], torque=[0, 0, 0]) is not None
        with pytest.raises(TypeError, match="missing"):
            robot.publish("arm", pos=HOME)
        with pytest.raises(TypeError, match=r"publish\('arm', pos=\.\.\.\)"):
            robot.publish("arm", HOME)
        with pytest.raises(ValueError, match="7 elements"):
            robot.publish("gripper", np.zeros(7))
        with pytest.raises(TypeError, match="cannot publish float64 as uint8"):
            robot.publish("wrist/cam", np.zeros((480, 640, 3)))
        with pytest.raises(TypeError, match="publish a str"):
            robot.publish("task", 3)
        with pytest.raises(ValueError, match="no part 'elbow'"):
            robot.publish("elbow", 1.0)
        assert robot.due("arm") is None, "nothing is leased: nothing is due"
        with pytest.raises(ValueError, match="not a commandable part"):
            robot.due("wrist/ft")

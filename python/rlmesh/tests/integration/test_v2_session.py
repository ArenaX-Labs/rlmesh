"""rlmesh.realtime on loopback: a Python robot and Python policies in one session.

The robot is the UR10e (the conformance fixture), faked: its control
loop publishes joints, wrench, gripper and two JPEG cameras, and applies
whatever its edge says is due. The policies bind to it by kind and path,
lease its parts and stream timed chunks.
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rlmesh.realtime as rlmesh
from rlmesh import _rlmesh

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
            "chunk": 10,
        },
        "gripper": {
            "kind": "gripper",
            "at": "gripper",
            "command": "position",
            "hz": 100,
            "chunk": 10,
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
            "chunk": 10,
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
    """The UR10e without physics: commanded joints become measured ones."""

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


def test_the_sessions_key_is_in_the_address_never_in_a_repr(host: rlmesh.Host) -> None:
    key = host.address.partition("#")[2]
    assert len(key) > 52
    with FakeUR10e(host) as ur:
        assert key in ur.robot.address
        for shown in (repr(host), str(host), repr(ur.robot)):
            assert key not in shown, shown


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
        assert binding.json["actions"][0]["chunk"] == 10
        assert len(binding.fingerprint) == 16
        thread, result = in_thread(lambda: policy.run(predict, duration=30.0))
        wait_until(
            "the arm driven by the sine",
            lambda: (
                raised(result)
                or ur.driven(lambda d: abs(float(np.asarray(d.value)[0])) <= 0.1001)
            ),
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
            lambda: (
                raised(result)
                or ur.driven(lambda d: float(np.asarray(d.value)[0]) == 0.25)
            ),
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
            lambda: (
                raised(operator_result)
                or ur.driven(lambda d: float(np.asarray(d.value)[0]) == 0.75)
            ),
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
            lambda: (
                raised(result)
                or ur.driven(lambda d: float(np.asarray(d.value)[0]) == 0.25)
            ),
            lambda: {"stats": model.stats(), "holds": model.holds},
        )
        resumed = ur.log[-1].lease_token
        assert resumed is not None and resumed > tokens["arm"]
        model.stop()
        thread.join(WAIT)
        assert isinstance(result[0], dict)


def test_a_recorder_writes_the_session_to_mcap_and_reads_it_back(
    host: rlmesh.Host, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "session.mcap"
    with rlmesh.SessionRecorder(path, session=host) as recorder:
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

    # The same recording as a collected episode, resampled for LeRobot on
    # capture time, and written through the lerobot package's dataset API.
    from rlmesh.realtime import lerobot

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

    datasets = fake_lerobot(monkeypatch)
    out = tmp_path / "dataset"
    assert lerobot.export(take.parent, out, repo_id="me/ur10e", fps=30) == 1
    [dataset] = datasets
    assert dataset.root == out and dataset.fps == 30
    assert dataset.robot_type == "ur10e"
    assert dataset.features["action"]["shape"] == (6,)
    assert dataset.features["observation.images.wrist_cam"]["shape"] == (480, 640, 3)
    [rows] = dataset.episodes
    assert rows and all(row["task"] == "hold" for row in rows)
    assert rows[-1]["observation.images.scene_cam"].shape == (720, 1280, 3)
    np.testing.assert_allclose(rows[-1]["action"], 0.25)
    assert dataset.finalized

    # The whole recording, with no episode.json, is one episode of `task`.
    segment = next(take.glob("*.mcap"))
    whole = fake_lerobot(monkeypatch)
    assert (
        lerobot.export(segment, tmp_path / "whole", repo_id="me/ur10e", task="wave")
        == 1
    )
    assert len(whole[0].episodes[0]) >= len(rows)
    assert whole[0].episodes[0][0]["task"] == "wave"


class FakeDataset:
    """What ``lerobot.datasets.lerobot_dataset.LeRobotDataset`` is asked to write."""

    def __init__(
        self,
        repo_id: str,
        fps: int,
        features: dict[str, Any],
        root: Path,
        robot_type: str,
    ) -> None:
        self.repo_id, self.fps, self.features = repo_id, fps, features
        self.root, self.robot_type = Path(root), robot_type
        self.episodes: list[list[dict[str, Any]]] = []
        self.frames: list[dict[str, Any]] = []
        self.finalized = False

    def add_frame(self, row: dict[str, Any]) -> None:
        for key, feature in self.features.items():
            assert tuple(np.shape(row[key])) == tuple(feature["shape"]), key
        self.frames.append(row)

    def save_episode(self) -> None:
        self.episodes.append(self.frames)
        self.frames = []

    def finalize(self) -> None:
        self.finalized = True


def fake_lerobot(monkeypatch: pytest.MonkeyPatch) -> list[FakeDataset]:
    """Stands in for the ``lerobot`` package; returns the datasets it creates."""
    import types

    created: list[FakeDataset] = []

    class LeRobotDataset:
        @staticmethod
        def create(
            repo_id: str,
            fps: int,
            *,
            features: dict[str, Any],
            root: Path,
            robot_type: str,
            use_videos: bool,
        ) -> FakeDataset:
            assert use_videos
            created.append(FakeDataset(repo_id, fps, features, root, robot_type))
            return created[-1]

    module = types.ModuleType("lerobot.datasets.lerobot_dataset")
    module.LeRobotDataset = LeRobotDataset  # type: ignore[attr-defined]
    for name in ("lerobot", "lerobot.datasets"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return created


def test_a_policy_publishes_annotations_under_its_name_and_never_binds_them(
    host: rlmesh.Host, tmp_path: Path
) -> None:
    gaze = {
        "hz": 25,
        "actions": {
            "gaze": {"kind": "joints", "at": "eyes", "command": "position", "hz": 50}
        },
    }
    target = {
        "target": {"kind": "x/target", "hz": 25, "fields": {"xy": ["float32", [2]]}}
    }
    path = tmp_path / "tracked.mcap"
    eye = rlmesh.Robot("eye-1", EYE, session=host)
    try:
        eye.arm()
        with (
            rlmesh.SessionRecorder(path, session=host),
            rlmesh.Policy(
                gaze, session=host, name="tracker", publishes=target
            ) as tracker,
        ):
            # One robot besides the tracker's own parts: bound with no robot
            # named, the eye alone, at its path.
            binding = tracker.bind()
            assert "binding tracker on eye-1" in str(binding), str(binding)
            with pytest.raises(ValueError, match="no part 'boxes'"):
                tracker.publish("boxes", np.zeros(2, np.float32))
            # For half a second: the recorder subscribes to a new part once
            # it sees it.
            sent = []
            for i in range(25):
                sent.append(
                    tracker.publish("target", np.array([0.01 * i, 0.5], np.float32))
                )
                time.sleep(0.02)
            assert any(seq is not None for seq in sent)
        with pytest.raises(ValueError, match="publishes nothing"):
            plain = rlmesh.Policy(gaze, session=host, name="plain")
            with plain:
                plain.bind("eye-1")
                plain.publish("target", np.zeros(2, np.float32))
    finally:
        eye.close()
    topics = rlmesh.read_recording(path)["topics"]
    assert topics["/tracker/target"]["messages"] >= 1
    assert topics["/tracker/target"]["undecodable"] == 0


def test_a_camera_publishes_frames_it_already_encoded(
    host: rlmesh.Host, tmp_path: Path
) -> None:
    pil = pytest.importorskip("PIL.Image")

    def jpeg(width: int, height: int) -> bytes:
        out = io.BytesIO()
        pil.new("RGB", (width, height), (200, 30, 30)).save(out, "JPEG")
        return out.getvalue()

    cam = {"cam": rlmesh.parts.camera(64, 48, hz=20, encoding="jpeg")}
    path = tmp_path / "jpeg.mcap"
    with (
        rlmesh.Robot("webcam", cam, session=host) as camera,
        rlmesh.SessionRecorder(path, session=host),
    ):
        frame = jpeg(64, 48)
        for _ in range(10):
            camera.publish_jpeg("cam", frame)
            time.sleep(0.06)
        with pytest.raises(ValueError, match="32x24"):
            camera.publish_jpeg("cam", jpeg(32, 24))
        with pytest.raises(ValueError, match="not a JPEG"):
            camera.publish_jpeg("cam", b"not a jpeg")
    with rlmesh.Robot(
        "raw", {"cam": rlmesh.parts.camera(64, 48, hz=20)}, session=host
    ) as raw:
        with pytest.raises(ValueError, match="encoding: jpeg"):
            raw.publish_jpeg("cam", frame)
    topics = rlmesh.read_recording(path)["topics"]
    assert topics["/webcam/cam"]["messages"] >= 1
    assert topics["/webcam/cam"]["undecodable"] == 0


def test_eyes_with_no_state_export_and_an_idle_eye_keeps_its_last_action(
    host: rlmesh.Host, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rlmesh.realtime import lerobot

    gaze = {
        "hz": 25,
        "actions": {
            "gaze": {"kind": "joints", "at": "eyes", "command": "position", "hz": 50}
        },
    }
    eye = {**EYE, "eyes": {**EYE["eyes"], "safe": "stop"}}
    path = tmp_path / "eyes.mcap"
    stop = threading.Event()
    eyes = [rlmesh.Robot(name, eye, session=host) for name in ("eye-1", "eye-2")]
    camera = rlmesh.Robot(
        "pi",
        {"cam": {"kind": "camera", "width": 64, "height": 48, "hz": 20}},
        session=host,
    )

    def loop() -> None:
        # No state: the eyes only take actions; the Pi only sees.
        rate = rlmesh.Rate(200)
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        tick = 0
        while not stop.is_set():
            for e in eyes:
                e.due("eyes")
            if tick % 10 == 0:
                camera.publish("cam", frame)
            tick += 1
            rate.sleep()

    try:
        eyes[0].arm()
        with rlmesh.SessionRecorder(path, session=host):
            ticker = threading.Thread(target=loop, daemon=True)
            ticker.start()
            with (
                rlmesh.Policy(gaze, session=host, name="one") as one,
                rlmesh.Policy(gaze, session=host, name="two") as two,
            ):
                one.bind("eye-1")
                two.bind("eye-2")
                runs = [
                    in_thread(
                        lambda p=p, v=v: p.run(
                            lambda obs, v=v: {"gaze": np.full((1, 2), v, np.float32)},
                            duration=30.0,
                        )
                    )
                    for p, v in ((one, 0.25), (two, 0.5))
                ]
                time.sleep(1.0)
                # Eye 2's policy stops: its edge holds, then goes safe and
                # dispatches nothing, while eye 1 is still driven.
                two.stop()
                runs[1][0].join(WAIT)
                time.sleep(1.5)
                one.stop()
                runs[0][0].join(WAIT)
                for _, out in runs:
                    assert not raised(out)
            stop.set()
            ticker.join(WAIT)
    finally:
        stop.set()
        for robot in (*eyes, camera):
            robot.close()

    datasets = fake_lerobot(monkeypatch)
    assert lerobot.export(path, tmp_path / "out", repo_id="me/eyes", task="track") == 1
    [dataset] = datasets
    assert "observation.state" not in dataset.features
    assert dataset.features["action"]["names"] == [
        "eye-1/eyes/_applied.pos[0]",
        "eye-1/eyes/_applied.pos[1]",
        "eye-2/eyes/_applied.pos[0]",
        "eye-2/eyes/_applied.pos[1]",
    ]
    assert dataset.features["observation.images.pi_cam"]["shape"] == (48, 64, 3)
    assert dataset.robot_type == "eye-1+eye-2+pi"
    [rows] = dataset.episodes
    # Eye 2 dispatched nothing for over a second while eye 1 was driven, and
    # those frames are all there: eye 2's last action carried over.
    dispatched: dict[str, list[int]] = {}
    for topic, _, capture_ns, fields in _rlmesh.V2Recorder.samples(str(path), False):
        if fields:
            dispatched.setdefault(topic, []).append(capture_ns)
    one_last = max(dispatched["eye-1/eyes/_applied"])
    two_last = max(dispatched["eye-2/eyes/_applied"])
    assert one_last - two_last > 1_000_000_000
    first = max(min(times) for times in dispatched.values())
    assert len(rows) >= 0.9 * (one_last - first) * 30 / 1e9, len(rows)
    np.testing.assert_allclose(rows[-1]["action"], [0.25, 0.25, 0.5, 0.5])


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


def test_a_local_estop_stops_the_robot_at_once_and_the_host_learns_why(
    host: rlmesh.Host,
) -> None:
    with FakeUR10e(host) as ur:
        ur.robot.estop("button")
        wait_until("the edge in estop", lambda: any(d.mode == "estop" for d in ur.log))
        wait_until(
            "ESTOP_LOCAL at the host",
            lambda: (ur.robot.mode, ur.robot.mode_reason) == ("estop", "estop_local"),
        )
        ur.robot.clear_estop()
        wait_until("the clear", lambda: ur.robot.mode_reason == "estop_cleared")
        assert ur.robot.mode == "estop", "leaving ESTOP still needs disarm and arm"
        ur.robot.disarm()
        ur.robot.arm()
        assert ur.robot.mode == "armed"


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


def test_frames_between_composes_two_placed_poses() -> None:
    from rlmesh.frames import between

    # The two-robot rig: in the arm's frame (turned 90° about z),
    # eye-l's frame is turned -90° and its origin reads (0.2, 0.365, 0.3).
    eye = {
        "at": "eyes/left",
        "pose": {"xyz": [0.035, 0.0, 0.30], "quat_xyzw": [0, 0, 0, 1]},
    }
    arm = {"xyz": [0.40, -0.20, 0.0], "quat_xyzw": [0, 0, 0.7071068, 0.7071068]}
    t = between(eye, arm)
    assert t["xyz"] == pytest.approx([0.2, 0.365, 0.3])
    assert t["quat_xyzw"] == pytest.approx([0, 0, -(0.5**0.5), 0.5**0.5])


#: A two-joint arm that waits in a lobby, and a policy holding it.
LOBBY_ARM = {
    "arm": {
        "kind": "joints",
        "count": 2,
        "fields": ["pos"],
        "hz": 100,
        "command": "position",
        "hold_max_ms": 200,
    }
}
LOBBY_HOLD = {
    "hz": 20,
    "inputs": {"q": {"kind": "joints", "at": "arm", "fields": ["pos"]}},
    "actions": {
        "arm": {
            "kind": "joints",
            "at": "arm",
            "command": "position",
            "hz": 20,
            "chunk": 2,
        }
    },
}


def _free_base() -> int:
    """A loopback port whose next three are free too (a daemon's classes)."""
    for _ in range(50):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            base = probe.getsockname()[1]
        if base > 65531:
            continue
        try:
            held = []
            for offset in range(4):
                s = socket.socket()
                held.append(s)
                s.bind(("127.0.0.1", base + offset))
        except OSError:
            continue
        finally:
            for s in held:
                s.close()
        return base
    raise RuntimeError("no four free ports")


@pytest.fixture
def daemon(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """An rlmeshd on loopback, whose ``local.key`` this process attaches with."""
    # Short: its local channel is a Unix socket in there (SUN_LEN).
    state = tempfile.mkdtemp(prefix="rlmd-", dir="/tmp")
    monkeypatch.setenv("RLMESH_STATE_DIR", state)
    base = _free_base()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "rlmesh",
            "host",
            "--lobby",
            "--no-record",
            "--listen",
            f"127.0.0.1:{base}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        assert proc.stdout is not None
        line = proc.stdout.readline()
        assert "rlmesh host on" in line, line
        yield f"127.0.0.1:{base}"
    finally:
        proc.terminate()
        proc.wait(timeout=WAIT)
        shutil.rmtree(state, ignore_errors=True)


@pytest.mark.skipif(
    not hasattr(_rlmesh, "V2Scene"), reason="the lobby and scenes need a fleet build"
)
def test_a_scene_runs_a_lobby_robot_and_a_python_policy_on_one_machine(
    daemon: str, tmp_path: Path
) -> None:
    from rlmesh.realtime import _fleet

    scene_file = tmp_path / "bench.toml"
    scene_file.write_text(
        '[scene]\nformat = 1\nname = "bench"\nwait = "20s"\n\n'
        '[members.arm]\nrole = "robot"\n\n'
        '[members.pilot]\nrole = "controller"\ndevice = "local"\n'
    )

    def robot() -> str | None:
        with _fleet.wait_in_lobby(
            "arm", LOBBY_ARM, home=daemon, timeout=3 * WAIT
        ) as arm:
            while _fleet.leaving(arm) is None:
                arm.publish("arm", pos=np.zeros(2, dtype=np.float32))
                time.sleep(0.01)
            return _fleet.leaving(arm)

    thread, out = in_thread(robot)
    with _fleet.run_scene(scene_file, home=daemon) as scene:
        assert scene.wait_until("binding", timeout=WAIT)["members"][0] == {
            "name": "arm",
            "stage": "declared",
            "error": None,
        }
        assert scene.address is not None
        with rlmesh.Policy(LOBBY_HOLD, session=scene.address, name="pilot") as policy:
            policy.bind("arm")
            assert scene.wait_until("running", timeout=WAIT)["error"] is None
            stats = policy.run(lambda obs: {"arm": np.zeros((2, 2))}, duration=1.0)
            assert stats["starts"] > 0
            with pytest.raises(rlmesh.SessionRefused, match="arm is in run") as busy:
                _fleet.run_scene(scene_file, home=daemon)
            assert busy.value.status == "ROBOT_BUSY"
        assert scene.stop()["stage"] == "ended"
    thread.join(WAIT)
    assert not raised(out)
    assert out == ["scene_ended"]


EYE = {
    "eyes": {
        "kind": "joints",
        "names": ["pan", "tilt"],
        "fields": ["pos"],
        "hz": 200,
        "command": "position",
        "safe": "hold",
        "hold_max_ms": 200,
    }
}

EYES = {
    "hz": 25,
    "actions": {
        "one": {"kind": "joints", "at": "eye-1:eyes", "command": "position", "hz": 50},
        "two": {"kind": "joints", "at": "eye-2:eyes", "command": "position", "hz": 50},
    },
}


def test_one_policy_bound_with_no_robot_drives_two_robots(host: rlmesh.Host) -> None:
    eyes = [rlmesh.Robot(name, EYE, session=host) for name in ("eye-1", "eye-2")]
    try:
        eyes[0].arm()
        with rlmesh.Policy(EYES, session=host, name="tracker") as policy:
            binding = policy.bind()
            assert "binding tracker on scene" in str(binding)
            gaze = np.full((1, 2), 0.25, dtype=np.float32)
            thread, result = in_thread(
                lambda: policy.run(
                    lambda obs: {"one": gaze, "two": gaze}, duration=30.0
                )
            )
            driven = [False, False]

            def both() -> bool:
                for i, eye in enumerate(eyes):
                    eye.publish("eyes", pos=np.zeros(2, dtype=np.float32))
                    due = eye.due("eyes")
                    if (
                        due is not None
                        and due.mode == "driving"
                        and due.value is not None
                    ):
                        driven[i] |= abs(float(np.asarray(due.value)[0]) - 0.25) < 1e-6
                return all(driven)

            wait_until(
                "both eyes driven by one policy", lambda: raised(result) or both()
            )
            policy.stop()
            thread.join(WAIT)
            assert not raised(result)
    finally:
        for eye in eyes:
            eye.close()


def test_a_robot_held_in_an_armed_session_needs_a_rearm_by_name(
    host: rlmesh.Host,
) -> None:
    first = rlmesh.Robot("eye-1", EYE, session=host)
    late = None
    try:
        with pytest.raises(rlmesh.SessionRefused, match="arm it first"):
            first.rearm("eye-1")
        first.arm()
        first.arm()
        with pytest.raises(rlmesh.SessionRefused, match="ghost") as unknown:
            first.rearm("ghost")
        assert unknown.value.status == "INVALID_REQUEST"
        first.rearm("eye-1")
        # A robot declaring commandable parts in an armed session is held.
        late = rlmesh.Robot("eye-2", EYE, session=host)
        actions = {"two": EYES["actions"]["two"]}
        with rlmesh.Policy(
            {**EYES, "actions": actions}, session=host, name="tracker"
        ) as policy:
            policy.bind()
            with pytest.raises(
                rlmesh.SessionRefused, match=r'robot\.rearm\("eye-2"\)'
            ) as held:
                policy.acquire()
            assert held.value.status == "NOT_READY"
            first.arm()
            with pytest.raises(rlmesh.SessionRefused, match="eye-2"):
                policy.acquire()
            first.rearm("eye-2")
            wait_until("every lease", lambda: bool(policy.acquire()))
    finally:
        first.close()
        if late is not None:
            late.close()


_STALLED_ROBOT = """
import threading

import numpy as np
from rlmesh import realtime

ARM = {
    "arm": {
        "kind": "joints", "names": ["a", "b"], "fields": ["pos"], "hz": 200,
        "command": "position", "safe": "stop", "hold_max_ms": 100,
        "max_action_age_ms": 1000,
    },
}
POLICY = {
    "hz": 20,
    "inputs": {"q": {"kind": "joints", "at": "arm", "fields": ["pos"], "names": ["a", "b"]}},
    "actions": {
        "arm": {"kind": "joints", "at": "arm", "command": "position",
                "names": ["a", "b"], "hz": 100, "chunk": 10},
    },
}

with realtime.Host() as host:
    robot = realtime.Robot("arm", ARM, session=host)
    robot.arm()
    stop = threading.Event()
    driven = threading.Event()

    def loop():
        q = np.zeros(2)
        rate = realtime.Rate(200)
        while not stop.is_set():
            robot.publish("arm", pos=q)
            due = robot.due("arm")
            if due is not None and isinstance(due.value, np.ndarray):
                driven.set()
            rate.sleep()

    thread = threading.Thread(target=loop)
    thread.start()

    def predict(obs):
        if driven.is_set() and not predict.said:
            predict.said = True
            print("DRIVING", flush=True)
        return {"arm": np.repeat(obs["q"][None], 10, axis=0)}

    predict.said = False
    with realtime.Policy(POLICY, session=host, name="hold") as policy:
        policy.bind("arm")
        policy.run(predict, duration=3.0)
    stop.set()
    thread.join()
    modes = robot.edge_stats("arm")["modes"]
    robot.close()
print("SAFE", modes["safe"][1], "DRIVING", modes["driving"][1], flush=True)
"""


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs SIGSTOP (POSIX)")
def test_a_driven_robot_is_driven_again_after_its_process_stalls() -> None:
    """A robot, its policy and their host frozen together never latch SAFE.

    A loaded machine deschedules a process for longer than a lease or the
    robot's hold_max. Frozen, it received nothing: back, the edge holds while
    the chunks and renewals sent meanwhile come in, instead of latching SAFE
    on its first tick and waiting for a person to arm it again.
    """
    import signal

    child = subprocess.Popen(
        [sys.executable, "-c", _STALLED_ROBOT],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        if "DRIVING" not in child.stdout.readline():
            _out, err = child.communicate(timeout=30)
            if "Operation not permitted" in err:
                pytest.skip("local tcp bind is not permitted in this environment")
            raise AssertionError(f"the robot was never driven: {err}")
        os.kill(child.pid, signal.SIGSTOP)
        time.sleep(0.5)
        os.kill(child.pid, signal.SIGCONT)
        out, err = child.communicate(timeout=60)
        assert child.returncode == 0, err
        report = out.split()
        assert report[:2] == ["SAFE", "0"], f"{out!r}: {err}"
        # Driven before the stall and again after it.
        assert int(report[3]) >= 2, out
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()

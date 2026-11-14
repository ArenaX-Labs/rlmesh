"""Shadow runs: a policy that predicts before it may command.

One program reads a camera, publishes where it aims as a part of its own,
and commands two eyes. Run in shadow on a disarmed session, it predicts and
publishes its aim (a viewer reads it) and commands nothing; armed, the same
run starts commanding both eyes.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pytest
import rlmesh.realtime as rlmesh

WAIT = 10.0

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

#: Reads the camera, commands both eyes.
TRACKER = {
    "hz": 25,
    "inputs": {"frame": {"kind": "camera", "at": "cam:image"}},
    "actions": {
        "one": {"kind": "joints", "at": "eye-1:eyes", "command": "position", "hz": 50},
        "two": {"kind": "joints", "at": "eye-2:eyes", "command": "position", "hz": 50},
    },
}

#: The tracker's own part: where it aims.
AIM = {"aim": {"kind": "x/aim", "hz": 25, "fields": {"xy": ["float32", [2]]}}}

#: Reads the tracker's aim and commands nothing.
VIEWER = {
    "hz": 25,
    "inputs": {
        "aim": {
            "kind": "x/aim",
            "at": "tracker:aim",
            "fields": {"xy": ["float32", [2]]},
        }
    },
}


def wait_until(what: str, done: Callable[[], bool]) -> None:
    deadline = time.monotonic() + WAIT
    while not done():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        time.sleep(0.01)


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
    if out and isinstance(out[0], BaseException):
        raise out[0]
    return False


@pytest.fixture
def host() -> Iterator[rlmesh.Host]:
    with rlmesh.Host(lease_grace_ms=1000) as host:
        yield host


class Rig:
    """A camera and two eyes, each with its own loop; nothing armed."""

    def __init__(self, host: rlmesh.Host) -> None:
        self.camera = rlmesh.Robot(
            "cam", {"image": rlmesh.parts.camera(32, 24, hz=30)}, session=host
        )
        self.eyes = [
            rlmesh.Robot(name, EYE, session=host) for name in ("eye-1", "eye-2")
        ]
        #: What each eye's edge dispatched, by eye.
        self.driven: list[list[float]] = [[], []]
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def __enter__(self) -> Rig:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(WAIT)
        for robot in (self.camera, *self.eyes):
            robot.close()

    def _loop(self) -> None:
        rate = rlmesh.Rate(100)
        frame = np.zeros((24, 32, 3), np.uint8)
        tick = 0
        while not self._stop.is_set():
            if tick % 3 == 0:
                frame[:, :, 0] = tick % 255
                self.camera.publish("image", frame)
            for i, eye in enumerate(self.eyes):
                eye.publish("eyes", pos=np.zeros(2, np.float32))
                due = eye.due("eyes")
                if due is not None and due.mode == "driving" and due.value is not None:
                    self.driven[i].append(float(np.asarray(due.value)[0]))
            tick += 1
            rate.sleep()

    def chunks_received(self) -> int:
        """Chunks the eyes' edges took in, accepted or refused."""
        total = 0
        for eye in self.eyes:
            stats = eye.edge_stats("eyes")
            total += stats["accepted"] + stats["rejected"]
        return total


def test_one_program_shadows_on_a_disarmed_session_and_commands_once_armed(
    host: rlmesh.Host,
) -> None:
    aims: list[np.ndarray[Any, Any]] = []
    steps: list[rlmesh.Step] = []
    with (
        Rig(host) as rig,
        rlmesh.Policy(TRACKER, session=host, name="tracker", publishes=AIM) as tracker,
        rlmesh.Policy(VIEWER, session=host, name="viewer") as viewer,
    ):
        tracker.bind()

        def track(obs: rlmesh.Observation) -> dict[str, Any]:
            # Where the frame's red channel says to look.
            x = float(obs["frame"][0, 0, 0]) / 255.0
            tracker.publish("aim", xy=np.array([x, 0.5], np.float32))
            gaze = np.array([[0.25, 0.0]], np.float32)
            return {"one": gaze, "two": gaze}

        thread, out = in_thread(
            lambda: tracker.run(
                track, duration=60.0, mode=rlmesh.RunMode.SHADOW, on_step=steps.append
            )
        )
        # The viewer is bound to the tracker's aim: it flows while disarmed.
        viewer.bind()
        view, viewed = in_thread(
            lambda: viewer.run(lambda obs: aims.append(obs["aim"]) or {}, duration=60.0)
        )
        wait_until(
            "the viewer reads the tracker's aim while disarmed",
            lambda: raised(out) or raised(viewed) or len(aims) >= 10,
        )
        wait_until(
            "shadow predictions", lambda: tracker.stats()["shadow_predictions"] >= 10
        )
        stats = tracker.stats()
        assert stats["chunks_sent"] == 0, stats
        assert stats["skipped"]["no_lease"] == 0, stats
        assert stats["waiting"] is not None
        assert rig.chunks_received() == 0
        assert not tracker.holds
        shadow = steps[0]
        assert shadow.shadow
        assert shadow.parts == {"one": "eye-1:eyes", "two": "eye-2:eyes"}
        assert shadow.seqs == {"one": None, "two": None}
        # The actions as each eye's edge would take them.
        pos = shadow.actions["one"]
        assert pos is not None
        np.testing.assert_allclose(pos["pos"], [[0.25, 0.0]])
        events = tracker.events()
        shadowing = [e for e in events if e["kind"] == "shadowing"]
        assert len(shadowing) == 1, events
        assert shadowing[0]["message"].startswith("shadowing: the session is DISARMED")

        # Armed: the same run commands both eyes, without a restart.
        rig.eyes[0].arm()
        wait_until(
            "both eyes driven by the same run",
            lambda: raised(out) or all(len(d) >= 5 for d in rig.driven),
        )
        assert thread.is_alive()
        assert all(abs(v - 0.25) < 1e-6 for d in rig.driven for v in d)
        assert any(e["kind"] == "commanding" for e in tracker.events())
        sent = [s for s in steps if not s.shadow]
        assert sent and all(isinstance(s.seqs["one"], int) for s in sent)
        assert tracker.stats()["chunks_sent"] > 0
        n = len(aims)
        wait_until("the aim still flows", lambda: len(aims) > n)

        tracker.stop()
        viewer.stop()
        thread.join(WAIT)
        view.join(WAIT)
        assert not raised(out) and not raised(viewed)


def test_in_command_mode_the_viewer_waits_for_the_aim_until_the_session_is_armed(
    host: rlmesh.Host,
) -> None:
    aims: list[np.ndarray[Any, Any]] = []
    with (
        Rig(host) as rig,
        rlmesh.Policy(TRACKER, session=host, name="tracker", publishes=AIM) as tracker,
        rlmesh.Policy(VIEWER, session=host, name="viewer") as viewer,
    ):
        tracker.bind()

        def track(obs: rlmesh.Observation) -> dict[str, Any]:
            tracker.publish("aim", xy=np.array([0.5, 0.5], np.float32))
            gaze = np.array([[0.25, 0.0]], np.float32)
            return {"one": gaze, "two": gaze}

        thread, out = in_thread(lambda: tracker.run(track, duration=60.0))
        viewer.bind()
        view, viewed = in_thread(
            lambda: viewer.run(lambda obs: aims.append(obs["aim"]) or {}, duration=60.0)
        )
        wait_until(
            "the tracker waits to be armed",
            lambda: raised(out) or tracker.stats()["skipped"]["no_lease"] >= 5,
        )
        wait_until(
            "the viewer waits for the aim",
            lambda: raised(viewed) or viewer.stats()["skipped"]["waiting_for"] >= 5,
        )
        assert aims == []
        assert tracker.stats()["completions"] == 0
        rig.eyes[0].arm()
        wait_until("the aim flows once armed", lambda: raised(viewed) or len(aims) >= 5)
        wait_until("both eyes driven", lambda: all(len(d) >= 5 for d in rig.driven))
        tracker.stop()
        viewer.stop()
        thread.join(WAIT)
        view.join(WAIT)
        assert not raised(out) and not raised(viewed)


def test_a_shadow_run_surfaces_an_action_the_binding_refuses(host: rlmesh.Host) -> None:
    with (
        Rig(host) as rig,
        rlmesh.Policy(TRACKER, session=host, name="tracker") as tracker,
    ):
        tracker.bind()

        def wrong(obs: rlmesh.Observation) -> dict[str, Any]:
            gaze = np.zeros((1, 2), np.float32)
            return {"one": gaze, "two": np.zeros((1, 3), np.float32)}

        with pytest.raises(ValueError, match="two"):
            tracker.run(wrong, duration=30.0, mode=rlmesh.RunMode.SHADOW)
        assert rig.chunks_received() == 0
        with pytest.raises(TypeError, match="RunMode"):
            tracker.run(wrong, mode="shadow")  # type: ignore[arg-type]

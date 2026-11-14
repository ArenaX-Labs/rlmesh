"""The session's live link health, read from a robot and from policies.

Two eyes, a tracker driving both, a viewer that only reads eye 1, and a
prober that sends eye 2 a value that is not a number: every participant's
link, and each eye's refusals by reason, reach a viewer that neither
publishes nor commands.
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

TRACKER = {
    "hz": 25,
    "actions": {
        "one": {
            "kind": "joints",
            "at": "eye-1:eyes",
            "command": "position",
            "hz": 50,
            "chunk": 2,
        },
        "two": {
            "kind": "joints",
            "at": "eye-2:eyes",
            "command": "position",
            "hz": 50,
            "chunk": 2,
        },
    },
}

VIEWER = {
    "hz": 25,
    "inputs": {"gaze": {"kind": "joints", "at": "eye-1:eyes", "fields": ["pos"]}},
}

PROBE = {
    "hz": 25,
    "inputs": {"gaze": {"kind": "joints", "at": "eye-2:eyes", "fields": ["pos"]}},
    "actions": {
        "two": {
            "kind": "joints",
            "at": "eye-2:eyes",
            "command": "position",
            "hz": 50,
            "chunk": 2,
        }
    },
}


def wait_until(what: str, done: Callable[[], bool]) -> None:
    deadline = time.monotonic() + WAIT
    while not done():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        time.sleep(0.01)


@pytest.fixture
def host() -> Iterator[rlmesh.Host]:
    with rlmesh.Host() as host:
        yield host


class Eyes:
    """Two eyes, each publishing its gaze and taking what is due."""

    def __init__(self, host: rlmesh.Host) -> None:
        self.eyes = [
            rlmesh.Robot(name, EYE, session=host) for name in ("eye-1", "eye-2")
        ]
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def __enter__(self) -> Eyes:
        self.eyes[0].arm()
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(WAIT)
        for eye in self.eyes:
            eye.close()

    def _loop(self) -> None:
        rate = rlmesh.Rate(200)
        while not self._stop.is_set():
            for eye in self.eyes:
                eye.publish("eyes", pos=np.zeros(2, np.float32))
                eye.due("eyes")
            rate.sleep()


def test_a_viewer_reads_every_link_and_each_robots_refusals(host: rlmesh.Host) -> None:
    def gaze(obs: rlmesh.Observation) -> dict[str, Any]:
        chunk = np.full((2, 2), 0.25, np.float32)
        return {"one": chunk, "two": chunk}

    with (
        Eyes(host) as rig,
        rlmesh.Policy(TRACKER, session=host, name="tracker") as tracker,
        rlmesh.Policy(VIEWER, session=host, name="viewer") as viewer,
        rlmesh.Policy(
            PROBE, session=host, name="prober", priority="operator"
        ) as prober,
    ):
        tracker.bind()
        viewer.bind()
        prober.bind()
        out: list[object] = []
        thread = threading.Thread(
            target=lambda: out.append(tracker.run(gaze, duration=30.0)), daemon=True
        )
        thread.start()
        wait_until("the tracker sending", lambda: tracker.stats()["chunks_sent"] > 5)

        # The prober takes eye 2 and sends it a value that is not a number.
        prober.acquire()
        two = rig.eyes[1]

        def refused() -> bool:
            if two.edge_stats("eyes")["rejected_by"].get("invalid_value", 0) > 0:
                return True
            try:
                obs = prober.observe()
            except rlmesh.StartSkipped:
                return False
            prober.send({"two": np.full((2, 2), np.nan, np.float32)}, obs)
            return False

        wait_until("eye 2 refusing a chunk for its value", refused)

        # The viewer neither publishes nor commands: its first read
        # subscribes and waits for a publication.
        health = viewer.health()
        deadline = time.monotonic() + WAIT
        while (
            health["links"]
            .get("eye-2", {})
            .get("rejected_by", {})
            .get("invalid_value", 0)
            == 0
        ):
            assert time.monotonic() < deadline, health
            health = viewer.health(wait=True)
        links = health["links"]
        assert {"eye-1", "eye-2", "tracker", "viewer", "prober"} <= set(links), links
        for name in ("eye-1", "eye-2", "tracker", "viewer", "prober"):
            link = links[name]
            assert link["heartbeats"] > 0, link
            assert link["rtt_ms"] is not None and link["rtt_ms"] > 0, link
            assert link["offset_error_ms"] is not None, link
            assert link["health"] == "ok", link
        assert links["eye-2"]["rejected"] >= 1
        assert links["eye-1"]["rejected_by"] == {}
        assert links["eye-1"]["last_chunk_accepted"] > 0
        assert links["eye-1"]["published"]["eye-1:eyes"]["published"] > 0
        assert links["tracker"]["run"]["chunks_sent"] > 0
        assert links["viewer"]["run"] is not None
        assert links["viewer"]["last_seq_seen"]["eye-1:eyes"] > 0
        assert links["eye-1"]["run"] is None

        # A robot reads the same; the next one comes later.
        now = rig.eyes[0].health()
        later = rig.eyes[0].health(wait=True)
        assert later["host_ns"] > now["host_ns"]
        assert set(later["links"]) == set(links)
        with pytest.raises(TimeoutError, match="health"):
            viewer.health(wait=True, timeout=0.001)

        tracker.stop()
        thread.join(WAIT)
        assert isinstance(out[0], dict)

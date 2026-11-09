"""Export recordings as a LeRobot dataset.

:func:`export` takes either

- a recording: an MCAP file (:class:`rlmesh.realtime.SessionRecorder`'s), or a
  directory of segments (what ``rlmesh host`` records under
  ``rolling/{session}/``, or an incident it kept), written as one episode of
  ``task``; or
- a collected dataset, ``collected/{dataset}`` under the recordings
  directory: one episode per take (its segments and an ``episode.json`` with
  ``task``, ``start_ns``, ``end_ns`` and ``success``).

Each episode is resampled at ``fps`` on capture time (session time, as the
recorder mapped each sample's capture), not on when the recorder received
it. A frame holds every stream's newest sample captured at or before the
frame's time:

- ``observation.state``: every numeric stream but the ``_applied`` ones,
  flattened and concatenated in topic order (or only the ``state``
  topics). Left out when there are none: a robot with no state stream (an
  eye's servos, driven open loop) still exports.
- ``action``: every ``_applied`` stream, what the edges dispatched. An edge
  that holds dispatches nothing; its last action carries over, so one idle
  robot never drops the others' frames.
- ``observation.images.{path}``: every camera, as video.

Frames before every stream has a value are left out. Takes marked failed
are skipped unless ``include_failed``. Recordings are decoded as they are
read, one segment held at a time. The dataset is written through the
``lerobot`` package (``pip install lerobot``), in the format that package
writes::

    python -m rlmesh.realtime.lerobot RECORDING OUT --repo-id me/eyes --task "track faces"
    python -m rlmesh.realtime.lerobot COLLECTED_DIR OUT --repo-id me/ur10e-sweeps --fps 30
"""

from __future__ import annotations

import argparse
import heapq
import importlib
import itertools
import json
import os
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .._rlmesh import V2Recorder

__all__ = ["export"]

#: ``(name, dtype, shape, bytes)``, as :meth:`V2Recorder.samples` gives it.
_Fields = list[tuple[str, str, list[int], bytes]]

#: How late a sample may reach the recorder after its capture and still be
#: placed by its capture time: samples are reordered within this window.
REORDER_NS = 1_000_000_000


@dataclass(frozen=True)
class _Take:
    path: Path
    task: str
    #: Session time; ``None``: from the first sample, or to the last.
    start_ns: int | None
    end_ns: int | None


def _takes(source: Path, include_failed: bool, task: str = "") -> list[_Take]:
    """The episodes under ``source``: a collected dataset's takes, or one recording."""
    metas = sorted(source.glob("*/episode.json")) if source.is_dir() else []
    if not metas:
        if source.is_dir() and not any(source.glob("*.mcap")):
            return []
        return [_Take(source, task, None, None)]
    takes: list[_Take] = []
    for meta in metas:
        episode = json.loads(meta.read_text())
        if episode.get("success") is False and not include_failed:
            continue
        takes.append(
            _Take(
                meta.parent,
                episode.get("task") or task,
                int(episode["start_ns"]),
                int(episode["end_ns"]),
            )
        )
    return takes


def _is_image(fields: _Fields) -> bool:
    return (
        len(fields) == 1
        and fields[0][1] == "uint8"
        and len(fields[0][2]) == 3
        and fields[0][2][2] == 3
    )


def _width(fields: _Fields) -> int:
    """How many numbers ``fields`` flatten to (text left out)."""
    return sum(
        int(np.prod(shape)) if shape else 1
        for _, dtype, shape, _ in fields
        if dtype != "utf8"
    )


def _numbers(fields: _Fields) -> np.ndarray[Any, Any]:
    parts = [
        np.frombuffer(data, dtype=dtype).astype(np.float32).ravel()
        for _, dtype, _, data in fields
        if dtype != "utf8"
    ]
    return np.concatenate(parts) if parts else np.zeros(0, np.float32)


def _names(topic: str, fields: _Fields) -> list[str]:
    names: list[str] = []
    for name, dtype, shape, _ in fields:
        if dtype == "utf8":
            continue
        n = int(np.prod(shape)) if shape else 1
        names += (
            [f"{topic}.{name}"]
            if n == 1
            else [f"{topic}.{name}[{i}]" for i in range(n)]
        )
    return names


def _by_capture(
    samples: Iterable[tuple[str, int, int, _Fields]],
) -> Iterator[tuple[int, str, _Fields] | int]:
    """Samples as ``(capture_ns, topic, fields)`` in capture order, and watermarks.

    A watermark (an ``int``) follows each sample read: every sample
    captured at or before it has been yielded by then. Samples reach the recorder in log order, up to :data:`REORDER_NS` after
    their capture; a later one is placed when it arrives.
    """
    heap: list[tuple[int, int, str, _Fields]] = []
    order = itertools.count()
    for topic, log_ns, capture_ns, fields in samples:
        heapq.heappush(heap, (capture_ns, next(order), topic, fields))
        watermark = log_ns - REORDER_NS
        while heap and heap[0][0] <= watermark:
            capture, _, t, f = heapq.heappop(heap)
            yield capture, t, f
        yield watermark
    while heap:
        capture, _, t, f = heapq.heappop(heap)
        yield capture, t, f


def _frames(
    take: _Take, fps: int, *, images: bool = True
) -> Iterator[dict[str, _Fields]]:
    """The newest fields per topic at each frame time of ``take``.

    An ``_applied`` tick that dispatched nothing keeps the topic's last
    dispatched fields; until the first, the topic has none.
    """
    step = round(1e9 / fps)
    end = take.end_ns
    at = take.start_ns
    latest: dict[str, _Fields] = {}
    # Each topic's newest capture: a sample later than the reorder window
    # never replaces a newer one.
    newest: dict[str, int] = {}
    events = _by_capture(V2Recorder.samples(os.fspath(take.path), images))
    for event in events:
        # A watermark: every frame up to it is complete.
        horizon = event if isinstance(event, int) else event[0] - 1
        while at is not None and at <= horizon and (end is None or at <= end):
            if latest:
                yield dict(latest)
            at += step
        if isinstance(event, int):
            continue
        capture, topic, fields = event
        if at is None:
            # A whole recording starts at its first sample.
            at = capture
        if (end is not None and capture > end) or capture < newest.get(topic, capture):
            continue
        newest[topic] = capture
        if fields:
            latest[topic] = fields
        else:
            latest.setdefault(topic, [])
    if at is None:
        return
    last = end if end is not None else at
    while at <= last and latest:
        yield dict(latest)
        at += step


def _layout(
    first: dict[str, _Fields], state: Sequence[str] | None
) -> tuple[list[str], list[str], dict[str, str]]:
    """State topics, action topics and camera topics (by feature key)."""
    robots = {topic.split("/", 1)[0] for topic in first}
    numeric = sorted(
        t for t, f in first.items() if f and not _is_image(f) and _width(f)
    )
    actions = [t for t in numeric if t.endswith("/_applied")]
    states = list(state) if state else [t for t in numeric if t not in actions]
    cameras: dict[str, str] = {}
    for topic in sorted(t for t, f in first.items() if f and _is_image(f)):
        path = topic.split("/", 1)[1] if len(robots) == 1 else topic
        cameras[f"observation.images.{path.replace('/', '_')}"] = topic
    return states, actions, cameras


def export(
    source: str | os.PathLike[str],
    out: str | os.PathLike[str],
    *,
    repo_id: str,
    fps: int = 30,
    task: str = "",
    state: Sequence[str] | None = None,
    include_failed: bool = False,
) -> int:
    """Write ``source`` (a recording, or a collected dataset) as a LeRobot dataset at ``out``.

    Returns how many episodes were written. ``task`` names a recording's
    episode, and any take without one.
    """
    lerobot: Any = importlib.import_module("lerobot.datasets.lerobot_dataset")
    takes = _takes(Path(source), include_failed, task)
    if not takes:
        raise ValueError(f"no episodes to export under {source}")
    # The features are every stream the first take holds, as last seen with
    # a value; camera frames are left encoded for this pass.
    last: dict[str, _Fields] = {}
    for frame in _frames(takes[0], fps, images=False):
        last.update((t, f) for t, f in frame.items() if f)
    if not last:
        raise ValueError(f"{takes[0].path} holds no samples")
    states, actions, cameras = _layout(last, state)
    if not actions:
        raise ValueError(
            "a LeRobot dataset needs an action: no `_applied` stream "
            f"(what an edge dispatched) in {sorted(last)}"
        )
    features: dict[str, dict[str, Any]] = {}
    if states:
        features["observation.state"] = {
            "dtype": "float32",
            "shape": (sum(_width(last[t]) for t in states),),
            "names": [n for t in states for n in _names(t, last[t])],
        }
    features["action"] = {
        "dtype": "float32",
        "shape": (sum(_width(last[t]) for t in actions),),
        "names": [n for t in actions for n in _names(t, last[t])],
    }
    for key, topic in cameras.items():
        features[key] = {
            "dtype": "video",
            "shape": tuple(last[topic][0][2]),
            "names": ["height", "width", "channels"],
        }
    robots = sorted({topic.split("/", 1)[0] for topic in last})
    written: Any = lerobot.LeRobotDataset.create(
        repo_id,
        fps,
        features=features,
        root=out,
        robot_type="+".join(robots),
        use_videos=True,
    )
    wanted = states + actions + list(cameras.values())
    count = 0
    for take in takes:
        frames = 0
        for frame in _frames(take, fps):
            # Frames before every stream's first value are left out.
            if any(not frame.get(t) for t in wanted):
                continue
            row: dict[str, Any] = {
                "action": np.concatenate([_numbers(frame[t]) for t in actions]),
                "task": take.task,
            }
            if states:
                row["observation.state"] = np.concatenate(
                    [_numbers(frame[t]) for t in states]
                )
            for key, topic in cameras.items():
                _, _, shape, data = frame[topic][0]
                pixels: Any = np.frombuffer(data, np.uint8)
                row[key] = pixels.reshape(shape)
            written.add_frame(row)
            frames += 1
        if frames:
            written.save_episode()
            count += 1
    written.finalize()
    return count


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m rlmesh.realtime.lerobot",
        description="Export a recording, or collected episodes, as a LeRobot dataset.",
    )
    parser.add_argument(
        "source",
        help="an MCAP file or a directory of segments (one episode), "
        "or collected/{dataset} under the recordings directory",
    )
    parser.add_argument("out", help="the new dataset's directory")
    parser.add_argument(
        "--repo-id", required=True, help="the dataset's Hub id, user/name"
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--task", default="", help="the task of a recording's episode")
    parser.add_argument(
        "--state",
        nargs="+",
        metavar="TOPIC",
        help="observation.state from only these {robot}/{path} streams",
    )
    parser.add_argument("--include-failed", action="store_true")
    args = parser.parse_args(argv)
    count = export(
        args.source,
        args.out,
        repo_id=args.repo_id,
        fps=args.fps,
        task=args.task,
        state=args.state,
        include_failed=args.include_failed,
    )
    print(f"{count} episodes written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

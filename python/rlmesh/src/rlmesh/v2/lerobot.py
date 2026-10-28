"""Export collected episodes as a LeRobot dataset.

The cockpit's episode controls keep each take under
``{recordings}/collected/{dataset}/{unix_ms}/``: the session's MCAP segments
and an ``episode.json`` (``task``, ``start_ns``, ``end_ns``, ``success``).
:func:`export` turns a dataset directory into one LeRobot episode per take,
resampled at ``fps``: each frame holds the newest sample of every stream at
or before its time.

- ``observation.state``: every numeric stream but the ``_applied`` ones,
  flattened and concatenated in topic order (or only the ``state`` topics).
- ``action``: every ``_applied`` stream, what the edges dispatched.
- ``observation.images.{path}``: every camera, as video.

Takes marked failed are skipped unless ``include_failed``. The dataset is
written through the ``lerobot`` package (``pip install lerobot``), in the
format that package writes::

    python -m rlmesh.v2.lerobot COLLECTED_DIR OUT --repo-id me/ur10e-sweeps --fps 30

``COLLECTED_DIR`` is ``collected/{dataset}`` under the recordings directory
(by default ``~/.local/share/rlmesh/recordings`` on Linux).
"""

from __future__ import annotations

import argparse
import collections
import importlib
import json
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .._rlmesh import V2Recorder

__all__ = ["export"]

#: ``(name, dtype, shape, bytes)``, as :meth:`V2Recorder.samples` gives it.
_Fields = list[tuple[str, str, list[int], bytes]]


@dataclass(frozen=True)
class _Take:
    dir: Path
    task: str
    start_ns: int
    end_ns: int


def _takes(dataset: Path, include_failed: bool) -> list[_Take]:
    takes: list[_Take] = []
    for meta in sorted(dataset.glob("*/episode.json")):
        episode = json.loads(meta.read_text())
        if episode.get("success") is False and not include_failed:
            continue
        takes.append(
            _Take(
                meta.parent,
                episode.get("task") or "",
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


def _frames(take: _Take, fps: float) -> Iterator[dict[str, _Fields]]:
    """The newest fields per topic at each frame time of ``take``."""
    step = round(1e9 / fps)
    at = take.start_ns
    latest: dict[str, _Fields] = {}
    for segment in sorted(take.dir.glob("*.mcap")):
        # One segment at a time: a minute of decoded camera frames at most.
        for topic, log_ns, fields in V2Recorder.samples(os.fspath(segment)):
            while at < log_ns and at <= take.end_ns:
                if latest:
                    yield dict(latest)
                at += step
            latest[topic] = fields
    while at <= take.end_ns and latest:
        yield dict(latest)
        at += step


def _layout(
    first: dict[str, _Fields], state: Sequence[str] | None
) -> tuple[list[str], list[str], dict[str, str]]:
    """State topics, action topics and camera topics (by feature key)."""
    robots = {topic.split("/", 1)[0] for topic in first}
    numeric = sorted(
        t for t, f in first.items() if not _is_image(f) and _numbers(f).size
    )
    actions = [t for t in numeric if t.endswith("/_applied")]
    states = list(state) if state else [t for t in numeric if t not in actions]
    cameras: dict[str, str] = {}
    for topic in sorted(t for t, f in first.items() if _is_image(f)):
        path = topic.split("/", 1)[1] if len(robots) == 1 else topic
        cameras[f"observation.images.{path.replace('/', '_')}"] = topic
    return states, actions, cameras


def export(
    dataset: str | os.PathLike[str],
    out: str | os.PathLike[str],
    *,
    repo_id: str,
    fps: float = 30.0,
    state: Sequence[str] | None = None,
    include_failed: bool = False,
) -> int:
    """Write the takes under ``dataset`` as a LeRobot dataset at ``out``; returns how many."""
    lerobot: Any = importlib.import_module("lerobot.datasets.lerobot_dataset")
    takes = _takes(Path(dataset), include_failed)
    if not takes:
        raise ValueError(f"no episodes to export under {dataset}")
    # The features are every stream the first take's last frame holds (its
    # first may precede some streams' first sample).
    tail = collections.deque(_frames(takes[0], fps), maxlen=1)
    if not tail:
        raise ValueError(f"{takes[0].dir} holds no samples")
    last = tail[0]
    states, actions, cameras = _layout(last, state)
    features: dict[str, dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (sum(_numbers(last[t]).size for t in states),),
            "names": [n for t in states for n in _names(t, last[t])],
        },
        "action": {
            "dtype": "float32",
            "shape": (sum(_numbers(last[t]).size for t in actions),),
            "names": [n for t in actions for n in _names(t, last[t])],
        },
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
        round(fps),
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
            if any(t not in frame for t in wanted):
                continue
            row: dict[str, Any] = {
                "observation.state": np.concatenate(
                    [_numbers(frame[t]) for t in states]
                ),
                "action": np.concatenate([_numbers(frame[t]) for t in actions]),
                "task": take.task,
            }
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
        prog="python -m rlmesh.v2.lerobot",
        description="Export collected episodes as a LeRobot dataset.",
    )
    parser.add_argument(
        "dataset", help="collected/{dataset} under the recordings directory"
    )
    parser.add_argument("out", help="the new dataset's directory")
    parser.add_argument(
        "--repo-id", required=True, help="the dataset's Hub id, user/name"
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--state",
        nargs="+",
        metavar="TOPIC",
        help="observation.state from only these {robot}/{path} streams",
    )
    parser.add_argument("--include-failed", action="store_true")
    args = parser.parse_args(argv)
    count = export(
        args.dataset,
        args.out,
        repo_id=args.repo_id,
        fps=args.fps,
        state=args.state,
        include_failed=args.include_failed,
    )
    print(f"{count} episodes written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

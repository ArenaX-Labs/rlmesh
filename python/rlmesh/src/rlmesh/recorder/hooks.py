"""Live capture: a :class:`~rlmesh.RunHooks` that records into a workload.

This is the ``session().run(hooks=...)`` tie-in. It observes the Python-driven eval
loop (the served / hand-driven path) and, per episode, records the outcome plus any
media the env exposes -- per-step image frames read through the session's own reader,
and/or an env-produced video file whose path the env leaves in the step ``info``.
Pure Rust ``.run()`` never surfaces per-step observations, so frame capture requires
this path; env-video-path capture works there too, since it is just an ``info`` key.

All capture is best-effort: a frame that fails to read is skipped, never fatal.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, cast

from .._models import RunHooks
from .constants import DEFAULT_CAMERA
from .frames import read_image, to_frame
from .schema import EpisodeRecord, MediaRef

if TYPE_CHECKING:
    from .._models import EpisodeResult, RunResult, StepEvent
    from .media import MediaStager
    from .schema import WorkloadRecord

FrameFn = Callable[["StepEvent"], Any]


class CaptureHooks(RunHooks):
    """Accumulate episodes (and staged media) into one :class:`WorkloadRecord`.

    Internal to the recorder; construct one via :meth:`Recorder.capture`, not directly.
    """

    def __init__(
        self,
        *,
        workload: WorkloadRecord,
        stager: MediaStager,
        cameras: list[str],
        frame_fn: FrameFn | None,
        video_keys: tuple[str, ...],
        included_in_metrics: bool,
    ) -> None:
        self._workload = workload
        self._stager = stager
        self._cameras = tuple(cameras)
        self._frame_fn = frame_fn
        self._video_keys = video_keys
        self._included = included_in_metrics
        #: camera -> frames buffered for the in-flight episode.
        self._buffers: dict[str, list[Any]] = {}
        self._video_path: str | None = None

    def on_episode_start(self, *, episode: int, seed: int | None) -> None:
        """Reset the per-episode media buffers."""
        self._buffers = {}
        self._video_path = None

    def on_step(self, event: StepEvent) -> None:
        """Note an env-produced video path and buffer any captured frames."""
        info = event.info
        for key in self._video_keys:
            value = info.get(key)
            if isinstance(value, str) and value:
                self._video_path = value
                break
        if self._frame_fn is not None:
            self._collect(self._frame_fn(event))
        elif self._cameras:
            for camera in self._cameras:
                frame = read_image(event, camera)
                if frame is not None:
                    self._buffers.setdefault(camera, []).append(frame)

    def _collect(self, produced: Any) -> None:
        """Buffer what a ``frame_fn`` returned.

        Accepts a single array (one default camera), a ``{camera: array}`` map, or
        ``None`` (skip the step).
        """
        if produced is None:
            return
        if isinstance(produced, Mapping):
            produced_map = cast("Mapping[str, object]", produced)
            for camera, value in produced_map.items():
                frame = to_frame(value)
                if frame is not None:
                    self._buffers.setdefault(camera, []).append(frame)
            return
        frame = to_frame(produced)
        if frame is not None:
            self._buffers.setdefault(DEFAULT_CAMERA, []).append(frame)

    def on_episode_end(self, result: EpisodeResult) -> None:
        """Stage this episode's media and append its record to the workload."""
        index = len(self._workload.episodes)
        media: list[MediaRef] = []
        for camera, frames in self._buffers.items():
            if frames:
                media.append(
                    self._stager.frames(
                        task=self._workload.task,
                        episode_index=index,
                        camera=camera,
                        frames=frames,
                    )
                )
        if self._video_path is not None:
            ref = self._stager.video(
                task=self._workload.task,
                episode_index=index,
                camera=DEFAULT_CAMERA,
                source=self._video_path,
            )
            if ref is not None:
                media.append(ref)
        self._workload.episodes.append(
            EpisodeRecord.from_result(
                result,
                index=index,
                included_in_metrics=self._included,
                media=tuple(media),
            )
        )
        self._buffers = {}
        self._video_path = None

    def on_run_end(self, result: RunResult) -> None:
        """No-op: episodes are accumulated per-episode, nothing to finalize."""

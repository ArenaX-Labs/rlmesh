"""Live capture: a :class:`~rlmesh.RunHooks` that records into a workload.

This is the ``session().run(hooks=...)`` tie-in. It observes the Python-driven eval
loop and, per episode, records the outcome plus any media the env exposes -- per-step
image frames read through the session's own reader and streamed straight into a native
AV1 writer (one frame in memory at a time), and/or an env-produced video file whose
path the env leaves in the step ``info``. Pure Rust ``.run()`` never surfaces per-step
observations, so frame capture needs this path; env-video capture works there too.

All capture is best-effort: a camera that fails to read or encode is warned once and
dropped, and the episode's outcome is still recorded -- capture never aborts the run.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .._models import RunHooks
from .constants import DEFAULT_CAMERA, RENDER_CAMERA
from .frames import image_roles, read_frame, render_source, to_frame_bytes
from .schema import EpisodeRecord, MediaRef

if TYPE_CHECKING:
    from collections.abc import Callable

    from .._models import EpisodeResult, RunResult, Session, StepEvent
    from .._rlmesh import PyVideoWriter
    from .media import MediaStager
    from .schema import WorkloadRecord


@dataclass
class _Cam:
    """A camera's in-flight AV1 writer plus where its file will land in the bundle."""

    writer: PyVideoWriter
    staged: str
    rel: str


def _render_camera_label(roles: list[str]) -> str:
    """Return a render label that cannot shadow a declared image role."""
    taken = set(roles)
    if RENDER_CAMERA not in taken:
        return RENDER_CAMERA
    label = f"{RENDER_CAMERA}()"
    if label not in taken:
        return label
    index = 2
    while True:
        label = f"{RENDER_CAMERA}({index})"
        if label not in taken:
            return label
        index += 1


class CaptureHooks(RunHooks):
    """Accumulate episodes (and streamed media) into one :class:`WorkloadRecord`.

    Internal to the recorder; construct one via :meth:`Recorder.capture`, not directly.
    """

    def __init__(
        self,
        *,
        workload: WorkloadRecord,
        stager: MediaStager,
        prefix: str,
        cameras: list[str] | None,
        session: Session[object, object] | None,
        video_keys: tuple[str, ...],
        included_in_metrics: bool,
    ) -> None:
        self._workload = workload
        self._stager = stager
        self._prefix = prefix
        self._cameras = tuple(cameras) if cameras is not None else ()
        self._session = session
        self._video_keys = video_keys
        self._included = included_in_metrics
        #: An explicit ``cameras`` list (even empty) is honored as-is; ``None`` defers
        #: discovery to the first step, when the session's contract is populated.
        self._resolved = cameras is not None
        #: Memoized ``render()`` thunk (resolved once from the session), or ``None``.
        self._render: Callable[[], object] | None = None
        self._render_resolved = False
        self._render_camera = RENDER_CAMERA if cameras is not None else None
        #: camera -> in-flight writer for the current episode.
        self._writers: dict[str, _Cam] = {}
        #: cameras that raised while encoding -- dropped for the whole run.
        self._disabled: set[str] = set()
        #: cameras that produced at least one frame (to flag silent no-capture).
        self._captured: set[str] = set()
        self._video_path: str | None = None

    def on_episode_start(self, *, episode: int, seed: int | None) -> None:
        """Reset the per-episode writers and env-video path."""
        self._writers = {}
        self._video_path = None

    def _render_thunk(self) -> Callable[[], object] | None:
        if not self._render_resolved:
            self._render_resolved = True
            if self._session is not None:
                self._render = render_source(self._session)
        return self._render

    def _resolve_cameras(self) -> None:
        if self._resolved:
            return
        self._resolved = True
        if self._session is None:
            return
        names: list[str] = []
        roles = image_roles(getattr(self._session, "_contract", None))
        if self._render_thunk() is not None:
            self._render_camera = _render_camera_label(roles)
            names.append(self._render_camera)
        else:
            self._render_camera = None
        names.extend(roles)
        self._cameras = tuple(names)
        if not self._cameras:
            warnings.warn(
                "recorder: no render() or image roles discovered from the session; "
                "recording metrics only",
                stacklevel=2,
            )

    def _read_source(
        self, event: StepEvent, camera: str
    ) -> tuple[bytes, int, int, int] | None:
        if camera == self._render_camera:
            thunk = self._render_thunk()
            if thunk is None:
                return None
            try:
                value = thunk()
            except Exception:
                return None
            return to_frame_bytes(value)
        return read_frame(event, camera)

    def on_step(self, event: StepEvent) -> None:
        """Note an env-produced video path and stream any captured frames."""
        self._resolve_cameras()
        info = event.info
        for key in self._video_keys:
            value = info.get(key)
            if isinstance(value, str) and value:
                self._video_path = value
                break
        for camera in self._cameras:
            if camera in self._disabled:
                continue
            try:
                frame = self._read_source(event, camera)
                if frame is not None:
                    self._write(camera, frame)
            except Exception as exc:
                warnings.warn(
                    f"recorder: disabling camera {camera!r} after an error: {exc}",
                    stacklevel=2,
                )
                self._disabled.add(camera)
                self._writers.pop(camera, None)

    def _write(self, camera: str, frame: tuple[bytes, int, int, int]) -> None:
        data, width, height, channels = frame
        cam = self._writers.get(camera)
        if cam is None:
            writer, staged, rel = self._stager.open_video(
                prefix=self._prefix,
                episode_index=len(self._workload.episodes),
                camera=camera,
                width=width,
                height=height,
            )
            cam = _Cam(writer=writer, staged=staged, rel=rel)
            self._writers[camera] = cam
        cam.writer.write_frame(data, width, height, channels)
        self._captured.add(camera)

    def on_episode_end(self, result: EpisodeResult) -> None:
        """Finalize this episode's media and append its record to the workload."""
        index = len(self._workload.episodes)
        media: list[MediaRef] = []
        for camera, cam in self._writers.items():
            try:
                meta = cam.writer.finish()
                if meta[0] > 0:
                    self._stager.commit(cam.staged, cam.rel)
                    media.append(
                        self._stager.video_ref(camera=camera, path=cam.rel, meta=meta)
                    )
            except Exception as exc:
                warnings.warn(
                    f"recorder: dropping video for {camera!r}: {exc}", stacklevel=2
                )
        if self._video_path is not None:
            try:
                ref = self._stager.carry_file(
                    prefix=self._prefix,
                    episode_index=index,
                    camera=DEFAULT_CAMERA,
                    source=self._video_path,
                )
                if ref is not None:
                    media.append(ref)
            except Exception as exc:
                warnings.warn(f"recorder: dropping env video: {exc}", stacklevel=2)
        self._workload.episodes.append(
            EpisodeRecord.from_result(
                result,
                index=index,
                included_in_metrics=self._included,
                media=tuple(media),
            )
        )
        self._writers = {}
        self._video_path = None

    def on_run_end(self, result: RunResult) -> None:
        """Flag any requested camera that never yielded a recordable frame."""
        for cam in self._writers.values():
            try:
                cam.writer.finish()
            except Exception:
                pass
        self._writers = {}
        missing = set(self._cameras) - self._captured - self._disabled
        if missing:
            warnings.warn(
                f"recorder: no frames captured for {sorted(missing)} "
                "(role absent or not a 1/3/4-channel image)",
                stacklevel=2,
            )

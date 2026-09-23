"""Live capture: a :class:`~rlmesh.RunHooks` that records into a workload.

This is the ``run(hooks=...)`` tie-in, on either loop. It observes the eval and, per
episode, records the outcome plus any media the env exposes -- per-step image frames
read through the run's own reader and streamed straight into a native AV1 writer (one
frame in memory at a time), and/or an env-produced video file whose path the env
leaves in the step ``info`` (that path comes from the env, which may be remote, so it
is copied only when it names a regular file inside the run directory). The env's
``render()`` frame is captured when the run can reach it: always on the session loop,
and on the native ``Model.run`` loop for a local env object (not an address target).
Frames come from one env: a vector env is refused unless ``cameras=[]``.

All capture is best-effort: a camera that fails to read or encode is warned once and
dropped, and the episode's outcome is still recorded -- capture never aborts the run.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .._models import RunContext, RunHooks
from .._models._view import FrameSources
from .constants import DEFAULT_CAMERA, RENDER_CAMERA
from .frames import as_frame, read_frame
from .schema import EpisodeRecord, MediaRef

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from .._models import EpisodeResult, RunResult, StepEvent
    from .._rlmesh import PyVideoWriter
    from .media import MediaStager
    from .schema import WorkloadRecord


@dataclass
class _Cam:
    """A camera's in-flight AV1 writer plus where its file will land in the bundle."""

    writer: PyVideoWriter
    staged: str
    rel: str


def _local_video_path(source: str) -> str | None:
    """The env-supplied video path, if it is a real file inside the run directory.

    ``video_keys`` reads a path out of a (possibly remote, possibly untrusted) env's
    step ``info``, and the recorder copies that file verbatim into the exported
    bundle -- so an env could otherwise name any file the eval process can read.
    The path is resolved against the run directory and accepted only if it stays
    inside it and names a regular file: an absolute path elsewhere, a ``..``
    escape, a symlink pointing out, and a directory or device are all refused.
    A path the OS itself rejects (an embedded NUL) and a run directory that has
    gone away are refused the same way, never raised out of the calling hook.
    """
    if not source:
        return None
    try:
        root = os.path.realpath(os.getcwd())
        resolved = os.path.realpath(os.path.join(root, source))
        if os.path.commonpath((root, resolved)) != root:
            return None
        return resolved if os.path.isfile(resolved) else None
    except (OSError, ValueError):
        return None


def _warn(message: str) -> None:
    """Emit a best-effort recorder warning that never propagates.

    Capture must never abort the run (the module contract), so even under
    warnings-as-errors (``python -W error``) a degradation notice must not raise
    out of a hook and kill the eval.
    """
    try:
        warnings.warn(message, stacklevel=3)
    except Exception:
        pass


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
        session: RunContext | None,
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
        self._explicit = cameras is not None
        #: Memoized ``render()`` thunk (resolved once from the session), or ``None``.
        self._render: Callable[[], object] | None = None
        self._render_resolved = False
        self._render_camera: str | None = RENDER_CAMERA if cameras is not None else None
        #: camera -> in-flight writer for the current episode.
        self._writers: dict[str, _Cam] = {}
        #: cameras that raised while encoding -- dropped for the whole run.
        self._disabled: set[str] = set()
        #: cameras that produced at least one frame (to flag silent no-capture).
        self._captured: set[str] = set()
        self._video_path: str | None = None

    def on_run_start(self, context: RunContext) -> None:
        """Adopt the run's context, so ``capture(session=...)`` is optional.

        An explicitly passed session wins (a hand-driven loop can still wire
        one in); otherwise the running loop's context is used for source
        discovery and the ``render()`` frame.
        """
        if self._session is None:
            self._session = context

    def on_episode_start(self, *, episode: int, seed: int | None) -> None:
        """Reset the per-episode writers and env-video path."""
        if (
            self._session is not None
            and (self._cameras or not self._explicit)
            and self._session.num_envs > 1
        ):
            raise ValueError(
                "recorder captures frames from a single env: a vector env's "
                "episodes interleave. Pass cameras=[] to record metrics only."
            )
        self._writers = {}
        self._video_path = None

    def _render_thunk(self) -> Callable[[], object] | None:
        if not self._render_resolved:
            self._render_resolved = True
            if self._session is not None:
                self._render = self._discover().render
        return self._render

    def _discover(self) -> FrameSources:
        """The run's frame sources, via the viewer's own discovery."""
        session = self._session
        if session is None:
            return FrameSources(roles=(), render_label=None, render=None)
        try:
            return session.frame_sources()
        except Exception:
            return FrameSources(roles=(), render_label=None, render=None)

    def _resolve_cameras(self) -> None:
        if self._resolved:
            return
        self._resolved = True
        if self._session is None:
            return
        sources = self._discover()
        self._render = sources.render
        self._render_resolved = True
        self._render_camera = sources.render_label
        self._cameras = sources.labels
        if not self._cameras:
            _warn(
                "recorder: no render() or image roles discovered from the session; "
                "recording metrics only"
            )

    def _read_source(
        self, event: StepEvent, camera: str
    ) -> tuple[Any, int, int, int] | None:
        """Read one camera's frame for this step, or ``None``.

        An explicit ``"render"`` camera prefers a declared image role of that
        name (mirroring auto-discovery, where a role named ``render`` is never
        shadowed) and only falls back to the env ``render()`` thunk.
        """
        if camera == self._render_camera:
            if self._explicit:
                frame = read_frame(event, camera)
                if frame is not None:
                    return frame
            thunk = self._render_thunk()
            if thunk is None:
                return None
            try:
                value = thunk()
            except Exception:
                return None
            return as_frame(value)
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
                _warn(f"recorder: disabling camera {camera!r} after an error: {exc}")
                self._disabled.add(camera)
                self._writers.pop(camera, None)

    def _write(self, camera: str, frame: tuple[Any, int, int, int]) -> None:
        data, height, width, channels = frame
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
        cam.writer.write_frame(memoryview(data).cast("B"), width, height, channels)
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
                _warn(f"recorder: dropping video for {camera!r}: {exc}")
        if self._video_path is not None:
            source = _local_video_path(self._video_path)
            try:
                ref = (
                    None
                    if source is None
                    else self._stager.carry_file(
                        prefix=self._prefix,
                        episode_index=index,
                        camera=DEFAULT_CAMERA,
                        source=source,
                    )
                )
                if ref is not None:
                    media.append(ref)
                else:
                    _warn(
                        f"recorder: env video {self._video_path!r} is not a readable "
                        "local file under the run directory; skipping"
                    )
            except Exception as exc:
                _warn(f"recorder: dropping env video: {exc}")
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
            _warn(
                f"recorder: no frames captured for {sorted(missing)} "
                "(role absent or not a 1/3/4-channel image)"
            )

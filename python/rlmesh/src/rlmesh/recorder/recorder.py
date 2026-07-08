"""The :class:`Recorder` -- capture eval runs, export a portable bundle.

Path-agnostic by design: every OSS execution path (pure ``rlmesh.run``, a
``session().run``, or a hand-driven session) converges on the same
:class:`~rlmesh.RunResult`, so :meth:`Recorder.add` records from that return value
with no runtime changes. :meth:`Recorder.capture` returns a
:class:`~rlmesh.RunHooks` for live per-episode capture (and media) on the
Python-driven paths. Either way, :meth:`Recorder.export` writes an
``rlmesh.result.v1`` bundle the managed platform's upload path can ingest.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .constants import DEFAULT_FPS, DEFAULT_VIDEO_INFO_KEYS
from .export import write_bundle
from .frames import image_roles
from .hooks import CaptureHooks, FrameFn
from .media import MediaStager
from .schema import EpisodeRecord, ResultSet, WorkloadRecord

if TYPE_CHECKING:
    from .._models import RunHooks, RunResult, Session


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Recorder:
    """Accumulates recorded workloads and exports them as one bundle.

    Each :meth:`add` / :meth:`capture` starts a new workload (one model x env x
    task cell). A single ``capture`` hooks object may drive several ``run`` calls;
    their episodes accumulate into that one workload, uniquely indexed.

    Example -- metrics only, path-agnostic::

        rec = rlmesh.Recorder()
        result = rlmesh.run(model, env, max_episodes=50)
        rec.add(result, model="smolvla", env="libero", task="libero-spatial-0")
        rec.export("results/run.zip")

    Example -- with media, on the session path::

        rec = rlmesh.Recorder()
        sess = rlmesh.session(model, env)
        sess.run(
            max_episodes=50,
            hooks=rec.capture(
                model="smolvla", env="libero", task="libero-spatial-0", session=sess
            ),
        )
        rec.export("results/run.zip")
    """

    def __init__(
        self,
        *,
        result_set_id: str | None = None,
        video: bool = False,
        fps: int = DEFAULT_FPS,
    ) -> None:
        """Create a recorder.

        ``video=True`` encodes captured frame stacks to H.264 mp4 (``fps`` playback
        rate) instead of shipping raw ``.npz`` stacks -- it needs ffmpeg, via the
        ``rlmesh[recorder]`` extra or a system binary (see :mod:`rlmesh.recorder.encode`).
        The default (``video=False``) stays dependency-free.
        """
        self._result_set = ResultSet(result_set_id=result_set_id or uuid.uuid4().hex)
        self._stager = MediaStager(video=video, fps=fps)

    @property
    def result_set_id(self) -> str:
        """The locally minted id anchoring this bundle (re-upload dedupe key)."""
        return self._result_set.result_set_id

    @property
    def workloads(self) -> tuple[WorkloadRecord, ...]:
        """The workloads recorded so far, in the order they were started."""
        return tuple(self._result_set.workloads)

    def _new_workload(
        self,
        model: str,
        env: str,
        task: str | None,
        config: dict[str, Any] | None,
    ) -> WorkloadRecord:
        workload = WorkloadRecord(
            model=model,
            env=env,
            task=task if task is not None else env,
            config=dict(config or {}),
        )
        self._result_set.workloads.append(workload)
        return workload

    def add(
        self,
        result: RunResult,
        *,
        model: str,
        env: str,
        task: str | None = None,
        config: dict[str, Any] | None = None,
        included_in_metrics: bool = True,
    ) -> WorkloadRecord:
        """Record a completed :class:`~rlmesh.RunResult` as a new workload (no media).

        The post-hoc path -- works for any run's return value, including pure
        ``rlmesh.run``. For video/frames, use :meth:`capture` instead.
        """
        workload = self._new_workload(model, env, task, config)
        for episode in result.episodes:
            workload.episodes.append(
                EpisodeRecord.from_result(
                    episode,
                    index=len(workload.episodes),
                    included_in_metrics=included_in_metrics,
                )
            )
        return workload

    def capture(
        self,
        *,
        model: str,
        env: str,
        task: str | None = None,
        config: dict[str, Any] | None = None,
        cameras: list[str] | None = None,
        frame_fn: FrameFn | None = None,
        session: Session[Any, Any] | None = None,
        video_info_keys: tuple[str, ...] = DEFAULT_VIDEO_INFO_KEYS,
        included_in_metrics: bool = True,
    ) -> RunHooks:
        """A :class:`~rlmesh.RunHooks` that records a live run into a new workload.

        Frame sourcing (all optional -- omit everything for metrics-only):

        * ``frame_fn`` -- full override: ``frame_fn(step_event)`` returns an HWC
          array, a ``{camera: array}`` map, or ``None`` to skip that step.
        * ``cameras`` -- explicit env image roles to read each step (HWC).
        * ``session`` -- when given and ``cameras``/``frame_fn`` are not, the env's
          declared image roles are auto-discovered from the session's contract.

        ``video_info_keys`` name the step-``info`` keys checked for an env-produced
        video file path (case 1: the env renders its own video); the file is copied
        into the bundle at export.
        """
        workload = self._new_workload(model, env, task, config)
        resolved = list(cameras) if cameras is not None else []
        if not resolved and frame_fn is None and session is not None:
            resolved = image_roles(getattr(session, "_contract", None))
        return CaptureHooks(
            workload=workload,
            stager=self._stager,
            cameras=resolved,
            frame_fn=frame_fn,
            video_keys=tuple(video_info_keys),
            included_in_metrics=included_in_metrics,
        )

    def to_dict(self, *, recorded_at: str | None = None) -> dict[str, Any]:
        """The ``rlmesh.result.v1`` document as a JSON-native dict."""
        return self._result_set.to_dict(recorded_at=recorded_at or _now_iso())

    def export(
        self,
        path: str | Path,
        *,
        archive: bool | str | None = None,
        recorded_at: str | None = None,
    ) -> Path:
        """Write the bundle to ``path`` and return it.

        Writes a folder by default, or a zip when ``path`` ends ``.zip`` or
        ``archive`` is ``True`` / ``"zip"``. ``archive=False`` forces a folder.
        """
        return write_bundle(
            self._result_set,
            Path(path),
            assets=self._stager.assets,
            recorded_at=recorded_at or _now_iso(),
            archive=archive,
        )

    def close(self) -> None:
        """Remove the temp staging dir holding captured frame stacks. Idempotent.

        Call after :meth:`export`. Staged files are also under the OS temp dir, so a
        missed ``close`` is cleaned by the OS eventually.
        """
        self._stager.cleanup()

    def __enter__(self) -> Recorder:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

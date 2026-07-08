"""Media staging: hold captured frames / referenced env videos until export.

Frames captured during a run cannot be written into the bundle until ``export`` is
told where the bundle goes, so a captured stack is written to a temp staging dir as
each episode completes (bounding memory to one episode) and remembered as an
``(source, bundle_relative_path)`` asset. Env-produced videos already exist on disk,
so only their path is remembered. ``export`` copies every asset into the bundle.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import TYPE_CHECKING

from .constants import MEDIA_DIR
from .frames import sanitize_part, write_frame_stack
from .schema import MediaRef

if TYPE_CHECKING:
    import numpy as np


class MediaStager:
    """Owns one temp staging dir and the list of assets to copy into the bundle.

    With ``video=True`` a captured frame stack is encoded to an mp4 (needs ffmpeg,
    see :mod:`rlmesh.recorder.encode`); otherwise it is saved as a compressed ``.npz``.
    """

    def __init__(self, *, video: bool = False, fps: int = 30) -> None:
        self._dir: str | None = None
        self._video = video
        self._fps = fps
        #: ``(absolute source path, bundle-relative destination)`` pairs.
        self.assets: list[tuple[str, str]] = []

    def _staging_dir(self) -> str:
        if self._dir is None:
            self._dir = tempfile.mkdtemp(prefix="rlmesh-recorder-")
        return self._dir

    @staticmethod
    def _rel(task: str, episode_index: int, filename: str) -> str:
        return f"{MEDIA_DIR}/{sanitize_part(task)}/ep{episode_index:05d}/{filename}"

    def frames(
        self,
        *,
        task: str,
        episode_index: int,
        camera: str,
        frames: list[np.ndarray],
    ) -> MediaRef:
        """Stage a per-step frame stack and return its manifest row.

        Encodes to mp4 when the stager is in ``video`` mode, else a compressed npz.
        """
        if self._video:
            return self._frames_mp4(task, episode_index, camera, frames)
        rel = self._rel(task, episode_index, f"{sanitize_part(camera)}.npz")
        handle, staged = tempfile.mkstemp(suffix=".npz", dir=self._staging_dir())
        os.close(handle)
        meta = write_frame_stack(staged, frames)
        self.assets.append((staged, rel))
        return MediaRef(
            camera=camera,
            kind="frames",
            path=rel,
            format="npz",
            frame_count=meta["frame_count"],
            width=meta["width"],
            height=meta["height"],
        )

    def _frames_mp4(
        self, task: str, episode_index: int, camera: str, frames: list[np.ndarray]
    ) -> MediaRef:
        from .encode import encode_frames_to_mp4

        rel = self._rel(task, episode_index, f"{sanitize_part(camera)}.mp4")
        handle, staged = tempfile.mkstemp(suffix=".mp4", dir=self._staging_dir())
        os.close(handle)
        meta = encode_frames_to_mp4(staged, frames, self._fps)
        self.assets.append((staged, rel))
        return MediaRef(
            camera=camera,
            kind="video",
            path=rel,
            format="mp4",
            frame_count=meta["frame_count"],
            width=meta["width"],
            height=meta["height"],
            fps=meta["fps"],
        )

    def video(
        self,
        *,
        task: str,
        episode_index: int,
        camera: str,
        source: str,
    ) -> MediaRef | None:
        """Reference an env-produced video file by path; ``None`` if it is missing.

        The file is copied into the bundle at export; here we only record where it
        lives and where it will land.
        """
        if not source or not os.path.isfile(source):
            return None
        ext = os.path.splitext(source)[1].lstrip(".").lower() or "bin"
        rel = self._rel(task, episode_index, f"{sanitize_part(camera)}.{ext}")
        self.assets.append((source, rel))
        return MediaRef(camera=camera, kind="video", path=rel, format=ext)

    def cleanup(self) -> None:
        """Remove the staging dir (staged npz files). Idempotent."""
        if self._dir is not None:
            shutil.rmtree(self._dir, ignore_errors=True)
            self._dir = None

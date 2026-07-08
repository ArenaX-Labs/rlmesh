"""Media staging: hold recorded AV1 videos / carried env videos until export.

Frames are encoded straight into a per-episode ``.mp4`` in a temp staging dir as
they arrive (via the native AV1 writer -- bounding memory to one frame), and
remembered as an ``(source, bundle_relative_path)`` asset. Env-produced videos
already exist on disk, so they are copied into staging at episode end (a fixed env
path that gets overwritten next episode can't clobber an earlier capture). ``export``
copies every asset into the bundle.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import TYPE_CHECKING

from .constants import MEDIA_DIR
from .frames import sanitize_part
from .schema import MediaRef

if TYPE_CHECKING:
    from .._rlmesh import PyVideoWriter


class MediaStager:
    """Owns one temp staging dir and the list of assets to copy into the bundle.

    Recorded frames are encoded to AV1 mp4 in process (pure Rust, no ffmpeg); the
    playback ``fps`` is stamped onto each recorded video's manifest row.
    """

    def __init__(self, *, fps: int = 30, quality: int = 60) -> None:
        self._dir: str | None = None
        #: Playback rate stamped on recorded videos (read by the manifest row).
        self.fps = fps
        #: AV1 record quality, 1..=100 (higher is better/larger).
        self.quality = quality
        #: ``(absolute source path, bundle-relative destination)`` pairs.
        self.assets: list[tuple[str, str]] = []
        #: Bundle-relative paths already handed out, to keep them unique.
        self._used: set[str] = set()

    def _staging_dir(self) -> str:
        if self._dir is None:
            self._dir = tempfile.mkdtemp(prefix="rlmesh-recorder-")
        return self._dir

    def _unique_rel(self, prefix: str, episode_index: int, filename: str) -> str:
        """A bundle-relative path guaranteed not to collide with an earlier one.

        Two camera names that differ only in characters :func:`sanitize_part` folds
        away (``cam/left`` and ``cam left`` -> ``cam_left``) would otherwise share a
        path and overwrite each other; a ``-2``/``-3`` suffix keeps them distinct.
        """
        rel = self._rel(prefix, episode_index, filename)
        if rel not in self._used:
            self._used.add(rel)
            return rel
        stem, dot, ext = filename.rpartition(".")
        i = 2
        while True:
            alt = self._rel(prefix, episode_index, f"{stem}-{i}{dot}{ext}")
            if alt not in self._used:
                self._used.add(alt)
                return alt
            i += 1

    @staticmethod
    def _rel(prefix: str, episode_index: int, filename: str) -> str:
        return f"{MEDIA_DIR}/{prefix}/ep{episode_index:05d}/{filename}"

    def open_video(
        self, *, prefix: str, episode_index: int, camera: str, width: int, height: int
    ) -> tuple[PyVideoWriter, str, str]:
        """Open a native AV1 writer for a camera; returns ``(writer, staged, rel)``.

        The staged file is registered as an asset only once the caller commits it via
        :meth:`commit` (a failed encode leaves no dangling manifest row).
        """
        from .._rlmesh import PyVideoWriter

        rel = self._unique_rel(prefix, episode_index, f"{sanitize_part(camera)}.mp4")
        handle, staged = tempfile.mkstemp(suffix=".mp4", dir=self._staging_dir())
        os.close(handle)
        writer = PyVideoWriter(staged, width, height, self.fps, self.quality)
        return writer, staged, rel

    def commit(self, staged: str, rel: str) -> None:
        """Register a finished staged file to be copied into the bundle at export."""
        self.assets.append((staged, rel))

    def carry_file(
        self, *, prefix: str, episode_index: int, camera: str, source: str
    ) -> MediaRef | None:
        """Copy an env-produced video into staging now; ``None`` if it is missing.

        Copying eagerly (rather than referencing the path until export) means an env
        that reuses one output path across episodes can't overwrite an earlier one.
        """
        if not source or not os.path.isfile(source):
            return None
        ext = os.path.splitext(source)[1].lstrip(".").lower() or "bin"
        rel = self._unique_rel(prefix, episode_index, f"{sanitize_part(camera)}.{ext}")
        handle, staged = tempfile.mkstemp(suffix=f".{ext}", dir=self._staging_dir())
        os.close(handle)
        shutil.copyfile(source, staged)
        self.assets.append((staged, rel))
        return MediaRef(camera=camera, kind="video", path=rel, format=ext)

    def video_ref(
        self, *, camera: str, path: str, meta: tuple[int, int, int]
    ) -> MediaRef:
        """Manifest row for a recorded AV1 video from its ``(frames, width, height)``."""
        frames, width, height = meta
        return MediaRef(
            camera=camera,
            kind="video",
            path=path,
            format="mp4",
            frame_count=frames,
            width=width,
            height=height,
            fps=float(self.fps),
        )

    def cleanup(self) -> None:
        """Remove the staging dir (staged videos). Idempotent."""
        if self._dir is not None:
            shutil.rmtree(self._dir, ignore_errors=True)
            self._dir = None

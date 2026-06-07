"""Shared viewer support for RLMesh remote clients."""

from __future__ import annotations

import math
import struct
import subprocess
import sys
import time
import warnings
from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal, Protocol, final

from ..specs import EnvContract
from ..types import Metadata

RenderPacket = bytes | bytearray | memoryview | None
ViewerFPS = float | None | Literal["env"]
EMPTY_METADATA: Mapping[str, object] = MappingProxyType({})


class RenderClient(Protocol):
    """Client capable of returning encoded render packets."""

    def render_packet(self, env_index: int = 0) -> RenderPacket:
        """Return one render packet for an environment index."""
        ...


@final
class ViewerProcess:
    """State for a local RLMesh viewer subprocess."""

    def __init__(
        self,
        process: subprocess.Popen[bytes],
        env_index: int,
        fps_limit: float | None,
    ) -> None:
        self.process = process
        self.env_index = env_index
        self.fps_limit = fps_limit
        self.last_frame_at: float | None = None


class ViewerMixin:
    """Mixin that streams render packets into the local RLMesh viewer."""

    _viewer: ViewerProcess | None = None
    _viewer_warning_emitted: bool = False

    @property
    def metadata(self) -> Metadata:
        """Endpoint metadata used to resolve viewer defaults."""
        raise NotImplementedError

    @property
    def env_contract(self) -> EnvContract:
        """Environment contract used to label the viewer."""
        raise NotImplementedError

    def _render_client(self) -> RenderClient:
        raise NotImplementedError

    def close_viewer(self) -> None:
        """Close the local render viewer if one is open."""
        self._shutdown_viewer()

    def open_viewer(self, *, env_index: int = 0, fps: ViewerFPS = "env") -> None:
        """Open a local render viewer and stream frames after reset, step, or render.

        Args:
            env_index: Environment index to view.
            fps: Frame-rate limit. Use ``"env"`` to read ``render_fps`` from
                environment metadata, a positive number for an explicit limit,
                or ``None`` to avoid pacing.
        """
        self.close_viewer()
        process = subprocess.Popen(
            [sys.executable, "-m", "rlmesh", "viewer", "--title", self.env_contract.id],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
        )
        if process.stdin is None:
            process.terminate()
            raise RuntimeError("failed to open render viewer stdin")

        self._viewer = ViewerProcess(
            process,
            env_index,
            self._resolve_viewer_fps_limit(fps),
        )
        self._viewer_warning_emitted = False

    def _refresh_viewer(self, *, pace: bool = False) -> None:
        viewer = self._viewer
        if viewer is None:
            return

        try:
            packet = self._render_client().render_packet(env_index=viewer.env_index)
        except Exception as exc:  # pragma: no cover - viewer best effort
            if not self._viewer_warning_emitted:
                warnings.warn(
                    f"render viewer update failed: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._viewer_warning_emitted = True
            return

        self._push_viewer_packet(packet, pace=pace)

    def _push_viewer_packet(self, packet: RenderPacket, *, pace: bool = False) -> None:
        viewer = self._viewer
        if viewer is None:
            return

        process = viewer.process
        if process.poll() is not None or process.stdin is None:
            self._shutdown_viewer(wait=False)
            return

        try:
            self._maybe_sleep_for_viewer_fps(viewer, pace=pace)
            if packet is None:
                _ = process.stdin.write(struct.pack("<BI", 0, 0))
                process.stdin.flush()
                viewer.last_frame_at = time.perf_counter()
                return

            raw = bytes(packet)
            _ = process.stdin.write(struct.pack("<BI", 1, len(raw)))
            _ = process.stdin.write(raw)
            process.stdin.flush()
            viewer.last_frame_at = time.perf_counter()
        except (BrokenPipeError, OSError):  # pragma: no cover - viewer best effort
            self._shutdown_viewer(wait=False)
        except Exception as exc:  # pragma: no cover - viewer best effort
            if not self._viewer_warning_emitted:
                warnings.warn(
                    f"render viewer packet push failed: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._viewer_warning_emitted = True

    def _resolve_viewer_fps_limit(self, fps: ViewerFPS) -> float | None:
        if fps == "env":
            return self._default_viewer_fps_limit()
        return self._coerce_viewer_fps_limit(fps, source="fps")

    def _default_viewer_fps_limit(self) -> float | None:
        return self._coerce_viewer_fps_limit(
            self.metadata.get("render_fps"),
            source="metadata['render_fps']",
            allow_none=True,
        )

    def _coerce_viewer_fps_limit(
        self,
        value: object,
        *,
        source: str,
        allow_none: bool = False,
    ) -> float | None:
        if value is None and allow_none:
            return None
        if value is None:
            return None
        if isinstance(value, bool):
            if allow_none:
                return None
            raise TypeError(f"{source} must be a positive number, None, or 'env'")
        if not isinstance(value, (int, float, str)):
            if allow_none:
                return None
            raise TypeError(f"{source} must be a positive number, None, or 'env'")

        try:
            fps = float(value)
        except (TypeError, ValueError):
            if allow_none:
                return None
            raise TypeError(
                f"{source} must be a positive number, None, or 'env'"
            ) from None

        if not math.isfinite(fps) or fps <= 0:
            if allow_none:
                return None
            raise ValueError(f"{source} must be a finite positive number")
        return fps

    def _maybe_sleep_for_viewer_fps(
        self, viewer: ViewerProcess, *, pace: bool = False
    ) -> None:
        if not pace or viewer.fps_limit is None or viewer.last_frame_at is None:
            return

        frame_interval = 1.0 / viewer.fps_limit
        remaining = frame_interval - (time.perf_counter() - viewer.last_frame_at)
        if remaining > 0:
            time.sleep(remaining)

    def _shutdown_viewer(self, *, wait: bool = True) -> None:
        viewer = self._viewer
        self._viewer = None
        if viewer is None:
            return

        process = viewer.process
        stdin = process.stdin
        if stdin is not None and not stdin.closed:
            try:
                stdin.close()
            except OSError:  # pragma: no cover - viewer best effort
                pass

        if not wait:
            return

        try:
            _ = process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:  # pragma: no cover - viewer best effort
            process.terminate()
            try:
                _ = process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                _ = process.wait()


__all__ = [
    "EMPTY_METADATA",
    "RenderClient",
    "RenderPacket",
    "ViewerFPS",
    "ViewerMixin",
    "ViewerProcess",
]

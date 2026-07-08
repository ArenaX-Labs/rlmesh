"""Frame acquisition and staging for the recorder (no video encoding).

This module only *collects* pixels the env already produces --
either a whole env-rendered video file (carried verbatim) or per-step image
observations grabbed through the very same read path the live viewer uses -- and
stages captured frames as a compressed uint8 ``(T, H, W, C)`` ``.npz`` stack. numpy
is imported lazily so importing the recorder never pulls the optional dependency;
frame capture is only reachable once an env is already handing back arrays.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

    from .._models import StepEvent

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_part(part: str) -> str:
    """A single path segment reduced to ``[A-Za-z0-9._-]`` (managed-layout safe)."""
    cleaned = _UNSAFE.sub("_", part).strip("._-")
    return cleaned or "unnamed"


def to_frame(value: object) -> np.ndarray | None:
    """Normalize one read/render array to a contiguous uint8 HWC frame, or ``None``.

    Delegates to the viewer's shared normalization so a captured frame is identical to
    what the live viewer would draw (range-normalizes float frames, keeps 1/3/4
    channels).
    """
    from .._models._view import normalize_frame

    return normalize_frame(value)


def image_roles(contract: object) -> list[str]:
    """The env's declared image roles (best-effort; ``[]`` on any failure).

    Wraps :func:`rlmesh._models._read.env_image_roles`, the same discovery the
    viewer uses, so ``cameras`` can be auto-derived from a session's contract.
    """
    try:
        from .._models._read import env_image_roles

        return list(env_image_roles(contract))
    except Exception:
        return []


def read_image(event: StepEvent, role: str) -> np.ndarray | None:
    """Grab one HWC uint8 frame for ``role`` from a step event, or ``None``.

    Uses ``event.read`` -- the session's own role-addressed reader -- so the frame
    comes back in the requested ``hwc`` layout whatever the env stores. Any read or
    conversion failure degrades to ``None`` (capture is best-effort, never fatal).
    """
    from ..adapters import Image

    try:
        value = event.read(Image(role, layout="hwc"))
    except Exception:
        return None
    return to_frame(value)


def write_frame_stack(path: str, frames: list[np.ndarray]) -> dict[str, Any]:
    """Save a list of HWC uint8 frames as a compressed ``(T, H, W, C)`` npz.

    Returns ``{frame_count, height, width}`` for the media manifest row. Raises if
    the frames have mismatched shapes (a real capture bug worth surfacing).
    """
    import numpy as np

    stack = np.stack(frames, axis=0)
    with open(path, "wb") as handle:
        np.savez_compressed(handle, frames=stack)
    count, height, width = int(stack.shape[0]), int(stack.shape[1]), int(stack.shape[2])
    return {"frame_count": count, "height": height, "width": width}

"""Frame acquisition for the recorder.

The recorder never encodes pixels in Python: it reads each step's image roles
through the very same read path the live viewer uses, normalizes them to a
contiguous uint8 HWC array (shared with the viewer via
:func:`rlmesh._models._view.normalize_frame`), and hands the buffer to the native
AV1 writer. numpy is imported lazily so importing the recorder never pulls it.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .._models import StepEvent

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_part(part: str) -> str:
    """A single path segment reduced to ``[A-Za-z0-9._-]`` (managed-layout safe).

    Sibling of the lowercasing socket-name sanitizer
    (``rlmesh._cli.serve_env._socket_label``); this one keeps case and dots
    because bundle paths are user-facing camera/model names.
    """
    cleaned = _UNSAFE.sub("_", part).strip("._-")
    return cleaned or "unnamed"


def as_frame(value: object) -> tuple[Any, int, int, int] | None:
    """Normalize any read/render value to ``(array, height, width, channels)``.

    ``array`` is the contiguous uint8 HWC array from
    :func:`~rlmesh._models._view.normalize_frame` (so a recorded frame is
    identical to what the viewer draws), passed to the native writer through
    the buffer protocol -- no ``tobytes`` copy. ``None`` when the value is not
    a 1/3/4-channel image (that frame is skipped).
    """
    from .._models._view import normalize_frame

    array = normalize_frame(value)
    if array is None:
        return None
    height, width, channels = (int(dim) for dim in array.shape)
    return array, height, width, channels


def read_frame(event: StepEvent, role: str) -> tuple[Any, int, int, int] | None:
    """Read one image-observation ``role`` frame, or ``None``.

    Uses ``event.read`` -- the session's own role-addressed reader -- so the frame
    comes back in the requested ``hwc`` layout whatever the env stores. Returns ``None``
    when the read fails, the role is absent, or the value is not a 1/3/4-channel image
    -- that frame is skipped (a transient miss never disables the camera).
    """
    from ..adapters import Image

    try:
        value = event.read(Image(role, layout="hwc"))
    except Exception:
        return None
    return as_frame(value)

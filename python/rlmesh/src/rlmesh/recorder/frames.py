"""Frame acquisition for the recorder.

The recorder never encodes pixels in Python: it reads each step's image roles
through the very same read path the live viewer uses, normalizes them to a
contiguous uint8 HWC buffer (shared with the viewer via
:func:`rlmesh._models._view.normalize_frame`), and hands the raw bytes to the native
AV1 writer. numpy is imported lazily so importing the recorder never pulls it.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .._models import StepEvent

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_part(part: str) -> str:
    """A single path segment reduced to ``[A-Za-z0-9._-]`` (managed-layout safe)."""
    cleaned = _UNSAFE.sub("_", part).strip("._-")
    return cleaned or "unnamed"


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


def to_frame_bytes(value: object) -> tuple[bytes, int, int, int] | None:
    """Normalize any read/render array to ``(bytes, width, height, channels)``.

    Reuses the viewer's converter (:func:`rlmesh._models._view._to_hwc_u8`), so a
    recorded frame matches what the viewer draws; only the ``(w, h)`` order differs to
    match the native writer's signature. ``None`` when the value is not a 1/3/4-channel
    image (that frame is skipped).
    """
    from .._models._view import _to_hwc_u8  # pyright: ignore[reportPrivateUsage]

    hwc = _to_hwc_u8(value)
    if hwc is None:
        return None
    data, height, width, channels = hwc
    return data, width, height, channels


def read_frame(event: StepEvent, role: str) -> tuple[bytes, int, int, int] | None:
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
    return to_frame_bytes(value)


def render_source(session: object) -> Callable[[], object] | None:
    """A zero-arg thunk for the env's rgb ``render()``, or ``None`` if unavailable.

    Mirrors the viewer's source discovery: only offered when the connected env exposes
    an rgb render mode, and resolved through the same :func:`_resolve_render` so a
    recorded render matches the viewer's default source.
    """
    client = getattr(session, "_client", None)
    if client is None:
        return None
    render_mode = getattr(client, "render_mode", None)
    if not (isinstance(render_mode, str) and "rgb" in render_mode.lower()):
        return None
    try:
        from .._models._view import (
            _resolve_render,  # pyright: ignore[reportPrivateUsage]
        )

        return _resolve_render(client)  # pyright: ignore[reportPrivateUsage]
    except Exception:
        return None

"""Built-in debug viewer wiring for the `Session` loop.

A :class:`View` config plus the :class:`ViewerDriver` that, when a session has
``view=`` set, feeds the selected source to the native ``PyViewer`` (terminal +
HTTP backends) each step. The sources are the env's ``render()`` frame (the pretty
human view, default) plus every observation image role the env **declares** -- so
the selector shows real role names, including custom ones. Strictly best-effort:
any failure disables the viewer with a warning and never breaks the eval.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# The pretty human-render source; always offered first when the env supports it.
_RENDER = "render"


@dataclass(frozen=True)
class View:
    """How to show a live eval. Pass to ``run`` / ``session`` as ``view=``.

    The common cases are the string shorthands ``"terminal"`` / ``"http"`` /
    ``"http:9000"`` / ``"both"``; construct a ``View`` directly only to tune.

    Attributes:
        backend: Where to draw -- ``"terminal"`` (in-place half-blocks),
            ``"http"`` (a local web page), or ``"both"``.
        port: HTTP port for the ``"http"`` / ``"both"`` backends.
        fps: Target frame rate; frames produced faster are dropped.
        source: Which source to show first by label (a render label or an image
            role); ``None`` shows the first available.
        format: Encoding for HTTP frames, ``"jpeg"`` or ``"png"``.
        quality: JPEG quality 1..100 (ignored for PNG).
    """

    backend: str = "terminal"
    port: int = 8008
    fps: int = 30
    source: str | None = None
    format: str = "jpeg"
    quality: int = 75

    def __post_init__(self) -> None:
        if self.backend not in ("terminal", "http", "both"):
            raise ValueError(
                f"View.backend must be 'terminal', 'http', or 'both'; got {self.backend!r}"
            )
        if not 0 < self.port < 65536:
            raise ValueError(f"View.port must be in 1..65535; got {self.port}")


def resolve_view(view: object) -> View | None:
    """Normalize the ``view=`` argument to a :class:`View` (or ``None`` = off)."""
    if view is None or view is False:
        return None
    if view is True:
        return View()
    if isinstance(view, View):
        return view
    if isinstance(view, str):
        spec = view.strip().lower()
        if spec in ("", "terminal", "term", "tty"):
            return View(backend="terminal")
        if spec == "both":
            return View(backend="both")
        if spec == "http" or spec.startswith("http:"):
            _, _, tail = spec.partition(":")
            if not tail:
                return View(backend="http")
            try:
                port = int(tail)
            except ValueError:
                raise ValueError(
                    f"invalid view port {tail!r} in view={view!r}; use 'http:PORT' "
                    "with an integer port"
                ) from None
            return View(backend="http", port=port)
        raise ValueError(
            f"unrecognized view={view!r}; use 'terminal', 'http', 'http:PORT', "
            "'both', or a View(...)"
        )
    raise TypeError(f"view must be str, bool, View, or None; got {type(view).__name__}")


class ViewerDriver:
    """Drive a native ``PyViewer`` from a session.

    Discovers the sources (render + declared image roles), then feeds the
    selected one each step.
    """

    def __init__(self, view: View) -> None:
        self._view = view
        self._pv: Any = None
        self._items: dict[str, Any] = {}
        self._render_ok = False
        self._render_label = _RENDER
        self._render_call: Callable[[], object] | None = None
        self._disabled = False

    def _ensure(self, contract: Any, client: Any) -> None:
        if self._pv is not None or self._disabled:
            return
        try:
            from ..adapters import Image
            from ._read import env_image_roles

            cameras = env_image_roles(contract)
            for role in cameras:
                self._items[role] = Image(role, layout="hwc")

            render_mode = getattr(client, "render_mode", None)
            self._render_ok = (
                isinstance(render_mode, str) and "rgb" in render_mode.lower()
            )

            if self._render_ok:
                self._render_label = "render()" if _RENDER in cameras else _RENDER
                self._render_call = _resolve_render(client)

            sources: list[str] = (
                [self._render_label] if self._render_ok else []
            ) + cameras
            if not sources:
                warnings.warn(
                    "rlmesh view: env declares no image roles and has no rgb "
                    "render mode; viewer disabled.",
                    stacklevel=2,
                )
                self._disabled = True
                return

            from .._rlmesh import PyViewer

            terminal = self._view.backend in ("terminal", "both")
            http_port = (
                self._view.port if self._view.backend in ("http", "both") else None
            )
            self._pv = PyViewer(
                terminal=terminal,
                http_port=http_port,
                fps=self._view.fps,
                format=self._view.format,
                quality=self._view.quality,
            )
            for warning in self._pv.warnings():
                warnings.warn(f"rlmesh view: {warning}", stacklevel=2)
            default = (
                sources.index(self._view.source) if self._view.source in sources else 0
            )
            self._pv.set_sources(sources, default)
        except Exception as exc:
            warnings.warn(
                f"rlmesh view: disabled after setup error: {exc}", stacklevel=2
            )
            self._disabled = True
            self._pv = None

    def feed(
        self,
        *,
        contract: Any,
        client: Any,
        obs: object,
        read: Callable[[object, object], object],
        steps: int,
        reward: float,
        outcome: str,
    ) -> None:
        self._ensure(contract, client)
        if self._pv is None:
            return
        try:
            if self._pv.wants_frame():
                frame = self._frame_for(obs, read, self._pv.selected_source())
                if frame is not None:
                    converted = _to_hwc_u8(frame)
                    if converted is not None:
                        data, height, width, channels = converted
                        self._pv.feed_frame(data, width, height, channels)
            self._pv.feed_hud(steps, reward, outcome)
        except Exception:
            pass
        if self._pv.should_quit():
            raise KeyboardInterrupt("rlmesh viewer: quit requested (q / Esc / Ctrl-C)")

    def _frame_for(
        self,
        obs: object,
        read: Callable[[object, object], object],
        selected: str | None,
    ) -> object:
        if selected is None:
            return None
        if selected == self._render_label and self._render_call is not None:
            try:
                return self._render_call()
            except Exception:
                return None
        item = self._items.get(selected)
        return read(obs, item) if item is not None else None

    def close(self) -> None:
        self._disabled = True
        if self._pv is not None:
            try:
                self._pv.close()
            finally:
                self._pv = None


def _resolve_render(client: Any) -> Callable[[], object]:
    """Resolve ``env.render()`` to a zero-arg call once, by signature.

    Returns a thunk calling ``render(env_index=0)`` (RemoteEnv) or ``render()``
    (gym/local), picked by inspecting the signature so the per-frame draw path
    neither re-probes nor relies on catching ``TypeError`` to choose -- which
    would also swallow a ``TypeError`` raised *inside* ``render()``.
    """
    import inspect

    render = getattr(client, "render", None)
    if render is None:
        return lambda: None
    try:
        params = inspect.signature(render).parameters
        wants_index = "env_index" in params or any(
            p.kind is p.VAR_KEYWORD for p in params.values()
        )
    except (TypeError, ValueError):
        wants_index = False
    if wants_index:
        return lambda: render(env_index=0)
    return lambda: render()


def _to_hwc_u8(frame: object) -> tuple[bytes, int, int, int] | None:
    """A render/read camera array (numpy / torch / jax) to ``(bytes, H, W, C)`` uint8.

    Returns ``None`` for anything not a 1/3/4-channel HWC image.
    """
    import numpy as np

    array: Any = frame
    if hasattr(array, "detach"):
        array = array.detach().to("cpu").numpy()
    array = np.asarray(array)
    if array.ndim != 3 or array.shape[2] not in (1, 3, 4):
        return None
    if array.dtype != np.uint8:
        a: Any = array.astype(np.float64)
        lo = float(a.min()) if a.size else 0.0
        hi = float(a.max()) if a.size else 0.0
        if lo >= 0.0 and hi <= 1.0 + 1e-6:
            a = a * 255.0
        elif lo >= -1.0 - 1e-6 and hi <= 1.0 + 1e-6:
            a = (a + 1.0) * 127.5
        elif not (lo >= 0.0 and hi <= 255.0 + 1e-6):
            span = hi - lo
            a = (a - lo) * (255.0 / span) if span > 1e-12 else a - lo
        array = a.clip(0.0, 255.0).astype(np.uint8)
    array = np.ascontiguousarray(array)
    height, width, channels = (int(dim) for dim in array.shape)
    return array.tobytes(), height, width, channels

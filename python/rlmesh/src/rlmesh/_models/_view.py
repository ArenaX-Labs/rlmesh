"""Built-in debug viewer wiring for the `Session` loop.

A :class:`View` config plus the :class:`ViewerDriver` that, when a session has
``view=`` set, feeds the selected source to the native ``PyViewer`` (terminal +
HTTP backends) each step. The sources are every observation image role the env
**declares** (the first one shows by default: it rides the step response, so
drawing it costs no extra round trip) plus the env's ``render()`` frame -- so the
selector shows real role names, including custom ones. Strictly best-effort:
any failure disables the viewer with a warning and never breaks the eval.
"""

from __future__ import annotations

import math
import sys
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

#: The pretty human-render source label; listed first when the env supports it.
#: Public vocabulary: ``cameras=["render"]`` on the recorder and
#: the viewer's source selector both name the env ``render()`` frame with it.
RENDER_SOURCE = "render"


@dataclass(frozen=True)
class FrameSources:
    """The frame sources one connected env offers, in display order.

    The single discovery result shared by the live viewer and the recorder,
    so the same env always yields the same camera vocabulary in the viewer
    HUD and in an exported bundle.

    Attributes:
        roles: The env's declared image observation roles.
        render_label: Label for the env ``render()`` source, or ``None`` when
            the env exposes no rgb render mode. ``"render()"`` when a declared
            role is itself named ``render`` (the role keeps the plain name).
        render: Zero-arg thunk returning the ``render()`` frame, or ``None``.
    """

    roles: tuple[str, ...]
    render_label: str | None
    render: Callable[[], object] | None

    @property
    def labels(self) -> tuple[str, ...]:
        """Every source label, render first -- the selector/camera vocabulary."""
        head = (self.render_label,) if self.render_label is not None else ()
        return head + self.roles

    @property
    def default_label(self) -> str | None:
        """The source a viewer shows first: the first declared image role.

        A role is read off the observation the step already returned, while the
        ``render()`` source costs one more env round trip per drawn frame, so
        ``render()`` is the default only for an env that declares no image role.
        ``None`` when the env offers no source at all.
        """
        if self.roles:
            return self.roles[0]
        return self.render_label


def discover_frame_sources(contract: Any, client: Any) -> FrameSources:
    """Discover the frame sources of a connected env (render + image roles).

    The one place the render-vs-role name collision is resolved: a declared
    image role named ``render`` keeps its plain name and the env ``render()``
    source is labeled ``render()`` instead, so a role can never be shadowed.
    """
    from ._read import env_image_roles

    roles = tuple(env_image_roles(contract))
    render_mode = getattr(client, "render_mode", None)
    if not (isinstance(render_mode, str) and "rgb" in render_mode.lower()):
        return FrameSources(roles=roles, render_label=None, render=None)
    label = f"{RENDER_SOURCE}()" if RENDER_SOURCE in roles else RENDER_SOURCE
    return FrameSources(roles=roles, render_label=label, render=_resolve_render(client))


@dataclass(frozen=True)
class View:
    """How to show a live eval. Pass to ``run`` / ``session`` as ``view=``.

    The common cases are the string shorthands ``"terminal"`` / ``"http"`` /
    ``"http:9000"`` / ``"both"``; construct a ``View`` directly only to tune.

    Attributes:
        backend: Where to draw -- ``"terminal"`` (in-place half-blocks),
            ``"http"`` (a local web page), or ``"both"``.
        port: HTTP port for the ``"http"`` / ``"both"`` backends.
        fps: Target frame rate; frames produced faster are dropped. This only
            thins the drawing: the eval itself still runs flat out (see
            ``step_hz``).
        source: Which source to show first by label (a render label or an image
            role); ``None`` shows the env's first declared image role, or its
            ``render()`` frame when it declares none.
        format: Encoding for HTTP frames, ``"jpeg"`` or ``"png"``.
        quality: JPEG quality 1..100 (ignored for PNG).
        hold: Keep the viewer up after the session ends, showing the final frame
            and HUD: ``True`` until you quit it (``q`` / ``Esc`` in the
            terminal, the quit button on the HTTP page, or Ctrl-C), a number of
            seconds to hold at most that long. ``False`` (the default) closes
            the viewer with the session. An unbounded hold needs a way to quit:
            with only the terminal backend and no interactive terminal it is
            skipped with a warning rather than blocking forever.
        step_hz: Pace the session's env steps to at most this many per second,
            so a fast sim plays back at a watchable rate (set it to the env's
            control rate for real time). The session sleeps after each
            ``step`` until the next step is due; a step slower than the period
            is never sped up, and the lost time is not made up with a burst.
            ``None`` (the default) runs as fast as it can.
    """

    backend: str = "terminal"
    port: int = 8008
    fps: int = 30
    source: str | None = None
    format: str = "jpeg"
    quality: int = 75
    hold: bool | float = False
    step_hz: float | None = None

    def __post_init__(self) -> None:
        if self.backend not in ("terminal", "http", "both"):
            raise ValueError(
                f"View.backend must be 'terminal', 'http', or 'both'; got {self.backend!r}"
            )
        if not 0 < self.port < 65536:
            raise ValueError(f"View.port must be in 1..65535; got {self.port}")
        if self.fps < 1:
            raise ValueError(f"View.fps must be >= 1; got {self.fps}")
        if self.format not in ("jpeg", "png"):
            raise ValueError(
                f"View.format must be 'jpeg' or 'png'; got {self.format!r}"
            )
        if not 1 <= self.quality <= 100:
            raise ValueError(f"View.quality must be in 1..100; got {self.quality}")
        if not isinstance(self.hold, bool) and not (
            math.isfinite(self.hold) and self.hold >= 0
        ):
            raise ValueError(
                "View.hold must be a bool or a finite number of seconds >= 0; "
                f"got {self.hold!r}"
            )
        if self.step_hz is not None and not (
            math.isfinite(self.step_hz) and self.step_hz > 0
        ):
            raise ValueError(
                f"View.step_hz must be a finite rate > 0 or None; got {self.step_hz!r}"
            )


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
        self._render_label = RENDER_SOURCE
        self._render_call: Callable[[], object] | None = None
        self._disabled = False
        self._quit = False
        #: Last drawn frame size, cached so the HUD can show the source resolution on
        #: throttled steps that fetch no new frame. 0 until the first frame is drawn.
        self._frame_w = 0
        self._frame_h = 0
        #: The last fed observation, its reader, and HUD fields: what a held viewer
        #: redraws after the session ends (and on a source switch while held).
        self._last_obs: object = None
        self._last_read: Callable[[object, object], object] | None = None
        self._last_hud: dict[str, Any] | None = None
        #: The source whose frame was last drawn (``None`` before the first).
        self._drawn_source: str | None = None
        #: ``step_hz`` pacing: when the next step is due (``perf_counter`` seconds).
        self._next_step_t: float | None = None

    def _ensure(self, contract: Any, client: Any) -> None:
        if self._pv is not None or self._disabled:
            return
        try:
            from ..adapters import Image

            discovered = discover_frame_sources(contract, client)
            for role in discovered.roles:
                self._items[role] = Image(role, layout="hwc")

            self._render_ok = discovered.render_label is not None
            if discovered.render_label is not None:
                self._render_label = discovered.render_label
                self._render_call = discovered.render

            sources = list(discovered.labels)
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
            wanted = (
                self._view.source
                if self._view.source in sources
                else discovered.default_label
            )
            self._pv.set_sources(sources, sources.index(wanted) if wanted else 0)
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
        model_ms: float = 0.0,
        env_ms: float = 0.0,
        sps: float = 0.0,
        elapsed_s: float = 0.0,
        episode: int = 0,
        episodes: int = 0,
        seed: int | None = None,
        chunk_pos: int = 0,
        chunk_len: int = 0,
    ) -> None:
        self._ensure(contract, client)
        if self._pv is None:
            return
        self._last_obs, self._last_read = obs, read
        self._last_hud = {
            "steps": steps,
            "reward": reward,
            "outcome": outcome,
            "model_ms": model_ms,
            "env_ms": env_ms,
            "sps": sps,
            "elapsed_s": elapsed_s,
            "episode": episode,
            "episodes": episodes,
            "seed": seed if seed is not None else -1,
            "chunk_pos": chunk_pos,
            "chunk_len": chunk_len,
        }
        try:
            if self._pv.wants_frame():
                self._draw(obs, read, self._pv.selected_source())
            self._feed_hud()
            if self._pv.should_quit():
                self._quit = True
        except Exception as exc:
            warnings.warn(
                f"rlmesh view: disabled after feed error: {exc}", stacklevel=2
            )
            self._disabled = True
            pv, self._pv = self._pv, None
            if pv is not None:
                try:
                    pv.close()
                except Exception:
                    pass

    def _draw(
        self,
        obs: object,
        read: Callable[[object, object], object],
        selected: str | None,
    ) -> None:
        """Fetch ``selected``'s frame for ``obs`` and hand it to the viewer."""
        frame = self._frame_for(obs, read, selected)
        if frame is None:
            return
        converted = _to_hwc_u8(frame)
        if converted is None:
            return
        data, height, width, channels = converted
        self._frame_w, self._frame_h = width, height
        self._drawn_source = selected
        self._pv.feed_frame(data, width, height, channels)

    def _feed_hud(self, outcome_suffix: str = "") -> None:
        """Push the last fed HUD fields (plus the current frame size)."""
        hud = self._last_hud
        if hud is None:
            return
        self._pv.feed_hud(
            hud["steps"],
            hud["reward"],
            f"{hud['outcome']} {outcome_suffix}".strip(),
            model_ms=hud["model_ms"],
            env_ms=hud["env_ms"],
            sps=hud["sps"],
            elapsed_s=hud["elapsed_s"],
            episode=hud["episode"],
            episodes=hud["episodes"],
            seed=hud["seed"],
            width=self._frame_w,
            height=self._frame_h,
            chunk_pos=hud["chunk_pos"],
            chunk_len=hud["chunk_len"],
        )

    def pace(self) -> float:
        """Sleep until the next step is due under :attr:`View.step_hz`.

        Called by the session after each env step; returns the seconds slept
        (``0.0`` with pacing off) so step timings can leave the sleep out.
        Deadlines advance by one period per step, so steady pacing does not
        drift; a step that overran its period restarts the schedule from now
        instead of bursting to catch up.
        """
        hz = self._view.step_hz
        if hz is None or self._quit:
            return 0.0
        period = 1.0 / hz
        now = time.perf_counter()
        due = self._next_step_t
        if due is None or due < now:
            self._next_step_t = now + period
            return 0.0
        time.sleep(due - now)
        self._next_step_t = due + period
        return due - now

    def quit_requested(self) -> bool:
        """Whether the viewer asked to stop the run (``q`` / ``Esc``; sticky).

        The session treats it as stop-early: the current episode is truncated and
        the eval loop returns the partial :class:`~rlmesh.RunResult` -- a real
        Ctrl-C outside the viewer still raises ``KeyboardInterrupt``.
        """
        return self._quit

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

    def consume_skip(self) -> bool:
        """Whether the viewer asked to end the current episode early (one-shot).

        Best-effort like the rest of the driver: any failure (or no viewer) reads as
        "no skip" and never disturbs the eval.
        """
        if self._pv is None:
            return False
        try:
            return bool(self._pv.take_skip())
        except Exception:
            return False

    def close(self) -> None:
        """Tear the viewer down, after the :attr:`View.hold` wait when one is set."""
        self._disabled = True
        if self._pv is not None:
            try:
                self._hold()
            finally:
                try:
                    self._pv.close()
                finally:
                    self._pv = None

    def _hold(self) -> None:
        """Keep serving the final frame and HUD until quit or the hold elapses.

        Redraws the last observation once (its frame may have been throttled)
        and again whenever the user switches source, so every source stays
        browsable. Skipped when the user already quit the run, and an unbounded
        hold is skipped (with a warning) when nobody could quit it. Ctrl-C ends
        the hold; any other failure warns and lets the viewer close.
        """
        hold = self._view.hold
        if hold is False or self._quit or self._last_hud is None:
            return
        if hold is True and not self._quittable():
            warnings.warn(
                "rlmesh view: hold=True needs an interactive terminal or the http "
                "backend to quit from; closing the viewer instead.",
                stacklevel=3,
            )
            return
        deadline = None if hold is True else time.monotonic() + float(hold)
        if deadline is not None and deadline <= time.monotonic():
            return
        pv = self._pv
        try:
            self._feed_hud("[held]")
            redraw = True
            while not pv.should_quit():
                if deadline is not None and time.monotonic() >= deadline:
                    break
                selected = pv.selected_source()
                if (redraw or selected != self._drawn_source) and pv.wants_frame():
                    redraw = False
                    if self._last_read is not None:
                        self._draw(self._last_obs, self._last_read, selected)
                    self._feed_hud("[held]")
                time.sleep(0.05)
        except KeyboardInterrupt:
            pass
        except Exception as exc:
            warnings.warn(f"rlmesh view: hold ended after error: {exc}", stacklevel=3)

    def _quittable(self) -> bool:
        """Whether a user can end an unbounded hold (an http page or a real tty)."""
        if self._view.backend in ("http", "both"):
            return True
        try:
            return sys.stdin.isatty() and sys.stdout.isatty()
        except (AttributeError, ValueError):
            return False


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


def normalize_frame(frame: object) -> Any:
    """A render/read camera array (numpy / torch / jax) to a contiguous uint8 HWC array.

    Returns ``None`` for anything not a 1/3/4-channel HWC image. Range-normalizes a
    float frame (``[0, 1]``, ``[-1, 1]``, or min-max) into ``[0, 255]``. Shared by the
    live viewer (:func:`_to_hwc_u8`) and the recorder so a captured frame is identical
    to what the viewer would draw.
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
    return np.ascontiguousarray(array)


def _to_hwc_u8(frame: object) -> tuple[bytes, int, int, int] | None:
    """A render/read camera array (numpy / torch / jax) to ``(bytes, H, W, C)`` uint8.

    Returns ``None`` for anything not a 1/3/4-channel HWC image.
    """
    array = normalize_frame(frame)
    if array is None:
        return None
    height, width, channels = (int(dim) for dim in array.shape)
    return array.tobytes(), height, width, channels

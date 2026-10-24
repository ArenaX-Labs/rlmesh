"""Idiomatic Python values over the Rust facade's manual binding."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any, Generic, cast

from .._value_conversion import ValueBridge
from ._eval import ActT, ObsT, RunHooks
from ._read import Reader, resolve_read_adapter
from ._view import FrameSources, ViewerDriver, discover_frame_sources, resolve_view

if TYPE_CHECKING:
    from rlmesh._rlmesh import PyBinding

__all__ = ["Binding"]


class Binding(Generic[ObsT, ActT]):
    """A model bound to one env, stepped by hand: :meth:`Model.bind` returns one.

    ``reset`` starts an episode; ``step()`` uses the model's action and
    ``step(action)`` uses yours. The shared Rust facade owns episode state,
    hooks, results and resource release. Closing leaves the model reusable.
    """

    _native: PyBinding
    _bridge: ValueBridge
    _target: Any
    _view: ViewerDriver | None
    _closed: bool
    _failed: bool
    _dispatching: bool
    _core: Any
    _read_cache: dict[Any, Reader]

    def __init__(self) -> None:
        raise TypeError("Binding is not constructed directly; use model.bind(target)")

    @classmethod
    def _create(
        cls,
        *,
        native: PyBinding,
        bridge: ValueBridge,
        hooks: RunHooks | None,
        view: object,
        target: Any,
    ) -> Binding[Any, Any]:
        from ._facade import FacadeHookRelay

        self = object.__new__(cls)
        self._native = native
        self._bridge = bridge
        self._target = target
        resolved = resolve_view(view)
        self._view = ViewerDriver(resolved) if resolved is not None else None
        self._closed = False
        self._failed = False
        self._dispatching = False
        self._read_cache = {}
        relay = FacadeHookRelay(hooks, self, weak_context=True)
        try:
            native.observe(relay)
        except BaseException:
            with contextlib.suppress(BaseException):
                native.close()
            raise
        return self

    @contextlib.contextmanager
    def _dispatch(self) -> Iterator[None]:
        previous = self._dispatching
        self._dispatching = True
        try:
            yield
        finally:
            self._dispatching = previous

    def _decode(self, value: object) -> Any:
        return self._bridge.decode(cast("Any", value))

    @property
    def _client(self) -> Any:
        return self._target.client if self._target.client is not None else self

    @property
    def contract(self) -> Any:
        """The env's contract."""
        return self._native.env_contract

    @property
    def num_envs(self) -> int:
        """Lanes a binding drives: one."""
        return 1

    @property
    def render_mode(self) -> str:
        """The env's render mode."""
        return str(getattr(self.contract, "render_mode", "") or "")

    def render(self) -> object:
        """The env's rendered frame, or ``None``."""
        self._require_open()
        if self._dispatching and self._target.client is not None:
            return self._target.client.render()
        return self._native.render()

    def read(self, observation: object, item: object) -> Any:
        """One role (or model-input leaf) read off ``observation``."""
        self._require_open()
        reader = self._read_cache.get(item)
        if reader is None:
            adapter, roles = resolve_read_adapter(self.contract, (item,), False)
            reader = Reader(adapter, roles, self._bridge)
            self._read_cache[item] = reader
        return reader(observation)[reader.roles[0]]

    def frame_sources(self) -> FrameSources:
        """The env's image roles and ``render()`` source."""
        return discover_frame_sources(self.contract, self)

    @property
    def done(self) -> bool:
        """Whether the episode ended (or none started yet)."""
        return self._native.done

    def reset(
        self, *, seed: int | None = None, options: Mapping[str, Any] | None = None
    ) -> tuple[ObsT, Mapping[str, Any]]:
        """Start the next episode: its first observation and the reset's info.

        ``options`` reach the env's ``reset(options=...)`` over the model's own
        :meth:`~rlmesh.Model.reset_options`.
        """
        self._require_open()
        with self._failing():
            observation, info = self._native.reset(
                seed, dict(options) if options else None
            )
        return cast(ObsT, self._decode(observation)), info

    def step(
        self, action: ActT | None = None
    ) -> tuple[ObsT, float, bool, bool, Mapping[str, Any]]:
        """Step the env under ``action``, or the model's when ``None``."""
        self._require_open()
        given = None if action is None else self._bridge.encode(action)
        with self._failing():
            observation, reward, terminated, truncated, info = self._native.step(given)
        return (
            cast(ObsT, self._decode(observation)),
            reward,
            terminated,
            truncated,
            info,
        )

    def close(self) -> None:
        """End the episode, release the env, and fire ``on_run_end`` (idempotent)."""
        if self._closed:
            return
        try:
            self._native.close()
        finally:
            self._closed = True
            if self._view is not None:
                self._view.close(hold=not self._failed)

    def __enter__(self) -> Binding[ObsT, ActT]:
        return self

    def __exit__(self, *exc: object) -> None:
        self._failed = self._failed or exc[0] is not None
        self.close()

    @contextlib.contextmanager
    def _failing(self) -> Iterator[None]:
        try:
            yield
        except BaseException:
            self._failed = True
            raise

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("the binding is closed")

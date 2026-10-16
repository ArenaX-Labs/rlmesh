"""One model's call queue and lifecycle: corner and hook calls run one at a time.

Every corner and hook call on one :class:`~rlmesh.Model` -- from its local
sessions, its native ``run`` loop and its ``serve`` -- passes through the
model's :class:`ModelCalls`, so two threaded sessions never overlap inside the
model, and a corner that calls back into its own model fails with
``RuntimeError`` instead of deadlocking. ``close()`` is terminal: it stops a
running ``serve``, waits for in-flight calls, fires the close hook once, and
every later corner call, ``run``, ``session`` or ``serve`` raises
``RuntimeError``.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any, TypeVar

_R = TypeVar("_R")

CLOSED_TEXT = "the model is closed"


def _reentry_text(what: str) -> str:
    return (
        f"{what} re-entered its own Model from inside a corner or hook; a model's "
        "calls are serialized, so this would deadlock"
    )


class ModelCalls:
    """The serialized call gate and close state of one model."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = threading.Condition()
        self._owner: int | None = None
        self._phase: str = "open"
        self._closing_thread: int | None = None
        self._refuse = False
        self._serves: dict[int, Callable[[], None]] = {}
        self._next_serve = 0

    # -- calls ---------------------------------------------------------------

    def in_call(self) -> bool:
        """Whether the current thread is inside one of this model's calls."""
        return self._owner == threading.get_ident()

    def check_not_reentrant(self, what: str) -> None:
        """Refuse ``what`` from inside this model's own corner or hook."""
        if self.in_call():
            raise RuntimeError(_reentry_text(what))

    def check_open(self, what: str) -> None:
        """Refuse ``what`` once ``close()`` has begun, or from inside a call."""
        self.check_not_reentrant(what)
        if self._phase != "open":
            raise RuntimeError(CLOSED_TEXT)

    @property
    def closed(self) -> bool:
        """Whether ``close()`` has begun refusing calls."""
        return self._refuse

    def _run(
        self,
        fn: Callable[..., _R],
        args: tuple[Any, ...],
        kwargs: dict[str, Any] | None = None,
    ) -> _R:
        with self._lock:
            if self._refuse:
                raise RuntimeError(CLOSED_TEXT)
            self._owner = threading.get_ident()
            try:
                return fn(*args, **(kwargs or {}))
            finally:
                self._owner = None

    def call(self, fn: Callable[..., _R], *args: Any, **kwargs: Any) -> _R:
        """Run one corner call behind every call already running."""
        self.check_not_reentrant("a corner call")
        if self._refuse:
            raise RuntimeError(CLOSED_TEXT)
        return self._run(fn, args, kwargs)

    def hook(self, fn: Callable[..., object], *args: Any) -> None:
        """Run one episode hook; after ``close()`` it submits nothing."""
        if self._refuse:
            return
        self.check_not_reentrant("a model hook")
        try:
            self._run(fn, args)
        except RuntimeError as error:
            if self._refuse and str(error) == CLOSED_TEXT:
                return
            raise

    def gated(self, fn: Callable[..., _R] | None) -> Callable[..., _R] | None:
        """``fn`` with every call through :meth:`call`."""
        if fn is None:
            return None
        corner = fn

        def through_gate(*args: Any, **kwargs: Any) -> _R:
            return self.call(corner, *args, **kwargs)

        through_gate.__wrapped__ = corner  # type: ignore[attr-defined]
        return through_gate

    # -- serves --------------------------------------------------------------

    def register_serve(self, stop: Callable[[], None]) -> int:
        """Record a running serve that ``close()`` stops; refused once closing."""
        self.check_open("serve()")
        with self._state:
            if self._phase != "open":
                raise RuntimeError(CLOSED_TEXT)
            token = self._next_serve
            self._next_serve += 1
            self._serves[token] = stop
            return token

    def deregister_serve(self, token: int) -> None:
        """A serve stopped."""
        with self._state:
            self._serves.pop(token, None)
            self._state.notify_all()

    # -- close ---------------------------------------------------------------

    def closing_here(self) -> bool:
        """Whether this thread is running the model's close (an inner ``super().close()``)."""
        return self._closing_thread == threading.get_ident()

    def begin_close(self) -> bool:
        """Start the one close: stop every serve and refuse new calls.

        Returns ``False`` when another close ran or is running (after waiting
        for it), so the caller fires nothing.
        """
        self.check_not_reentrant("close()")
        with self._state:
            if self._phase == "closed":
                return False
            if self._phase == "closing":
                # Another thread's close; finish_close() flips the phase.
                self._state.wait_for(lambda: self._phase == "closed")
                return False
            self._phase = "closing"
            self._closing_thread = threading.get_ident()
            stops = list(self._serves.values())
        for stop in stops:
            stop()
        with self._state:
            while self._serves:
                self._state.wait()
        self._refuse = True
        return True

    def run_close_hook(self, hook: Callable[[], object]) -> None:
        """Fire the close hook behind every in-flight call."""
        with self._lock:
            self._owner = threading.get_ident()
            try:
                hook()
            finally:
                self._owner = None

    def finish_close(self) -> None:
        """Mark the model closed."""
        with self._state:
            self._refuse = True
            self._phase = "closed"
            self._closing_thread = None
            self._state.notify_all()

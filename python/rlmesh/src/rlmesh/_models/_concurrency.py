"""Overlapping predicts: an async ``predict`` on its own loop, a sync one on threads."""

from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import inspect
import threading
from collections.abc import Callable, Coroutine, Sequence
from typing import Any

#: The cap an ``async def predict`` runs under when the model sets no ``concurrency``.
DEFAULT_ASYNC_CONCURRENCY = 32


def is_async(fn: object) -> bool:
    """Whether ``fn`` is an ``async def`` function or method."""
    return inspect.iscoroutinefunction(fn)


def resolve_limit(declared: object, *, asynchronous: bool, owner: str) -> int:
    """The number of predicts that may overlap: ``declared``, else the async default or 1."""
    if declared is None:
        return DEFAULT_ASYNC_CONCURRENCY if asynchronous else 1
    if isinstance(declared, bool) or not isinstance(declared, int) or declared < 1:
        raise TypeError(
            f"{owner}: concurrency must be a positive int (got {declared!r})"
        )
    return declared


class Concurrency:
    """Runs one model's predicts, up to ``limit`` at a time."""

    def __init__(self, limit: int, *, asynchronous: bool, name: str) -> None:
        self.limit = limit
        self.asynchronous = asynchronous
        self._name = name
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._pool: concurrent.futures.ThreadPoolExecutor | None = None

    def sync(self, fn: Callable[..., Coroutine[Any, Any, Any]]) -> Callable[..., Any]:
        """``fn`` as a blocking call: one predict, awaited on the model's loop."""

        @functools.wraps(fn)
        def call(*args: Any) -> Any:
            return asyncio.run_coroutine_threadsafe(
                fn(*args), self._ensure_loop()
            ).result()

        return call

    def map(self, call: Callable[[Any], Any], rows: Sequence[Any]) -> list[Any]:
        """``call`` over ``rows``, overlapping; a row that raised holds its exception."""
        if self.asynchronous:
            return asyncio.run_coroutine_threadsafe(
                self._gather(call, rows), self._ensure_loop()
            ).result()
        futures = [self._ensure_pool().submit(call, row) for row in rows]
        return [_outcome(future) for future in futures]

    async def _gather(
        self, call: Callable[[Any], Any], rows: Sequence[Any]
    ) -> list[Any]:
        gate = asyncio.Semaphore(self.limit)

        async def one(row: Any) -> Any:
            async with gate:
                return await call(row)

        return await asyncio.gather(*(one(row) for row in rows), return_exceptions=True)

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is None:
                loop = asyncio.new_event_loop()
                thread = threading.Thread(
                    target=loop.run_forever, name=f"{self._name}-predict", daemon=True
                )
                thread.start()
                self._loop, self._thread = loop, thread
            return self._loop

    def _ensure_pool(self) -> concurrent.futures.ThreadPoolExecutor:
        with self._lock:
            if self._pool is None:
                self._pool = concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.limit, thread_name_prefix=f"{self._name}-predict"
                )
            return self._pool

    def close(self) -> None:
        """Stop the loop or pool; a later predict starts a fresh one."""
        with self._lock:
            loop, thread, pool = self._loop, self._thread, self._pool
            self._loop = self._thread = self._pool = None
        if pool is not None:
            pool.shutdown(wait=True)
        if loop is not None and thread is not None:
            loop.call_soon_threadsafe(loop.stop)
            thread.join()
            loop.close()


def _outcome(future: concurrent.futures.Future[Any]) -> Any:
    error = future.exception()
    return error if error is not None else future.result()

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
    """Whether ``fn`` is an ``async def`` function, method or ``__call__``."""
    if inspect.iscoroutinefunction(fn):
        return True
    call = inspect.getattr_static(fn, "__call__", None)
    return not inspect.isroutine(fn) and inspect.iscoroutinefunction(call)


def check_limit(declared: object, *, owner: str) -> int | None:
    """``declared`` as a concurrency, refusing anything but ``None`` or a positive int."""
    if declared is None:
        return None
    if isinstance(declared, bool) or not isinstance(declared, int) or declared < 1:
        raise TypeError(
            f"{owner}: concurrency must be a positive int (got {declared!r})"
        )
    return declared


def resolve_limit(declared: object, *, asynchronous: bool, owner: str) -> int:
    """The number of predicts that may overlap: ``declared``, else the async default or 1."""
    limit = check_limit(declared, owner=owner)
    if limit is None:
        return DEFAULT_ASYNC_CONCURRENCY if asynchronous else 1
    return limit


class _Escaped:
    """A ``BaseException`` caught on the loop thread, which it would otherwise stop."""

    def __init__(self, error: BaseException) -> None:
        self.error = error


async def _contained(awaitable: Coroutine[Any, Any, Any]) -> Any:
    try:
        return await awaitable
    except (Exception, asyncio.CancelledError):
        raise
    except BaseException as error:
        return _Escaped(error)


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
        self._local = threading.local()

    def owns_current_thread(self) -> bool:
        """Whether the calling thread is one this model runs its predicts on."""
        return getattr(self._local, "owned", False)

    def _own(self) -> None:
        self._local.owned = True

    def sync(self, fn: Callable[..., Coroutine[Any, Any, Any]]) -> Callable[..., Any]:
        """``fn`` as a blocking call: one predict, awaited on the model's loop."""

        @functools.wraps(fn)
        def call(*args: Any) -> Any:
            result = asyncio.run_coroutine_threadsafe(
                _contained(fn(*args)), self._ensure_loop()
            ).result()
            if isinstance(result, _Escaped):
                raise result.error
            return result

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
                result = await _contained(call(row))
            return result.error if isinstance(result, _Escaped) else result

        return await asyncio.gather(*(one(row) for row in rows), return_exceptions=True)

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._thread is not None and not self._thread.is_alive():
                self._loop, self._thread = None, None
            if self._loop is None:
                loop = asyncio.new_event_loop()

                def run() -> None:
                    self._own()
                    loop.run_forever()

                thread = threading.Thread(
                    target=run, name=f"{self._name}-predict", daemon=True
                )
                thread.start()
                self._loop, self._thread = loop, thread
            return self._loop

    def _ensure_pool(self) -> concurrent.futures.ThreadPoolExecutor:
        with self._lock:
            if self._pool is None:
                self._pool = concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.limit,
                    thread_name_prefix=f"{self._name}-predict",
                    initializer=self._own,
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
            if thread.is_alive():
                asyncio.run_coroutine_threadsafe(_drain(), loop).result()
                loop.call_soon_threadsafe(loop.stop)
                thread.join()
            loop.close()


async def _drain() -> None:
    current = asyncio.current_task()
    leftover = [task for task in asyncio.all_tasks() if task is not current]
    for task in leftover:
        task.cancel()
    await asyncio.gather(*leftover, return_exceptions=True)
    await asyncio.get_running_loop().shutdown_asyncgens()


def _outcome(future: concurrent.futures.Future[Any]) -> Any:
    error = future.exception()
    return error if error is not None else future.result()

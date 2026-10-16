"""Admission order includes callers waiting for the currently active corner."""

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from rlmesh._models._calls import ModelCalls


def _waiting(thread_id):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        frame = sys._current_frames().get(thread_id)
        while frame is not None:
            if frame.f_code.co_name == "_run":
                return
            frame = frame.f_back
        time.sleep(0.001)
    raise AssertionError("call never entered the queue")


def test_close_follows_admitted_predicts_and_hooks_in_order() -> None:
    calls = ModelCalls()
    active = threading.Event()
    release = threading.Event()
    log = []

    def predict():
        active.set()
        assert release.wait(2)
        log.append("active")

    def waiting(kind):
        ids[kind] = threading.get_ident()
        entered[kind].set()
        if kind == "predict":
            calls.call(lambda: log.append(kind))
        else:
            calls.hook(lambda: log.append(kind))

    def close():
        if calls.begin_close():
            try:
                calls.run_close_hook(lambda: log.append("close"))
            finally:
                calls.finish_close()

    ids = {}
    entered = {kind: threading.Event() for kind in ("predict", "hook")}
    with ThreadPoolExecutor(4) as pool:
        first = pool.submit(calls.call, predict)
        assert active.wait(2)
        queued = []
        for kind in entered:
            queued.append(pool.submit(waiting, kind))
            assert entered[kind].wait(2)
            _waiting(ids[kind])
        closing = pool.submit(close)
        deadline = time.monotonic() + 2
        while calls._phase == "open" and time.monotonic() < deadline:
            time.sleep(0.001)
        release.set()
        for future in [first, *queued, closing]:
            future.result(timeout=3)
    assert log == ["active", "predict", "hook", "close"]

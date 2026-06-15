from __future__ import annotations

import os
import sys
import time
from typing import Any, cast

import pytest


class _FakeProcess:
    """Minimal stand-in for subprocess.Popen exposing a writable stdin pipe."""

    def __init__(self, write_fd: int) -> None:
        self.stdin = os.fdopen(write_fd, "wb", buffering=0)

    def poll(self) -> None:
        return None


@pytest.mark.skipif(
    sys.platform == "win32", reason="non-blocking pipe write test is POSIX-only"
)
def test_push_viewer_packet_does_not_hang_on_stalled_viewer() -> None:
    from rlmesh.client import _viewer as viewer_mod
    from rlmesh.client._viewer import ViewerMixin, ViewerProcess

    read_fd, write_fd = os.pipe()
    os.set_blocking(write_fd, False)
    process = _FakeProcess(write_fd)

    mixin = ViewerMixin()
    mixin._viewer = ViewerProcess(cast(Any, process), env_index=0, fps_limit=None)

    # A large packet that cannot fit in the (undrained) OS pipe buffer.
    big_packet = b"\x00" * (4 * 1024 * 1024)

    start = time.monotonic()
    with pytest.warns(RuntimeWarning, match="not draining frames"):
        mixin._push_viewer_packet(big_packet)
    elapsed = time.monotonic() - start

    # The reader never drains, so the push must time out and drop the viewer
    # well within a small multiple of the configured write timeout instead of
    # blocking forever.
    assert elapsed < viewer_mod.VIEWER_WRITE_TIMEOUT_SECONDS + 1.5
    assert mixin._viewer is None

    os.close(read_fd)


@pytest.mark.skipif(
    sys.platform == "win32", reason="non-blocking pipe write test is POSIX-only"
)
def test_push_viewer_packet_writes_complete_frame_when_drained() -> None:
    from rlmesh.client._viewer import ViewerMixin, ViewerProcess

    read_fd, write_fd = os.pipe()
    os.set_blocking(write_fd, False)
    process = _FakeProcess(write_fd)

    mixin = ViewerMixin()
    mixin._viewer = ViewerProcess(cast(Any, process), env_index=0, fps_limit=None)

    payload = b"frame-bytes"
    mixin._push_viewer_packet(payload)

    # Header is 5 bytes ("<BI"), followed by the payload.
    header = os.read(read_fd, 5)
    body = os.read(read_fd, len(payload))
    assert header[0] == 1
    assert int.from_bytes(header[1:5], "little") == len(payload)
    assert body == payload
    assert mixin._viewer is not None

    os.close(read_fd)
    mixin._shutdown_viewer(wait=False)

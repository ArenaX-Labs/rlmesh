"""Shared pytest setup for the Python SDK suites."""

from __future__ import annotations

import atexit
import os
import shutil
import socket
import tempfile

import pytest
from rlmesh._editions import WORKFLOW_EDITION_ENV_VAR

# These suites are not authored against a sealed edition -- they exercise
# whatever this build implements -- so they take the documented "deliberately
# undeclared" switch rather than a declaration. It keeps the one-time float
# warning out of every unrelated test (and out of the example subprocesses,
# which inherit this environment) while leaving it assertable: a test that
# wants the warning deletes the variable with `monkeypatch.delenv`.
os.environ.setdefault(WORKFLOW_EDITION_ENV_VAR, "")

# A loopback host leaves its address key for this user's other processes in
# the runtime directory: keep the suites' (and their subprocesses') out of the
# user's real one.
if "RLMESH_RUNTIME_DIR" not in os.environ:
    _runtime_dir = tempfile.mkdtemp(prefix="rlmesh-test-runtime-")
    atexit.register(shutil.rmtree, _runtime_dir, ignore_errors=True)
    os.environ["RLMESH_RUNTIME_DIR"] = _runtime_dir


@pytest.fixture
def session_port() -> int:
    """A free local port, on TCP and UDP both: a session host listens on it
    (and offers boards UDP on the same number)."""
    for _ in range(64):
        with (
            socket.socket() as tcp,
            socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp,
        ):
            tcp.bind(("127.0.0.1", 0))
            port = int(tcp.getsockname()[1])
            try:
                udp.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("no free local port")

"""Shared pytest setup for the Python SDK suites."""

from __future__ import annotations

import os
import socket

import pytest
from rlmesh._editions import WORKFLOW_EDITION_ENV_VAR

# These suites are not authored against a sealed edition -- they exercise
# whatever this build implements -- so they take the documented "deliberately
# undeclared" switch rather than a declaration. It keeps the one-time float
# warning out of every unrelated test (and out of the example subprocesses,
# which inherit this environment) while leaving it assertable: a test that
# wants the warning deletes the variable with `monkeypatch.delenv`.
os.environ.setdefault(WORKFLOW_EDITION_ENV_VAR, "")


@pytest.fixture
def session_port() -> int:
    """A free local port whose next three are free too: a session host
    listens on all four (one per traffic class)."""
    for _ in range(64):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = int(probe.getsockname()[1])
        held: list[socket.socket] = []
        try:
            for offset in range(1, 4):
                sock = socket.socket()
                held.append(sock)
                sock.bind(("127.0.0.1", port + offset))
        except OSError:
            continue
        finally:
            for sock in held:
                sock.close()
        return port
    raise RuntimeError("no four consecutive free local ports")

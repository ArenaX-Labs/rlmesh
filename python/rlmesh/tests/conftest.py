from __future__ import annotations

import atexit
import os
import shutil
import socket
import tempfile

import pytest
from rlmesh._editions import WORKFLOW_EDITION_ENV_VAR

os.environ.setdefault(WORKFLOW_EDITION_ENV_VAR, "")

if "RLMESH_RUNTIME_DIR" not in os.environ:
    _runtime_dir = tempfile.mkdtemp(prefix="rlmesh-test-runtime-")
    atexit.register(shutil.rmtree, _runtime_dir, ignore_errors=True)
    os.environ["RLMESH_RUNTIME_DIR"] = _runtime_dir


@pytest.fixture
def session_port() -> int:
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

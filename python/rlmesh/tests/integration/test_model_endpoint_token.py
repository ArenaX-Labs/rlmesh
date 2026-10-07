"""The Python serving path enforces the deployment endpoint credential."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("token", ["", " \t\n", "deployment-token"])
def test_model_serve_reads_endpoint_token(token: str) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import rlmesh; "
            "rlmesh.Model(lambda observation: 0).serve('127.0.0.1:0', "
            "options=rlmesh.ServeOptions(idle_timeout_seconds=0.1))",
        ],
        env={**os.environ, "RLMESH_MODEL_ENDPOINT_TOKEN": token},
        capture_output=True,
        text=True,
        timeout=30,
    )
    if token.strip():
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0
        assert "RLMESH_MODEL_ENDPOINT_TOKEN must not be empty" in result.stderr

"""The example envs' tags the 0.1 reader check reads are the ones they publish now."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("numpy")

SCRIPT = (
    Path(__file__).resolve().parents[4]
    / "crates"
    / "rlmesh-adapters-compat"
    / "write_example_tags.py"
)


@pytest.mark.skipif(not SCRIPT.exists(), reason="needs the repository checkout")
def test_recorded_example_env_tags_are_current() -> None:
    done = subprocess.run(
        [sys.executable, str(SCRIPT), "--check"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr

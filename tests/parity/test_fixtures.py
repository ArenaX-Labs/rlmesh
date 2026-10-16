"""Run shared behavioural contracts through Python and the two C ABI consumers."""

from __future__ import annotations

import json
import os
import pathlib
import subprocess

import pytest

from .python_runner import execute
from .support import ROOT, compare, native_args, scenarios

CASES = scenarios()


@pytest.fixture(autouse=True)
def quiet_edition(monkeypatch):
    """Make floating deliberate so only the fixture's warning is emitted."""
    monkeypatch.setenv("RLMESH_WORKFLOW_EDITION", "")


@pytest.mark.parametrize("language", ["python", "c", "cpp"])
@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_scenario(language, case):
    """Every applicable scenario must run and compare its observed output."""
    if language not in case.get("languages", ["python", "rust", "c", "cpp"]):
        pytest.skip(case["why"])
    if language == "python":
        observed = execute(case)
    else:
        binary = (
            pathlib.Path(os.environ.get("RLMESH_PARITY_BIN", ROOT / "target/parity"))
            / language
        )
        result = subprocess.run(
            [str(binary), *native_args(case)],
            check=True,
            text=True,
            capture_output=True,
            timeout=30,
        )
        observed = json.loads(result.stdout.splitlines()[-1])
    compare(case, observed)

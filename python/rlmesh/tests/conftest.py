"""Shared pytest setup for the Python SDK suites."""

from __future__ import annotations

import os

from rlmesh._editions import WORKFLOW_EDITION_ENV_VAR

# These suites are not authored against a sealed edition -- they exercise
# whatever this build implements -- so they take the documented "deliberately
# undeclared" switch rather than a declaration. It keeps the one-time float
# warning out of every unrelated test (and out of the example subprocesses,
# which inherit this environment) while leaving it assertable: a test that
# wants the warning deletes the variable with `monkeypatch.delenv`.
os.environ.setdefault(WORKFLOW_EDITION_ENV_VAR, "")

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import rlmesh.__main__ as cli_main


def test_python_entrypoint_marks_wheel_distribution_and_forwards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def run_cli(argv: list[str]) -> int:
        captured["argv"] = argv
        captured["distribution"] = os.environ["RLMESH_CLI_DISTRIBUTION"]
        return 23

    monkeypatch.delenv("RLMESH_CLI_DISTRIBUTION", raising=False)
    monkeypatch.setattr(cli_main, "find_repo_root", lambda: None)
    monkeypatch.setattr(cli_main, "_load_extension_cli", lambda: run_cli)

    assert cli_main.main(["version"]) == 23
    assert captured == {
        "argv": ["version"],
        "distribution": "python-wheel",
    }


def test_python_entrypoint_preserves_existing_distribution_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RLMESH_CLI_DISTRIBUTION", "custom")
    monkeypatch.setattr(cli_main, "find_repo_root", lambda: None)
    monkeypatch.setattr(cli_main, "_load_extension_cli", lambda: (lambda argv: 0))

    assert cli_main.main(["version"]) == 0
    assert os.environ["RLMESH_CLI_DISTRIBUTION"] == "custom"


def test_python_entrypoint_uses_cargo_fallback_for_editable_checkout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def load_extension_cli() -> object:
        raise ImportError("native extension unavailable")

    def run_subprocess(command: list[str], *, check: bool, cwd: Path) -> object:
        captured["command"] = command
        captured["check"] = check
        captured["cwd"] = cwd
        captured["distribution"] = os.environ["RLMESH_CLI_DISTRIBUTION"]
        return SimpleNamespace(returncode=7)

    monkeypatch.delenv("RLMESH_CLI_DISTRIBUTION", raising=False)
    monkeypatch.setattr(cli_main, "find_repo_root", lambda: tmp_path)
    monkeypatch.setattr(cli_main, "_load_extension_cli", load_extension_cli)
    monkeypatch.setattr(cli_main.shutil, "which", lambda name: "/usr/bin/cargo")
    monkeypatch.setattr(cli_main.subprocess, "run", run_subprocess)

    assert cli_main.main(["version"]) == 7
    assert captured == {
        "command": [
            "/usr/bin/cargo",
            "run",
            "-p",
            "rlmesh-cli",
            "--bin",
            "rlmesh",
            "--",
            "version",
        ],
        "check": False,
        "cwd": tmp_path,
        "distribution": "python-source",
    }


def test_python_entrypoint_errors_without_native_module_or_checkout(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def load_extension_cli() -> object:
        raise ImportError("native extension unavailable")

    monkeypatch.setattr(cli_main, "find_repo_root", lambda: None)
    monkeypatch.setattr(cli_main, "_load_extension_cli", load_extension_cli)
    monkeypatch.setattr(cli_main.shutil, "which", lambda name: None)

    assert cli_main.main(["version"]) == 1
    captured = capsys.readouterr()
    assert "native module could not be imported" in captured.err


def test_import_error_from_cli_execution_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ImportError raised while the CLI runs is a failure inside the CLI, not a
    missing native module; it must propagate rather than being swallowed into the
    cargo/native-module fallback path."""

    def run_cli(argv: list[str]) -> int:
        raise ImportError("boom from inside the CLI")

    monkeypatch.setattr(cli_main, "find_repo_root", lambda: None)
    monkeypatch.setattr(cli_main, "_load_extension_cli", lambda: run_cli)

    with pytest.raises(ImportError, match="boom from inside the CLI"):
        cli_main.main(["version"])

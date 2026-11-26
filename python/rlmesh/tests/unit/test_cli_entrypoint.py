from __future__ import annotations

import os
import sys
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
    monkeypatch.delenv("RLMESH_PYTHON", raising=False)
    monkeypatch.setattr(cli_main, "find_repo_root", lambda: None)
    monkeypatch.setattr(cli_main, "_load_extension_cli", lambda: run_cli)

    assert cli_main.main(["version"]) == 23
    assert captured == {
        "argv": ["version"],
        "distribution": "python-wheel",
    }
    assert os.environ["RLMESH_PYTHON"] == sys.executable


def test_wheel_entrypoint_execs_the_bundled_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def execv(path: str, args: list[str]) -> None:
        captured["exec"] = (path, args)
        captured["python"] = os.environ["RLMESH_PYTHON"]
        raise SystemExit(0)

    monkeypatch.delenv("RLMESH_PYTHON", raising=False)
    monkeypatch.setattr(cli_main, "find_repo_root", lambda: None)
    monkeypatch.setattr(cli_main, "_bundled_cli", lambda: "/site/rlmesh/_bin/rlmesh")
    monkeypatch.setattr(cli_main.os, "execv", execv)

    with pytest.raises(SystemExit):
        cli_main.credential_helper_main(["get"])
    assert captured == {
        "exec": (
            "/site/rlmesh/_bin/rlmesh",
            [
                "/site/rlmesh/_bin/rlmesh",
                "registry",
                "credential-helper",
                "get",
            ],
        ),
        "python": sys.executable,
    }


def test_checkout_entrypoint_never_execs_a_bundled_cli(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(cli_main, "find_repo_root", lambda: tmp_path)
    monkeypatch.setattr(cli_main, "_bundled_cli", lambda: "/site/rlmesh/_bin/rlmesh")
    monkeypatch.setattr(cli_main.os, "execv", pytest.fail)
    monkeypatch.setattr(cli_main, "_load_extension_cli", lambda: (lambda argv: 5))

    assert cli_main.main(["version"]) == 5


def test_bundled_cli_is_macos_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_main.sys, "platform", "linux")
    assert cli_main._bundled_cli() is None


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
    def run_cli(argv: list[str]) -> int:
        raise ImportError("boom from inside the CLI")

    monkeypatch.setattr(cli_main, "find_repo_root", lambda: None)
    monkeypatch.setattr(cli_main, "_load_extension_cli", lambda: run_cli)

    with pytest.raises(ImportError, match="boom from inside the CLI"):
        cli_main.main(["version"])

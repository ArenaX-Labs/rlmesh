"""The editable-install stale-extension check in ``rlmesh._load_native``."""

from __future__ import annotations

import subprocess
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from rlmesh import _load_native

BUILT = "aaaaaaaaaaaa"
MOVED = "bbbbbbbbbbbb"


def _checkout(tmp_path: Path) -> Path:
    """A fake rlmesh checkout holding an in-tree extension; return the .so path."""
    (tmp_path / "rlmesh.toml").write_text("")
    (tmp_path / ".git").write_text("gitdir: elsewhere\n")
    package = tmp_path / "python" / "rlmesh" / "src" / "rlmesh"
    package.mkdir(parents=True)
    extension = package / "_rlmesh.abi3.so"
    extension.write_bytes(b"")
    return extension


def _native(extension: Path, *, source: str = "git", git: str | None = BUILT) -> Any:
    info = SimpleNamespace(build_source=source, git=git)
    return SimpleNamespace(__file__=str(extension), build_info=lambda: info)


@pytest.fixture(autouse=True)
def _fresh(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_load_native, "_stale_checked", False)
    monkeypatch.delenv(_load_native.STALE_CHECK_ENV_VAR, raising=False)


def _head(monkeypatch: pytest.MonkeyPatch, head: str | None) -> list[Path]:
    calls: list[Path] = []

    def fake(checkout: Path) -> str | None:
        calls.append(checkout)
        return head

    monkeypatch.setattr(_load_native, "_git_head", fake)
    return calls


def test_warns_once_when_head_moved_past_the_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _head(monkeypatch, MOVED)
    native = _native(_checkout(tmp_path), git=f"{BUILT}.dirty.0123456789abcdef")

    with pytest.warns(UserWarning, match="native extension is stale") as record:
        _load_native._warn_if_stale(native)
    message = str(record[0].message)
    assert BUILT in message
    assert MOVED in message
    assert "mise run build:python:develop" in message
    assert "RLMESH_STALE_CHECK=0" in message
    assert calls == [tmp_path.resolve()]

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _load_native._warn_if_stale(native)
    assert len(calls) == 1, "the check runs once per process"


@pytest.mark.parametrize(
    ("head", "source", "git"),
    [
        pytest.param(BUILT, "git", BUILT, id="current"),
        pytest.param(
            BUILT, "git", f"{BUILT}.dirty.0123456789abcdef", id="current-dirty"
        ),
        pytest.param(None, "git", BUILT, id="git-unavailable"),
        pytest.param(MOVED, "release", None, id="release-build"),
        pytest.param(MOVED, "package", None, id="package-build"),
    ],
)
def test_quiet_when_current_or_not_a_source_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    head: str | None,
    source: str,
    git: str | None,
) -> None:
    _head(monkeypatch, head)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _load_native._warn_if_stale(
            _native(_checkout(tmp_path), source=source, git=git)
        )


def test_wheels_are_never_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Even a git-stamped wheel installed into a venv inside the checkout.
    calls = _head(monkeypatch, MOVED)
    (tmp_path / "rlmesh.toml").write_text("")
    (tmp_path / ".git").mkdir()
    package = tmp_path / ".venv" / "lib" / "python3.12" / "site-packages" / "rlmesh"
    package.mkdir(parents=True)
    extension = package / "_rlmesh.abi3.so"
    extension.write_bytes(b"")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _load_native._warn_if_stale(_native(extension))
    assert calls == []


def test_outside_a_checkout_is_not_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _head(monkeypatch, MOVED)
    extension = tmp_path / "_rlmesh.abi3.so"
    extension.write_bytes(b"")
    _load_native._warn_if_stale(_native(extension))
    assert calls == []


@pytest.mark.parametrize("value", ["0", "false", "No", " off "])
def test_opt_out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    calls = _head(monkeypatch, MOVED)
    monkeypatch.setenv(_load_native.STALE_CHECK_ENV_VAR, value)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _load_native._warn_if_stale(_native(_checkout(tmp_path)))
    assert calls == []


def test_never_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken() -> Any:
        raise RuntimeError("build_info exploded")

    native = SimpleNamespace(__file__=str(_checkout(tmp_path)), build_info=broken)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _load_native._warn_if_stale(native)


def test_git_head_swallows_git_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing(*_args: Any, **_kwargs: Any) -> Any:
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", missing)
    assert _load_native._git_head(tmp_path) is None

    def timeout(*_args: Any, **_kwargs: Any) -> Any:
        raise subprocess.TimeoutExpired("git", 2)

    monkeypatch.setattr(subprocess, "run", timeout)
    assert _load_native._git_head(tmp_path) is None

    def failed(*_args: Any, **_kwargs: Any) -> Any:
        return subprocess.CompletedProcess(["git"], 128, stdout="", stderr="not a repo")

    monkeypatch.setattr(subprocess, "run", failed)
    assert _load_native._git_head(tmp_path) is None


def test_load_native_still_returns_the_symbol() -> None:
    import rlmesh._rlmesh as native

    assert _load_native.load_native("build_info") is native.build_info

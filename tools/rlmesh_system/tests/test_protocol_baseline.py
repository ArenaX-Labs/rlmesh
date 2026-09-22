from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from scripts import protocol_baseline


def test_regenerating_snapshot_cannot_hide_a_wire_break(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = "crates/rlmesh-proto/proto/rlmesh/test.proto"
    original = b'syntax = "proto3"; package test; message Value { string text = 1; }\n'
    live = tmp_path / name
    live.parent.mkdir(parents=True)
    live.write_bytes(original)
    (tmp_path / "rlmesh.toml").write_text(
        '[protocol]\ncurrent_generation = "rlmesh-wire-v1"\n'
    )
    run = subprocess.run

    def tagged_run(command: list[str], **kwargs: Any) -> Any:
        if command[0] == "git":
            assert command[-1] == "refs/tags/v0.1.0^{commit}"
            return subprocess.CompletedProcess(command, 0)
        return run(command, **kwargs)

    def tagged_output(command: list[str], **kwargs: Any) -> str | bytes:
        if command[1] == "ls-tree":
            assert command[4] == "v0.1.0"
            return name + "\n"
        assert command == ["git", "show", f"v0.1.0:{name}"]
        return original

    monkeypatch.setattr(subprocess, "run", tagged_run)
    monkeypatch.setattr(subprocess, "check_output", tagged_output)
    assert protocol_baseline.breaking(tmp_path) == 0
    live.write_bytes(original.replace(b"text = 1", b"text = 2"))
    assert protocol_baseline.regen(tmp_path) == 0
    assert protocol_baseline.verify(tmp_path) == 0
    assert protocol_baseline.breaking(tmp_path) != 0


def test_later_release_requires_the_sealed_tag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "Cargo.toml").write_text('[workspace.package]\nversion = "0.1.1"\n')
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 1),
    )
    with pytest.raises(SystemExit, match=r"v0\.1\.0 tag is required"):
        protocol_baseline.breaking(tmp_path)

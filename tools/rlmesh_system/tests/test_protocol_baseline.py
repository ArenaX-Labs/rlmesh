from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from scripts import protocol_baseline


def test_regenerating_snapshot_cannot_hide_a_wire_break(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = "crates/rlmesh-session-core/proto/rlmesh/test.proto"
    original = b'syntax = "proto3"; package test; message Value { string text = 1; }\n'
    live = tmp_path / name
    live.parent.mkdir(parents=True)
    live.write_bytes(original)
    (tmp_path / "rlmesh.toml").write_text(
        '[protocols."rlmesh-realtime-v1"]\n'
        'proto = "crates/rlmesh-session-core/proto"\n'
        'baseline = "baseline"\nstatus = "frozen"\nfrozen_in = "v0.2.0"\n'
    )
    run = subprocess.run

    def tagged_run(command: list[str], **kwargs: Any) -> Any:
        if command[0] == "git":
            assert command[-1] == "refs/tags/v0.2.0^{commit}"
            return subprocess.CompletedProcess(command, 0)
        return run(command, **kwargs)

    def tagged_output(command: list[str], **kwargs: Any) -> str | bytes:
        if command[1] == "ls-tree":
            assert command[4] == "v0.2.0"
            return name + "\n"
        assert command == ["git", "show", f"v0.2.0:{name}"]
        return original

    monkeypatch.setattr(subprocess, "run", tagged_run)
    monkeypatch.setattr(subprocess, "check_output", tagged_output)
    assert protocol_baseline.breaking_protocols(tmp_path) == 0
    live.write_bytes(original.replace(b"text = 1", b"text = 2"))
    assert protocol_baseline.regen_protocol(tmp_path, "rlmesh-realtime-v1") == 0
    assert protocol_baseline.verify(tmp_path) == 0
    assert protocol_baseline.breaking_protocols(tmp_path) != 0


def test_frozen_protocol_requires_its_release_tag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "rlmesh.toml").write_text(
        '[protocols."rlmesh-realtime-v1"]\n'
        'proto = "crates/rlmesh-session-core/proto"\n'
        'baseline = "baseline"\nstatus = "frozen"\nfrozen_in = "v0.2.0"\n'
    )

    def missing_tag(command: list[str], **kwargs: Any) -> Any:
        raise subprocess.CalledProcessError(128, command)

    monkeypatch.setattr(subprocess, "check_output", missing_tag)
    with pytest.raises(SystemExit, match=r"cannot read v0\.2\.0"):
        protocol_baseline.breaking_protocols(tmp_path)

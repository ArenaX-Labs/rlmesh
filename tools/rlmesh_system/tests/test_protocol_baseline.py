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


def test_a_wire_change_without_a_revision_bump_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    added = (
        "diff --git a/v/control.txt b/v/control.txt\n--- a/v/control.txt\n"
        "+++ b/v/control.txt\n@@ -3,0 +4 @@\n+Attach#2 0a01\n"
        "diff --git a/v/new.txt b/v/new.txt\n--- /dev/null\n+++ b/v/new.txt\n"
        "@@ -0,0 +1 @@\n+Ping#1 00\n"
    )
    changed = added + (
        "diff --git a/v/model.txt b/v/model.txt\n--- a/v/model.txt\n"
        "+++ b/v/model.txt\n@@ -1 +1 @@\n-Resolve#1 0a\n+Resolve#1 0b\n"
    )
    assert protocol_baseline._wire_break([], added) == []
    assert protocol_baseline._wire_break([], changed) == ["v/model.txt"]
    assert protocol_baseline._wire_break(["p/session.proto"], "") == ["p/session.proto"]

    version = "pub const WIRE_REVISION: u32 = {};\n"
    released = {"revision": 2}

    def git(root: Path, *args: str) -> str:
        if args[0] == "describe":
            return "v0.2.0-alpha.2\n"
        if args[0] == "show":
            return version.format(released["revision"])
        return "p/session.proto\n" if "--name-only" in args else ""

    root = Path(protocol_baseline.__file__).resolve().parents[1]
    current = protocol_baseline._wire_revision(
        (root / protocol_baseline.WIRE_VERSION).read_text(encoding="utf-8")
    )
    monkeypatch.setattr(protocol_baseline, "_git", git)
    entry = {"proto": "p"}
    released["revision"] = current
    [error] = protocol_baseline._unbumped_wire_change(root, entry)
    assert "WIRE_REVISION is still" in error, error
    released["revision"] = current - 1
    assert protocol_baseline._unbumped_wire_change(root, entry) == []


def test_the_wire_check_fails_loudly_without_a_release_tag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def git(root: Path, *args: str) -> str:
        raise subprocess.CalledProcessError(128, ["git", *args])

    monkeypatch.setattr(protocol_baseline, "_git", git)
    with pytest.raises(SystemExit, match="shallow clone"):
        protocol_baseline._unbumped_wire_change(Path("."), {"proto": "p"})

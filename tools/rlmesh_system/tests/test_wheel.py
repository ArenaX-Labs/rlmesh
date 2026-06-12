from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from rlmesh_system.support.wheel import (
    WheelStatus,
    inspect_wheel,
    select_wheel,
    wheel_tag_prefix,
)


def _touch_wheel(path: Path, *, mtime: float) -> None:
    path.write_bytes(b"")
    os.utime(path, (mtime, mtime))


def test_wheel_tag_prefix_maps_python_versions() -> None:
    assert wheel_tag_prefix("3.10") == "cp310"
    assert wheel_tag_prefix("3.11") == "cp311"
    assert wheel_tag_prefix("3.12") == "cp311"


def test_select_wheel_matches_python_tag(tmp_path: Path) -> None:
    cp310 = tmp_path / "rlmesh-0.1.0-cp310-cp310-linux_x86_64.whl"
    cp311 = tmp_path / "rlmesh-0.1.0-cp311-abi3-linux_x86_64.whl"
    _touch_wheel(cp310, mtime=1000.0)
    _touch_wheel(cp311, mtime=1000.0)

    assert select_wheel(tmp_path, "3.10") == cp310
    assert select_wheel(tmp_path, "3.11") == cp311


def test_select_wheel_prefers_newest_matching(tmp_path: Path) -> None:
    older = tmp_path / "rlmesh-0.1.0-cp311-abi3-linux_x86_64.whl"
    newer = tmp_path / "rlmesh-0.2.0-cp311-abi3-linux_x86_64.whl"
    _touch_wheel(older, mtime=1000.0)
    _touch_wheel(newer, mtime=2000.0)

    assert select_wheel(tmp_path, "3.11") == newer


def test_select_wheel_returns_none_without_match(tmp_path: Path) -> None:
    _touch_wheel(tmp_path / "rlmesh-0.1.0-cp310-cp310-linux.whl", mtime=1000.0)

    assert select_wheel(tmp_path, "3.11") is None


def test_wheel_status_flags_stale_wheel() -> None:
    built = datetime(2026, 1, 1, tzinfo=timezone.utc)
    commit = datetime(2026, 6, 1, tzinfo=timezone.utc)
    status = WheelStatus(
        path=Path("rlmesh-0.1.0-cp311-abi3.whl"),
        modified_at=built,
        source_commit_at=commit,
    )

    assert status.is_stale
    header = "\n".join(status.header_lines())
    assert "WARNING" in header
    assert "rlmesh-0.1.0-cp311-abi3.whl" in header


def test_wheel_status_fresh_wheel_has_no_warning() -> None:
    status = WheelStatus(
        path=Path("rlmesh-0.1.0-cp311-abi3.whl"),
        modified_at=datetime(2026, 6, 2, tzinfo=timezone.utc),
        source_commit_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )

    assert not status.is_stale
    assert all("WARNING" not in line for line in status.header_lines())


def test_wheel_status_without_commit_is_not_stale() -> None:
    status = WheelStatus(
        path=Path("rlmesh-0.1.0-cp311-abi3.whl"),
        modified_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        source_commit_at=None,
    )

    assert not status.is_stale
    assert all("WARNING" not in line for line in status.header_lines())


def test_inspect_wheel_reports_selected_wheel(tmp_path: Path) -> None:
    wheel = tmp_path / "rlmesh-0.1.0-cp311-abi3-linux_x86_64.whl"
    _touch_wheel(wheel, mtime=1000.0)

    status = inspect_wheel(tmp_path, "3.11", root=tmp_path)

    assert status is not None
    assert status.path == wheel
    assert status.modified_at == datetime.fromtimestamp(1000.0, tz=timezone.utc)


def test_inspect_wheel_returns_none_without_wheel(tmp_path: Path) -> None:
    assert inspect_wheel(tmp_path, "3.11", root=tmp_path) is None

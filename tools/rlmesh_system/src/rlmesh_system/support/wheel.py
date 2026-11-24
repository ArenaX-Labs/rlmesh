"""Wheel selection and staleness checks for the system runner.

``uv pip install --find-links`` silently picks whatever matching wheel sits in the
wheel directory; these helpers surface the selection and warn when it predates
the sources it is built from."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# Source trees whose newest commit a wheel must not predate.
SOURCE_PATHS: tuple[str, ...] = ("python", "crates")


def wheel_tag_prefix(python_version: str) -> str:
    return "cp310" if python_version == "3.10" else "cp311"


def select_wheel(wheel_dir: Path, python_version: str) -> Path | None:
    prefix = wheel_tag_prefix(python_version)
    candidates = [
        wheel for wheel in wheel_dir.glob("rlmesh-*.whl") if prefix in wheel.name
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda wheel: wheel.stat().st_mtime)


def newest_source_commit_time(root: Path) -> datetime | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "log", "-1", "--format=%ct", "--", *SOURCE_PATHS],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    stamp = result.stdout.strip()
    if not stamp:
        return None
    return datetime.fromtimestamp(int(stamp), tz=timezone.utc)


@dataclass(frozen=True)
class WheelStatus:
    path: Path
    modified_at: datetime
    source_commit_at: datetime | None

    @property
    def is_stale(self) -> bool:
        if self.source_commit_at is None:
            return False
        return self.modified_at < self.source_commit_at

    def header_lines(self) -> list[str]:
        lines = [
            f"wheel: {self.path.name}",
            f"wheel built: {self.modified_at.isoformat()}",
        ]
        if self.source_commit_at is not None:
            lines.append(f"newest source commit: {self.source_commit_at.isoformat()}")
        if self.is_stale:
            lines.append(
                "WARNING: selected wheel is OLDER than the newest source commit; "
                "rebuild with build:python:local before trusting these numbers"
            )
        return lines


def inspect_wheel(
    wheel_dir: Path, python_version: str, *, root: Path
) -> WheelStatus | None:
    wheel = select_wheel(wheel_dir, python_version)
    if wheel is None:
        return None
    modified_at = datetime.fromtimestamp(wheel.stat().st_mtime, tz=timezone.utc)
    return WheelStatus(
        path=wheel,
        modified_at=modified_at,
        source_commit_at=newest_source_commit_time(root),
    )

"""Wheel selection and staleness checks for the system runner.

A perf run is only as trustworthy as the wheel it installs. ``uv pip install
--find-links dist/`` silently picks whatever matching wheel sits in ``dist/``,
so a stale wheel left over from a previous build is invisible until the numbers
look wrong. These helpers surface the selected wheel and warn when it predates
the newest commit touching the Python or Rust sources it is built from.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# Source trees whose newest commit a wheel must not predate. ``python/`` covers
# the pure-Python SDK; ``crates/`` and ``python/rlmesh/rust/`` cover the
# compiled extension.
SOURCE_PATHS: tuple[str, ...] = ("python", "crates")


def wheel_tag_prefix(python_version: str) -> str:
    """Return the wheel interpreter-tag prefix uv selects for a Python version."""
    return "cp310" if python_version == "3.10" else "cp311"


def select_wheel(wheel_dir: Path, python_version: str) -> Path | None:
    """Pick the wheel uv would install for ``python_version``.

    Matches uv's behavior closely enough for reporting: filter to wheels whose
    interpreter tag matches the target Python, then take the most recently
    modified one (uv prefers the highest version; for a single local build the
    newest mtime is the same wheel).
    """
    prefix = wheel_tag_prefix(python_version)
    candidates = [
        wheel for wheel in wheel_dir.glob("rlmesh-*.whl") if prefix in wheel.name
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda wheel: wheel.stat().st_mtime)


def newest_source_commit_time(root: Path) -> datetime | None:
    """Return the commit time of the newest commit touching wheel sources.

    Returns ``None`` when git is unavailable or no matching commit is found,
    so the caller degrades to printing the wheel without a staleness verdict
    rather than failing the run.
    """
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
    """Selected wheel plus the staleness verdict for the run header."""

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
    """Resolve the wheel that will be installed and its staleness verdict."""
    wheel = select_wheel(wheel_dir, python_version)
    if wheel is None:
        return None
    modified_at = datetime.fromtimestamp(wheel.stat().st_mtime, tz=timezone.utc)
    return WheelStatus(
        path=wheel,
        modified_at=modified_at,
        source_commit_at=newest_source_commit_time(root),
    )

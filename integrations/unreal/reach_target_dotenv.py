"""Minimal .env read/write for the Reach Target app (stdlib only)."""

from __future__ import annotations

from pathlib import Path


def load_dotenv(path: Path) -> dict[str, str]:
    """Parse KEY=VAL lines; ignore blanks and comments. Does not mutate os.environ."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip("'").strip('"')
        if key:
            values[key] = val
    return values


def apply_dotenv(path: Path) -> dict[str, str]:
    """Load .env into os.environ for keys that are not already set."""
    import os

    values = load_dotenv(path)
    for key, val in values.items():
        os.environ.setdefault(key, val)
    return values


def upsert_dotenv(path: Path, updates: dict[str, str]) -> None:
    """Create or update KEY=VAL entries in a .env file, preserving other lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    if path.is_file():
        lines = path.read_text(encoding="utf-8").splitlines()

    remaining = dict(updates)
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.partition("=")[0].strip()
            if key in remaining:
                out.append(f"{key}={remaining.pop(key)}")
                continue
        out.append(line)

    if out and out[-1].strip():
        out.append("")
    for key, val in remaining.items():
        out.append(f"{key}={val}")

    text = "\n".join(out).rstrip() + "\n"
    path.write_text(text, encoding="utf-8")

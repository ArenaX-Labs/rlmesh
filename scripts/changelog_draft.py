#!/usr/bin/env python3
"""Draft Unreleased changelog bullets from commits since the last release tag.

Appends a `<!-- DRAFT -->`-marked scaffold under `## [Unreleased]` in CHANGELOG.md. It
never writes final entries: rewrite each bullet in your own words and delete every
`<!-- DRAFT -->` marker. `mise run release` refuses to ship while any marker remains.

Usage:
    python scripts/changelog_draft.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHANGELOG = ROOT / "CHANGELOG.md"

# conventional-commit type -> Keep a Changelog section
SECTION = {"feat": "Added", "fix": "Fixed", "perf": "Changed", "refactor": "Changed"}
ORDER = ["Breaking", "Added", "Changed", "Deprecated", "Removed", "Fixed", "Security"]


def last_release_tag() -> str | None:
    r = subprocess.run(
        ["git", "tag", "--sort=-creatordate", "--list", "v*"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    tags = [t for t in r.stdout.split() if t]
    return tags[0] if tags else None


def commit_subjects(since: str | None) -> list[str]:
    rng = [f"{since}..HEAD"] if since else []
    r = subprocess.run(
        ["git", "log", "--no-merges", "--pretty=%s", *rng],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in r.stdout.splitlines() if line.strip()]


def classify(subject: str) -> tuple[str, str] | None:
    m = re.match(r"(\w+)(\([^)]*\))?(!)?:\s*(.+)", subject)
    if not m:
        return None
    typ, _scope, bang, summary = m.groups()
    if bang or "BREAKING" in subject:
        return "Breaking", summary
    section = SECTION.get(typ)
    return (section, summary) if section else None


def main() -> None:
    tag = last_release_tag()
    print(f"drafting from {tag or 'repository start'}..HEAD", file=sys.stderr)

    buckets: dict[str, list[str]] = {}
    for subject in commit_subjects(tag):
        hit = classify(subject)
        if hit:
            section, summary = hit
            buckets.setdefault(section, []).append(summary)

    if not buckets:
        print("no user-facing commits found", file=sys.stderr)
        return

    lines = ["<!-- DRAFT: rewrite each bullet, then delete every DRAFT marker -->"]
    for section in ORDER:
        if section in buckets:
            lines.append(f"\n### {section}\n")
            lines.extend(f"- <!-- DRAFT --> {s}" for s in buckets[section])
    block = "\n".join(lines) + "\n"

    text = CHANGELOG.read_text()
    marker = "## [Unreleased]\n"
    if marker not in text:
        sys.exit("CHANGELOG.md has no '## [Unreleased]' section")
    CHANGELOG.write_text(text.replace(marker, marker + "\n" + block, 1))

    total = sum(len(v) for v in buckets.values())
    print(f"appended {total} draft bullets under [Unreleased]", file=sys.stderr)


if __name__ == "__main__":
    main()

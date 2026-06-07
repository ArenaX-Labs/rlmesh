#!/usr/bin/env python3
"""Validate RLMesh Python wheel ABI tags."""

from __future__ import annotations

import argparse
import sys
import zipfile
from collections import defaultdict
from email.parser import Parser
from pathlib import Path

EXPECTED_REQUIRES_PYTHON = ">=3.10"
EXPECTED_TAGS = {
    ("cp310", "cp310"),
    ("cp311", "abi3"),
}
LOCAL_ONLY_PLATFORM_PREFIXES = ("linux_",)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel_dir", type=Path)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="validate wheel tags and metadata without requiring both wheel families",
    )
    parser.add_argument(
        "--allow-local-platform-tags",
        action="store_true",
        help=(
            "allow local-only platform tags such as linux_x86_64; release "
            "validation should leave this disabled"
        ),
    )
    args = parser.parse_args(argv)

    wheels = sorted(args.wheel_dir.glob("rlmesh-*.whl"))
    if not wheels:
        print(f"no rlmesh wheels found in {args.wheel_dir}", file=sys.stderr)
        return 1

    errors: list[str] = []
    platforms: dict[str, set[tuple[str, str]]] = defaultdict(set)

    for wheel in wheels:
        try:
            python_tag, abi_tag, platform_tag = wheel_tags(wheel)
        except ValueError as exc:
            errors.append(f"{wheel.name}: {exc}")
            continue

        tag = (python_tag, abi_tag)
        if tag == ("cp310", "abi3"):
            errors.append(
                f"{wheel.name}: cp310-abi3 is not supported by RLMesh Tensor buffers"
            )
        elif tag not in EXPECTED_TAGS:
            errors.append(
                f"{wheel.name}: expected cp310-cp310 or cp311-abi3, "
                f"got {python_tag}-{abi_tag}"
            )

        if not args.allow_local_platform_tags and is_local_only_platform(platform_tag):
            errors.append(
                f"{wheel.name}: platform tag {platform_tag} is local-only and "
                "cannot be uploaded to PyPI; build release wheels with an "
                "uploadable platform tag such as manylinux, musllinux, macosx, or win"
            )

        requires_python = wheel_requires_python(wheel)
        if requires_python != EXPECTED_REQUIRES_PYTHON:
            errors.append(
                f"{wheel.name}: expected Requires-Python "
                f"{EXPECTED_REQUIRES_PYTHON}, got {requires_python or '<missing>'}"
            )

        platforms[platform_tag].add(tag)

    if not args.allow_partial:
        for platform_tag, tags in sorted(platforms.items()):
            missing = EXPECTED_TAGS - tags
            if missing:
                missing_text = ", ".join(format_tag(tag) for tag in sorted(missing))
                errors.append(f"{platform_tag}: missing {missing_text}")

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    for platform_tag, tags in sorted(platforms.items()):
        tag_text = ", ".join(format_tag(tag) for tag in sorted(tags))
        print(f"{platform_tag}: {tag_text}")
    return 0


def wheel_tags(wheel: Path) -> tuple[str, str, str]:
    stem = wheel.name.removesuffix(".whl")
    parts = stem.split("-")
    if len(parts) < 5:
        raise ValueError("invalid wheel filename")
    return parts[-3], parts[-2], parts[-1]


def wheel_requires_python(wheel: Path) -> str | None:
    with zipfile.ZipFile(wheel) as archive:
        metadata_name = next(
            (
                name
                for name in archive.namelist()
                if name.endswith(".dist-info/METADATA")
            ),
            None,
        )
        if metadata_name is None:
            return None
        metadata = archive.read(metadata_name).decode("utf-8")
    return Parser().parsestr(metadata).get("Requires-Python")


def is_local_only_platform(platform_tag: str) -> bool:
    return any(
        tag.startswith(LOCAL_ONLY_PLATFORM_PREFIXES)
        for tag in platform_tag.split(".")
    )


def format_tag(tag: tuple[str, str]) -> str:
    return f"{tag[0]}-{tag[1]}"


if __name__ == "__main__":
    raise SystemExit(main())

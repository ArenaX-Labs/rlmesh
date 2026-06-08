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
PLATFORM_SETS = {
    "all": (
        "macosx_10_12_x86_64",
        "macosx_11_0_arm64",
        "manylinux_2_17_aarch64.manylinux2014_aarch64",
        "manylinux_2_17_x86_64.manylinux2014_x86_64",
        "musllinux_1_2_aarch64",
        "musllinux_1_2_x86_64",
        "win_amd64",
    ),
    "linux-windows": (
        "manylinux_2_17_aarch64.manylinux2014_aarch64",
        "manylinux_2_17_x86_64.manylinux2014_x86_64",
        "musllinux_1_2_aarch64",
        "musllinux_1_2_x86_64",
        "win_amd64",
    ),
    "macos": (
        "macosx_10_12_x86_64",
        "macosx_11_0_arm64",
    ),
}
LOCAL_ONLY_PLATFORM_PREFIXES = ("linux_",)
REQUIRED_WHEEL_FILES = (
    "rlmesh/py.typed",
    "rlmesh/_rlmesh.pyi",
)
FORBIDDEN_WHEEL_PREFIXES = (
    ".pytest_cache/",
    "dist/",
    "rust/",
    "tests/",
)
FORBIDDEN_WHEEL_SEGMENTS = (
    "__pycache__",
    "rust",
    "tests",
)
NATIVE_EXTENSION_PREFIX = "rlmesh/_rlmesh."
NATIVE_EXTENSION_SUFFIXES = (".pyd", ".so")


def main(argv: list[str] | None = None) -> int:
    """Run wheel validation from command-line arguments."""
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
    parser.add_argument(
        "--platform-set",
        action="append",
        choices=sorted(PLATFORM_SETS),
        default=[],
        help="require all platform tags for a named release platform set",
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

        errors.extend(
            f"{wheel.name}: {error}" for error in wheel_payload_errors(wheel)
        )

        platforms[platform_tag].add(tag)

    if not args.allow_partial:
        for platform_tag, tags in sorted(platforms.items()):
            missing = EXPECTED_TAGS - tags
            if missing:
                missing_text = ", ".join(format_tag(tag) for tag in sorted(missing))
                errors.append(f"{platform_tag}: missing {missing_text}")

    required_platforms = set()
    for platform_set in args.platform_set:
        required_platforms.update(PLATFORM_SETS[platform_set])
    for platform_tag in sorted(required_platforms - set(platforms)):
        errors.append(f"{platform_tag}: missing platform")

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    for platform_tag, tags in sorted(platforms.items()):
        tag_text = ", ".join(format_tag(tag) for tag in sorted(tags))
        print(f"{platform_tag}: {tag_text}")
    return 0


def wheel_tags(wheel: Path) -> tuple[str, str, str]:
    """Return the Python, ABI, and platform tags from a wheel filename."""
    stem = wheel.name.removesuffix(".whl")
    parts = stem.split("-")
    if len(parts) < 5:
        raise ValueError("invalid wheel filename")
    return parts[-3], parts[-2], parts[-1]


def wheel_requires_python(wheel: Path) -> str | None:
    """Return the wheel metadata Requires-Python value."""
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


def wheel_payload_errors(wheel: Path) -> list[str]:
    """Return payload validation errors for a wheel archive."""
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()

    errors: list[str] = []
    name_set = set(names)
    for required in REQUIRED_WHEEL_FILES:
        if required not in name_set:
            errors.append(f"missing required wheel file {required}")

    native_extensions = [
        name
        for name in names
        if name.startswith(NATIVE_EXTENSION_PREFIX)
        and name.endswith(NATIVE_EXTENSION_SUFFIXES)
    ]
    if len(native_extensions) != 1:
        joined = ", ".join(native_extensions) if native_extensions else "<none>"
        errors.append(f"expected exactly one native extension, found {joined}")

    for name in names:
        if name.endswith(".pyc"):
            errors.append(f"forbidden bytecode file {name}")
        if any(name.startswith(prefix) for prefix in FORBIDDEN_WHEEL_PREFIXES):
            errors.append(f"forbidden wheel payload path {name}")
            continue
        segments = tuple(segment for segment in name.split("/") if segment)
        forbidden = sorted(set(segments) & set(FORBIDDEN_WHEEL_SEGMENTS))
        if forbidden:
            errors.append(
                f"forbidden wheel payload path {name} "
                f"(contains {', '.join(forbidden)})"
            )
    return errors


def is_local_only_platform(platform_tag: str) -> bool:
    """Return true when a wheel platform tag is only valid for local installs."""
    return any(
        tag.startswith(LOCAL_ONLY_PLATFORM_PREFIXES)
        for tag in platform_tag.split(".")
    )


def format_tag(tag: tuple[str, str]) -> str:
    """Format a Python and ABI tag pair for diagnostics."""
    return f"{tag[0]}-{tag[1]}"


if __name__ == "__main__":
    raise SystemExit(main())

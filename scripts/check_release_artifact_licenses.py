#!/usr/bin/env python3
"""Validate release artifact license payloads before publishing."""

from __future__ import annotations

import argparse
import stat
import sys
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import tomllib

LICENSE_FILES = ("LICENSE-APACHE", "LICENSE-MIT")
PYTHON_NOTICE_FILES = ("THIRD_PARTY_NOTICES.md",)


@dataclass(frozen=True)
class Artifact:
    """A publishable artifact declared by rlmesh.toml."""

    ecosystem: str
    name: str
    version: str


def main(argv: list[str] | None = None) -> int:
    """Run release artifact license validation."""
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=repo_root / "rlmesh.toml",
        help="release policy manifest to read",
    )
    parser.add_argument(
        "--crate-dir",
        type=Path,
        default=repo_root / "target/package",
        help="directory containing generated .crate archives",
    )
    parser.add_argument(
        "--wheel-dir",
        type=Path,
        default=repo_root / "python/rlmesh/dist",
        help="directory containing generated Python wheels",
    )
    args = parser.parse_args(argv)

    errors: list[str] = []
    license_bytes = {name: (repo_root / name).read_bytes() for name in LICENSE_FILES}
    artifacts = publishable_artifacts(args.manifest)

    cargo_artifacts = [
        artifact for artifact in artifacts if artifact.ecosystem == "cargo"
    ]
    pypi_artifacts = [
        artifact for artifact in artifacts if artifact.ecosystem == "pypi"
    ]

    for artifact in cargo_artifacts:
        archive = args.crate_dir / f"{artifact.name}-{artifact.version}.crate"
        errors.extend(validate_crate_archive(archive, artifact, license_bytes))

    if len(pypi_artifacts) != 1:
        errors.append(
            f"expected exactly one publishable pypi artifact, got {len(pypi_artifacts)}"
        )
    else:
        errors.extend(
            validate_python_wheels(
                args.wheel_dir,
                pypi_artifacts[0],
                repo_root / "python/rlmesh",
                license_bytes,
            )
        )

    if errors:
        for error in errors:
            print(f"release artifact license error: {error}", file=sys.stderr)
        return 1

    crate_count = len(cargo_artifacts)
    wheel_count = (
        len(list(args.wheel_dir.glob(f"{pypi_artifacts[0].name}-*.whl")))
        if pypi_artifacts
        else 0
    )
    print(
        f"validated license payloads for {crate_count} crates and {wheel_count} wheels"
    )
    return 0


def publishable_artifacts(manifest_path: Path) -> list[Artifact]:
    """Return publishable artifacts from the release policy manifest."""
    manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = []
    for raw in manifest.get("artifact", []):
        if not raw.get("publish", False):
            continue
        artifacts.append(
            Artifact(
                ecosystem=required_str(raw, "ecosystem"),
                name=required_str(raw, "name"),
                version=required_str(raw, "version"),
            )
        )
    return artifacts


def validate_crate_archive(
    archive: Path,
    artifact: Artifact,
    license_bytes: dict[str, bytes],
) -> list[str]:
    """Return validation errors for a Cargo .crate archive."""
    if not archive.exists():
        return [f"{archive}: missing crate archive"]

    errors: list[str] = []
    prefix = f"{artifact.name}-{artifact.version}"
    try:
        with tarfile.open(archive, mode="r:gz") as package:
            for license_name, expected in license_bytes.items():
                member_name = f"{prefix}/{license_name}"
                try:
                    member = package.getmember(member_name)
                except KeyError:
                    errors.append(f"{archive.name}: missing {member_name}")
                    continue

                if not member.isfile():
                    errors.append(
                        f"{archive.name}: {member_name} is not a regular file"
                    )
                    continue

                extracted = package.extractfile(member)
                if extracted is None:
                    errors.append(f"{archive.name}: could not read {member_name}")
                    continue
                if extracted.read() != expected:
                    errors.append(
                        f"{archive.name}: {member_name} does not match repo root"
                    )
    except tarfile.TarError as exc:
        return [f"{archive}: invalid crate archive: {exc}"]

    return errors


def validate_python_wheels(
    wheel_dir: Path,
    artifact: Artifact,
    package_root: Path,
    license_bytes: dict[str, bytes],
) -> list[str]:
    """Return validation errors for Python wheel license payloads."""
    wheels = sorted(wheel_dir.glob(f"{artifact.name}-{artifact.version}-*.whl"))
    if not wheels:
        return [f"{wheel_dir}: no wheels found for {artifact.name} {artifact.version}"]

    package_notice_files = {
        name: (package_root / name).read_bytes() for name in PYTHON_NOTICE_FILES
    }
    third_party_files = {
        file.relative_to(package_root).as_posix(): file.read_bytes()
        for file in sorted((package_root / "third_party_licenses").rglob("*"))
        if file.is_file()
    }
    expected_files = {
        **license_bytes,
        **package_notice_files,
        **third_party_files,
    }

    errors: list[str] = []
    for wheel in wheels:
        errors.extend(validate_wheel_archive(wheel, expected_files))
    return errors


def validate_wheel_archive(wheel: Path, expected_files: dict[str, bytes]) -> list[str]:
    """Return validation errors for one wheel archive."""
    errors: list[str] = []
    try:
        with zipfile.ZipFile(wheel) as archive:
            infos = {info.filename: info for info in archive.infolist()}
            names = set(infos)
            dist_info_dirs = sorted(
                {
                    name.split("/", 1)[0]
                    for name in names
                    if name.endswith(".dist-info/METADATA")
                }
            )
            if len(dist_info_dirs) != 1:
                return [
                    f"{wheel.name}: expected exactly one .dist-info directory, got {len(dist_info_dirs)}"
                ]

            licenses_prefix = f"{dist_info_dirs[0]}/licenses"
            for relative_name, expected in expected_files.items():
                archive_name = f"{licenses_prefix}/{relative_name}"
                info = infos.get(archive_name)
                if info is None:
                    errors.append(f"{wheel.name}: missing {archive_name}")
                    continue
                if not zip_member_is_regular_file(info):
                    errors.append(f"{wheel.name}: {archive_name} is not a regular file")
                    continue
                if archive.read(archive_name) != expected:
                    errors.append(
                        f"{wheel.name}: {archive_name} does not match source file"
                    )
    except zipfile.BadZipFile as exc:
        return [f"{wheel}: invalid wheel archive: {exc}"]

    return errors


def zip_member_is_regular_file(info: zipfile.ZipInfo) -> bool:
    """Return true when a zip member is a regular file or has no POSIX mode."""
    mode = (info.external_attr >> 16) & 0o170000
    return mode == 0 or stat.S_IFMT(mode) == stat.S_IFREG


def required_str(data: dict[str, object], key: str) -> str:
    """Return a required string value from a TOML table."""
    value = data.get(key)
    if not isinstance(value, str):
        raise ValueError(f"expected {key} to be a string")
    return value


if __name__ == "__main__":
    raise SystemExit(main())

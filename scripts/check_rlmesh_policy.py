#!/usr/bin/env python3
"""Validate the canonical RLMesh project policy manifest."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomllib

VERSION_RE = re.compile(
    r"^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)"
    r"(?:(?:[-._]?(?:alpha|beta|rc|a|b)\.?\d+))?"
    r"(?:\+[A-Za-z0-9.-]+)?$",
    re.IGNORECASE,
)
STR_CONST_RE = re.compile(r'pub const (?P<name>[A-Z0-9_]+): &str = "(?P<value>[^"]+)";')
STR_SLICE_CONST_RE = re.compile(
    r"pub const (?P<name>[A-Z0-9_]+): &\[&str\]\s*=\s*&\[(?P<values>[^\]]*)\];",
    re.DOTALL,
)
PY_STR_CONST_RE = re.compile(r'(?P<name>[A-Z0-9_]+)\s*=\s*"(?P<value>[^"]+)"')
INT_CONST_RE = re.compile(r"(?P<name>[A-Z0-9_]+)\s*=\s*(?P<value>\d+)")


@dataclass(frozen=True)
class Artifact:
    id: str
    name: str
    ecosystem: str
    manifest: Path
    version: str
    role: str
    publish: bool

    @classmethod
    def from_manifest(cls, repo_root: Path, raw: dict[str, Any]) -> "Artifact":
        return cls(
            id=_required_str(raw, "id"),
            name=_required_str(raw, "name"),
            ecosystem=_required_str(raw, "ecosystem"),
            manifest=repo_root / _required_str(raw, "manifest"),
            version=_required_str(raw, "version"),
            role=_required_str(raw, "role"),
            publish=bool(raw.get("publish", False)),
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="rlmesh.toml",
        help="RLMesh project policy manifest to validate",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = (repo_root / args.manifest).resolve()
    errors = validate_rlmesh_policy(repo_root=repo_root, manifest_path=manifest_path)
    if errors:
        for error in errors:
            print(f"rlmesh policy error: {error}", file=sys.stderr)
        return 1
    return 0


def validate_rlmesh_policy(*, repo_root: Path, manifest_path: Path) -> list[str]:
    errors: list[str] = []
    manifest = _read_toml(manifest_path)

    project = _required_table(manifest, "project", errors)
    release = _required_table(manifest, "release", errors)
    workflow = _required_table(manifest, "workflow", errors)
    protocol = _required_table(manifest, "protocol", errors)
    api_surface = _required_table(manifest, "api_surface", errors)
    raw_artifacts = manifest.get("artifact")
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        errors.append("manifest must define at least one [[artifact]] entry")
        raw_artifacts = []

    if project.get("name") != "rlmesh":
        errors.append("[project].name must be 'rlmesh'")

    package_family = release.get("package_family")
    if not isinstance(package_family, str):
        errors.append("[release].package_family must be a string")
        package_family = ""

    artifacts: list[Artifact] = []
    for raw in raw_artifacts:
        if not isinstance(raw, dict):
            errors.append("[[artifact]] entries must be TOML tables")
            continue
        try:
            artifacts.append(Artifact.from_manifest(repo_root, raw))
        except ValueError as exc:
            errors.append(str(exc))

    artifact_ids = {artifact.id for artifact in artifacts}
    workspace_version = _workspace_version(repo_root / "Cargo.toml", errors)
    workspace_dependencies = _workspace_dependencies(repo_root / "Cargo.toml", errors)

    for artifact in artifacts:
        errors.extend(_validate_artifact(artifact, package_family, workspace_version))
        if artifact.ecosystem == "cargo" and artifact.name in workspace_dependencies:
            expected = f"={artifact.version}"
            actual = workspace_dependencies[artifact.name]
            if actual != expected:
                errors.append(
                    f"workspace dependency {artifact.name} is {actual}, expected {expected}"
                )

    errors.extend(_validate_protocol_and_workflow(repo_root, protocol, workflow))
    errors.extend(
        _validate_api_surface(
            repo_root, api_surface, artifact_ids, release, package_family
        )
    )

    return errors


def _validate_artifact(
    artifact: Artifact, package_family: str, workspace_version: str | None
) -> list[str]:
    errors: list[str] = []
    if not artifact.manifest.exists():
        return [f"{artifact.id}: {artifact.manifest} does not exist"]

    try:
        actual_name, actual_version = _artifact_version(artifact, workspace_version)
    except ValueError as exc:
        return [f"{artifact.id}: {exc}"]

    if actual_name != artifact.name:
        errors.append(
            f"{artifact.manifest}: package name is {actual_name}, manifest declares {artifact.name}"
        )
    if actual_version != artifact.version:
        errors.append(
            f"{artifact.manifest}: version is {actual_version}, manifest declares {artifact.version}"
        )

    try:
        family = _package_family_for_version(artifact.version)
    except ValueError as exc:
        errors.append(f"{artifact.id}: {exc}")
        return errors

    if family != package_family:
        errors.append(
            f"{artifact.id} {artifact.version} belongs to package family {family}, "
            f"manifest family is {package_family}"
        )

    if artifact.role == "core" and not artifact.publish:
        errors.append(f"{artifact.id}: core artifact must be publish=true")

    return errors


def _artifact_version(
    artifact: Artifact, workspace_version: str | None
) -> tuple[str, str]:
    data = _read_toml(artifact.manifest)
    project = data.get("project")
    package = data.get("package")

    if artifact.ecosystem == "pypi":
        if not isinstance(project, dict):
            raise ValueError("missing [project] table")
        return _required_str(project, "name"), _required_str(project, "version")

    if artifact.ecosystem == "cargo":
        if not isinstance(package, dict):
            raise ValueError("missing [package] table")
        name = _required_str(package, "name")
        raw_version = package.get("version")
        if isinstance(raw_version, str):
            return name, raw_version
        if isinstance(raw_version, dict) and raw_version.get("workspace") is True:
            if workspace_version is None:
                raise ValueError(
                    "uses workspace version but workspace version was not found"
                )
            return name, workspace_version
        raise ValueError("missing package version")

    raise ValueError(f"unsupported ecosystem {artifact.ecosystem}")


def _validate_protocol_and_workflow(
    repo_root: Path, protocol: dict[str, Any], workflow: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    source = repo_root / "crates/rlmesh-proto/src/lib.rs"
    source_text = source.read_text(encoding="utf-8")
    constants = dict(STR_CONST_RE.findall(source_text))
    expected = {
        "PROTOCOL_GENERATION": (
            "[protocol].current_generation",
            protocol.get("current_generation"),
        ),
        "MIN_SUPPORTED_PROTOCOL_GENERATION": (
            "[protocol].minimum_generation",
            protocol.get("minimum_generation"),
        ),
        "CURRENT_WORKFLOW_EDITION": (
            "[workflow].current_edition",
            workflow.get("current_edition"),
        ),
    }
    for name, (manifest_key, value) in expected.items():
        if not isinstance(value, str):
            errors.append(f"{manifest_key} must be a string")
            continue
        if constants.get(name) != value:
            errors.append(
                f"{name} is {constants.get(name)!r}, manifest declares {value!r}"
            )

    supported = workflow.get("supported_editions")
    if not isinstance(supported, list) or not all(
        isinstance(item, str) for item in supported
    ):
        errors.append("[workflow].supported_editions must be a list of strings")
    elif constants.get("CURRENT_WORKFLOW_EDITION") not in supported:
        errors.append(
            "[workflow].supported_editions must include [workflow].current_edition"
        )
    else:
        supported_constant = _rust_str_slice_const(
            source_text, "SUPPORTED_WORKFLOW_EDITIONS", constants
        )
        if supported_constant is None:
            errors.append(f"{source}: missing SUPPORTED_WORKFLOW_EDITIONS string slice")
        elif supported_constant != supported:
            errors.append(
                "SUPPORTED_WORKFLOW_EDITIONS is "
                f"{supported_constant!r}, manifest declares {supported!r}"
            )

    for token in _forbidden_unpublished_protocol_tokens():
        if token in source_text:
            errors.append(
                f"{source}: remove unpublished legacy protocol token {token!r}"
            )

    baseline = protocol.get("public_baseline")
    if not isinstance(baseline, str):
        errors.append("[protocol].public_baseline must be a string")
    else:
        baseline_path = repo_root / baseline
        if not baseline_path.exists():
            errors.append(f"[protocol].public_baseline does not exist: {baseline}")

    service_generations = protocol.get("service_generations")
    if not isinstance(service_generations, dict):
        errors.append("[protocol].service_generations must be an inline table")
    else:
        for key in ("env", "model", "spaces"):
            if not isinstance(service_generations.get(key), str):
                errors.append(f"[protocol].service_generations.{key} must be a string")

    buf_config = repo_root / "buf.yaml"
    if not buf_config.exists():
        errors.append("buf.yaml is required for proto lint and breaking-change policy")
    elif "breaking:" not in buf_config.read_text(encoding="utf-8"):
        errors.append("buf.yaml must include a breaking-change policy")

    return errors


def _rust_str_slice_const(
    source_text: str, name: str, constants: dict[str, str]
) -> list[str] | None:
    for match in STR_SLICE_CONST_RE.finditer(source_text):
        if match.group("name") != name:
            continue
        values: list[str] = []
        for raw_item in match.group("values").split(","):
            item = raw_item.strip()
            if not item:
                continue
            if item in constants:
                values.append(constants[item])
            elif item.startswith('"') and item.endswith('"'):
                values.append(item[1:-1])
            else:
                return None
        return values
    return None


def _validate_api_surface(
    repo_root: Path,
    api_surface: dict[str, Any],
    artifact_ids: set[str],
    release: dict[str, Any],
    package_family: str,
) -> list[str]:
    errors: list[str] = []
    python = api_surface.get("python")
    if not isinstance(python, dict):
        return ["missing [api_surface.python] table"]

    tool_source = (
        repo_root / "tools/rlmesh_api_surface/src/rlmesh_api_surface/api_surface.py"
    )
    tool_text = tool_source.read_text(encoding="utf-8")
    string_constants = dict(PY_STR_CONST_RE.findall(tool_text))
    int_constants = {
        name: int(value) for name, value in INT_CONST_RE.findall(tool_text)
    }
    expected_strings = {
        "DOCS_API_SURFACE_KIND": ("[api_surface.python].kind", python.get("kind")),
        "DOCS_API_SURFACE_MANIFEST_KIND": (
            "[api_surface.python].manifest_kind",
            python.get("manifest_kind"),
        ),
    }
    for name, (manifest_key, value) in expected_strings.items():
        if not isinstance(value, str):
            errors.append(f"{manifest_key} must be a string")
            continue
        if string_constants.get(name) != value:
            errors.append(
                f"{name} is {string_constants.get(name)!r}, manifest declares {value!r}"
            )

    schema_version = python.get("schema_version")
    if not isinstance(schema_version, int):
        errors.append("[api_surface.python].schema_version must be an integer")
    elif int_constants.get("DOCS_API_SURFACE_SCHEMA_VERSION") != schema_version:
        errors.append(
            "DOCS_API_SURFACE_SCHEMA_VERSION is "
            f"{int_constants.get('DOCS_API_SURFACE_SCHEMA_VERSION')!r}, "
            f"manifest declares {schema_version!r}"
        )

    if api_surface.get("stability_policy") != "stable-labels":
        errors.append("[api_surface].stability_policy must be 'stable-labels'")

    if release.get("status") not in {"alpha", "beta", "stable"}:
        errors.append("[release].status must be alpha, beta, or stable")

    for key in ("metadata", "contract_snapshot"):
        value = python.get(key)
        if not isinstance(value, str):
            errors.append(f"[api_surface.python].{key} must be a string")
            continue
        if not (repo_root / value).exists():
            errors.append(f"[api_surface.python].{key} does not exist: {value}")

    package_artifact = python.get("package_artifact")
    if not isinstance(package_artifact, str):
        errors.append("[api_surface.python].package_artifact must be a string")
    elif package_artifact not in artifact_ids:
        errors.append(
            f"[api_surface.python].package_artifact {package_artifact!r} "
            "does not match any [[artifact]].id"
        )

    metadata_path = python.get("metadata")
    if isinstance(metadata_path, str) and (repo_root / metadata_path).exists():
        metadata = _read_json(repo_root / metadata_path)
        if "release_maturity" in metadata:
            errors.append(f"{metadata_path}: release_maturity belongs in rlmesh.toml")
        if (
            metadata.get("package", {}).get("name")
            and metadata["package"]["name"] != "rlmesh"
        ):
            errors.append(f"{metadata_path}: package.name must be rlmesh")

    if package_family and not package_family.startswith("0."):
        errors.append("[release].package_family must be 0.minor before 1.0")

    return errors


def _workspace_version(path: Path, errors: list[str]) -> str | None:
    data = _read_toml(path)
    workspace = data.get("workspace")
    if not isinstance(workspace, dict):
        errors.append("root Cargo.toml is missing [workspace]")
        return None
    package = workspace.get("package")
    if not isinstance(package, dict):
        errors.append("root Cargo.toml is missing [workspace.package]")
        return None
    version = package.get("version")
    if not isinstance(version, str):
        errors.append("root Cargo.toml is missing [workspace.package].version")
        return None
    return version


def _workspace_dependencies(path: Path, errors: list[str]) -> dict[str, str]:
    data = _read_toml(path)
    workspace = data.get("workspace")
    if not isinstance(workspace, dict):
        return {}
    raw_dependencies = workspace.get("dependencies")
    if not isinstance(raw_dependencies, dict):
        errors.append("root Cargo.toml is missing [workspace.dependencies]")
        return {}

    dependencies: dict[str, str] = {}
    for name, raw in raw_dependencies.items():
        if (
            isinstance(raw, dict)
            and "path" in raw
            and isinstance(raw.get("version"), str)
        ):
            dependencies[name] = raw["version"]
    return dependencies


def _package_family_for_version(version: str) -> str:
    match = VERSION_RE.match(version)
    if match is None:
        raise ValueError(f"unsupported version spelling: {version}")
    major = int(match.group("major"))
    minor = int(match.group("minor"))
    if major == 0:
        return f"0.{minor}"
    return str(major)


def _forbidden_unpublished_protocol_tokens() -> list[str]:
    return [
        "_".join(["LEGACY", "0", "1", "A" + "BI", "VERSION"]),
        "".join(["A", "BI", "_VERSION"]),
        "_".join(["MIN", "SUPPORTED", "A" + "BI", "VERSION"]),
        "_".join(["is", "a" + "bi", "compatible"]),
        ".".join(["rlmesh", "v1"]),
    ]


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _required_table(
    data: dict[str, Any], key: str, errors: list[str]
) -> dict[str, Any]:
    table = data.get(key)
    if not isinstance(table, dict):
        errors.append(f"missing [{key}] table")
        return {}
    return table


def _required_str(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"missing required string field {key}")
    return value


if __name__ == "__main__":
    raise SystemExit(main())

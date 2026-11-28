#!/usr/bin/env python3
"""Validate the canonical RLMesh project policy manifest."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
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
RUST_SEMVER_RE = re.compile(
    r"^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)"
    r"(?:-(?P<channel>alpha|beta|rc)\.(?P<number>\d+))?$"
)
STABLE_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")
STR_CONST_RE = re.compile(
    r'pub const (?P<name>[A-Z0-9_]+): &str =\s*"(?P<value>[^"]+)";'
)
PY_STR_CONST_RE = re.compile(r'(?P<name>[A-Z0-9_]+)\s*=\s*"(?P<value>[^"]+)"')
INT_CONST_RE = re.compile(r"(?P<name>[A-Z0-9_]+)\s*=\s*(?P<value>\d+)")
EDITION_BASE_RE = re.compile(r"^[0-9]{4}\.[0-9]{2}$")
PRERELEASE_LITERAL_RE = re.compile(
    r"\d+\.\d+\.\d+(?:-(?:alpha|beta|rc)\.\d+|(?:a|b|rc)\d+)"
)


def _canonical_spec_sha256(spec_path: Path) -> str:
    r"""SHA-256 of a sealed spec document over its canonical bytes.

    Canonicalization: decode UTF-8 rejecting a BOM, normalize CRLF/CR to LF,
    ensure exactly one trailing ``\\n``, then sha256. It is
    whitespace/encoding-only — never semantic Markdown normalization, which
    could erase a real spec difference. ``.gitattributes``
    (`docs/editions/*.md text eol=lf`) keeps the on-disk bytes stable across
    checkouts.
    """
    raw = spec_path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError(f"{spec_path} has a UTF-8 BOM; edition specs must be BOM-free")
    text = raw.decode("utf-8")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.rstrip("\n") + "\n"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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
    def from_manifest(cls, repo_root: Path, raw: dict[str, Any]) -> Artifact:
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
    parser.add_argument(
        "--selfcheck",
        action="store_true",
        help="run this checker's own unit tests instead of validating the manifest",
    )
    args = parser.parse_args(argv)

    if args.selfcheck:
        selfcheck()
        return 0

    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = (repo_root / args.manifest).resolve()
    errors = validate_rlmesh_policy(repo_root=repo_root, manifest_path=manifest_path)
    if errors:
        for error in errors:
            print(f"rlmesh policy error: {error}", file=sys.stderr)
        return 1
    return 0


def selfcheck() -> None:
    """Unit tests for the edition rules."""
    lib = 'pub const CURRENT_WORKFLOW_EDITION: &str = "2026.10";\n' + (
        'Edition::E2026_06 => "2026.06",\nEdition::E2026_10 => "2026.10",\n'
    )
    assert _edition_arms(lib) == ["2026.06", "2026.10"]
    assert not _check_edition_source(lib, "2026.10", ["2026.06", "2026.10"])
    assert _check_edition_source(lib, "2026.06", ["2026.06", "2026.10"])
    assert _check_edition_source(lib, "2026.10", ["2026.10"])
    assert _check_edition_source(lib, "2026.10", ["2026.06", "2026.10", "2027.04"])

    previous = """[workflow]
current_edition = "2026.10"

[workflow.editions."2026.06"]
status = "sealed"
sealed_in = "0.1.0"

[workflow.editions."2026.10"]
status = "provisional"
"""
    assert _sealed_editions(previous) == {"2026.06"}
    sealed = tomllib.loads(previous)["workflow"]["editions"]
    reason = {"2026.06": "0.2 is a major break"}
    assert not _check_dropped_editions(reason, sealed)
    assert _check_dropped_editions({"2026.06": " "}, sealed)
    assert _check_dropped_editions({"2026.07": "typo"}, sealed)
    assert _known_editions(sealed, reason) == ["2026.10"]
    assert _known_editions(sealed, {}) == ["2026.06", "2026.10"]

    assert not _changed_sealed_records(previous, sealed)
    for field, value in {
        "status": "provisional",
        "spec": "replacement.md",
        "sealed_in": "0.2.0",
        "spec_sha256": "rewritten checksum",
    }.items():
        changed = {name: dict(entry) for name, entry in sealed.items()}
        changed["2026.06"][field] = value
        assert _changed_sealed_records(previous, changed) == ["2026.06"]
    assert _changed_sealed_records(previous, {}) == ["2026.06"]
    assert not _changed_sealed_records("", {})

    assert not _check_rust_version("1.99", "1.99")
    assert not _check_rust_version({"version": "1.99.0"}, "1.99")
    assert _check_rust_version("1.100", "1.99")
    assert _check_rust_version("stable", "1.99")
    assert _check_rust_version(None, "1.99")
    assert _check_rust_version("1.99", None)
    assert not _check_capi_version(
        '#define RLMESH_VERSION "0.2.0-rc.1"\n', "0.2.0-rc.1"
    )
    assert _check_capi_version('#define RLMESH_VERSION "0.1.0"\n', "0.2.0")
    assert _check_capi_version("/* no macro */\n", "0.2.0")

    print("policy self-check passed")


def validate_rlmesh_policy(*, repo_root: Path, manifest_path: Path) -> list[str]:
    errors: list[str] = []
    manifest = _read_toml(manifest_path)

    release = _required_table(manifest, "release", errors)
    workflow = _required_table(manifest, "workflow", errors)
    api_surface = _required_table(manifest, "api_surface", errors)
    raw_artifacts = manifest.get("artifact")
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        errors.append("manifest must define at least one [[artifact]] entry")
        raw_artifacts = []

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

    errors.extend(_validate_release_crate_order(repo_root, artifacts))
    errors.extend(_validate_buf_policy(repo_root))
    errors.extend(
        _validate_workflow_editions(repo_root, workflow, release, workspace_version)
    )
    errors.extend(
        _validate_sealed_edition_retention(repo_root, manifest_path, workflow)
    )
    errors.extend(_validate_adapters(repo_root))
    errors.extend(_validate_python_public_modules(repo_root))
    errors.extend(
        _validate_api_surface(
            repo_root, api_surface, artifact_ids, release, package_family
        )
    )
    errors.extend(_validate_doc_versions(repo_root, workspace_version))
    errors.extend(_validate_rust_version(repo_root))
    errors.extend(_validate_capi_version(repo_root, workspace_version))
    errors.extend(_validate_binding_versions(repo_root, workspace_version))

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


def _validate_adapters(repo_root: Path) -> list[str]:
    errors: list[str] = []
    keys = repo_root / "crates/rlmesh-adapters/src/keys.rs"
    if not keys.exists():
        return [f"adapter metadata keys file missing: {keys}"]
    text = keys.read_text(encoding="utf-8")
    for const in ("ENV_METADATA_KEY", "MODEL_METADATA_KEY"):
        match = re.search(rf'{const}: &str = "(?P<value>[^"]+)"', text)
        if match is None:
            errors.append(f"{keys}: missing {const}")
        elif not match.group("value").startswith("rlmesh.adapters.v1."):
            errors.append(
                f"{keys}: {const} is {match.group('value')!r}; the adapter spec format is v1"
            )
    vectors = repo_root / "crates/rlmesh-adapters/conformance/v1"
    if not vectors.is_dir():
        errors.append(f"adapter conformance vectors missing: {vectors}")
    return errors


def _validate_python_public_modules(repo_root: Path) -> list[str]:
    """Every shipped public (non-``_``) rlmesh module must define ``__all__``.

    Without ``__all__`` a module implicitly exports every non-underscore top-level
    name, so internal classes leak into IDE autocomplete/auto-import. The ``_``
    prefix hides private *modules*; ``__all__`` curates what the public ones
    expose.
    """
    errors: list[str] = []
    pkg_root = repo_root / "python/rlmesh/src/rlmesh"
    if not pkg_root.is_dir():
        return [f"python package root missing: {pkg_root}"]
    for entry in sorted(pkg_root.iterdir()):
        name = entry.name
        if name.startswith("_"):
            continue
        if entry.is_file() and name.endswith(".py"):
            init = entry
        elif entry.is_dir() and (entry / "__init__.py").is_file():
            init = entry / "__init__.py"
        else:
            continue
        if not _defines_dunder_all(init):
            errors.append(
                f"{init.relative_to(repo_root)}: public module must define __all__ "
                "(curate the public surface; _-prefix internal modules)"
            )
    return errors


def _validate_release_crate_order(
    repo_root: Path, artifacts: list[Artifact]
) -> list[str]:
    """Cross-check the [[artifact]] cargo inventory against release.py CRATE_ORDER.

    Every crate the release driver publishes must be a declared artifact (so its
    version and license payload are validated), and every declared cargo artifact
    must be in the publish order (so it cannot be silently skipped at release).
    """
    release_py = repo_root / "scripts" / "release.py"
    crate_order = _release_crate_order(release_py)
    if crate_order is None:
        return [f"{release_py.relative_to(repo_root)}: CRATE_ORDER list not found"]

    declared = {a.name for a in artifacts if a.ecosystem == "cargo" and a.publish}
    ordered = set(crate_order)
    errors: list[str] = []
    for name in sorted(ordered - declared):
        errors.append(
            f"release.py CRATE_ORDER publishes {name} but rlmesh.toml has no "
            f"publish=true cargo artifact for it"
        )
    for name in sorted(declared - ordered):
        errors.append(
            f"rlmesh.toml declares cargo artifact {name} but release.py "
            f"CRATE_ORDER does not publish it"
        )
    errors.extend(_crate_order_violations(repo_root, crate_order))
    return errors


def _crate_order_violations(repo_root: Path, crate_order: list[str]) -> list[str]:
    """Each crate must publish after the published crates it depends on."""
    position = {name: index for index, name in enumerate(crate_order)}
    errors: list[str] = []
    for name in crate_order:
        manifest = repo_root / "crates" / name / "Cargo.toml"
        if not manifest.exists():
            continue
        cargo = tomllib.loads(manifest.read_text(encoding="utf-8"))
        tables = [cargo.get("dependencies", {}), cargo.get("build-dependencies", {})]
        tables += [
            target.get(kind, {})
            for target in cargo.get("target", {}).values()
            for kind in ("dependencies", "build-dependencies")
        ]
        for dep in sorted({dep for table in tables for dep in table}):
            if dep in position and position[dep] > position[name]:
                errors.append(
                    f"release.py CRATE_ORDER publishes {name} before {dep}, "
                    f"which it depends on"
                )
    return errors


def _release_crate_order(path: Path) -> list[str] | None:
    if not path.exists():
        return None
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(t, ast.Name) and t.id == "CRATE_ORDER" for t in node.targets
        ):
            continue
        value = ast.literal_eval(node.value)
        if isinstance(value, list) and all(isinstance(v, str) for v in value):
            return value
    return None


def _defines_dunder_all(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
            return True
    return False


EDITION_ARM_RE = re.compile(r'Edition::E\d{4}_\d{2} => "(?P<name>[0-9]{4}\.[0-9]{2})"')


def _edition_arms(lib_text: str) -> list[str]:
    """The editions ``rlmesh-proto`` names arms for, in source order."""
    return sorted(set(EDITION_ARM_RE.findall(lib_text)))


def _check_edition_source(lib_text: str, current: str, known: list[str]) -> list[str]:
    """The build knows exactly the manifest's editions and runs at its current one."""
    errors: list[str] = []
    constants = {
        match.group("name"): match.group("value")
        for match in STR_CONST_RE.finditer(lib_text)
    }
    if constants.get("CURRENT_WORKFLOW_EDITION") != current:
        errors.append(
            f"CURRENT_WORKFLOW_EDITION is {constants.get('CURRENT_WORKFLOW_EDITION')!r}, "
            f"[workflow].current_edition is {current!r}"
        )
    arms = _edition_arms(lib_text)
    if arms != sorted(known):
        errors.append(
            f"the Edition enum knows {arms}, but [workflow.editions] minus "
            f"[workflow.dropped_editions] is {sorted(known)}"
        )
    return errors


def _validate_buf_policy(repo_root: Path) -> list[str]:
    errors: list[str] = []
    buf_config = repo_root / "buf.yaml"
    if not buf_config.exists():
        errors.append("buf.yaml is required for proto lint and breaking-change policy")
    else:
        buf_lines = buf_config.read_text(encoding="utf-8").splitlines()
        if not any(line.split("#", 1)[0].strip() == "breaking:" for line in buf_lines):
            errors.append(
                "buf.yaml must include an active (uncommented) breaking-change policy"
            )

    return errors


def _known_editions(editions: dict[str, Any], dropped: Any) -> list[str]:
    """The editions a build knows: every recorded edition not dropped."""
    gone = set(dropped) if isinstance(dropped, dict) else set()
    return sorted(edition for edition in editions if edition not in gone)


def _validate_workflow_editions(
    repo_root: Path,
    workflow: dict[str, Any],
    release: dict[str, Any],
    workspace_version: str | None,
) -> list[str]:
    errors: list[str] = []
    editions = workflow.get("editions")
    if not isinstance(editions, dict):
        return ["missing [workflow.editions] table"]
    known = _known_editions(editions, workflow.get("dropped_editions"))

    current_edition = workflow.get("current_edition")
    if not isinstance(current_edition, str) or not EDITION_BASE_RE.match(
        current_edition
    ):
        errors.append("[workflow].current_edition must be `YYYY.MM` (zero-padded)")
        current_edition = ""
    elif current_edition not in known:
        errors.append(
            f"[workflow].current_edition {current_edition!r} must be a recorded, "
            "undropped [workflow.editions] entry"
        )
    elif known and max(known) != current_edition:
        errors.append(
            f"[workflow].current_edition is {current_edition!r}, but {max(known)!r} is "
            "newer; the current edition is the newest one a build knows"
        )

    lib = repo_root / "crates/rlmesh-proto/src/lib.rs"
    errors.extend(
        f"{lib.relative_to(repo_root)}: {error}"
        for error in _check_edition_source(
            lib.read_text(encoding="utf-8"), current_edition, known
        )
    )

    release_status = release.get("status")
    if workspace_version is not None:
        expected_status = _release_status_for_version(workspace_version)
        if expected_status is None:
            errors.append(
                f"workspace version {workspace_version!r} is not valid Rust SemVer"
            )
        elif release_status != expected_status:
            errors.append(
                f"[release].status is {release_status!r}, expected {expected_status!r} "
                f"from workspace version {workspace_version!r}"
            )

    for edition, entry in editions.items():
        prefix = f'[workflow.editions."{edition}"]'
        if not isinstance(entry, dict):
            errors.append(f"{prefix} must be a table")
            continue
        if not EDITION_BASE_RE.match(edition):
            errors.append(f"{prefix}: an edition is named `YYYY.MM` (zero-padded)")

        status = entry.get("status")
        if status not in {"provisional", "sealed"}:
            errors.append(f"{prefix}.status must be 'provisional' or 'sealed'")
        if status == "provisional" and release_status == "stable":
            errors.append(
                f"{prefix}: stable releases must not ship provisional editions; "
                "seal the edition first"
            )

        sealed_in = entry.get("sealed_in")
        if status == "sealed":
            if not isinstance(sealed_in, str) or not STABLE_VERSION_RE.match(sealed_in):
                errors.append(
                    f"{prefix}.sealed_in must be the stable (non-prerelease) version "
                    "that sealed the edition"
                )
        elif sealed_in is not None:
            errors.append(
                f"{prefix}.sealed_in is only recorded once the edition is sealed"
            )

        spec = entry.get("spec")
        if not isinstance(spec, str):
            errors.append(f"{prefix}.spec must be a string path")
            continue
        spec_path = repo_root / spec
        if not spec_path.exists():
            errors.append(f"{prefix}.spec does not exist: {spec}")
            continue

        spec_sha256 = entry.get("spec_sha256")
        if status == "sealed":
            try:
                actual = _canonical_spec_sha256(spec_path)
            except ValueError as exc:
                errors.append(f"{prefix}: {exc}")
                continue
            if not isinstance(spec_sha256, str):
                errors.append(f"{prefix}.spec_sha256 is required for sealed editions")
            elif actual != spec_sha256:
                errors.append(
                    f"{prefix}: sealed spec {spec} canonical sha256 is {actual}, "
                    f"manifest declares {spec_sha256}"
                )
        elif spec_sha256 is not None:
            errors.append(
                f"{prefix}.spec_sha256 is only recorded once the edition is sealed"
            )

    errors.extend(_check_dropped_editions(workflow.get("dropped_editions"), editions))
    return errors


def _sealed_editions(manifest_text: str) -> set[str]:
    """Editions marked ``status = "sealed"`` in a manifest's ``[workflow.editions]``."""
    manifest = tomllib.loads(manifest_text)
    workflow = manifest.get("workflow")
    editions = workflow.get("editions") if isinstance(workflow, dict) else None
    if not isinstance(editions, dict):
        return set()
    return {
        name
        for name, entry in editions.items()
        if isinstance(entry, dict) and entry.get("status") == "sealed"
    }


def _changed_sealed_records(
    previous_manifest_text: str, editions: dict[str, Any]
) -> list[str]:
    """Find released seal records changed alongside a spec or its checksum."""
    previous = (
        tomllib.loads(previous_manifest_text).get("workflow", {}).get("editions", {})
    )
    fields = ("status", "spec", "sealed_in", "spec_sha256")
    return sorted(
        edition
        for edition in _sealed_editions(previous_manifest_text)
        if not isinstance(editions.get(edition), dict)
        or any(
            editions[edition].get(field) != previous[edition].get(field)
            for field in fields
        )
    )


def _check_dropped_editions(dropped: Any, editions: dict[str, Any]) -> list[str]:
    """``[workflow.dropped_editions]`` is the one way a build stops knowing a sealed edition.

    A deliberate major break may drop a sealed edition, but only by naming it
    with its reason; the edition keeps its sealed record (and checked spec hash).
    """
    if dropped is None:
        return []
    if not isinstance(dropped, dict):
        return ["[workflow.dropped_editions] must be a table of edition = reason"]
    errors: list[str] = []
    for edition, reason in dropped.items():
        prefix = f'[workflow.dropped_editions]."{edition}"'
        if not isinstance(reason, str) or not reason.strip():
            errors.append(f"{prefix} must give the reason the edition is dropped")
        entry = editions.get(edition)
        if not isinstance(entry, dict) or entry.get("status") != "sealed":
            errors.append(
                f"{prefix} names no sealed edition; keep the dropped edition's "
                "sealed [workflow.editions] record"
            )
    return errors


def _validate_sealed_edition_retention(
    repo_root: Path, manifest_path: Path, workflow: dict[str, Any]
) -> list[str]:
    """Keep every sealed record the last release tag had, unchanged.

    A build knows every recorded edition that is not dropped, so a sealed edition
    stays known as long as its record does. Comparing against the tag catches a
    deleted ``[workflow.editions."…"]`` table and a spec rewritten together with
    its checksum. Skipped when there is no tag yet (or no git).
    """
    tag = _last_release_tag(repo_root)
    if tag is None:
        return []
    try:
        tracked = manifest_path.relative_to(repo_root).as_posix()
    except ValueError:
        return []
    previous = _git_output(repo_root, ["show", f"{tag}:{tracked}"])
    if previous is None:
        return []
    editions = workflow.get("editions")
    if not isinstance(editions, dict):
        return []
    return [
        f"sealed edition {edition!r} record differs from {tag}; "
        "status, spec, sealed_in and spec_sha256 must remain unchanged, even when dropped"
        for edition in _changed_sealed_records(previous, editions)
    ]


def _int_expr(expr: str) -> int | None:
    """Value of a Rust integer expression made of literals, ``*`` and ``+``."""

    def value(node: ast.expr) -> int | None:
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult | ast.Add):
            left, right = value(node.left), value(node.right)
            if left is None or right is None:
                return None
            return left * right if isinstance(node.op, ast.Mult) else left + right
        return None

    try:
        return value(ast.parse(expr.strip(), mode="eval").body)
    except SyntaxError:
        return None


def _git_output(repo_root: Path, args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout if result.returncode == 0 else None


def _last_release_tag(repo_root: Path) -> str | None:
    tag = _git_output(repo_root, ["describe", "--tags", "--abbrev=0", "--match", "v*"])
    return (tag.strip() or None) if tag is not None else None


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

    if release.get("status") not in {"alpha", "beta", "rc", "stable"}:
        errors.append("[release].status must be alpha, beta, rc, or stable")

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

    errors.extend(_validate_native_api_surfaces(repo_root, api_surface, artifact_ids))
    return errors


NATIVE_API_SURFACES = {
    "rust": ("crate",),
    "c": ("header", "standard"),
    "cpp": ("header", "standard", "namespace"),
}


def _validate_native_api_surfaces(
    repo_root: Path, api_surface: dict[str, Any], artifact_ids: set[str]
) -> list[str]:
    errors: list[str] = []
    for language, source_keys in NATIVE_API_SURFACES.items():
        table = api_surface.get(language)
        if table is None:
            continue
        prefix = f"[api_surface.{language}]"
        if not isinstance(table, dict):
            errors.append(f"{prefix} must be a table")
            continue
        for key, expected in (
            ("kind", f"rlmesh.{language}-api-surface"),
            ("manifest_kind", f"rlmesh.{language}-api-surface-manifest"),
            ("access", "early-access"),
        ):
            if table.get(key) != expected:
                errors.append(f"{prefix}.{key} must be {expected!r}")
        if table.get("stability") not in {"Preview", "Experimental"}:
            errors.append(f"{prefix}.stability must be Preview or Experimental")
        if table.get("package_artifact") not in artifact_ids:
            errors.append(
                f"{prefix}.package_artifact does not match any [[artifact]].id"
            )
        for key in ("package_name", "summary", *source_keys):
            if not isinstance(table.get(key), str) or not table[key]:
                errors.append(f"{prefix}.{key} must be a non-empty string")
        header = table.get("header")
        if isinstance(header, str) and not (repo_root / header).exists():
            errors.append(f"{prefix}.header does not exist: {header}")
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


def _release_status_for_version(version: str) -> str | None:
    match = RUST_SEMVER_RE.match(version)
    if match is None:
        return None
    return match.group("channel") or "stable"


def _pep440(version: str) -> str:
    """PEP 440 spelling of a Rust SemVer version (mirrors bump_version.pep440)."""
    for tag, short in (("-alpha.", "a"), ("-beta.", "b"), ("-rc.", "rc")):
        if tag in version:
            base, suffix = version.split(tag, 1)
            return f"{base}{short}{suffix}"
    return version


def _prose_version_files(repo_root: Path) -> list[Path]:
    """User-facing prose whose prerelease literals must track the release.

    The top-level + crate READMEs, example READMEs, and docs — excluding the
    changelog and the version-stamped edition specs (both legitimately name old
    versions). Mirrors ``prose_version_files`` in ``bump_version.py``.
    """
    files = [repo_root / "README.md"]
    files += sorted(repo_root.glob("crates/*/README.md"))
    files += sorted((repo_root / "examples").rglob("README.md"))
    files += [
        md
        for md in sorted((repo_root / "docs").rglob("*.md"))
        if md.relative_to(repo_root).as_posix() != "docs/changelog.md"
        and not md.relative_to(repo_root).as_posix().startswith("docs/editions/")
    ]
    return [path for path in files if path.exists()]


def _validate_doc_versions(repo_root: Path, workspace_version: str | None) -> list[str]:
    """Every prerelease version literal in prose must equal the current release.

    ``bump_version.py`` rewrites these on a bump; this guard makes its "fails
    loudly if any version-bearing spot was missed" promise real — a stale ``rc.N``
    left in the docs is caught here, not shipped. Stable ``X.Y.Z`` literals are
    allowed (forward/historical references).
    """
    if workspace_version is None:
        return []
    allowed = {workspace_version, _pep440(workspace_version)}
    errors: list[str] = []
    for path in _prose_version_files(repo_root):
        stale = sorted(
            {
                match.group(0)
                for match in PRERELEASE_LITERAL_RE.finditer(
                    path.read_text(encoding="utf-8")
                )
                if match.group(0) not in allowed
            }
        )
        for literal in stale:
            errors.append(
                f"{path.relative_to(repo_root)}: stale version literal {literal!r}; "
                f"expected {workspace_version!r} — run `mise run bump`, or use "
                "release-neutral phrasing"
            )
    return errors


def _validate_rust_version(repo_root: Path) -> list[str]:
    mise_rust = _read_toml(repo_root / "mise.toml").get("tools", {}).get("rust")
    package = (
        _read_toml(repo_root / "Cargo.toml").get("workspace", {}).get("package", {})
    )
    return _check_rust_version(mise_rust, package.get("rust-version"))


CAPI_HEADER = Path("crates/rlmesh-capi/include/rlmesh.h")


def _validate_capi_version(repo_root: Path, workspace_version: str | None) -> list[str]:
    header = repo_root / CAPI_HEADER
    if workspace_version is None or not header.is_file():
        return []
    return _check_capi_version(header.read_text(encoding="utf-8"), workspace_version)


def _check_capi_version(header_text: str, workspace_version: str) -> list[str]:
    """``RLMESH_VERSION`` in the hand-written C header is the workspace version."""
    match = re.search(r'^#define RLMESH_VERSION "([^"]*)"$', header_text, re.MULTILINE)
    if match is None:
        return [f"{CAPI_HEADER} must #define RLMESH_VERSION as a string literal"]
    if match.group(1) != workspace_version:
        return [
            f"{CAPI_HEADER} RLMESH_VERSION is {match.group(1)!r} but the workspace "
            f"version is {workspace_version!r}; scripts/bump_version.py keeps them equal"
        ]
    return []


BINDING_VERSIONS = (
    (Path("javascript/rlmesh/package.json"), r'^  "version": "([^"]*)"'),
    (Path("integrations/unity/package.json"), r'^  "version": "([^"]*)"'),
    (Path("integrations/unity/package.json"), r'^    "com\.rlmesh\.rlmesh": "([^"]*)"'),
    (Path("csharp/src/RLMesh/RLMesh.csproj"), r"<Version>([^<]*)</Version>"),
)


def _validate_binding_versions(
    repo_root: Path, workspace_version: str | None
) -> list[str]:
    """Binding manifests that scripts/bump_version.py rewrites carry the workspace version."""
    if workspace_version is None:
        return []
    errors: list[str] = []
    for path, pattern in BINDING_VERSIONS:
        manifest = repo_root / path
        if not manifest.is_file():
            continue
        match = re.search(pattern, manifest.read_text(encoding="utf-8"), re.MULTILINE)
        if match is None:
            errors.append(f"{path}: no version matching {pattern!r}")
        elif match.group(1) != workspace_version:
            errors.append(
                f"{path} has {match.group(1)!r} but the workspace version is "
                f"{workspace_version!r}; scripts/bump_version.py keeps them equal"
            )
    return errors


def _check_rust_version(mise_rust: Any, rust_version: Any) -> list[str]:
    """``[workspace.package].rust-version`` must equal the mise rust toolchain.

    Rust has no compatibility promise yet, so the MSRV is not a separate floor:
    it moves with the toolchain CI builds on. Compared at major.minor.
    """
    toolchain = mise_rust.get("version") if isinstance(mise_rust, dict) else mise_rust
    if not isinstance(toolchain, str) or not re.match(r"^\d+\.\d+", toolchain):
        return [f"mise.toml [tools].rust must pin a numeric version, got {toolchain!r}"]
    if not isinstance(rust_version, str):
        return ["root Cargo.toml is missing [workspace.package].rust-version"]
    if toolchain.split(".")[:2] != rust_version.split(".")[:2]:
        return [
            f"Cargo.toml rust-version is {rust_version!r} but mise.toml pins rust "
            f"{toolchain!r}; keep them in lockstep"
        ]
    return []


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

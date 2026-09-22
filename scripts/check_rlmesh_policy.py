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
from itertools import chain
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
STR_CONST_RE = re.compile(r'pub const (?P<name>[A-Z0-9_]+): &str =\s*"(?P<value>[^"]+)";')
PY_STR_CONST_RE = re.compile(r'(?P<name>[A-Z0-9_]+)\s*=\s*"(?P<value>[^"]+)"')
INT_CONST_RE = re.compile(r"(?P<name>[A-Z0-9_]+)\s*=\s*(?P<value>\d+)")
# A proto package path: `rlmesh.<pkg>.vN`. The generation token must NOT match
# this shape (it is an opaque handshake value, not a package namespace).
PACKAGE_PATH_TOKEN_RE = re.compile(r"^rlmesh\.[a-z]+\.v\d+$")
PROTO_PACKAGE_RE = re.compile(r"^\s*package\s+(?P<package>[A-Za-z0-9_.]+)\s*;", re.MULTILINE)
PROTO_BLOCK_RE = re.compile(r"^(?P<kind>message|enum|oneof)\s+(?P<name>\w+)\s*\{")
# An enum value (`NAME = 1;`) or a field/oneof arm (`Type name = 1;`), with any
# label words and an optional `[option]` block; `reserved` lines never match.
PROTO_ENTRY_RE = re.compile(r"^(?:[\w.]+(?:<[^>]*>)?\s+)*\w+\s*=\s*\d+\s*(?:\[.*\])?\s*;")
MAX_MESSAGE_SIZE_RE = re.compile(r"pub const MAX_MESSAGE_SIZE: usize = (?P<expr>[^;]+);")
# Oneofs whose new arm an old peer cannot refuse: `MetaValue.kind` reads as null
# and `SpaceSpec.spec` as an unset spec, so their arm counts stay manifest-pinned.
FROZEN_ONEOFS = ("rlmesh.spaces.v1.MetaValue.kind", "rlmesh.spaces.v1.SpaceSpec.spec")
# Workflow edition name shapes. A SEALED edition is the bare `YYYY.MM` base; an
# official PROVISIONAL prerelease edition appends the full Rust SemVer
# prerelease cohort (`YYYY.MM-X.Y.Z-{alpha,beta,rc}.N`). Local dev cohorts are
# generated at build time and are not committed to `rlmesh.toml`.
EDITION_BASE_RE = re.compile(r"^[0-9]{4}\.[0-9]{2}$")
EDITION_PRERELEASE_SUFFIX_RE = re.compile(
    r"^\d+\.\d+\.\d+-(?:alpha|beta|rc)\.\d+$"
)
# A prerelease version literal in prose: `X.Y.Z-rc.N` (SemVer) or `X.Y.ZrcN`
# (PEP 440). Bare stable `X.Y.Z` is intentionally NOT matched — forward/historical
# references to stable versions are fine; only the moving prerelease cohort must
# track the current release.
PRERELEASE_LITERAL_RE = re.compile(
    r"\d+\.\d+\.\d+(?:-(?:alpha|beta|rc)\.\d+|(?:a|b|rc)\d+)"
)


def _split_edition_name(name: str) -> tuple[str, str | None]:
    """Split a workflow edition name into ``(base, suffix)`` at its first ``-``.

    A bare ``YYYY.MM`` yields ``(name, None)``; ``YYYY.MM-<cohort>`` yields
    ``(base, suffix)``. Mirrors ``edition_sort_key`` in ``rlmesh-proto/src/lib.rs``.
    """
    base, sep, suffix = name.partition("-")
    return (base, suffix if sep else None)


def _edition_sort_key(name: str) -> tuple[str, bool, str]:
    """Ordering key for a workflow edition name, mirroring ``edition_sort_key``.

    Base date first (zero-padded ``YYYY.MM``, so lexicographic is chronological),
    then a suffixed cohort above its bare base, then the suffix itself as a
    deterministic tiebreak.
    """
    base, suffix = _split_edition_name(name)
    return (base, suffix is not None, suffix or "")


def _current_edition_not_highest(current_edition: str, supported: list[str]) -> str | None:
    """The error when ``current_edition`` does not sort highest among ``supported``.

    A build declares ``current_edition`` as its WANT and offers ``supported`` as
    its CAN set, so a retained edition sorting above the current one could never
    be selected and would make this build's own offer cap its peers.
    """
    highest = max(supported, key=_edition_sort_key, default=current_edition)
    if _edition_sort_key(highest) > _edition_sort_key(current_edition):
        return (
            f"[workflow].current_edition is {current_edition!r}, but retained edition "
            f"{highest!r} sorts above it; current_edition must be the highest supported"
        )
    return None


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
    """Unit tests for the manifest-vs-generated-list rules (mirrors bump_version.py)."""
    lib = 'const L: &str = env!("RLMESH_SUPPORTED_WORKFLOW_EDITIONS");'
    build = "supported_editions -> cargo:rustc-env=RLMESH_SUPPORTED_WORKFLOW_EDITIONS"
    rc1 = "2026.09-0.2.0-rc.1"

    # The generated list is the current cohort first, then every retained edition.
    assert _generated_supported_editions(rc1, [rc1, "2026.06"]) == [rc1, "2026.06"]
    assert _generated_supported_editions("2026.06", ["2026.06"]) == ["2026.06"]
    assert not _check_generated_supported_editions(rc1, [rc1, "2026.06"], lib, build)
    # A hand-written Rust literal is no longer the source of the list.
    assert _check_generated_supported_editions(rc1, [rc1], "&[CURRENT]", build)
    # build.rs must be the generator.
    assert _check_generated_supported_editions(rc1, [rc1], lib, "fn main() {}")
    # current_edition missing from the manifest's retained set: the generated
    # list then carries an edition rlmesh.toml does not retain.
    assert _check_generated_supported_editions(rc1, ["2026.06"], lib, build)
    # A repeat would be offered twice on the wire.
    assert _check_generated_supported_editions(rc1, [rc1, "2026.06", "2026.06"], lib, build)

    # The crate's edition file is current_edition first, then the other retained
    # editions; a crates.io build takes its base edition from that first line.
    retained = [rc1, "2026.06"]
    assert not _check_packaged_supported_editions(f"{rc1}\n2026.06\n", rc1, retained)
    assert not _check_packaged_supported_editions(
        f"{rc1}\n2026.06\n", rc1, ["2026.06", rc1]
    )
    assert _check_packaged_supported_editions(f"{rc1}\n", rc1, retained)
    assert "must start with" in _check_packaged_supported_editions(
        f"2026.06\n{rc1}\n", rc1, retained
    )[0]
    assert "must start with" in _check_packaged_supported_editions("", rc1, retained)[0]

    previous = """[workflow]
supported_editions = ["2026.09-0.2.0-rc.1", "2026.06"]

[workflow.editions."2026.06"]
status = "sealed"
sealed_in = "0.1.0"

[workflow.editions."2026.09-0.2.0-rc.1"]
status = "provisional"
"""
    assert _sealed_editions(previous) == {"2026.06"}
    assert not _dropped_sealed_editions(previous, ["2026.09", "2026.06"])
    assert _dropped_sealed_editions(previous, ["2026.09"]) == ["2026.06"]
    # A provisional cohort may be retired; only sealed editions are sticky.
    assert not _dropped_sealed_editions(previous, ["2026.06"])

    # build.rs scans the manifest without a TOML parser: its reading of the
    # retained list must equal tomllib's for both array spellings, or the build
    # would silently offer fewer editions than the manifest retains.
    single = 'supported_editions = ["2026.09-0.2.0-rc.1", "2026.06"]\n'
    multi = (
        "supported_editions = [\n"
        '  "2026.09-0.2.0-rc.1", # the moving cohort\n'
        '  "2026.06",\n'
        "]\n"
    )
    for spelling in (single, multi):
        text = "[workflow]\n" + spelling
        parsed = tomllib.loads(text)["workflow"]["supported_editions"]
        assert _scan_manifest_string_list(text, "supported_editions") == parsed
    assert _scan_manifest_string_list("[workflow]\n" + single, "current_edition") == []

    # current_edition is this build's WANT, so nothing it retains may sort above
    # it — same ordering rlmesh-proto's `edition_sort_key` applies.
    assert _current_edition_not_highest(rc1, [rc1, "2026.06"]) is None
    assert _current_edition_not_highest("2026.06", ["2026.06"]) is None
    assert _current_edition_not_highest("2026.06", [rc1, "2026.06"]) is not None
    # A suffixed cohort sorts above its own bare base, either way round.
    cohort = "2026.06-0.2.0-rc.1"
    assert _current_edition_not_highest("2026.06", ["2026.06", cohort]) is not None
    assert _current_edition_not_highest(cohort, [cohort, "2026.06"]) is None

    # The proto line scan: nested enums are qualified by their message, comments
    # and map/labelled fields are not entries, a service block nests nothing.
    sample = """syntax = "proto3";
package rlmesh.t.v1;
// enum Ghost { A = 0; }
message Outer {
  oneof kind {
    int64 integer = 1; // a = 9;
    Inner inner = 2 [deprecated = true];
  }
  enum Mode {
    MODE_UNSPECIFIED = 0;
    MODE_ON = 1;
  }
  map<string, Outer> entries = 3;
  optional uint64 total_ns = 4;
  reserved 5;
}
message Empty {}
enum Top {
  TOP_UNSPECIFIED = 0;
  TOP_A = 1;
  TOP_B = 10;
}
service S {
  rpc Call(Outer) returns (stream Empty) {}
}
"""
    enums, oneofs = _scan_proto_vocabulary(sample)
    assert enums == {"rlmesh.t.v1.Outer.Mode": 2, "rlmesh.t.v1.Top": 3}
    assert oneofs == {"rlmesh.t.v1.Outer.kind": 2}

    frozen = {name: 1 for name in FROZEN_ONEOFS}
    wire: dict[str, Any] = {
        "max_message_bytes": 256 * 1024 * 1024,
        "enums": {"rlmesh.t.v1.Outer.Mode": 2, "rlmesh.t.v1.Top": 3},
        "oneofs": {"rlmesh.t.v1.Outer.kind": 2, **frozen},
    }
    live_oneofs = {**oneofs, **frozen}
    assert not _check_wire_vocabulary(wire, enums, live_oneofs)
    # A new enum value, a new enum, a stale entry, and a new oneof arm each need
    # the manifest edited in the same change.
    grown = {**enums, "rlmesh.t.v1.Top": 4}
    assert _check_wire_vocabulary(wire, grown, live_oneofs) == [
        '[wire.enums]."rlmesh.t.v1.Top" is 3, the proto declares 4 values; a new '
        "value is a wire-v1 vocabulary change: record the count here in the same "
        "change and emit it only when the target leg's edition covers it "
        '(docs/editions/index.md, "Additive-forever rules")'
    ]
    assert _check_wire_vocabulary(wire, {**enums, "rlmesh.t.v1.New": 1}, live_oneofs)
    assert _check_wire_vocabulary(wire, {"rlmesh.t.v1.Top": 3}, live_oneofs)
    assert _check_wire_vocabulary(wire, enums, {**live_oneofs, "rlmesh.t.v1.Outer.kind": 3})
    # An error-code enum names the silent fold in its message.
    fold = _check_wire_vocabulary(
        {**wire, "enums": {**wire["enums"], "rlmesh.t.v1.EnvErrorCode": 2}},
        {**enums, "rlmesh.t.v1.EnvErrorCode": 3},
        live_oneofs,
    )
    assert len(fold) == 1 and "folds an unknown value to UNSPECIFIED" in fold[0]
    # The frozen oneofs must stay recorded even if the manifest is trimmed.
    assert _check_wire_vocabulary(
        {**wire, "oneofs": {"rlmesh.t.v1.Outer.kind": 2}}, enums, live_oneofs
    ) == [f'[wire.oneofs] must record "{name}"' for name in FROZEN_ONEOFS]
    # A listed oneof only the manifest knows is stale, and a type error is named.
    assert _check_wire_vocabulary(wire, enums, frozen)
    assert _check_wire_vocabulary({**wire, "enums": []}, enums, live_oneofs)

    # The gRPC cap: the Rust constant expression must equal the pinned bytes.
    lib = "pub const MAX_MESSAGE_SIZE: usize = 256 * 1024 * 1024;"
    assert _int_expr("256 * 1024 * 1024") == 268435456
    assert _int_expr("1 + 2") == 3
    assert _int_expr("1 << 20") is None
    assert not _check_message_cap(wire, lib)
    assert _check_message_cap({**wire, "max_message_bytes": 4 * 1024 * 1024}, lib)
    assert _check_message_cap(wire, "pub const MAX_MESSAGE_SIZE: usize = 1 << 28;")
    assert _check_message_cap(wire, "")
    assert _check_message_cap({**wire, "max_message_bytes": "256 MiB"}, lib)

    print("policy self-check passed")


def validate_rlmesh_policy(*, repo_root: Path, manifest_path: Path) -> list[str]:
    errors: list[str] = []
    manifest = _read_toml(manifest_path)

    release = _required_table(manifest, "release", errors)
    workflow = _required_table(manifest, "workflow", errors)
    protocol = _required_table(manifest, "protocol", errors)
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
    errors.extend(_validate_protocol_and_workflow(repo_root, protocol))
    errors.extend(_validate_workflow_editions(repo_root, workflow, release, workspace_version))
    errors.extend(
        _validate_generated_supported_editions(repo_root, manifest_path, workflow)
    )
    errors.extend(_validate_packaged_supported_editions(repo_root, workflow))
    errors.extend(_validate_sealed_edition_retention(repo_root, manifest_path, workflow))
    errors.extend(_validate_wire(repo_root, _required_table(manifest, "wire", errors)))
    errors.extend(_validate_adapters(repo_root))
    errors.extend(_validate_python_public_modules(repo_root))
    errors.extend(
        _validate_api_surface(
            repo_root, api_surface, artifact_ids, release, package_family
        )
    )
    errors.extend(_validate_doc_versions(repo_root, workspace_version))

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
    # The version-stamped metadata key string, not any source-module path, is the
    # adapter spec-format discriminator, so guard the v1 token here. A v2 bump
    # must be deliberate and keep reading v1.
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


def _validate_release_crate_order(repo_root: Path, artifacts: list[Artifact]) -> list[str]:
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
    return errors


def _release_crate_order(path: Path) -> list[str] | None:
    if not path.exists():
        return None
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "CRATE_ORDER" for t in node.targets):
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


def _validate_protocol_and_workflow(repo_root: Path, protocol: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    source = repo_root / "crates/rlmesh-proto/src/lib.rs"
    source_text = source.read_text(encoding="utf-8")
    constants = dict(STR_CONST_RE.findall(source_text))
    expected = {
        "PROTOCOL_GENERATION": (
            "[protocol].current_generation",
            protocol.get("current_generation"),
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

    if "CURRENT_WORKFLOW_EDITION_SPEC_SHA256" in source_text:
        errors.append(
            f"{source}: remove CURRENT_WORKFLOW_EDITION_SPEC_SHA256; provisional "
            "editions are release/dev cohorts, not docs hashes"
        )
    if 'env!("RLMESH_CURRENT_WORKFLOW_EDITION")' not in source_text:
        errors.append(
            f"{source}: CURRENT_WORKFLOW_EDITION must come from build-time cohort env"
        )

    # Guard the protocol-generation token's *shape*. The token is an opaque
    # handshake value, not a proto package path: it must not look like a
    # per-package namespace (`rlmesh.<pkg>.vN`) and must not collide with any
    # declared proto package. This keeps Axis 1 (generation) visually and
    # semantically distinct from Axis 4 (package vN). See versioning-governance §3.
    declared_packages = _declared_proto_packages(repo_root)
    for manifest_key, value in (
        ("[protocol].current_generation", protocol.get("current_generation")),
    ):
        if not isinstance(value, str):
            continue
        if PACKAGE_PATH_TOKEN_RE.match(value):
            errors.append(
                f"{manifest_key} {value!r} is shaped like a proto package path "
                "(rlmesh.<pkg>.vN); the generation token must be an opaque "
                "non-package string (e.g. rlmesh-wire-v1)"
            )
        if value in declared_packages:
            errors.append(
                f"{manifest_key} {value!r} collides with a declared proto package; "
                "the generation token must not name a real package"
            )

    for token in _forbidden_unpublished_protocol_tokens():
        if token in source_text:
            errors.append(
                f"{source}: remove unpublished legacy protocol token {token!r}"
            )

    buf_config = repo_root / "buf.yaml"
    if not buf_config.exists():
        errors.append("buf.yaml is required for proto lint and breaking-change policy")
    else:
        buf_lines = buf_config.read_text(encoding="utf-8").splitlines()
        if not any(
            line.split("#", 1)[0].strip() == "breaking:" for line in buf_lines
        ):
            errors.append(
                "buf.yaml must include an active (uncommented) breaking-change policy"
            )

    return errors


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

    base_edition = workflow.get("base_edition")
    if not isinstance(base_edition, str):
        errors.append("[workflow].base_edition must be a string")
        base_edition = ""
    elif not EDITION_BASE_RE.match(base_edition):
        errors.append("[workflow].base_edition must be `YYYY.MM` (zero-padded)")

    current_edition = workflow.get("current_edition")
    if not isinstance(current_edition, str):
        errors.append("[workflow].current_edition must be a string")
        current_edition = ""

    if workspace_version is not None and base_edition:
        expected_current = _official_current_edition(base_edition, workspace_version)
        if expected_current is None:
            errors.append(f"workspace version {workspace_version!r} is not valid Rust SemVer")
        elif current_edition and current_edition != expected_current:
            errors.append(
                f"[workflow].current_edition is {current_edition!r}, expected "
                f"{expected_current!r} for workspace version {workspace_version!r}"
            )

    supported = workflow.get("supported_editions")
    if not isinstance(supported, list) or not all(
        isinstance(item, str) for item in supported
    ):
        errors.append("[workflow].supported_editions must be a list of strings")
        supported = []
    elif current_edition and current_edition not in supported:
        errors.append("[workflow].supported_editions must include [workflow].current_edition")

    if current_edition and supported:
        not_highest = _current_edition_not_highest(current_edition, supported)
        if not_highest is not None:
            errors.append(not_highest)

    for edition in supported if isinstance(supported, list) else []:
        if edition not in editions:
            errors.append(
                f'supported edition {edition!r} has no [workflow.editions."{edition}"] entry'
            )

    release_status = release.get("status")
    if workspace_version is not None:
        expected_status = _release_status_for_version(workspace_version)
        if expected_status is None:
            errors.append(f"workspace version {workspace_version!r} is not valid Rust SemVer")
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

        status = entry.get("status")
        if status not in {"provisional", "sealed"}:
            errors.append(f"{prefix}.status must be 'provisional' or 'sealed'")
        if status == "provisional" and release_status == "stable":
            errors.append(
                f"{prefix}: stable releases must not ship provisional editions; "
                "seal the edition first"
            )

        base, suffix = _split_edition_name(edition)
        if not EDITION_BASE_RE.match(base):
            errors.append(
                f"{prefix}: edition name base {base!r} must be `YYYY.MM` (zero-padded)"
            )
        if suffix is None:
            if status != "sealed":
                errors.append(
                    f"{prefix}: a bare `YYYY.MM` name is sealed, but status is {status!r}; "
                    "a provisional edition's name must carry a SemVer prerelease suffix"
                )
        else:
            if not EDITION_PRERELEASE_SUFFIX_RE.match(suffix):
                errors.append(
                    f"{prefix}: provisional suffix {suffix!r} must be full Rust SemVer "
                    "prerelease (`X.Y.Z-{alpha,beta,rc}.N`)"
                )
            if status != "provisional":
                errors.append(
                    f"{prefix}: a suffixed edition is provisional, but status is {status!r}; "
                    "a sealed edition is named by its bare `YYYY.MM` base alone"
                )

        sealed_in = entry.get("sealed_in")
        if status == "sealed":
            if not isinstance(sealed_in, str) or not STABLE_VERSION_RE.match(sealed_in):
                errors.append(
                    f"{prefix}.sealed_in must be the stable (non-prerelease) version "
                    "that sealed the edition"
                )
            if edition not in (supported if isinstance(supported, list) else []):
                errors.append(
                    f"{prefix}: a sealed edition must stay in [workflow].supported_editions "
                    "(retained forever, never pruned)"
                )
        elif sealed_in is not None:
            errors.append(f"{prefix}.sealed_in is only recorded once the edition is sealed")

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

    seen_provisional_by_base: dict[str, str] = {}
    for edition in supported:
        base, suffix = _split_edition_name(edition)
        if suffix is None:
            continue
        prior = seen_provisional_by_base.get(base)
        if prior is not None:
            errors.append(
                f"[workflow].supported_editions has two provisional editions for "
                f"date {base!r} ({prior!r} and {edition!r}); keep at most one "
                "moving cohort per base, plus an optional sealed fallback"
            )
        else:
            seen_provisional_by_base[base] = edition

    return errors


def _scan_manifest_string_list(text: str, key: str) -> list[str]:
    """Mirror of ``manifest_string_list`` in ``crates/rlmesh-proto/build_manifest.rs``.

    The build script reads ``rlmesh.toml`` with a line scan rather than a TOML
    parser, so its reading of an array must agree with ``tomllib`` on every
    spelling the manifest may carry — single-line or spread over lines, with
    comments and trailing commas. ``selfcheck`` asserts that agreement; keep the
    two implementations in step.
    """
    prefix = f"{key} = ["
    lines = iter(line.strip() for line in text.splitlines())
    for line in lines:
        if line.startswith(prefix):
            first = line[len(prefix) :]
            break
    else:
        return []

    items = ""
    for line in chain([first], lines):
        line = line.split("#", 1)[0]
        head, closer, _ = line.partition("]")
        items += head
        if closer:
            break
        # A line break separates two entries exactly as a comma does.
        items += ","

    values: list[str] = []
    for item in items.split(","):
        item = item.strip()
        if not item.startswith('"'):
            continue
        value, quote, _ = item[1:].partition('"')
        if quote:
            values.append(value)
    return values


def _generated_supported_editions(current_edition: str, supported: list[str]) -> list[str]:
    """The list ``build.rs`` generates: this build's cohort first, then the rest.

    Mirrors ``supported_workflow_editions`` in ``crates/rlmesh-proto/build.rs``. A
    source build spells its own cohort into the first slot and drops the manifest's
    ``current_edition`` from the tail; the release spelling used here is the list a
    published build offers.
    """
    return [current_edition] + [e for e in supported if e != current_edition]


def _check_generated_supported_editions(
    current_edition: str, supported: list[str], lib_text: str, build_text: str
) -> list[str]:
    errors: list[str] = []
    if 'env!("RLMESH_SUPPORTED_WORKFLOW_EDITIONS")' not in lib_text:
        errors.append(
            "SUPPORTED_WORKFLOW_EDITIONS must be generated from the build-time "
            "RLMESH_SUPPORTED_WORKFLOW_EDITIONS env, not written as a literal"
        )
    if not (
        "RLMESH_SUPPORTED_WORKFLOW_EDITIONS" in build_text
        and "supported_editions" in build_text
    ):
        errors.append(
            "build.rs must emit RLMESH_SUPPORTED_WORKFLOW_EDITIONS from "
            "rlmesh.toml's [workflow].supported_editions"
        )

    generated = _generated_supported_editions(current_edition, supported)
    if len(set(generated)) != len(generated):
        errors.append(f"generated SUPPORTED_WORKFLOW_EDITIONS {generated} repeats an edition")
    if set(generated) != set(supported):
        errors.append(
            f"generated SUPPORTED_WORKFLOW_EDITIONS {generated} does not match "
            f"[workflow].supported_editions {supported}"
        )
    return errors


def _validate_generated_supported_editions(
    repo_root: Path, manifest_path: Path, workflow: dict[str, Any]
) -> list[str]:
    """``SUPPORTED_WORKFLOW_EDITIONS`` is generated, and generated losslessly.

    Both the offer and the accept read that one list, so it must be the manifest's
    retained set — generated by ``build.rs`` — rather than a hand-maintained Rust
    literal free to drift from it. ``build.rs`` has no TOML parser, so this also
    checks that its scan of THIS manifest sees the same array tomllib does: a
    spelling it cannot read would silently shrink the offer while every rule
    below (all reading tomllib) stayed green.
    """
    current_edition = workflow.get("current_edition")
    supported = workflow.get("supported_editions")
    if not isinstance(current_edition, str) or not isinstance(supported, list):
        return []
    if not all(isinstance(edition, str) for edition in supported):
        return []
    scanned = _scan_manifest_string_list(
        manifest_path.read_text(encoding="utf-8"), "supported_editions"
    )
    if scanned != supported:
        return [
            f"{manifest_path.name}: build.rs reads [workflow].supported_editions as "
            f"{scanned}, the manifest declares {supported}; write the array so the "
            "build script's scan reads it (see crates/rlmesh-proto/build_manifest.rs)"
        ]
    lib = repo_root / "crates/rlmesh-proto/src/lib.rs"
    build = repo_root / "crates/rlmesh-proto/build.rs"
    return [
        f"{lib.relative_to(repo_root)}: {error}"
        for error in _check_generated_supported_editions(
            current_edition,
            supported,
            lib.read_text(encoding="utf-8"),
            build.read_text(encoding="utf-8"),
        )
    ]


PACKAGED_SUPPORTED_EDITIONS = "crates/rlmesh-proto/supported_editions.txt"


def _check_packaged_supported_editions(
    text: str, current_edition: str, supported: list[str]
) -> list[str]:
    packaged = [line.strip() for line in text.splitlines() if line.strip()]
    fix = "run `python scripts/bump_version.py --sync-editions`"
    if packaged[:1] != [current_edition]:
        return [
            f"{PACKAGED_SUPPORTED_EDITIONS} must start with [workflow].current_edition "
            f"{current_edition!r}, its first line is {packaged[:1]}; {fix}"
        ]
    expected = _generated_supported_editions(current_edition, supported)
    if packaged == expected:
        return []
    return [f"{PACKAGED_SUPPORTED_EDITIONS} lists {packaged}, expected {expected}; {fix}"]


def _validate_packaged_supported_editions(
    repo_root: Path, workflow: dict[str, Any]
) -> list[str]:
    """The crate's edition file is ``current_edition`` then the other retained editions.

    Every build reads it, and a published crate (no ``rlmesh.toml``) also takes
    its base edition from the first line; ``build.rs`` refuses a source build
    whose file disagrees with the manifest.
    """
    current_edition = workflow.get("current_edition")
    supported = workflow.get("supported_editions")
    if not isinstance(current_edition, str) or not isinstance(supported, list):
        return []
    path = repo_root / PACKAGED_SUPPORTED_EDITIONS
    if not path.is_file():
        return [f"{PACKAGED_SUPPORTED_EDITIONS} is missing"]
    return _check_packaged_supported_editions(
        path.read_text(encoding="utf-8"), current_edition, supported
    )


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


def _dropped_sealed_editions(previous_manifest_text: str, supported: list[str]) -> list[str]:
    return sorted(_sealed_editions(previous_manifest_text) - set(supported))


def _validate_sealed_edition_retention(
    repo_root: Path, manifest_path: Path, workflow: dict[str, Any]
) -> list[str]:
    """No edition sealed by the previous release may leave ``supported_editions``.

    A sealed edition is retained forever: a peer pinned to it must keep finding it in
    every later build's offer. The intra-manifest rule in ``_validate_workflow_editions``
    only sees editions this manifest still declares, so deleting a whole
    ``[workflow.editions."…"]`` table slips past it — this compares against the previous
    release tag instead. Skipped when there is no tag yet (or no git).
    """
    supported = workflow.get("supported_editions")
    if not isinstance(supported, list) or not all(
        isinstance(edition, str) for edition in supported
    ):
        return []
    tag = _last_release_tag(repo_root)
    if tag is None:
        return []
    try:
        tracked = manifest_path.relative_to(repo_root).as_posix()
    except ValueError:
        # A manifest from outside the repo has no history to compare against.
        return []
    previous = _git_output(repo_root, ["show", f"{tag}:{tracked}"])
    if previous is None:
        return []
    return [
        f"[workflow].supported_editions dropped sealed edition {edition!r}, retained in "
        f"{tag}; sealed editions are kept forever"
        for edition in _dropped_sealed_editions(previous, supported)
    ]


def _validate_wire(repo_root: Path, wire: dict[str, Any]) -> list[str]:
    """The additive-forever shapes of ``rlmesh-wire-v1`` match ``[wire]``.

    The message cap, every enum's value count, and the frozen oneofs' arm counts
    are pinned in the manifest so none of them changes without a deliberate edit
    in the same change. The manifest edit is the paperwork; emitting the new
    value only to a peer whose edition covers it is the house rule the gate
    cannot see (``docs/editions/index.md``).
    """
    enums: dict[str, int] = {}
    oneofs: dict[str, int] = {}
    proto_root = repo_root / "crates/rlmesh-proto/proto"
    for proto_file in sorted(proto_root.rglob("*.proto")):
        file_enums, file_oneofs = _scan_proto_vocabulary(
            proto_file.read_text(encoding="utf-8")
        )
        enums.update(file_enums)
        oneofs.update(file_oneofs)
    lib = repo_root / "crates/rlmesh-grpc/src/lib.rs"
    return _check_wire_vocabulary(wire, enums, oneofs) + [
        f"{lib.relative_to(repo_root)}: {error}"
        for error in _check_message_cap(wire, lib.read_text(encoding="utf-8"))
    ]


def _scan_proto_vocabulary(text: str) -> tuple[dict[str, int], dict[str, int]]:
    """Enum value counts and oneof arm counts of one proto file, package-qualified.

    A line scan, like ``build_manifest.rs``: comments are stripped, each ``{``
    opens a block and each ``}`` closes one, and an entry line counts toward the
    innermost enum or oneof. Keys are ``<package>.<Message...>.<name>``.
    """
    package = ""
    stack: list[tuple[str, str]] = []
    enums: dict[str, int] = {}
    oneofs: dict[str, int] = {}
    for raw in text.splitlines():
        line = raw.split("//", 1)[0].strip()
        if not line:
            continue
        if declared := PROTO_PACKAGE_RE.match(line):
            package = declared.group("package")
        if opened := PROTO_BLOCK_RE.match(line):
            kind, name = opened.group("kind"), opened.group("name")
            stack.append((kind, name))
            path = ".".join([package, *(n for _, n in stack if n)])
            if kind == "enum":
                enums[path] = 0
            elif kind == "oneof":
                oneofs[path] = 0
        elif "{" in line:
            stack.append(("block", ""))
        elif stack and PROTO_ENTRY_RE.match(line):
            kind = stack[-1][0]
            path = ".".join([package, *(n for _, n in stack if n)])
            if kind == "enum":
                enums[path] += 1
            elif kind == "oneof":
                oneofs[path] += 1
        for _ in range(line.count("}")):
            if stack:
                stack.pop()
    return enums, oneofs


def _check_wire_vocabulary(
    wire: dict[str, Any], enums: dict[str, int], oneofs: dict[str, int]
) -> list[str]:
    errors: list[str] = []
    recorded_enums = wire.get("enums")
    if not isinstance(recorded_enums, dict):
        errors.append("[wire.enums] must be a table of `\"<package>.<Enum>\" = <value count>`")
        recorded_enums = {}
    recorded_oneofs = wire.get("oneofs")
    if not isinstance(recorded_oneofs, dict):
        errors.append(
            "[wire.oneofs] must be a table of `\"<package>.<Message>.<oneof>\" = <arm count>`"
        )
        recorded_oneofs = {}
    for name in FROZEN_ONEOFS:
        if name not in recorded_oneofs:
            errors.append(f'[wire.oneofs] must record "{name}"')

    for table, live, noun in (
        ("enums", enums, "value"),
        ("oneofs", oneofs, "arm"),
    ):
        recorded = recorded_enums if table == "enums" else recorded_oneofs
        for name in sorted(recorded.keys() - live.keys()):
            errors.append(
                f'[wire.{table}]."{name}" names no {table[:-1]} under crates/rlmesh-proto/proto'
            )
        for name, count in sorted(live.items()):
            if name not in recorded:
                if table == "enums":
                    errors.append(
                        f'[wire.enums] has no entry for proto enum "{name}" ({count} '
                        "values); every wire-v1 enum's vocabulary is pinned: record it"
                    )
                continue
            pinned = recorded[name]
            if not isinstance(pinned, int):
                errors.append(f'[wire.{table}]."{name}" must be an integer {noun} count')
            elif pinned != count:
                errors.append(
                    f'[wire.{table}]."{name}" is {pinned}, the proto declares {count} '
                    f"{noun}s; a new {noun} is a wire-v1 vocabulary change: record the "
                    "count here in the same change and emit it only when the target "
                    "leg's edition covers it (docs/editions/index.md, "
                    '"Additive-forever rules")'
                    + (
                        "; a 0.1.0 peer folds an unknown value to UNSPECIFIED, so a new "
                        "code must never carry error semantics an old peer needs"
                        if name.endswith("ErrorCode")
                        else ""
                    )
                )
    return errors


def _check_message_cap(wire: dict[str, Any], lib_text: str) -> list[str]:
    pinned = wire.get("max_message_bytes")
    if not isinstance(pinned, int):
        return ["[wire].max_message_bytes must be an integer byte count"]
    match = MAX_MESSAGE_SIZE_RE.search(lib_text)
    if match is None:
        return ["MAX_MESSAGE_SIZE constant not found"]
    actual = _int_expr(match.group("expr"))
    if actual is None:
        return [
            f"MAX_MESSAGE_SIZE expression {match.group('expr').strip()!r} is not a "
            "sum or product of integer literals; policy:check cannot pin it"
        ]
    if actual != pinned:
        return [
            f"MAX_MESSAGE_SIZE is {actual} bytes, rlmesh.toml [wire].max_message_bytes "
            f"pins {pinned}; the gRPC message cap is part of the rlmesh-wire-v1 "
            "contract, so change both deliberately in the same change"
        ]
    return []


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


def _declared_proto_packages(repo_root: Path) -> set[str]:
    """Collect every `package` declared under the proto tree."""
    proto_root = repo_root / "crates/rlmesh-proto/proto"
    packages: set[str] = set()
    if not proto_root.exists():
        return packages
    for proto_file in proto_root.rglob("*.proto"):
        text = proto_file.read_text(encoding="utf-8")
        packages.update(match.group("package") for match in PROTO_PACKAGE_RE.finditer(text))
    return packages


def _forbidden_unpublished_protocol_tokens() -> list[str]:
    """Legacy protocol identifiers that must never reappear in `rlmesh-proto`.

    Each token is assembled from fragments ON PURPOSE, not for cleverness: the
    whole point of this guard is that these legacy names are eradicated from the
    tree, so the literals must not appear verbatim *anywhere* — including in this
    checker. A literal here would make a plain "is `<token>` gone yet?" grep/audit
    false-positive on the guard's own source, and would self-match if the scan
    were ever widened beyond `rlmesh-proto/src/lib.rs`. Read the fragments below to
    recover each legacy name; do not inline them.
    """
    return [
        "_".join(["LEGACY", "0", "1", "A" + "BI", "VERSION"]),
        "".join(["A", "BI", "_VERSION"]),
        "_".join(["MIN", "SUPPORTED", "A" + "BI", "VERSION"]),
        "_".join(["is", "a" + "bi", "compatible"]),
        ".".join(["rlmesh", "v1"]),
    ]


def _release_status_for_version(version: str) -> str | None:
    match = RUST_SEMVER_RE.match(version)
    if match is None:
        return None
    return match.group("channel") or "stable"


def _official_current_edition(base_edition: str, workspace_version: str) -> str | None:
    status = _release_status_for_version(workspace_version)
    if status is None:
        return None
    if status == "stable":
        return base_edition
    return f"{base_edition}-{workspace_version}"


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

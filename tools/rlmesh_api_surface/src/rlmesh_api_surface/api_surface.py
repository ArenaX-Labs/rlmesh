"""Static public API extraction for RLMesh Python docs and snapshots."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import griffe
import tomllib

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[3]
NATIVE_MODULE = "rlmesh._rlmesh"
METADATA_PATH = PACKAGE_ROOT / "api_metadata.json"
RLMESH_MANIFEST_PATH = REPO_ROOT / "rlmesh.toml"
DOCSTRING_PARSER = "google"
DOCS_API_SURFACE_SCHEMA_VERSION = 2
DOCS_API_SURFACE_KIND = "rlmesh.python-api-surface"
DOCS_API_SURFACE_MANIFEST_KIND = "rlmesh.python-api-surface-manifest"
SOURCE_REPO = "ArenaX-Labs/rlmesh"

STABLE = "Stable"
INTERNAL = "Internal"
_IGNORED_MEMBER_NAMES = {"__enter__", "__exit__", "__repr__", "__del__"}


@dataclass(frozen=True)
class SourceLocation:
    repo: str
    path: str
    commit: str | None = None
    line: int | None = None
    end_line: int | None = None

    def to_docs_api_surface(self) -> dict[str, object]:
        snapshot: dict[str, object] = {
            "repo": self.repo,
            "path": self.path,
        }
        if self.commit:
            snapshot["commit"] = self.commit
        if self.line is not None:
            snapshot["line"] = self.line
        if self.end_line is not None and self.end_line != self.line:
            snapshot["endLine"] = self.end_line
        return snapshot


@dataclass(frozen=True)
class ApiMember:
    name: str
    kind: str
    signature: str
    documentation: str = ""
    defined_at: SourceLocation | None = None

    def to_contract(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "signature": self.signature,
        }

    def to_docs_api_surface(self) -> dict[str, object]:
        snapshot: dict[str, object] = {
            "name": self.name,
            "kind": self.kind,
            "signature": self.signature,
        }
        if self.documentation:
            snapshot["documentation"] = self.documentation
        if self.defined_at is not None:
            snapshot["definedAt"] = self.defined_at.to_docs_api_surface()
        return snapshot


@dataclass(frozen=True)
class ApiExport:
    name: str
    kind: str
    stability: str
    signature: str = ""
    documentation: str = ""
    qualified_name: str = ""
    defined_at: SourceLocation | None = None
    exported_at: SourceLocation | None = None
    members: list[ApiMember] = field(default_factory=list)

    def to_contract(self) -> dict[str, object]:
        snapshot: dict[str, object] = {
            "kind": self.kind,
            "stability": self.stability,
        }
        if self.signature:
            snapshot["signature"] = self.signature
        if self.members:
            snapshot["members"] = {
                member.name: member.to_contract() for member in self.members
            }
        return snapshot

    def to_docs_api_surface(self) -> dict[str, object]:
        snapshot: dict[str, object] = {
            "name": self.name,
            "kind": self.kind,
            "stability": self.stability,
            "documentation": self.documentation,
            "members": [member.to_docs_api_surface() for member in self.members],
        }
        if self.signature:
            snapshot["signature"] = self.signature
        if self.qualified_name:
            snapshot["qualifiedName"] = self.qualified_name
        if self.defined_at is not None:
            snapshot["definedAt"] = self.defined_at.to_docs_api_surface()
        if self.exported_at is not None:
            snapshot["exportedAt"] = self.exported_at.to_docs_api_surface()
        return snapshot


@dataclass(frozen=True)
class ApiModule:
    name: str
    title: str
    description: str
    stability: str
    status: str
    documentation: str
    exports: list[ApiExport]

    def to_contract(self) -> dict[str, object]:
        return {
            "exports": [export.name for export in self.exports],
            "objects": {export.name: export.to_contract() for export in self.exports},
        }

    def to_docs_api_surface(self) -> dict[str, object]:
        return {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "stability": self.stability,
            "status": self.status,
            "documentation": self.documentation,
            "exports": [export.to_docs_api_surface() for export in self.exports],
        }


@dataclass(frozen=True)
class ApiSurface:
    modules: list[ApiModule]
    native_exports: list[ApiExport]

    def to_contract(self) -> dict[str, object]:
        return {module.name: module.to_contract() for module in self.modules}

    def missing_stable_documentation(self) -> list[str]:
        missing: list[str] = []
        for module in self.modules:
            for export in module.exports:
                if export.stability != STABLE:
                    continue
                if export.kind == "module":
                    continue
                if not export.documentation:
                    missing.append(f"{module.name}.{export.name}")
        return missing


def collect_python_api_surface(
    *,
    repo_root: Path | None = None,
    metadata_path: Path | None = None,
) -> ApiSurface:
    root = repo_root or REPO_ROOT
    metadata = _load_metadata(metadata_path or METADATA_PATH)
    collector = _Collector(root, metadata)
    return collector.collect()


def api_surface_contract_json(surface: ApiSurface) -> str:
    return json.dumps(surface.to_contract(), indent=2, sort_keys=True) + "\n"


def docs_api_surface_json(
    surface: ApiSurface,
    *,
    package_version: str,
    metadata: dict[str, Any],
    manifest: dict[str, Any],
) -> str:
    return _json_dumps(
        docs_api_surface_payload(
            surface,
            package_version=package_version,
            metadata=metadata,
            manifest=manifest,
        )
    )


def docs_api_surface_payload(
    surface: ApiSurface,
    *,
    package_version: str,
    metadata: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, object]:
    release = _release_policy(manifest)
    return {
        "schemaVersion": DOCS_API_SURFACE_SCHEMA_VERSION,
        "kind": DOCS_API_SURFACE_KIND,
        "language": "python",
        "release": release,
        "package": {
            "name": metadata.get("package", {}).get("name", "rlmesh"),
            "version": package_version,
        },
        "surface": {
            "modules": [module.to_docs_api_surface() for module in surface.modules],
            "nativeExports": [
                export.to_docs_api_surface() for export in surface.native_exports
            ],
        },
        "features": metadata.get("features", {}),
    }


def write_docs_api_surface(
    *,
    output_dir: Path,
    repo_root: Path,
    manifest_path: Path,
    metadata_path: Path | None,
    package_version: str | None,
    check: bool,
) -> int:
    manifest = _load_manifest(manifest_path)
    python_policy = _python_api_surface_policy(manifest)
    resolved_metadata_path = metadata_path or repo_root / python_policy["metadata"]
    metadata = _load_metadata(resolved_metadata_path)
    version = package_version or _package_version_for_artifact(
        repo_root, manifest, python_policy["package_artifact"]
    )
    surface = collect_python_api_surface(
        repo_root=repo_root, metadata_path=resolved_metadata_path
    )
    snapshot_text = docs_api_surface_json(
        surface,
        package_version=version,
        metadata=metadata,
        manifest=manifest,
    )
    docs_manifest_text = _docs_manifest_json(output_dir, latest=version)
    snapshot_path = output_dir / f"{version}.json"
    docs_manifest_path = output_dir / "manifest.json"

    if check:
        failed = False
        failed |= _check_text(snapshot_path, snapshot_text)
        failed |= _check_text(docs_manifest_path, docs_manifest_text)
        return 1 if failed else 0

    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(snapshot_text, encoding="utf-8")
    docs_manifest_path.write_text(docs_manifest_text, encoding="utf-8")
    return 0


class _Collector:
    def __init__(self, repo_root: Path, metadata: dict[str, Any]) -> None:
        self.repo_root = repo_root
        self.package_root = repo_root / "python" / "rlmesh"
        self.src_root = self.package_root / "src"
        self.metadata = metadata
        self.source_repo = str(metadata.get("source", {}).get("repo", SOURCE_REPO))
        self.source_commit = _git_commit(repo_root)
        self.modules = griffe.ModulesCollection()

    def collect(self) -> ApiSurface:
        modules = [self._collect_module(name) for name in self.metadata["module_order"]]
        native_exports = self._collect_native_exports()
        return ApiSurface(modules=modules, native_exports=native_exports)

    def _collect_module(self, module_name: str) -> ApiModule:
        module = self._load_module(module_name)
        module_metadata = self.metadata["modules"][module_name]
        exports = [
            self._collect_export(module_name, module, name)
            for name in _module_exports(module)
        ]
        return ApiModule(
            name=module_name,
            title=module_metadata["title"],
            description=module_metadata["description"],
            stability=module_metadata["stability"],
            status=module_metadata["status"],
            documentation=_documentation_for(module, module_metadata["description"]),
            exports=exports,
        )

    def _collect_native_exports(self) -> list[ApiExport]:
        module = self._load_module(NATIVE_MODULE)
        descriptions = self.metadata["native"]["descriptions"]
        classes: list[ApiExport] = []
        for name in self.metadata["native"]["classes"]:
            obj = _resolved_object(module.members[name])
            description = descriptions.get(name, "")
            classes.append(
                self._export_from_object(
                    module_name=NATIVE_MODULE,
                    export_name=name,
                    obj=obj,
                    fallback_description=description,
                    stability=STABLE,
                )
            )
        return classes

    def _collect_export(
        self, module_name: str, module: griffe.Module, export_name: str
    ) -> ApiExport:
        exported_obj = module.members[export_name]
        obj = _resolved_object(exported_obj)
        fq_name = f"{module_name}.{export_name}"
        export_metadata = self.metadata.get("exports", {}).get(fq_name, {})
        module_metadata = self.metadata["modules"][module_name]
        stability = export_metadata.get("stability", module_metadata["stability"])
        description = export_metadata.get("description", "")
        return self._export_from_object(
            module_name=module_name,
            export_name=export_name,
            obj=obj,
            fallback_description=description,
            stability=stability,
            exported_at=self._source_location(exported_obj),
        )

    def _export_from_object(
        self,
        *,
        module_name: str,
        export_name: str,
        obj: Any,
        fallback_description: str,
        stability: str,
        exported_at: SourceLocation | None = None,
    ) -> ApiExport:
        kind = _kind_for(obj)
        documentation = _documentation_for(obj, fallback_description)
        if not documentation and kind == "class":
            documentation = _inherited_class_documentation(obj)
        return ApiExport(
            name=export_name,
            kind=kind,
            stability=stability,
            signature=_signature_for(export_name, obj, kind),
            documentation=documentation,
            qualified_name=f"{module_name}.{export_name}",
            defined_at=self._source_location(obj),
            exported_at=_distinct_source_location(
                exported_at, self._source_location(obj)
            ),
            members=_class_members(obj, self._source_location)
            if kind == "class"
            else [],
        )

    def _load_module(self, module_name: str) -> griffe.Module:
        return cast(
            griffe.Module,
            griffe.load(
                module_name,
                search_paths=[self.src_root],
                allow_inspection=False,
                docstring_parser=DOCSTRING_PARSER,
                modules_collection=self.modules,
                resolve_aliases=True,
            ),
        )

    def _source_location(self, obj: Any) -> SourceLocation | None:
        return _source_location(
            obj,
            repo_root=self.repo_root,
            repo=self.source_repo,
            commit=self.source_commit,
        )


def _load_metadata(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _python_api_surface_policy(manifest: dict[str, Any]) -> dict[str, Any]:
    api_surface = manifest.get("api_surface", {})
    if not isinstance(api_surface, dict):
        raise ValueError("missing [api_surface] table in rlmesh manifest")
    python = api_surface.get("python", {})
    if not isinstance(python, dict):
        raise ValueError("missing [api_surface.python] table in rlmesh manifest")
    return python


def _release_policy(manifest: dict[str, Any]) -> dict[str, object]:
    release = manifest.get("release", {})
    api_surface = manifest.get("api_surface", {})
    if not isinstance(release, dict):
        release = {}
    if not isinstance(api_surface, dict):
        api_surface = {}
    return {
        "status": release.get("status", "beta"),
        "packageFamily": release.get("package_family", "0.1"),
        "stablePolicy": api_surface.get("stability_policy", "stable-labels"),
    }


def _package_version_for_artifact(
    repo_root: Path, manifest: dict[str, Any], artifact_id: str
) -> str:
    artifacts = manifest.get("artifact", [])
    if not isinstance(artifacts, list):
        raise ValueError("missing [[artifact]] entries in rlmesh manifest")
    for artifact in artifacts:
        if not isinstance(artifact, dict) or artifact.get("id") != artifact_id:
            continue
        manifest_path = artifact.get("manifest")
        if not isinstance(manifest_path, str):
            raise ValueError(f"{artifact_id}: missing manifest path")
        return _read_project_version(repo_root / manifest_path)
    raise ValueError(f"missing artifact {artifact_id!r} in rlmesh manifest")


def _git_commit(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def _source_location(
    obj: Any, *, repo_root: Path, repo: str, commit: str | None
) -> SourceLocation | None:
    filepath = getattr(obj, "filepath", None)
    if filepath is None:
        return None

    try:
        qualified_name = Path(filepath).resolve().relative_to(repo_root.resolve())
    except ValueError:
        return None

    return SourceLocation(
        repo=repo,
        commit=commit,
        path=qualified_name.as_posix(),
        line=_optional_int(getattr(obj, "lineno", None)),
        end_line=_optional_int(getattr(obj, "endlineno", None)),
    )


def _distinct_source_location(
    location: SourceLocation | None, defined_at: SourceLocation | None
) -> SourceLocation | None:
    if location is None or defined_at is None:
        return location
    if location.to_docs_api_surface() == defined_at.to_docs_api_surface():
        return None
    return location


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _read_project_version(path: Path) -> str:
    in_project = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            in_project = line == "[project]"
            continue
        if in_project and line.startswith("version"):
            _, value = line.split("=", 1)
            version = ast.literal_eval(value.strip())
            if isinstance(version, str) and version:
                return version
    raise ValueError(f"missing [project].version in {path}")


def _docs_manifest_json(output_dir: Path, *, latest: str) -> str:
    versions: list[str] = []
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw_versions = raw_manifest.get("versions", [])
        if isinstance(raw_versions, list):
            versions.extend(str(version) for version in raw_versions)
    if latest not in versions:
        versions.append(latest)
    manifest = {
        "schemaVersion": DOCS_API_SURFACE_SCHEMA_VERSION,
        "kind": DOCS_API_SURFACE_MANIFEST_KIND,
        "language": "python",
        "latest": latest,
        "versions": versions,
    }
    return _json_dumps(manifest)


def _check_text(path: Path, expected: str) -> bool:
    actual = path.read_text(encoding="utf-8") if path.exists() else ""
    if actual == expected:
        return False
    print(f"stale docs API snapshot file: {path}")
    return True


def _json_dumps(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=False) + "\n"


def _module_exports(module: griffe.Module) -> list[str]:
    all_attribute = module.members.get("__all__")
    if all_attribute is None:
        return []
    return list(ast.literal_eval(str(cast(Any, all_attribute).value)))


def _resolved_object(obj: Any) -> Any:
    return getattr(obj, "target", None) or obj


def _kind_for(obj: Any) -> str:
    labels = getattr(obj, "labels", set()) or set()
    if "property" in labels:
        return "property"
    kind = str(getattr(obj, "kind", "")).removeprefix("Kind.").lower()
    if kind == "attribute":
        if "module-attribute" in labels:
            return "type alias"
        return "attribute"
    return kind or "object"


def _signature_for(export_name: str, obj: Any, kind: str) -> str:
    if kind == "class":
        return _class_signature(export_name, obj)
    if kind == "function" and hasattr(obj, "signature"):
        return _normalize_signature(str(obj.signature()))
    if kind == "type alias":
        return _type_alias_signature(export_name, obj)
    if kind == "property":
        return _attribute_signature(export_name, obj)
    return ""


def _class_signature(export_name: str, obj: Any) -> str:
    if hasattr(obj, "signature"):
        signature = str(obj.signature())
        if signature:
            return _normalize_signature(signature)

    init = _find_class_constructor(obj)
    if init is None:
        return ""

    signature = str(init.signature())
    if signature.startswith("__init__"):
        signature = f"{export_name}{signature.removeprefix('__init__')}"
    elif signature.startswith("__new__"):
        signature = f"{export_name}{signature.removeprefix('__new__')}"
    return _normalize_signature(signature)


def _find_class_constructor(obj: Any) -> Any | None:
    candidates = [obj]
    if hasattr(obj, "mro"):
        try:
            candidates.extend(obj.mro())
        except Exception:
            pass

    for cls in candidates:
        members = getattr(cls, "members", {})
        for name in ("__init__", "__new__"):
            constructor = members.get(name)
            if constructor is not None:
                return constructor
    return None


def _type_alias_signature(export_name: str, obj: Any) -> str:
    annotation = getattr(obj, "annotation", None)
    value = getattr(obj, "value", None)
    target = (
        value if str(annotation) == "TypeAlias" and value is not None else annotation
    )
    target = target or value
    if target is None:
        return ""
    return _normalize_signature(f"{export_name} = {target}")


def _attribute_signature(export_name: str, obj: Any) -> str:
    annotation = getattr(obj, "annotation", None)
    if annotation is None:
        return export_name
    if _kind_for(obj) == "property":
        return _normalize_signature(f"property {export_name}: {annotation}")
    return _normalize_signature(f"{export_name}: {annotation}")


def _class_members(obj: Any, source_location: Any | None = None) -> list[ApiMember]:
    if not hasattr(obj, "members"):
        return []

    members: list[ApiMember] = []
    seen: set[str] = set()
    for cls in _class_and_bases(obj):
        for name, member in getattr(cls, "members", {}).items():
            if name in seen or not _is_public_member(name, member):
                continue
            kind = _kind_for(member)
            signature = (
                _normalize_signature(str(member.signature()))
                if kind == "function" and hasattr(member, "signature")
                else _attribute_signature(name, member)
            )
            members.append(
                ApiMember(
                    name=name,
                    kind=kind,
                    signature=signature,
                    documentation=_documentation_for(member),
                    defined_at=source_location(member) if source_location else None,
                )
            )
            seen.add(name)
    return members


def _class_and_bases(obj: Any) -> list[Any]:
    classes = [obj]
    if hasattr(obj, "mro"):
        try:
            classes.extend(obj.mro())
        except Exception:
            pass
    return classes


def _is_public_member(name: str, member: Any) -> bool:
    if name.startswith("_") or name in _IGNORED_MEMBER_NAMES:
        return False
    kind = _kind_for(member)
    return kind in {"attribute", "function", "property"}


def _documentation_for(obj: Any, fallback: str = "") -> str:
    docstring = getattr(obj, "docstring", None)
    if docstring is None:
        return fallback
    rendered = _render_docstring(docstring)
    return rendered or fallback


def _inherited_class_documentation(obj: Any) -> str:
    for cls in _class_and_bases(obj)[1:]:
        rendered = _documentation_for(cls)
        if rendered:
            return rendered
    return ""


def _render_docstring(docstring: griffe.Docstring) -> str:
    try:
        sections = docstring.parse(DOCSTRING_PARSER, warnings=False)
    except Exception:
        return _clean_text(docstring.value)

    lines: list[str] = []
    for section in sections:
        kind = str(section.kind).removeprefix("DocstringSectionKind.")
        value = section.value
        if kind == "text":
            _extend_paragraph(lines, _clean_text(str(value)))
        elif kind in {"parameters", "other_parameters"}:
            _extend_items(lines, "Parameters", value)
        elif kind == "returns":
            _extend_items(lines, "Returns", value)
        elif kind == "raises":
            _extend_items(lines, "Raises", value)
        elif kind == "warns":
            _extend_items(lines, "Warnings", value)
        elif kind == "examples":
            _extend_examples(lines, value)
        else:
            _extend_paragraph(lines, _clean_text(str(value)))
    return "\n".join(lines).strip()


def _extend_items(lines: list[str], title: str, items: Any) -> None:
    if not items:
        return
    _add_blank(lines)
    lines.append(f"**{title}**")
    lines.append("")
    for item in items:
        name = getattr(item, "name", "")
        annotation = getattr(item, "annotation", None)
        description = _inline_text(getattr(item, "description", ""))
        annotation_label = (
            f"`{_normalize_signature(str(annotation))}`" if annotation else ""
        )
        if name and annotation_label:
            label = f"`{name}` ({annotation_label})"
        elif name:
            label = f"`{name}`"
        else:
            label = annotation_label
        if label and description:
            lines.append(f"- {label}: {description}")
        elif label:
            lines.append(f"- {label}")
        elif description:
            lines.append(f"- {description}")


def _extend_examples(lines: list[str], examples: Any) -> None:
    if not examples:
        return
    _add_blank(lines)
    lines.append("**Examples**")
    for _, example in examples:
        _add_blank(lines)
        lines.extend(["```python", textwrap.dedent(str(example)).strip(), "```"])


def _extend_paragraph(lines: list[str], value: str) -> None:
    if not value:
        return
    _add_blank(lines)
    lines.extend(value.splitlines())


def _add_blank(lines: list[str]) -> None:
    if lines and lines[-1] != "":
        lines.append("")


def _clean_text(value: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", value.strip()).replace("``", "`")


def _inline_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).replace("``", "`")


def _normalize_signature(signature: str) -> str:
    return (
        signature.replace("builtins.", "")
        .replace("typing.", "")
        .replace("collections.abc.", "")
        .replace("SequenceType", "Sequence")
        .replace(" = ...", " = ...")
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--print-api-surface",
        action="store_true",
        help="print API surface contract JSON",
    )
    parser.add_argument(
        "--write-api-surface",
        type=Path,
        help="write API surface contract JSON to a file",
    )
    subparsers = parser.add_subparsers(dest="command")
    docs_parser = subparsers.add_parser(
        "docs-api-surface",
        help="write the versioned docs API surface and manifest",
    )
    docs_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="directory containing versioned docs API snapshots",
    )
    docs_parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="RLMesh repository root to extract from",
    )
    docs_parser.add_argument(
        "--manifest",
        type=Path,
        default=RLMESH_MANIFEST_PATH,
        help="RLMesh project policy manifest",
    )
    docs_parser.add_argument(
        "--metadata-path",
        type=Path,
        help="API metadata JSON path; defaults to [api_surface.python].metadata",
    )
    docs_parser.add_argument(
        "--version",
        help="package version override; defaults to python/rlmesh/pyproject.toml",
    )
    docs_parser.add_argument(
        "--check",
        action="store_true",
        help="validate generated files without writing them",
    )
    args = parser.parse_args()

    if args.command == "docs-api-surface":
        return write_docs_api_surface(
            output_dir=args.output_dir,
            repo_root=args.repo_root,
            manifest_path=args.manifest,
            metadata_path=args.metadata_path,
            package_version=args.version,
            check=args.check,
        )

    surface = collect_python_api_surface()
    if args.print_api_surface:
        print(api_surface_contract_json(surface), end="")
        return 0
    if args.write_api_surface is not None:
        args.write_api_surface.write_text(
            api_surface_contract_json(surface), encoding="utf-8"
        )
        return 0

    missing = surface.missing_stable_documentation()
    if missing:
        for name in missing:
            print(f"missing stable API documentation: {name}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

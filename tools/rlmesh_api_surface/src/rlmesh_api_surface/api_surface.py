"""Static public API extraction for RLMesh Python docs and snapshots."""

from __future__ import annotations

import argparse
import ast
import json
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import griffe

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[3]
NATIVE_MODULE = "rlmesh._rlmesh"
METADATA_PATH = PACKAGE_ROOT / "api_metadata.json"
DOCSTRING_PARSER = "google"

STABLE = "Stable"
INTERNAL = "Internal"
_IGNORED_MEMBER_NAMES = {"__enter__", "__exit__", "__repr__", "__del__"}


@dataclass(frozen=True)
class ApiMember:
    name: str
    kind: str
    signature: str
    documentation: str = ""

    def to_snapshot(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "signature": self.signature,
        }


@dataclass(frozen=True)
class ApiExport:
    name: str
    kind: str
    stability: str
    signature: str = ""
    documentation: str = ""
    source_path: str = ""
    members: list[ApiMember] = field(default_factory=list)

    def to_snapshot(self) -> dict[str, object]:
        snapshot: dict[str, object] = {
            "kind": self.kind,
            "stability": self.stability,
        }
        if self.signature:
            snapshot["signature"] = self.signature
        if self.members:
            snapshot["members"] = {
                member.name: member.to_snapshot() for member in self.members
            }
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

    def to_snapshot(self) -> dict[str, object]:
        return {
            "exports": [export.name for export in self.exports],
            "objects": {export.name: export.to_snapshot() for export in self.exports},
        }


@dataclass(frozen=True)
class ApiSurface:
    modules: list[ApiModule]
    native_classes: list[ApiExport]

    def to_snapshot(self) -> dict[str, object]:
        return {module.name: module.to_snapshot() for module in self.modules}

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


def snapshot_json(surface: ApiSurface) -> str:
    return json.dumps(surface.to_snapshot(), indent=2, sort_keys=True) + "\n"


class _Collector:
    def __init__(self, repo_root: Path, metadata: dict[str, Any]) -> None:
        self.repo_root = repo_root
        self.package_root = repo_root / "python" / "rlmesh"
        self.src_root = self.package_root / "src"
        self.metadata = metadata
        self.modules = griffe.ModulesCollection()

    def collect(self) -> ApiSurface:
        modules = [self._collect_module(name) for name in self.metadata["module_order"]]
        native_classes = self._collect_native_classes()
        return ApiSurface(modules=modules, native_classes=native_classes)

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

    def _collect_native_classes(self) -> list[ApiExport]:
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
        obj = _resolved_object(module.members[export_name])
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
        )

    def _export_from_object(
        self,
        *,
        module_name: str,
        export_name: str,
        obj: Any,
        fallback_description: str,
        stability: str,
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
            source_path=str(getattr(obj, "path", "")),
            members=_class_members(obj) if kind == "class" else [],
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


def _load_metadata(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _class_members(obj: Any) -> list[ApiMember]:
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
    parser.add_argument("--snapshot", action="store_true", help="print snapshot JSON")
    parser.add_argument(
        "--write-snapshot", type=Path, help="write snapshot JSON to a file"
    )
    args = parser.parse_args()

    surface = collect_python_api_surface()
    if args.snapshot:
        print(snapshot_json(surface), end="")
        return 0
    if args.write_snapshot is not None:
        args.write_snapshot.write_text(snapshot_json(surface), encoding="utf-8")
        return 0

    missing = surface.missing_stable_documentation()
    if missing:
        for name in missing:
            print(f"missing stable API documentation: {name}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

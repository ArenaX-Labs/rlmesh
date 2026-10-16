"""Public API extraction for the Rust `rlmesh` facade crate, from rustdoc JSON.

Only the facade is documented: the other workspace crates are internal, so an
item they define appears only where the facade re-exports it. rustdoc JSON does
not inline cross-crate re-exports, so each workspace crate the facade re-exports
from is documented to JSON too and its items are resolved by path. Third-party
re-exports (`async_trait`, `CancellationToken`) become stubs pointing at docs.rs.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rlmesh_api_surface.api_surface import (
    ApiExport,
    ApiMember,
    ApiModule,
    SourceLocation,
)

# rustdoc's JSON output is unstable and versioned; a toolchain bump that moves
# it fails here with a clear message instead of producing a wrong reference.
RUSTDOC_FORMAT_VERSION = 61

# Item kinds that are documented as pages, by their rustdoc JSON key.
_EXPORT_KINDS = {
    "struct": "struct",
    "enum": "enum",
    "union": "union",
    "trait": "trait",
    "trait_alias": "trait alias",
    "function": "function",
    "constant": "constant",
    "static": "static",
    "type_alias": "type alias",
    "macro": "macro",
    "proc_macro": "macro",
}


@dataclass(frozen=True)
class RustSurfaceConfig:
    crate: str
    stability: str
    skipped_modules: tuple[str, ...]
    source_repo: str
    source_commit: str | None
    document_hidden: bool = False


def collect_rust_api_surface(
    *,
    repo_root: Path,
    config: RustSurfaceConfig,
    json_dir: Path | None = None,
) -> list[ApiModule]:
    """Document the facade crate (building rustdoc JSON unless `json_dir` holds it)."""
    loader = _CrateLoader(repo_root, json_dir, document_hidden=config.document_hidden)
    facade = loader.load(config.crate)
    return _FacadeWalker(loader, facade, config).collect()


class _Crate:
    def __init__(self, data: dict[str, Any]) -> None:
        version = data.get("format_version")
        if version != RUSTDOC_FORMAT_VERSION:
            raise ValueError(
                f"rustdoc JSON format {version}, expected {RUSTDOC_FORMAT_VERSION}: "
                "update rust_surface.py for the new toolchain's format"
            )
        self.index: dict[str, Any] = data["index"]
        self.paths: dict[str, Any] = data["paths"]
        self.external: dict[str, Any] = data["external_crates"]
        self.root = self.index[str(data["root"])]
        self.name: str = self.root["name"]
        self._by_path = {
            tuple(summary["path"]): item_id
            for item_id, summary in self.paths.items()
            if summary["crate_id"] == 0
        }

    def item(self, item_id: object) -> dict[str, Any] | None:
        return self.index.get(str(item_id))

    def item_by_path(self, path: tuple[str, ...]) -> dict[str, Any] | None:
        item_id = self._by_path.get(path)
        return None if item_id is None else self.item(item_id)

    def external_target(self, item_id: object) -> tuple[str, tuple[str, ...]] | None:
        summary = self.paths.get(str(item_id))
        if summary is None or summary["crate_id"] == 0:
            return None
        crate = self.external[str(summary["crate_id"])]["name"]
        return crate, tuple(summary["path"])


class _CrateLoader:
    def __init__(
        self, repo_root: Path, json_dir: Path | None, *, document_hidden: bool = False
    ) -> None:
        self.repo_root = repo_root
        self.document_hidden = document_hidden
        self.json_dir = json_dir
        self.workspace, self.target_dir = _cargo_workspace(repo_root)
        self._loaded: dict[str, _Crate] = {}

    def is_workspace(self, crate: str) -> bool:
        return crate.replace("_", "-") in self.workspace

    def load(self, crate: str) -> _Crate:
        key = crate.replace("-", "_")
        if key not in self._loaded:
            self._loaded[key] = _Crate(self._read_json(key))
        return self._loaded[key]

    def _read_json(self, crate: str) -> dict[str, Any]:
        if self.json_dir is not None:
            path = self.json_dir / f"{crate}.json"
        else:
            path = self._build_json(crate)
        return json.loads(path.read_text(encoding="utf-8"))

    def _build_json(self, crate: str) -> Path:
        package = crate.replace("_", "-")
        env = {**os.environ, "RUSTC_BOOTSTRAP": "1"}
        subprocess.run(
            [
                "cargo",
                "rustdoc",
                "--quiet",
                "--package",
                package,
                "--lib",
                "--",
                "-Z",
                "unstable-options",
                "--output-format",
                "json",
                *(["--document-hidden-items"] if self.document_hidden else []),
            ],
            cwd=self.repo_root,
            env=env,
            check=True,
        )
        return self.target_dir / "doc" / f"{crate}.json"


def _cargo_workspace(repo_root: Path) -> tuple[set[str], Path]:
    """The workspace's package names and its target dir (where rustdoc writes `doc/`)."""
    metadata = json.loads(
        subprocess.run(
            ["cargo", "metadata", "--format-version", "1", "--no-deps"],
            cwd=repo_root,
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        ).stdout
    )
    names = {package["name"] for package in metadata["packages"]}
    return names, Path(metadata["target_directory"])


@dataclass
class _Placement:
    crate: _Crate | None
    item: dict[str, Any] | None
    name: str
    paths: list[str]
    use_item: dict[str, Any] | None
    external: tuple[str, tuple[str, ...]] | None = None


class _FacadeWalker:
    def __init__(
        self, loader: _CrateLoader, facade: _Crate, config: RustSurfaceConfig
    ) -> None:
        self.loader = loader
        self.facade = facade
        self.config = config
        self.module_order: list[tuple[str, _Crate, dict[str, Any]]] = []
        self.placements: dict[tuple[str, str], _Placement] = {}
        self._walked: set[tuple[str, str, str]] = set()

    def collect(self) -> list[ApiModule]:
        self._walk(self.facade.name, self.facade, self.facade.root)
        by_module: dict[str, list[_Placement]] = {
            path: [] for path, _, _ in self.module_order
        }
        for placement in self.placements.values():
            canonical = _canonical_path(placement.paths)
            module_path = canonical.rsplit("::", 1)[0]
            by_module[module_path].append(placement)
        modules = []
        for path, crate, module in self.module_order:
            exports = [self._export(p) for p in by_module[path]]
            exports.sort(
                key=lambda export: (_KIND_ORDER.get(export.kind, 99), export.name)
            )
            docs = _rust_docs(module, crate)
            modules.append(
                ApiModule(
                    name=path,
                    title=path,
                    description=_first_sentence(docs),
                    stability=self.config.stability,
                    status=self.config.stability.lower(),
                    documentation=docs,
                    exports=exports,
                )
            )
        return modules

    def _walk(self, path: str, crate: _Crate, module: dict[str, Any]) -> None:
        key = (path, crate.name, str(module["id"]))
        if key in self._walked:
            return
        self._walked.add(key)
        if not any(existing == path for existing, _, _ in self.module_order):
            self.module_order.append((path, crate, module))
        for item_id in module["inner"]["module"]["items"]:
            item = crate.item(item_id)
            if item is None or item["visibility"] != "public":
                continue
            kind = _kind(item)
            if kind == "module":
                if item["name"] not in self.config.skipped_modules:
                    self._walk(f"{path}::{item['name']}", crate, item)
            elif kind == "use":
                self._use(path, crate, item)
            elif kind in _EXPORT_KINDS:
                self._place(path, crate, item, item["name"], use_item=None)

    def _use(self, path: str, crate: _Crate, use: dict[str, Any]) -> None:
        inner = use["inner"]["use"]
        if inner.get("id") is None:
            return
        target_crate, target = self._resolve(crate, inner["id"])
        if inner["is_glob"]:
            if target is not None and _kind(target) == "module":
                self._walk(path, target_crate, target)
            return
        if target is None:
            external = crate.external_target(inner["id"])
            if external is None:
                return
            key = (external[0], "::".join(external[1]))
            placement = self.placements.setdefault(
                key, _Placement(None, None, inner["name"], [], use, external)
            )
            placement.paths.append(f"{path}::{inner['name']}")
            return
        if _kind(target) == "module":
            if inner["name"] not in self.config.skipped_modules:
                self._walk(f"{path}::{inner['name']}", target_crate, target)
            return
        if _kind(target) in _EXPORT_KINDS:
            self._place(path, target_crate, target, inner["name"], use_item=use)

    def _resolve(
        self, crate: _Crate, item_id: object
    ) -> tuple[_Crate, dict[str, Any] | None]:
        local = crate.item(item_id)
        if local is not None:
            return crate, local
        external = crate.external_target(item_id)
        if external is None or not self.loader.is_workspace(external[0]):
            return crate, None
        other = self.loader.load(external[0])
        return other, other.item_by_path(external[1])

    def _place(
        self,
        path: str,
        crate: _Crate,
        item: dict[str, Any],
        name: str,
        *,
        use_item: dict[str, Any] | None,
    ) -> None:
        key = (crate.name, str(item["id"]))
        placement = self.placements.setdefault(
            key, _Placement(crate, item, name, [], use_item)
        )
        placement.paths.append(f"{path}::{name}")

    def _export(self, placement: _Placement) -> ApiExport:
        canonical = _canonical_path(placement.paths)
        aliases = sorted(p for p in placement.paths if p != canonical)
        if placement.crate is None or placement.item is None:
            return self._external_export(placement, canonical, aliases)
        crate, item = placement.crate, placement.item
        kind = _EXPORT_KINDS[_kind(item)]
        docs = _rust_docs(item, crate)
        defined_at = self._location(item)
        exported_at = (
            self._location(placement.use_item)
            if placement.use_item is not None
            else None
        )
        signature = _item_signature(item, placement.name, crate)
        if kind == "constant" and not self._value_in_source(item):
            # A value computed at build time (`env!`, `concat!`) varies per build;
            # the snapshot keeps only the type.
            signature = signature.split(" = ", 1)[0]
        return ApiExport(
            name=placement.name,
            kind=kind,
            stability=self.config.stability,
            signature=signature,
            documentation=_with_aliases(docs, aliases),
            qualified_name=canonical,
            aliases=aliases,
            defined_at=defined_at,
            exported_at=exported_at if exported_at != defined_at else None,
            members=self._members(crate, item),
        )

    def _external_export(
        self, placement: _Placement, canonical: str, aliases: list[str]
    ) -> ApiExport:
        assert placement.external is not None
        crate, path = placement.external
        url = f"https://docs.rs/{crate.replace('_', '-')}/latest/{'/'.join(path[:-1])}/"
        docs = (
            f"Re-exported from the `{crate}` crate as `{'::'.join(path)}`; "
            f"see its documentation on [docs.rs]({url})."
        )
        return ApiExport(
            name=placement.name,
            kind="re-export",
            stability=self.config.stability,
            signature=f"pub use {'::'.join(path)};",
            documentation=_with_aliases(docs, aliases),
            qualified_name=canonical,
            aliases=aliases,
            exported_at=self._location(placement.use_item)
            if placement.use_item is not None
            else None,
        )

    def _members(self, crate: _Crate, item: dict[str, Any]) -> list[ApiMember]:
        kind = _kind(item)
        inner = item["inner"][kind]
        members: list[ApiMember] = []
        if kind in ("struct", "union"):
            # Only named fields are listed; a tuple struct's fields read as its signature.
            if kind == "union":
                fields = inner.get("fields", [])
            elif isinstance(inner["kind"], dict) and "plain" in inner["kind"]:
                fields = inner["kind"]["plain"]["fields"]
            else:
                fields = []
            for field_id in fields:
                field = crate.item(field_id)
                if field is None or field["visibility"] != "public":
                    continue
                members.append(
                    self._member(
                        crate,
                        field,
                        "field",
                        f"{field['name']}: {_ty(field['inner']['struct_field'])}",
                    )
                )
        if kind == "enum":
            for variant_id in inner["variants"]:
                variant = crate.item(variant_id)
                if variant is not None:
                    members.append(
                        self._member(
                            crate,
                            variant,
                            "variant",
                            _variant_signature(crate, variant),
                        )
                    )
        if kind == "trait":
            for assoc_id in inner["items"]:
                assoc = crate.item(assoc_id)
                if assoc is not None:
                    members.append(self._assoc_member(crate, assoc, in_trait=True))
        for impl_id in inner.get("impls", []):
            impl = crate.item(impl_id)
            if impl is None:
                continue
            body = impl["inner"]["impl"]
            if body["is_synthetic"] or body["blanket_impl"] is not None:
                continue
            if body["trait"] is None:
                for assoc_id in body["items"]:
                    assoc = crate.item(assoc_id)
                    if assoc is not None and assoc["visibility"] == "public":
                        members.append(self._assoc_member(crate, assoc, in_trait=False))
            else:
                members.append(
                    ApiMember(
                        name=_path(body["trait"]),
                        kind="trait implementation",
                        signature=_impl_signature(body),
                    )
                )
        return members

    def _assoc_member(
        self, crate: _Crate, assoc: dict[str, Any], *, in_trait: bool
    ) -> ApiMember:
        kind = _kind(assoc)
        if kind == "function":
            function = assoc["inner"]["function"]
            takes_self = (
                bool(function["sig"]["inputs"])
                and function["sig"]["inputs"][0][0] == "self"
            )
            if in_trait:
                member_kind = (
                    "provided method" if function["has_body"] else "required method"
                )
            else:
                member_kind = "method" if takes_self else "associated function"
            signature = _function_signature(assoc["name"], function)
        elif kind == "assoc_type":
            member_kind = "associated type"
            signature = _assoc_type_signature(assoc)
        else:
            member_kind = "associated constant"
            inner = assoc["inner"].get(kind, {})
            signature = f"const {assoc['name']}: {_ty(inner.get('type'))}"
        return self._member(crate, assoc, member_kind, signature)

    def _member(
        self, crate: _Crate, item: dict[str, Any], kind: str, signature: str
    ) -> ApiMember:
        return ApiMember(
            name=item["name"] or "",
            kind=kind,
            signature=signature,
            documentation=_rust_docs(item, crate),
            defined_at=self._location(item),
        )

    def _value_in_source(self, item: dict[str, Any]) -> bool:
        expr = item["inner"]["constant"].get("const", {}).get("expr")
        location = self._location(item)
        if not expr or location is None:
            return False
        lines = (
            (self.loader.repo_root / location.path)
            .read_text(encoding="utf-8")
            .splitlines()
        )
        start, end = (location.line or 1) - 1, location.end_line or location.line or 1
        return expr in "\n".join(lines[start:end])

    def _location(self, item: dict[str, Any]) -> SourceLocation | None:
        span = item.get("span")
        if not span:
            return None
        filename = Path(span["filename"])
        if filename.is_absolute():
            try:
                filename = filename.resolve().relative_to(
                    self.loader.repo_root.resolve()
                )
            except ValueError:
                return None
        return SourceLocation(
            repo=self.config.source_repo,
            commit=self.config.source_commit,
            path=filename.as_posix(),
            line=span["begin"][0],
            end_line=span["end"][0],
        )


_KIND_ORDER = {
    "struct": 0,
    "enum": 1,
    "union": 2,
    "trait": 3,
    "trait alias": 4,
    "type alias": 5,
    "function": 6,
    "constant": 7,
    "static": 8,
    "macro": 9,
    "re-export": 10,
}


def _canonical_path(paths: list[str]) -> str:
    """An item's home: its shallowest facade path below the crate root, else the root.

    `rlmesh::RemoteEnv` and `rlmesh::env::RemoteEnv` document under `rlmesh::env`;
    `rlmesh::spaces::SpaceSpec` beats the module-wide `rlmesh::spaces::types::SpaceSpec`.
    First seen wins a tie.
    """
    # The prelude is a convenience alias, never an item's canonical home.
    paths = [
        path for path in paths if not path.startswith("rlmesh::prelude::")
    ] or paths
    nested = [path for path in paths if path.count("::") > 1]
    return min(nested or paths, key=lambda path: path.count("::"))


def _with_aliases(docs: str, aliases: list[str]) -> str:
    if not aliases:
        return docs
    also = ", ".join(f"`{alias}`" for alias in aliases)
    note = f"Also available as {also}."
    return f"{docs}\n\n{note}" if docs else note


def _kind(item: dict[str, Any]) -> str:
    return next(iter(item["inner"]))


# ---- documentation ---------------------------------------------------------

_LINK = re.compile(r"\[([^\[\]]+)\](\(([^()]*)\)|\[([^\[\]]*)\])?")
_FENCE = re.compile(r"^(\s*)```(.*)$")


def _rust_docs(item: dict[str, Any], crate: _Crate) -> str:
    """Item docs as plain Markdown: intra-doc links unwrapped, doctests tidied."""
    docs = item.get("docs") or ""
    links: dict[str, Any] = item.get("links") or {}
    return _tidy_doctests(_unwrap_intra_doc_links(docs, links)).strip()


def _unwrap_intra_doc_links(docs: str, links: dict[str, Any]) -> str:
    # rustdoc resolves [`Foo`] / [Foo](crate::Foo) against `links`; the site has
    # no rustdoc link targets, so a resolved link keeps its text and drops the
    # target. URLs and unresolved brackets are left alone.
    def replace(match: re.Match[str]) -> str:
        text, target = match.group(1), match.group(3) or match.group(4)
        if target is not None and re.match(r"^[a-z]+://", target):
            return match.group(0)
        key = target or text
        return text if key in links or key.strip("`") in links else match.group(0)

    out_lines = []
    in_code = False
    for line in docs.splitlines():
        if _FENCE.match(line):
            in_code = not in_code
            out_lines.append(line)
            continue
        out_lines.append(line if in_code else _LINK.sub(replace, line))
    return "\n".join(out_lines)


def _tidy_doctests(docs: str) -> str:
    # A bare ``` fence is a Rust doctest; its `# ` lines are hidden setup.
    out_lines: list[str] = []
    rust_block = False
    in_block = False
    for line in docs.splitlines():
        fence = _FENCE.match(line)
        if fence:
            if not in_block:
                info = fence.group(2).strip()
                tags = {tag.strip() for tag in info.split(",") if tag.strip()}
                rust_block = not tags or bool(
                    tags
                    & {
                        "rust",
                        "no_run",
                        "ignore",
                        "should_panic",
                        "compile_fail",
                        "edition2021",
                        "edition2024",
                    }
                )
                in_block = True
                out_lines.append(f"{fence.group(1)}```{'rust' if rust_block else info}")
            else:
                in_block = False
                out_lines.append(line)
            continue
        if in_block and rust_block:
            stripped = line.strip()
            if stripped == "#" or stripped.startswith("# "):
                continue
            if stripped.startswith("##"):
                line = line.replace("##", "#", 1)
        out_lines.append(line)
    return "\n".join(out_lines)


def _first_sentence(docs: str) -> str:
    paragraph = docs.split("\n\n", 1)[0].replace("\n", " ").strip()
    match = re.match(r"(.+?[.!?])(\s|$)", paragraph)
    return match.group(1) if match else paragraph


# ---- signatures ------------------------------------------------------------


def _item_signature(item: dict[str, Any], name: str, crate: _Crate) -> str:
    kind = _kind(item)
    inner = item["inner"][kind]
    if kind == "function":
        return _function_signature(name, inner)
    if (
        kind == "struct"
        and isinstance(inner["kind"], dict)
        and "tuple" in inner["kind"]
    ):
        # A tuple struct's fields are its constructor, so they belong in the
        # signature; rustdoc nulls out the ones it stripped as private.
        params, where = _generics(inner.get("generics"))
        fields = []
        for field_id in inner["kind"]["tuple"]:
            field = crate.item(field_id) if field_id is not None else None
            if field is None or field["visibility"] != "public":
                fields.append("/* private field */")
            else:
                fields.append(f"pub {_ty(field['inner']['struct_field'])}")
        return f"struct {name}{params}({', '.join(fields)}){where}"
    if kind in ("struct", "enum", "union"):
        params, where = _generics(inner.get("generics"))
        return f"{kind} {name}{params}{where}"
    if kind == "trait":
        params, where = _generics(inner.get("generics"))
        bounds = _bounds(inner.get("bounds", []))
        unsafe = "unsafe " if inner.get("is_unsafe") else ""
        return f"{unsafe}trait {name}{params}{': ' + bounds if bounds else ''}{where}"
    if kind == "type_alias":
        params, where = _generics(inner.get("generics"))
        return f"type {name}{params}{where} = {_ty(inner['type'])}"
    if kind == "constant":
        value = inner.get("const", {})
        literal = f" = {value['expr']}" if value.get("is_literal") else ""
        return f"const {name}: {_ty(inner['type'])}{literal}"
    if kind == "static":
        mutable = "mut " if inner.get("is_mutable") else ""
        return f"static {mutable}{name}: {_ty(inner['type'])}"
    if kind in ("macro", "proc_macro"):
        return f"macro {name}"
    return name


def _function_signature(name: str, function: dict[str, Any]) -> str:
    header = function["header"]
    prefix = "".join(
        word
        for flag, word in (
            ("is_const", "const "),
            ("is_async", "async "),
            ("is_unsafe", "unsafe "),
        )
        if header.get(flag)
    )
    abi = header.get("abi")
    if abi and abi != "Rust":
        abi_name = abi if isinstance(abi, str) else next(iter(abi))
        prefix += f'extern "{abi_name}" '
    params, where = _generics(function["generics"])
    inputs = ", ".join(
        _input(arg_name, ty) for arg_name, ty in function["sig"]["inputs"]
    )
    output = function["sig"].get("output")
    returns = f" -> {_ty(output)}" if output is not None else ""
    return f"{prefix}fn {name}{params}({inputs}){returns}{where}"


def _input(name: str, ty: dict[str, Any]) -> str:
    if name == "self":
        if ty == {"generic": "Self"}:
            return "self"
        ref = ty.get("borrowed_ref")
        if ref is not None and ref["type"] == {"generic": "Self"}:
            lifetime = f"{ref['lifetime']} " if ref.get("lifetime") else ""
            return f"&{lifetime}{'mut ' if ref['is_mutable'] else ''}self"
        return f"self: {_ty(ty)}"
    return f"{name}: {_ty(ty)}"


def _generics(generics: dict[str, Any] | None) -> tuple[str, str]:
    if not generics:
        return "", ""
    params = []
    for param in generics.get("params", []):
        kind = param["kind"]
        if "lifetime" in kind:
            outlives = kind["lifetime"].get("outlives") or []
            params.append(
                param["name"] + (": " + " + ".join(outlives) if outlives else "")
            )
        elif "type" in kind:
            if kind["type"].get("is_synthetic"):
                continue
            bounds = _bounds(kind["type"].get("bounds", []))
            default = kind["type"].get("default")
            params.append(
                param["name"]
                + (f": {bounds}" if bounds else "")
                + (f" = {_ty(default)}" if default else "")
            )
        elif "const" in kind:
            params.append(f"const {param['name']}: {_ty(kind['const']['type'])}")
    predicates = []
    for predicate in generics.get("where_predicates", []):
        if "bound_predicate" in predicate:
            bound = predicate["bound_predicate"]
            # `for<'a> &'a T: Trait` carries its binder on the predicate itself.
            hrtb = _hrtb(bound.get("generic_params", []))
            predicates.append(f"{hrtb}{_ty(bound['type'])}: {_bounds(bound['bounds'])}")
        elif "lifetime_predicate" in predicate:
            bound = predicate["lifetime_predicate"]
            predicates.append(f"{bound['lifetime']}: {' + '.join(bound['outlives'])}")
        elif "eq_predicate" in predicate:
            bound = predicate["eq_predicate"]
            rhs = bound["rhs"].get("type", bound["rhs"])
            predicates.append(f"{_ty(bound['lhs'])} = {_ty(rhs)}")
    return (
        f"<{', '.join(params)}>" if params else "",
        f" where {', '.join(predicates)}" if predicates else "",
    )


def _bounds(bounds: list[dict[str, Any]]) -> str:
    rendered = []
    for bound in bounds:
        if "trait_bound" in bound:
            trait_bound = bound["trait_bound"]
            modifier = {"maybe": "?", "maybe_const": "~const "}.get(
                trait_bound.get("modifier", "none"), ""
            )
            hrtb = _hrtb(trait_bound.get("generic_params", []))
            rendered.append(f"{hrtb}{modifier}{_path(trait_bound['trait'])}")
        elif "outlives" in bound:
            rendered.append(bound["outlives"])
        elif "use" in bound:
            rendered.append(f"use<{', '.join(str(arg) for arg in bound['use'])}>")
    return " + ".join(rendered)


def _hrtb(params: list[dict[str, Any]]) -> str:
    names = [param["name"] for param in params]
    return f"for<{', '.join(names)}> " if names else ""


def _path(path: dict[str, Any]) -> str:
    # A path written relative to its defining crate (`crate::dtype::DType`,
    # `super::Metric`) means nothing at the facade, so it renders as its name.
    name = path["path"]
    if name.split("::", 1)[0] in ("crate", "super", "self"):
        name = name.rsplit("::", 1)[-1]
    return name + _generic_args(path.get("args"))


def _generic_args(args: dict[str, Any] | None) -> str:
    if not args:
        return ""
    if "angle_bracketed" in args:
        body = args["angle_bracketed"]
        rendered = []
        for arg in body.get("args", []):
            if arg == "infer":
                rendered.append("_")
            elif "type" in arg:
                rendered.append(_ty(arg["type"]))
            elif "lifetime" in arg:
                rendered.append(arg["lifetime"])
            elif "const" in arg:
                rendered.append(arg["const"].get("expr", "_"))
        for constraint in body.get("constraints", []):
            binding = constraint["binding"]
            name = constraint["name"] + _generic_args(constraint.get("args"))
            if "equality" in binding:
                value = binding["equality"]
                rendered.append(f"{name} = {_ty(value.get('type', value))}")
            else:
                rendered.append(f"{name}: {_bounds(binding['constraint'])}")
        return f"<{', '.join(rendered)}>" if rendered else ""
    if "parenthesized" in args:
        body = args["parenthesized"]
        inputs = ", ".join(_ty(ty) for ty in body["inputs"])
        output = body.get("output")
        return f"({inputs})" + (f" -> {_ty(output)}" if output else "")
    return ""


def _ty(ty: Any) -> str:
    if ty is None:
        return "()"
    if ty == "infer":
        return "_"
    ((kind, value),) = ty.items()
    if kind == "resolved_path":
        return _path(value)
    if kind == "generic":
        return value
    if kind == "primitive":
        return value
    if kind == "tuple":
        if len(value) == 1:
            return f"({_ty(value[0])},)"
        return f"({', '.join(_ty(item) for item in value)})"
    if kind == "slice":
        return f"[{_ty(value)}]"
    if kind == "array":
        return f"[{_ty(value['type'])}; {value['len']}]"
    if kind == "borrowed_ref":
        lifetime = f"{value['lifetime']} " if value.get("lifetime") else ""
        mutable = "mut " if value["is_mutable"] else ""
        return f"&{lifetime}{mutable}{_ty(value['type'])}"
    if kind == "raw_pointer":
        return f"*{'mut' if value['is_mutable'] else 'const'} {_ty(value['type'])}"
    if kind == "impl_trait":
        return f"impl {_bounds(value)}"
    if kind == "dyn_trait":
        traits = [
            _hrtb(poly.get("generic_params", [])) + _path(poly["trait"])
            for poly in value["traits"]
        ]
        if value.get("lifetime"):
            traits.append(value["lifetime"])
        return f"dyn {' + '.join(traits)}"
    if kind == "function_pointer":
        sig = value["sig"]
        inputs = ", ".join(_ty(arg) for _, arg in sig["inputs"])
        output = sig.get("output")
        unsafe = "unsafe " if value["header"].get("is_unsafe") else ""
        return f"{_hrtb(value.get('generic_params', []))}{unsafe}fn({inputs})" + (
            f" -> {_ty(output)}" if output else ""
        )
    if kind == "qualified_path":
        trait = value.get("trait")
        self_ty = _ty(value["self_type"])
        base = f"<{self_ty} as {_path(trait)}>" if trait else self_ty
        return f"{base}::{value['name']}{_generic_args(value.get('args'))}"
    if kind == "pat":
        return _ty(value["type"])
    return kind


def _variant_signature(crate: _Crate, variant: dict[str, Any]) -> str:
    inner = variant["inner"]["variant"]
    kind = inner["kind"]
    name = variant["name"]
    discriminant = inner.get("discriminant")
    suffix = f" = {discriminant['expr']}" if discriminant else ""
    if kind == "plain":
        return name + suffix
    if "tuple" in kind:
        fields = [
            crate.item(field_id) for field_id in kind["tuple"] if field_id is not None
        ]
        return (
            f"{name}({', '.join(_ty(f['inner']['struct_field']) for f in fields if f)})"
        )
    fields = [crate.item(field_id) for field_id in kind["struct"]["fields"]]
    body = ", ".join(
        f"{f['name']}: {_ty(f['inner']['struct_field'])}"
        for f in fields
        if f is not None
    )
    return f"{name} {{ {body} }}"


def _assoc_type_signature(assoc: dict[str, Any]) -> str:
    inner = assoc["inner"]["assoc_type"]
    bounds = _bounds(inner.get("bounds", []))
    default = inner.get("type")
    return (
        f"type {assoc['name']}"
        + (f": {bounds}" if bounds else "")
        + (f" = {_ty(default)}" if default else "")
    )


def _impl_signature(impl: dict[str, Any]) -> str:
    params, where = _generics(impl["generics"])
    negative = "!" if impl.get("is_negative") else ""
    unsafe = "unsafe " if impl.get("is_unsafe") else ""
    return f"{unsafe}impl{params} {negative}{_path(impl['trait'])} for {_ty(impl['for'])}{where}"

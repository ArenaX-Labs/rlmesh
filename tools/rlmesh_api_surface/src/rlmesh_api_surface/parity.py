"""Reviewed language mapping and static parity checks (E-1 and E-2).

The manifest is source, never refreshed by the check. Extraction uses the same
rustdoc/libclang readers as the native reference. Rust defaults and runtime
error inheritance belong to the behavioural fixtures, not rustdoc signatures.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import tomllib

from rlmesh_api_surface.api_surface import REPO_ROOT, _collect_native_modules
from rlmesh_api_surface.header_surface import HeaderSurfaceConfig, _parse

LANGUAGES = ("rust", "c", "cpp")
RESERVED = {
    "as",
    "async",
    "await",
    "box",
    "break",
    "const",
    "continue",
    "crate",
    "default",
    "delete",
    "do",
    "dyn",
    "else",
    "enum",
    "export",
    "extern",
    "false",
    "final",
    "fn",
    "for",
    "if",
    "impl",
    "in",
    "let",
    "loop",
    "match",
    "mod",
    "move",
    "mut",
    "new",
    "override",
    "priv",
    "pub",
    "ref",
    "return",
    "self",
    "sizeof",
    "static",
    "struct",
    "super",
    "trait",
    "true",
    "try",
    "type",
    "typeof",
    "union",
    "unsafe",
    "unsized",
    "use",
    "virtual",
    "where",
    "while",
    "yield",
}
# Union of Rust's strict/reserved/weak keywords and C11/C++20 keywords.
# A field keeps one spelling even when only one native language reserves it.
RESERVED.update(
    """
Self abstract alignas alignof and and_eq asm atomic_cancel atomic_commit atomic_noexcept
 auto become bitand bitor bool case catch char char8_t char16_t char32_t class compl
 concept consteval constexpr constinit const_cast co_await co_return co_yield decltype
 double dynamic_cast explicit float friend gen goto inline int long macro macro_rules
 mutable namespace noexcept not not_eq nullptr offsetof operator or or_eq private
 protected public raw register reinterpret_cast requires safe short signed static_assert
 static_cast switch synchronized template this thread_local throw typedef typeid typename
 unsigned using void volatile wchar_t xor xor_eq _Alignas _Alignof _Atomic _Bool
 _Complex _Generic _Imaginary _Noreturn _Static_assert _Thread_local
""".split()
)

ERROR_BASES = {
    "RLMeshException": "RuntimeError",
    "ProtocolException": "RLMeshException",
    "EnvironmentException": "RLMeshException",
    "RecoverableEnvironmentException": "EnvironmentException",
    "ModelException": "RLMeshException",
    "ValueError": "std::invalid_argument",
    "AdapterResolutionError": "ValueError",
    "TypeError": "std::logic_error",
    "RuntimeError": "std::runtime_error",
    "OSError": "std::runtime_error",
    "ConnectionError": "OSError",
    "TimeoutError": "OSError",
    "KeyboardInterrupt": "",
}
TOLERANT = {
    "RLMeshResample",
    "RLMeshCropMode",
    "RLMeshChannelOrder",
    "RLMeshFrame",
    "RLMeshReference",
    "RLMeshProvenance",
    "RLMeshFitMode",
    "RLMeshRotationEncoding",
    "RLMeshDType",
}
# Output-only mirrors retain wire zero values (R-9).
OUTPUT_ENUMS = {
    "RLMeshErrorCode",
    "RLMeshModelErrorCode",
    "RLMeshSpaceKind",
    "RLMeshBoundsKind",
    "RLMeshValueKind",
    "RLMeshAutoresetMode",
}


@dataclass
class Item:
    kind: str
    signature: str = ""
    params: dict[str, str] = field(default_factory=dict)
    fields: list[str] = field(default_factory=list)
    default: str | None = None
    values: dict[str, int] = field(default_factory=dict)
    base: str = ""
    protocols: list[str] = field(default_factory=list)
    callbacks: dict[str, list[str]] = field(default_factory=dict)
    alias_of: str = ""
    protocol_params: dict[str, dict[str, str]] = field(default_factory=dict)
    returns: str = ""


def split(text: str) -> list[str]:
    """Split comma-separated declarations without splitting nested types/literals."""
    parts, start, stack, quote, escaped = [], 0, [], "", False
    for i, ch in enumerate(text):
        if quote:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = ""
        elif ch in "\"'" and not (
            ch == "'"
            and (lifetime := re.match(r"'[a-zA-Z_]\w*", text[i:]))
            and text[i + len(lifetime[0]) : i + len(lifetime[0]) + 1] != "'"
        ):
            quote = ch
        elif ch in "([{<":
            stack.append(ch)
        elif ch in ")]}>" and stack:
            stack.pop()
        elif ch == "," and not stack:
            parts.append(text[start:i].strip())
            start = i + 1
    return [p for p in [*parts, text[start:].strip()] if p]


def python_entries(snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Stable objects and members, with every signature parameter and default."""
    result = {}
    for module, body in snapshot.items():
        for name, obj in body["objects"].items():
            if obj.get("stability") != "Stable":
                continue
            for suffix, value in [
                (name, obj),
                *[
                    (f"{name}.{n}", m)
                    for n, m in obj.get("members", {}).items()
                    if m.get("stability", "Stable") == "Stable"
                ],
            ]:
                params = {}
                signature = value.get("signature", "")
                if value["kind"] in ("class", "function") and "(" in signature:
                    start, end = signature.find("("), signature.rfind(")")
                    for param in split(signature[start + 1 : end]):
                        if param in ("*", "/"):
                            continue
                        key = (
                            param.split(":", 1)[0].split("=", 1)[0].strip().lstrip("*")
                        )
                        if key in ("self", "cls"):
                            continue
                        params[key] = (
                            param.split("=", 1)[1].strip() if "=" in param else None
                        )
                result[f"{module}:{suffix}"] = {"kind": value["kind"], "params": params}
    return result


def native_name(name: str, language: str) -> str:
    if name.endswith("_seconds") and language != "c":
        name = name.removesuffix("_seconds")
    return name + "_" if name in RESERVED else name


def collect(root: Path, json_dir: Path | None = None) -> dict[str, dict[str, Item]]:
    policy = tomllib.loads((root / "rlmesh.toml").read_text())["api_surface"]
    policy["rust"] = {
        **policy["rust"],
        "document_hidden": True,
        "skipped_modules": ["binding"],
    }
    surfaces: dict[str, dict[str, Item]] = {}
    for language in LANGUAGES:
        items: dict[str, Item] = {}
        with header_compilers(language):
            for module in _collect_native_modules(
                language, policy[language], root, json_dir
            ):
                if language == "rust":
                    items[module.name] = Item("module")
                for export in module.exports:
                    path = export.qualified_name
                    items[path] = Item(export.kind, export.signature)
                    for alias in export.aliases:
                        items[alias] = Item("alias", alias_of=path)
                    for member in export.members:
                        member_path = f"{path}::{member.name}"
                        # Preserve all overloads' parameters, not the last overload.
                        entry = items.setdefault(
                            member_path, Item(member.kind, member.signature)
                        )
                        if member.kind == "field":
                            items[path].fields.append(member.name)
                        if language == "rust":
                            entry.params.update(rust_params(member.signature))
                    if language == "rust":
                        items[path].params.update(rust_params(export.signature))
            if language != "rust":
                config = HeaderSurfaceConfig(
                    policy[language]["header"],
                    "c++" if language == "cpp" else "c",
                    policy[language]["standard"],
                    "Stable",
                    policy[language].get("namespace"),
                    "",
                    None,
                )
                tu = _parse(root / config.header, config)
                header_metadata(
                    tu.cursor, (root / config.header).resolve(), items, language
                )
        if language == "cpp":
            text = (root / policy[language]["header"]).read_text()
            for owner, vtable, receiver in (
                ("rlmesh::Policy", "RLMeshModelVtable", "P"),
                ("rlmesh::EnvFactory", "RLMeshEnvFactoryVtable", "F"),
                ("rlmesh::Env", "RLMeshEnvVtable", "E"),
                ("rlmesh::VectorEnv", "RLMeshVectorEnvVtable", "E"),
            ):
                protocol_metadata(text, items[owner], surfaces["c"], vtable, receiver)
            for path in list(items):
                parts = path.split("::")
                for end in range(2, len(parts)):
                    scope = "::".join(parts[:end])
                    if scope not in items and parts[end - 1][:1].islower():
                        items[scope] = Item("module")
        if language == "rust":
            error_bases(root, items)
        surfaces[language] = items
    return surfaces


def protocol_metadata(
    text: str, item: Item, c_items: dict[str, Item], vtable: str, receiver: str
) -> None:
    """Detect actual authored calls and cross-check their arity with the ABI slot.

    C++ duck types do not declare parameter names. The named C callback slot is
    the shared R-12 contract; only input parameters supported by a real detected
    expression are exposed on the C++ protocol mapping.
    """
    calls: dict[str, int] = {}
    pattern = rf"(?:declval<(?:{receiver}|T)&>\(\)\.|static_cast<{receiver}\*>\(\w+\)->)(\w+)\("
    for match in re.finditer(pattern, text):
        start, depth, end = match.end(), 1, match.end()
        while end < len(text) and depth:
            depth += (text[end] == "(") - (text[end] == ")")
            end += 1
        calls[match[1]] = max(calls.get(match[1], 0), len(split(text[start : end - 1])))
    item.protocols = []
    item.protocol_params = {}
    for method, arity in calls.items():
        slot = c_items.get(vtable + "::" + method) or c_items.get(
            vtable.removesuffix("Vtable") + "Attributes::" + method
        )
        if slot is None:
            continue
        item.protocols.append(method)
        inputs = {
            n: t
            for n, t in slot.params.items()
            if n != "user_data"
            and not output_parameter(n, t)
            and not n.endswith("_len")
            and n not in {"count", "capacity"}
        }
        if len(inputs) <= arity:
            item.protocol_params[method] = inputs


@contextmanager
def header_compilers(language: str):
    """Optional local header driver overrides, never passed into Cargo builds."""
    before = {key: os.environ.get(key) for key in ("CC", "CXX")}
    try:
        if language != "rust":
            for key in before:
                override = os.environ.get("RLMESH_PARITY_HEADER_" + key)
                if override:
                    os.environ[key] = override
        yield
    finally:
        for key, value in before.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def error_bases(root: Path, items: dict[str, Item]) -> None:
    """Read the small ErrorClass::base match; rustdoc records only its signature.

    Fail closed if an arm is no longer in this simple form. This is not a Rust
    evaluator. Runtime exceptions, including the C forwarding implementation,
    are additionally exercised by the Errors fixture family.
    """
    source = (root / "crates/rlmesh/src/error.rs").read_text()
    body = source.split("pub fn base(self)", 1)[1].split("/// Class inheritance", 1)[0]
    found = {}
    for names, base in re.findall(
        r"((?:Self::\w+\s*\|?\s*)+) =>\s*\{?\s*(Some\(Self::\w+\)|None)", body
    ):
        parent = (
            base.removeprefix("Some(Self::").removesuffix(")") if base != "None" else ""
        )
        for name in re.findall(r"Self::(\w+)", names):
            found[name] = parent
    for name in ERROR_BASES:
        item = items.get("rlmesh::ErrorClass::" + name)
        if item is not None:
            item.base = found.get(name, "UNPARSED")


def rust_params(signature: str) -> dict[str, str]:
    match = re.search(r"\bfn\s+\w+(?:<.*?)?\(", signature)
    if not match:
        return {}
    start = match.end()
    depth, end = 1, start
    while end < len(signature) and depth:
        depth += (signature[end] == "(") - (signature[end] == ")")
        end += 1
    return {
        p.split(":", 1)[0].strip(): p.split(":", 1)[1].strip()
        for p in split(signature[start : end - 1])
        if ":" in p
    }


def header_metadata(
    cursor: Any, header: Path, items: dict[str, Item], language: str
) -> None:
    for child in cursor.get_children():
        if (
            child.location.file is None
            or Path(child.location.file.name).resolve() != header
        ):
            continue
        names = [child.spelling]
        parent = child.semantic_parent
        while parent is not None and parent.kind.name != "TRANSLATION_UNIT":
            if parent.spelling:
                names.insert(0, parent.spelling)
            parent = parent.semantic_parent
        path = "::".join(names)
        kind = child.kind.name
        if path in items:
            item = items[path]
            if kind in (
                "FUNCTION_DECL",
                "CXX_METHOD",
                "CONSTRUCTOR",
                "FUNCTION_TEMPLATE",
                "FIELD_DECL",
            ):
                item.returns = child.result_type.spelling
                item.params.update(
                    {
                        p.spelling: p.type.spelling
                        for p in child.get_children()
                        if p.kind.name == "PARM_DECL"
                        and p.location.offset > child.location.offset
                    }
                )
                item.callbacks.update(
                    {
                        p.spelling: [
                            c.spelling
                            for c in p.get_children()
                            if c.kind.name == "PARM_DECL"
                        ]
                        for p in child.get_children()
                        if p.kind.name == "PARM_DECL"
                        and p.spelling.removeprefix("out_").startswith("free_")
                    }
                )
            if kind == "FIELD_DECL":
                tokens = [t.spelling for t in child.get_tokens()]
                if child.spelling in tokens:
                    tokens = tokens[
                        len(tokens) - 1 - tokens[::-1].index(child.spelling) + 1 :
                    ]
                if "=" in tokens:
                    item.default = " ".join(tokens[tokens.index("=") + 1 :])
                elif "{" in tokens:
                    item.default = " ".join(tokens[tokens.index("{") :])
            if kind == "ENUM_DECL":
                item.values = {
                    c.spelling: c.enum_value
                    for c in child.get_children()
                    if c.kind.name == "ENUM_CONSTANT_DECL"
                }
            if kind == "MACRO_DEFINITION":
                item.signature = " ".join(t.spelling for t in child.get_tokens())
            for base in child.get_children():
                if base.kind.name == "CXX_BASE_SPECIFIER":
                    item.base = base.type.spelling.removeprefix("class ").removeprefix(
                        "rlmesh::"
                    )
        header_metadata(child, header, items, language)


def literal(text: str) -> Any:
    """Normalize native literals; refuse unknown expressions instead of guessing."""
    text = text.strip().rstrip(";")
    if text in ("NULL", "nullptr", "std::nullopt", "None", "{}", "{ }"):
        return None
    text = {"true": "True", "false": "False"}.get(text, text)
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?[fFuUlL]*", text):
        text = re.sub(r"[fFuUlL]+$", "", text)
    return ast.literal_eval(text)


def spec_ids() -> tuple[set[str], set[str], set[str]]:
    """Only IDs actually declared by the spec are citations, never R-n as decisions."""
    directory = REPO_ROOT / "docs/specs/language-parity"
    decisions, gaps, rules = set(), set(), set()
    for file in directory.glob("*.md"):
        text = file.read_text()
        decisions.update(
            re.findall(
                r"^\| (D-\d+|(?:COR|SPC|ENV|MDL|RUN|RMT|ADP)-\d+)\s*\|", text, re.M
            )
        )
        gaps.update(re.findall(r"^\| (G-\d+)\s*\|", text, re.M))
        rules.update(re.findall(r"^### (R-\d+)\b", text, re.M))
    return decisions, gaps, rules


def bridge_scope(path: str, parameter: str | None = None) -> bool:
    module, _, symbol = path.partition(":")
    if parameter is not None:
        return (
            parameter in {"bridge", "bridges", "input_bridge", "output_bridge"}
            or (
                symbol in {"EnvServer", "EnvFactory.serve"}
                and parameter in {"framework", "device", "vectorization_mode"}
            )
            or (
                symbol.endswith(".__dlpack__")
                and parameter in {"max_version", "stream", "dl_device", "copy"}
            )
        )
    if path in {"rlmesh:types", "rlmesh:Model.device"}:
        return True
    if module in {"rlmesh.numpy", "rlmesh.torch", "rlmesh.jax"}:
        return True
    if symbol.rsplit(".", 1)[-1] in {
        "to_gymnasium",
        "from_gymnasium",
        "to_gymnasium_space",
        "from_gymnasium_space",
        "to_gym_space",
    }:
        return True
    # R-24 covers Python structural typing/union conveniences, not native data.
    return module == "rlmesh.types" and symbol.split(".")[0] in {
        "HasAddress",
        "HasEnvContract",
        "SpaceLike",
        "SupportsMake",
        "SupportsResetStep",
        "PrimitiveValue",
        "ViewArg",
        "LocalEnvTarget",
    }


def output_parameter(name: str, ty: str) -> bool:
    return (name == "out" or name.startswith("out_")) and (
        (
            "*" in ty
            and (not ty.strip().startswith("const ") or "**" in ty.replace(" ", ""))
        )
        or ("[" in ty and not ty.strip().startswith("const "))
    )


def extension_scope(path: str, language: str, item: Item) -> bool:
    if fenced(path, language):
        return True
    if language == "c":
        if path in {"RLMESH_ABI_VERSION", "rlmesh_abi_version", "rlmesh_abi_check"}:
            return True
        if re.fullmatch(r"rlmesh_(remote_env|remote_vector_env)_cancel_handle", path):
            return True
    owner, _, member = path.rpartition("::")
    owner = owner.rsplit("::", 1)[-1].removeprefix("RLMesh")
    if member in {"cancel", "with_cancel"}:
        return owner in {
            "RunOptions",
            "ModelServeOptions",
            "EnvFactoryServeOptions",
        }
    if member == "cancel_handle":
        return owner in {"RemoteEnv", "RemoteVectorEnv"}
    return member == "ext" and (
        owner in {"EnvServerOptions", "ModelServeOptions"}
        or (language == "c" and owner == "EnvFactoryServeOptions")
    )


def also_structural(
    target: str, language: str, items: dict[str, Item], mapped: set[str]
) -> bool:
    if target in mapped or fenced(target, language):
        return True
    item = items.get(target, Item("missing"))
    if item.kind in {
        "alias",
        "trait implementation",
        "variant",
        "enum",
        "struct",
        "class",
        "type alias",
        "module",
    }:
        return True
    owner, _, member = target.rpartition("::")
    fields = {
        p.removeprefix(owner + "::") for p in mapped if p.startswith(owner + "::")
    }
    if member.startswith("_reserved") or member == "struct_size":
        return True
    if member.startswith("has_") and member[4:] in fields:
        return True
    if member.endswith("_len") and member[:-4] in fields:
        return True
    if member.startswith("with_") and member[5:] in fields:
        return True
    if member in fields:  # Rust options setters share their field's name.
        return True
    # Enum spellings, standard trait impls and C sizing macros are translations.
    if items.get(owner, Item("missing")).kind == "enum":
        return True
    if language == "c" and target.endswith(("_INIT", "_MIN_SIZE")):
        return True
    return False


def check_order(
    path: str, python: dict, mapping: dict, item: Item, language: str, fail: Any
) -> None:
    native = mapping.get("path", "")
    translations = mapping.get("translations", {})
    names = {
        target.partition("#")[2]: name
        for name, target in mapping.get("params", {}).items()
        if target.partition("#")[0] == native and name not in translations
    }
    native_params = item.protocol_params.get(mapping.get("protocol", ""), item.params)
    actual = [
        names[n]
        for n in native_params
        if n in names
        and not structural(n, native_params[n], native_params, path, language)
    ]
    wanted = [n for n in python["params"] if n in actual]
    if actual != wanted:
        fail(
            2,
            f"{path}.{language}: positional order {actual} != {wanted}; edit params (R-6/R-7/R-12)",
        )
    method = path.rsplit(".", 1)[-1]
    if (
        language == "c"
        and method in {"reset", "step"}
        and path.partition(":")[2].split(".")[0]
        in {"Binding", "RemoteEnv", "RemoteVectorEnv", "EnvLike", "VectorEnvLike"}
    ):
        actual_outputs = [
            n.removeprefix("out_")
            for n, ty in item.params.items()
            if output_parameter(n, ty) and n != "out_len"
        ]
        actual_outputs = ["reward" if n == "rewards" else n for n in actual_outputs]
        wanted_outputs = (
            ["observation", "info"]
            if method == "reset"
            else ["observation", "reward", "terminated", "truncated", "info"]
        )
        if actual_outputs != wanted_outputs:
            fail(
                2,
                f"{path}.{language}: output order {actual_outputs} != {wanted_outputs}; edit params/path (R-15)",
            )


def check_result_tuple(
    path: str, mapping: dict, items: dict[str, Item], language: str, fail: Any
) -> None:
    expected = mapping.get("result_tuple")
    if expected is None:
        return
    native = mapping.get("path", "")
    signature = items.get(native, Item("missing")).signature
    if language == "rust":
        match = re.search(r"->\s*Result<\((.*)\)>$", signature)
    else:
        result = mapping.get("returns", "")
        signature = items.get("rlmesh::" + result, Item("missing")).signature
        match = re.search(r"std::tuple<(.*)>$", signature)
    actual = split(match[1]) if match else []

    def normalize(values):
        return [re.sub(r"\s+", "", value) for value in values]

    if normalize(actual) != normalize(expected):
        fail(
            2,
            f"{path}.{language}: result_tuple order/types {actual} != {expected}; edit result_tuple/returns or native tuple (R-15)",
        )


def check(
    expected: dict[str, Any],
    manifest: dict[str, Any],
    surfaces: dict[str, dict[str, Item]],
) -> list[str]:
    errors: list[str] = []
    covered = {lang: set() for lang in LANGUAGES}
    decisions, gaps, rule_ids = spec_ids()
    mapped_targets = {lang: set() for lang in LANGUAGES}
    for path, row in manifest.items():
        if ":" not in path:
            continue
        for lang in LANGUAGES:
            mapping = row.get(lang, {})
            if mapping.get("status") != "present":
                continue
            mapped_targets[lang].add(mapping.get("path", ""))
            mapped_targets[lang].update(
                t.replace("#", "::") for t in mapping.get("params", {}).values()
            )

    def fail(number: int, message: str) -> None:
        guidance = {
            1: "edit symbol parameters/status/rule/bridges or missing.gap (E-1, R-1/R-24)",
            2: "edit language path/params/also/also_rules/translations (R-4/R-7/R-15)",
            3: "edit missing status/gap to present with path/params (R-1)",
            4: "edit mapped params/also, [ext] or [binding] (R-25)",
            5: "edit parameters.default/expression or translations/default_skip, or fix native defaults (R-6/R-19)",
            6: "edit the enum's path/also or fix native enum (R-9)",
            7: "edit error path/also or fix native error hierarchy (R-21)",
        }
        errors.append(f"E-2.{number}: {message}; {guidance[number]}")

    for path, python in expected.items():
        row = manifest.get(path)
        if row is None:
            fail(1, f"{path}: no mapping")
            continue
        params = python["params"]
        if set(params) != set(row.get("parameters", {})):
            fail(
                1,
                f"{path}: parameter coverage differs: {set(params) ^ set(row.get('parameters', {}))}",
            )
        for name, source_default in params.items():
            stored = row.get("parameters", {}).get(name, {})
            if (
                source_default is not None
                and stored.get("expression", stored.get("default")) != source_default
            ):
                fail(5, f"{path}#{name}: manifest default is stale relative to Python")
        for language in LANGUAGES:
            mapping = row.get(language, {})
            status, native = mapping.get("status"), mapping.get("path", "")
            items = surfaces[language]
            rules = row.get("rule", []) + mapping.get("rule", [])
            if status not in ("present", "missing", "bridge", "exempt"):
                fail(1, f"{path}.{language}: invalid or absent status")
                continue
            if status == "bridge" and "R-24" not in rules:
                fail(1, f"{path}.{language}: bridge requires R-24")
            if status == "bridge" and not bridge_scope(path):
                fail(
                    1,
                    f"{path}.{language}: outside R-24 scope; implement a present mapping",
                )
            if status == "exempt" and not decisions.intersection(
                mapping.get("rule", [])
            ):
                fail(1, f"{path}.{language}: exemption requires a decision")
            if status == "missing":
                if mapping.get("gap") not in gaps:
                    fail(
                        1,
                        f"{path}.{language}: missing requires a gap from the spec table, got {mapping.get('gap')!r}",
                    )
                if native in items:
                    fail(3, f"{path}.{language}: stale gap: {native} exists")
            for name, citations in mapping.get("translations", {}).items():
                if not decisions.intersection(citations):
                    fail(
                        1,
                        f"{path}.{language}.translations.{name}: requires its own decision",
                    )
            for name in mapping.get("bridges", []):
                if "R-24" not in rules or not bridge_scope(path, name):
                    fail(1, f"{path}.{language}.bridges: {name} is outside R-24 scope")
            for target in mapping.get("also", []):
                citations = mapping.get("also_rules", {}).get(target, [])
                if not also_structural(
                    target, language, items, mapped_targets[language]
                ) and not (decisions | rule_ids).intersection(citations):
                    fail(
                        4,
                        f"{path}.{language}: also entry {target} needs a mapped field/param or also_rules citation (R-25)",
                    )
                if target not in items:
                    fail(2, f"{path}.{language}: {target} does not exist")
                covered[language].add(target)
            if status != "present":
                continue
            protocol = mapping.get("protocol")
            if protocol and (
                native not in items or protocol not in items[native].protocols
            ):
                fail(
                    2, f"{path}.{language}: missing detected protocol member {protocol}"
                )
            for target in [native]:
                if target not in items:
                    fail(2, f"{path}.{language}: {target} does not exist")
                covered[language].add(target)
            mapped_params = mapping.get("params", {})
            for name in params:
                target = mapped_params.get(name)
                if target is None:
                    if name in mapping.get("bridges", []) and "R-24" in rules:
                        continue
                    fail(1, f"{path}.{language}: parameter {name} has no mapping")
                    continue
                owner, sep, field_name = target.partition("#")
                translated = bool(
                    decisions.intersection(
                        mapping.get("translations", {}).get(name, [])
                    )
                )
                if field_name != native_name(name, language) and not translated:
                    fail(
                        2,
                        f"{path}.{language}: {name} maps to {field_name}, violates R-4",
                    )
                if name in mapping.get("factory_switch", {}):
                    for factory in mapping["factory_switch"][name]:
                        if factory not in items:
                            fail(2, f"{path}.{language}: missing factory {factory}")
                        covered[language].add(factory)
                    continue
                exists = owner in items and (
                    field_name
                    in (
                        items[owner].protocol_params.get(protocol, {})
                        if protocol
                        else items[owner].params
                    )
                    if sep
                    else False
                )
                if owner + "::" + field_name in items:
                    exists = True
                    covered[language].add(owner + "::" + field_name)
                    covered[language].add(owner)
                if (
                    name.endswith("_seconds")
                    and language != "c"
                    and field_name == name.removesuffix("_seconds")
                ):
                    member = items.get(owner + "::" + field_name)
                    native_type = (
                        member.signature
                        if member is not None
                        else items.get(owner, Item("missing")).params.get(
                            field_name, ""
                        )
                    )
                    if (
                        "Duration" not in native_type
                        and "chrono::duration" not in native_type
                    ):
                        fail(
                            2,
                            f"{path}.{language}: {name} may drop _seconds only for a duration type",
                        )
                if not exists:
                    fail(
                        2, f"{path}.{language}: parameter/field {target} does not exist"
                    )
            check_result_tuple(path, mapping, items, language, fail)
            if native in items and mapping.get("returns") is not None:
                actual_return = items[native].returns
                if language == "rust":
                    match = re.search(
                        r"\)\s*->\s*(.*?)(?:\s+where\b|$)", items[native].signature
                    )
                    actual_return = match[1] if match else "()"
                if re.sub(r"\s+", "", actual_return) != re.sub(
                    r"\s+", "", mapping["returns"]
                ):
                    fail(
                        2,
                        f"{path}.{language}: returns {actual_return!r} != {mapping['returns']!r}; edit returns (R-3/R-15)",
                    )
            if native in items and python["kind"] == "function":
                check_order(path, python, mapping, items[native], language, fail)
            # Every positional native input needs a Python counterpart or a
            # structural exemption, including additions made after review.
            if native in items and python["kind"] == "function" and not protocol:
                mapped = {
                    v.partition("#")[2]
                    for v in mapped_params.values()
                    if v.partition("#")[0] == native
                }
                for name, ty in items[native].params.items():
                    if name in mapped or structural(
                        name, ty, items[native].params, path, language
                    ):
                        continue
                    if (
                        name == "i"
                        and mapping.get("collection") in ("SPC-30", "ADP-23")
                        and native.endswith(("_get", "_key_at", "_value_at"))
                    ):
                        continue
                    if name in mapping.get("injected", {}) and mapping["injected"][
                        name
                    ] in ("MDL-3", "MDL-4"):
                        continue
                    fail(
                        2,
                        f"{path}.{language}: native parameter {native}#{name} has no Python counterpart",
                    )
            if (
                python["kind"] == "function"
                and native
                and not protocol
                and not mapping.get("collection")
            ):
                name = path.partition(":")[2].rsplit(".", 1)[-1]
                tail = native.rsplit("::", 1)[-1]
                if language == "c":
                    wanted = snake(name)
                    matches = tail.endswith("_" + wanted) or tail == wanted
                else:
                    matches = tail == native_name(name, language)
                if not matches:
                    fail(2, f"{path}.{language}: callable name {tail} violates R-4/D-6")
            if python["kind"] in ("property", "attribute") and native:
                name = path.rsplit(".", 1)[-1]
                tail = native.rsplit("::", 1)[-1]
                if (
                    (
                        language != "c"
                        or items.get(native, Item("missing")).kind == "field"
                    )
                    and not protocol
                    and tail != native_name(name, language)
                ):
                    fail(2, f"{path}.{language}: member name {tail} violates R-4")
    for section in ("ext", "binding"):
        for language in LANGUAGES:
            for row in manifest.get(section, {}).get(language, []):
                target = row["path"]
                if section == "ext" and not extension_scope(
                    target, language, surfaces[language].get(target, Item("missing"))
                ):
                    fail(
                        4, f"[ext].{language}: {target} is outside the R-25 closed list"
                    )
                if section == "binding" and not (
                    (
                        language == "rust"
                        and (
                            target.startswith("rlmesh::binding::")
                            or target == "rlmesh::Error::Attached"
                        )
                    )
                    or (
                        language == "c"
                        and target.split("::")[0]
                        in {
                            "DLPackVersion",
                            "DLDeviceType",
                            "DLDevice",
                            "DLDataType",
                            "DLDataTypeCode",
                            "DLTensor",
                            "DLManagedTensor",
                            "DLManagedTensorVersioned",
                            "RLMESH_HOST_RUNTIME_JSON",
                        }
                    )
                ):
                    fail(
                        4,
                        f"[binding].{language}: {target} is not R-12/R-25/SPC-19 plumbing",
                    )
                if row.get("rule") not in decisions | rule_ids:
                    fail(
                        4,
                        f"[{section}].{language}: {target} needs a spec rule/decision",
                    )
                if target not in surfaces[language]:
                    fail(2, f"{section}.{language}: {target} does not exist")
                covered[language].add(target)
    for language, items in surfaces.items():
        for path, item in items.items():
            if (
                path in covered[language]
                or fenced(path, language)
                or item.kind == "module"
            ):
                continue
            fail(4, f"{language}: unrecorded public item {path}")
    check_defaults(manifest, surfaces, fail)
    for path, item in surfaces["c"].items():
        if item.kind != "enum" or not path.startswith("RLMesh"):
            continue
        zero = [n for n, value in item.values.items() if value == 0]
        if (
            path not in OUTPUT_ENUMS
            and path != "RLMeshStatus"
            and (len(zero) != 1 or not zero[0].endswith("_INVALID"))
        ):
            fail(6, f"{path}: input enum has no INVALID = 0")
        if path == "RLMeshStatus" and item.values.get("RLMESH_OK") != 0:
            fail(6, "RLMeshStatus: RLMESH_OK must be zero")
        if path not in TOLERANT and any(n.endswith("_OTHER") for n in item.values):
            fail(6, f"{path}: unapproved tolerant OTHER enumerator")
    for path, item in surfaces["c"].items():
        callbacks = dict(item.callbacks)
        name = path.rsplit("::", 1)[-1]
        if item.kind == "field" and name.startswith("free_"):
            callbacks[name] = list(item.params)
        for name, parameters in callbacks.items():
            freed = name.removeprefix("out_").removeprefix("free_")
            if parameters != [freed]:
                fail(2, f"{path}: {name} must name its freed argument {freed}")
    check_errors(surfaces, fail)
    return errors


def structural(
    name: str, ty: str, params: dict[str, str], python: str, language: str
) -> bool:
    if name in ("self", "struct_size") or (
        language == "c" and output_parameter(name, ty)
    ):
        return True
    if name.endswith("_len") and "*" in params.get(name[:-4], ""):
        return True
    if name == "capacity" and any(output_parameter(n, t) for n, t in params.items()):
        return True
    if name in ("len", "count") and any("*" in t for t in params.values()):
        return True
    if name in ("user_data", "ctx") and ("*" in ty or "c_void" in ty):
        return True
    if (
        name == "free_user_data"
        and "user_data" in params
        and "*" in ty
        and any(
            n != name and ("(*" in t or t.endswith("Fn")) for n, t in params.items()
        )
    ):
        return True
    if re.search(r"\b\w+Options\b", ty):
        return True
    owner = python.partition(":")[2].split(".")[0]
    receivers = {owner, owner + "Vtable"}
    if owner in ("Box", "Discrete", "Dict"):
        receivers.add("Space")
    if owner == "AdapterBase":
        receivers.add("Adapter")
    if (
        language == "c"
        and "." in python.partition(":")[2]
        and name == next(iter(params), None)
        and "*" in ty
        and any(
            t.removeprefix("RLMesh") in receivers for t in re.findall(r"RLMesh\w+", ty)
        )
    ):
        return True
    return False


def snake(name: str) -> str:
    return re.sub(
        r"([a-z0-9])([A-Z])", r"\1_\2", re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    ).lower()


def fenced(path: str, language: str) -> bool:
    if language == "c":
        return path.startswith(("rlmesh_ext_", "RLMeshExt", "RLMESH_EXT_"))
    scopes = (
        ("rlmesh::ext", "rlmesh::binding") if language == "rust" else ("rlmesh::ext",)
    )
    return any(path == scope or path.startswith(scope + "::") for scope in scopes)


def default_value(text: str, item: Item, c_items: dict[str, Item]) -> Any:
    """Decode enum and aggregate literals, without evaluating native code."""
    text = text.strip()
    enum = None
    member = None
    cast = re.fullmatch(r"\(\s*(RLMesh\w+)\s*\)\s*(\d+)", text)
    cpp = re.fullmatch(r"(?:\w+\s*::\s*)*(\w+)\s*::\s*(\w+)", text)
    if cast:
        enum = c_items.get(cast[1])
        if enum is not None:
            member = next(
                (n for n, v in enum.values.items() if v == int(cast[2])), None
            )
    elif text.startswith("RLMESH_"):
        member = text
        enum = next((e for e in c_items.values() if member in e.values), None)
    elif cpp:
        return snake(cpp[2])
    if member and enum:
        prefix = os.path.commonprefix(list(enum.values))
        # The longest common prefix can include part of the first word.
        prefix = prefix[: prefix.rfind("_") + 1]
        # INVALID is present on every input vocabulary and fixes its prefix.
        invalid = next((n for n in enum.values if n.endswith("_INVALID")), None)
        if invalid:
            prefix = invalid.removesuffix("INVALID")
        word = member.removeprefix(prefix).lower()
        if word == "invalid":
            return None
        if word == "off":
            return False
        return word
    if re.fullmatch(r"\{\s*(?:NULL\s*,\s*)?(?:NULL|0)\s*\}", text):
        return None
    if text.startswith("{") and text.endswith("}") and text not in ("{}", "{ }"):
        if "RLMESH_SPEC_KIND_NONE" in text:
            return None
        return tuple(
            default_value(v, Item("literal"), c_items) for v in split(text[1:-1])
        )
    if text in ("{}", "{ }"):
        if "optional" in item.signature or "OptionalList" in item.signature:
            return None
        if any(t in item.signature for t in ("vector", "Vec<")):
            return ()
        if "Kwargs" in item.signature or "Metadata" in item.signature:
            return {}
    return literal(text)


def decisions_for_default(mapping: dict, parameter: str) -> bool:
    return "RUN-20" in mapping.get("translations", {}).get(parameter, [])


def check_defaults(
    manifest: dict[str, Any], surfaces: dict[str, dict[str, Item]], fail: Any
) -> None:
    for path, item in surfaces["c"].items():
        if not path.endswith("Options") or item.kind != "struct":
            continue
        stem = "RLMESH_" + snake(path.removeprefix("RLMesh")).upper()
        if not item.fields or item.fields[0] != "struct_size":
            fail(5, f"{path}: struct_size must be first")
        for suffix in ("_INIT", "_MIN_SIZE"):
            if stem + suffix not in surfaces["c"]:
                fail(5, f"{path}: missing {stem + suffix}")
    for python_path, row in manifest.items():
        if ":" not in python_path:
            continue
        for language in ("c", "cpp"):
            mapping = row.get(language, {})
            for param, target in mapping.get("params", {}).items():
                default = row.get("parameters", {}).get(param, {}).get("default")
                owner, _, name = target.partition("#")
                if default is None or owner not in surfaces[language]:
                    continue
                item = surfaces[language].get(owner + "::" + name)
                if (
                    item is None
                    or item.kind != "field"
                    or surfaces[language][owner].kind not in ("struct", "class")
                ):
                    continue
                actual = item.default
                if language == "c":
                    stem = (
                        "RLMESH_"
                        + snake(owner.removeprefix("RLMesh")).upper()
                        + "_INIT"
                    )
                    macro = surfaces["c"].get(stem)
                    if macro is None:
                        continue
                    text = macro.signature
                    values = split(text[text.find("{") + 1 : text.rfind("}")])
                    fields = surfaces["c"][owner].fields
                    if name not in fields or fields.index(name) >= len(values):
                        fail(5, f"{target}: missing initializer")
                        continue
                    actual = values[fields.index(name)]
                    if "has_" + name in fields:
                        guard = values[fields.index("has_" + name)]
                        if literal(guard) in (False, 0):
                            actual = "None"
                if actual is None:
                    fail(5, f"{target}: missing default initializer")
                    continue
                try:
                    want = ast.literal_eval(default)
                    if isinstance(want, list):
                        want = tuple(want)
                    got = default_value(actual, item, surfaces["c"])
                    if language == "c" and got is None and want in ((), {}):
                        got = want
                except (ValueError, SyntaxError):
                    fail(
                        5,
                        f"{target}: cannot decode native default {actual!r} for {default}",
                    )
                else:
                    if want != got:
                        if (
                            language == "c"
                            and decisions_for_default(mapping, param)
                            and actual == "NULL"
                        ):
                            continue
                        fail(5, f"{target}: default {actual!r} != {default}")


def check_errors(surfaces: dict[str, dict[str, Item]], fail: Any) -> None:
    for name, base in ERROR_BASES.items():
        rust = surfaces["rust"].get("rlmesh::ErrorClass::" + name)
        c = surfaces["c"].get("RLMeshErrorClass")
        c_name = "RLMESH_ERROR_CLASS_" + snake(name).upper().replace(
            "RL_MESH", "RLMESH"
        )
        cpp_name = "rlmesh::" + (
            "ext::CancelledError" if name == "KeyboardInterrupt" else name
        )
        cpp = surfaces["cpp"].get(cpp_name)
        if (
            rust is None
            or c is None
            or c.values.get(c_name) != list(ERROR_BASES).index(name) + 1
            or cpp is None
        ):
            fail(7, f"{name}: missing error class")
        rust_base = "" if base.startswith("std::") else base
        if rust is not None and rust.base != rust_base:
            fail(7, f"ErrorClass::{name}: base {rust.base!r} != {rust_base!r}")
        if cpp is not None and cpp.base != base:
            fail(7, f"{cpp_name}: base {cpp.base!r} != {base!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--rustdoc-json-dir", type=Path)
    parser.add_argument(
        "--dump", type=Path, help="Save extracted inputs for manifest review"
    )
    args = parser.parse_args()
    surfaces = collect(args.root, args.rustdoc_json_dir)
    if args.dump:
        args.dump.write_text(
            json.dumps(
                {
                    lang: {p: asdict(i) for p, i in items.items()}
                    for lang, items in surfaces.items()
                },
                indent=2,
            )
            + "\n"
        )
    snapshot = json.loads(
        (
            args.root / "python/rlmesh/tests/api_surface/snapshots/api_surface.json"
        ).read_text()
    )
    manifest = tomllib.loads(
        (args.root / "docs/specs/language-parity/manifest.toml").read_text()
    )
    errors = check(python_entries(snapshot), manifest, surfaces)
    print(
        "\n".join(errors)
        if errors
        else f"Parity check passed: {sum(':' in p for p in manifest)} Stable mappings"
    )
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())

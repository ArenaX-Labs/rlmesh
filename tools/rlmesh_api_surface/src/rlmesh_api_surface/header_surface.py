"""Public API extraction for the C ABI (`rlmesh.h`) and C++ wrapper (`rlmesh.hpp`).

libclang parses each header as its consumers compile it. Documentation is the
comment block directly above a declaration (or trailing a field/enumerator on
its line), which is how both headers are written. A `---- name ----` divider
comment starts a section, and each section becomes one module page.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from clang import cindex as ci

from rlmesh_api_surface.api_surface import (
    ApiExport,
    ApiMember,
    ApiModule,
    SourceLocation,
)

_DIVIDER = re.compile(r"^\s*(?:/\*|//)\s*-{3,}\s*(.+?)\s*-*\s*(?:\*/)?\s*$")
# Header-plumbing macros: export decoration, include guards, and internals.
_PLUMBING_MACRO = re.compile(
    r"^RLMESH_(API|H|HPP|STATIC|BUILD_DLL)$|_DETAIL_|^RLMESH_DETAIL"
)


@dataclass(frozen=True)
class HeaderSurfaceConfig:
    header: str  # repo-relative
    language: str  # "c" or "c++"
    standard: str  # "c11", "c++17", "c++20"
    stability: str
    namespace: str | None  # the C++ namespace to document; None for C
    source_repo: str
    source_commit: str | None


def collect_header_api_surface(
    *, repo_root: Path, config: HeaderSurfaceConfig
) -> list[ApiModule]:
    path = repo_root / config.header
    tu = _parse(path, config)
    lines = path.read_text(encoding="utf-8").splitlines()
    collector = _HeaderCollector(repo_root, path, lines, config)
    return collector.collect(tu)


def _parse(path: Path, config: HeaderSurfaceConfig) -> ci.TranslationUnit:
    args = [
        "-x",
        config.language,
        f"-std={config.standard}",
        f"-I{path.parent}",
        *_library_defines(config.language),
        *[
            arg
            for directory in _system_includes(config.language)
            for arg in ("-isystem", directory)
        ],
    ]
    tu = ci.Index.create().parse(
        str(path),
        args=args,
        options=ci.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
        | ci.TranslationUnit.PARSE_SKIP_FUNCTION_BODIES,
    )
    errors = [d for d in tu.diagnostics if d.severity >= ci.Diagnostic.Error]
    if errors:
        detail = "\n".join(str(d) for d in errors[:10])
        raise ValueError(f"{path}: libclang could not parse the header:\n{detail}")
    return tu


def _system_includes(language: str) -> list[str]:
    """The selected CC/CXX's include search path, which the libclang wheel lacks."""
    compiler = shlex.split(
        os.environ.get("CXX" if language == "c++" else "CC")
        or ("c++" if language == "c++" else "cc")
    )
    stderr = subprocess.run(
        [*compiler, "-E", "-v", "-x", language, "/dev/null"],
        capture_output=True,
        text=True,
        check=True,
    ).stderr
    match = re.search(
        r"#include <\.\.\.> search starts here:\n(.*?)End of search list", stderr, re.S
    )
    if match is None:
        raise ValueError(f"could not read {shlex.join(compiler)}'s include search path")
    return [
        line.strip().split(" (")[0]
        for line in match.group(1).splitlines()
        if line.strip()
    ]


def _library_defines(language: str) -> list[str]:
    """Preserve driver-provided libc++ configuration (not compiler builtins).

    Zig configures libc++ on the command line instead of in __config_site.
    Feeding only its include paths to libclang loses that configuration.
    """
    compiler = shlex.split(
        os.environ.get("CXX" if language == "c++" else "CC")
        or ("c++" if language == "c++" else "cc")
    )
    output = subprocess.run(
        [*compiler, "-dM", "-E", "-x", language, "/dev/null"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [
        "-D" + match[1] + ("=" + match[2] if match[2] else "")
        for line in output.splitlines()
        if (match := re.match(r"#define (_LIBCPP_\w+)(?:\s+(.*))?", line))
    ]


@dataclass
class _Entry:
    line: int
    export: ApiExport


class _HeaderCollector:
    def __init__(
        self, repo_root: Path, path: Path, lines: list[str], config: HeaderSurfaceConfig
    ) -> None:
        self.repo_root = repo_root
        self.path = path
        self.lines = lines
        self.config = config
        self.entries: list[_Entry] = []
        self.seen: set[str] = set()

    def collect(self, tu: ci.TranslationUnit) -> list[ApiModule]:
        scopes = [tu.cursor]
        if self.config.namespace is not None:
            scopes = [
                c
                for c in tu.cursor.get_children()
                if c.kind.name == "NAMESPACE"
                and c.spelling == self.config.namespace
                and self._here(c)
            ]
        for scope in scopes:
            self._scope(scope)
        if self.config.namespace is None:
            for cursor in tu.cursor.get_children():
                if cursor.kind.name == "MACRO_DEFINITION" and self._here(cursor):
                    self._macro(cursor)
        return self._modules()

    def _scope(self, scope: ci.Cursor) -> None:
        typedef_names = {
            c.underlying_typedef_type.get_declaration().spelling: c
            for c in scope.get_children()
            if c.kind.name == "TYPEDEF_DECL" and self._here(c)
        }
        for cursor in scope.get_children():
            if self._here(cursor):
                self._top_level(cursor, typedef_names)

    def _here(self, cursor: ci.Cursor) -> bool:
        location = cursor.location
        return (
            location.file is not None
            and Path(location.file.name).resolve() == self.path.resolve()
        )

    # ---- declarations ------------------------------------------------------

    def _top_level(self, cursor: ci.Cursor, typedefs: dict[str, ci.Cursor]) -> None:
        kind = cursor.kind.name
        name = cursor.spelling
        if not name or name.startswith(("_", "detail")):
            return
        if kind == "NAMESPACE":
            self._scope(cursor)
        elif kind in ("FUNCTION_DECL", "FUNCTION_TEMPLATE"):
            if name.startswith("operator"):
                return
            self._add(cursor, name, "function", self._declaration_text(cursor))
        elif kind in (
            "STRUCT_DECL",
            "CLASS_DECL",
            "CLASS_TEMPLATE",
            "CLASS_TEMPLATE_PARTIAL_SPECIALIZATION",
            "UNION_DECL",
        ):
            self._record(cursor, name)
        elif kind == "ENUM_DECL":
            self._enum(cursor, name)
        elif kind == "TYPEDEF_DECL":
            target = cursor.underlying_typedef_type.get_declaration()
            if (
                target.kind.name in ("STRUCT_DECL", "ENUM_DECL", "UNION_DECL")
                and target.spelling == name
            ):
                if target.is_definition() or name in self.seen:
                    return  # `typedef struct X {...} X;` documents as the struct itself
                self._add(
                    cursor,
                    name,
                    "opaque type",
                    self._statement_text(cursor),
                    members=[],
                )
            else:
                self._add(cursor, name, "type alias", self._statement_text(cursor))
        elif kind in ("TYPE_ALIAS_DECL", "TYPE_ALIAS_TEMPLATE_DECL", "NAMESPACE_ALIAS"):
            self._add(cursor, name, "type alias", self._statement_text(cursor))
        elif kind == "CONCEPT_DECL":
            self._add(cursor, name, "concept", self._statement_text(cursor))
        elif kind == "VAR_DECL":
            self._add(cursor, name, "constant", self._statement_text(cursor))
        elif kind == "UNEXPOSED_DECL":
            # libclang 18 exposes C++17 constexpr variable templates this way
            # rather than as VAR_DECL (for example the public Policy trait).
            text = self._statement_text(cursor)
            if re.match(r"template\s*<.*>\s*(?:inline\s+)?constexpr\b", text):
                self._add(cursor, name, "constant", text)

    def _record(self, cursor: ci.Cursor, name: str) -> None:
        if not cursor.is_definition():
            return  # a forward declaration; the typedef (C) or definition documents it
        if cursor.kind.name != "CLASS_TEMPLATE" and "<" in cursor.displayname:
            # An explicit or partial specialization (`Result<void>`) has its own
            # members, so it documents as its own entry under its full name. Its
            # kind says so whichever of it and the primary is declared first.
            name = cursor.displayname
        keyword = {
            "CLASS_DECL": "class",
            "CLASS_TEMPLATE": "class",
            "CLASS_TEMPLATE_PARTIAL_SPECIALIZATION": "class",
            "UNION_DECL": "union",
        }.get(cursor.kind.name, "struct")
        if (
            self.config.language == "c++"
            and keyword == "struct"
            and _has_methods(cursor)
        ):
            keyword = "class"
        members = self._record_members(
            cursor, default_public=cursor.kind.name == "STRUCT_DECL"
        )
        self._add(cursor, name, keyword, self._head_text(cursor), members=members)

    def _enum(self, cursor: ci.Cursor, name: str) -> None:
        if not cursor.is_definition() or name in self.seen:
            return
        members = [
            ApiMember(
                name=c.spelling,
                kind="enumerator",
                signature=f"{c.spelling} = {c.enum_value}",
                documentation=self._docs(c, trailing=True),
                defined_at=self._location(c),
            )
            for c in cursor.get_children()
            if c.kind.name == "ENUM_CONSTANT_DECL"
        ]
        self._add(cursor, name, "enum", self._head_text(cursor), members=members)

    def _macro(self, cursor: ci.Cursor) -> None:
        name = cursor.spelling
        if not name.startswith("RLMESH_") or _PLUMBING_MACRO.search(name):
            return
        if name in self.seen:
            return  # one branch of a compiler-detection #if chain is enough
        text = " ".join(t.spelling for t in cursor.get_tokens())
        head = re.match(r"^(\w+(?:\s*\([^)]*\))?)", text)
        definition = (
            text if len(text) <= 100 else (head.group(1) if head else name) + " …"
        )
        self._add(cursor, name, "macro", f"#define {definition}")

    def _record_members(
        self, cursor: ci.Cursor, *, default_public: bool
    ) -> list[ApiMember]:
        members: list[ApiMember] = []
        public = default_public
        for child in cursor.get_children():
            if child.kind.name == "CXX_ACCESS_SPEC_DECL":
                public = child.access_specifier.name == "PUBLIC"
                continue
            if self.config.language == "c++" and not public:
                continue
            member = self._member(child, cursor.spelling)
            if member is not None:
                members.append(member)
        return members

    def _member(self, child: ci.Cursor, owner: str) -> ApiMember | None:
        kind = child.kind.name
        if kind == "FIELD_DECL":
            return self._member_entry(
                child, "field", self._statement_text(child), trailing=True
            )
        if kind == "CONSTRUCTOR":
            if _is_deleted(self._statement_text(child)):
                return None
            return self._member_entry(
                child, "constructor", self._declaration_text(child, constructor=True)
            )
        if kind in ("CXX_METHOD", "FUNCTION_TEMPLATE", "CONVERSION_FUNCTION"):
            constructor = child.spelling == owner
            text = self._declaration_text(child, constructor=constructor)
            if _is_deleted(text) or (child.spelling.startswith("operator=")):
                return None
            member_kind = "static method" if child.is_static_method() else "method"
            if kind == "FUNCTION_TEMPLATE" and any(
                token.spelling == "static"
                and token.location.offset < child.location.offset
                for token in child.get_tokens()
            ):
                member_kind = "static method"
            if constructor:
                member_kind = "constructor"
            if kind == "CONVERSION_FUNCTION" or child.spelling.startswith("operator"):
                member_kind = "operator"
            return self._member_entry(child, member_kind, text)
        if kind in ("TYPE_ALIAS_DECL", "TYPEDEF_DECL"):
            return self._member_entry(child, "type alias", self._statement_text(child))
        if (
            kind in ("STRUCT_DECL", "CLASS_DECL")
            and child.is_definition()
            and child.spelling
        ):
            return self._member_entry(child, "nested type", self._head_text(child))
        return None

    def _member_entry(
        self, child: ci.Cursor, kind: str, signature: str, *, trailing: bool = False
    ) -> ApiMember:
        return ApiMember(
            name=child.spelling,
            kind=kind,
            signature=signature,
            documentation=self._docs(child, trailing=trailing),
            defined_at=self._location(child),
        )

    def _add(
        self,
        cursor: ci.Cursor,
        name: str,
        kind: str,
        signature: str,
        *,
        members: list[ApiMember] | None = None,
    ) -> None:
        if self.config.namespace is not None:
            namespaces: list[str] = []
            parent = cursor.semantic_parent
            while parent is not None and parent.kind.name == "NAMESPACE":
                if parent.spelling == self.config.namespace:
                    break
                namespaces.insert(0, parent.spelling)
                parent = parent.semantic_parent
            name = "::".join([*namespaces, name])
        if name in self.seen:
            return
        self.seen.add(name)
        qualified = (
            f"{self.config.namespace}::{name}" if self.config.namespace else name
        )
        self.entries.append(
            _Entry(
                line=cursor.extent.start.line,
                export=ApiExport(
                    name=name,
                    kind=kind,
                    stability=self.config.stability,
                    signature=signature,
                    documentation=self._docs(cursor),
                    qualified_name=qualified,
                    defined_at=self._location(cursor),
                    members=members or [],
                ),
            )
        )

    # ---- sections ----------------------------------------------------------

    def _modules(self) -> list[ApiModule]:
        header = self.path.name
        sections = [(0, header, self._file_comment())]
        sections += [
            (number, match.group(1), "")
            for number, line in enumerate(self.lines, start=1)
            if (match := _DIVIDER.match(line))
        ]
        modules = []
        for index, (start, title, docs) in enumerate(sections):
            end = (
                sections[index + 1][0]
                if index + 1 < len(sections)
                else len(self.lines) + 1
            )
            exports = [e.export for e in self.entries if start <= e.line < end]
            if not exports:
                continue
            name = header if index == 0 else f"{header}: {title}"
            modules.append(
                ApiModule(
                    name=name,
                    title=title if index else header,
                    description=_first_sentence(docs)
                    if docs
                    else f"{title} ({header}).",
                    stability=self.config.stability,
                    status=self.config.stability.lower(),
                    documentation=docs,
                    exports=exports,
                )
            )
        return modules

    def _file_comment(self) -> str:
        block = []
        for line in self.lines:
            stripped = line.strip()
            if not stripped:
                if block:
                    break
                continue
            if not _is_comment_line(stripped) and not block:
                break
            if not _is_comment_line(stripped):
                break
            block.append(line)
        return _clean_comment(block)

    # ---- source text -------------------------------------------------------

    def _docs(self, cursor: ci.Cursor, *, trailing: bool = False) -> str:
        docs = self._leading_comment(cursor.extent.start.line - 2)
        if not docs and not trailing:
            # One comment above a run of declarations (no blank line between them)
            # documents the whole run: `rlmesh_last_error_message` and its
            # `_is_recoverable` sibling, or the RLMESH_F32..RLMESH_BOOL constants.
            index = cursor.extent.start.line - 2
            while (
                index >= 0
                and self.lines[index].strip()
                and not _is_comment_line(self.lines[index].strip())
            ):
                index -= 1
            if index < cursor.extent.start.line - 2:
                docs = self._leading_comment(index)
        if not docs and trailing:
            line = self.lines[cursor.extent.end.line - 1]
            match = re.search(
                r"(?:/\*(.*?)\*/|//(.*))\s*$", line[cursor.extent.end.column - 1 :]
            )
            if match:
                docs = (
                    (match.group(1) or match.group(2) or "").strip().lstrip("<").strip()
                )
        return docs

    def _leading_comment(self, index: int) -> str:
        """The comment block ending on 0-based line `index`, or "" if there is none."""
        block: list[str] = []
        in_block_comment = False
        while index >= 0:
            stripped = self.lines[index].strip()
            if in_block_comment:
                block.insert(0, self.lines[index])
                if stripped.startswith("/*"):
                    in_block_comment = False
                index -= 1
                continue
            if _is_trailing_comment(stripped):
                break  # code with a comment after it documents itself, not this
            if stripped.endswith("*/") and not _DIVIDER.match(stripped):
                block.insert(0, self.lines[index])
                in_block_comment = not stripped.startswith("/*")
                index -= 1
                continue
            if stripped.startswith("//") and not _DIVIDER.match(stripped):
                block.insert(0, self.lines[index])
                index -= 1
                continue
            break
        return _clean_comment(block)

    def _text(self, cursor: ci.Cursor) -> str:
        start, end = cursor.extent.start, cursor.extent.end
        if start.line == end.line:
            return self.lines[start.line - 1][start.column - 1 : end.column - 1]
        parts = [self.lines[start.line - 1][start.column - 1 :]]
        parts += self.lines[start.line : end.line - 1]
        parts.append(self.lines[end.line - 1][: end.column - 1])
        return "\n".join(parts)

    def _declaration_text(self, cursor: ci.Cursor, *, constructor: bool = False) -> str:
        """A function's declaration: its text up to the body or `;`.

        Read from the source rather than the cursor's extent, which libclang can
        end at the name when it skips bodies (`template <class F> auto and_then`),
        line by line until the declaration's own terminator, however long it is.
        Only a constructor has a member-initializer list to drop; elsewhere a
        top-level `:` is an expression (`-> std::integral_constant<int, c ? 1 : 2>`).
        """
        start = cursor.extent.start
        source = self.lines[start.line - 1][start.column - 1 :]
        index = start.line
        while True:
            text = _strip_line_comments(source)
            masked = _mask_strings(text)
            cut = _top_level_cut(masked, "{;")
            # A terminator inside a block comment still open at its line isn't one.
            if (cut < len(text) and "/*" not in masked[:cut]) or index >= len(
                self.lines
            ):
                break
            source += "\n" + self.lines[index]
            index += 1
        text = text[:cut]
        if constructor:
            init = _init_list_start(_mask_strings(text))
            if init is not None:
                text = text[:init]
        return _normalize(text)

    def _statement_text(self, cursor: ci.Cursor) -> str:
        return _normalize(_strip_line_comments(self._text(cursor)))

    def _head_text(self, cursor: ci.Cursor) -> str:
        text = _strip_line_comments(self._text(cursor))
        return _normalize(text[: _top_level_cut(text, "{")])

    def _location(self, cursor: ci.Cursor) -> SourceLocation:
        return SourceLocation(
            repo=self.config.source_repo,
            commit=self.config.source_commit,
            path=self.config.header,
            line=cursor.extent.start.line,
            end_line=cursor.extent.end.line,
        )


def _has_methods(cursor: ci.Cursor) -> bool:
    return any(
        c.kind.name in ("CXX_METHOD", "CONSTRUCTOR", "FUNCTION_TEMPLATE")
        for c in cursor.get_children()
    )


def _is_deleted(text: str) -> bool:
    return re.search(r"=\s*delete\s*;?\s*$", text) is not None


def _is_comment_line(stripped: str) -> bool:
    if _is_trailing_comment(stripped):
        return False
    return stripped.startswith(("/*", "*", "//")) or stripped.endswith("*/")


def _is_trailing_comment(stripped: str) -> bool:
    """Code followed by a comment on its line (`int x; /* why */`, `int x;  // why`)."""
    if stripped.startswith(("/*", "*", "//")):
        return False
    code = _mask_strings(stripped)
    if code.endswith("*/"):
        # A block comment's last line (`See https://rlmesh.dev */`) unless this
        # same line opened it after some code.
        return "/*" in code
    return "//" in code


def _clean_comment(block: list[str]) -> str:
    out = []
    for line in block:
        stripped = line.strip()
        stripped = re.sub(r"^/{2,}!?<?\s?", "", stripped)
        stripped = re.sub(r"^/\*+!?<?\s?", "", stripped)
        stripped = re.sub(r"\s*\*+/$", "", stripped)
        stripped = re.sub(r"^\*(?!/)\s?", "", stripped)
        out.append(stripped)
    text = "\n".join(out).strip()
    return re.sub(r"\n{3,}", "\n\n", text)


# A string or character literal, or a comment: matched together so a `//` or
# `/*` inside a literal (a URL in a default argument) stays code.
_LITERAL_OR_COMMENT = re.compile(
    r'"(?:\\.|[^"\\\n])*"' + r"|'(?:\\.|[^'\\\n])*'" + r"|/\*.*?\*/|//[^\n]*", re.S
)


def _strip_line_comments(text: str) -> str:
    return _LITERAL_OR_COMMENT.sub(
        lambda m: m.group(0) if m.group(0)[0] in "\"'" else " ", text
    )


def _mask_strings(text: str) -> str:
    """`text` with each literal's contents blanked, same length, for punctuation scans."""
    return _LITERAL_OR_COMMENT.sub(
        lambda m: m.group(0)[0] + "_" * (len(m.group(0)) - 2) + m.group(0)[-1]
        if m.group(0)[0] in "\"'"
        else m.group(0),
        text,
    )


def _top_level_cut(text: str, stops: str) -> int:
    depth = 0
    for index, char in enumerate(text):
        if char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
        elif depth == 0 and char in stops:
            return index
    return len(text)


def _init_list_start(text: str) -> int | None:
    """Where a constructor's member-initializer list starts: the first single `:`
    at the top level after the parameter list, or None."""
    depth = 0
    after_params = False
    for index, char in enumerate(text):
        if char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
            if depth == 0:
                after_params = True
        elif (
            char == ":"
            and depth == 0
            and after_params
            and text[index - 1 : index] != ":"
            and text[index + 1 : index + 2] != ":"
        ):
            return index
    return None


def _normalize(text: str) -> str:
    text = re.sub(r"\bRLMESH_API\s+", "", text)
    text = re.sub(r"\s+", " ", text).strip().rstrip(";").strip()
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"\s+,", ",", text)
    return re.sub(r"<\s+", "<", text)


def _first_sentence(docs: str) -> str:
    paragraph = docs.split("\n\n", 1)[0].replace("\n", " ").strip()
    match = re.match(r"(.+?[.!?])(\s|$)", paragraph)
    return match.group(1) if match else paragraph

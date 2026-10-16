from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from clang import cindex as ci
from rlmesh_api_surface.api_surface import REPO_ROOT, write_native_docs_api_surface
from rlmesh_api_surface.header_surface import (
    HeaderSurfaceConfig,
    _parse,
    collect_header_api_surface,
)
from rlmesh_api_surface.rust_surface import (
    RUSTDOC_FORMAT_VERSION,
    RustSurfaceConfig,
    _canonical_path,
    _Crate,
    _function_signature,
    _generics,
    _item_signature,
    _ty,
    _unwrap_intra_doc_links,
    collect_rust_api_surface,
)


def test_c_prototype_and_callback_parameters_are_named() -> None:
    header = "crates/rlmesh-capi/include/rlmesh.h"
    tu = _parse(
        REPO_ROOT / header,
        HeaderSurfaceConfig(header, "c", "c11", "Stable", None, "", None),
    )
    unnamed = [
        cursor.location.line
        for cursor in tu.cursor.walk_preorder()
        if cursor.kind == ci.CursorKind.PARM_DECL
        and cursor.location.file is not None
        and Path(cursor.location.file.name).resolve() == (REPO_ROOT / header).resolve()
        and not cursor.spelling
    ]
    assert unnamed == []
    functions = {
        cursor.spelling: [arg.spelling for arg in cursor.get_arguments()]
        for cursor in tu.cursor.get_children()
        if cursor.kind == ci.CursorKind.FUNCTION_DECL
    }
    assert functions["rlmesh_adapters_resolve"] == [
        "env_tags",
        "observation_space",
        "action_space",
        "model_spec",
        "options",
        "out",
    ]
    assert functions["rlmesh_run"] == ["model", "env", "options", "out"]
    assert functions["rlmesh_model_run"] == [
        "model",
        "env_or_address",
        "options",
        "out",
    ]
    # Output names follow the order the library writes them in.
    assert functions["rlmesh_text_limits"] == [
        "space",
        "out_min_length",
        "out_max_length",
        "out_charset",
    ]
    assert functions["rlmesh_reader_call"] == ["reader", "observation", "out"]

    parameters = [
        cursor
        for cursor in tu.cursor.walk_preorder()
        if cursor.kind == ci.CursorKind.PARM_DECL
        and cursor.location.file is not None
        and Path(cursor.location.file.name).resolve() == (REPO_ROOT / header).resolve()
    ]
    assert [p.spelling for p in parameters if re.search(r"_[12]$", p.spelling)] == []
    for name, args in functions.items():
        if name.endswith("_equals"):
            assert args[:2] == ["a", "b"], name
    assert functions["rlmesh_run_result_episodes_get"] == ["run_result", "i", "out"]
    assert functions["rlmesh_run_result_telemetry_get"] == ["run_result", "i", "out"]
    assert functions["rlmesh_env_server_new"][:2] == ["envs", "envs_len"]
    assert functions["rlmesh_remote_env_step"][0] == "remote_env"
    assert functions["rlmesh_remote_vector_env_step"][0] == "remote_vector_env"
    assert functions["rlmesh_model_new"][0] == "vtable"
    assert functions["rlmesh_model_vtable_describe"][0] == "vtable"
    assert functions["rlmesh_model_describe"][0] == "model"
    assert functions["rlmesh_env_factory_describe"][0] == "env_factory"
    # Deleters name the object they free; user_data/ctx are not exempt here.
    declarations = [
        cursor
        for cursor in tu.cursor.walk_preorder()
        if cursor.kind in (ci.CursorKind.PARM_DECL, ci.CursorKind.FIELD_DECL)
        and cursor.location.file is not None
        and Path(cursor.location.file.name).resolve() == (REPO_ROOT / header).resolve()
    ]
    for parameter in declarations:
        if parameter.spelling.startswith(("free_", "out_free_")):
            freed = parameter.spelling.removeprefix("out_").removeprefix("free_")
            callback_args = [
                child.spelling
                for child in parameter.get_children()
                if child.kind == ci.CursorKind.PARM_DECL
            ]
            assert callback_args == [freed], parameter.spelling


def test_rust_glob_reexports_include_private_module_contents(
    tmp_path: Path, monkeypatch
) -> None:
    from rlmesh_api_surface import rust_surface

    data = _minimal_rustdoc()
    data["index"]["0"]["inner"]["module"]["items"] = [3, 4]
    data["index"]["1"]["visibility"] = "crate"
    data["index"]["3"]["inner"]["use"].update({"name": "*", "id": 1, "is_glob": True})
    data["index"]["4"] = {
        **data["index"]["3"],
        "id": 4,
    }
    # A recursive glob must terminate and must not duplicate the module page.
    data["index"]["1"]["inner"]["module"]["items"].append(5)
    data["index"]["5"] = {
        **data["index"]["3"],
        "id": 5,
        "inner": {"use": {"name": "*", "source": "crate::*", "id": 0, "is_glob": True}},
    }
    (tmp_path / "rlmesh.json").write_text(json.dumps(data))
    monkeypatch.setattr(
        rust_surface, "_cargo_workspace", lambda root: ({"rlmesh"}, tmp_path)
    )
    modules = collect_rust_api_surface(
        repo_root=tmp_path,
        config=RustSurfaceConfig("rlmesh", "Stable", (), "", None),
        json_dir=tmp_path,
    )
    assert [module.name for module in modules] == ["rlmesh"]
    assert [
        export.qualified_name for module in modules for export in module.exports
    ] == ["rlmesh::RemoteEnv"]


def test_c_field_keeps_its_own_trailing_comment() -> None:
    # The previous field's trailing comment must not read as the end of a block
    # comment and drag the struct's docs (and source lines) onto this field.
    _, exports = _header("rlmesh.h", "c", "c11", None)
    fields = {member.name: member for member in exports["RLMeshRunOptions"].members}
    assert fields["instruction"].documentation == "NULL = None; copied."
    assert (
        fields["view"].documentation
        == "NULL = None; copied and validated by the facade."
    )


def test_cpp_signatures_read_past_a_truncated_extent() -> None:
    # libclang ends this template's extent at its name when skipping bodies.
    _, exports = _header("rlmesh.hpp", "c++", "c++20", "rlmesh")
    (from_config,) = [m for m in exports["Model"].members if m.name == "from_config"]
    assert from_config.signature == (
        "template <class P> static Model from_config(const Kwargs& config)"
    )
    # A constructor stops before its member-initializer list.
    (error,) = [m for m in exports["RuntimeError"].members if m.kind == "constructor"]
    assert error.signature == ("explicit RuntimeError(const std::string& message)")
    constructors = [
        m.signature for m in exports["Model"].members if m.kind == "constructor"
    ]
    assert (
        "template <class P> requires Policy<P> "
        "explicit Model(std::shared_ptr<P> source, ModelOptions options = {})"
    ) in constructors


def test_cpp_partial_specialization_documents_its_members() -> None:
    _, exports = _header("rlmesh.hpp", "c++", "c++20", "rlmesh")
    function = exports["SharedFunction<R (A...), EmptyError>"]
    assert function.qualified_name == "rlmesh::SharedFunction<R (A...), EmptyError>"
    assert "SharedFunction() = default" in [m.signature for m in function.members]
    assert (
        function.signature == "template <class R, class EmptyError, class... A> "
        "class SharedFunction<R(A...), EmptyError>"
    )


def test_rust_tuple_struct_shows_its_fields() -> None:
    crate = _Crate(_minimal_rustdoc())
    generics = {"params": [], "where_predicates": []}
    crate.index["5"] = {
        "id": 5,
        "name": "0",
        "visibility": "public",
        "inner": {"struct_field": {"primitive": "i32"}},
    }
    tuple_struct = {
        "inner": {
            "struct": {"kind": {"tuple": [5, None]}, "generics": generics, "impls": []}
        }
    }
    assert (
        _item_signature(tuple_struct, "Mode", crate)
        == "struct Mode(pub i32, /* private field */)"
    )


def test_rust_higher_ranked_where_predicate_keeps_its_binder() -> None:
    lifetime = {"name": "'a", "kind": {"lifetime": {"outlives": []}}}
    reference = {
        "borrowed_ref": {
            "lifetime": "'a",
            "is_mutable": False,
            "type": {"generic": "T"},
        }
    }
    bound = {
        "trait_bound": {
            "trait": {"path": "IntoIterator", "id": 9, "args": None},
            "generic_params": [],
            "modifier": "none",
        }
    }
    generics = {
        "params": [],
        "where_predicates": [
            {
                "bound_predicate": {
                    "type": reference,
                    "bounds": [bound],
                    "generic_params": [lifetime],
                }
            }
        ],
    }
    assert _generics(generics) == ("", " where for<'a> &'a T: IntoIterator")


def _header(header: str, language: str, standard: str, namespace: str | None):
    modules = collect_header_api_surface(
        repo_root=REPO_ROOT,
        config=HeaderSurfaceConfig(
            header=f"crates/rlmesh-capi/include/{header}",
            language=language,
            standard=standard,
            stability="Experimental",
            namespace=namespace,
            source_repo="ArenaX-Labs/rlmesh",
            source_commit=None,
        ),
    )
    return modules, {
        export.name: export for module in modules for export in module.exports
    }


def test_c_header_sections_and_declarations() -> None:
    modules, exports = _header("rlmesh.h", "c", "c11", None)

    assert [module.name for module in modules][:4] == [
        "rlmesh.h",
        "rlmesh.h: status + errors",
        "rlmesh.h: build info, editions, warnings",
        "rlmesh.h: dtype + tensor",
    ]
    status = exports["RLMeshStatus"]
    assert status.kind == "enum"
    assert {member.signature for member in status.members} >= {
        "RLMESH_OK = 0",
        "RLMESH_ERR_INTERNAL = 99",
    }
    assert exports["RLMeshValue"].kind == "opaque type"
    assert (
        exports["rlmesh_last_error_message"].signature
        == "const char* rlmesh_last_error_message(void)"
    )
    # One comment documents the run of declarations below it.
    assert exports["rlmesh_last_error_is_recoverable"].documentation.startswith(
        "Most recent failing call"
    )
    tensor = exports["RLMeshTensor"]
    assert tensor.kind == "opaque type"
    assert tensor.members == []
    options = exports["RLMeshToDlpackOptions"]
    assert options.kind == "struct"
    assert {member.signature for member in options.members} >= {
        "size_t struct_size",
        "bool copy",
        "int32_t dl_device_type",
    }
    assert exports["RLMESH_TO_DLPACK_OPTIONS_INIT"].kind == "macro"
    sections = {
        export.name: module.name for module in modules for export in module.exports
    }
    assert sections["RLMeshTensor"] == "rlmesh.h: dtype + tensor"
    assert sections["RLMeshToDlpackOptions"] == "rlmesh.h: dtype + tensor"
    assert (
        sections["RLMeshRunOptions"]
        == "rlmesh.h: Running: facade targets, observers and results"
    )
    # Header plumbing stays out of the reference.
    assert "RLMESH_API" not in exports
    assert "RLMESH_H" not in exports


def test_cpp_header_classes_and_members() -> None:
    modules, exports = _header("rlmesh.hpp", "c++", "c++20", "rlmesh")

    assert modules[0].name == "rlmesh.hpp"
    model = exports["Model"]
    assert model.qualified_name == "rlmesh::Model"
    signatures = {member.name: member for member in model.members}
    assert signatures["from_config"].kind == "static method"
    assert (
        signatures["from_config"].signature
        == "template <class P> static Model from_config(const Kwargs& config)"
    )
    assert exports["EnvFactory"].signature == (
        "template <class F> class EnvFactory : public EnvFactoryBase"
    )
    assert exports["Policy"].kind == "concept"
    assert "concept Policy =" in exports["Policy"].signature
    assert (
        exports["RuntimeError"].signature
        == "class RuntimeError : public std::runtime_error"
    )
    assert exports["ext::CancelledError"].signature == "class CancelledError"
    assert exports["spaces::Box"].qualified_name == "rlmesh::spaces::Box"
    assert exports["specs::EnvContract"].qualified_name == "rlmesh::specs::EnvContract"
    assert (
        exports["adapters::ModelSpec"].qualified_name == "rlmesh::adapters::ModelSpec"
    )
    assert exports["recorder::SCHEMA"].kind == "constant"
    assert exports["adapters::PARTS"].kind == "function"
    assert exports["adapters::ROTATION_DIMS"].kind == "function"
    assert "Result" not in exports
    assert "Error" not in exports
    # Internal namespaces stay out, including those reopened later in the header.
    assert not any(
        part.startswith("detail") for name in exports for part in name.split("::")
    )


def test_cpp_snapshot_uses_the_manifest_standard(tmp_path: Path) -> None:
    assert (
        write_native_docs_api_surface(
            language="cpp",
            output_dir=tmp_path,
            repo_root=REPO_ROOT,
            manifest_path=REPO_ROOT / "rlmesh.toml",
            package_version="dev",
            check=False,
        )
        == 0
    )
    snapshot = json.loads((tmp_path / "dev.json").read_text(encoding="utf-8"))
    assert snapshot["kind"] == "rlmesh.cpp-api-surface"
    exports = {
        export["qualifiedName"]
        for module in snapshot["surface"]["modules"]
        for export in module["exports"]
    }
    assert exports >= {"rlmesh::Model", "rlmesh::Policy", "rlmesh::spaces::Box"}


def test_rust_signatures_render_like_source() -> None:
    connect = {
        "sig": {
            "inputs": [
                [
                    "address",
                    {
                        "borrowed_ref": {
                            "lifetime": None,
                            "is_mutable": False,
                            "type": {"primitive": "str"},
                        }
                    },
                ]
            ],
            "output": {
                "resolved_path": {
                    "path": "Result",
                    "id": 1,
                    "args": {
                        "angle_bracketed": {
                            "args": [{"type": {"generic": "Self"}}],
                            "constraints": [],
                        }
                    },
                }
            },
            "is_c_variadic": False,
        },
        "generics": {"params": [], "where_predicates": []},
        "header": {
            "is_const": False,
            "is_unsafe": False,
            "is_async": True,
            "abi": "Rust",
        },
        "has_body": True,
    }
    assert (
        _function_signature("connect", connect)
        == "async fn connect(address: &str) -> Result<Self>"
    )
    # A path relative to an internal crate renders as its name.
    assert (
        _ty({"resolved_path": {"path": "crate::dtype::DType", "id": 2, "args": None}})
        == "DType"
    )
    assert _ty({"slice": {"primitive": "u8"}}) == "[u8]"


def test_rust_items_document_at_their_module() -> None:
    assert (
        _canonical_path(["rlmesh::RemoteEnv", "rlmesh::env::RemoteEnv"])
        == "rlmesh::env::RemoteEnv"
    )
    assert (
        _canonical_path(
            [
                "rlmesh::SpaceSpec",
                "rlmesh::spaces::SpaceSpec",
                "rlmesh::spaces::types::SpaceSpec",
            ]
        )
        == "rlmesh::spaces::SpaceSpec"
    )
    assert _canonical_path(["rlmesh::Error"]) == "rlmesh::Error"


def test_rust_intra_doc_links_keep_their_text() -> None:
    links = {"`EnvServer`": 1, "crate::Error": 2}
    docs = "Use [`EnvServer`], or [the error](crate::Error). See [docs](https://rlmesh.dev)."
    assert (
        _unwrap_intra_doc_links(docs, links)
        == "Use `EnvServer`, or the error. See [docs](https://rlmesh.dev)."
    )


def _minimal_rustdoc() -> dict[str, object]:
    def item(
        item_id: int, name: str | None, inner: dict[str, object], docs: str = ""
    ) -> dict[str, object]:
        return {
            "id": item_id,
            "crate_id": 0,
            "name": name,
            "span": {
                "filename": "crates/rlmesh/src/lib.rs",
                "begin": [1, 1],
                "end": [2, 1],
            },
            "visibility": "public",
            "docs": docs,
            "links": {},
            "attrs": [],
            "deprecation": None,
            "inner": inner,
        }

    generics = {"params": [], "where_predicates": []}
    return {
        "format_version": RUSTDOC_FORMAT_VERSION,
        "root": 0,
        "index": {
            "0": item(
                0,
                "rlmesh",
                {"module": {"is_crate": True, "items": [1, 3], "is_stripped": False}},
                "The facade.",
            ),
            "1": item(
                1,
                "env",
                {"module": {"is_crate": False, "items": [2], "is_stripped": False}},
                "Envs.",
            ),
            "2": item(
                2,
                "RemoteEnv",
                {"struct": {"kind": "unit", "generics": generics, "impls": []}},
                "A client.",
            ),
            "3": item(
                3,
                "RemoteEnv",
                {
                    "use": {
                        "source": "env::RemoteEnv",
                        "name": "RemoteEnv",
                        "id": 2,
                        "is_glob": False,
                    }
                },
            ),
        },
        "paths": {
            "2": {
                "crate_id": 0,
                "path": ["rlmesh", "env", "RemoteEnv"],
                "kind": "struct",
            }
        },
        "external_crates": {},
    }


def test_rust_snapshot_envelope_is_early_access(tmp_path: Path) -> None:
    json_dir = tmp_path / "rustdoc"
    json_dir.mkdir()
    (json_dir / "rlmesh.json").write_text(
        json.dumps(_minimal_rustdoc()), encoding="utf-8"
    )

    status = write_native_docs_api_surface(
        language="rust",
        output_dir=tmp_path / "out",
        repo_root=REPO_ROOT,
        manifest_path=REPO_ROOT / "rlmesh.toml",
        package_version="dev",
        check=False,
        rustdoc_json_dir=json_dir,
    )

    assert status == 0
    snapshot = json.loads((tmp_path / "out" / "dev.json").read_text(encoding="utf-8"))
    assert snapshot["kind"] == "rlmesh.rust-api-surface"
    assert snapshot["access"]["level"] == "early-access"
    assert snapshot["access"]["stability"] == "Preview"
    # The root re-export is an alias, not a second page; the empty root is dropped.
    assert [module["name"] for module in snapshot["surface"]["modules"]] == [
        "rlmesh::env"
    ]
    remote_env = snapshot["surface"]["modules"][0]["exports"][0]
    assert remote_env["qualifiedName"] == "rlmesh::env::RemoteEnv"
    assert "Also available as `rlmesh::RemoteEnv`." in remote_env["documentation"]


def test_rust_rejects_an_unknown_rustdoc_format(tmp_path: Path) -> None:
    data = _minimal_rustdoc()
    data["format_version"] = RUSTDOC_FORMAT_VERSION + 1
    (tmp_path / "rlmesh.json").write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="rustdoc JSON format"):
        write_native_docs_api_surface(
            language="rust",
            output_dir=tmp_path / "out",
            repo_root=REPO_ROOT,
            manifest_path=REPO_ROOT / "rlmesh.toml",
            package_version="dev",
            check=False,
            rustdoc_json_dir=tmp_path,
        )


def _fixture(tmp_path: Path, source: str, language: str = "c++"):
    name = "fixture.hpp" if language == "c++" else "fixture.h"
    (tmp_path / name).write_text(source, encoding="utf-8")
    modules = collect_header_api_surface(
        repo_root=tmp_path,
        config=HeaderSurfaceConfig(
            header=name,
            language=language,
            standard="c++20" if language == "c++" else "c11",
            stability="Experimental",
            namespace="ns" if language == "c++" else None,
            source_repo="ArenaX-Labs/rlmesh",
            source_commit=None,
        ),
    )
    return {export.name: export for module in modules for export in module.exports}


def test_specialization_naming_ignores_declaration_order(tmp_path: Path) -> None:
    exports = _fixture(
        tmp_path,
        "namespace ns {\n"
        "template <class T> class Result;\n"
        "/// The void result.\n"
        "template <> class Result<void> { public: bool ok() const; };\n"
        "/// A value or an error.\n"
        "template <class T> class Result { public: T value() const; };\n"
        "}\n",
    )
    assert exports["Result"].qualified_name == "ns::Result"
    assert [m.name for m in exports["Result"].members] == ["value"]
    assert exports["Result<void>"].qualified_name == "ns::Result<void>"
    assert [m.name for m in exports["Result<void>"].members] == ["ok"]


def test_cpp_reopened_namespaces_keep_distinct_exports(tmp_path: Path) -> None:
    exports = _fixture(
        tmp_path,
        "namespace ns {\n"
        "namespace detail { struct Hidden {}; }\n"
        "namespace a { struct Options { int count; }; }\n"
        "}\n"
        "namespace ns {\n"
        "namespace a { int open(); }\n"
        "namespace b { struct Options { bool enabled; }; }\n"
        "namespace b { namespace detail_impl { struct Hidden {}; } }\n"
        "namespace alias = a;\n"
        "template <class T> using List = T*;\n"
        "template <class T> concept Policy = requires(T p) { p.predict(); };\n"
        "}\n",
    )
    assert set(exports) == {
        "a::Options",
        "a::open",
        "b::Options",
        "alias",
        "List",
        "Policy",
    }
    assert exports["a::Options"].qualified_name == "ns::a::Options"
    assert [m.name for m in exports["a::Options"].members] == ["count"]
    assert [m.name for m in exports["b::Options"].members] == ["enabled"]
    assert exports["alias"].signature == "namespace alias = a"
    assert exports["List"].signature == "template <class T> using List = T*"
    assert exports["Policy"].kind == "concept"
    assert exports["Policy"].signature.endswith("requires(T p) { p.predict(); }")


def test_url_ending_a_block_comment_keeps_the_docs(tmp_path: Path) -> None:
    exports = _fixture(
        tmp_path,
        "/* Opens a session.\n   See https://rlmesh.dev */\nint open_session(void);\n",
        language="c",
    )
    assert (
        exports["open_session"].documentation
        == "Opens a session.\nSee https://rlmesh.dev"
    )


def test_long_declaration_is_read_to_its_end(tmp_path: Path) -> None:
    params = ",\n".join(f"    int p{i}" for i in range(46))
    exports = _fixture(tmp_path, f"/// Many.\nint many(\n{params});\n", language="c")
    assert exports["many"].signature.endswith("int p44, int p45)")


def test_only_constructors_lose_an_initializer_list(tmp_path: Path) -> None:
    exports = _fixture(
        tmp_path,
        "#include <type_traits>\nnamespace ns {\n/// A thing.\nclass Thing {\n public:\n"
        "  /// Builds.\n  Thing(int v) : v_(v) {}\n"
        "  /// Picks.\n  auto pick() -> std::integral_constant<int, true ? 1 : 2>;\n"
        '  /// Greets.\n  void greet(const char* url = "https://x;{") const;\n'
        " private:\n  int v_;\n};\n}\n",
    )
    signatures = {m.name: m.signature for m in exports["Thing"].members}
    assert signatures["Thing"] == "Thing(int v)"
    assert (
        signatures["pick"] == "auto pick() -> std::integral_constant<int, true ? 1 : 2>"
    )
    assert signatures["greet"] == 'void greet(const char* url = "https://x;{") const'


def test_library_defines_preserve_valueless_configuration(monkeypatch) -> None:
    from types import SimpleNamespace

    from rlmesh_api_surface import header_surface

    monkeypatch.setenv("CXX", "custom-cxx --target=test")
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert kwargs == {"capture_output": True, "text": True, "check": True}
        return SimpleNamespace(
            stdout=(
                "#define _LIBCPP_ABI_VERSION 1\n"
                "#define _LIBCPP_HAS_NO_THREADS\n"
                "#define _LIBCPP_EMPTY \n"
                "#define __COMPILER_BUILTIN__ 1\n"
            )
        )

    monkeypatch.setattr(header_surface.subprocess, "run", run)
    assert header_surface._library_defines("c++") == [
        "-D_LIBCPP_ABI_VERSION=1",
        "-D_LIBCPP_HAS_NO_THREADS",
        "-D_LIBCPP_EMPTY",
    ]
    assert calls == [
        ["custom-cxx", "--target=test", "-dM", "-E", "-x", "c++", "/dev/null"]
    ]

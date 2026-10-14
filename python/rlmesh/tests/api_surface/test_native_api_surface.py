from __future__ import annotations

import json
from pathlib import Path

import pytest
from rlmesh_api_surface.api_surface import REPO_ROOT, write_native_docs_api_surface
from rlmesh_api_surface.header_surface import (
    HeaderSurfaceConfig,
    collect_header_api_surface,
)
from rlmesh_api_surface.rust_surface import (
    RUSTDOC_FORMAT_VERSION,
    _canonical_path,
    _Crate,
    _function_signature,
    _generics,
    _item_signature,
    _ty,
    _unwrap_intra_doc_links,
)


def test_c_field_keeps_its_own_trailing_comment() -> None:
    # The previous field's `/* required */` must not read as the end of a block
    # comment and drag the struct's docs (and source lines) onto this field.
    _, exports = _header("rlmesh.h", "c", "c11", None)
    fields = {member.name: member for member in exports["RLMeshEnvConfig"].members}
    assert fields["observation_space"].documentation == "required"
    assert fields["action_space"].documentation == "required"


def test_cpp_signatures_read_past_a_truncated_extent() -> None:
    # libclang ends this template's extent at its name when skipping bodies.
    _, exports = _header("rlmesh.hpp", "c++", "c++17", "rlmesh")
    (and_then,) = [m for m in exports["Result"].members if m.name == "and_then"]
    assert and_then.signature.startswith("template <class F> auto and_then(F&& fn) &&")
    # A constructor stops before its member-initializer list.
    (error,) = [m for m in exports["Error"].members if m.kind == "constructor"]
    assert error.signature == (
        "Error(RLMeshStatus code, std::string message, bool recoverable = false)"
    )


def test_cpp_explicit_specialization_documents_separately() -> None:
    _, exports = _header("rlmesh.hpp", "c++", "c++17", "rlmesh")
    void = exports["Result<void>"]
    assert void.qualified_name == "rlmesh::Result<void>"
    assert "Result() = default" in [member.signature for member in void.members]
    assert (
        exports["Result"].signature == "template <class T> class [[nodiscard]] Result"
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

    assert [module.name for module in modules][:3] == [
        "rlmesh.h",
        "rlmesh.h: status + errors",
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
    assert tensor.kind == "struct"
    assert "const void* data" in [member.signature for member in tensor.members]
    # Header plumbing stays out of the reference.
    assert "RLMESH_API" not in exports
    assert "RLMESH_H" not in exports


def test_cpp_header_classes_and_members() -> None:
    modules, exports = _header("rlmesh.hpp", "c++", "c++17", "rlmesh")

    assert modules[0].name == "rlmesh.hpp: errors + results"
    model = exports["Model"]
    assert model.qualified_name == "rlmesh::Model"
    signatures = {member.name: member for member in model.members}
    assert signatures["from_predict"].kind == "static method"
    assert (
        signatures["from_predict"].signature
        == "static Result<Model> from_predict(PredictFn predict)"
    )
    assert (
        exports["Result"].signature == "template <class T> class [[nodiscard]] Result"
    )
    # Nothing from rlmesh::detail is documented.
    assert not any(name.startswith("detail") for name in exports)


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
            standard="c++17" if language == "c++" else "c11",
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

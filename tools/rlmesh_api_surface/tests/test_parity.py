from __future__ import annotations

from copy import deepcopy

import pytest
from rlmesh_api_surface.parity import (
    ERROR_BASES,
    Item,
    check,
    extension_scope,
    python_entries,
    snake,
)


def fixture():
    expected = {"rlmesh:run": {"kind": "function", "params": {"episodes": "1"}}}
    surfaces = {lang: {} for lang in ("rust", "c", "cpp")}
    manifest = {
        "rlmesh:run": {"area": "running", "parameters": {"episodes": {"default": "1"}}}
    }
    for lang, owner, function in (
        ("rust", "rlmesh::RunOptions", "rlmesh::run"),
        ("c", "RLMeshRunOptions", "rlmesh_run"),
        ("cpp", "rlmesh::RunOptions", "rlmesh::run"),
    ):
        items = surfaces[lang]
        items[function] = Item("function", params={"options": owner + "*"})
        items[owner] = Item(
            "struct",
            fields=["struct_size", "episodes"] if lang == "c" else ["episodes"],
        )
        items[owner + "::episodes"] = Item("field", default="1")
        manifest["rlmesh:run"][lang] = {
            "status": "present",
            "path": function,
            "also": [owner],
            "params": {"episodes": owner + "#episodes"},
        }
    surfaces["c"]["RLMESH_RUN_OPTIONS_INIT"] = Item(
        "macro", "RLMESH_RUN_OPTIONS_INIT { sizeof(RLMeshRunOptions), 1 }"
    )
    surfaces["c"]["RLMESH_RUN_OPTIONS_MIN_SIZE"] = Item("macro")
    surfaces["c"]["RLMeshDevice"] = Item(
        "enum", values={"RLMESH_DEVICE_INVALID": 0, "RLMESH_DEVICE_CPU": 1}
    )
    surfaces["c"]["RLMeshErrorClass"] = Item(
        "enum", values={"RLMESH_ERROR_CLASS_INVALID": 0}
    )
    for index, (name, base) in enumerate(ERROR_BASES.items(), start=1):
        surfaces["rust"]["rlmesh::ErrorClass::" + name] = Item(
            "variant", base="" if base.startswith("std::") else base
        )
        surfaces["c"]["RLMeshErrorClass"].values[
            "RLMESH_ERROR_CLASS_" + snake(name).upper().replace("RL_MESH", "RLMESH")
        ] = index
        cpp = "rlmesh::" + (
            "ext::CancelledError" if name == "KeyboardInterrupt" else name
        )
        surfaces["cpp"][cpp] = Item("class", base=base)
    # All synthetic support items are explicitly inventoried on the mapping.
    for lang, items in surfaces.items():
        manifest["rlmesh:run"][lang]["also"] = list(
            items.keys() - {manifest["rlmesh:run"][lang]["path"]}
        )
    return expected, manifest, surfaces


def test_passing_case():
    assert check(*fixture()) == []


@pytest.mark.parametrize("condition", range(1, 8))
def test_each_failure_condition(condition):
    expected, manifest, surfaces = fixture()
    if condition == 1:
        expected["rlmesh:run"]["params"]["new_keyword"] = "None"
    elif condition == 2:
        surfaces["rust"]["rlmesh::run"].params["wrong_name"] = "u64"
    elif condition == 3:
        manifest["rlmesh:run"]["rust"].update(status="missing", gap="G-1")
    elif condition == 4:
        surfaces["rust"]["rlmesh::unfenced"] = Item("function")
    elif condition == 5:
        surfaces["c"][
            "RLMESH_RUN_OPTIONS_INIT"
        ].signature = "RLMESH_RUN_OPTIONS_INIT { sizeof(RLMeshRunOptions), 2 }"
    elif condition == 6:
        surfaces["c"]["RLMeshDevice"].values = {"RLMESH_DEVICE_CPU": 0}
    elif condition == 7:
        surfaces["cpp"]["rlmesh::ConnectionError"].base = "RLMeshException"
    assert any(
        e.startswith(f"E-2.{condition}:") for e in check(expected, manifest, surfaces)
    )


def test_missing_member_and_parameter_mapping():
    expected, manifest, surfaces = fixture()
    expected["rlmesh:RunResult.reward"] = {"kind": "property", "params": {}}
    del manifest["rlmesh:run"]["cpp"]["params"]["episodes"]
    errors = check(expected, manifest, surfaces)
    assert any("RunResult.reward: no mapping" in e for e in errors)
    assert any("parameter episodes has no mapping" in e for e in errors)


@pytest.mark.parametrize(
    "mutation", ["size", "init", "minimum", "cpp_default", "other", "error"]
)
def test_additional_static_guards(mutation):
    expected, manifest, surfaces = fixture()
    if mutation == "size":
        surfaces["c"]["RLMeshRunOptions"].fields.reverse()
    elif mutation == "init":
        del surfaces["c"]["RLMESH_RUN_OPTIONS_INIT"]
    elif mutation == "minimum":
        del surfaces["c"]["RLMESH_RUN_OPTIONS_MIN_SIZE"]
    elif mutation == "cpp_default":
        surfaces["cpp"]["rlmesh::RunOptions::episodes"].default = "2"
    elif mutation == "other":
        surfaces["c"]["RLMeshDevice"].values["RLMESH_DEVICE_OTHER"] = 1000
    elif mutation == "error":
        del surfaces["rust"]["rlmesh::ErrorClass::TypeError"]
    assert check(expected, manifest, surfaces)


def test_bridge_requires_rule_and_missing_requires_gap():
    expected, manifest, surfaces = fixture()
    manifest["rlmesh:run"]["rust"] = {"status": "bridge"}
    manifest["rlmesh:run"]["c"] = {"status": "missing", "path": "unknown"}
    errors = check(expected, manifest, surfaces)
    assert any("bridge requires R-24" in e for e in errors)
    assert any("missing requires a gap" in e for e in errors)


def test_snapshot_filters_stability_and_keeps_parameter_defaults():
    snapshot = {
        "rlmesh": {
            "objects": {
                "run": {
                    "kind": "function",
                    "stability": "Stable",
                    "signature": "run(model: object, *, seeds: list[int] | None = None)",
                },
                "X": {
                    "kind": "class",
                    "stability": "Stable",
                    "members": {
                        "new": {
                            "kind": "function",
                            "signature": "new()",
                            "stability": "Experimental",
                        },
                        "reward": {"kind": "property"},
                    },
                },
                "experimental": {"kind": "function", "stability": "Experimental"},
            }
        }
    }
    assert python_entries(snapshot) == {
        "rlmesh:run": {"kind": "function", "params": {"model": None, "seeds": "None"}},
        "rlmesh:X": {"kind": "class", "params": {}},
        "rlmesh:X.reward": {"kind": "property", "params": {}},
    }


def test_manifest_is_not_mutated():
    inputs = fixture()
    original = deepcopy(inputs)
    check(*inputs)
    assert inputs == original


def test_new_member_on_mapped_type_needs_inventory_entry():
    expected, manifest, surfaces = fixture()
    surfaces["rust"]["rlmesh::RunOptions::new_field"] = Item("field")
    assert any(e.startswith("E-2.4:") for e in check(expected, manifest, surfaces))


def test_mapped_field_name_must_follow_python():
    expected, manifest, surfaces = fixture()
    surfaces["c"]["RLMeshRunOptions::count"] = Item("field")
    manifest["rlmesh:run"]["c"]["params"]["episodes"] = "RLMeshRunOptions#count"
    assert any("violates R-4" in e for e in check(expected, manifest, surfaces))


def test_rust_error_base_must_follow_hierarchy():
    expected, manifest, surfaces = fixture()
    surfaces["rust"]["rlmesh::ErrorClass::ConnectionError"].base = "RLMeshException"
    assert any(e.startswith("E-2.7:") for e in check(expected, manifest, surfaces))


def test_nested_extension_namespace_does_not_bypass_fencing():
    expected, manifest, surfaces = fixture()
    surfaces["rust"]["rlmesh::spaces::ext::hidden_extra"] = Item("function")
    assert any(e.startswith("E-2.4:") for e in check(expected, manifest, surfaces))


def test_stale_python_default_fails_even_when_native_matches_manifest():
    expected, manifest, surfaces = fixture()
    expected["rlmesh:run"]["params"]["episodes"] = "2"
    assert any(
        "stale relative to Python" in e for e in check(expected, manifest, surfaces)
    )


def test_only_header_drivers_see_local_override(monkeypatch):
    import os

    from rlmesh_api_surface.parity import header_compilers

    monkeypatch.setenv("CC", "cc")
    monkeypatch.setenv("RLMESH_PARITY_HEADER_CC", "zig cc")
    with header_compilers("rust"):
        assert os.environ["CC"] == "cc"
    with header_compilers("c"):
        assert os.environ["CC"] == "zig cc"
    assert os.environ["CC"] == "cc"


def test_zero_enum_value_cannot_alias_a_real_input():
    expected, manifest, surfaces = fixture()
    surfaces["c"]["RLMeshDevice"].values["RLMESH_DEVICE_CPU"] = 0
    assert any(e.startswith("E-2.6:") for e in check(expected, manifest, surfaces))


def test_deleter_parameter_names_the_freed_object():
    expected, manifest, surfaces = fixture()
    surfaces["c"]["rlmesh_run"].callbacks["free_value"] = ["user_data"]
    assert any(
        "must name its freed argument value" in e
        for e in check(expected, manifest, surfaces)
    )


def test_existing_but_wrong_callable_cannot_be_mapped():
    expected, manifest, surfaces = fixture()
    surfaces["rust"]["rlmesh::wrong_name"] = surfaces["rust"]["rlmesh::run"]
    manifest["rlmesh:run"]["rust"]["path"] = "rlmesh::wrong_name"
    assert any(
        "callable name wrong_name" in e for e in check(expected, manifest, surfaces)
    )


@pytest.mark.parametrize(
    "name,expected",
    [
        ("type", "type_"),
        ("export", "export_"),
        ("class", "class_"),
        ("namespace", "namespace_"),
        ("macro_rules", "macro_rules_"),
    ],
)
def test_reserved_names_have_one_native_spelling(name, expected):
    from rlmesh_api_surface.parity import native_name

    assert [native_name(name, language) for language in ("rust", "c", "cpp")] == [
        expected
    ] * 3


def test_length_companion_requires_a_pointer():
    from rlmesh_api_surface.parity import structural

    assert structural(
        "values_len", "size_t", {"values": "const double*"}, "rlmesh:run", "c"
    )
    assert not structural(
        "values_len", "size_t", {"values": "double"}, "rlmesh:run", "c"
    )


def test_seconds_suffix_requires_native_duration_type():
    expected, manifest, surfaces = fixture()
    expected["rlmesh:run"]["params"]["timeout_seconds"] = "None"
    manifest["rlmesh:run"]["parameters"]["timeout_seconds"] = {"default": "None"}
    manifest["rlmesh:run"]["rust"]["params"]["timeout_seconds"] = (
        "rlmesh::RunOptions#timeout"
    )
    surfaces["rust"]["rlmesh::RunOptions::timeout"] = Item(
        "field", "timeout: Option<f64>"
    )
    assert any(
        "only for a duration type" in e for e in check(expected, manifest, surfaces)
    )


def test_only_the_receiving_handle_is_exempt():
    from rlmesh_api_surface.parity import structural

    params = {"model": "RLMeshModel*", "value": "RLMeshValue*"}
    assert structural("model", params["model"], params, "rlmesh:Model.run", "c")
    assert not structural("value", params["value"], params, "rlmesh:Model.run", "c")


def test_new_prelude_alias_needs_its_own_inventory_entry():
    expected, manifest, surfaces = fixture()
    surfaces["rust"]["rlmesh::prelude::Surprise"] = Item(
        "alias", alias_of="rlmesh::RunOptions"
    )
    assert any(e.startswith("E-2.4:") for e in check(expected, manifest, surfaces))


def test_new_public_trait_implementation_needs_inventory_entry():
    expected, manifest, surfaces = fixture()
    surfaces["rust"]["rlmesh::RunOptions::From<WireOptions>"] = Item(
        "trait implementation"
    )
    assert any(e.startswith("E-2.4:") for e in check(expected, manifest, surfaces))


@pytest.mark.parametrize("rule", [None, ["R-4"]])
def test_review_exempt_needs_own_decision(rule):
    expected, manifest, surfaces, _ = reset_fixture("cpp")
    mapping = manifest["rlmesh:Binding.reset"]["cpp"]
    mapping.update(status="exempt")
    manifest["rlmesh:Binding.reset"]["rule"] = ["D-8"]
    mapping.pop("rule", None)
    if rule is not None:
        mapping["rule"] = rule
    del surfaces["cpp"][mapping["path"]]
    assert any("exemption requires" in e for e in check(expected, manifest, surfaces))


def test_review_bridge_cannot_hide_deleted_shared_method():
    expected, manifest, surfaces, _ = reset_fixture("cpp")
    mapping = manifest["rlmesh:Binding.reset"]["cpp"]
    del surfaces["cpp"][mapping["path"]]
    mapping.update(status="bridge", rule=["R-24"])
    assert any("R-24 scope" in e for e in check(expected, manifest, surfaces))


@pytest.mark.parametrize("language", ["c", "cpp"])
def test_review_also_cannot_hide_arbitrary_options_field(language):
    expected, manifest, surfaces = fixture()
    owner = "RLMeshRunOptions" if language == "c" else "rlmesh::RunOptions"
    extra = owner + "::sneaky"
    surfaces[language][extra] = Item("field", default="0")
    surfaces[language][owner].fields.append("sneaky")
    manifest["rlmesh:run"][language]["also"].append(extra)
    if language == "c":
        surfaces[language][
            "RLMESH_RUN_OPTIONS_INIT"
        ].signature = "RLMESH_RUN_OPTIONS_INIT { sizeof(RLMeshRunOptions), 1, 0 }"
    assert any(
        "also_rules" in e and "sneaky" in e for e in check(expected, manifest, surfaces)
    )


def test_review_ext_cannot_hide_unfenced_method():
    expected, manifest, surfaces = fixture()
    extra = "rlmesh::Binding::frobnicate"
    surfaces["rust"][extra] = Item("method")
    manifest["ext"] = {"rust": [{"path": extra, "rule": "R-25"}]}
    assert any("R-25 closed list" in e for e in check(expected, manifest, surfaces))


def reset_fixture(language):
    expected, manifest, surfaces = fixture()
    path = "rlmesh:Binding.reset"
    native = "rlmesh_binding_reset" if language == "c" else "rlmesh::Binding::reset"
    inputs = {"seed": "const int64_t *", "options": "const RLMeshValue *"}
    if language == "c":
        inputs = {
            "binding": "RLMeshBinding *",
            **inputs,
            "out_observation": "RLMeshValue **",
            "out_info": "RLMeshValue **",
        }
    surfaces[language][native] = Item("function", params=inputs)
    expected[path] = {
        "kind": "function",
        "params": {"seed": "None", "options": "None"},
    }
    manifest[path] = {
        "parameters": {n: {"default": "None"} for n in ("seed", "options")}
    }
    for lang in surfaces:
        manifest[path][lang] = (
            {
                "status": "present",
                "path": native,
                "params": {n: native + "#" + n for n in ("seed", "options")},
            }
            if lang == language
            else {"status": "missing", "gap": "G-1"}
        )
    return expected, manifest, surfaces, native


def test_review_positional_reset_order():
    expected, manifest, surfaces, native = reset_fixture("rust")
    surfaces["rust"][native].params = dict(
        reversed(list(surfaces["rust"][native].params.items()))
    )
    assert any("R-7" in e and "order" in e for e in check(expected, manifest, surfaces))


def test_review_c_reset_output_order():
    expected, manifest, surfaces, native = reset_fixture("c")
    params = surfaces["c"][native].params
    surfaces["c"][native].params = {
        n: params[n]
        for n in ("binding", "seed", "options", "out_info", "out_observation")
    }
    assert any(
        "R-15" in e and "order" in e for e in check(expected, manifest, surfaces)
    )


def test_translation_requires_decision_not_naming_rule():
    expected, manifest, surfaces = fixture()
    manifest["rlmesh:run"]["c"]["translations"] = {"episodes": ["R-4"]}
    assert any(
        "translations" in e and "decision" in e
        for e in check(expected, manifest, surfaces)
    )


def test_missing_gap_must_be_in_spec_table():
    expected, manifest, surfaces = fixture()
    manifest["rlmesh:run"]["rust"] = {"status": "missing", "gap": "G-99999"}
    assert any(
        "gap" in e and "G-99999" in e for e in check(expected, manifest, surfaces)
    )


def test_out_prefix_does_not_exempt_scalar_or_const_input():
    from rlmesh_api_surface.parity import structural

    assert not structural("out_capacity", "size_t", {}, "rlmesh:run", "c")
    assert not structural("out_data", "const double *", {}, "rlmesh:run", "c")
    assert structural("out_data", "double *", {}, "rlmesh:run", "c")


def test_protocol_parameters_must_exist():
    expected, manifest, surfaces = fixture()
    mapping = manifest["rlmesh:run"]["cpp"]
    mapping["protocol"] = "run"
    surfaces["cpp"][mapping["path"]].protocols = ["run"]
    mapping["params"]["episodes"] = "rlmesh::run#typo"
    mapping["translations"] = {"episodes": ["D-6"]}
    assert any(
        "does not exist" in e and "typo" in e
        for e in check(expected, manifest, surfaces)
    )


def test_also_missing_target_is_reported_once():
    expected, manifest, surfaces = fixture()
    manifest["rlmesh:run"]["cpp"]["also"].append("rlmesh::absent")
    errors = check(expected, manifest, surfaces)
    assert sum("rlmesh::absent does not exist" in e for e in errors) == 1


def test_recorded_return_type_is_checked():
    expected, manifest, surfaces = fixture()
    manifest["rlmesh:run"]["cpp"]["returns"] = "RunResult"
    surfaces["cpp"]["rlmesh::run"].returns = "void"
    assert any("returns 'void'" in e for e in check(expected, manifest, surfaces))


def test_ext_closed_list_cannot_be_used_through_binding():
    expected, manifest, surfaces = fixture()
    surfaces["rust"]["rlmesh::Binding::frobnicate"] = Item("method")
    manifest["binding"] = {
        "rust": [{"path": "rlmesh::Binding::frobnicate", "rule": "R-25"}]
    }
    assert any("not R-12/R-25" in e for e in check(expected, manifest, surfaces))


@pytest.mark.parametrize("parameter", ["seed", "options"])
def test_arbitrary_input_cannot_be_dropped_as_bridge(parameter):
    expected, manifest, surfaces, native = reset_fixture("rust")
    mapping = manifest["rlmesh:Binding.reset"]["rust"]
    mapping["params"].pop(parameter)
    mapping["bridges"] = [parameter]
    mapping["rule"] = ["R-24"]
    assert any("outside R-24 scope" in e for e in check(expected, manifest, surfaces))


def test_review_rust_tuple_output_order():
    expected, manifest, surfaces, native = reset_fixture("rust")
    manifest["rlmesh:Binding.reset"]["rust"]["result_tuple"] = ["Value", "InfoDict"]
    surfaces["rust"][
        native
    ].signature = "fn reset(seed: Option<i64>, options: Option<&InfoDict>) -> Result<(InfoDict, Value)>"
    assert any(
        "result_tuple order" in e and "R-15" in e
        for e in check(expected, manifest, surfaces)
    )


def test_review_cpp_tuple_alias_order():
    expected, manifest, surfaces, native = reset_fixture("cpp")
    manifest["rlmesh:Binding.reset"]["cpp"].update(
        returns="ResetResult", result_tuple=["Value", "InfoDict"]
    )
    surfaces["cpp"][native].returns = "ResetResult"
    surfaces["cpp"]["rlmesh::ResetResult"] = Item(
        "type alias", "using ResetResult = std::tuple<InfoDict, Value>"
    )
    assert any(
        "result_tuple order" in e and "R-15" in e
        for e in check(expected, manifest, surfaces)
    )


def test_detected_protocol_params_require_a_real_slot_and_call_arity():
    from rlmesh_api_surface.parity import protocol_metadata

    item = Item("constant")
    c_items = {
        "RLMeshModelVtable::predict": Item(
            "field",
            params={
                "user_data": "void *",
                "observation": "const RLMeshValue *",
                "context": "const RLMeshPredictContext *",
                "out_action": "RLMeshValue **",
            },
        )
    }
    source = "decltype(std::declval<P&>().predict(std::declval<const Value&>(), std::declval<const PredictContext&>()));"
    protocol_metadata(source, item, c_items, "RLMeshModelVtable", "P")
    assert list(item.protocol_params["predict"]) == ["observation", "context"]
    protocol_metadata(
        "decltype(std::declval<P&>().predict());",
        item,
        c_items,
        "RLMeshModelVtable",
        "P",
    )
    assert "predict" not in item.protocol_params
    protocol_metadata(source, item, {}, "RLMeshModelVtable", "P")
    assert "predict" not in item.protocols


@pytest.mark.parametrize("language", ["rust", "c", "cpp"])
def test_factory_host_extension_is_c_only(language):
    prefix = "RLMesh" if language == "c" else "rlmesh::"
    assert extension_scope(
        prefix + "EnvFactoryServeOptions::ext", language, Item("field")
    ) == (language == "c")


def test_a_keyword_kept_only_to_refuse_needs_a_decision_not_a_mapping():
    expected, manifest, surfaces = fixture()
    expected["rlmesh:run"]["params"]["session"] = "None"
    manifest["rlmesh:run"]["parameters"]["session"] = {"default": "None"}
    assert any(
        "parameter session has no mapping" in e
        for e in check(expected, manifest, surfaces)
    )
    for lang in ("rust", "c", "cpp"):
        manifest["rlmesh:run"][lang]["refused"] = {"session": ["D-4"]}
    assert check(expected, manifest, surfaces) == []
    for lang in ("rust", "c", "cpp"):
        manifest["rlmesh:run"][lang]["refused"] = {"session": ["D-999"]}
    assert any(
        "parameter session has no mapping" in e
        for e in check(expected, manifest, surfaces)
    )


def test_gymnasium_only_env_members_are_bridges():
    from rlmesh_api_surface.parity import bridge_scope

    assert bridge_scope("rlmesh:RemoteVectorEnv.closed")
    assert bridge_scope("rlmesh:RemoteEnv.spec")
    assert not bridge_scope("rlmesh:RemoteEnv.reset")


def test_a_header_libclang_cannot_read_through_host_headers_parses_through_zig(
    monkeypatch, capsys
):
    from pathlib import Path

    from rlmesh_api_surface import header_surface as hs

    calls = []

    def parse_with(path, config, compiler):
        calls.append(compiler)
        if compiler != hs.FALLBACK_DRIVERS["c++"]:
            raise hs._ParseError("system header error", in_system_headers=True)
        return "parsed"

    monkeypatch.setattr(hs, "_parse_with", parse_with)
    config = hs.HeaderSurfaceConfig(
        header="include/rlmesh.hpp",
        language="c++",
        standard="c++17",
        stability="experimental",
        namespace="rlmesh",
        source_repo="",
        source_commit=None,
    )
    assert hs._parse(Path("include/rlmesh.hpp"), config) == "parsed"
    assert calls[-1] == hs.FALLBACK_DRIVERS["c++"]
    assert "parsed it against mise exec zig@0.13.0" in capsys.readouterr().err

    def own_error(path, config, compiler):
        raise hs._ParseError("error in rlmesh.hpp", in_system_headers=False)

    monkeypatch.setattr(hs, "_parse_with", own_error)
    with pytest.raises(hs._ParseError, match=r"rlmesh\.hpp"):
        hs._parse(Path("include/rlmesh.hpp"), config)

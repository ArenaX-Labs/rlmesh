"""Check real manifest defaults and exercise generator coverage/failure behavior."""

import ast
import importlib
import inspect

import pytest
import tomllib

from .generate_defaults import MANIFEST, generate, options_defaults


def test_manifest_options_defaults_match_python():
    """Mapped options defaults record live Python defaults, including constants."""
    manifest = tomllib.loads(MANIFEST.read_text())
    for symbol, entry in manifest.items():
        if "parameters" not in entry:
            continue
        mappings = entry.get("rust", {}).get("params", {})
        keywords = {
            keyword: parameter
            for keyword, parameter in entry["parameters"].items()
            if (
                mappings.get(keyword, "").split("#")[0].endswith("Options")
                or any(
                    item.endswith(f"Options::{keyword}")
                    for item in entry.get("rust", {}).get("also", [])
                )
            )
            and "default" in parameter
        }
        if not keywords:
            continue
        module, path = symbol.split(":")
        function = importlib.import_module(module)
        for name in path.split("."):
            function = getattr(function, name)
        parameters = inspect.signature(function).parameters
        for keyword, parameter in keywords.items():
            assert parameters[keyword].default == ast.literal_eval(
                parameter["default"]
            ), (
                symbol,
                keyword,
            )


def example(default="None"):
    """Minimal E-1 input, kept in the unit test rather than a bootstrap manifest."""
    return {
        "rlmesh:run": {
            "rust": {"params": {"hooks": "rlmesh::RunOptions#hooks"}},
            "parameters": {"hooks": {"default": default}},
        }
    }


def test_generator_covers_every_manifest_options_default():
    manifest = tomllib.loads(MANIFEST.read_text())
    fields = options_defaults(manifest)
    source = generate(manifest)
    for native_type, defaults in fields.items():
        constructors = [
            entry["rust"]["default_constructor"]
            for symbol, entry in manifest.items()
            if ":" in symbol
            and entry.get("rust", {})
            .get("default_constructor", "")
            .startswith(native_type + "::")
        ]
        constructor = next(iter(constructors), native_type + "::default()")
        assert source.count(f"let value = {constructor};") == 1
        for field in defaults:
            assert f"value.{field}" in source


def test_generator_translates_field_names_and_whole_options_arguments():
    manifest = {
        "rlmesh:run": {
            "parameters": {
                "max_episode_seconds": {"default": "None"},
                "model": {},
            },
            "rust": {
                "params": {
                    "max_episode_seconds": "rlmesh::RunOptions#max_episode",
                    "model": "rlmesh::run#model",
                },
            },
        },
        "rlmesh:Model.serve": {
            "parameters": {"options": {"default": "None"}},
            "rust": {
                "params": {"options": "rlmesh::Model::serve#options"},
                "also": ["rlmesh::ModelServeOptions::options"],
            },
        },
    }
    source = generate(manifest)
    assert "value.max_episode.is_none()" in source
    assert "value.max_episode_seconds" not in source
    assert "value.options.is_none()" in source
    assert "value.model" not in source


def test_generator_deduplicates_and_rejects_conflicts():
    manifest = example()
    manifest["rlmesh:Model.run"] = example()["rlmesh:run"]
    assert generate(manifest).count("value.hooks.is_none()") == 1
    manifest["rlmesh:Model.run"] = example("False")["rlmesh:run"]
    with pytest.raises(ValueError, match="conflicting defaults"):
        generate(manifest)


def test_generator_rejects_unsupported_defaults():
    """A new unsupported default cannot silently disappear from the Rust test."""
    with pytest.raises((ValueError, SyntaxError)):
        generate(example("object()"))
    with pytest.raises(ValueError, match="no Rust options defaults"):
        generate({})


def test_review_view_default_is_checked_outside_options():
    manifest = {
        "rlmesh:View": {
            "parameters": {"port": {"default": "8008"}},
            "rust": {"params": {"port": "rlmesh::View#port"}},
        }
    }
    source = generate(manifest)
    assert "rlmesh::View::default()" in source
    assert 'assert_eq!(value.port, 8008, "rlmesh:View.port")' in source


def test_unchecked_callable_default_requires_explicit_reason():
    manifest = {
        "rlmesh:Session.reset": {
            "parameters": {"seed": {"default": "None"}},
            "rust": {"params": {"seed": "rlmesh::Session::reset#seed"}},
        }
    }
    with pytest.raises(ValueError, match="default_skip"):
        generate(manifest)
    manifest["rlmesh:Session.reset"]["rust"]["default_skip"] = {"seed": ""}
    with pytest.raises(ValueError, match="requires a reason"):
        generate(manifest)

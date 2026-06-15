"""``rlmesh.export``: build a recipe to an image, for both env and model kinds.

The recipe classes are module-level (not local) so ``to_recipe()``'s
import-safety guard is satisfied, exactly like ``test_model_recipe.py``. The
native build layer is stubbed so these run without a Docker daemon -- the
crate-level Rust tests cover the real build/tag.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
import rlmesh
from rlmesh.models import ModelRecipe
from rlmesh.recipes import Build, PipInstall


class _ExportEnv(rlmesh.EnvRecipe):
    name = "export/env"
    build = Build(pip=[PipInstall(["gymnasium"])])

    def make(self, **kwargs: object) -> object:
        raise NotImplementedError


class _ExportPolicy(ModelRecipe):
    name = "export/policy"
    build = Build(pip=[PipInstall(["numpy"])])
    spec = None

    def load(self) -> None: ...

    def predict(self, observation: Any) -> Any:
        return observation


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    def fake_build(display: str, **kwargs: Any) -> dict[str, Any]:
        seen.clear()
        seen.update(kwargs)
        seen["display"] = display
        kind = json.loads(kwargs["recipe_json"])["kind"]
        return {
            "requested_source": display,
            "resolved_source": display,
            "image": f"rlmesh-sandbox-recipe:{kind}123456789",
            "alias": kwargs.get("tag"),
            "image_id": "sha256:abc",
        }

    monkeypatch.setattr("rlmesh.sandbox._export._sandbox_build_image", fake_build)
    return seen


def test_export_env_bakes_env_kind(captured: dict[str, Any]) -> None:
    result = rlmesh.export(_ExportEnv, tag="me/env:v1")

    assert json.loads(captured["recipe_json"])["kind"] == "env"
    assert captured["recipe_provenance"] == "installed"
    assert captured["tag"] == "me/env:v1"
    assert result.alias == "me/env:v1"
    assert result.image.startswith("rlmesh-sandbox-recipe:env")
    assert result.image_id == "sha256:abc"


def test_export_model_bakes_model_kind(captured: dict[str, Any]) -> None:
    result = rlmesh.export(_ExportPolicy, tag="me/policy:v1")

    assert json.loads(captured["recipe_json"])["kind"] == "model"
    assert captured["recipe_provenance"] == "installed"
    assert result.alias == "me/policy:v1"
    assert result.image.startswith("rlmesh-sandbox-recipe:model")


def test_export_without_tag_has_no_alias(captured: dict[str, Any]) -> None:
    result = rlmesh.export(_ExportPolicy)

    assert captured["tag"] is None
    assert result.alias is None


def test_export_forwards_build_options(captured: dict[str, Any]) -> None:
    rlmesh.export(
        _ExportEnv,
        base_image="python:3.11-slim",
        packages=["extra-pkg"],
        build_memory="4g",
    )

    assert captured["base_image"] == "python:3.11-slim"
    assert captured["packages"] == ["extra-pkg"]
    assert captured["build_memory"] == "4g"


def test_export_resolves_registered_model_name(
    captured: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    rlmesh.register(_ExportPolicy)
    try:
        rlmesh.export("export/policy")
    finally:
        from rlmesh.recipes import _registry

        monkeypatch.undo()
        _registry.unregister("export/policy")

    assert json.loads(captured["recipe_json"])["kind"] == "model"

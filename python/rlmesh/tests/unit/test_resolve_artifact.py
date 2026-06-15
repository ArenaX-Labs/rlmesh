"""The materialize() seam: resolve_artifact precedence + the run-contract override.

Covers the resolution ladder (env override > in-container mount/fetch > host
local_dir/uri) and the env-var helpers, monkeypatching `_snapshot_hf` so no network.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

import pytest
import rlmesh
from rlmesh.models import ArtifactInput
from rlmesh.recipes import _artifacts
from rlmesh.recipes._artifacts import (
    _env_key,
    _materialized_path_from_env,
    resolve_artifact,
)
from rlmesh.server import EnvLike

_OVERRIDE_ENVS = (
    "RLMESH_INPUT_WEIGHTS_PATH",
    "RLMESH_MODEL_INPUT_WEIGHTS_PATH",
    "RLMESH_INPUT_CHECKPOINT_PATH",
    "RLMESH_MODEL_INPUT_CHECKPOINT_PATH",
    "RLMESH_MODEL_CHECKPOINT_PATH",
)


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _OVERRIDE_ENVS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def fake_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        _artifacts, "_snapshot_hf", lambda repo, rev, **_k: f"/cache/{repo}@{rev}"
    )


def test_env_override_wins_everywhere(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RLMESH_INPUT_WEIGHTS_PATH", "/override")
    art = ArtifactInput("weights", "/target", uri="hf://org/repo")
    assert resolve_artifact(art, in_container=True) == "/override"
    assert resolve_artifact(art, in_container=False) == "/override"


def test_back_compat_alias_and_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    art = ArtifactInput("weights", "/target", uri="hf://org/repo")
    monkeypatch.setenv("RLMESH_MODEL_INPUT_WEIGHTS_PATH", "/alias")
    assert resolve_artifact(art, in_container=True) == "/alias"
    # canonical wins over the managed alias when both are set
    monkeypatch.setenv("RLMESH_INPUT_WEIGHTS_PATH", "/canonical")
    assert resolve_artifact(art, in_container=True) == "/canonical"


def test_checkpoint_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RLMESH_MODEL_CHECKPOINT_PATH", "/ckpt")
    art = ArtifactInput("checkpoint", "/target", uri="hf://org/repo")
    assert resolve_artifact(art, in_container=True) == "/ckpt"


def test_in_container_local_dir_is_target_path() -> None:
    art = ArtifactInput("w", "/target", local_dir="/host")
    assert resolve_artifact(art, in_container=True) == "/target"


def test_in_container_uri_only_fetches(fake_snapshot: None) -> None:
    art = ArtifactInput("w", "/target", uri="hf://org/repo")
    assert resolve_artifact(art, in_container=True) == "/cache/org/repo@None"


def test_in_container_no_source_is_target_path() -> None:
    art = ArtifactInput("w", "/target")
    assert resolve_artifact(art, in_container=True) == "/target"


def test_host_local_dir_wins() -> None:
    art = ArtifactInput("w", "/target", local_dir="/host")
    assert resolve_artifact(art, in_container=False) == "/host"


def test_host_uri_fetches(fake_snapshot: None) -> None:
    art = ArtifactInput("w", "/target", uri="hf://org/repo@abc")
    assert resolve_artifact(art, in_container=False) == "/cache/org/repo@abc"


def test_host_required_unresolved_raises() -> None:
    art = ArtifactInput("w", "/target")  # required default True, no uri/local_dir
    with pytest.raises(FileNotFoundError, match="required"):
        resolve_artifact(art, in_container=False)


def test_host_optional_unresolved_is_none() -> None:
    art = ArtifactInput("w", "/target", required=False)
    assert resolve_artifact(art, in_container=False) is None


def test_env_key_sanitization() -> None:
    assert _env_key("weights") == "WEIGHTS"
    assert _env_key("my-weights.v2") == "MY_WEIGHTS_V2"
    assert _env_key("a/b c") == "A_B_C"


def test_materialized_path_from_env_absent() -> None:
    assert _materialized_path_from_env("weights") is None


def test_snapshot_hf_missing_dep_message(monkeypatch: pytest.MonkeyPatch) -> None:
    # None in sys.modules makes `from huggingface_hub import ...` raise ImportError.
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(ImportError, match=r"rlmesh\[hf\]"):
        _artifacts._snapshot_hf("org/repo", None)


# ── env symmetry: an authored EnvRecipe resolves inputs through the same seam ──


class _Env:
    def reset(self, *, seed: object = None, options: object = None) -> object:
        return 0, {}

    def step(self, action: object) -> object:
        return 0, 0.0, False, False, {}


def _assets_env(local_dir: str, captured: dict[str, str]) -> type[rlmesh.EnvRecipe]:
    class AssetsEnv(rlmesh.EnvRecipe):
        name = "test/assets-env"
        inputs = (ArtifactInput("assets", "/assets", local_dir=local_dir),)

        def make(self, **kwargs: object) -> EnvLike:
            captured["path"] = self.input_path("assets")
            return cast(EnvLike, cast(object, _Env()))

    return AssetsEnv


def test_env_recipe_host_input_path_resolves_local_dir(tmp_path: Path) -> None:
    from rlmesh.recipes.authoring.env import construct_authored

    captured: dict[str, str] = {}
    _ = construct_authored(_assets_env(str(tmp_path), captured))
    assert captured["path"] == str(tmp_path)


def test_env_recipe_in_container_uses_target_path() -> None:
    captured: dict[str, str] = {}
    _ = _assets_env("/unused", captured)._rlmesh_construct()
    assert captured["path"] == "/assets"


def test_env_sandbox_inputs_flow_to_mounts(tmp_path: Path) -> None:
    # 1.5b: an env recipe's local_dir inputs reach the sandbox as bind-mounts.
    from rlmesh.recipes import PyMake, Recipe
    from rlmesh.recipes._artifacts import local_dir_mounts
    from rlmesh.sandbox._export import resolve_recipe_source

    rec = Recipe(
        name="t/assets-sandbox",
        kind="env",
        make=PyMake(entrypoint="m:C._rlmesh_construct"),
        inputs=(ArtifactInput("assets", "/assets", local_dir=str(tmp_path)),),
    )
    _, _, _, _, inputs = resolve_recipe_source(rec)
    assert inputs == rec.inputs
    assert local_dir_mounts(inputs) == [(str(tmp_path), "/assets")]


def test_resolve_uri_handles_file_netloc_forms() -> None:
    from rlmesh.recipes._artifacts import _resolve_uri

    assert _resolve_uri("file:///abs/path") == "/abs/path"
    assert _resolve_uri("file://localhost/abs/path") == "/abs/path"
    # A '#'/'?' is a legal path char on the host and must survive (urlparse would
    # truncate it into a fragment/query).
    assert _resolve_uri("file:///data/run#2/ckpt") == "/data/run#2/ckpt"
    with pytest.raises(NotImplementedError):
        _resolve_uri("file://remotehost/path")


def test_merged_inputs_override_keeps_declared_target_path() -> None:
    from rlmesh.recipes._artifacts import merged_inputs

    declared = ArtifactInput("weights", "/declared/target", uri="hf://org/repo")
    override = ArtifactInput("weights", "/ignored/target", local_dir="/host/ckpt")
    merged = merged_inputs((declared,), (override,))
    result = merged["weights"]
    # The override repoints the source; the container target stays the declared one
    # (matching local_dir_mounts, so local and sandbox resolution agree).
    assert result.target_path == "/declared/target"
    assert result.local_dir == "/host/ckpt"
    assert result.uri is None

    # An override naming an undeclared input is rejected, matching local_dir_mounts.
    extra = ArtifactInput("extra", "/extra", local_dir="/host/extra")
    with pytest.raises(ValueError, match="matches no declared input"):
        merged_inputs((declared,), (extra,))

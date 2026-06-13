from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, ClassVar, cast

import pytest
from rlmesh.recipes import Build, EnvRecipe, ProjectInstall


def test_sandbox_cleanup_runs_on_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rlmesh import sandbox

    stopped: list[str] = []
    captured: dict[str, object] = {}

    class InterruptingRemote:
        @classmethod
        def _connect_for_sandbox(
            cls,
            address: str,
            *,
            connect_timeout_seconds: float,
        ) -> object:
            captured["address"] = address
            captured["connect_timeout_seconds"] = connect_timeout_seconds
            raise KeyboardInterrupt

    class SandboxUnderTest(sandbox.SandboxSessionBase[object]):
        _remote_env_cls: ClassVar[type[InterruptingRemote]] = InterruptingRemote

    monkeypatch.setattr(sandbox, "_sandbox_start_env", _start_result)
    monkeypatch.setattr(
        sandbox,
        "_sandbox_stop_env",
        lambda *, container_id: stopped.append(container_id),
    )

    with pytest.raises(KeyboardInterrupt):
        SandboxUnderTest("CartPole-v1")

    assert stopped == ["container-1"]
    assert captured == {
        "address": "tcp://127.0.0.1:50051",
        "connect_timeout_seconds": sandbox.SANDBOX_REMOTE_CONNECT_TIMEOUT_SECONDS,
    }


def test_sandbox_cleanup_runs_on_remote_attach_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rlmesh import sandbox

    stopped: list[str] = []

    class FailingRemote:
        def __init__(self, address: str) -> None:
            _ = address
            raise RuntimeError("attach failed")

    class SandboxUnderTest(sandbox.SandboxSessionBase[object]):
        _remote_env_cls: ClassVar[type[FailingRemote]] = FailingRemote

    monkeypatch.setattr(sandbox, "_sandbox_start_env", _start_result)
    monkeypatch.setattr(
        sandbox,
        "_sandbox_stop_env",
        lambda *, container_id: stopped.append(container_id),
    )

    with pytest.raises(RuntimeError, match="attach failed"):
        SandboxUnderTest("CartPole-v1")

    assert stopped == ["container-1"]


def test_sandbox_package_spec_alias_sets_rlmesh_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rlmesh import sandbox

    captured: dict[str, object] = {}
    stopped: list[str] = []

    class Remote:
        def __init__(self, address: str) -> None:
            self.address = address

        def close(self) -> None:
            pass

    class SandboxUnderTest(sandbox.SandboxSessionBase[object]):
        _remote_env_cls: ClassVar[type[Remote]] = Remote

    def start_result(*_args: object, **kwargs: object) -> dict[str, str]:
        captured.update(kwargs)
        return _start_result()

    monkeypatch.setattr(sandbox, "_sandbox_start_env", start_result)
    monkeypatch.setattr(
        sandbox,
        "_sandbox_stop_env",
        lambda *, container_id: stopped.append(container_id),
    )

    with SandboxUnderTest(
        "CartPole-v1",
        package_spec="local",
        render_mode="rgb_array",
    ):
        pass

    assert captured["rlmesh_package"] == "local"
    assert json.loads(cast(str, captured["kwargs_json"])) == {
        "render_mode": "rgb_array",
    }
    assert stopped == ["container-1"]


def test_sandbox_package_spec_alias_rejects_ambiguous_rlmesh_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rlmesh import sandbox

    class Remote:
        def __init__(self, address: str) -> None:
            self.address = address

    class SandboxUnderTest(sandbox.SandboxSessionBase[object]):
        _remote_env_cls: ClassVar[type[Remote]] = Remote

    monkeypatch.setattr(
        sandbox,
        "_sandbox_start_env",
        lambda *_args, **_kwargs: pytest.fail("sandbox should not start"),
    )

    with pytest.raises(TypeError, match=r"both rlmesh_package=.*package_spec"):
        SandboxUnderTest(
            "CartPole-v1",
            rlmesh_package="local",
            package_spec="wheel.whl",
        )


def test_sandbox_retries_close_after_transient_stop_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rlmesh import sandbox

    stop_calls: list[str] = []

    class Remote:
        def __init__(self, address: str) -> None:
            self.address = address

        def close(self) -> None:
            pass

    class SandboxUnderTest(sandbox.SandboxSessionBase[object]):
        _remote_env_cls: ClassVar[type[Remote]] = Remote

    def flaky_stop(*, container_id: str) -> None:
        stop_calls.append(container_id)
        if len(stop_calls) == 1:
            raise RuntimeError("docker daemon unavailable")

    monkeypatch.setattr(sandbox, "_sandbox_start_env", _start_result)
    monkeypatch.setattr(sandbox, "_sandbox_stop_env", flaky_stop)

    session = SandboxUnderTest("CartPole-v1")

    # First close attempt fails while stopping the container.
    with pytest.raises(RuntimeError, match="docker daemon unavailable"):
        session.close()
    # Session must not be marked closed, so the container is not leaked.
    assert session._closed is False

    # A retry succeeds and stops the container.
    session.close()
    assert session._closed is True
    assert stop_calls == ["container-1", "container-1"]


@pytest.mark.parametrize("field", ["packages", "imports"])
def test_sandbox_rejects_bare_str_packages_imports(
    field: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rlmesh import sandbox

    class Remote:
        def __init__(self, address: str) -> None:
            self.address = address

    class SandboxUnderTest(sandbox.SandboxSessionBase[object]):
        _remote_env_cls: ClassVar[type[Remote]] = Remote

    monkeypatch.setattr(
        sandbox,
        "_sandbox_start_env",
        lambda *_args, **_kwargs: pytest.fail("sandbox should not start"),
    )

    with pytest.raises(TypeError, match=rf"{field}= expects a sequence of strings"):
        kwargs: dict[str, Any] = {field: "ale-py"}
        SandboxUnderTest("CartPole-v1", **kwargs)


@pytest.mark.parametrize("field", ["packages", "imports"])
def test_sandbox_accepts_string_sequence_packages_imports(
    field: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rlmesh import sandbox

    captured: dict[str, object] = {}
    stopped: list[str] = []

    class Remote:
        def __init__(self, address: str) -> None:
            self.address = address

        def close(self) -> None:
            pass

    class SandboxUnderTest(sandbox.SandboxSessionBase[object]):
        _remote_env_cls: ClassVar[type[Remote]] = Remote

    def start_result(*_args: object, **kwargs: object) -> dict[str, str]:
        captured.update(kwargs)
        return _start_result()

    monkeypatch.setattr(sandbox, "_sandbox_start_env", start_result)
    monkeypatch.setattr(
        sandbox,
        "_sandbox_stop_env",
        lambda *, container_id: stopped.append(container_id),
    )

    kwargs: dict[str, Any] = {field: ["ale-py"]}
    with SandboxUnderTest("CartPole-v1", **kwargs):
        pass

    assert captured[field] == ["ale-py"]


def test_resolve_recipe_source_bakes_make_kwargs_into_document() -> None:
    # A recipe source carries make kwargs in the document (make.kwargs), since the
    # recipe bootstrap payload never threads kwargs_json into the recipe build.
    from rlmesh import recipes
    from rlmesh.recipes import GymMake, Recipe
    from rlmesh.sandbox import _resolve_recipe_source

    recipe = Recipe(name="cart/pole", make=GymMake(env_id="CartPole-v1"))
    _, recipe_json, provenance, _ = _resolve_recipe_source(
        recipe, {"render_mode": "rgb_array"}
    )

    assert provenance == "installed"
    assert recipe_json is not None
    document = recipes.Recipe.from_json(recipe_json)
    assert document.make is not None
    assert dict(document.make.kwargs) == {"render_mode": "rgb_array"}


def test_resolve_recipe_source_merges_over_existing_make_kwargs() -> None:
    # Caller kwargs win over the recipe's own baked make kwargs, like
    # rlmesh.make(recipe, **kwargs).
    from rlmesh.recipes import GymMake, Recipe
    from rlmesh.sandbox import _resolve_recipe_source

    recipe = Recipe(
        name="cart/pole",
        make=GymMake(env_id="CartPole-v1", kwargs={"render_mode": "human", "g": 9.8}),
    )
    _, recipe_json, _, _ = _resolve_recipe_source(recipe, {"render_mode": "rgb_array"})

    assert recipe_json is not None
    document = Recipe.from_json(recipe_json)
    assert document.make is not None
    assert dict(document.make.kwargs) == {"render_mode": "rgb_array", "g": 9.8}


def test_resolve_recipe_source_build_only_base_with_kwargs_raises() -> None:
    # A build-only base recipe (make=None) has nowhere to bake make kwargs.
    from rlmesh.recipes import Build, Recipe
    from rlmesh.sandbox import _resolve_recipe_source

    base = Recipe(name="acme/base", build=Build(base="python:3.11-slim"))
    with pytest.raises(TypeError, match="build-only base"):
        _resolve_recipe_source(base, {"render_mode": "rgb_array"})


def test_resolve_recipe_source_rejects_hf_make_before_build() -> None:
    # An HfMake recipe must be rejected up front, not only after a full image build
    # by the in-container build().
    from rlmesh.recipes import HfMake, Recipe, UnsupportedRecipeError
    from rlmesh.sandbox import _resolve_recipe_source

    recipe = Recipe(
        name="acme/hf",
        make=HfMake(repo="acme/env", revision="a" * 40),
    )
    with pytest.raises(UnsupportedRecipeError, match="HfMake"):
        _resolve_recipe_source(recipe, {})


def test_resolve_recipe_source_rejects_setup_files_before_build() -> None:
    # setup.files is not applied anywhere yet (the in-container build() -> apply_setup
    # raises on it too), so it must be rejected up front -- before any image build.
    from rlmesh.recipes import GymMake, Recipe, Setup, UnsupportedRecipeError
    from rlmesh.recipes._schema import FileWrite
    from rlmesh.sandbox import _resolve_recipe_source

    recipe = Recipe(
        name="acme/with-files",
        make=GymMake(env_id="CartPole-v1"),
        setup=Setup(files=[FileWrite(path="x.txt", contents="hi")]),
    )
    with pytest.raises(
        UnsupportedRecipeError, match=r"setup\.files is not applied yet"
    ):
        _resolve_recipe_source(recipe, {})


def test_resolve_recipe_source_from_recipe_uses_base_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # When a task recipe inlines a `from_recipe` base, the base supplies the entire
    # build (incl. its ProjectInstall, whose src is relative to the BASE's tree). The
    # staged context_root must therefore be the BASE's origin (dir A), not the task's
    # registration dir / cwd (dir B).
    from rlmesh import recipes
    from rlmesh.recipes import Build, GymMake, ProjectInstall, PyMake, Recipe
    from rlmesh.sandbox import _resolve_recipe_source

    base = Recipe(
        name="acme/base",
        make=PyMake(entrypoint="acme_env:make"),
        build=Build(project=ProjectInstall(src=".")),
    )
    task = Recipe(
        name="acme/task",
        make=GymMake(env_id="CartPole-v1"),
        build=Build(from_recipe="acme/base"),
    )
    recipes.register(base)
    recipes.register(task)
    # Force the two recipes to have *different* recorded origins so we can prove the
    # base's origin (dir A) wins over the task's (dir B / cwd).
    origins = {"acme/base": "/dir/A", "acme/task": "/dir/B"}
    monkeypatch.setattr(
        "rlmesh.recipes._registry.recipe_origin_dir",
        lambda name: origins.get(name),
    )
    try:
        _, _, provenance, context_root = _resolve_recipe_source("acme/task", {})
    finally:
        recipes.unregister("acme/base")
        recipes.unregister("acme/task")

    assert provenance == "installed"
    assert context_root == "/dir/A"


def test_resolve_recipe_source_chained_from_recipe_uses_terminal_base_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A chained task -> base_b -> base_a inlines the TERMINAL base's build (base_a,
    # the first whose build.from_recipe is None). base_a owns the ProjectInstall, so
    # the staged context_root must be base_a's origin (dir A) -- not base_b's (dir B)
    # or the task's (dir C / cwd). The immediate base (base_b) carries no build of
    # its own here.
    from rlmesh import recipes
    from rlmesh.recipes import Build, GymMake, ProjectInstall, PyMake, Recipe
    from rlmesh.sandbox import _resolve_recipe_source

    base_a = Recipe(
        name="acme/base-a",
        make=PyMake(entrypoint="acme_env:make"),
        build=Build(project=ProjectInstall(src=".")),
    )
    base_b = Recipe(name="acme/base-b", build=Build(from_recipe="acme/base-a"))
    task = Recipe(
        name="acme/task",
        make=GymMake(env_id="CartPole-v1"),
        build=Build(from_recipe="acme/base-b"),
    )
    recipes.register(base_a)
    recipes.register(base_b)
    recipes.register(task)
    # Distinct recorded origins so we can prove the terminal base's origin (dir A)
    # wins over the intermediate base (dir B) and the task (dir C / cwd).
    origins = {"acme/base-a": "/dir/A", "acme/base-b": "/dir/B", "acme/task": "/dir/C"}
    monkeypatch.setattr(
        "rlmesh.recipes._registry.recipe_origin_dir",
        lambda name: origins.get(name),
    )
    try:
        _, _, provenance, context_root = _resolve_recipe_source("acme/task", {})
    finally:
        recipes.unregister("acme/base-a")
        recipes.unregister("acme/base-b")
        recipes.unregister("acme/task")

    assert provenance == "installed"
    assert context_root == "/dir/A"


def test_resolve_recipe_source_authored_project_uses_module_dir() -> None:
    # An authored EnvRecipe with a ProjectInstall stages from its defining module's
    # directory, not the launching process's cwd.
    from pathlib import Path

    from rlmesh.sandbox import _resolve_recipe_source

    _, recipe_json, provenance, context_root = _resolve_recipe_source(
        _ProjectRecipe, {}
    )

    assert provenance == "installed"
    assert recipe_json is not None
    expected = str(Path(__file__).resolve().parent)
    assert context_root == expected


def test_resolve_recipe_source_registered_name_uses_registrant_dir() -> None:
    # A recipe resolved by registered name stages a ProjectInstall from the
    # registrant's module directory (recorded at register() time), not the cwd.
    from pathlib import Path

    from rlmesh import recipes
    from rlmesh.recipes import Build, ProjectInstall, PyMake, Recipe
    from rlmesh.sandbox import _resolve_recipe_source

    recipe = Recipe(
        name="acme/by-name",
        make=PyMake(entrypoint="acme_env:make"),
        build=Build(project=ProjectInstall(src=".")),
    )
    recipes.register(recipe)
    try:
        _, _, provenance, context_root = _resolve_recipe_source("acme/by-name", {})
    finally:
        recipes.unregister("acme/by-name")

    assert provenance == "installed"
    # register() above runs from this test module, so its origin is this directory.
    assert context_root == str(Path(__file__).resolve().parent)


def test_resolve_recipe_source_plain_id_unchanged() -> None:
    # A plain gym id that is not a registered recipe stays an ordinary source.
    from rlmesh.sandbox import _resolve_recipe_source

    assert _resolve_recipe_source("CartPole-v1", {"render_mode": "rgb_array"}) == (
        "CartPole-v1",
        None,
        None,
        None,
    )


def test_start_sandbox_recipe_path_omits_kwargs_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # On the recipe path the make kwargs ride the document (make.kwargs); they must
    # not ALSO be shipped via kwargs_json, which the recipe bootstrap ignores.
    from rlmesh import sandbox
    from rlmesh.recipes import GymMake, Recipe

    captured: dict[str, object] = {}

    def start_result(*_args: object, **kwargs: object) -> dict[str, str]:
        captured.update(kwargs)
        return _start_result()

    monkeypatch.setattr(sandbox, "_sandbox_start_env", start_result)

    recipe = Recipe(name="cart/pole", make=GymMake(env_id="CartPole-v1"))
    sandbox._start_sandbox(
        recipe,
        base_image=None,
        rlmesh_package=None,
        packages=None,
        imports=None,
        trust_remote_code=False,
        allow_unpinned_hf=False,
        num_envs=1,
        vectorization_mode=None,
        gym_make_kwargs={"render_mode": "rgb_array"},
    )

    assert captured["kwargs_json"] is None
    baked = Recipe.from_json(cast(str, captured["recipe_json"]))
    assert baked.make is not None
    assert dict(baked.make.kwargs) == {"render_mode": "rgb_array"}


class _ProjectRecipe(EnvRecipe):
    """A test EnvRecipe whose build stages its own package source tree."""

    name = "acme/project"
    build = Build(project=ProjectInstall(src="."))

    def make(self, **kwargs: object) -> object:
        return SimpleNamespace(reset=lambda: None, step=lambda action: None)


def _start_result(*_args: object, **_kwargs: object) -> dict[str, str]:
    return {
        "requested_source": "gym://CartPole-v1",
        "resolved_source": "gym://CartPole-v1",
        "address": "tcp://127.0.0.1:50051",
        "container_id": "container-1",
    }

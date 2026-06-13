from __future__ import annotations

import pytest
from rlmesh.recipes import (
    Build,
    Fetch,
    FileWrite,
    GymMake,
    HfMake,
    PipInstall,
    ProjectInstall,
    PyMake,
    Recipe,
    RecipeValidationError,
    Requires,
    Setup,
)


def _rich_recipe() -> Recipe:
    return Recipe(
        name="acme/libero-10",
        make=PyMake(
            entrypoint="robot_env.factory:make", kwargs={"task": "pick", "n": 3}
        ),
        build=Build(
            base="nvidia/cuda:12.4.1-runtime-ubuntu22.04",
            system=["cmake", "g++"],
            system_runtime=["libglew2.2"],
            pip=[
                PipInstall(
                    packages=["torch", "torchvision"],
                    index_url="https://download.pytorch.org/whl/cu124",
                ),
                PipInstall(packages=["libero==1.0.0"], no_deps=True),
            ],
            project=ProjectInstall(
                src=".", dest="/opt/robot_env", include=["assets/**"]
            ),
            fetch=[
                Fetch(
                    kind="git",
                    repo="https://github.com/x/LIBERO.git",
                    ref="a" * 40,
                    dest="/opt/LIBERO",
                    pip_install=True,
                )
            ],
            env={"MUJOCO_GL": "egl"},
            pythonpath=["/opt/LIBERO"],
            gpu=True,
            run_as=1000,
        ),
        setup=Setup(
            env={"LIBERO_TASK": "libero_10/0"},
            files=[FileWrite(path="cfg.json", contents="{}", if_absent=True)],
        ),
        summary="LIBERO task 10",
    )


def test_gym_one_liner_round_trip() -> None:
    recipe = Recipe(
        name="safety/point-goal",
        make=GymMake(env_id="SafetyPointGoal1-v0"),
        build=Build(pip=[PipInstall(packages=["safety-gymnasium==1.0.0"])]),
        requires=Requires(imports=["safety_gymnasium"]),
        summary="Safety Gymnasium point goal",
    )
    assert Recipe.from_json(recipe.to_json()) == recipe


def test_rich_recipe_round_trip() -> None:
    recipe = _rich_recipe()
    assert Recipe.from_json(recipe.to_json()) == recipe


def test_make_kind_tags() -> None:
    assert Recipe(name="a", make=GymMake("E-v0")).to_dict()["make"] == {
        "kind": "gym",
        "env_id": "E-v0",
        "kwargs": {},
    }
    py = Recipe(name="a", make=PyMake("m:f")).to_dict()["make"]
    assert isinstance(py, dict) and py["kind"] == "py"
    hf = Recipe(name="a", make=HfMake(repo="org/env", suite="s", task="t")).to_dict()[
        "make"
    ]
    assert isinstance(hf, dict) and hf["kind"] == "hf"


def test_build_only_base_has_null_make() -> None:
    recipe = Recipe(
        name="base/cuda", build=Build(base="nvidia/cuda:12.4.1-runtime-ubuntu22.04")
    )
    assert recipe.make is None
    assert recipe.to_dict()["make"] is None
    assert Recipe.from_json(recipe.to_json()) == recipe


def test_from_dict_is_lenient_about_missing_keys() -> None:
    recipe = Recipe.from_dict(
        {"name": "minimal", "make": {"kind": "gym", "env_id": "E-v0"}}
    )
    assert recipe.name == "minimal"
    assert recipe.build == Build()
    assert recipe.setup == Setup()
    assert recipe.requires.imports == ()
    assert recipe.recipe_version == 1


def test_kwargs_must_be_json_only() -> None:
    with pytest.raises(RecipeValidationError, match="JSON-only"):
        GymMake(env_id="E-v0", kwargs={"f": lambda: 1})


def test_kwargs_reject_numpy_scalar() -> None:
    np = pytest.importorskip("numpy")
    with pytest.raises(RecipeValidationError, match="JSON-only"):
        GymMake(env_id="E-v0", kwargs={"x": np.float64(1.0)})


def test_kwargs_reject_non_str_keys() -> None:
    with pytest.raises(RecipeValidationError, match="keys must be str"):
        GymMake(env_id="E-v0", kwargs={1: "v"})


def test_kwargs_nested_json_round_trips() -> None:
    recipe = Recipe(
        name="a", make=GymMake("E-v0", kwargs={"nested": {"xs": [1, 2, 3]}})
    )
    assert Recipe.from_json(recipe.to_json()) == recipe


def test_name_rejects_at_sign() -> None:
    with pytest.raises(RecipeValidationError, match="@variant"):
        Recipe(name="acme/env@v2")


def test_pymake_forbids_requires_imports() -> None:
    with pytest.raises(RecipeValidationError, match="forbidden for PyMake"):
        Recipe(name="a", make=PyMake("m:f"), requires=Requires(imports=["robot_env"]))


def test_gym_make_allows_requires_imports() -> None:
    recipe = Recipe(name="a", make=GymMake("E-v0"), requires=Requires(imports=["pkg"]))
    assert recipe.requires.imports == ("pkg",)


def test_pymake_entrypoint_needs_colon() -> None:
    with pytest.raises(RecipeValidationError, match="module:callable"):
        PyMake("not_a_factory")


def test_build_base_and_from_recipe_mutually_exclusive() -> None:
    with pytest.raises(RecipeValidationError, match="mutually exclusive"):
        Build(base="img", from_recipe="other/recipe")


def test_build_dockerfile_excludes_structured_fields() -> None:
    with pytest.raises(RecipeValidationError, match="mutually exclusive"):
        Build(dockerfile="FROM x\n", pip=[PipInstall(packages=["a"])])
    # dockerfile alone is fine
    assert Build(dockerfile="FROM x\n").dockerfile == "FROM x\n"


def test_apt_name_validation() -> None:
    with pytest.raises(RecipeValidationError):
        Build(system=["bad name; rm -rf /"])
    assert Build(system=["lib-foo+bar.baz"]).system == ("lib-foo+bar.baz",)


def test_pip_package_rejects_option_injection() -> None:
    with pytest.raises(RecipeValidationError, match="not empty or an option"):
        PipInstall(packages=["--index-url=https://evil"])


def test_pip_requires_non_empty_packages() -> None:
    with pytest.raises(RecipeValidationError, match="non-empty"):
        PipInstall(packages=[])


def test_fetch_git_requires_repo_and_validates_ref() -> None:
    with pytest.raises(RecipeValidationError, match="requires repo"):
        Fetch(kind="git")
    with pytest.raises(RecipeValidationError):
        Fetch(kind="git", repo="https://x/r.git", ref="not a sha; echo")


def test_fetch_url_validates_sha256() -> None:
    with pytest.raises(RecipeValidationError):
        Fetch(kind="url", url="https://x/a.tar.gz", sha256="short")
    assert (
        Fetch(kind="url", url="https://x/a.tar.gz", sha256="0" * 64).sha256 == "0" * 64
    )


def test_index_url_must_be_url() -> None:
    with pytest.raises(RecipeValidationError):
        PipInstall(packages=["a"], index_url="not-a-url")


def test_sequence_fields_reject_bare_str() -> None:
    with pytest.raises(RecipeValidationError, match="not a bare str"):
        Requires(imports="pkg")  # type: ignore[arg-type]
    with pytest.raises(RecipeValidationError, match="not a bare str"):
        Build(system="cmake")  # type: ignore[arg-type]


def test_setup_env_name_validation() -> None:
    with pytest.raises(RecipeValidationError):
        Setup(env={"BAD NAME": "v"})
    assert Setup(env={"OK_NAME": "v"}).env == {"OK_NAME": "v"}


def test_sequences_normalize_to_tuples() -> None:
    build = Build(system=["a"], pythonpath=["/p"], commands=["echo hi"])
    assert isinstance(build.system, tuple)
    assert isinstance(build.pythonpath, tuple)
    assert isinstance(build.commands, tuple)


def test_installer_must_be_pip_or_uv() -> None:
    with pytest.raises(RecipeValidationError, match=r"pip.*uv"):
        Build(installer="conda")
    assert Build(installer="uv").installer == "uv"


def test_recipe_equality_independent_of_input_container_types() -> None:
    a = Recipe(
        name="a", make=GymMake("E-v0", kwargs={"xs": (1, 2)}), build=Build(system=["x"])
    )
    b = Recipe(
        name="a",
        make=GymMake("E-v0", kwargs={"xs": [1, 2]}),
        build=Build(system=("x",)),
    )
    assert a == b


def test_gym_env_id_allows_colon_load_form() -> None:
    # gymnasium.make accepts module:Name ids (e.g. sai_pygame:SquidHunt-v0); make
    # is a strict superset, so GymMake.env_id must too.
    assert GymMake(env_id="sai_pygame:SquidHunt-v0").env_id == "sai_pygame:SquidHunt-v0"
    assert GymMake(env_id="ALE/Breakout-v5").env_id == "ALE/Breakout-v5"


def test_gym_env_id_still_rejects_whitespace_and_metachars() -> None:
    for bad in ["has space", "a;rm -rf", "-leadingdash"]:
        with pytest.raises(RecipeValidationError):
            GymMake(env_id=bad)


def test_pymake_rejects_malformed_entrypoints() -> None:
    for bad in ["nocolon", "mod:", ":fn", "mod:fn.", "mod:.fn", "mod:a..b"]:
        with pytest.raises(RecipeValidationError):
            PyMake(entrypoint=bad)


def test_pymake_accepts_dotted_classmethod_entrypoint() -> None:
    assert PyMake(entrypoint="mod:Class._rlmesh_construct").entrypoint == (
        "mod:Class._rlmesh_construct"
    )

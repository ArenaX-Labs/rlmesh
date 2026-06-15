"""ModelRecipe + Model: authoring, projection, and the local eval loop.

These classes are defined at module level (not in a fixture/local scope) so
``to_recipe()``'s import-safety guard is satisfied, exactly like EnvRecipe.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import rlmesh
from rlmesh.adapters import (
    ACTION_GRIPPER,
    IMAGE_PRIMARY,
    ActionComponent,
    ActionLayout,
    AdapterResolutionError,
    EnvTags,
    ImageInput,
    ImageTag,
    ModelSpec,
    TextInput,
)
from rlmesh.models import (
    DELEGATED,
    ArtifactInput,
    ModelRecipe,
    RunResult,
    construct_authored_model,
)
from rlmesh.numpy import Model
from rlmesh.recipes import Build, PipInstall, PyMake, Recipe, RecipeValidationError

_SPEC = ModelSpec(
    inputs=(ImageInput("image", role=IMAGE_PRIMARY, size=8),),
    action=ActionLayout(ActionComponent(ACTION_GRIPPER, 1)),
)
_TAGS = EnvTags(
    observation={"image": ImageTag()},
    action=ActionLayout(ActionComponent(ACTION_GRIPPER, 1)),
)


class TinyPolicy(ModelRecipe):
    """A spec'd policy used for projection/serde (one class IS the policy)."""

    name = "policy/tiny"
    build = Build(pip=[PipInstall(["numpy"])])
    spec = _SPEC

    def load(self) -> None:
        self._loaded = True

    def predict(self, observation: Any) -> Any:
        return np.array([0.5], dtype="float32")


class LoopPolicy(ModelRecipe):
    """A spec-less policy (no adapter) used to drive the local eval loop."""

    name = "policy/loop"
    spec = None

    def load(self) -> None:
        self._reset_calls = 0
        self._closed = False

    def predict(self, observation: Any) -> Any:
        return np.array([0.5], dtype="float32")

    def reset(self) -> None:
        self._reset_calls += 1

    def close(self) -> None:
        self._closed = True


class InlinePolicy(ModelRecipe):
    """A local-only spec (InlineCustomInput) -- cannot be projected/sandboxed."""

    name = "policy/inline"

    def load(self) -> None: ...

    def predict(self, observation: Any) -> Any:
        return observation


# InlineCustomInput holds an in-process callable, so the spec is local-only.
def _inline_spec() -> ModelSpec:
    from rlmesh.adapters import InlineCustomInput

    return ModelSpec(
        inputs=(InlineCustomInput("x", lambda obs: obs),),
        action=ActionLayout(ActionComponent(ACTION_GRIPPER, 1)),
    )


InlinePolicy.spec = _inline_spec()


class _Contract:
    def __init__(
        self, metadata: dict[str, Any] | None = None, num_envs: int = 1
    ) -> None:
        self.metadata = metadata or {}
        self.num_envs = num_envs
        self.observation_space: Any = None
        self.action_space: Any = None


class _LoopEnv:
    """A tiny in-process env (reset/step/env_contract) for the eval-loop test."""

    def __init__(self, horizon: int = 4, contract: _Contract | None = None) -> None:
        self.horizon = horizon
        self.t = 0
        self.resets = 0
        self.closed = False
        self.last_seed: int | None = None
        self._contract = contract if contract is not None else _Contract()

    @property
    def env_contract(self) -> _Contract:
        return self._contract

    def reset(
        self, *, seed: int | None = None
    ) -> tuple[dict[str, int], dict[str, Any]]:
        self.t = 0
        self.resets += 1
        self.last_seed = seed
        return {"obs": 0}, {}

    def step(
        self, action: Any
    ) -> tuple[dict[str, int], float, bool, bool, dict[str, Any]]:
        self.t += 1
        terminated = self.t >= self.horizon
        return {"obs": self.t}, 1.0, terminated, False, {}

    def close(self) -> None:
        self.closed = True


# ── projection / serde ──────────────────────────────────────────────────────


def test_to_recipe_is_model_kind() -> None:
    recipe = TinyPolicy.to_recipe()
    assert recipe.kind == "model"
    assert recipe.name == "policy/tiny"
    assert isinstance(recipe.make, PyMake)
    assert recipe.make.entrypoint.endswith(":TinyPolicy._rlmesh_load")
    # at authoring, recipe.adapter holds the ModelSpec INSTANCE...
    assert recipe.adapter is _SPEC
    # ...which serializes to a BARE dict (not the to_metadata wrapper)
    adapter_dict = recipe.to_dict()["adapter"]
    assert isinstance(adapter_dict, dict)
    assert set(adapter_dict) == {"inputs", "action"}


def test_recipe_json_round_trip() -> None:
    recipe = TinyPolicy.to_recipe()
    back = type(recipe).from_json(recipe.to_json())
    assert back.kind == "model"
    assert back.name == recipe.name
    # from_dict rehydrates adapter as a raw Mapping equal to the projected dict
    assert back.adapter == recipe.to_dict()["adapter"]


def test_recipe_inputs_kind_agnostic_for_authored() -> None:
    # An authored (PyMake) env may declare runtime inputs, same as a model.
    env = Recipe(
        name="x/y",
        kind="env",
        make=PyMake(entrypoint="m:C._rlmesh_construct"),
        inputs=(ArtifactInput("assets", "/assets"),),
    )
    assert env.inputs[0].name == "assets"


def test_recipe_inputs_require_authored_recipe() -> None:
    # A gym/hf SOURCE env has no input_path to resolve mounts, so reject inputs.
    from rlmesh.recipes import GymMake

    with pytest.raises(RecipeValidationError, match="authored"):
        Recipe(
            name="x/y",
            kind="env",
            make=GymMake("CartPole-v1"),
            inputs=(ArtifactInput("w", "/w"),),
        )


def test_local_only_spec_rejected_at_projection() -> None:
    with pytest.raises(RecipeValidationError, match="local-only"):
        InlinePolicy.to_recipe()


# ── construction (load populates self; the instance IS the policy) ───────────


def test_construct_authored_model_runs_load() -> None:
    policy = construct_authored_model(TinyPolicy)
    assert isinstance(policy, TinyPolicy)
    assert policy._loaded is True
    out = policy.predict({"image": np.zeros((8, 8, 3), dtype="uint8")})
    assert out.tolist() == [0.5]


def test_input_path_unknown_raises() -> None:
    policy = construct_authored_model(TinyPolicy)
    with pytest.raises(RecipeValidationError, match="no such ArtifactInput"):
        policy.input_path("missing")


def test_artifact_input_local_dir_resolves(tmp_path: Any) -> None:
    class WithWeights(ModelRecipe):
        name = "policy/with-weights"
        inputs = (ArtifactInput("ckpt", "/weights", local_dir=str(tmp_path)),)
        ckpt_dir: str

        def load(self) -> None:
            self.ckpt_dir = self.input_path("ckpt")

        def predict(self, observation: Any) -> Any:
            return observation

    policy = construct_authored_model(WithWeights)
    assert policy.ckpt_dir == str(tmp_path)


def test_artifact_override_wins(tmp_path: Any) -> None:
    other = tmp_path / "other"
    other.mkdir()

    class WithWeights(ModelRecipe):
        name = "policy/override"
        inputs = (ArtifactInput("ckpt", "/weights", local_dir=str(tmp_path)),)
        ckpt_dir: str

        def load(self) -> None:
            self.ckpt_dir = self.input_path("ckpt")

        def predict(self, observation: Any) -> Any:
            return observation

    policy = construct_authored_model(
        WithWeights,
        artifacts=(ArtifactInput("ckpt", "/weights", local_dir=str(other)),),
    )
    assert policy.ckpt_dir == str(other)


# ── the local eval loop (Model.run returns a typed RunResult) ──────────


def test_model_server_run_returns_runresult_and_wires_lifecycle() -> None:
    server = Model(LoopPolicy)
    env = _LoopEnv(horizon=4)
    result = server.run(env, seeds=[1, 2, 3], close_env=True)

    assert isinstance(result, RunResult)
    assert result.num_episodes == 3
    assert result.total_steps == 12  # 3 episodes x 4 steps
    assert result.mean_reward == pytest.approx(4.0)
    assert all(e.terminated for e in result.episodes)
    assert [e.seed for e in result.episodes] == [1, 2, 3]
    # the policy's per-episode reset() fired once per episode; close() fired once
    assert server._policy._reset_calls == 3
    assert server._policy._closed is True
    assert env.resets == 3
    assert env.closed is True  # close_env=True


def test_run_seeds_threaded_to_env() -> None:
    env = _LoopEnv(horizon=2)
    Model(LoopPolicy).run(env, seeds=[7])
    assert env.last_seed == 7  # sole guard the seed reaches env.reset(seed=)


# ── spec=None vs DELEGATED (loud failure, FINAL_API_SPEC §12 B9) ─────────────


def test_spec_none_against_untagged_env_runs() -> None:
    result = Model(LoopPolicy).run(_LoopEnv(horizon=2))
    assert result.num_episodes == 1


def test_spec_none_against_tagged_env_fails_loud() -> None:
    tagged = _LoopEnv(contract=_Contract(metadata=dict(_TAGS.to_metadata())))
    with pytest.raises(AdapterResolutionError, match="spec=None"):
        Model(LoopPolicy).run(tagged)


def test_delegated_skips_resolution_even_when_tagged() -> None:
    class SelfAdapt(ModelRecipe):
        name = "policy/self-adapt"
        spec = DELEGATED

        def load(self) -> None: ...

        def predict(self, observation: Any) -> Any:
            return observation

    tagged = _LoopEnv(horizon=2, contract=_Contract(metadata=dict(_TAGS.to_metadata())))
    result = Model(SelfAdapt).run(tagged)
    assert result.num_episodes == 1


# ── registration + name-based construction ──────────────────────────────────


def test_register_class_and_run_by_name() -> None:
    rlmesh.register(LoopPolicy, overwrite=True)
    result = Model("policy/loop").run(_LoopEnv(horizon=3))
    assert result.total_steps == 3


# ── review regressions ──────────────────────────────────────────────────────


def test_check_passes_for_model_recipe_with_spec() -> None:
    # Regression: check() compared dataclasses, but adapter is a ModelSpec instance
    # pre-round-trip and a dict post, so it spuriously failed. Must not raise now.
    TinyPolicy.check()
    assert InlinePolicy is not None  # keep the local-only policy referenced


def test_flat_hf_form_projects_importable_recipe() -> None:
    import rlmesh.models._registry as reg
    from rlmesh.recipes import resolve as resolve_recipe

    rlmesh.register(
        "policy/flat-openvla",
        hf="org/openvla",
        spec=_SPEC,
        revision="abc123def",
        loader="transformers:AutoModel",
        overwrite=True,
    )
    recipe = resolve_recipe("policy/flat-openvla")
    assert recipe.kind == "model"
    # auto-declared weights mount with the pinned hf uri (per-run artifacts override it)
    assert any(
        a.name == "weights" and a.uri == "hf://org/openvla@abc123def"
        for a in recipe.inputs
    )
    # the projected entrypoint class is bound into the module, so a fresh-process /
    # sandbox import of "rlmesh.models._registry:<Class>._rlmesh_load" resolves
    assert isinstance(recipe.make, PyMake)
    cls_name = recipe.make.entrypoint.split(":", 1)[1].rsplit(".", 1)[0]
    assert cls_name.isidentifier()
    assert hasattr(reg, cls_name)


# ── the REAL adapter path end-to-end (resolve + transform in the loop) ───────


class _TaggedEnv:
    """A steppable env with real gymnasium spaces + published tags.

    Exercises the full O(N+M) path: Model resolves the adapter from this
    env's tags x the model's spec, then transform_obs/transform_action wrap predict.
    """

    def __init__(self, horizon: int = 3) -> None:
        import gymnasium as gym

        self.horizon = horizon
        self.t = 0
        self._obs_space = gym.spaces.Dict(
            {
                "image": gym.spaces.Box(
                    low=0, high=255, shape=(16, 16, 3), dtype=np.uint8
                )
            }
        )
        self._action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        tags = EnvTags(
            observation={"image": ImageTag(role=IMAGE_PRIMARY)},
            action=ActionLayout(ActionComponent(ACTION_GRIPPER, 1)),
        )
        self._contract = _Contract(metadata=dict(tags.to_metadata()))
        # the resolver reads spaces off the contract
        self._contract.observation_space = self._obs_space
        self._contract.action_space = self._action_space
        self.seen_payload_keys: set[str] = set()

    @property
    def env_contract(self) -> _Contract:
        return self._contract

    def reset(
        self, *, seed: int | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self.t = 0
        return {"image": np.zeros((16, 16, 3), dtype=np.uint8)}, {}

    def step(
        self, action: Any
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self.t += 1
        # the action must be the ENV-format action: a length-1 array in [-1, 1]
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        assert arr.shape == (1,)
        terminated = self.t >= self.horizon
        return (
            {"image": np.zeros((16, 16, 3), dtype=np.uint8)},
            1.0,
            terminated,
            False,
            {},
        )


class AdaptedPolicy(ModelRecipe):
    """A spec'd policy: predict sees the MODEL-format payload (image resized to 8x8)."""

    name = "policy/adapted"
    spec = ModelSpec(
        inputs=(ImageInput("image", role=IMAGE_PRIMARY, size=8),),
        action=ActionLayout(ActionComponent(ACTION_GRIPPER, 1)),
    )

    def load(self) -> None:
        self.shapes: list[tuple[int, ...]] = []

    def predict(self, observation: Any) -> Any:
        # the adapter resized the env's 16x16 image to the model's declared 8x8
        self.shapes.append(np.asarray(observation["image"]).shape)
        return np.array([0.3], dtype="float32")


def test_model_server_resolves_and_runs_real_adapter() -> None:
    server = Model(AdaptedPolicy)
    env = _TaggedEnv(horizon=3)
    result = server.run(env)
    assert result.num_episodes == 1
    assert result.total_steps == 3
    # predict saw the MODEL-format observation: image resized to the declared 8x8
    assert server._policy.shapes
    assert all(shape[:2] == (8, 8) for shape in server._policy.shapes)


def test_instruction_seam_delivers_text_to_model() -> None:
    captured: list[str] = []

    class WithText(ModelRecipe):
        name = "policy/with-text"
        # TextInput carries a default so resolve() succeeds when the env does not
        # tag text; run(instruction=) overrides it per episode (the language seam).
        spec = ModelSpec(
            inputs=(
                ImageInput("image", role=IMAGE_PRIMARY, size=8),
                TextInput("task", default=""),
            ),
            action=ActionLayout(ActionComponent(ACTION_GRIPPER, 1)),
        )

        def load(self) -> None: ...

        def predict(self, observation: Any) -> Any:
            captured.append(observation.get("task"))
            return np.array([0.0], dtype="float32")

    server = Model(WithText)
    server.run(_TaggedEnv(horizon=2), instruction="pick up the red block")
    assert captured and all(c == "pick up the red block" for c in captured)


# ---------------------------------------------------------------------------
# Artifact bind-mounts (SandboxModel local_dir -> container)
# ---------------------------------------------------------------------------


def test_local_dir_mounts_binds_only_local_dir_inputs(tmp_path: Any) -> None:
    from rlmesh.recipes._artifacts import local_dir_mounts
    from rlmesh.recipes._schema import ArtifactInput

    weights = tmp_path / "weights"
    weights.mkdir()
    inputs = (
        ArtifactInput("weights", "/rlmesh/input/model/weights", local_dir=str(weights)),
        ArtifactInput("remote", "/rlmesh/input/model/remote", uri="hf://org/repo@abc"),
    )
    # Only the local_dir input mounts; the uri-backed one resolves in-container.
    assert local_dir_mounts(inputs) == [
        (str(weights.resolve()), "/rlmesh/input/model/weights")
    ]


def test_local_dir_mounts_override_supplies_local_dir(tmp_path: Any) -> None:
    from rlmesh.recipes._artifacts import local_dir_mounts
    from rlmesh.recipes._schema import ArtifactInput

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    inputs = (
        ArtifactInput(
            "weights", "/rlmesh/input/model/weights", uri="hf://org/repo@abc"
        ),
    )
    override = ArtifactInput("weights", "/ignored", local_dir=str(ckpt))
    # The override's local_dir mounts at the DECLARED target the container resolves.
    assert local_dir_mounts(inputs, (override,)) == [
        (str(ckpt.resolve()), "/rlmesh/input/model/weights")
    ]


@pytest.mark.parametrize(
    ("case", "exc", "match"),
    [
        ("missing_dir", FileNotFoundError, "local_dir is not a directory"),
        ("unknown_override", ValueError, "matches no declared input"),
        ("unsafe_target", ValueError, "absolute container path without"),
    ],
)
def test_local_dir_mounts_loud_failures(
    tmp_path: Any, case: str, exc: type[Exception], match: str
) -> None:
    from rlmesh.recipes._artifacts import local_dir_mounts
    from rlmesh.recipes._schema import ArtifactInput

    here = str(tmp_path)
    inputs: Any
    overrides: Any = ()
    if case == "missing_dir":
        inputs = (ArtifactInput("weights", "/t", local_dir=str(tmp_path / "nope")),)
    elif case == "unknown_override":
        inputs = (ArtifactInput("weights", "/rlmesh/input/model/weights"),)
        overrides = (ArtifactInput("typo", "/x", local_dir=here),)
    else:  # unsafe_target
        inputs = (ArtifactInput("weights", "/rlmesh/../etc", local_dir=here),)

    with pytest.raises(exc, match=match):
        local_dir_mounts(inputs, overrides)

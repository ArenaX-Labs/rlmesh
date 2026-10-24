from __future__ import annotations

from typing import Any, cast

import pytest
import rlmesh


def test_episode_success_reads_gymnasium_info_keys() -> None:
    from rlmesh._models._eval import episode_success

    assert episode_success({"is_success": True}) is True
    assert episode_success({"success": False}) is False
    assert episode_success({}) is None  # no signal -> unknown, never inferred


def test_success_rate_counts_only_the_env_reported_outcome() -> None:
    from rlmesh import RunResult
    from rlmesh._models._eval import EpisodeResult

    episodes = (
        # terminated but the env reported NOT a success
        EpisodeResult(0, None, 5, 1.0, terminated=True, truncated=False, success=False),
        # truncated but the env reported success
        EpisodeResult(1, None, 5, 1.0, terminated=False, truncated=True, success=True),
    )
    assert RunResult(episodes).success_rate == pytest.approx(0.5)


def test_success_rate_is_none_when_any_episode_lacks_a_signal() -> None:
    from rlmesh import RunResult
    from rlmesh._models._eval import EpisodeResult

    reported = EpisodeResult(
        0, None, 5, 1.0, terminated=True, truncated=False, success=True
    )
    # terminated, but the env never said whether that was a success
    unknown = EpisodeResult(1, None, 5, 1.0, terminated=True, truncated=False)
    assert RunResult((reported, unknown)).success_rate is None
    assert RunResult((unknown,)).success_rate is None
    assert RunResult(()).success_rate is None
    assert RunResult((reported,)).success_rate == 1.0


def test_local_contract_builds_from_a_bare_env_without_forwarding_lookups() -> None:
    # A local env (no env_contract) yields a contract synthesized from its spaces
    # and metadata; num_envs must NOT be probed via __getattr__, so a gymnasium
    # wrapper does not emit its deprecated attribute-forwarding warning.
    from rlmesh._models._connect import local_contract

    forwarded: list[str] = []

    class ForwardingEnv:
        observation_space = "obs"
        action_space = "act"

        def __init__(self) -> None:
            self.metadata = {"k": "v"}

        def __getattr__(self, name: str) -> object:  # gymnasium warns in here
            forwarded.append(name)
            raise AttributeError(name)

        def reset(self) -> None: ...
        def step(self, action: object) -> None: ...

    contract = local_contract(ForwardingEnv())
    assert contract.num_envs == 1
    assert contract.observation_space == "obs"
    assert contract.metadata == {"k": "v"}
    assert forwarded == []  # nothing reached the forwarding __getattr__


def test_shutdown_calls_shutdown_plainly_with_no_reason() -> None:
    # shutdown_env calls the owner-level shutdown() with no arguments; a remote
    # owner's shutdown(reason=...) keeps its own default.
    from rlmesh._models._connect import shutdown_env

    calls: list[tuple[object, ...]] = []

    class Owner:
        def shutdown(self, reason: str = "owner shutdown") -> None:
            calls.append((reason,))

    shutdown_env(Owner())
    assert calls == [("owner shutdown",)]


def test_shutdown_falls_back_to_close() -> None:
    from rlmesh._models._connect import shutdown_env

    calls: list[tuple[object, ...]] = []

    class CloseOnly:
        def close(self) -> None:
            calls.append(())

    shutdown_env(CloseOnly())
    assert calls == [()]


def test_predict_can_pass_optional_context_with_episode_seed() -> None:
    import rlmesh
    from rlmesh.numpy import Model

    seen: list[dict[str, object]] = []

    class Env:
        observation_space = rlmesh.spaces.Discrete(3)
        action_space = rlmesh.spaces.Discrete(2)

        def __init__(self) -> None:
            self.step_count = 0

        def reset(
            self, *, seed: int | None = None, options: dict[str, object] | None = None
        ):
            _ = seed, options
            self.step_count = 0
            return 0, {}

        def step(self, action: object):
            _ = action
            self.step_count += 1
            return 0, 1.0, self.step_count >= 1, False, {}

        def close(self) -> None:
            return None

    def predict(observation: object, context: dict[str, object]) -> int:
        seen.append({"observation": observation, **context})
        return 0

    result = Model(predict).run(Env(), seeds=[7, 8], episodes=2)

    assert result.num_episodes == 2
    assert [episode.seed for episode in result.episodes] == [7, 8]
    assert [item["episode_seed"] for item in seen] == [7, 8]
    assert all(isinstance(item["episode_id"], str) for item in seen)


def test_bound_predict_context_carries_stable_episode_identity() -> None:
    import rlmesh
    from rlmesh.numpy import Model

    seen: list[dict[str, object]] = []

    class Env:
        observation_space = rlmesh.spaces.Discrete(5)
        action_space = rlmesh.spaces.Discrete(2)

        def __init__(self) -> None:
            self.step_count = 0

        def reset(
            self, *, seed: int | None = None, options: dict[str, object] | None = None
        ):
            _ = seed, options
            self.step_count = 0
            return 0, {}

        def step(self, action: object):
            _ = action
            self.step_count += 1
            return 0, 1.0, self.step_count >= 3, False, {}

        def close(self) -> None:
            return None

    def predict(observation: object, context: dict[str, object]) -> int:
        seen.append(dict(context))
        return 0

    with Model(predict).bind(Env()) as binding:
        binding.reset(seed=7)
        while not binding.done:
            binding.step()

    # One id for the whole episode, the seed riding on every step, and the
    # re-plan ordinal counting up.
    assert len(seen) == 3
    ids = {context["episode_id"] for context in seen}
    assert len(ids) == 1 and ids != {""}
    assert [context["episode_seed"] for context in seen] == [7, 7, 7]
    assert [context["predict_index"] for context in seen] == [0, 1, 2]
    assert [context["predict_seed"] for context in seen] == [
        rlmesh.predict_seed(7, index) for index in range(3)
    ]


def test_no_adapter_sentinel_skips_adapter_resolution_for_tagged_env() -> None:
    from types import SimpleNamespace

    import rlmesh
    import rlmesh.adapters as adapt
    from rlmesh._models._eval import resolve_adapter

    tags = adapt.EnvTags(
        observation={}, action=adapt.Action(adapt.Actuator(adapt.ACTION_GRIPPER, dim=1))
    )
    contract = SimpleNamespace(metadata=tags.to_metadata())

    assert resolve_adapter(rlmesh.NO_ADAPTER, cast(Any, contract), False) is None


def test_spec_none_rejects_tagged_env_with_no_adapter_hint() -> None:
    from types import SimpleNamespace

    import rlmesh.adapters as adapt
    from rlmesh._models._eval import resolve_adapter

    tags = adapt.EnvTags(
        observation={}, action=adapt.Action(adapt.Actuator(adapt.ACTION_GRIPPER, dim=1))
    )
    contract = SimpleNamespace(metadata=tags.to_metadata())

    with pytest.raises(adapt.AdapterResolutionError, match="spec=NO_ADAPTER"):
        resolve_adapter(None, cast(Any, contract), False)


def test_random_sample_samples_action_space_and_skips_adapter_on_tagged_env() -> None:
    # A RANDOM_SAMPLE baseline adapts nothing, so it drives a tagged env without
    # raising AdapterResolutionError, stepping it under actions of its space.
    import rlmesh
    import rlmesh.adapters as adapt

    tags = adapt.EnvTags(
        observation={}, action=adapt.Action(adapt.Actuator(adapt.ACTION_GRIPPER, dim=1))
    )
    actions: list[object] = []

    class Env:
        observation_space = rlmesh.spaces.Dict({"x": rlmesh.spaces.Discrete(3)})
        action_space = rlmesh.spaces.Box(-1.0, 1.0, shape=(1,))
        metadata = tags.to_metadata()

        def reset(
            self, *, seed: object = None, options: object = None
        ) -> tuple[object, dict[str, object]]:
            return {"x": 0}, {}

        def step(self, action: object) -> tuple[object, float, bool, bool, dict]:
            actions.append(action)
            return {"x": 0}, 1.0, True, False, {}

        def close(self) -> None:
            return None

    with rlmesh.Model(rlmesh.RANDOM_SAMPLE).bind(Env()) as binding:
        binding.reset(seed=0)
        binding.step()
    assert len(actions) == 1
    assert Env.action_space.contains(actions[0])


def test_invalid_model_spec_mentions_no_adapter_sentinel() -> None:
    from types import SimpleNamespace

    import rlmesh.adapters as adapt
    from rlmesh._models._eval import resolve_adapter

    with pytest.raises(adapt.AdapterResolutionError, match="ModelSpec or NO_ADAPTER"):
        resolve_adapter(object(), cast(Any, SimpleNamespace(metadata={})), False)


# rlmesh moves every obs tensor leaf onto the model's `device` before predict,
# so authors delete `.to(self.device)`. The dual of the env-side action
# placement (test_env_framework.py); real-GPU round-trips are the integration
# suite's, these are fast wiring checks.


def test_to_device_places_torch_obs_leaves_and_passes_non_tensors() -> None:
    torch = pytest.importorskip("torch")
    import rlmesh.torch

    class M(rlmesh.torch.Model):
        spec = rlmesh.NO_ADAPTER

        def predict(self, observation: Any) -> Any:
            return observation

    m = M()
    m.device = "cpu"
    placed = cast("dict[str, Any]", m._to_device({"img": torch.zeros(2), "n": 5}))
    assert placed["img"].device.type == "cpu"
    assert placed["n"] == 5  # non-tensor leaf passes through unchanged


def test_to_device_reads_device_dynamically(monkeypatch: pytest.MonkeyPatch) -> None:
    # The served worker is built (in __init__) BEFORE load() runs on the served
    # path, so the corners must read `device` at predict time, not snapshot it.
    # Mirror that: set `device` AFTER construction and confirm it is honored.
    pytest.importorskip("torch")
    import rlmesh.torch

    class M(rlmesh.torch.Model):
        spec = rlmesh.NO_ADAPTER

        def predict(self, observation: Any) -> Any:
            return observation

    m = M()
    seen: list[object] = []
    monkeypatch.setattr(
        type(m)._bridge, "to_device", lambda value, device: seen.append(device) or value
    )
    m.device = "cuda:7"  # set late, as load() would
    m._to_device({"x": 1})
    assert seen == ["cuda:7"]  # read the post-construction value, not a snapshot


def test_device_is_noop_and_guard_rejects_for_numpy_model() -> None:
    import numpy as np
    import rlmesh
    import rlmesh.numpy

    class M(rlmesh.numpy.Model):
        spec = rlmesh.NO_ADAPTER

        def predict(self, observation: Any) -> Any:
            return observation

    m = M()
    m.device = "cpu"
    # numpy has no device: placement is a silent no-op (same array back)...
    arr = np.zeros(3)
    assert cast("dict[str, Any]", m._to_device({"x": arr}))["x"] is arr
    # ...but the guard rejects the meaningless device at a drive entry point.
    with pytest.raises(ValueError, match="device"):
        m._require_device_support()


def test_device_guard_allows_a_torch_model() -> None:
    pytest.importorskip("torch")
    import rlmesh.torch

    class M(rlmesh.torch.Model):
        spec = rlmesh.NO_ADAPTER

        def predict(self, observation: Any) -> Any:
            return observation

    m = M()
    m.device = "cpu"
    m._require_device_support()  # does not raise


def test_model_construction_builds_no_native_worker() -> None:
    """The native worker (and its runtime) is lazy: a Model that is only ever
    driven locally must not pay for serve-path machinery at construction."""
    import rlmesh

    model = rlmesh.Model(lambda obs: 0)
    assert model._worker is None
    assert model._install_worker() is model._worker
    assert model._install_worker() is model._worker  # cached, built once


def test_allow_fusion_reaches_the_native_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``Model.allow_fusion`` is the author's cross-route fusion permission: it
    must reach the native ``PyModel`` worker as the ``allow_fusion`` kwarg --
    True by default, False when the author opts a shape-pinned batched forward
    out (grouped predicts then serve per route)."""
    import rlmesh
    import rlmesh._load_native as load_native_mod

    captured: dict[str, Any] = {}

    def fake_worker(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(load_native_mod, "load_native", lambda name: fake_worker)

    rlmesh.Model(lambda obs: 0)._install_worker()
    assert captured["allow_fusion"] is True

    class OptedOut(rlmesh.Model):
        allow_fusion = False

        def predict(self, observation: object) -> object:
            return 0

    captured.clear()
    OptedOut()._install_worker()
    assert captured["allow_fusion"] is False


def test_native_chunk_is_read_after_load_not_at_class_definition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """K usually comes off the loaded checkpoint, so the worker must read
    ``native_chunk`` AFTER ``load()`` ran -- two instances whose ``load()`` picked
    different K build two workers declaring different K."""
    import rlmesh
    import rlmesh._load_native as load_native_mod

    captured: list[Any] = []

    def fake_worker(**kwargs: Any) -> Any:
        captured.append(kwargs.get("native_chunk"))
        return object()

    monkeypatch.setattr(load_native_mod, "load_native", lambda name: fake_worker)

    class FromCheckpoint(rlmesh.Model):
        def __init__(self, k: int) -> None:
            self._k = k
            super().__init__()

        def load(self) -> None:
            self.native_chunk = self._k

        def predict(self, observation: object) -> object:
            return 0

        def predict_chunk(self, observation: object) -> object:
            return [0]

    FromCheckpoint(30)._install_worker()
    FromCheckpoint(50)._install_worker()
    assert captured == [30, 50]

    # Undeclared stays undeclared: the elastic contract reaches the worker as None.
    captured.clear()
    rlmesh.Model(lambda obs: 0)._install_worker()
    assert captured == [None]


def test_native_chunk_without_a_chunk_corner_is_rejected_at_construction() -> None:
    """A class-level K on a model with no chunk corner could never be honored --
    and nothing downstream would ever check it -- so it fails where the corners
    are known."""
    import rlmesh

    class NoCorner(rlmesh.Model):
        native_chunk = 8

        def predict(self, observation: object) -> object:
            return 0

    with pytest.raises(TypeError, match="no predict_chunk"):
        NoCorner()


def test_native_chunk_below_one_is_rejected_at_worker_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rlmesh
    import rlmesh._load_native as load_native_mod

    monkeypatch.setattr(
        load_native_mod, "load_native", lambda name: lambda **kw: object()
    )

    class Zero(rlmesh.Model):
        native_chunk = 0

        def predict(self, observation: object) -> object:
            return 0

        def predict_chunk(self, observation: object) -> object:
            return [0]

    with pytest.raises(ValueError, match="native_chunk must be >= 1"):
        Zero()._install_worker()


def test_a_duck_typed_policy_declares_its_own_native_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``Model(policy)`` reads K off the coerced policy, so a wrapped object
    declares exactly the way a subclass does."""
    import rlmesh
    import rlmesh._load_native as load_native_mod

    captured: list[Any] = []

    def fake_worker(**kwargs: Any) -> Any:
        captured.append(kwargs.get("native_chunk"))
        return object()

    monkeypatch.setattr(load_native_mod, "load_native", lambda name: fake_worker)

    class Policy:
        native_chunk = 12

        def predict(self, observation: object) -> object:
            return 0

        def predict_chunk(self, observation: object) -> object:
            return [0]

    rlmesh.Model(Policy())._install_worker()
    assert captured == [12]


def test_a_subclass_reset_hook_is_refused_at_class_creation() -> None:
    with pytest.raises(TypeError, match="on_episode_end"):

        class Stale(rlmesh.Model):
            def reset(self) -> None: ...


def test_a_wrapped_callable_object_keeps_its_reset_and_close_hooks() -> None:
    from rlmesh._models._coerce import coerce_model

    class Policy:
        def __call__(self, observation: object) -> int:
            return 0

        def reset(self) -> None: ...

        def close(self) -> None: ...

    policy = Policy()
    coerced = coerce_model(policy, spec=None)
    assert coerced.on_episode_end == policy.reset
    assert coerced.on_close == policy.close


@pytest.mark.parametrize("invalid", [-1, 0, 1025, 2**64])
def test_loaded_execution_horizon_is_validated_by_rust(invalid):
    from rlmesh import Model
    from rlmesh._rlmesh import resolve_execution_horizon

    class Invalid(Model):
        def load(self):
            self.execution_horizon = invalid

        def predict(self, observation):
            return 0

    with pytest.raises(ValueError, match="execution_horizon must be"):
        Invalid()
    with pytest.raises(ValueError, match="exceeds the model's declared native_chunk"):
        resolve_execution_horizon(None, 4, 2, 3)
    assert resolve_execution_horizon(1, 4, 2, 3) == 1

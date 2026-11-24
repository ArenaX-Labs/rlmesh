from __future__ import annotations

from typing import Any, cast

import pytest
import rlmesh


def test_episode_success_reads_gymnasium_info_keys() -> None:
    from rlmesh._models._eval import episode_success

    assert episode_success({"is_success": True}) is True
    assert episode_success({"success": False}) is False
    assert episode_success({}) is None


def test_success_rate_counts_only_the_env_reported_outcome() -> None:
    from rlmesh import RunResult
    from rlmesh._models._eval import EpisodeResult

    episodes = (
        EpisodeResult(0, None, 5, 1.0, terminated=True, truncated=False, success=False),
        EpisodeResult(1, None, 5, 1.0, terminated=False, truncated=True, success=True),
    )
    assert RunResult(episodes).success_rate == pytest.approx(0.5)


def test_success_rate_counts_a_missing_signal_as_a_failure() -> None:
    from rlmesh import RunResult
    from rlmesh._models._eval import EpisodeResult

    reported = EpisodeResult(
        0, None, 5, 1.0, terminated=True, truncated=False, success=True
    )
    unknown = EpisodeResult(1, None, 5, 1.0, terminated=True, truncated=False)
    assert RunResult((reported, unknown)).success_rate == pytest.approx(0.5)
    assert RunResult((unknown,)).success_rate is None
    assert RunResult(()).success_rate is None
    assert RunResult((reported,)).success_rate == 1.0


def test_local_contract_builds_from_a_bare_env_without_forwarding_lookups() -> None:
    from rlmesh._models._connect import local_contract

    forwarded: list[str] = []

    class ForwardingEnv:
        observation_space = "obs"
        action_space = "act"

        def __init__(self) -> None:
            self.metadata = {"k": "v"}

        def __getattr__(self, name: str) -> object:
            forwarded.append(name)
            raise AttributeError(name)

        def reset(self) -> None: ...
        def step(self, action: object) -> None: ...

    contract = local_contract(ForwardingEnv())
    assert contract.num_envs == 1
    assert contract.observation_space == "obs"
    assert contract.metadata == {"k": "v"}
    assert forwarded == []


def test_shutdown_calls_shutdown_plainly_with_no_reason() -> None:
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
    assert placed["n"] == 5


def test_to_device_reads_device_dynamically(monkeypatch: pytest.MonkeyPatch) -> None:
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
    m.device = "cuda:7"
    m._to_device({"x": 1})
    assert seen == ["cuda:7"]


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
    arr = np.zeros(3)
    assert cast("dict[str, Any]", m._to_device({"x": arr}))["x"] is arr
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
    m._require_device_support()


def test_model_construction_builds_no_native_worker() -> None:
    import rlmesh

    model = rlmesh.Model(lambda obs: 0)
    assert model._worker is None
    assert model._install_worker() is model._worker
    assert model._install_worker() is model._worker


def test_allow_fusion_reaches_the_native_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    captured.clear()
    rlmesh.Model(lambda obs: 0)._install_worker()
    assert captured == [None]


def test_native_chunk_without_a_chunk_corner_is_rejected_at_construction() -> None:
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

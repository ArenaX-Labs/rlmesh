"""A model's lifecycle: serialized calls, terminal close, and declarations read once.

Pins the language-parity Model changes (MDL-P1 to MDL-P9) on the local paths:
the authored close runs once, close is terminal, every corner and hook call
on one model is serialized and refuses re-entry, the episode store drops an
env's entries on teardown, ``native_chunk`` is read once after ``load``, the
declarations are checked before ``load``, a wrapped policy's declarations are
the model's, and a tuple is one chunk frame.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest
import rlmesh
from rlmesh._models._episodes import EpisodeStore


class _Env:
    """A local env: three steps, then terminates."""

    def __init__(self) -> None:
        from rlmesh import spaces

        self.observation_space = spaces.Discrete(4)
        self.action_space = spaces.Discrete(2)
        self._t = 0

    def reset(
        self, *, seed: object = None, options: object = None
    ) -> tuple[int, dict[str, Any]]:
        _ = (seed, options)
        self._t = 0
        return 0, {}

    def step(self, action: object) -> tuple[int, float, bool, bool, dict[str, Any]]:
        _ = action
        self._t += 1
        return self._t, 1.0, self._t >= 3, False, {}

    def close(self) -> None:
        pass


def test_an_authored_close_runs_at_most_once() -> None:
    calls: list[str] = []

    class M(rlmesh.Model):
        def predict(self, observation: Any) -> Any:
            return 0

        def close(self) -> None:
            calls.append("close")
            super().close()

    model = M()
    model.close()
    model.close()
    assert calls == ["close"]


def test_an_authored_close_in_a_subclass_chain_runs_each_body_once() -> None:
    calls: list[str] = []

    class A(rlmesh.Model):
        def predict(self, observation: Any) -> Any:
            return 0

        def close(self) -> None:
            calls.append("A")
            super().close()

    class B(A):
        def close(self) -> None:
            calls.append("B")
            super().close()

    model = B()
    model.close()
    model.close()
    assert calls == ["B", "A"]


def test_on_close_replaces_an_authored_close() -> None:
    calls: list[str] = []

    class M(rlmesh.Model):
        def predict(self, observation: Any) -> Any:
            return 0

        def close(self) -> None:
            calls.append("authored")

    model = M(on_close=lambda: calls.append("option"))
    model.close()
    model.close()
    assert calls == ["option"]


def test_close_is_terminal_for_runs_bindings_and_open_bindings() -> None:
    ended: list[str] = []
    model = rlmesh.Model(
        lambda obs: 0, on_episode_end=lambda episode_id: ended.append(episode_id)
    )
    binding = model.bind(_Env())
    binding.reset()
    binding.step()
    model.close()
    with pytest.raises(RuntimeError, match="the model is closed"):
        binding.step()
    binding.close()
    assert ended == [], "a binding closed after the model fires no on_episode_end"
    with pytest.raises(RuntimeError, match="the model is closed"):
        model.bind(_Env())
    with pytest.raises(RuntimeError, match="the model is closed"):
        model.run(_Env())
    with pytest.raises(RuntimeError, match="the model is closed"):
        model.serve("127.0.0.1:0")


def test_a_corner_that_re_enters_its_own_model_raises_instead_of_deadlocking() -> None:
    seen: list[BaseException] = []
    holder: dict[str, Any] = {}

    def predict(observation: Any) -> Any:
        try:
            holder["model"].close()
        except RuntimeError as error:
            seen.append(error)
        return 0

    model = rlmesh.Model(predict)
    holder["model"] = model
    with model.bind(_Env()) as binding:
        binding.reset()
        binding.step()
    assert len(seen) == 1
    assert "re-entered its own Model" in str(seen[0])
    model.close()


def test_corner_calls_on_one_model_are_serialized_across_bindings() -> None:
    lock = threading.Lock()
    active = {"now": 0, "most": 0}

    def predict(observation: Any) -> Any:
        with lock:
            active["now"] += 1
            active["most"] = max(active["most"], active["now"])
        time.sleep(0.01)
        with lock:
            active["now"] -= 1
        return 0

    model = rlmesh.Model(predict)

    def drive() -> None:
        with model.bind(_Env()) as binding:
            for _ in range(2):
                binding.reset()
                while not binding.done:
                    binding.step()

    threads = [threading.Thread(target=drive) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert active["most"] == 1


def test_the_store_drops_an_envs_entries_without_firing_the_hook() -> None:
    ended: list[str] = []
    store = EpisodeStore(ended.append)
    store.context({"episode_id": "a", "episode_seed": None, "env_id": "x"})
    store.context({"episode_id": "b", "episode_seed": None, "env_id": "y"})
    store.drop_env("x")
    assert len(store) == 1
    assert ended == []
    assert store.context({"episode_id": "b", "env_id": "y"})["predict_index"] == 1


def test_a_loaded_native_chunk_is_checked_once_after_load() -> None:
    class Zero(rlmesh.Model):
        def load(self) -> None:
            self.native_chunk = 0

        def predict(self, observation: Any) -> Any:
            return 0

        def predict_chunk(self, observation: Any) -> Any:
            return [0]

    with pytest.raises(ValueError, match="native_chunk must be >= 1"):
        Zero()

    class NoChunkCorner(rlmesh.Model):
        def load(self, k: int = 2) -> None:
            self.native_chunk = k

        def predict(self, observation: Any) -> Any:
            return 0

    with pytest.raises(TypeError, match="defines no predict_chunk"):
        NoChunkCorner()
    with pytest.raises(TypeError, match="defines no predict_chunk"):
        NoChunkCorner.from_config(k=3)

    class Loaded(rlmesh.Model):
        def load(self) -> None:
            self.native_chunk = 4

        def predict(self, observation: Any) -> Any:
            return 0

        def predict_chunk(self, observation: Any) -> Any:
            return [0] * 4

    model = Loaded()
    model.native_chunk = 9
    assert model._native_chunk() == 4  # pyright: ignore[reportPrivateUsage]


def test_declarations_are_checked_before_load() -> None:
    loads: list[str] = []

    class Bad(rlmesh.Model):
        native_chunk = 4

        def load(self) -> None:
            loads.append("load")

        def predict(self, observation: Any) -> Any:
            return 0

    with pytest.raises(TypeError, match="defines no predict_chunk"):
        Bad()
    assert loads == []


def test_a_wrapped_policy_declarations_are_the_models() -> None:
    class Policy:
        native_chunk = 2
        allow_fusion = False
        workflow_edition = "2026.01"

        def predict(self, observation: Any) -> Any:
            return 0

        def predict_chunk(self, observation: Any) -> Any:
            return [0, 0]

    model = rlmesh.Model(Policy())
    assert model.native_chunk == 2
    assert model.allow_fusion is False
    assert model.workflow_edition == "2026.01"
    bare = rlmesh.Model(lambda obs: 0)
    assert (bare.native_chunk, bare.allow_fusion, bare.workflow_edition) == (
        None,
        True,
        None,
    )

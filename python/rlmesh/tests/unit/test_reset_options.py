from __future__ import annotations

from typing import Any, ClassVar, cast

import pytest
import rlmesh


class _TinyEnv:
    metadata: ClassVar[dict[str, Any]] = {"render_modes": []}

    def __init__(self) -> None:
        from rlmesh import spaces

        self.observation_space = spaces.Discrete(1)
        self.action_space = spaces.Discrete(1)
        self.resets: list[dict[str, Any]] = []

    def reset(
        self, *, seed: object = None, options: Any = None
    ) -> tuple[int, dict[str, object]]:
        self.resets.append({"seed": seed, "options": options})
        return 0, {}

    def step(self, action: object) -> tuple[int, float, bool, bool, dict[str, object]]:
        return 0, 1.0, True, False, {}

    def close(self) -> None:
        pass


class _PlainFactory(rlmesh.EnvFactory):
    def make(self, **kwargs: Any) -> Any:
        return _TinyEnv()


class _TrialFactory(rlmesh.EnvFactory):
    reset_options = ("trial_index",)

    def make(self, **kwargs: Any) -> Any:
        return _TinyEnv()


def test_reset_options_key_matches_the_runtime_constant() -> None:
    assert rlmesh.ENV_RESET_OPTIONS_KEY == "rlmesh.env.v1.reset_options"


def test_make_stamps_the_declaration_without_mutating_the_class() -> None:
    class_metadata = dict(_TinyEnv.metadata)
    env = _TrialFactory().make()

    assert env.metadata[rlmesh.ENV_RESET_OPTIONS_KEY] == ["trial_index"]
    assert _TinyEnv.metadata == class_metadata
    assert rlmesh.ENV_RESET_OPTIONS_KEY not in _TinyEnv.metadata
    assert env.metadata["render_modes"] == []


def test_make_stamps_nothing_when_no_option_is_declared() -> None:
    env = _PlainFactory().make()
    assert rlmesh.ENV_RESET_OPTIONS_KEY not in dict(env.metadata)


def test_trial_index_helper_reads_the_reserved_key() -> None:
    assert rlmesh.trial_index({"trial_index": 12}) == 12
    assert rlmesh.trial_index({"trial_index": 12.0}) == 12
    assert rlmesh.trial_index({"other": 1}) is None
    assert rlmesh.trial_index({"trial_index": True}) is None
    assert rlmesh.trial_index(None) is None
    assert rlmesh.trial_index({"trial_index": 12.5}) is None
    assert rlmesh.trial_index({"trial_index": "12"}) is None


def test_trial_index_helper_accepts_any_integral_number() -> None:
    np = pytest.importorskip("numpy")

    value = rlmesh.trial_index({"trial_index": np.int64(7)})
    assert value == 7
    assert type(value) is int
    assert rlmesh.trial_index({"trial_index": np.uint32(3)}) == 3
    assert rlmesh.trial_index({"trial_index": np.bool_(True)}) is None


def test_trial_index_helper_reads_the_multi_lane_list_by_lane() -> None:
    options = {"trial_index": [4, 5, 6]}
    assert [rlmesh.trial_index(options, lane=i) for i in range(3)] == [4, 5, 6]
    with pytest.raises(ValueError, match="pass lane="):
        rlmesh.trial_index(options)
    with pytest.raises(ValueError, match="lane=3"):
        rlmesh.trial_index(options, lane=3)
    assert rlmesh.trial_index({"trial_index": 9}, lane=0) == 9
    with pytest.raises(ValueError, match="lane=1"):
        rlmesh.trial_index({"trial_index": 9}, lane=1)
    assert rlmesh.trial_index({}, lane=2) is None


class _NoOptionsEnv:
    def __init__(self) -> None:
        from rlmesh import spaces

        self.observation_space = spaces.Discrete(1)
        self.action_space = spaces.Discrete(1)

    def reset(self, *, seed: object = None) -> tuple[int, dict[str, object]]:
        return 0, {}

    def step(self, action: object) -> tuple[int, float, bool, bool, dict[str, object]]:
        return 0, 1.0, True, False, {}

    def close(self) -> None:
        pass


def test_served_env_without_an_options_parameter_still_resets() -> None:
    from rlmesh.numpy import RemoteEnv

    try:
        server = rlmesh.EnvServer(cast("Any", _NoOptionsEnv()), ":0/env")
    except ConnectionError as exc:
        if "Operation not permitted" in str(exc):
            pytest.skip("local tcp bind is not permitted in this environment")
        raise
    server.start()
    try:
        handle = RemoteEnv(server.address)
        obs, _info = handle.reset(options={"trial_index": 3})
        assert obs is not None
        handle.close()
    finally:
        server.shutdown()


def test_native_run_delivers_the_ordinal_over_the_wire() -> None:
    from rlmesh.numpy import Model

    env = _TrialFactory().make()
    try:
        server = rlmesh.EnvServer(env, ":0/env")
    except ConnectionError as exc:
        if "Operation not permitted" in str(exc):
            pytest.skip("local tcp bind is not permitted in this environment")
        raise
    server.start()
    try:
        worker = Model(lambda obs: 0)._install_worker()  # pyright: ignore[reportPrivateUsage]
        report = worker.run_local_for_episodes(
            server.address, 3, 1, None, None, None, False, 40
        )
    finally:
        server.shutdown()

    assert [reset["options"] for reset in env.resets] == [
        {"trial_index": 40},
        {"trial_index": 41},
        {"trial_index": 42},
    ]
    assert [episode["trial"] for episode in report["episodes"]] == [40, 41, 42]


def _run_native(env: _TinyEnv, **kwargs: Any) -> rlmesh.RunResult:
    from rlmesh.numpy import Model

    try:
        return Model(lambda obs: 0).run(env, **kwargs)
    except ConnectionError as exc:
        if "Operation not permitted" in str(exc):
            pytest.skip("local tcp bind is not permitted in this environment")
        raise


def test_native_run_delivers_the_ordinal_by_default() -> None:
    env = _TrialFactory().make()
    result = _run_native(env, episodes=3)

    assert [reset["options"] for reset in env.resets] == [
        {"trial_index": 0},
        {"trial_index": 1},
        {"trial_index": 2},
    ]
    assert [episode.trial for episode in result.episodes] == [0, 1, 2]


def test_native_run_walks_the_ordinals_from_the_base() -> None:
    env = _TrialFactory().make()
    result = _run_native(env, episodes=2, trial_index_base=5)

    assert [reset["options"] for reset in env.resets] == [
        {"trial_index": 5},
        {"trial_index": 6},
    ]
    assert [episode.trial for episode in result.episodes] == [5, 6]

    with pytest.raises(ValueError, match="trial_index_base"):
        _run_native(env, episodes=1, trial_index_base=-1)


def test_native_run_reports_zero_based_episode_indices() -> None:
    env = _TrialFactory().make()
    result = _run_native(env, seeds=range(3), trial_index_base=7)

    assert [episode.index for episode in result.episodes] == [0, 1, 2]
    assert [episode.trial for episode in result.episodes] == [
        7 + episode.index for episode in result.episodes
    ]


def test_native_run_never_sends_the_key_to_an_undeclared_env() -> None:
    env = _PlainFactory().make()
    result = _run_native(env, episodes=2)

    assert all(reset["options"] is None for reset in env.resets)
    assert [episode.trial for episode in result.episodes] == [0, 1]

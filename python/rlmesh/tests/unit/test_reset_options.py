"""Declared reset options: the `trial_index` ordinal and its declaration gate.

An env opts into a reserved `reset(options=)` key by naming it in
`EnvFactory.reset_options`; the stamp rides the env's metadata, and every sender
(the Rust driver, the local Session) checks the declaration before delivering.
"""

from __future__ import annotations

import warnings
from typing import Any, ClassVar, cast

import pytest
import rlmesh
from rlmesh._models._connect import declares_reset_option, reset_env


class _TinyEnv:
    """A minimal local env that records what its reset was called with."""

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
    # Pinned against `ENV_RESET_OPTIONS_KEY` in crates/rlmesh-runtime/src/spec.rs
    # (this name is re-exported from it, so the literal is the contract).
    assert rlmesh.ENV_RESET_OPTIONS_KEY == "rlmesh.env.v1.reset_options"


def test_make_stamps_the_declaration_without_mutating_the_class() -> None:
    class_metadata = dict(_TinyEnv.metadata)
    env = _TrialFactory().make()

    assert env.metadata[rlmesh.ENV_RESET_OPTIONS_KEY] == ["trial_index"]
    # Copy-and-assign: the shared class attribute is untouched, so a second env
    # of the same class does not inherit another factory's stamp.
    assert _TinyEnv.metadata == class_metadata
    assert rlmesh.ENV_RESET_OPTIONS_KEY not in _TinyEnv.metadata
    # The render modes the env already published survive the merge.
    assert env.metadata["render_modes"] == []


def test_make_stamps_nothing_when_no_option_is_declared() -> None:
    env = _PlainFactory().make()
    assert rlmesh.ENV_RESET_OPTIONS_KEY not in dict(env.metadata)


def test_declares_reset_option_reads_the_stamped_list() -> None:
    class _Contract:
        def __init__(self, metadata: Any) -> None:
            self.metadata = metadata

    assert declares_reset_option(
        _Contract({rlmesh.ENV_RESET_OPTIONS_KEY: ["trial_index"]}), "trial_index"
    )
    assert declares_reset_option(
        _Contract({rlmesh.ENV_RESET_OPTIONS_KEY: "trial_index"}), "trial_index"
    )
    assert not declares_reset_option(
        _Contract({rlmesh.ENV_RESET_OPTIONS_KEY: ["something_else"]}), "trial_index"
    )
    assert not declares_reset_option(_Contract({}), "trial_index")
    assert not declares_reset_option(_Contract(None), "trial_index")


def test_reset_env_forwards_options_only_when_given() -> None:
    env = _TinyEnv()
    reset_env(env, 3, None)
    reset_env(env, None, {"trial_index": 7})

    assert env.resets == [
        {"seed": 3, "options": None},
        {"seed": None, "options": {"trial_index": 7}},
    ]


def test_session_reset_sends_the_ordinal_to_a_declaring_env() -> None:
    env = _TrialFactory().make()
    with rlmesh.session(rlmesh.Model(lambda obs: 0), env) as sess:
        sess.reset(seed=1, trial_index=4)
    assert env.resets[-1]["options"] == {"trial_index": 4}


def test_session_reset_warns_and_drops_the_ordinal_for_an_undeclared_env() -> None:
    env = _PlainFactory().make()
    with rlmesh.session(rlmesh.Model(lambda obs: 0), env) as sess:
        with pytest.warns(UserWarning, match="declares no"):
            sess.reset(trial_index=4)
    assert env.resets[-1]["options"] is None


def test_session_run_walks_trial_ordinals_by_default() -> None:
    # The ordinal is on by default (base 0): a declaring env receives 0, 1, 2
    # without the caller asking for anything.
    declared = _TrialFactory().make()
    with rlmesh.session(rlmesh.Model(lambda obs: 0), declared) as sess:
        result = sess.run(max_episodes=3)
    assert [episode.trial for episode in result.episodes] == [0, 1, 2]
    assert [reset["options"] for reset in declared.resets] == [
        {"trial_index": 0},
        {"trial_index": 1},
        {"trial_index": 2},
    ]


def test_session_run_never_sends_the_key_to_an_undeclared_env() -> None:
    # Delivery is what the declaration gates; the result still records the
    # ordinal each episode walked, as the native loop's report does.
    plain = _PlainFactory().make()
    with rlmesh.session(rlmesh.Model(lambda obs: 0), plain) as sess:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = sess.run(max_episodes=2)
    assert [episode.trial for episode in result.episodes] == [0, 1]
    assert all(reset["options"] is None for reset in plain.resets)


def test_session_run_walks_trial_ordinals_from_the_base() -> None:
    declared = _TrialFactory().make()
    with rlmesh.session(rlmesh.Model(lambda obs: 0), declared) as sess:
        result = sess.run(max_episodes=2, trial_index_base=10)
        with pytest.raises(ValueError, match="trial_index_base"):
            sess.run(max_episodes=1, trial_index_base=-1)
    assert [episode.trial for episode in result.episodes] == [10, 11]
    assert [reset["options"] for reset in declared.resets] == [
        {"trial_index": 10},
        {"trial_index": 11},
    ]


def test_trial_index_helper_reads_the_reserved_key() -> None:
    assert rlmesh.trial_index({"trial_index": 12}) == 12
    assert rlmesh.trial_index({"trial_index": 12.0}) == 12
    assert rlmesh.trial_index({"other": 1}) is None
    assert rlmesh.trial_index({"trial_index": True}) is None
    assert rlmesh.trial_index(None) is None


class _NoOptionsEnv:
    """A served env whose `reset` takes no `options` -- the backstop's subject."""

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
        server = rlmesh.EnvServer(
            cast("Any", _NoOptionsEnv()), host="127.0.0.1", port=0
        )
    except ConnectionError as exc:
        if "Operation not permitted" in str(exc):
            pytest.skip("local tcp bind is not permitted in this environment")
        raise
    server.start()
    try:
        handle = RemoteEnv(server.address)
        # A hand-driven client forwards whatever options it is given; the server
        # drops them (warning once) rather than raising TypeError mid-run.
        obs, _info = handle.reset(options={"trial_index": 3})
        assert obs is not None
        handle.close()
    finally:
        server.shutdown()


def test_native_run_delivers_the_ordinal_over_the_wire() -> None:
    """The full path: RuntimeSessionSpec.trial_index_base -> ResetRequest.options
    -> the served env's reset -> EpisodeResult.trial."""
    from rlmesh.numpy import Model

    env = _TrialFactory().make()
    try:
        server = rlmesh.EnvServer(env, host="127.0.0.1", port=0)
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
    """`Model.run` on the native loop, serving `env` on a loopback port."""
    from rlmesh.numpy import Model

    try:
        return Model(lambda obs: 0).run(env, **kwargs)
    except ConnectionError as exc:
        if "Operation not permitted" in str(exc):
            pytest.skip("local tcp bind is not permitted in this environment")
        raise


def test_native_run_delivers_the_ordinal_by_default() -> None:
    env = _TrialFactory().make()
    result = _run_native(env, max_episodes=3)

    assert [reset["options"] for reset in env.resets] == [
        {"trial_index": 0},
        {"trial_index": 1},
        {"trial_index": 2},
    ]
    assert [episode.trial for episode in result.episodes] == [0, 1, 2]


def test_native_run_walks_the_ordinals_from_the_base() -> None:
    env = _TrialFactory().make()
    result = _run_native(env, max_episodes=2, trial_index_base=5)

    assert [reset["options"] for reset in env.resets] == [
        {"trial_index": 5},
        {"trial_index": 6},
    ]
    assert [episode.trial for episode in result.episodes] == [5, 6]

    with pytest.raises(ValueError, match="trial_index_base"):
        _run_native(env, max_episodes=1, trial_index_base=-1)


def test_native_run_reports_zero_based_episode_indices() -> None:
    # The runtime mints 1-based slot ordinals; EpisodeResult.index is 0-based on
    # every path (Session.run and the hooks included), so `trial` stays
    # `trial_index_base + index`.
    env = _TrialFactory().make()
    result = _run_native(env, seeds=range(3), trial_index_base=7)

    assert [episode.index for episode in result.episodes] == [0, 1, 2]
    assert [episode.trial for episode in result.episodes] == [
        7 + episode.index for episode in result.episodes
    ]


def test_native_run_never_sends_the_key_to_an_undeclared_env() -> None:
    env = _PlainFactory().make()
    result = _run_native(env, max_episodes=2)

    assert all(reset["options"] is None for reset in env.resets)
    # Minted and reported either way, so the sweep can be read off the result.
    assert [episode.trial for episode in result.episodes] == [0, 1]

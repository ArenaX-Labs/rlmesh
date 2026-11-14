"""Drive a C#-served env through the supported Python surface."""

import builtins
import sys

import rlmesh
import rlmesh.numpy

address, scenario, *expected_text = sys.argv[1:]


def remote_env() -> rlmesh.numpy.RemoteEnv:
    return rlmesh.numpy.RemoteEnv(address)


def random_episodes(count: int) -> None:
    """Random episodes, each to its end."""
    with remote_env() as env:
        for _ in range(count):
            env.reset()
            done = False
            while not done:
                _, _, terminated, truncated, _ = env.step(env.action_space.sample())
                done = terminated or truncated


if scenario == "normal":
    random_episodes(3)
elif scenario == "dict":
    import numpy as np

    with remote_env() as env:
        observation, info = env.reset()
        assert info["z"] == 9 and info["a"] == 8, info
        step = env.step(0)
        assert step[4]["z"] == 9 and step[4]["a"] == 8, step[4]
        for value in (observation, step[0]):
            assert list(value) == ["image", "state"], list(value)
            np.testing.assert_array_equal(
                value["image"], np.array([1, 2, 3, 4], dtype=np.uint8).reshape(2, 2, 1)
            )
            np.testing.assert_array_equal(
                value["state"], np.array([5, 6], dtype=np.uint8)
            )
    random_episodes(2)
else:
    with remote_env() as env:
        env.reset()
        try:
            env.step(0)
        except rlmesh.EnvironmentException as error:
            assert error.code == "INTERNAL", error.code
            assert error.is_recoverable == (scenario == "recoverable")
            assert isinstance(error, rlmesh.RecoverableEnvironmentException) == (
                scenario == "recoverable"
            )
            expected = expected_text[0]
            builtin = getattr(builtins, scenario, None)
            if builtin is not None:
                # The env's builtin class arrives with its own message.
                assert isinstance(error, builtin), type(error).__mro__
                assert str(error) == expected and error.phase == "step", str(error)
            else:
                assert str(error) == f"[INTERNAL] step failed: {expected}", str(error)
        else:
            raise AssertionError("expected an environment failure")

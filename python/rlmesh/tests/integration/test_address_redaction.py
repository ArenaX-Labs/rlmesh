"""A session's key stays in its address: no repr or refusal shows it."""

from __future__ import annotations

from collections.abc import Callable

import gymnasium as gym
import pytest
import rlmesh
import rlmesh.numpy
import rlmesh.realtime
from rlmesh._sandbox.session import SandboxInfo
from rlmesh.launch import launch_identity


class _Env:
    observation_space = gym.spaces.Discrete(4)
    action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        return 0, {}

    def step(self, action):
        return 1, 0.0, False, False, {}

    def close(self):
        pass


def _keyless(key: str, shown: dict[str, str]) -> None:
    for name, text in shown.items():
        assert key not in text, f"{name} shows the key: {text.replace(key, '<KEY>')}"
        assert "#…" in text, f"{name}: {text.replace(key, '<KEY>')}"


def test_reprs_leave_the_key_out() -> None:
    server = rlmesh.EnvServer(_Env(), ":0/env")
    server.start()
    try:
        address = server.address
        key = address.partition("#")[2].partition("/")[0]
        assert key
        session = address.rpartition("/")[0]
        info = SandboxInfo("gym:CartPole-v1", "gym:CartPole-v1", address, "c")
        policy = rlmesh.realtime.Policy({"hz": 10}, session=session, name="p")
        shown = {
            "EnvServer": repr(server),
            "SandboxInfo": repr(info),
            "Policy": repr(policy),
        }
        # One driver holds the env at a time.
        with rlmesh.RemoteEnv(address) as env:
            shown["RemoteEnv"] = repr(env)
        with rlmesh.numpy.RemoteEnv(address) as numpy_env:
            shown["numpy RemoteEnv"] = repr(numpy_env)
        _keyless(key, shown)
    finally:
        server.shutdown()


def test_a_refused_keyed_address_is_not_repeated() -> None:
    _, key = launch_identity()
    calls: dict[str, Callable[[], object]] = {
        "Model(v1)": lambda: rlmesh.Model(f"tcp://10.0.0.5:1#{key}/m"),
        "Model(no model)": lambda: rlmesh.Model(f"10.0.0.5:1#{key}"),
        "Model(served)(obs)": lambda: rlmesh.Model(f"10.0.0.5:1#{key}/m")(0),
        "Model.serve without a name": lambda: rlmesh.Model(lambda obs: 0).serve(
            f"10.0.0.5:1#{key}"
        ),
        "RemoteEnv(v1)": lambda: rlmesh.RemoteEnv(f"tcp://10.0.0.5:1#{key}/env"),
        "EnvServer(v1)": lambda: rlmesh.EnvServer(
            _Env(), f"tcp://10.0.0.5:1#{key}/env"
        ),
    }
    for name, call in calls.items():
        with pytest.raises((ValueError, TypeError)) as raised:
            call()
        text = str(raised.value)
        assert key not in text, f"{name}: {text.replace(key, '<KEY>')}"

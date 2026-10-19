"""Drive a C#-served env through the supported Python surface."""

import importlib.util
import sys
from pathlib import Path

import grpc
import rlmesh
import rlmesh.numpy

address, scenario, *expected_text = sys.argv[1:]
if scenario == "normal":
    # Reuse the repository's raw protobuf probe codec; no generated stubs or
    # dependency on Python's private env-client internals is needed.
    spec = importlib.util.spec_from_file_location(
        "crossver", Path(__file__).resolve().parents[2] / "tests/system/crossver.py"
    )
    assert spec is not None and spec.loader is not None
    codec = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = codec
    spec.loader.exec_module(codec)
    info = rlmesh.build_info()
    core = codec._text(1, info.protocol_generation)
    for edition in info.supported_workflow_editions:
        core += codec._text(4, edition)
    with grpc.insecure_channel(address.removeprefix("tcp://")) as channel:
        raw = channel.unary_unary("/rlmesh.env.v1.EnvService/Handshake")(
            codec._delimited(1, core), timeout=10
        )
    base = codec._message(codec._decode(raw), 1)
    peer = codec._message(base, 2)
    assert bool(base.get(1, [0])[0])
    # ABI 5's host_runtime_json changes the served describe envelope only.
    # PeerInfo still belongs to the native process; E-3's assertion that this
    # option changes handshake language is not supported by the real ABI.
    assert codec._string(peer, 3), "missing native PeerInfo language"

    result = rlmesh.run(rlmesh.RANDOM_SAMPLE, address, episodes=3)
    assert result is not None
elif scenario == "dict":
    import numpy as np

    with rlmesh.numpy.RemoteEnv(address) as env:
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
    result = rlmesh.run(rlmesh.RANDOM_SAMPLE, address, episodes=2)
    assert result is not None
else:
    with rlmesh.numpy.RemoteEnv(address) as env:
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
            assert str(error) == f"[INTERNAL] step failed: {expected}", str(error)
        else:
            raise AssertionError("expected an environment failure")

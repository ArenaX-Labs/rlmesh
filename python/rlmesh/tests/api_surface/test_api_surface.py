from __future__ import annotations


def test_root_namespace_is_small() -> None:
    import rlmesh

    assert rlmesh.__all__ == [
        "NO_ADAPTER",
        "RANDOM_SAMPLE",
        "EnvFactory",
        "EnvServer",
        "EpisodeResult",
        "Model",
        "Param",
        "ParamSpec",
        "RemoteEnv",
        "RemoteModel",
        "RemoteVectorEnv",
        "RunResult",
        "SandboxEnv",
        "SandboxModel",
        "SandboxOptions",
        "SandboxVectorEnv",
        "ServeOptions",
        "Session",
        "Tensor",
        "__version__",
        "adapters",
        "run",
        "session",
        "spaces",
        "types",
    ]

    # EnvFactory is the thin runtime env-authoring base; models are authored by
    # subclassing rlmesh.Model. The removed build DSL stays gone, and the internal
    # modules are _-prefixed (server/serving/client/sandbox/models): none of those are
    # part of the public top-level surface.
    for name in (
        "Recipe",
        "register",
        "serving",
        "server",
        "client",
        "sandbox",
        "models",
        "Box",
        "Discrete",
        "Dict",
        "Space",
        "SpaceSpec",
        "EnvContract",
        "RemoteShutdown",
        "Value",
        "PrimitiveValue",
        "EnvLike",
        "VectorEnvLike",
        "SpaceLike",
    ):
        assert not hasattr(rlmesh, name)


def test_spaces_namespace_contains_space_family() -> None:
    from rlmesh import spaces

    assert spaces.__all__ == [
        "Box",
        "Dict",
        "Discrete",
        "MultiBinary",
        "MultiDiscrete",
        "Space",
        "SpaceSpec",
        "Text",
        "Tuple",
        "from_gymnasium_space",
        "space_from_spec",
        "to_gymnasium_space",
    ]


def test_types_namespace_contains_typing_contracts_only() -> None:
    from rlmesh import types

    assert types.__all__ == [
        "EnvLike",
        "InfoDict",
        "Metadata",
        "PrimitiveValue",
        "SpaceLike",
        "Value",
        "VectorEnvLike",
    ]
    assert not hasattr(types, "Tensor")


def test_backend_namespaces_do_not_export_adapters() -> None:
    import rlmesh.numpy as rlmesh_numpy
    import rlmesh.torch as rlmesh_torch

    assert rlmesh_numpy.__all__ == [
        "EnvFactory",
        "Model",
        "NumpyValue",
        "RemoteEnv",
        "RemoteModel",
        "RemoteVectorEnv",
        "SandboxEnv",
        "SandboxInfo",
        "SandboxModel",
        "SandboxOptions",
        "SandboxVectorEnv",
        "asarray",
        "ensure_available",
        "from_array",
        "space_from_spec",
    ]
    assert "NumpyAdapter" not in rlmesh_numpy.__all__
    assert not hasattr(rlmesh_numpy, "NumpyAdapter")

    assert rlmesh_torch.__all__ == [
        "EnvFactory",
        "Model",
        "RemoteEnv",
        "RemoteModel",
        "RemoteVectorEnv",
        "SandboxEnv",
        "SandboxInfo",
        "SandboxModel",
        "SandboxOptions",
        "SandboxVectorEnv",
        "TorchValue",
        "as_tensor",
        "ensure_available",
        "from_tensor",
        "space_from_spec",
    ]
    assert "TorchAdapter" not in rlmesh_torch.__all__
    assert not hasattr(rlmesh_torch, "TorchAdapter")

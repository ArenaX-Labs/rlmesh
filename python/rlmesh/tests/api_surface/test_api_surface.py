from __future__ import annotations


def test_root_namespace_is_small() -> None:
    import rlmesh

    assert rlmesh.__all__ == [
        "EnvRecipe",
        "EnvServer",
        "Model",
        "ModelRecipe",
        "ModelServer",
        "Recipe",
        "RemoteEnv",
        "RemoteVectorEnv",
        "ServeOptions",
        "Tensor",
        "__version__",
        "adapters",
        "make",
        "models",
        "recipes",
        "register",
        "serving",
        "spaces",
        "types",
    ]

    for name in (
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
        "SandboxEnv",
        "SandboxVectorEnv",
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
        "SpaceBridge",
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
        "Model",
        "NumpyValue",
        "RemoteEnv",
        "RemoteVectorEnv",
        "SandboxEnv",
        "SandboxInfo",
        "SandboxModel",
        "SandboxVectorEnv",
        "asarray",
        "from_array",
        "space_from_spec",
    ]
    assert "NumpyAdapter" not in rlmesh_numpy.__all__
    assert not hasattr(rlmesh_numpy, "NumpyAdapter")

    assert rlmesh_torch.__all__ == [
        "Model",
        "RemoteEnv",
        "RemoteVectorEnv",
        "SandboxEnv",
        "SandboxInfo",
        "SandboxVectorEnv",
        "TorchValue",
        "as_tensor",
        "from_tensor",
        "space_from_spec",
    ]
    assert "TorchAdapter" not in rlmesh_torch.__all__
    assert not hasattr(rlmesh_torch, "TorchAdapter")

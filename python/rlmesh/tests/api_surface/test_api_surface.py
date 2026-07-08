from __future__ import annotations


def test_root_namespace_is_small() -> None:
    import rlmesh

    assert rlmesh.__all__ == [
        "DESCRIBE_METADATA_KEY",
        "DESCRIBE_SCHEMA_VERSION",
        "NO_ADAPTER",
        "RANDOM_SAMPLE",
        "EnvFactory",
        "EnvServer",
        "EpisodeResult",
        "Model",
        "Param",
        "ParamSpec",
        "Reader",
        "Recorder",
        "RemoteEnv",
        "RemoteModel",
        "RemoteVectorEnv",
        "RunHooks",
        "RunResult",
        "SandboxBuild",
        "SandboxEnv",
        "SandboxModel",
        "SandboxRuntime",
        "SandboxVectorEnv",
        "ServeOptions",
        "Session",
        "StepEvent",
        "Tensor",
        "Variant",
        "Vector",
        "View",
        "__build__",
        "__version__",
        "adapters",
        "describe",
        "describe_json",
        "params",
        "run",
        "sanitize_metadata",
        "session",
        "spaces",
        "specs",
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
        "EnvTarget",
        "HasAddress",
        "HasEnvContract",
        "InfoDict",
        "LocalEnvTarget",
        "Metadata",
        "PrimitiveValue",
        "SpaceLike",
        "SpecArg",
        "SupportsMake",
        "SupportsResetStep",
        "Value",
        "VectorEnvLike",
        "ViewArg",
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
        "SandboxBuild",
        "SandboxEnv",
        "SandboxInfo",
        "SandboxModel",
        "SandboxRuntime",
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
        "SandboxBuild",
        "SandboxEnv",
        "SandboxInfo",
        "SandboxModel",
        "SandboxRuntime",
        "SandboxVectorEnv",
        "TorchValue",
        "as_tensor",
        "ensure_available",
        "from_tensor",
        "space_from_spec",
    ]
    assert "TorchAdapter" not in rlmesh_torch.__all__
    assert not hasattr(rlmesh_torch, "TorchAdapter")


def test_token_auth_surface_stays_removed() -> None:
    """The token-auth surface was deliberately stripped from the Python SDK
    (the Rust core keeps its internals). Beyond snapshot omission, pin it
    explicitly: the connecting classes accept no credential parameter, and no
    snapshotted public module exports an auth-named symbol.
    """
    import importlib
    import inspect
    import json
    from pathlib import Path

    import rlmesh

    for cls in (rlmesh.RemoteEnv, rlmesh.RemoteModel, rlmesh.RemoteVectorEnv):
        parameters = inspect.signature(cls.__init__).parameters
        for name in ("token", "api_key", "bearer", "auth"):
            assert name not in parameters, (cls.__name__, name)

    snapshot = Path(__file__).parent / "snapshots" / "api_surface.json"
    for module_name in json.loads(snapshot.read_text(encoding="utf-8")):
        module = importlib.import_module(module_name)
        for export in module.__all__:
            lowered = export.lower()
            for needle in ("token", "auth", "credential"):
                assert needle not in lowered, (module_name, export)

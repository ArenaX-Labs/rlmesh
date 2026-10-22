from __future__ import annotations


def test_root_namespace_is_small() -> None:
    import rlmesh

    assert rlmesh.__all__ == [
        "DESCRIBE_METADATA_KEY",
        "DESCRIBE_SCHEMA_VERSION",
        "ENV_RESET_OPTIONS_KEY",
        "NO_ADAPTER",
        "RANDOM_SAMPLE",
        "Binding",
        "BuildInfo",
        "EnvFactory",
        "EnvServer",
        "EnvironmentException",
        "EpisodeResult",
        "FrameSources",
        "Model",
        "ModelException",
        "Param",
        "ParamSpec",
        "ProtocolException",
        "RLMeshException",
        "Reader",
        "Recorder",
        "RecoverableEnvironmentException",
        "RemoteEnv",
        "RemoteModel",
        "RemoteVectorEnv",
        "RunContext",
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
        "TelemetryRow",
        "Tensor",
        "Variant",
        "Vector",
        "View",
        "WorkflowEditionWarning",
        "__build__",
        "__version__",
        "adapters",
        "build_info",
        "current_workflow_edition",
        "describe",
        "describe_json",
        "params",
        "predict_seed",
        "run",
        "sanitize_metadata",
        "session",
        "spaces",
        "specs",
        "trial_index",
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


def test_lazy_exports_are_discoverable() -> None:
    """describe/describe_json load lazily, so dir() must still complete them."""
    import rlmesh

    listing = dir(rlmesh)
    assert "describe" in listing
    assert "describe_json" in listing


def test_exception_family_is_public() -> None:
    """The exception family is importable from rlmesh, not just rlmesh._rlmesh."""
    import rlmesh
    from rlmesh import _rlmesh

    assert rlmesh.RLMeshException is _rlmesh.RLMeshException
    assert issubclass(rlmesh.EnvironmentException, rlmesh.RLMeshException)
    assert issubclass(rlmesh.ProtocolException, rlmesh.RLMeshException)
    assert issubclass(
        rlmesh.RecoverableEnvironmentException, rlmesh.EnvironmentException
    )
    assert issubclass(rlmesh.ModelException, rlmesh.RLMeshException)
    assert rlmesh.ModelException is _rlmesh.ModelException


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
        "BatchPredictContext",
        "EnvLike",
        "EnvTarget",
        "HasAddress",
        "HasEnvContract",
        "InfoDict",
        "LocalEnvTarget",
        "Metadata",
        "PredictContext",
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


def test_api_key_never_appears_in_repr_or_errors() -> None:
    """The remote clients take a shared ``api_key`` (RMT-22): it must never
    reach a repr, an attribute dump or error text, and no snapshotted public
    module exports a token- or credential-named symbol.
    """
    import importlib
    import inspect
    import json
    from pathlib import Path

    import pytest
    import rlmesh

    for cls in (rlmesh.RemoteEnv, rlmesh.RemoteVectorEnv):
        parameters = inspect.signature(cls.__init__).parameters
        assert parameters["api_key"].default is None, cls.__name__
        assert parameters["tls_ca_file"].default is None, cls.__name__
        for name in ("token", "bearer", "auth"):
            assert name not in parameters, (cls.__name__, name)

    for bad in ("hidden-key\n", " hidden-key", "hidden\u00e9key"):
        with pytest.raises(ValueError) as raised:
            rlmesh.RemoteEnv("https://localhost:1", api_key=bad)
        assert "hidden" not in str(raised.value), str(raised.value)

    snapshot = Path(__file__).parent / "snapshots" / "api_surface.json"
    for module_name in json.loads(snapshot.read_text(encoding="utf-8")):
        module = importlib.import_module(module_name)
        for export in module.__all__:
            lowered = export.lower()
            for needle in ("token", "auth", "credential"):
                assert needle not in lowered, (module_name, export)

from __future__ import annotations


def test_native_module_exports_only_reachable_classes() -> None:
    """The private _native module exports only reachable classes."""
    import rlmesh._native as native

    assert native.__all__ == [
        "Model",
        "RemoteEnv",
        "RemoteVectorEnv",
        "SandboxEnv",
        "SandboxModel",
        "SandboxVectorEnv",
    ]

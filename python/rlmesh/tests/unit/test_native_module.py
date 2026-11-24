from __future__ import annotations


def test_native_module_exports_only_reachable_classes() -> None:
    import rlmesh._native as native

    assert native.__all__ == [
        "Model",
        "RemoteEnv",
        "RemoteModel",
        "RemoteVectorEnv",
        "SandboxEnv",
        "SandboxModel",
        "SandboxVectorEnv",
    ]

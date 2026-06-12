from __future__ import annotations


def test_native_module_exports_only_reachable_classes() -> None:
    """The private _native module must not carry unreachable public-looking API.

    SandboxEnv/SandboxVectorEnv were defined here and listed in __all__ but
    never re-exported from any public module, so they were dead code that
    looked importable. They are removed; the dependency-free sandbox base
    classes remain available via rlmesh.sandbox and the numpy/torch backends.
    """
    import rlmesh._native as native

    assert native.__all__ == ["Model", "RemoteEnv", "RemoteVectorEnv"]
    assert not hasattr(native, "SandboxEnv")
    assert not hasattr(native, "SandboxVectorEnv")

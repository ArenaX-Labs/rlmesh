from __future__ import annotations

import importlib.util
from pathlib import Path


def _types_module_path() -> Path:
    return Path(__file__).resolve().parents[2] / "src" / "rlmesh" / "types.py"


def test_types_module_imports_without_compiled_extension() -> None:
    """rlmesh.types imports without the compiled extension."""
    spec = importlib.util.spec_from_file_location(
        "rlmesh_types_standalone_under_test", _types_module_path()
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.InfoDict == dict[str, object]
    assert hasattr(module, "Value")
    assert hasattr(module, "EnvLike")
    assert hasattr(module, "VectorEnvLike")
    assert hasattr(module, "SpaceLike")

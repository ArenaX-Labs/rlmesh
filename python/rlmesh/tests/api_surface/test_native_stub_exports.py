from __future__ import annotations

import ast
import importlib
from pathlib import Path

EXPECTED_NATIVE_EXPORTS = [
    "EnvContract",
    "EnvironmentException",
    "ProtocolException",
    "PyEnvClient",
    "PyEnvServer",
    "PyModel",
    "PyVectorEnvClient",
    "PyVectorEnvServer",
    "RLMeshException",
    "ServeOptions",
    "Space",
    "SpaceSpec",
    "Tensor",
    "box_space_spec",
    "dict_space_spec",
    "discrete_space_spec",
    "multi_binary_space_spec",
    "multi_discrete_space_spec",
    "run_cli",
    "sandbox_start_env",
    "sandbox_stop_env",
    "space_spec_from_gym_space",
    "text_space_spec",
    "tuple_space_spec",
]


def test_native_runtime_exports_are_represented_in_stub() -> None:
    native = importlib.import_module("rlmesh._rlmesh")
    stub = _parse_stub_exports()

    for name in EXPECTED_NATIVE_EXPORTS:
        assert hasattr(native, name), name
        assert name in stub.names, name
        assert name in stub.all_exports, name

    assert "RemoteShutdown" not in dir(native)
    assert "RemoteShutdown" not in stub.names
    assert "RemoteShutdown" not in stub.all_exports


def test_native_exception_hierarchy() -> None:
    native = importlib.import_module("rlmesh._rlmesh")

    assert issubclass(native.RLMeshException, RuntimeError)
    assert issubclass(native.ProtocolException, native.RLMeshException)
    assert issubclass(native.EnvironmentException, native.RLMeshException)


class StubExports:
    def __init__(self, names: set[str], all_exports: set[str]) -> None:
        self.names = names
        self.all_exports = all_exports


def _parse_stub_exports() -> StubExports:
    stub_path = Path(__file__).resolve().parents[2] / "src" / "rlmesh" / "_rlmesh.pyi"
    tree = ast.parse(stub_path.read_text(encoding="utf-8"), filename=str(stub_path))
    names: set[str] = set()
    all_exports: set[str] = set()

    for node in tree.body:
        if isinstance(node, ast.ClassDef | ast.FunctionDef):
            names.add(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
                    if target.id == "__all__":
                        all_exports = set(ast.literal_eval(node.value))

    return StubExports(names=names, all_exports=all_exports)

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, cast

import pytest

if TYPE_CHECKING:
    import torch

    TorchTensor = torch.Tensor


def test_torch_tensor_roundtrip_and_rejects_unknown_dtype() -> None:
    torch = pytest.importorskip("torch")
    from rlmesh import Tensor
    from rlmesh import torch as rlmesh_torch

    source = torch.tensor([[1.5, 2.5]], dtype=torch.float16)
    tensor = rlmesh_torch.from_tensor(source)

    assert isinstance(tensor, Tensor)
    assert tensor.dtype == "float16"

    copied = rlmesh_torch.as_tensor(tensor, copy=True)
    torch.testing.assert_close(copied, source)
    copied[0, 0] = 9.0
    torch.testing.assert_close(
        _torch_as_tensor_without_readonly_warning(rlmesh_torch, tensor),
        source,
    )

    restored = _torch_as_tensor_without_readonly_warning(rlmesh_torch, tensor)
    torch.testing.assert_close(restored, source)
    restored[0, 0] = 9.0
    expected = source.clone()
    expected[0, 0] = 9.0
    torch.testing.assert_close(
        _torch_as_tensor_without_readonly_warning(rlmesh_torch, tensor),
        expected,
    )

    with pytest.raises(ValueError, match="unsupported tensor dtype"):
        rlmesh_torch._torch_dtype("complex64")


def test_torch_space_from_spec_samples_and_contains_torch_values() -> None:
    torch = pytest.importorskip("torch")
    from rlmesh import spaces
    from rlmesh import torch as rlmesh_torch

    space = rlmesh_torch.space_from_spec(
        spaces.Box(-1.0, 1.0, shape=[2], dtype="float32").spec
    )

    sample = space.sample()

    assert isinstance(sample, torch.Tensor)
    sample_tensor = cast("TorchTensor", sample)
    assert tuple(sample_tensor.shape) == (2,)
    assert sample_tensor.dtype == torch.float32
    assert space.contains(sample_tensor)
    assert space.contains(torch.tensor([0.25, -0.25], dtype=torch.float32))


def _torch_as_tensor_without_readonly_warning(rlmesh_torch: Any, tensor: object) -> Any:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The given buffer is not writable.*",
            category=UserWarning,
        )
        return rlmesh_torch.as_tensor(tensor)


@pytest.mark.parametrize(
    ("dtype_name", "values"),
    [
        ("bool", [True, False, True]),
        ("uint8", [0, 127, 255]),
        ("int8", [-128, 0, 127]),
        ("int16", [-5, 0, 999]),
        ("int32", [-5, 0, 70_000]),
        ("int64", [-5, 0, 2**40]),
        ("uint16", [0, 1, 65_535]),
        ("uint32", [0, 1, 70_000]),
        ("uint64", [0, 1, 2**40]),
        ("float16", [1.5, -2.0, 0.25]),
        ("bfloat16", [1.0, -2.0, 0.5]),
        ("float32", [1.5, -2.0, 0.25]),
        ("float64", [1.5, -2.0, 0.25]),
    ],
)
def test_torch_tensor_roundtrip_all_dtypes(
    dtype_name: str, values: list[object]
) -> None:
    torch = pytest.importorskip("torch")
    from rlmesh import Tensor
    from rlmesh import torch as rlmesh_torch

    torch_dtype = getattr(torch, dtype_name, None)
    if torch_dtype is None:
        pytest.skip(f"torch {torch.__version__} lacks {dtype_name}")

    source = torch.tensor(values, dtype=torch_dtype)
    tensor = rlmesh_torch.from_tensor(source)

    assert isinstance(tensor, Tensor)
    assert tensor.dtype == dtype_name
    assert tensor.shape == [len(values)]

    restored = rlmesh_torch.as_tensor(tensor)
    assert restored.dtype == torch_dtype
    assert torch.equal(restored, source)


def test_torch_as_tensor_shares_memory_via_dlpack() -> None:
    pytest.importorskip("torch")
    from rlmesh import Tensor
    from rlmesh import torch as rlmesh_torch

    tensor = Tensor(bytes(8), [2], "float32")
    view = rlmesh_torch.as_tensor(tensor)
    assert view.data_ptr() == rlmesh_torch.as_tensor(tensor).data_ptr()

    copied = rlmesh_torch.as_tensor(tensor, copy=True)
    assert copied.data_ptr() != view.data_ptr()


def test_torch_bridge_decode_is_owned_writable_copy() -> None:
    """predict() receives an owned, writable tensor: an in-place op (the common
    VLA normalize idiom) must not write through into the shared wire buffer."""
    pytest.importorskip("torch")
    import torch

    from rlmesh import Tensor
    from rlmesh.torch import _torch_bridge

    tensor = Tensor(bytes(8), [2], "float32")
    decoded = _torch_bridge.decode(tensor)
    assert isinstance(decoded, torch.Tensor)
    decoded.add_(1.0)  # in-place; must not reach the wire bytes
    assert tensor.tobytes() == bytes(8)


def test_torch_export_works_without_numpy() -> None:
    """from_tensor must not require numpy (regression for the .numpy() path)."""
    import subprocess
    import sys
    import textwrap

    pytest.importorskip("torch")

    script = textwrap.dedent(
        """
        import importlib.abc, sys

        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name == "numpy" or name.startswith("numpy."):
                    raise ModuleNotFoundError(name + " is blocked")
                return None

        sys.modules.pop("numpy", None)
        sys.meta_path.insert(0, Block())

        import torch
        from rlmesh import torch as rlmesh_torch

        source = torch.tensor([[1.5, -2.5]], dtype=torch.float32)
        encoded = rlmesh_torch.from_tensor(source)
        assert encoded.dtype == "float32", encoded.dtype
        restored = rlmesh_torch.as_tensor(encoded)
        assert restored.tolist() == [[1.5, -2.5]], restored
        assert rlmesh_torch.from_tensor(torch.tensor(7)) == 7
        bools = rlmesh_torch.from_tensor(torch.tensor([True, False]))
        assert bools.tobytes() == b"\\x01\\x00", bools.tobytes()
        print("NO-NUMPY-OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "NO-NUMPY-OK" in result.stdout

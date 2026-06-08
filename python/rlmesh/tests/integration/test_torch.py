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
        rlmesh_torch._torch_dtype("int16")


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

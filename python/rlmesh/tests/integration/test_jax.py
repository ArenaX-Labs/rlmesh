from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

jax = pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402

# int64/uint64/float64 round-trips require 64-bit mode; set it before any op.
jax.config.update("jax_enable_x64", True)

if TYPE_CHECKING:
    JaxArray = jax.Array


def test_jax_tensor_roundtrip() -> None:
    from rlmesh import Tensor
    from rlmesh import jax as rlmesh_jax

    source = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
    tensor = rlmesh_jax.from_array(source)

    assert isinstance(tensor, Tensor)
    assert tensor.shape == [2, 3]
    assert tensor.dtype == "float32"

    restored = rlmesh_jax.asarray(tensor)
    assert restored.dtype == source.dtype
    assert jnp.array_equal(restored, source)


@pytest.mark.parametrize(
    ("dtype", "values"),
    [
        ("bool", [True, False, True]),
        ("uint8", [0, 127, 255]),
        ("int8", [-128, 0, 127]),
        ("int16", [-5, 0, 999]),
        ("uint16", [0, 1, 65_535]),
        ("int32", [-5, 0, 70_000]),
        ("uint32", [0, 1, 70_000]),
        ("int64", [-5, 0, 2**40]),
        ("uint64", [0, 1, 2**40]),
        ("float16", [1.5, -2.0, 0.25]),
        ("bfloat16", [1.0, -2.0, 0.5]),
        ("float32", [1.5, -2.0, 0.25]),
        ("float64", [1.5, -2.0, 0.25]),
    ],
)
def test_jax_tensor_roundtrip_all_dtypes(dtype: str, values: list[object]) -> None:
    from rlmesh import Tensor
    from rlmesh import jax as rlmesh_jax

    source = jnp.asarray(values, dtype=dtype)
    tensor = rlmesh_jax.from_array(source)

    assert isinstance(tensor, Tensor)
    assert tensor.dtype == dtype
    assert tensor.shape == [len(values)]

    restored = rlmesh_jax.asarray(tensor)
    assert restored.dtype == source.dtype
    assert jnp.array_equal(restored, source)


def test_jax_scalar_from_array_returns_primitive() -> None:
    from rlmesh import jax as rlmesh_jax

    assert rlmesh_jax.from_array(jnp.asarray(3, dtype=jnp.int32)) == 3
    assert rlmesh_jax.from_array(jnp.asarray(2.5, dtype=jnp.float32)) == 2.5


def test_jax_from_array_rejects_non_jax_values() -> None:
    np = pytest.importorskip("numpy")
    from rlmesh import jax as rlmesh_jax

    with pytest.raises(TypeError, match=r"jax\.Array"):
        rlmesh_jax.from_array([1, 2, 3])
    with pytest.raises(TypeError, match=r"jax\.Array"):
        rlmesh_jax.from_array(np.zeros(2))


def test_jax_space_from_spec_samples_and_contains_jax_values() -> None:
    from rlmesh import jax as rlmesh_jax
    from rlmesh import spaces

    space = rlmesh_jax.space_from_spec(
        spaces.Box(-1.0, 1.0, shape=[2], dtype="float32").spec
    )

    sample = space.sample()

    assert isinstance(sample, jax.Array)
    sample_array = cast("JaxArray", sample)
    assert sample_array.shape == (2,)
    assert sample_array.dtype == jnp.float32
    assert space.contains(sample_array)
    assert space.contains(jnp.asarray([0.25, -0.25], dtype=jnp.float32))


def test_jax_version_floor_is_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    from rlmesh import jax as rlmesh_jax

    monkeypatch.setattr(jax, "__version__", "0.4.18")
    with pytest.raises(ImportError, match=r"0\.4\.24"):
        rlmesh_jax.ensure_available()

    monkeypatch.setattr(jax, "__version__", "0.4.24")
    rlmesh_jax.ensure_available()

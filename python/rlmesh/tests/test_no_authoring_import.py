from __future__ import annotations

import importlib

import pytest


def test_recipes_package_is_gone() -> None:
    with pytest.raises((ImportError, ModuleNotFoundError)):
        importlib.import_module("rlmesh.recipes")


def test_runtime_spec_package_is_gone() -> None:
    with pytest.raises((ImportError, ModuleNotFoundError)):
        importlib.import_module("rlmesh._spec")


def test_kept_surface_works() -> None:
    import rlmesh

    model = rlmesh.Model(lambda obs: 0)
    assert model is not None

    assert not hasattr(rlmesh, "make")
    assert not hasattr(rlmesh, "export")
    assert not hasattr(rlmesh, "DELEGATED")
    assert not hasattr(rlmesh, "hf_load")

    import rlmesh._models

    assert rlmesh.NO_ADAPTER is rlmesh._models.NO_ADAPTER

"""Regression: the kept v0.1 surface works and recipe authoring is gone.

Recipe authoring (``rlmesh.recipes``) has been removed from the tree. The kept
core (``import rlmesh``, ``Model``, serving, sandbox) must work, and the
``rlmesh.recipes`` package must no longer be importable.
"""

from __future__ import annotations

import pytest


def test_recipes_package_is_gone() -> None:
    with pytest.raises((ImportError, ModuleNotFoundError)):
        import rlmesh.recipes  # noqa: F401


def test_kept_surface_works() -> None:
    import rlmesh

    model = rlmesh.Model(lambda obs: 0)
    assert model is not None

    assert not hasattr(rlmesh, "make")
    assert not hasattr(rlmesh, "export")

    # DELEGATED is one shared instance from the kept neutral core.
    import rlmesh._models
    import rlmesh._spec

    assert rlmesh._models.DELEGATED is rlmesh._spec.DELEGATED

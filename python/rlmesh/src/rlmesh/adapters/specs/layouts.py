"""Image axis layout vocabulary.

``IMAGE_LAYOUTS`` is defined once, in the ``rlmesh-adapters`` crate
(``ImageLayout::ALL`` in ``v1/spec/layouts.rs``); this module re-exports
it through the native bindings. ``ImageLayout`` is the Python-side typing
view of the same value set.
"""

from typing import Literal, TypeAlias

from ..._rlmesh import IMAGE_LAYOUTS

ImageLayout: TypeAlias = Literal["hwc", "chw"]

__all__ = ["IMAGE_LAYOUTS", "ImageLayout"]

"""Small kept runtime-spec core.

Kept in the published wheel: the ``DELEGATED`` self-adapting-model sentinel and
the ``ArtifactInput`` runtime weight mount, which the kept features
(``Model``/``SandboxModel``, the artifact resolver) depend on. The recipe-document
schema and authoring are not part of this build.
"""

from __future__ import annotations

from ._core import DELEGATED, ArtifactInput

__all__ = ["DELEGATED", "ArtifactInput"]

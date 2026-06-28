"""Internal adapter helper package.

Nothing here is part of the public adapters API.
"""

from __future__ import annotations


def render_placement(segments: tuple[str | int, ...]) -> str:
    """Render a tree position as the canonical native ``NodePath`` string.

    Mirrors ``rlmesh_adapters::path::NodePath`` ``Display``: dot-joined keys,
    ``[i]`` for tuple indices, and ``<root>`` for the empty path (a bare leaf).
    The single source of truth for the ``NodePath::to_string()`` format both the
    resolver (error/wire spec) and the adapter (served-route customs map key)
    agree on.
    """
    if not segments:
        return "<root>"
    out = ""
    for position, segment in enumerate(segments):
        if isinstance(segment, int):
            out += f"[{segment}]"
        else:
            out += ("." if position > 0 else "") + segment
    return out

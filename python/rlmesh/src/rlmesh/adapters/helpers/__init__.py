"""Internal adapter helper package.

Nothing here is part of the public adapters API.
"""

from __future__ import annotations


def render_placement(segments: tuple[str | int, ...]) -> str:
    """Render a tree position as the canonical native ``NodePath`` string.

    Mirrors ``rlmesh_adapters::path::NodePath`` ``Display``: dot-joined keys,
    ``[i]`` for tuple indices, and ``<root>`` for the empty path (a bare leaf).
    Dict keys are not escaped (neither is the native ``Display``), so a key
    containing ``'.'`` or ``'['`` renders identically to the nested path it
    spells; ``Adapter.serve_route`` guards against that collision where the
    string is a lookup key.
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

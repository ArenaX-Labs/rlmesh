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

    Dict keys are deliberately not escaped (the native ``Display`` does not
    escape either, and dotted keys like GR00T's ``"state.x"`` are first-class),
    so a key containing ``'.'`` or ``'['`` renders identically to the nested
    path it spells. That is harmless for error text; the one consumer that uses
    the rendered string as a lookup key (``Adapter.serve_route``'s customs map)
    guards against an actual collision and raises there.
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

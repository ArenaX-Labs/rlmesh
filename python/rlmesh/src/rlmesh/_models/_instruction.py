"""Instruction injection into the model payload tree.

The ``instruction=`` override is written into every text leaf the model spec
declares, at its tree placement and in its declared shape. Pure tree machinery,
used by the per-step predict assembly in :mod:`rlmesh._models._eval`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple, cast

from ._adapter_mode import NO_ADAPTER

# Cross-module surface for ``_eval`` (see note in ``_connect``).
__all__ = ["TextPlacement", "text_placements", "tree_set"]


class TextPlacement(NamedTuple):
    """Where (and how) the ``instruction=`` override lands in the model payload.

    ``segments`` is the text leaf's position in the model input tree (str for a
    Dict key, int for a Tuple index; the empty tuple is a bare-root text input,
    whose payload *is* the text leaf). ``as_list`` is True when the leaf declares
    ``container='list'`` (inject ``[instruction]``, not a bare ``str``, to keep
    the model's declared shape).
    """

    segments: tuple[str | int, ...]
    as_list: bool


def text_placements(spec: object | None) -> tuple[TextPlacement, ...]:
    """Find every text leaf the ``instruction=`` override should be written into.

    Walks the model spec's input tree locally (a public structure: a leaf
    dataclass, a ``dict`` Dict node, or a ``tuple`` Tuple node), so the override
    reaches *every* text leaf -- bare-root, top-level, and nested -- and carries
    each leaf's ``container`` so a list-shaped leaf gets ``[instruction]``. A
    spec-less / ``NO_ADAPTER`` model declares no text inputs, so none.
    """
    if spec is None or spec is NO_ADAPTER:
        return ()
    from ..adapters import Text

    input_tree = getattr(spec, "input", None)
    if input_tree is None:
        return ()
    placements: list[TextPlacement] = []

    def walk(node: Any, segments: tuple[str | int, ...]) -> None:
        if isinstance(node, Text):
            placements.append(TextPlacement(segments, node.container == "list"))
        elif isinstance(node, Mapping):
            for key, child in cast("Mapping[str, Any]", node).items():
                walk(child, (*segments, key))
        elif isinstance(node, tuple):
            for index, child in enumerate(cast("tuple[Any, ...]", node)):
                walk(child, (*segments, index))

    walk(input_tree, ())
    return tuple(placements)


def tree_set(tree: Any, segments: tuple[str | int, ...], value: Any) -> Any:
    """Return ``tree`` with the value at ``segments`` replaced by ``value``.

    A small structured set over the payload tree (dict for str segments, list for
    int segments). Rebuilds only the path it touches, so the env's observation is
    never mutated; the empty path replaces the whole payload (a bare-root leaf).
    """
    if not segments:
        return value
    head, rest = segments[0], segments[1:]
    if isinstance(head, int):
        items: list[Any] = list(cast("Sequence[Any]", tree))
        items[head] = tree_set(items[head], rest, value)
        return items
    node: dict[str, Any] = (
        dict(cast("Mapping[str, Any]", tree)) if isinstance(tree, Mapping) else {}
    )
    # A subtree may not exist yet (e.g. injecting into a missing nested key);
    # descend into an empty dict rather than indexing a missing key.
    node[head] = tree_set(node.get(head, {}), rest, value)
    return node

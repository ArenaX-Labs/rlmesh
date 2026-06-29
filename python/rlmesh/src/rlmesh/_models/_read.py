"""Role-addressed, read-only views over an env's observations.

:class:`Reader` (built by :meth:`rlmesh._models._eval.Session.reader`) maps a raw
observation to a ``{role: value}`` dict through the same adapter pipeline a model
uses, pointed at the consumer. Plus the read-item normalization and the
read-adapter resolution that reuse the model adapter pipeline.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

from .._value_conversion import from_value

if TYPE_CHECKING:
    from .._value_conversion import ValueBridge
    from ..adapters import Action

# Cross-module surface for ``_eval.Session`` (see note in ``_connect``).
__all__ = ["Reader", "env_image_roles", "resolve_read_adapter"]


class Reader:
    """A resolved, role-addressed read over an env's observations (read-only).

    Built by :meth:`Session.reader`. Calling it maps one raw env observation to a
    ``{role: value}`` dict, each value in the encoding its read item declared -- a
    bare role keeps the env's native encoding; an :class:`~rlmesh.adapters.Image` /
    :class:`~rlmesh.adapters.State` leaf converts (``Image(IMAGE_PRIMARY,
    layout="hwc")`` yields an HWC image whatever the env stores). It is the same
    adapter pipeline a model uses, pointed at the consumer: resolved once, reused
    every step, and identical in the native core. Values come back in the env's own
    framework (numpy for a gym env, torch for a torch route).
    """

    def __init__(
        self, adapter: Any, roles: tuple[str, ...], bridge: ValueBridge
    ) -> None:
        self._adapter = adapter
        self._roles = roles
        self._bridge = bridge

    @property
    def roles(self) -> tuple[str, ...]:
        """The roles this reader extracts, in declaration order."""
        return self._roles

    def __call__(self, observation: object) -> dict[str, Any]:
        """Extract ``{role: value}`` from one raw env observation (never mutated)."""
        value = self._adapter.transform_obs_value(
            observation, input_bridge=self._bridge, custom_bridge=self._bridge
        )
        return cast("dict[str, Any]", from_value(value, self._bridge))


def _obs_tag_node(node: Any, role: str) -> Any:
    """The env obs tag node (ImageTag/StateTag/TextTag/Split) carrying ``role``.

    Walks the env's published observation tags -- authoritative, so a custom or
    prefix-less role (e.g. ``INSTRUCTION``) resolves correctly -- and returns None
    if no leaf carries the role. Returns the node itself (not just a kind) so a
    bare-role read can honor the env's declared image layout.
    """
    from ..adapters import ImageTag, Split, StateTag, TextTag

    if isinstance(node, Mapping):
        children: Any = cast("Any", node).values()
    elif isinstance(node, tuple):
        children = cast("Any", node)
    else:
        if isinstance(node, (ImageTag, StateTag, TextTag)):
            return node if node.role == role else None
        if isinstance(node, Split):
            return node if any(f.role == role for f in node.fields) else None
        return None
    for child in children:
        found = _obs_tag_node(child, role)
        if found is not None:
            return found
    return None


def _read_leaf(item: object, env_tags: Any) -> tuple[str, Any]:
    """Normalize a read item to ``(role, model-input leaf)``.

    A single-role model-input leaf (Image/State/Text) is taken as-is; a bare role
    string is desugared to the env-native leaf matching the env's tag for that role.
    """
    from ..adapters import (
        AdapterResolutionError,
        Image,
        ImageTag,
        Split,
        State,
        StateTag,
        Text,
        TextTag,
    )

    if isinstance(item, str):
        node = _obs_tag_node(env_tags.observation, item)
        if isinstance(node, ImageTag):
            # A bare image role keeps the env's native encoding -- carry the env's
            # declared layout so the read does not transpose a chw frame to hwc.
            return item, Image(item, layout=node.layout)
        if isinstance(node, (StateTag, Split)):
            return item, State(item)
        if isinstance(node, TextTag):
            return item, Text(item)
        raise AdapterResolutionError(
            f"the env declares no observation role {item!r} to read; pass an "
            f"explicit leaf (e.g. State({item!r})) or use a role the env tags"
        )
    role = getattr(item, "role", None)
    if not isinstance(role, str):
        raise TypeError(
            "a read item must be a role string or a single-role model-input leaf "
            f"(Image/State/Text); got {type(item).__name__}"
        )
    return role, item


def env_image_roles(contract: Any) -> list[str]:
    """The image roles an env declares, in observation-tag declaration order.

    Reads the env's published adapter tags (the same :class:`~rlmesh.adapters.EnvTags`
    the reader resolves against) and walks the observation tree for every
    :class:`~rlmesh.adapters.ImageTag`. Returns ``[]`` when the env publishes no tags.
    Unlike probing a fixed role list, this finds custom image roles too, and only the
    ones this env actually declares.
    """
    from ..adapters import EnvTags, ImageTag

    env_tags = EnvTags.from_metadata(getattr(contract, "metadata", None) or {})
    if env_tags is None:
        return []
    roles: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, Mapping):
            for child in cast("Any", node).values():
                walk(child)
        elif isinstance(node, tuple):
            for child in cast("Any", node):
                walk(child)
        elif isinstance(node, ImageTag):
            roles.append(node.role)

    walk(env_tags.observation)
    return roles


def _read_only_action(action: Action) -> Action:
    """The env action with its env-side-only clamps dropped, fit to ride a read along.

    A read reuses the model adapter pipeline with the env's own action as a *no-op*
    output (a read emits no action). But ``clip`` -- the global :attr:`Action.clip`
    and a per-:class:`~rlmesh.adapters.Actuator` ``clip`` alike -- stays env-side
    only: the resolver forbids a *model* action declaration from setting it. Echoing
    the env action verbatim therefore makes resolution reject *every* read against a
    clip-declaring env (the live viewer and ``record`` both read by role, so both go
    blank). Dropping the clamps is sound -- no action is written on a read -- and
    leaves the actuator layout the read genuinely needs untouched.
    """
    components = tuple(
        replace(c, clip=False) if c.clip else c for c in action.components
    )
    return type(action)(*components, clip=None)


def resolve_read_adapter(
    contract: Any, items: tuple[object, ...], trust: bool
) -> tuple[Any, tuple[str, ...]]:
    """Resolve a read-only adapter for ``items`` against the env contract.

    Reuses the model adapter pipeline: the read items become an obs-only
    :class:`~rlmesh.adapters.ModelSpec` input, the env's own action layout rides
    along as a no-op output (so the obs-only intent satisfies ModelSpec without
    inventing an action -- see :func:`_read_only_action` for why its env-side clamps
    are stripped first), and :func:`~rlmesh.adapters.resolve_from_contract` derives
    the same plan a model would.
    """
    from ..adapters import (
        AdapterResolutionError,
        EnvTags,
        ModelSpec,
        resolve_from_contract,
    )

    if contract is None:
        raise AdapterResolutionError(
            "reading by role needs an env contract, but the env exposes none"
        )
    env_tags = EnvTags.from_metadata(getattr(contract, "metadata", None) or {})
    if env_tags is None:
        raise AdapterResolutionError(
            "the env publishes no adapter tags, so there are no roles to read; "
            "serve it with rlmesh.adapters.tag(...)"
        )
    pairs = [_read_leaf(item, env_tags) for item in items]
    roles = tuple(role for role, _ in pairs)
    if len(set(roles)) != len(roles):
        raise AdapterResolutionError(f"a role is read more than once: {list(roles)}")
    spec = ModelSpec(
        input={role: leaf for role, leaf in pairs},
        output=_read_only_action(env_tags.action),
    )
    return resolve_from_contract(contract, spec, trust_entrypoints=trust), roles

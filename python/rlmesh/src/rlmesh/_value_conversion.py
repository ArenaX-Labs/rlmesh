"""Internal value conversion shared by Python framework backends.

Framework modules (``rlmesh.numpy``, ``rlmesh.torch``, ``rlmesh.jax``) supply
leaf encoders/decoders for their tensor type. This module owns the shared
tree-walking logic for RLMesh value payloads so runtime, model, and adapter
paths agree on one canonical Python value shape.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Final, Protocol, TypeVar, cast

from ._rlmesh import Tensor
from .specs import SpaceSpec
from .types import PrimitiveValue, Value

UNHANDLED: Final = object()
ValueT = TypeVar("ValueT")


class ValueBridge(Protocol):
    name: str

    def ensure_available(self) -> None: ...

    def decode(self, value: Value | None) -> object: ...

    def encode(self, value: object) -> Value: ...

    def tree_stack(self, trees: Sequence[object]) -> object: ...

    def tree_unstack(self, value: object, n: int) -> list[object]: ...

    def supports_device(self) -> bool: ...

    def to_device(self, value: object, device: object) -> object: ...

    def to_host(self, value: object) -> object: ...


class IdentityBridge:
    name: str = "rlmesh"

    def ensure_available(self) -> None:
        return None

    def decode(self, value: Value | None) -> Value | None:
        return value

    def encode(self, value: object) -> Value:
        return cast(Value, value)

    def tree_stack(self, trees: Sequence[object]) -> object:
        # Opaque native Values can't be fused into a batched tensor, so the
        # dependency-free Value model sees the per-lane list and returns one.
        return list(trees)

    def tree_unstack(self, value: object, n: int) -> list[object]:
        return list(cast("Iterable[object]", value))

    def supports_device(self) -> bool:
        return False

    def to_device(self, value: object, device: object) -> object:
        return value

    def to_host(self, value: object) -> object:
        return value


def decode_tree(
    value: Value | None, leaf_decoder: Callable[[Tensor], ValueT]
) -> (
    ValueT
    | PrimitiveValue
    | list[object]
    | tuple[object, ...]
    | dict[str, object]
    | None
):
    if value is None:
        return None
    if isinstance(value, Tensor):
        return leaf_decoder(value)
    if isinstance(value, list):
        return [decode_tree(item, leaf_decoder) for item in value]
    if isinstance(value, tuple):
        return tuple(decode_tree(item, leaf_decoder) for item in value)
    if isinstance(value, dict):
        return {key: decode_tree(item, leaf_decoder) for key, item in value.items()}
    return value


def encode_tree(value: object, leaf_encoder: Callable[[object], object]) -> Value:
    if isinstance(value, Tensor):
        return value
    if isinstance(value, dict):
        raw_mapping = cast(Mapping[object, object], value)
        return {
            str(key): encode_tree(item, leaf_encoder)
            for key, item in raw_mapping.items()
        }
    if isinstance(value, list):
        raw_items = cast(list[object], value)
        return [encode_tree(item, leaf_encoder) for item in raw_items]
    if isinstance(value, tuple):
        raw_items = cast(tuple[object, ...], value)
        return tuple(encode_tree(item, leaf_encoder) for item in raw_items)
    encoded = leaf_encoder(value)
    if encoded is UNHANDLED:
        return cast(Value, value)
    return cast(Value, encoded)


def tree_stack(
    trees: Sequence[object], stack_leaf: Callable[[list[object]], object]
) -> object:
    """Fuse N identically-shaped decoded trees into one with batched leaves.

    Containers (dict/list/tuple) recurse structurally; aligned leaves across the N
    trees are handed to ``stack_leaf`` (the framework's stack op), which gives each
    leaf a new leading batch axis. A Dict observation therefore fuses to
    ``{key: array[N, ...]}`` -- the "keys get batched" shape every RL/VLA runtime
    hands a policy. Lanes share an observation space, so structures align.
    """
    first = trees[0]
    if isinstance(first, dict):
        maps = [cast("Mapping[str, object]", t) for t in trees]
        first_map = cast("Mapping[str, object]", first)
        return {
            key: tree_stack([m[key] for m in maps], stack_leaf) for key in first_map
        }
    if isinstance(first, (list, tuple)):
        seqs = [cast("Sequence[object]", t) for t in trees]
        first_seq = cast("Sequence[object]", first)
        fused: list[object] = [
            tree_stack([s[i] for s in seqs], stack_leaf) for i in range(len(first_seq))
        ]
        return tuple(fused) if isinstance(first, tuple) else fused
    return stack_leaf(list(trees))


def tree_unstack(
    value: object, n: int, unstack_leaf: Callable[[object, int], list[object]]
) -> list[object]:
    """Split one batched tree back into N per-lane trees (inverse of tree_stack).

    Containers recurse; each batched leaf is split on its leading axis by
    ``unstack_leaf``. A batched action ``array[N, ...]`` becomes N ``array[...]``;
    a batched chunk ``array[N, horizon, ...]`` splits the batch axis only, leaving
    each lane's chunk (``horizon``) axis intact.
    """
    if isinstance(value, dict):
        mapping = cast("Mapping[str, object]", value)
        cols = {
            key: tree_unstack(item, n, unstack_leaf) for key, item in mapping.items()
        }
        return [{key: cols[key][i] for key in mapping} for i in range(n)]
    if isinstance(value, (list, tuple)):
        seq = cast("Sequence[object]", value)
        cols = [tree_unstack(item, n, unstack_leaf) for item in seq]
        lanes: list[object] = []
        for i in range(n):
            lane: list[object] = [col[i] for col in cols]
            lanes.append(tuple(lane) if isinstance(value, tuple) else lane)
        return lanes
    return list(unstack_leaf(value, n))


def tree_map(value: object, leaf_fn: Callable[[object], object]) -> object:
    """Apply ``leaf_fn`` to every leaf of a tree, preserving its container shape.

    Used to slice a leading axis off each leaf -- e.g. de-chunking an action chunk
    to a single action (``leaf[0]``) or its batched form (``leaf[:, 0]``).
    """
    if isinstance(value, dict):
        mapping = cast("Mapping[str, object]", value)
        return {key: tree_map(item, leaf_fn) for key, item in mapping.items()}
    if isinstance(value, (list, tuple)):
        seq = cast("Sequence[object]", value)
        mapped: list[object] = [tree_map(item, leaf_fn) for item in seq]
        return tuple(mapped) if isinstance(value, tuple) else mapped
    return leaf_fn(value)


identity_bridge: ValueBridge = IdentityBridge()


def _bridge_or_identity(bridge: ValueBridge | None) -> ValueBridge:
    return identity_bridge if bridge is None else bridge


def to_value(value: object, bridge: ValueBridge | None = None) -> Value:
    """Encode a backend/native Python payload into the canonical Value tree."""
    return _bridge_or_identity(bridge).encode(value)


def from_value(value: Value | None, bridge: ValueBridge | None = None) -> object:
    """Decode a canonical Value tree into the requested backend."""
    return _bridge_or_identity(bridge).decode(value)


def encode_framework_array_batch(
    value: object,
    *,
    bridge: ValueBridge | None,
    space: SpaceSpec,
    num_envs: int,
) -> object:
    """Encode a single framework array batch into per-env Value leaves.

    Vector environment APIs commonly pass one array shaped
    ``(num_envs, *action_shape)`` for leaf array spaces. Splitting here keeps
    that backend policy with the value conversion system instead of each client
    guessing how to identify and encode framework batches.

    Non-array batches, composite spaces, text spaces, identity/native values, and
    count mismatches pass through unchanged so the native batched codec remains
    the final authority.
    """
    active_bridge = _bridge_or_identity(bridge)
    if active_bridge.name == identity_bridge.name:
        return value
    if space.kind in ("dict", "tuple", "text"):
        return value
    if not _is_framework_array_batch(value):
        return value
    try:
        per_env = list(cast(Iterable[object], value))
    except TypeError:
        return value
    if len(per_env) != num_envs:
        return value
    return [active_bridge.encode(item) for item in per_env]


def _is_framework_array_batch(value: object) -> bool:
    """Return true for a single framework array batch, not a container batch."""
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        return False
    if isinstance(value, (list, tuple)):
        return False
    if isinstance(value, Sequence):
        return False
    return hasattr(value, "__len__") and hasattr(value, "__iter__")


class FrameworkBridge:
    """Tree-walking value conversion for one array framework.

    Implements the internal ``ValueBridge`` protocol. Tensor leaves decode
    through ``decode_leaf``; arbitrary leaves encode through ``encode_leaf``,
    which returns ``UNHANDLED`` to pass a value through unchanged.
    Availability is checked once per ``decode``/``encode`` call.
    """

    def __init__(
        self,
        *,
        name: str,
        ensure_available: Callable[[], None],
        decode_leaf: Callable[[Tensor], object],
        encode_leaf: Callable[[object], object],
        stack_leaf: Callable[[list[object]], object],
        unstack_leaf: Callable[[object, int], list[object]],
        to_device_leaf: Callable[[object, object], object] | None = None,
        to_host_leaf: Callable[[object], object] | None = None,
    ) -> None:
        self.name = name
        self._ensure_available = ensure_available
        self._decode_leaf = decode_leaf
        self._encode_leaf = encode_leaf
        self._stack_leaf = stack_leaf
        self._unstack_leaf = unstack_leaf
        # Optional device/host ops -- supplied by frameworks with a device concept
        # (torch, jax). Used by the env serving wrapper to place a decoded action on
        # the requested device and to move a reward/done leaf back to host.
        self._to_device_leaf = to_device_leaf
        self._to_host_leaf = to_host_leaf

    def ensure_available(self) -> None:
        self._ensure_available()

    def decode(self, value: Value | None) -> object:
        self._ensure_available()
        return decode_tree(value, self._decode_leaf)

    def encode(self, value: object) -> Value:
        self._ensure_available()
        return encode_tree(value, self._encode_leaf)

    def tree_stack(self, trees: Sequence[object]) -> object:
        self._ensure_available()
        return tree_stack(trees, self._stack_leaf)

    def tree_unstack(self, value: object, n: int) -> list[object]:
        self._ensure_available()
        return tree_unstack(value, n, self._unstack_leaf)

    def supports_device(self) -> bool:
        return self._to_device_leaf is not None

    def to_device(self, value: object, device: object) -> object:
        """Move every framework tensor leaf of ``value`` onto ``device``.

        A no-op when this framework has no device concept or ``device`` is None;
        non-tensor leaves (e.g. a Discrete action's int) pass through unchanged.
        """
        if self._to_device_leaf is None or device is None:
            return value
        leaf = self._to_device_leaf
        return tree_map(value, lambda item: leaf(item, device))

    def to_host(self, value: object) -> object:
        """Coerce a reward/terminated/truncated leaf back to a host scalar/list.

        Keeps a per-element device sync off the hot path for a GPU-resident
        reward/done tensor; a plain Python scalar passes through unchanged.
        """
        if self._to_host_leaf is None:
            return value
        return self._to_host_leaf(value)


def resolve_bridge(framework: str | ValueBridge) -> ValueBridge:
    """Resolve a framework name (or a bridge) to its :class:`ValueBridge` singleton.

    Accepts ``"numpy"``/``"torch"``/``"jax"`` (lazy-imported so the resolver does
    not pull in a framework that isn't asked for) or a ready ``ValueBridge``,
    which passes through. Unknown names raise ``ValueError`` at the call site.
    """
    if not isinstance(framework, str):
        return framework
    name = framework.strip().lower()
    if name in ("numpy", "np"):
        from .numpy import _numpy_bridge  # pyright: ignore[reportPrivateUsage]

        return _numpy_bridge
    if name == "torch":
        from .torch import _torch_bridge  # pyright: ignore[reportPrivateUsage]

        return _torch_bridge
    if name == "jax":
        from .jax import _jax_bridge  # pyright: ignore[reportPrivateUsage]

        return _jax_bridge
    raise ValueError(
        f"unknown framework {framework!r}; expected 'numpy', 'torch', 'jax', "
        f"or a rlmesh ValueBridge"
    )


__all__ = [
    "UNHANDLED",
    "FrameworkBridge",
    "ValueBridge",
    "encode_framework_array_batch",
    "from_value",
    "identity_bridge",
    "resolve_bridge",
    "to_value",
    "tree_map",
    "tree_stack",
    "tree_unstack",
]

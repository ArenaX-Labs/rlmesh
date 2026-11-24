from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

from ._rlmesh import Tensor
from ._value_conversion import tree_map

if TYPE_CHECKING:
    from ._value_conversion import ValueBridge

# Other array-like obs leaves still work but are slower and may diverge from the numpy path, so they warn once.
_PRIMITIVE_LEAF = (bool, int, float, str, bytes)


def _tuple_subspaces(space: object) -> Sequence[object] | None:
    inner = getattr(space, "spaces", None)
    if isinstance(inner, (list, tuple)):
        return cast("Sequence[object]", inner)
    return None


def _obs_info_split(
    result: object, observation_space: object = None
) -> tuple[Any, Any] | None:
    """Split a gym ``(obs, info)`` reset (info is always a Mapping) from a bare obs."""
    if isinstance(result, tuple):
        items = cast("tuple[Any, ...]", result)
        if len(items) == 2 and isinstance(items[1], Mapping):
            subspaces = _tuple_subspaces(observation_space)
            head: object = items[0]
            head_is_pair = (
                isinstance(head, tuple) and len(cast("tuple[object, ...]", head)) == 2
            )
            if subspaces is not None and len(subspaces) == 2 and not head_is_pair:
                return None
            return items[0], cast("Any", items[1])
    return None


class BridgedEnv:
    """Wrap an env so its obs/action seam speaks one array framework.

    Only ``reset``/``step`` are bridged. ``render()`` (like every other
    attribute) delegates raw to the wrapped env, and the native layer imports
    CPU arrays only -- so a framework env should return numpy/CPU frames from
    ``render()``.

    Args:
        env: The wrapped env (scalar or vector). All attributes other than
            ``reset``/``step`` delegate to it, so the native server's contract
            parse and scalar-vs-vector detection see the real env.
        bridge: The framework :class:`~rlmesh._value_conversion.ValueBridge`.
        device: Optional device for the *incoming* action; framework tensor
            leaves are moved onto it before ``step`` (a no-op for ``None`` or a
            non-tensor leaf such as a Discrete action's int).
    """

    def __init__(
        self, env: Any, bridge: ValueBridge, device: object | None = None
    ) -> None:
        bridge.ensure_available()
        self._env: Any = env
        self._bridge = bridge
        self._device = device
        self._warned_foreign: set[str] = set()
        self._foreign_scan_settled = False

    def reset(self, **kwargs: Any) -> object:
        """Reset the wrapped env and encode only its observation (info stays raw)."""
        # Encode only the observation: info is freeform metadata read via .tolist(), which a native Tensor lacks.
        result = self._env.reset(**kwargs)
        split = _obs_info_split(result, getattr(self._env, "observation_space", None))
        if split is not None:
            obs, info = split
            return (self._bridge.encode(obs), info)
        return self._bridge.encode(result)

    def step(self, action: Any) -> object:
        """Decode the action onto the env's framework, step, and encode the observation."""
        decoded = self._bridge.decode(action)
        if self._device is not None:
            decoded = self._bridge.to_device(decoded, self._device)
        # Accept the gymnasium 5-tuple and legacy 4-tuple (done = terminated, truncated False); native always gets a 5-tuple.
        result = self._env.step(decoded)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
        elif len(result) == 4:
            obs, reward, terminated, info = result
            truncated = False
        else:
            raise ValueError(
                f"env.step() must return a 4- or 5-tuple, got a "
                f"{len(result)}-element result"
            )
        encoded_obs = self._bridge.encode(obs)
        self._warn_foreign_leaves(encoded_obs)
        return (
            encoded_obs,
            self._bridge.to_host(reward),
            self._bridge.to_host(terminated),
            self._bridge.to_host(truncated),
            info,
        )

    def _warn_foreign_leaves(self, encoded_obs: object) -> None:
        if self._foreign_scan_settled:
            return

        def check(leaf: object) -> object:
            if (
                not isinstance(leaf, Tensor)
                and not isinstance(leaf, _PRIMITIVE_LEAF)
                and leaf is not None
                and hasattr(leaf, "ndim")
            ):
                key = f"{type(leaf).__module__}.{type(leaf).__qualname__}"
                if key not in self._warned_foreign:
                    self._warned_foreign.add(key)
                    fw = self._bridge.name
                    warnings.warn(
                        f"a {fw} env returned a non-{fw} array leaf of type {key!r} "
                        f"in its observation; it falls back to a slower per-element "
                        f"encode and may differ byte-for-byte from the numpy path. "
                        f"Return {fw} tensors (or rlmesh Tensors) for every obs leaf.",
                        stacklevel=3,
                    )
            return leaf

        known_before = len(self._warned_foreign)
        tree_map(encoded_obs, check)
        if len(self._warned_foreign) == known_before:
            self._foreign_scan_settled = True

    def __getattr__(self, name: str) -> Any:
        # An underscore name raises AttributeError without touching self._env, so probing _env before
        # __init__ (pickle/copy) cannot recurse.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._env, name)


__all__ = ["BridgedEnv"]

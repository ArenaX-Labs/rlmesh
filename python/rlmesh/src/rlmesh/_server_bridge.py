"""Framework-bridged env wrapper for the serving side.

``BridgedEnv`` is the seam-swapped dual of the model's predict bridge: where a
model decodes its input observation and encodes its output action, an env decodes
its input *action* and encodes its output *observation*. Wrapping an env in this
lets an author write ``reset``/``step`` in their framework's tensors (torch, jax)
-- including on a GPU -- while the Rust server stays framework-neutral
(``ValueBackend::Native``): Rust hands native ``Tensor`` leaves to ``step`` and
keeps the native leaves this wrapper returns from ``reset``/``step``.

The same wrapper serves scalar and vector envs unchanged: ``encode``/``decode``
are batch-axis-agnostic tree ops, and a vector obs is one fused ``[N, ...]`` leaf
that the native codec splits per lane.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

from ._rlmesh import Tensor
from ._value_conversion import tree_map

if TYPE_CHECKING:
    from ._value_conversion import ValueBridge

# Obs leaves that are fine to pass through unbridged: native tensors and Python
# scalars. Anything else array-like (a numpy array inside a torch obs, a foreign
# tensor) is a frameworks-mixed leaf -- it still works (native per-element encode)
# but is slower and may diverge from the numpy path, so it earns a one-time warn.
_PRIMITIVE_LEAF = (bool, int, float, str, bytes)


def _tuple_subspaces(space: object) -> Sequence[object] | None:
    """The ordered subspaces of a tuple-kind observation space, else ``None``.

    Matches both the rlmesh ``spaces.Tuple`` wrapper and gymnasium's ``Tuple``
    (whose ``spaces`` is a list/tuple); a ``Dict`` space's ``spaces`` mapping is
    excluded by the sequence check.
    """
    inner = getattr(space, "spaces", None)
    if isinstance(inner, (list, tuple)):
        return cast("Sequence[object]", inner)
    return None


def _obs_info_split(
    result: object, observation_space: object = None
) -> tuple[Any, Any] | None:
    """Split a gym ``(obs, info)`` reset (info is always a Mapping) from a bare obs.

    Returns ``None`` for a bare obs -- including a Tuple-space obs tuple -- so it
    encodes whole; only the (obs, info) shape splits, keeping info raw metadata.

    A legacy bare ``Tuple(x, Dict)`` obs is shaped exactly like ``(obs, info)``,
    so when the env declares a 2-entry tuple observation space the modern shape
    is recognized by its first element carrying the space's arity (a 2-tuple);
    a bare obs (first element not a 2-tuple) encodes whole.
    """
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
        # Fail loud at construction if the framework isn't importable, mirroring
        # ModelBase.__init__ (so a missing torch surfaces here, not mid-episode).
        bridge.ensure_available()
        self._env: Any = env
        self._bridge = bridge
        self._device = device
        self._warned_foreign: set[str] = set()
        self._foreign_scan_settled = False

    def reset(self, **kwargs: Any) -> object:
        """Reset the wrapped env and encode only its observation (info stays raw)."""
        # Encode only the observation, never the info dict. info is freeform
        # metadata that the native metadata path reads via numpy's .tolist(); a
        # native Tensor there has no such conversion, so encoding an array leaf
        # inside info would crash the reset RPC. Mirror step()'s split: a gym
        # ``(obs, info)`` return (info is always a Mapping) keeps info raw, while a
        # bare obs (including a Tuple-space obs tuple) encodes whole. kwargs forward
        # verbatim (only what the server set -- seed and/or options).
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
        # Accept the gymnasium 5-tuple and the legacy gym 4-tuple, exactly as the
        # native normalize_*_step_result does (conversion.rs): a 4-tuple's ``done``
        # is ``terminated`` with ``truncated`` False. The bridge always hands native
        # a 5-tuple, so the server stays 5-tuple-only downstream.
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
        # Warn (once per leaf type) if an obs leaf isn't this framework's tensor:
        # after encode, the framework's own tensors are native Tensors, so a leaf
        # that is still a foreign array slipped through unbridged. Checked on step
        # only -- the reset obs shares the same structure, so the first step covers
        # it.
        self._warn_foreign_leaves(encoded_obs)
        return (
            encoded_obs,
            self._bridge.to_host(reward),
            self._bridge.to_host(terminated),
            self._bridge.to_host(truncated),
            info,
        )

    def _warn_foreign_leaves(self, encoded_obs: object) -> None:
        """Warn once per foreign obs leaf type, then retire the per-step walk.

        The obs structure is stable for a served env, so the first step whose
        full walk discovers no new foreign leaf type settles the scan and every
        later step skips the tree walk entirely (keeping the hot path free of
        it). Foreign 0-d array scalars (e.g. ``np.float32``) count: they carry
        ``ndim`` like any array leaf and flow unencoded just the same.
        """
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
        # Delegate public attributes (observation_space/action_space/single_*/
        # num_envs/spec/metadata/render_mode/render/close/...) to the real env.
        # An underscore name raises AttributeError WITHOUT touching ``self._env``,
        # so a probe for ``_env`` itself before __init__ has run (pickle/copy build
        # an instance via __new__, then read __setstate__/__reduce_ex__) gets a
        # clean miss instead of recursing into __getattr__ forever. The public
        # attrs set in __init__ never route here -- __getattr__ only fires on a miss.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._env, name)


__all__ = ["BridgedEnv"]

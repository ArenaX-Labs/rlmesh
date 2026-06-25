"""The runtime adapter that applies a resolved plan to observations and actions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast, final

from .._value_conversion import ValueBridge, from_value, to_value
from ..numpy import NumpyArray, ensure_available
from ..types import Value
from .specs import ObsTransform, RotationTransform

if TYPE_CHECKING:
    from .._rlmesh import AdapterPlan

ActionT = TypeVar("ActionT")

# A raw env observation handed to an adapter: a mapping of named leaves for a
# Dict space, or a single bare leaf (array) for a flat (non-Dict) space.
RawObs = Mapping[str, Any] | NumpyArray


def _numpy_value_bridge() -> ValueBridge:
    from ..numpy import _numpy_bridge  # pyright: ignore[reportPrivateUsage]

    return _numpy_bridge


def _serve_custom(
    transform: ObsTransform, bridge: ValueBridge
) -> Callable[[Value], Value]:
    """Wrap a custom-input transform as a neutral Value-tree callable.

    The served engine passes the full per-lane observation as a neutral tree;
    this bridges it into the framework, runs the user transform, and bridges the
    result back -- so the host-language transform stays exactly as written.
    """

    def call(observation: Value) -> Value:
        return bridge.encode(transform(cast("Any", bridge.decode(observation))))

    return call


@dataclass(frozen=True)
class ObsEncShim:
    """Repack one observation payload key from its base encoding to custom."""

    model_key: str
    base: str
    width: int
    name: str
    dtype: str
    from_base: RotationTransform


@dataclass(frozen=True)
class ActEncShim:
    """Repack a model action slice from custom back to its base encoding.

    Applied before the native conversion; the slice offset is model-declared.
    """

    offset: int
    base: str
    width: int
    name: str
    to_base: RotationTransform


class AdapterBase(ABC, Generic[ActionT]):
    """Base class for env-to-model adapters.

    :func:`rlmesh.adapters.resolve` derives the spec-driven implementation
    (:class:`Adapter`); subclass this directly to plug a fully custom
    pairing into anything built around adapters instead. Implement the two
    transforms; ``wrap_predict`` comes for free. Custom adapters may hold
    state across steps (e.g. cache proprio from ``transform_obs`` for use
    in ``transform_action``) -- that is their power over declarative specs.
    Override :meth:`reset` to clear any such state at episode boundaries.
    """

    @abstractmethod
    def transform_obs(self, raw_obs: RawObs) -> dict[str, Any]:
        """Convert a raw env observation into the model input payload.

        ``raw_obs`` is a mapping for a Dict space, or a bare array/leaf for a
        flat (non-Dict) space.
        """

    @abstractmethod
    def transform_action(self, raw_action: object) -> ActionT:
        """Convert a model action output into the env action."""

    def reset(self, env_index: int | None = None) -> None:
        """Clear episode-scoped state, optionally for a single lane.

        ``env_index`` identifies the vector lane whose episode rolled, or
        ``None`` for a whole-vector reset. The default does nothing (resolved
        adapters are stateless). Stateful custom adapters override this and
        wire it to the model worker's per-lane reset so a single lane's
        autoreset never wipes the other still-running lanes' state.
        """

    @property
    def is_stateful(self) -> bool:
        """Whether the adapter carries per-stream state across steps.

        A stateful adapter must keep affinity to its lane (one instance per
        ``(session, route, env_index)``) and so cannot yet be shared across
        the lanes of a vector env. Custom adapters default to stateful (the
        safe assumption); :class:`Adapter` derives this from its frame
        history. The per-lane affinity manager that makes vectorized stateful
        adapters correct is not implemented yet.
        """
        return True

    def describe(self) -> str:
        """Return a human-readable summary of the adapter."""
        return f"{type(self).__name__} (custom adapter)"

    def wrap_predict(
        self, predict_fn: Callable[[dict[str, Any]], object]
    ) -> Callable[[Any], ActionT]:
        """Wrap a model predict function with both transforms.

        The returned callable takes a raw env observation -- a mapping, or a
        bare array/leaf for a flat (non-Dict) env -- and returns an env-ready
        action, suitable for :class:`rlmesh.numpy.Model`.
        """

        def predict(raw_obs: Any) -> ActionT:
            return self.transform_action(predict_fn(self.transform_obs(raw_obs)))

        return predict


@final
class Adapter(AdapterBase[NumpyArray]):
    """A resolved env-to-model adapter; build instances with ``resolve()``.

    Declarative plans run in the native ``rlmesh-adapters`` core; custom
    inputs run their host-language transforms on the raw Python
    observation, exactly as before.
    """

    def __init__(
        self,
        plan: AdapterPlan,
        customs: Mapping[str, ObsTransform],
        stacks: Mapping[str, int] | None = None,
        obs_enc_shims: tuple[ObsEncShim, ...] = (),
        action_enc_shims: tuple[ActEncShim, ...] = (),
    ) -> None:
        self._plan = plan
        self._customs = dict(customs)
        # Per-key frame-history depth (>1 only) and the rolling buffers that
        # back it. Stacking happens host-side, after the native transform.
        self._stacks = {key: n for key, n in (stacks or {}).items() if n > 1}
        self._buffers: dict[str, deque[Any]] = {}
        # Host-side custom-encoding shims: the native plan resolves each to a
        # base encoding; these repack the field at the boundary (obs after the
        # native transform, action before it).
        self._obs_enc_shims = obs_enc_shims
        self._action_enc_shims = action_enc_shims

    def transform_obs_value(
        self,
        raw_obs: RawObs,
        *,
        input_bridge: ValueBridge | None = None,
        custom_bridge: ValueBridge | None = None,
    ) -> dict[str, Value]:
        """Convert a raw env observation into a canonical Value-tree payload.

        Only the observation keys the plan actually reads are encoded and
        sent across the native boundary, so an unused -- possibly
        unencodable -- observation key never aborts a step. Custom inputs
        still see the full raw observation. Inputs that request frame history
        are stacked here from a rolling buffer, cleared by :meth:`reset`.
        """
        # A flat (non-Dict) observation is a single leaf: present it under the
        # reserved "." key the plan references for a StateLayout-tagged env.
        obs: Mapping[str, Any] = (
            raw_obs if isinstance(raw_obs, Mapping) else {".": raw_obs}
        )
        selected = {
            key: obs[key] for key in self._plan.referenced_obs_keys() if key in obs
        }
        payload = cast(
            "dict[str, Value]",
            self._plan.transform_obs(to_value(selected, input_bridge)),
        )
        # Repack custom-encoded keys from their resolved base encoding into the
        # model's convention, before stacking (so a stacked input stacks the
        # model's representation).
        numpy_bridge = _numpy_value_bridge()
        for shim in self._obs_enc_shims:
            if shim.model_key in payload:
                payload[shim.model_key] = to_value(
                    self._apply_obs_enc(
                        shim, from_value(payload[shim.model_key], numpy_bridge)
                    ),
                    numpy_bridge,
                )
        for key, depth in self._stacks.items():
            if key in payload:
                payload[key] = to_value(
                    self._stack_frames(
                        key, from_value(payload[key], numpy_bridge), depth
                    ),
                    numpy_bridge,
                )
        for key, transform in self._customs.items():
            # Custom inputs see the full observation (not just the plan's
            # referenced keys), normalized to a mapping -- identical to raw_obs
            # for a Dict env, and {".": leaf} for a flat one.
            payload[key] = to_value(transform(obs), custom_bridge)
        return payload

    def transform_obs(self, raw_obs: RawObs) -> dict[str, Any]:
        """Convert a raw env observation into the model input payload."""
        bridge = _numpy_value_bridge()
        return cast(
            "dict[str, Any]",
            from_value(
                self.transform_obs_value(
                    raw_obs, input_bridge=bridge, custom_bridge=bridge
                ),
                bridge,
            ),
        )

    def _stack_frames(self, key: str, frame: Any, depth: int) -> NumpyArray:
        import numpy as np

        frames = self._buffers.get(key)
        if frames is None:
            frames = deque[Any](maxlen=depth)
            self._buffers[key] = frames
        if not frames:
            # Pad the start of an episode with copies of the first frame so
            # the stack is full from step zero.
            for _ in range(depth - 1):
                frames.append(frame)
        frames.append(frame)
        return cast(
            "NumpyArray", np.stack(cast("list[NumpyArray]", list(frames)), axis=0)
        )

    def _apply_obs_enc(self, shim: ObsEncShim, value: Any) -> NumpyArray:
        import numpy as np

        base = cast("NumpyArray", np.asarray(value))
        if int(base.size) != shim.width:
            raise ValueError(
                f"custom encoding {shim.name!r} for {shim.model_key!r} expected "
                f"a width-{shim.width} {shim.base} value, got size {int(base.size)}"
            )
        out = cast("NumpyArray", np.asarray(shim.from_base(base)))
        # A rotation field is a flat width-N vector; reject a non-1-D return
        # (e.g. a stray reshape) rather than silently flattening it, which could
        # reorder elements.
        if out.ndim != 1 or int(out.size) != shim.width:
            raise ValueError(
                f"custom encoding {shim.name!r} for {shim.model_key!r} must "
                f"return a flat width-{shim.width} vector, from_base returned "
                f"shape {out.shape}"
            )
        # Restore the model's declared dtype: the native core already cast the
        # state to it, but from_base may have produced float64 (e.g. a fresh
        # array or a Python list).
        return cast("NumpyArray", out.astype(np.dtype(shim.dtype)))

    def reset(self, env_index: int | None = None) -> None:
        """Clear the frame-history buffers at an episode boundary.

        The rolling buffers are not lane-indexed (the per-lane affinity manager
        is unimplemented, see :attr:`is_stateful`). A stateful adapter is now
        rejected on ``num_envs>1`` at :func:`resolve_route_adapter`, so a resolved
        stateful adapter is always a single lane (env_index 0). Clear on a
        whole-vector reset or lane 0's roll; ignore any other lane so a
        mid-vector autoreset never wipes the shared buffers.
        """
        if env_index is None or env_index == 0:
            self._buffers.clear()

    @property
    def is_stateful(self) -> bool:
        """A resolved adapter is stateful only when it stacks frame history."""
        return bool(self._stacks)

    def _apply_action_enc(self, raw_action: object) -> object:
        if not self._action_enc_shims:
            return raw_action
        import numpy as np

        if isinstance(raw_action, Mapping):
            raise TypeError(
                "a CustomEncoding on an action component requires the model to "
                "emit a flat array action, not a mapping"
            )
        # A fresh float64 copy: the in-place slice writes below must not mutate
        # the array the caller still holds (np.asarray would alias it).
        action = cast("NumpyArray", np.array(raw_action, dtype=np.float64))
        if action.ndim != 1:
            raise ValueError(
                "a CustomEncoding on an action component requires a 1-D action "
                f"vector, got shape {action.shape}"
            )
        for shim in self._action_enc_shims:
            stop = shim.offset + shim.width
            segment = cast("NumpyArray", action[shim.offset : stop])
            converted = cast("NumpyArray", np.asarray(shim.to_base(segment)))
            # Reject a non-1-D / wrong-width return rather than reshaping it.
            if converted.ndim != 1 or int(converted.size) != shim.width:
                raise ValueError(
                    f"custom encoding {shim.name!r} must return a flat width-"
                    f"{shim.width} vector, to_base returned shape {converted.shape}"
                )
            action[shim.offset : stop] = converted
        return action

    def transform_action_value(
        self,
        raw_action: object,
        *,
        action_bridge: ValueBridge | None = None,
    ) -> Value:
        """Convert a model action vector into a canonical env action value."""
        if self._action_enc_shims:
            ensure_available()
            numpy_bridge = _numpy_value_bridge()
            raw_action = self._apply_action_enc(
                from_value(to_value(raw_action, action_bridge), numpy_bridge)
            )
            action_value = to_value(raw_action, numpy_bridge)
        else:
            action_value = to_value(raw_action, action_bridge)
        return cast("Value", self._plan.transform_action(action_value))

    def transform_action(self, raw_action: object) -> NumpyArray:
        """Convert a model action vector into the env action vector."""
        bridge = _numpy_value_bridge()
        return cast(
            "NumpyArray",
            from_value(
                self.transform_action_value(raw_action, action_bridge=bridge),
                bridge,
            ),
        )

    def serve_route(self, bridge: ValueBridge) -> dict[str, object]:
        """The served-route payload the native engine drives.

        The engine applies the native plan, frame-stacking, customs, and
        encoding shims in Rust; this hands it the native plan plus the host
        (Python) holes as neutral Value-tree callables (framework/numpy bridging
        stays here). Used only on the serve path; :meth:`transform_obs_value`
        remains the in-process run(env) single-lane path.
        """
        customs = {
            key: _serve_custom(transform, bridge)
            for key, transform in self._customs.items()
        }
        return {
            "plan": self._plan,
            "customs": customs,
            "obs_encodings": self._serve_obs_encodings if self._obs_enc_shims else None,
            "action_encodings": (
                self._serve_action_encodings if self._action_enc_shims else None
            ),
        }

    def _serve_obs_encodings(self, payload: dict[str, Value]) -> dict[str, Value]:
        """Repack enc-shimmed obs payload keys (neutral in, neutral out)."""
        numpy_bridge = _numpy_value_bridge()
        result = dict(payload)
        for shim in self._obs_enc_shims:
            if shim.model_key in result:
                result[shim.model_key] = to_value(
                    self._apply_obs_enc(
                        shim, from_value(result[shim.model_key], numpy_bridge)
                    ),
                    numpy_bridge,
                )
        return result

    def _serve_action_encodings(self, action: Value) -> Value:
        """Repack enc-shimmed action segments back to base (neutral in/out)."""
        numpy_bridge = _numpy_value_bridge()
        raw = self._apply_action_enc(from_value(action, numpy_bridge))
        return cast("Value", to_value(raw, numpy_bridge))

    def describe(self) -> str:
        """Return a human-readable summary of the resolved transformations."""
        native = self._plan.describe()
        if not self._obs_enc_shims and not self._action_enc_shims:
            return native
        lines = [native, "host-side encodings:"]
        for shim in self._obs_enc_shims:
            lines.append(f"  obs    {shim.model_key!r}: {shim.base} -> {shim.name}")
        for shim in self._action_enc_shims:
            stop = shim.offset + shim.width
            lines.append(f"  action [{shim.offset}:{stop}]: {shim.name} -> {shim.base}")
        return "\n".join(lines)

    def advisories(self) -> list[str]:
        """Per-env data-loss / fabrication notes for this resolved adapter.

        The "warn" subset of :meth:`describe` -- e.g. a camera the env did not
        provide that is being zero-filled, or an aspect crop that drops pixels.
        A managed runner can log these without failing; empty when nothing
        noteworthy happened.
        """
        return list(self._plan.advisories())


__all__ = ["Adapter", "AdapterBase"]

"""The runtime adapter that applies a resolved plan to observations and actions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast, final

from .._value_conversion import ValueBridge, from_value, to_value
from ..numpy import NumpyArray, ensure_available
from ..types import Value
from .helpers import render_placement
from .specs import ObsTransform, RotationTransform

if TYPE_CHECKING:
    from .._rlmesh import AdapterPlan, Advisory

ActionT = TypeVar("ActionT")

RawObs = Mapping[str, Any] | NumpyArray

Placement = tuple[str | int, ...]

# Reserved raw-obs envelope key for a flat (non-Dict) observation (``rlmesh_adapters::plans::OBS_ROOT_KEY``).
_OBS_ROOT_KEY = "<obs>"


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
    """Repack one slice of an observation payload leaf from base to custom.

    Keyed by the leaf's ``segments`` path in the input tree (each step a ``str``
    Dict key or ``int`` Tuple index; the empty tuple is a bare-leaf input).

    ``offset`` is where this field starts inside the ``native_width``-wide
    assembled leaf: a single-part state is the whole leaf, a part of a
    multi-part ``Concat`` is the slice ``[offset, offset + width)``.
    """

    segments: tuple[str | int, ...]
    base: str
    width: int
    name: str
    dtype: str
    from_base: RotationTransform
    offset: int
    native_width: int


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


def _tree_get(tree: Any, segments: Placement) -> Any:
    """Read the value at ``segments`` within a Value tree (dict/list/leaf)."""
    node = tree
    for segment in segments:
        node = node[segment]
    return node


def _tree_contains(tree: Any, segments: Placement) -> bool:
    node = tree
    for segment in segments:
        if isinstance(segment, int):
            if not isinstance(node, (list, tuple)):
                return False
            if segment >= len(cast("Sequence[object]", node)):
                return False
        elif not isinstance(node, Mapping) or segment not in node:
            return False
        node = node[segment]
    return True


def _tree_set(tree: Any, segments: Placement, value: Any) -> Any:
    """Return ``tree`` with the value at ``segments`` replaced by ``value``.

    The empty path replaces the whole tree. An ``int`` segment requires a list
    node and grows it with empty-dict placeholders; a ``str`` segment requires a
    map; an empty placeholder takes whichever kind its segment needs.
    """
    if not segments:
        return value
    head, rest = segments[0], segments[1:]
    if isinstance(head, int):
        if isinstance(tree, Mapping) and not tree:
            tree = []
        if not isinstance(tree, (list, tuple)):
            raise ValueError(
                f"payload placement conflict: expected a list to place index [{head}]"
            )
        items = list(cast("Sequence[Any]", tree))
        if head >= len(items):
            items.extend({} for _ in range(head + 1 - len(items)))
        items[head] = _tree_set(items[head], rest, value)
        return items
    if isinstance(tree, (list, tuple)) and not tree:
        tree = {}
    if not isinstance(tree, Mapping):
        raise ValueError(
            f"payload placement conflict: expected a mapping to place key {head!r}"
        )
    node = dict(cast("Mapping[Any, Any]", tree))
    node[head] = _tree_set(node.get(head, {}), rest, value)
    return node


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
    def transform_obs(self, raw_obs: RawObs) -> Any:
        """Convert a raw env observation into the model input payload.

        ``raw_obs`` is a mapping for a Dict space, or a bare array/leaf for a
        flat (non-Dict) space. The return is the model input payload *tree* (a
        dict, a list, or a bare leaf -- whatever the model spec's input tree
        declares).
        """

    @abstractmethod
    def transform_action(self, raw_action: object) -> ActionT:
        """Convert a model action output into the env action."""

    def reset(self) -> None:
        """Clear episode-scoped state at an episode boundary.

        The base adapter holds no episode-scoped state; a stateful custom adapter
        overrides this to clear its per-episode state. It is driven on the
        single-env local loop, so there is no per-lane bookkeeping to do (the
        served path stacks natively with episode-keyed buffers in the core).
        """

    def observe(self, raw_obs: RawObs) -> None:
        """Advance episode-scoped state for a step that predicted nothing.

        Called once per env step whose action came from a replayed chunk, so a
        frame history still sees consecutive steps. A no-op by default.
        """

    def explain(self) -> str:
        """Return a human-readable summary of the adapter."""
        return f"{type(self).__name__} (custom adapter)"

    def wrap_predict(
        self, predict_fn: Callable[[Any], object]
    ) -> Callable[[Any], ActionT]:
        """Wrap a model predict function with both transforms.

        ``predict_fn`` receives the model input payload *tree* -- a dict for a
        Dict-shaped input, a list for a Tuple-shaped input, or a bare leaf for a
        single-leaf input (whatever :meth:`transform_obs` produces). The
        returned callable takes a raw env observation -- a mapping, or a
        bare array/leaf for a flat (non-Dict) env -- and returns an env-ready
        action, suitable for :class:`rlmesh.numpy.Model`.

        The execution horizon is a runtime decision (``execution_horizon`` on
        ``ResolveAdapter``, owned by the runtime driver) and the served engine
        emits the chunk. This direct wrapper applies one action per step.
        """

        def predict(raw_obs: Any) -> ActionT:
            raw_action = predict_fn(self.transform_obs(raw_obs))
            return self.transform_action(raw_action)

        return predict


@final
class Adapter(AdapterBase[NumpyArray]):
    """A resolved env-to-model adapter; build instances with ``resolve()``.

    Declarative plans run in the native ``rlmesh-adapters`` core; custom
    inputs run their host-language transforms on the raw Python
    observation.
    """

    def __init__(
        self,
        plan: AdapterPlan,
        customs: Mapping[Placement, ObsTransform],
        obs_enc_shims: tuple[ObsEncShim, ...] = (),
        action_enc_shims: tuple[ActEncShim, ...] = (),
    ) -> None:
        self._plan = plan
        self._customs = dict(customs)
        self._history_keys = tuple(plan.history_keys())
        self._obs_enc_shims = obs_enc_shims
        self._action_enc_shims = action_enc_shims

    def transform_obs_value(
        self,
        raw_obs: RawObs,
        *,
        input_bridge: ValueBridge | None = None,
        custom_bridge: ValueBridge | None = None,
    ) -> Value:
        """Convert a raw env observation into a canonical Value-tree payload.

        The result is the model input payload *tree* (a nested dict/list, or a
        bare leaf for a single-leaf model input) -- the same shape the model's
        ``predict`` receives. Only the observation keys the plan actually reads
        are encoded and sent across the native boundary, so an unused -- possibly
        unencodable -- observation key never aborts a step. Custom inputs still
        see the full raw observation. Inputs that request frame history are
        stacked here from a rolling buffer, cleared by :meth:`reset`.
        """
        obs: Mapping[str, Any] = (
            raw_obs if isinstance(raw_obs, Mapping) else {_OBS_ROOT_KEY: raw_obs}
        )
        selected = {
            key: obs[key] for key in self._plan.referenced_obs_keys() if key in obs
        }
        payload = cast(
            "Value", self._plan.transform_obs(to_value(selected, input_bridge))
        )
        numpy_bridge = _numpy_value_bridge()
        for shim in self._obs_enc_shims:
            if _tree_contains(payload, shim.segments):
                repacked = to_value(
                    self._apply_obs_enc(
                        shim,
                        from_value(_tree_get(payload, shim.segments), numpy_bridge),
                    ),
                    numpy_bridge,
                )
                payload = _tree_set(payload, shim.segments, repacked)
        for segments, transform in self._customs.items():
            value = to_value(transform(obs), custom_bridge)
            payload = _tree_set(payload, segments, value)
        return payload

    def transform_obs(self, raw_obs: RawObs) -> Any:
        """Convert a raw env observation into the model input payload.

        Returns the payload tree (a dict for a Dict-shaped model input, a bare
        array/scalar for a single-leaf input, a list for a Tuple-shaped input).
        """
        bridge = _numpy_value_bridge()
        return from_value(
            self.transform_obs_value(
                raw_obs, input_bridge=bridge, custom_bridge=bridge
            ),
            bridge,
        )

    def _apply_obs_enc(self, shim: ObsEncShim, value: Any) -> NumpyArray:
        import numpy as np

        placement = render_placement(shim.segments)
        leaf = cast("NumpyArray", np.asarray(value))
        if leaf.ndim != 1 or int(leaf.size) != shim.native_width:
            raise ValueError(
                f"custom encoding {shim.name!r} for {placement!r} expected a flat "
                f"width-{shim.native_width} state, got shape {leaf.shape}"
            )
        stop = shim.offset + shim.width
        base = cast("NumpyArray", leaf[shim.offset : stop])
        out = cast("NumpyArray", np.asarray(shim.from_base(base)))
        # Reject a non-1-D return rather than flattening it, which could reorder elements.
        if out.ndim != 1 or int(out.size) != shim.width:
            raise ValueError(
                f"custom encoding {shim.name!r} for {placement!r} must "
                f"return a flat width-{shim.width} vector, from_base returned "
                f"shape {out.shape}"
            )
        # Copy: the slice write below must not mutate the payload the caller holds.
        repacked = cast("NumpyArray", leaf.astype(np.dtype(shim.dtype)))
        repacked[shim.offset : stop] = out
        return repacked

    def observe(
        self, raw_obs: RawObs, *, input_bridge: ValueBridge | None = None
    ) -> None:
        """Advance the frame windows for a step that predicted nothing.

        Every input that declares a ``stack`` still sees its frame, so windows
        hold consecutive steps. Only the observation keys the plan reads are
        encoded and no custom transform runs. The step is counted, so the next
        converted action is recorded as executed here for a part reading the
        previous action. A no-op for a plan with no per-episode history. Without
        ``input_bridge`` the observation is read as NumPy.
        """
        if not self._history_keys:
            return
        obs: Mapping[str, Any] = (
            raw_obs if isinstance(raw_obs, Mapping) else {_OBS_ROOT_KEY: raw_obs}
        )
        selected = {
            key: obs[key] for key in self._plan.referenced_obs_keys() if key in obs
        }
        bridge = input_bridge if input_bridge is not None else _numpy_value_bridge()
        self._plan.transform_history(to_value(selected, bridge))

    def history_windows(self) -> tuple[tuple[str, int, int], ...]:
        """``(placement, span, frame_bytes)`` per frame window, per live episode.

        ``span`` is how many *consecutive* frames the window holds to reach its
        oldest offset (4 for ``stack=4``, 7 for ``stack=4, stride=2``);
        ``frame_bytes`` is one processed frame, or ``0`` when the env's camera
        resolution was not derivable. A caller sizes an admission budget from
        ``num_envs * sum(span * frame_bytes)``.
        """
        return tuple(self._plan.history_windows())

    def history_keys(self) -> tuple[str, ...]:
        """Canonical placements of the inputs that hold a frame window.

        Empty when the model declares no ``stack``; :meth:`observe` is then a
        no-op and a caller may skip it.
        """
        return self._history_keys

    def reset(self) -> None:
        """Clear the frame-history windows at an episode boundary.

        Used on the local per-episode loop (a single env); the served path keeps
        its own episode-keyed buffers in the engine, so this is the local-path
        counterpart.
        """
        self._plan.reset_history()

    def _apply_action_enc(self, raw_action: object) -> object:
        if not self._action_enc_shims:
            return raw_action
        import numpy as np

        if isinstance(raw_action, Mapping):
            raise TypeError(
                "a CustomEncoding on an action component requires the model to "
                "emit a flat array action, not a mapping"
            )
        # Copy: the slice writes below must not mutate the caller's array.
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

        Hands the engine the native plan plus the host (Python) holes as
        Value-tree callables. Serve path only; :meth:`transform_obs_value` is the
        in-process path.
        """
        # The served engine looks host holes up by ``NodePath::to_string()``.
        customs = {
            render_placement(segments): _serve_custom(transform, bridge)
            for segments, transform in self._customs.items()
        }
        if len(customs) != len(self._customs):
            rendered = sorted(render_placement(s) for s in self._customs)
            colliding = sorted({r for r in rendered if rendered.count(r) > 1})
            raise ValueError(
                "custom input placements render to colliding route keys "
                f"{colliding}; the served engine routes by the rendered path, so "
                "Dict keys containing '.' or '[' must not shadow another custom "
                "input's placement -- rename the colliding keys"
            )
        return {
            "plan": self._plan,
            "customs": customs,
            "obs_encodings": self._serve_obs_encodings if self._obs_enc_shims else None,
            "action_encodings": (
                self._serve_action_encodings if self._action_enc_shims else None
            ),
        }

    def _serve_obs_encodings(self, payload: Value) -> Value:
        """Repack enc-shimmed obs payload leaves (neutral tree in, tree out).

        The served engine hands the whole assembled payload tree; scatter each
        shim into its placement path within that tree.
        """
        numpy_bridge = _numpy_value_bridge()
        result: Value = payload
        for shim in self._obs_enc_shims:
            if _tree_contains(result, shim.segments):
                repacked = to_value(
                    self._apply_obs_enc(
                        shim, from_value(_tree_get(result, shim.segments), numpy_bridge)
                    ),
                    numpy_bridge,
                )
                result = _tree_set(result, shim.segments, repacked)
        return result

    def _serve_action_encodings(self, action: Value) -> Value:
        """Repack enc-shimmed action segments back to base (neutral in/out)."""
        numpy_bridge = _numpy_value_bridge()
        raw = self._apply_action_enc(from_value(action, numpy_bridge))
        return to_value(raw, numpy_bridge)

    def explain(self) -> str:
        """Return a human-readable summary of the resolved transformations."""
        native = self._plan.describe()
        if not self._obs_enc_shims and not self._action_enc_shims:
            return native
        lines = [native, "host-side encodings:"]
        for shim in self._obs_enc_shims:
            placement = render_placement(shim.segments)
            stop = shim.offset + shim.width
            lines.append(
                f"  obs    {placement!r}[{shim.offset}:{stop}]: "
                f"{shim.base} -> {shim.name}"
            )
        for shim in self._action_enc_shims:
            stop = shim.offset + shim.width
            lines.append(f"  action [{shim.offset}:{stop}]: {shim.name} -> {shim.base}")
        return "\n".join(lines)

    def advisories(self) -> list[Advisory]:
        """Per-env data-loss / fabrication notes for this resolved adapter.

        The "warn" subset of :meth:`describe` -- e.g. a camera the env did not
        provide that is being zero-filled, or an aspect crop that drops pixels.
        Each note carries a ``severity``: ``"info"`` for benign hints and
        ``"caution"`` when the adapter substituted or fabricated model-visible
        data (a role-rebound camera, a zero-filled frame) -- the tier a managed
        runner or eval harness may want to hard-fail on. ``str(note)`` is the
        message; empty when nothing noteworthy happened.
        """
        return list(self._plan.advisories())


__all__ = ["Adapter", "AdapterBase"]

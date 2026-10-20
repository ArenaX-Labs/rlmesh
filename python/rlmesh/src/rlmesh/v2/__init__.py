"""Real-time sessions (protocol v2): robots, policies and the edge.

The v0.2 physical-AI API, side by side with the v1 SDK (Gym envs, model
servers) in the same package. A robot is a flat map ``path -> part``; a
policy declares what it wants by kind and path, binds to a robot, and sends
timed action chunks that the robot's edge plays, holds, or replaces with
its safe behaviour. ``import rlmesh.v2 as rlmesh`` runs the spec's examples
as written::

    import rlmesh.v2 as rlmesh

    with rlmesh.Host() as host:
        robot = rlmesh.Robot("ur10e", UR10E, session=host)
        robot.arm()
        policy = rlmesh.Policy(HOLD, session=host, name="hold")
        print(policy.bind("ur10e"))  # the binding table
        # robot loop: robot.publish("arm", pos=q, vel=dq); due = robot.due("arm")
        policy.run(lambda obs: {"arm": obs["q"][None]}, duration=5.0)

A robot can also run in lockstep (``Robot(..., clock="lockstep")``, §12):
each step is one call of its ``step(commands)``, and :class:`GymView` drives
it as a Gymnasium env. Existing Gym envs and v1 models reach lockstep
through their own servers and runs, unchanged, at ``lockstep://`` addresses.

Every rule (normalisation, binding, conversions, leases, the edge, the run
loop) runs in Rust; this module converts NumPy arrays at the boundary.
Observations are read-only arrays over the runtime's buffers (no copy when
aligned); ``publish`` copies its values before returning, so arrays can be
reused at once.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from types import TracebackType
from typing import TYPE_CHECKING, Any, NamedTuple, cast

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError("rlmesh.v2 requires numpy. Install rlmesh[numpy].") from exc

from .._rlmesh import (
    BindingRefused,
    SessionException,
    SessionRefused,
    StartSkipped,
    Tensor,
    V2Controller,
    V2Due,
    V2GymView,
    V2Host,
    V2LockstepRobot,
    V2Observation,
    V2Rate,
    V2Recorder,
    V2Robot,
)

if TYPE_CHECKING:
    from typing import TypeAlias

    from typing_extensions import Self

    Array: TypeAlias = np.ndarray[Any, Any]
else:
    Array = object

__all__ = [
    "Binding",
    "BindingRefused",
    "Due",
    "GymView",
    "Host",
    "Meta",
    "Observation",
    "Policy",
    "Rate",
    "Recorder",
    "Robot",
    "SessionException",
    "SessionRefused",
    "StartSkipped",
    "read_recording",
    "run",
]

#: A predict function: an observation in, one ``[horizon, ...]`` array (or a
#: dict of arrays) per action entry out.
Predict = Callable[["Observation"], Mapping[str, Any]]


class Host:
    """A session host in this process, listening on local TCP ports.

    Participants (robots, policies, operators) connect to its ``address``,
    in this process or another. Use it as a context manager, or ``close()``.
    """

    def __init__(
        self,
        *,
        ip: str = "127.0.0.1",
        tenant: str = "local",
        lease_grace_ms: int = 250,
    ) -> None:
        self._native: V2Host = V2Host(ip, tenant, lease_grace_ms)

    @property
    def address(self) -> str:
        """``{ip}:{port}/{tenant}/{session}``: what participants connect to."""
        return self._native.address

    def close(self) -> None:
        """Stop the host."""
        self._native.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def _address(session: Host | str) -> str:
    return session.address if isinstance(session, Host) else session


class _Field(NamedTuple):
    name: str
    dtype: np.dtype[Any] | None
    shape: tuple[int, ...]
    size: int


def _fields(raw: Sequence[tuple[str, str, Sequence[int]]]) -> tuple[_Field, ...]:
    out: list[_Field] = []
    for name, dtype, shape in raw:
        dims = tuple(int(d) for d in shape)
        out.append(
            _Field(
                name,
                None if dtype == "utf8" else np.dtype(dtype),
                dims,
                int(np.prod(dims, dtype=np.int64)),
            )
        )
    return tuple(out)


def _bytes(path: str, field: _Field, value: object) -> memoryview:
    """One field as C-contiguous bytes of its declared dtype (same-kind casts only)."""
    if field.dtype is None:
        raise TypeError(f"{path}.{field.name} is text: publish a str")
    array: Array = np.asarray(value)
    if array.dtype != field.dtype:
        if not np.can_cast(array.dtype, field.dtype, casting="same_kind"):
            raise TypeError(
                f"{path}.{field.name}: cannot publish {array.dtype} as {field.dtype}"
            )
        array = array.astype(field.dtype)
    if array.size != field.size:
        raise ValueError(
            f"{path}.{field.name}: {array.shape} has {array.size} elements, "
            f"the part declares {list(field.shape)}"
        )
    return _flat_bytes(array)


def _flat_bytes(array: Array) -> memoryview:
    """The array's elements as one C-contiguous run of bytes (copied only if not contiguous)."""
    flat: Array = np.ascontiguousarray(array).ravel()
    return flat.data.cast("B")


def _array(tensor: object) -> Array:
    """A native tensor as a read-only NumPy array over the same buffer."""
    return np.asarray(cast("Tensor", tensor))


def _value(value: object) -> object:
    if isinstance(value, Tensor):
        return _array(value)
    if isinstance(value, dict):
        fields = cast("dict[str, object]", value)
        return {name: _array(tensor) for name, tensor in fields.items()}
    return value


def _action(entry: str, value: object) -> tuple[str, list[int], memoryview]:
    array: Array = np.asarray(value)
    dtype: np.dtype[Any] = array.dtype
    if dtype.kind not in "biuf":
        raise TypeError(f"{entry}: {dtype} is not a numeric array")
    return (str(dtype), [int(d) for d in array.shape], _flat_bytes(array))


def _actions(actions: Mapping[str, Any]) -> dict[str, object]:
    if not isinstance(actions, Mapping):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError("predict must return a dict keyed by action name")
    packed: dict[str, object] = {}
    for entry, value in actions.items():
        if isinstance(value, Mapping):
            fields = cast("Mapping[str, object]", value)
            packed[entry] = {
                name: _action(f"{entry}.{name}", array)
                for name, array in fields.items()
            }
        else:
            packed[entry] = _action(entry, value)
    return packed


class Due:
    """What the robot does on this control tick (returned by :meth:`Robot.due`).

    ``mode`` is ``driving``, ``hold``, ``safe``, ``estop`` or ``idle``.
    ``value`` is the command to dispatch to ``path`` (one array, or a dict
    for a multi-field command), after the edge's validation and clamping;
    ``None`` when nothing is dispatched. ``safe`` names the declared safe
    behaviour to run now. ``handover`` is ``(group, from_path, to_path)`` on
    the tick a command-group handover starts. ``lease_token``,
    ``chunk_seq`` and ``index`` name the chunk and action the command came
    from, ``based_on`` the input samples it was computed from (``(path,
    seq_first, seq_last)``), ``received_ns`` and ``slot_ns`` when the edge
    received that chunk and when this action's slot starts (the robot's
    clock).
    """

    __slots__ = (
        "based_on",
        "chunk_seq",
        "handover",
        "index",
        "lease_token",
        "mode",
        "path",
        "received_ns",
        "safe",
        "slot_ns",
        "value",
    )

    def __init__(self, native: V2Due) -> None:
        self.path: str = native.path
        self.mode: str = native.mode
        self.safe: str | None = native.safe
        self.handover: tuple[str, str, str] | None = native.handover
        self.lease_token: int | None = native.lease_token
        self.chunk_seq: int | None = native.chunk_seq
        self.index: int | None = native.index
        self.received_ns: int | None = native.received_ns
        self.slot_ns: int | None = native.slot_ns
        self.based_on: list[tuple[str, int, int]] = native.based_on
        fields = native.fields
        self.value: Array | dict[str, Array] | None
        if fields is None:
            self.value = None
        elif len(fields) == 1:
            self.value = _array(next(iter(fields.values())))
        else:
            self.value = {name: _array(t) for name, t in fields.items()}

    def __repr__(self) -> str:
        return f"Due(path={self.path!r}, mode={self.mode!r}, value={self.value!r}, safe={self.safe!r})"


class Robot:
    """One robot: its map declared, its parts published, its commands served.

    ``parts`` is the robot map (§3): ``{path: {"kind": ..., ...}}``. The
    host normalises it; every commandable part gets an edge. The robot's
    control loop publishes its sensors and asks :meth:`due` what to dispatch
    on each control tick. ``on_handover(group, from_path, to_path)`` runs
    when a command group's lease moves between members (switch the device's
    control mode there); the group dispatches nothing until it returns.
    """

    def __init__(
        self,
        name: str,
        parts: Mapping[str, Any],
        *,
        session: Host | str,
        timeout: float = 10.0,
        on_handover: Callable[[str, str, str], None] | None = None,
        clock: str = "wall",
    ) -> None:
        if clock not in ("wall", "lockstep"):
            raise ValueError(f"clock {clock!r}: wall or lockstep")
        self._lockstep: V2LockstepRobot | None = None
        if clock == "lockstep":
            self._lockstep = V2LockstepRobot(_address(session), name, json.dumps(parts))
            self._native = cast("V2Robot", self._lockstep)
        else:
            self._native = V2Robot(_address(session), name, json.dumps(parts), timeout)
        self._on_handover = on_handover
        self._fields: dict[str, tuple[_Field, ...]] = {}
        self._kinds: dict[str, str] = {}
        for part in self._native.parts():
            path = cast("str", part["path"])
            self._kinds[path] = cast("str", part["kind"])
            self._fields[path] = _fields(
                cast("list[tuple[str, str, list[int]]]", part["fields"])
            )

    @property
    def name(self) -> str:
        """The robot's participant id."""
        return self._native.name

    @property
    def parts(self) -> dict[str, str]:
        """Each part's kind, by path, in authored order."""
        return dict(self._kinds)

    @property
    def mode(self) -> str:
        """The session mode (``disarmed``, ``armed``, ``active``, ``safe``, ``estop``)."""
        return self._native.mode

    def publish(
        self,
        path: str,
        value: object = None,
        /,
        *,
        capture_ns: int | None = None,
        **fields: object,
    ) -> int | None:
        """Publish one sample of ``path``; never blocks on the network.

        A single-field part takes the value itself (``publish("wrist/cam",
        frame)``, ``publish("task", "pick")``); a multi-field part takes one
        keyword per field (``publish("arm", pos=q, vel=dq)``) or a dict.
        Values are cast to the declared dtype when the cast is same-kind
        (``float64`` to ``float32``) and must have the declared element
        count. ``capture_ns`` is ``time.monotonic_ns()`` at capture (now by
        default). Returns the sample's seq, or ``None`` when it was
        coalesced or handed to the image encoder.
        """
        declared = self._fields.get(path)
        if declared is None:
            raise ValueError(
                f"{self.name} has no part {path!r} (its parts: {', '.join(self._fields)})"
            )
        if value is not None and fields:
            raise TypeError(f"{path}: pass the value or keyword fields, not both")
        if isinstance(value, Mapping):
            values = dict(cast("Mapping[str, object]", value))
        elif value is not None:
            if len(declared) != 1:
                names = ", ".join(f.name for f in declared)
                raise TypeError(
                    f"{path} has fields {names}: publish({path!r}, {declared[0].name}=...)"
                )
            values = {declared[0].name: value}
        else:
            values = dict(fields)
        unknown = set(values) - {f.name for f in declared}
        missing = [f.name for f in declared if f.name not in values]
        if unknown or missing:
            raise TypeError(
                f"{path}: fields {', '.join(f.name for f in declared)} "
                f"(missing {missing or 'none'}, unknown {sorted(unknown) or 'none'})"
            )
        buffers: list[str | memoryview] = []
        for field in declared:
            item = values[field.name]
            if field.dtype is None:
                if not isinstance(item, str):
                    raise TypeError(f"{path} is text: publish a str")
                buffers.append(item)
            else:
                buffers.append(_bytes(path, field, item))
        if self._lockstep is not None:
            self._lockstep.publish(path, buffers)
            return None
        return self._native.publish(path, buffers, capture_ns)

    def serve_lockstep(
        self,
        step: Callable[[dict[str, Any]], Any],
        reset: Callable[[int | None, dict[str, Any] | None], Any] | None = None,
        *,
        background: bool = False,
        timeout: float = 10.0,
    ) -> None:
        """Serve the lockstep exchange (§12): one ``step(commands)`` per step.

        ``commands`` maps each commanded part to its command (the array of a
        single-field command, else a dict of fields). Inside ``step`` and
        ``reset(seed, options)`` the robot :meth:`publish`es its parts; each
        returns ``None`` or ``(reward, terminated, truncated[, info])``
        (``reset`` may return just ``info``). Declares and arms first; with
        ``background``, returns once serving (stop with :meth:`close`).
        """
        if self._lockstep is None:
            raise RuntimeError(
                f"{self.name} runs on the wall clock: Robot(..., clock='lockstep')"
            )

        def outcome(result: Any) -> tuple[float, bool, bool, str]:
            if result is None:
                return (0.0, False, False, "")
            if isinstance(result, Mapping):
                return (0.0, False, False, json.dumps(result) if result else "")
            reward, terminated, truncated, *rest = result
            info = rest[0] if rest else None
            return (
                float(reward),
                bool(terminated),
                bool(truncated),
                json.dumps(info) if info else "",
            )

        def on_step(commands: dict[str, Any]) -> tuple[float, bool, bool, str]:
            return outcome(
                step({path: _value(value) for path, value in commands.items()})
            )

        def on_reset(
            seed: int | None, options_json: str
        ) -> tuple[float, bool, bool, str]:
            options = json.loads(options_json) if options_json else None
            return outcome(reset(seed, options) if reset is not None else None)

        self._lockstep.serve(on_step, on_reset, background, timeout)

    def due(self, path: str) -> Due | None:
        """Tick the edge serving ``path`` (and its command group): what to dispatch now.

        Call it once per control tick per command group, at the part's
        control rate: the call is the edge's tick, so its lateness is the
        robot's. ``None`` when there is nothing to do.
        """
        native = self._native.due(path)
        if native is None:
            return None
        due = Due(native)
        if due.handover is not None:
            if self._on_handover is not None:
                self._on_handover(*due.handover)
            self._native.handover_done(path)
        return due

    def arm(self) -> None:
        """DISARMED or SAFE to ARMED, in a fresh epoch."""
        self._native.arm()

    def disarm(self) -> None:
        """Back to DISARMED."""
        self._native.disarm()

    def estop(self, reason: str = "estop") -> None:
        """Emergency stop: latched until :meth:`clear`, then disarm and arm."""
        self._native.estop(reason)

    def clear(self) -> None:
        """Release a latched e-stop (the robot's local clear)."""
        self._native.clear()

    def report_safe(self, reason: str) -> None:
        """Tell the host the robot fell back on its own (a device watchdog fired)."""
        self._native.report_safe(reason)

    def now_ns(self) -> int:
        """This robot's clock: ``time.monotonic_ns()``."""
        return self._native.now_ns()

    def edge_stats(self, path: str) -> dict[str, Any]:
        """The counters of the edge serving ``path``: accepted, rejected, mode ticks."""
        return self._native.edge_stats(path)

    def close(self) -> None:
        """Detach from the session."""
        self._native.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class Meta:
    """A snapshot's metadata (``obs.meta``): ages, staleness and provenance."""

    __slots__ = (
        "ages_ns",
        "based_on",
        "blank",
        "padded",
        "skew_ns",
        "stale",
        "time_ns",
    )

    def __init__(self, native: V2Observation) -> None:
        #: Each input's conservative age at the snapshot (``None``: blank).
        self.ages_ns: dict[str, int | None] = native.ages
        #: Inputs delivered stale under ``allow: hold``.
        self.stale: frozenset[str] = frozenset(native.stale)
        #: Window slots padded by ``allow: warmup``, by entry.
        self.padded: dict[str, int] = native.padded
        #: Inputs bound to a constant (``allow: blank``).
        self.blank: frozenset[str] = frozenset(native.blank)
        #: The newest capture minus the oldest, across inputs.
        self.skew_ns: int = native.skew_ns
        #: Per stream, ``(stream id, seq_first, seq_last)``: the chunk's ``based_on``.
        self.based_on: list[tuple[int, int, int]] = native.based_on
        #: The session time the snapshot was taken at.
        self.time_ns: int = native.time_ns


class Observation(Mapping[str, Any]):
    """What ``predict`` receives: each input by entry name, plus ``meta``.

    Arrays are read-only and stay valid while the observation lives; text
    inputs are ``str``; several fields that are not one concatenable array
    arrive as a dict (§6.1).
    """

    __slots__ = ("_native", "_values", "meta")

    def __init__(self, native: V2Observation) -> None:
        self._native = native
        raw = cast("dict[str, object]", native.values)
        self._values: dict[str, Any] = {entry: _value(v) for entry, v in raw.items()}
        self.meta: Meta = Meta(native)

    def __getitem__(self, entry: str) -> Any:
        return self._values[entry]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        return f"Observation({', '.join(self._values)})"


class Binding:
    """A binding table (§6.8): what each entry is bound to and how it is delivered."""

    __slots__ = ("fingerprint", "json", "replan_hz", "table", "warnings")

    def __init__(self, native: V2Controller) -> None:
        #: The human table, as the cockpit shows it.
        self.table: str = native.table
        #: The binding fingerprint (16 hex digits).
        self.fingerprint: str = native.fingerprint
        #: The machine-readable table the fingerprint is taken over.
        self.json: dict[str, Any] = cast(
            "dict[str, Any]", json.loads(native.table_json)
        )
        #: Warnings the binding was accepted with.
        self.warnings: list[str] = native.warnings
        #: The replan rate R.
        self.replan_hz: float = native.replan_hz

    def __str__(self) -> str:
        return self.table

    def __repr__(self) -> str:
        return f"Binding({self.json['policy']!r} on {self.json['robot']!r}, {self.fingerprint})"


class Policy:
    """A policy map (§6) and, once bound, the controller that runs it.

    ``priority`` is the lease class it attaches with: ``model`` (default),
    ``operator`` (preempts models: teleop), ``safety`` or ``scripted``.
    """

    def __init__(
        self,
        policy: Mapping[str, Any],
        *,
        session: Host | str,
        name: str = "policy",
        priority: str = "model",
        timeout: float = 10.0,
    ) -> None:
        self._policy_json = json.dumps(policy)
        self._address = _address(session)
        self._name = name
        self._priority = priority
        self._timeout = timeout
        self._native: V2Controller | None = None
        self._binding: Binding | None = None

    @property
    def name(self) -> str:
        """The policy's participant id."""
        return self._name

    def bind(
        self,
        robot: str,
        *,
        under: str | None = None,
        at: Mapping[str, str] | None = None,
        names: Mapping[str, str] | None = None,
        allow: Sequence[str] | None = None,
        **captures: str,
    ) -> Binding:
        """Attach, bind to ``robot``'s parts and declare the binding.

        ``under`` scopes relative selectors, ``at`` overrides an entry's
        path, ``names={"entry": "positional"}`` binds joints by index,
        ``allow`` adds conversions for every entry, and keyword captures
        fill ``{name}`` segments (§6.4). Raises :class:`BindingRefused`
        with the binder's explanation (§6.7).
        """
        if self._native is not None:
            raise SessionException(f"policy {self._name} is already bound")
        positional: list[str] = []
        for entry, mode in (names or {}).items():
            if mode != "positional":
                raise ValueError(
                    f'names={{{entry!r}: {mode!r}}}: only "positional" is an override; '
                    "joint name lists belong in the policy map"
                )
            positional.append(entry)
        self._native = V2Controller(
            self._address,
            self._name,
            self._policy_json,
            robot,
            priority=self._priority,
            under=under,
            at=dict(at or {}),
            positional=positional,
            allow=list(allow or ()),
            captures=dict(captures),
            timeout_s=self._timeout,
        )
        self._binding = Binding(self._native)
        return self._binding

    def _bound(self) -> V2Controller:
        if self._native is None:
            raise SessionException(
                f"policy {self._name} is not bound: call bind() first"
            )
        return self._native

    @property
    def binding(self) -> Binding:
        """The binding table (after :meth:`bind`)."""
        self._bound()
        return cast("Binding", self._binding)

    def observe(self) -> Observation:
        """A policy start now; raises :class:`StartSkipped` when one is skipped (§8.8)."""
        return Observation(self._bound().observe())

    def send(
        self, actions: Mapping[str, Any], observation: Observation
    ) -> dict[str, int | None]:
        """Send one timed chunk per action entry, computed from ``observation``.

        ``actions`` maps each action entry to its ``[horizon, ...]`` array
        (or a dict of arrays). Returns each entry's chunk seq, or ``None``
        when this policy does not hold the part's lease.
        """
        native = observation._native  # pyright: ignore[reportPrivateUsage]
        return self._bound().send(_actions(actions), native)

    def acquire(self) -> dict[str, int]:
        """Take the binding's leases now (preempting a lower class), then keep them.

        Returns the lease token per part; raises :class:`SessionRefused`
        (``status`` ``LEASE_BUSY``, ``GROUP_BUSY``, ``NOT_READY``, ...)
        when they cannot be granted.
        """
        return dict(self._bound().acquire())

    def release(self) -> None:
        """Give the leases back; they are not re-requested until acquire() or run()."""
        self._bound().release()

    @property
    def holds(self) -> bool:
        """Whether this policy holds every lease its binding commands."""
        return self._bound().holds()

    def run(self, predict: Predict, *, duration: float | None = None) -> dict[str, Any]:
        """Run the policy loop (§8.9) on this thread; returns the run's counters.

        ``predict(obs)`` is called at most at the replan rate, one inference
        in flight; a start that falls while it runs is skipped, never
        queued. Leases are requested once the inputs are warm, re-requested
        after a preemption ends, and released when the run returns. If
        ``predict`` raises, the policy stops sending, releases its leases
        (the robot holds, then goes safe) and the exception propagates.
        Stops after ``duration`` seconds or on :meth:`stop`.
        """
        native = self._bound()

        def call(raw: V2Observation) -> dict[str, Any]:
            return _actions(predict(Observation(raw)))

        native.run(call, duration)
        return self.stats()

    def stop(self) -> None:
        """Ask a :meth:`run` in progress (on another thread) to return."""
        self._bound().stop()

    def stats(self) -> dict[str, Any]:
        """Starts, completions, skips by reason, chunks sent and latency (§8.10).

        ``aligned_to`` names the input whose publication the starts are
        phased just after (wall clock), ``phase_offset_ms`` how far.
        """
        return self._bound().stats()

    def events(self) -> list[dict[str, Any]]:
        """Take the events reported so far: ``rate_drift``, ``inference_pending``."""
        return self._bound().events()

    def close(self) -> None:
        """Release the leases and detach."""
        if self._native is not None:
            self._native.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


try:
    import gymnasium as _gym
except ImportError:  # pragma: no cover - optional
    _gym = None

if TYPE_CHECKING:
    from gymnasium import Env as _GymBase

    _GymEnv: TypeAlias = _GymBase[dict[str, Any], dict[str, Any]]
else:
    _GymEnv = _gym.Env if _gym is not None else object


def _space(kind: str, *described: Any) -> Any:
    assert _gym is not None
    spaces = _gym.spaces
    if kind == "text":
        return spaces.Text(max_length=4096)
    if kind == "dict":
        return spaces.Dict(
            {name: _space("array", d, sh) for name, d, sh in described[0]}
        )
    dtype, shape = np.dtype(described[0]), tuple(int(d) for d in described[1])
    box: Any = spaces.Box
    if dtype == np.bool_:
        return box(0, 1, shape, dtype=np.bool_)
    if dtype.kind == "f":
        return box(-np.inf, np.inf, shape, dtype=dtype)
    info = np.iinfo(dtype)
    return box(info.min, info.max, shape, dtype=dtype)


class GymView(_GymEnv):
    """A v2-native lockstep robot as a Gymnasium env (§12.6).

    ``policy`` (a policy map) keys the observation space by its input names
    and the action space by its action names (``[horizon, ...]``, squeezed
    when the horizon is 1); without one, every part is keyed by its path
    (``/`` as ``.``): a single-field part is its field, a multi-field part a
    dict of fields. One :meth:`step` with horizon H plays H lockstep steps
    (stopping at the episode's end), rewards summed. ``info`` carries
    ``ages_ns``, ``stale`` and ``edge_modes``, plus the robot's own info.
    The robot must serve lockstep (``Robot(..., clock="lockstep")``).
    """

    def __init__(
        self,
        session: Host | str,
        policy: Mapping[str, Any] | None = None,
        *,
        robot: str | None = None,
        name: str = "gym",
        timeout: float = 10.0,
    ) -> None:
        if _gym is None:
            raise ImportError("GymView requires gymnasium. Install gymnasium.")
        self._native = V2GymView(
            _address(session),
            robot,
            json.dumps(policy) if policy is not None else None,
            name,
            timeout,
        )
        spaces = _gym.spaces
        observations = self._native.observation_entries()
        actions = self._native.action_entries()
        self.observation_space = spaces.Dict(
            {entry: _space(*described) for entry, described in observations}
        )
        self.action_space = spaces.Dict(
            {entry: _space(*described) for entry, described in actions}
        )

    @property
    def table(self) -> str | None:
        """The binding table (§6.8), with a policy."""
        return self._native.table()

    @staticmethod
    def _out(raw: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info, robot_info = raw
        observation = {entry: _value(value) for entry, value in obs.items()}
        merged = dict(info)
        if robot_info:
            merged.update(json.loads(robot_info))
        return observation, float(reward), bool(terminated), bool(truncated), merged

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset the robot (``seed`` and ``options`` reach its reset callback)."""
        super().reset(seed=seed)
        obs, _, _, _, info = self._out(
            self._native.reset(seed, json.dumps(options) if options else "")
        )
        return obs, info

    def step(
        self, action: Mapping[str, Any]
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """One step: ``action`` keyed by action entry (one horizon's chunk each)."""
        return self._out(self._native.step(_actions(action)))

    def close(self) -> None:
        """Release the lease and detach."""
        self._native.close()


class Recorder:
    """An MCAP recording of the session (§13), attached as an observer.

    Records every robot's streams (``/{robot}/{path}``, with the edge's
    ``/{robot}/{path}/_applied``), every action chunk sent to an edge
    (``/{robot}/{path}/_chunks``), and the authority and binding changes
    (``/_session/authority``, ``/_session/bindings``), each envelope as it
    travelled. It declares no delivery contract: a stream a policy reads at
    ``latest@R`` is recorded at the rate it is published. Use it as a
    context manager, or :meth:`close` to finish the file.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        session: Host | str,
        name: str = "recorder",
        timeout: float = 10.0,
    ) -> None:
        self._native = V2Recorder(_address(session), os.fspath(path), name, timeout)
        self.stats: dict[str, Any] | None = None

    @property
    def path(self) -> str:
        """The MCAP file."""
        return self._native.path

    def close(self) -> dict[str, Any] | None:
        """Finish the file; returns ``messages``, ``channels``, ``bytes`` (also kept in ``stats``)."""
        stats = self._native.close()
        if stats is not None:
            self.stats = stats
        return self.stats

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def read_recording(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read a recording back: messages per topic, time ordering, the lease timeline.

    ``topics`` maps each topic to its ``schema``, ``messages``,
    ``log_time_monotonic``, ``publish_time_monotonic`` and ``undecodable``
    count; ``leases`` lists every lease change (``part``, ``holder``,
    ``priority``, ``token``) in order.
    """
    return V2Recorder.read(os.fspath(path))


class Rate:
    """A fixed-rate loop timer on ``time.monotonic_ns()`` (a robot's control loop).

    :meth:`sleep` waits for the next period with the GIL released; a call
    more than a period late skips the missed periods instead of bursting.
    """

    __slots__ = ("_native",)

    def __init__(self, hz: float) -> None:
        self._native = V2Rate(hz)

    def sleep(self) -> int:
        """Wait for the next period; returns how late this call was, in ns."""
        return self._native.sleep()

    @property
    def missed(self) -> int:
        """Periods skipped so far."""
        return self._native.missed


def run(
    policy: Mapping[str, Any],
    robot: str,
    predict: Predict,
    *,
    session: Host | str,
    name: str = "policy",
    priority: str = "model",
    under: str | None = None,
    at: Mapping[str, str] | None = None,
    names: Mapping[str, str] | None = None,
    allow: Sequence[str] | None = None,
    duration: float | None = None,
    timeout: float = 10.0,
    **captures: str,
) -> dict[str, Any]:
    """Bind ``policy`` to ``robot`` and run ``predict`` until ``duration`` (§6.4 options).

    Shorthand for :class:`Policy` + :meth:`Policy.bind` + :meth:`Policy.run`.
    """
    with Policy(
        policy, session=session, name=name, priority=priority, timeout=timeout
    ) as p:
        p.bind(robot, under=under, at=at, names=names, allow=allow, **captures)
        return p.run(predict, duration=duration)

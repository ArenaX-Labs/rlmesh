"""Real-time sessions: robots, policies and the edge (Experimental).

The physical-AI API, beside the evaluation API (Gym envs, model servers)
of the top-level package. A robot is a flat map ``path -> part``; a policy
declares what it wants by kind and path, binds to a robot, and sends timed
action chunks that the robot's edge plays, holds, or replaces with its safe
behaviour::

    from rlmesh import realtime
    from rlmesh.realtime import actions, inputs, parts

    ARM = {"arm": parts.joints(NAMES, hz=500, command="position", safe=parts.Safe.HOLD)}

    with realtime.Host() as host:
        robot = realtime.Robot("ur10e", ARM, session=host)
        robot.arm()
        policy = realtime.Policy(
            hz=50,
            inputs={"q": inputs.joints("ur10e:arm", fields=["pos"])},
            actions={"arm": actions.joints("ur10e:arm", command="position", chunk=10)},
            session=host,
        )
        print(policy.bind("ur10e"))  # the binding table
        # robot loop: robot.publish("arm", pos=q, vel=dq); due = robot.due("arm")
        policy.run(lambda obs: {"arm": obs["q"][None]}, duration=5.0)

:mod:`~rlmesh.realtime.parts`, :mod:`~rlmesh.realtime.inputs` and
:mod:`~rlmesh.realtime.actions` each return the plain dict they stand for
(``inputs.joints("ur10e:arm")`` is ``{"kind": "joints", "at":
"ur10e:arm"}``), and a dict written by hand is accepted wherever one is.

A robot can also run in lockstep (``Robot(..., clock=Clock.LOCKSTEP)``):
each step is one call of its ``step(commands)``, and ``RemoteEnv`` at its
participant address (``robot.address``) drives it as a Gymnasium env.
Existing Gym envs and v1 models reach lockstep through their own servers
and runs, unchanged.

Every rule (normalisation, binding, conversions, leases, the edge, the run
loop) runs in Rust; this module converts NumPy arrays at the boundary.
Observations are read-only arrays over the runtime's buffers (no copy when
aligned); ``publish`` copies its values before returning, so arrays can be
reused at once.
"""

from __future__ import annotations

import enum
import inspect
import json
import logging
import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, cast

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError("rlmesh.realtime requires numpy. Install rlmesh[robot].") from exc

from .._client._endpoint import redact_address
from .._rlmesh import (
    BindingRefused,
    SessionException,
    SessionLost,
    SessionRefused,
    StartSkipped,
    Tensor,
    V2Controller,
    V2Due,
    V2Host,
    V2LockstepRobot,
    V2Observation,
    V2Rate,
    V2Recorder,
    V2Robot,
    parse_address,
)
from . import actions as actions
from . import inputs as inputs
from . import parts as parts
from . import results as results
from ._entries import Conversion
from .results import (
    EdgeStats,
    Event,
    HandBack,
    Health,
    Held,
    HostLink,
    PolicyStats,
)

if TYPE_CHECKING:
    from typing import TypeAlias

    from typing_extensions import Self

    Array: TypeAlias = np.ndarray[Any, Any]
else:
    Array = object

__all__ = [
    "BindingRefused",
    "BindingTable",
    "BoardData",
    "Clock",
    "Due",
    "Host",
    "Meta",
    "Observation",
    "Policy",
    "Rate",
    "Resume",
    "Robot",
    "RobotPart",
    "RunMode",
    "SessionException",
    "SessionLost",
    "SessionRecorder",
    "SessionRefused",
    "StartSkipped",
    "Step",
    "actions",
    "drive",
    "inputs",
    "parse_address",
    "parts",
    "read_recording",
    "results",
]

#: A predict function: an observation in (and, when it takes a second
#: argument, the :class:`Policy` running it), one ``[chunk, ...]`` array (or a
#: dict of arrays) per action entry out.
Predict = (
    Callable[["Observation"], Mapping[str, Any]]
    | Callable[["Observation", "Policy"], Mapping[str, Any]]
)

_log = logging.getLogger(__name__)


def _close_leaving(close: Callable[[], object], exc: BaseException | None) -> None:
    """Close on leaving a ``with`` block; an exception already in flight stays the one raised."""
    if exc is None:
        close()
        return
    try:
        close()
    except Exception as failed:
        _log.warning("closing after %s also failed: %s", type(exc).__name__, failed)


class BoardData(enum.Enum):
    """How a host carries boards' realtime parts (``rlmesh host --board-data``)."""

    #: Over the host's datagram path, UDP on its port + 1, back to TCP while
    #: that path is broken. The default.
    UDP = "udp"
    #: Over the session's TCP links: no UDP socket. For a sandbox or network
    #: that forbids UDP.
    TCP = "tcp"
    #: Over the datagram path only, for measuring it: never back to TCP.
    UDP_ONLY = "udp-only"


class Host:
    """The session at ``address`` (``[host/][session]``), by the hosting rule.

    ``":0"``, the default, hosts a session in this process on an OS-assigned
    port; an address without a host joins the local host on port 7447, or
    hosts the session here when nothing listens there. Participants
    (robots, policies, operators) connect to its ``address``, in this
    process or another. Use it as a context manager, or ``close()``.

    It listens on loopback only: an address on another interface is refused
    unless ``insecure=True`` (sessions other machines join are ``rlmesh host``'s).
    On loopback every link presents the password the address's key derives;
    a host bound beyond it gates no link, and only the handshake admits.

    ``board_data`` is how a session hosted here carries boards' realtime
    parts (:class:`BoardData` or its value, ``"tcp"``); ``None`` takes
    ``RLMESH_BOARD_DATA`` from the environment, else ``udp``, so one
    variable sets it for every host a process starts for itself.
    """

    def __init__(
        self,
        address: str = ":0",
        *,
        lease_grace_ms: int = 250,
        insecure: bool = False,
        board_data: BoardData | Literal["udp", "tcp", "udp-only"] | None = None,
    ) -> None:
        if isinstance(board_data, BoardData):
            board_data = board_data.value
        self._native: V2Host = V2Host(address, lease_grace_ms, insecure, board_data)

    @property
    def address(self) -> str:
        """``{ip}:{port}[/{session}]#{key}``: what participants connect to.

        The key is a credential (its holder attaches as device ``local``):
        pass the address to the processes you mean to join, not to logs.
        """
        return self._native.address

    def __repr__(self) -> str:
        return f"Host({redact_address(self.address)!r})"

    def health(self) -> list[HostLink]:
        """Each participant's link as this host measured it, about once a second.

        Dicts of ``participant``, ``heartbeats``, ``rtt_ms``,
        ``last_heartbeat_ms``, ``data_path`` (a robot's realtime parts:
        ``"tcp"`` or ``"udp"``), ``chunks_accepted``, ``chunk_timing`` (per
        commanded part, by ``robot:path``: ``received``, ``accepted``,
        ``lost``, ``late_ms``, ``late_max_ms``, ``gap_max_ms``, as in
        :meth:`Robot.edge_stats`), ``datagram_board`` / ``datagram_host``
        (each end of a board's datagram path, or ``None``), ``path_time``
        (a robot offered the datagram path: ``udp_s`` and ``tcp_s`` its
        realtime parts rode each path since it attached, and the host's
        ``fallbacks`` and ``returns`` between them; else ``None``) and
        ``datagram_relay`` (how long the host took to relay its chunks,
        subscriber to socket: ``ms``, counts in buckets ``[0, 1)``,
        ``[1, 2)``, ``[2, 4)`` ... ``[64, inf)`` ms, and ``max_ms``; else
        ``None``). Empty for a session another process hosts.
        """
        return cast("list[HostLink]", self._native.health())

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
        _close_leaving(self.close, exc)


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


def _plain(value: Any) -> Any:
    """A NumPy value or a ``Safe`` as plain JSON data (``json.dumps``'s default)."""
    if isinstance(value, np.ndarray | np.generic):
        return cast("Any", value).tolist()
    if isinstance(value, parts.Safe):
        return value.name
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _json(value: Any) -> str:
    """A map, ``info`` or reset ``options`` as JSON; NumPy values become lists and numbers."""
    return json.dumps(value, default=_plain)


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


def _sample(
    owner: str,
    parts: Mapping[str, tuple[_Field, ...]],
    path: str,
    value: object,
    fields: Mapping[str, object],
) -> list[str | memoryview]:
    """One sample of ``owner``'s part ``path`` as its fields' buffers, in declaration order."""
    declared = parts.get(path)
    if declared is None:
        raise ValueError(
            f"{owner} has no part {path!r} (its parts: {', '.join(parts)})"
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
    return buffers


class Clock(enum.Enum):
    """Who owns a robot's time: the world, or whoever steps it.

    The robot chooses, where it is declared; a policy or env learns it from
    what it connects to.
    """

    #: The world runs whether anyone acts or not: the robot publishes what
    #: it measures and plays what its edge says is due, at its own rate. A
    #: :class:`Policy` drives it. The default.
    WALL = "wall"
    #: Time advances one step per call of the robot's ``step(commands)``,
    #: and waits for it: :meth:`Robot.serve` serves the steps, and
    #: ``RemoteEnv`` at :attr:`Robot.address` drives them as a Gymnasium env.
    LOCKSTEP = "lockstep"


class Robot:
    """One robot: its map declared, its parts published, its commands served.

    ``parts`` is the robot map: ``{path: part}``, each part built with
    :mod:`~rlmesh.realtime.parts` or written as its dict. The host
    normalises it; every commandable part gets an edge. The robot's control
    loop publishes its sensors and asks :meth:`due` what to dispatch on each
    control tick; :meth:`part` gives a handle that does the same for one
    part. ``on_handover(group, from_path, to_path)`` runs when a command
    group's lease moves between members (switch the device's control mode
    there); the group dispatches nothing until it returns.

    ``clock`` is a :class:`Clock` (or its value, ``"wall"`` or
    ``"lockstep"``). A lockstep robot publishes inside its ``step`` and is
    served with :meth:`serve`.

    When the host drops the robot (a network gap past its presence grace)
    it joins again on its own; meanwhile its edge holds, then goes safe.
    The host has it wait for :meth:`rearm` unless ``rlmesh host --rearm-robots``.
    """

    def __init__(
        self,
        name: str,
        parts: Mapping[str, Any],
        *,
        session: Host | str = "",
        timeout: float = 10.0,
        on_handover: Callable[[str, str, str], None] | None = None,
        clock: Clock | Literal["wall", "lockstep"] = Clock.WALL,
    ) -> None:
        try:
            clock = Clock(clock)
        except ValueError:
            raise ValueError(f"clock={clock!r}: Clock.WALL or Clock.LOCKSTEP") from None
        self._timeout = timeout
        self._lockstep: V2LockstepRobot | None = None
        if clock is Clock.LOCKSTEP:
            self._lockstep = V2LockstepRobot(_address(session), name, _json(parts))
            self._native = cast("V2Robot", self._lockstep)
        else:
            self._native = V2Robot(_address(session), name, _json(parts), timeout)
        self._adopt(on_handover)

    def _adopt(self, on_handover: Callable[[str, str, str], None] | None) -> None:
        self._on_handover = on_handover
        self._fields: dict[str, tuple[_Field, ...]] = {}
        self._kinds: dict[str, str] = {}
        self._commanded: set[str] = set()
        self._jpeg: set[str] = set()
        for part in self._native.parts():
            path = cast("str", part["path"])
            self._kinds[path] = cast("str", part["kind"])
            self._fields[path] = _fields(
                cast("list[tuple[str, str, list[int]]]", part["fields"])
            )
            if part.get("command") is not None:
                self._commanded.add(path)
            if part.get("encoding") == "jpeg":
                self._jpeg.add(path)

    @property
    def name(self) -> str:
        """The robot's participant id."""
        return self._native.name

    @property
    def address(self) -> str:
        """The robot's participant address, with its host's real port.

        It carries its session's key: a credential, as :attr:`Host.address`
        says.
        """
        return self._native.address

    @property
    def parts(self) -> dict[str, str]:
        """Each part's kind, by path, in authored order."""
        return dict(self._kinds)

    @property
    def mode(self) -> str:
        """The session mode (``disarmed``, ``armed``, ``active``, ``safe``, ``estop``)."""
        return self._native.mode

    @property
    def mode_reason(self) -> str:
        """Why the mode is what it is (``robot_lost``, ``estop_local``, ...; ``unspecified`` before any change)."""
        return self._native.mode_reason

    @property
    def held(self) -> list[Held]:
        """Every robot held until re-armed by name: ``robot``, ``reason``, ``detail``, ``at_ns``, ``message``."""
        return cast("list[Held]", self._native.held)

    def part(self, path: str) -> RobotPart:
        """A handle on the part at ``path``: its own :meth:`~RobotPart.publish`, :meth:`~RobotPart.due` and :meth:`~RobotPart.edge_stats`.

        Raises ``ValueError`` at once for a path the robot does not have.
        """
        if path not in self._kinds:
            raise ValueError(
                f"{self.name} has no part {path!r} (its parts: {', '.join(self._kinds)})"
            )
        return RobotPart(self, path)

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
        buffers = _sample(self.name, self._fields, path, value, fields)
        if self._lockstep is not None:
            self._lockstep.publish(path, buffers)
            return None
        return self._native.publish(path, buffers, capture_ns)

    def publish_jpeg(
        self,
        path: str,
        jpeg: bytes | bytearray | memoryview,
        /,
        *,
        capture_ns: int | None = None,
    ) -> None:
        """Publish one camera frame already encoded as JPEG, sent as it is.

        For a camera that hands out JPEG (a Pi camera, a USB webcam in MJPEG
        mode): no decode, no re-encode. ``path`` is a camera declared
        ``encoding="jpeg"``; the frame's header must give its declared width
        and height. The same rate ceiling and bitrate budget apply as to
        :meth:`publish`; never blocks on the network.
        """
        if self._lockstep is not None:
            raise TypeError("publish_jpeg is for a wall-clock robot")
        self._native.publish_jpeg(path, jpeg, capture_ns)

    def serve(
        self,
        step: Callable[[dict[str, Any]], Any],
        reset: Callable[[int | None, dict[str, Any] | None], Any] | None = None,
        *,
        background: bool = False,
        timeout: float = 10.0,
    ) -> None:
        """Serve a lockstep robot's steps: one ``step(commands)`` per step.

        ``commands`` maps each commanded part to its command (the array of a
        single-field command, else a dict of fields), validated and clamped
        to the part's ``limits`` as a wall-clock edge would; a step with an
        invalid command (a NaN) is refused before ``step`` sees it. Inside ``step`` and
        ``reset(seed, options)`` the robot :meth:`publish`es its parts; each
        returns ``None`` or ``(reward, terminated, truncated[, info])``
        (``reset`` may return just ``info``). Declares and arms first; with
        ``background``, returns once serving (stop with :meth:`close`).
        Whoever drives it (``RemoteEnv`` at :attr:`address`) owns its time.
        """
        if self._lockstep is None:
            raise RuntimeError(
                f"{self.name} runs on the wall clock: Robot(..., clock=Clock.LOCKSTEP)"
            )

        def outcome(result: Any) -> tuple[float, bool, bool, str]:
            if result is None:
                return (0.0, False, False, "")
            if isinstance(result, Mapping):
                return (0.0, False, False, _json(result))
            reward, terminated, truncated, *rest = result
            info = rest[0] if rest else None
            return (
                float(reward),
                bool(terminated),
                bool(truncated),
                "" if info is None else _json(info),
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

    def serve_lockstep(
        self,
        step: Callable[[dict[str, Any]], Any],
        reset: Callable[[int | None, dict[str, Any] | None], Any] | None = None,
        *,
        background: bool = False,
        timeout: float = 10.0,
    ) -> None:
        """:meth:`serve`'s former name: the same call."""
        self.serve(step, reset, background=background, timeout=timeout)

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
        """DISARMED or SAFE to ARMED, in a fresh epoch; nothing when already armed.

        It clears no robot's hold: see :meth:`rearm`.
        """
        self._native.arm()

    def rearm(self, robot: str) -> None:
        """Let ``robot``, stopped on its own, take leases again.

        A robot that faulted, restarted or declared commandable parts while
        the session was armed is held until it is re-armed by name. The
        session must already be armed; a robot that is not held is left as
        it is.
        """
        self._native.rearm(robot)

    def disarm(self) -> None:
        """Back to DISARMED."""
        self._native.disarm()

    def keep(self, label: str = "keep") -> None:
        """Ask the host to keep its recording of the last minutes as an incident named ``label``.

        Kept out of the rolling purge's reach, like the cockpit's Keep. At
        most every 5 s; a host that records nothing refuses it.
        """
        self._native.keep(label)

    def estop(self, reason: str = "estop") -> None:
        """Emergency stop, latched here with no network (any thread, a button handler too).

        The next :meth:`due` purges every queue and returns ``estop``, and
        the host is told in the background. Latched until
        :meth:`clear_estop`, then disarm and arm.
        """
        self._native.estop(reason)

    def clear_estop(self) -> None:
        """Clear the local e-stop latch and tell the host (after any unsent e-stop); then disarm and arm."""
        self._native.clear_estop()

    def report_safe(self, reason: str) -> None:
        """Tell the host the robot fell back on its own (a device watchdog fired)."""
        self._native.report_safe(reason)

    def now_ns(self) -> int:
        """This robot's clock: ``time.monotonic_ns()``."""
        return self._native.now_ns()

    def health(self, *, wait: bool = False, timeout: float | None = None) -> Health:
        """The session's live link health, as the host last published it (once a second).

        The same as :meth:`Policy.health`, read through this robot's link.
        """
        if self._lockstep is not None:
            raise SessionException(
                f"robot {self.name} runs lockstep: its session has no live link health"
            )
        return cast(
            "Health",
            self._native.health(wait, self._timeout if timeout is None else timeout),
        )

    def edge_stats(self, path: str) -> EdgeStats:
        """The counters of the edge serving ``path``.

        Accepted, rejected (and ``rejected_by`` reason), mode ticks, and
        ``clock_refused``: while its clock error refuses chunks, a dict of
        ``part``, ``participant``, ``error_ms``, ``bound_ms`` and ``message``.
        How ``path``'s chunks arrived: ``received`` (any outcome), ``lost``
        (gaps in their seqs under one lease), ``late_ms`` (accepted chunks
        by lateness on arrival against their first slot, in buckets of
        ``[0, 1)``, ``[1, 2)``, ``[2, 4)`` ... ``[64, inf)`` ms),
        ``late_max_ms`` and ``gap_max_ms`` (the longest time between two
        accepted chunks).
        """
        return cast("EdgeStats", self._native.edge_stats(path))

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
        _close_leaving(self.close, exc)


class RobotPart:
    """One part of a :class:`Robot`, by path (:meth:`Robot.part`).

    The robot's per-part calls without the path::

        servos = robot.part("servos")
        servos.publish(pos=q)
        due = servos.due()

    A call the part cannot take (``due`` on a part that takes no commands,
    ``publish_jpeg`` on one that is not a JPEG camera) raises ``TypeError``.
    """

    __slots__ = ("_robot", "path")

    def __init__(self, robot: Robot, path: str) -> None:
        self._robot = robot
        #: The part's path in its robot's map.
        self.path: str = path

    @property
    def kind(self) -> str:
        """The part's kind (``joints``, ``camera``, ``ns/name``, ...)."""
        return self._robot._kinds[self.path]  # pyright: ignore[reportPrivateUsage]

    @property
    def commanded(self) -> bool:
        """Whether the part takes commands (it declared a ``command`` mode)."""
        return self.path in self._robot._commanded  # pyright: ignore[reportPrivateUsage]

    def publish(
        self,
        value: object = None,
        /,
        *,
        capture_ns: int | None = None,
        **fields: object,
    ) -> int | None:
        """Publish one sample, as :meth:`Robot.publish` does."""
        return self._robot.publish(self.path, value, capture_ns=capture_ns, **fields)

    def publish_jpeg(
        self, jpeg: bytes | bytearray | memoryview, /, *, capture_ns: int | None = None
    ) -> None:
        """Publish one frame already encoded as JPEG, as :meth:`Robot.publish_jpeg` does."""
        if self.path not in self._robot._jpeg:  # pyright: ignore[reportPrivateUsage]
            raise TypeError(
                f"{self.path} is not a camera declared encoding='jpeg': publish() its frames"
            )
        self._robot.publish_jpeg(self.path, jpeg, capture_ns=capture_ns)

    def due(self) -> Due | None:
        """Tick this part's edge: what to dispatch now, as :meth:`Robot.due` does."""
        self._commands("due")
        return self._robot.due(self.path)

    def edge_stats(self) -> EdgeStats:
        """The counters of this part's edge, as :meth:`Robot.edge_stats` gives them."""
        self._commands("edge_stats")
        return self._robot.edge_stats(self.path)

    def _commands(self, call: str) -> None:
        if not self.commanded:
            raise TypeError(
                f"{self.path} takes no commands: {call}() is for a commanded part"
            )
        if self._robot._lockstep is not None:  # pyright: ignore[reportPrivateUsage]
            raise TypeError(
                f"{self.path}: {call}() is for a wall-clock robot; a lockstep robot "
                "gets its commands in step()"
            )

    def __repr__(self) -> str:
        return f"RobotPart({self._robot.name}:{self.path}, {self.kind})"


class Meta:
    """A snapshot's metadata (``obs.meta``): ages, staleness and provenance."""

    __slots__ = (
        "ages_ns",
        "based_on",
        "blank",
        "padded",
        "seqs",
        "skew_ns",
        "stale",
        "time_ns",
    )

    def __init__(self, native: V2Observation) -> None:
        #: Each input's conservative age at the snapshot, by entry name
        #: (``None``: blank).
        self.ages_ns: dict[str, int | None] = native.ages
        #: Each input's seq range delivered, ``(seq_first, seq_last)``, by
        #: entry name (``None``: blank). Entry names hold across a rebind;
        #: stream ids do not.
        self.seqs: dict[str, tuple[int, int] | None] = native.seqs
        #: Inputs delivered stale under ``allow: hold``.
        self.stale: frozenset[str] = frozenset(native.stale)
        #: Window slots padded by ``allow: warmup``, by entry.
        self.padded: dict[str, int] = native.padded
        #: Inputs bound to a constant (``allow: blank``).
        self.blank: frozenset[str] = frozenset(native.blank)
        #: The newest capture minus the oldest, across inputs.
        self.skew_ns: int = native.skew_ns
        #: Per stream, ``(stream id, seq_first, seq_last)``: the chunk's
        #: ``based_on``, by the ids of the binding in force (:attr:`Policy.binding`).
        self.based_on: list[tuple[int, int, int]] = native.based_on
        #: The session time the snapshot was taken at.
        self.time_ns: int = native.time_ns


class Observation(Mapping[str, Any]):
    """What ``predict`` receives: each input by entry name, plus ``meta``.

    Arrays are read-only and stay valid while the observation lives; text
    inputs are ``str``; several fields that are not one concatenable array
    arrive as a dict.
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

    def age_ms(self, name: str) -> float | None:
        """How old input ``name`` was at the snapshot, in ms; ``None`` for a blank input.

        The conservative age :attr:`Meta.ages_ns` holds, in milliseconds.
        Raises ``KeyError`` for a name the policy has no input for.
        """
        ages = self.meta.ages_ns
        if name not in ages:
            raise KeyError(
                f"no input {name!r} (the inputs: {', '.join(ages) or 'none'})"
            )
        age = ages[name]
        return None if age is None else age / 1e6

    def __repr__(self) -> str:
        return f"Observation({', '.join(self._values)})"


class RunMode(enum.Enum):
    """What :meth:`Policy.run` does with a start it cannot command.

    A start cannot command while the policy holds none of its leases: the
    session is not armed, a higher class preempted it, or its robot is held.
    """

    #: The start is skipped (``no_lease``): ``predict`` runs only while the
    #: policy can command. The default.
    COMMAND = "command"
    #: ``predict`` runs on the same rate and inputs, and its actions are
    #: converted and checked as for sending, then dropped: nothing reaches a
    #: robot. The run commands from the first start whose inputs were read
    #: with a lease held, without a restart.
    SHADOW = "shadow"


class Resume(enum.Enum):
    """What :meth:`Policy.run` does once a higher class took a part from it and let go.

    A part is taken when a participant of a higher priority class preempts
    the policy (an operator jumping in). Losing a lease for any other reason
    (its own lapse, a disarm, its robot held or restarting) never parks it:
    it is asked for again as soon as it can be.
    """

    #: The part stays parked: the policy does not take it back when the
    #: holder releases it, detaches or dies, until :meth:`Policy.resume`,
    #: or until the holder hands it back (``release(to="model")``). The
    #: robot's part goes to its safe behaviour meanwhile. The default.
    MANUAL = "manual"
    #: The part is asked for again as soon as it is free: the robot moves
    #: again the moment its holder lets go, however it lets go (its
    #: process dying too). Prefer the default and an explicit hand-back.
    AUTO = "auto"


class Step:
    """A completed start, as ``on_step`` receives it (:meth:`Policy.run`)."""

    __slots__ = ("actions", "parts", "seqs", "shadow", "time_ns")

    def __init__(self, native: Mapping[str, Any]) -> None:
        #: A shadow start: its actions were checked and not sent.
        self.shadow: bool = native["shadow"]
        #: The session time of the start's snapshot.
        self.time_ns: int = native["time_ns"]
        #: Each action entry's chunk as the part's edge receives it: one
        #: ``[chunk, ...]`` array per command field (``None``: a dropped entry).
        self.actions: dict[str, dict[str, Array] | None] = {
            entry: None if value is None else cast("dict[str, Array]", _value(value))
            for entry, value in native["actions"].items()
        }
        #: The part each action entry commands, ``robot:path`` (``None``: dropped).
        self.parts: dict[str, str | None] = native["parts"]
        #: Each entry's chunk seq when it was sent (``None``: not sent).
        self.seqs: dict[str, int | None] = native["seqs"]

    def __repr__(self) -> str:
        kind = "shadow" if self.shadow else "sent"
        return f"Step({kind}, {', '.join(self.actions)})"


class BindingTable:
    """A binding table: what each entry is bound to and how it is delivered."""

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
        return f"BindingTable({self.json['policy']!r} on {self.json['robot']!r}, {self.fingerprint})"


def _arity(predict: Callable[..., Any]) -> int:
    """2 for ``predict(obs, policy)``: two required positional parameters; else 1."""
    try:
        params = list(inspect.signature(predict).parameters.values())
    except (TypeError, ValueError):  # a builtin without a signature
        return 1
    ordered = (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
    positional = [p for p in params if p.kind in ordered]
    required = [p for p in positional if p.default is inspect.Parameter.empty]
    if len(required) > 2:
        raise TypeError(
            f"predict takes {len(required)} arguments: def predict(obs) or "
            "def predict(obs, policy)"
        )
    if len(required) == 2:
        return 2
    if not positional and not any(
        p.kind is inspect.Parameter.VAR_POSITIONAL for p in params
    ):
        raise TypeError(
            "predict takes no argument: def predict(obs) or def predict(obs, policy)"
        )
    return 1


class Policy:
    """A policy map and, once bound, the controller that runs it.

    The map is given by keyword, its entries built with
    :mod:`~rlmesh.realtime.inputs` and :mod:`~rlmesh.realtime.actions`::

        tracker = realtime.Policy(
            hz=15,
            inputs={"cam": inputs.camera("webcam:cam")},
            actions={
                "eye": actions.joints("eye-1:eyes", command="position", hz=50, chunk=4)
            },
            session="",
        )

    or as one dict, its first argument, with the same keys: ``hz`` (the
    replan rate), ``inputs``, ``actions`` and ``allow`` (conversions every
    entry may use). Passing both raises ``TypeError``.

    ``priority`` is the lease class it attaches with: ``model`` (default),
    ``operator`` (preempts models: teleop), ``safety`` or ``scripted``.

    ``timeout`` is how long :meth:`bind` waits for the robots to join, and
    :meth:`acquire` and a hand-back wait for what they need; ``None`` waits
    for as long as it takes (Ctrl-C interrupts it).

    ``publishes`` declares parts of the policy's own to :meth:`publish`
    annotations on (a detector's boxes, a tracker's target), built with
    :mod:`~rlmesh.realtime.parts`, none commandable:
    ``{"target": parts.custom("x/target", hz=30, fields={"xy":
    parts.array("float32", 2)})}``. They appear under the policy's name
    (``tracker/target``): recorded, shown in the cockpit, readable by other
    policies (:func:`rlmesh.realtime.inputs.of`), and never bound to this one.
    """

    def __init__(
        self,
        policy: Mapping[str, Any] | None = None,
        *,
        hz: float | None = None,
        inputs: Mapping[str, Any] | None = None,
        actions: Mapping[str, Any] | None = None,
        allow: Sequence[Conversion] | None = None,
        session: Host | str,
        name: str = "policy",
        priority: str = "model",
        timeout: float | None = 10.0,
        publishes: Mapping[str, Any] | None = None,
    ) -> None:
        keywords = {"hz": hz, "inputs": inputs, "actions": actions, "allow": allow}
        given = [key for key, value in keywords.items() if value is not None]
        if policy is not None and given:
            raise TypeError(
                "Policy takes the policy map as one dict or as keywords, not both: "
                f"drop {'=, '.join(given)}= or the dict"
            )
        if policy is None:
            if hz is None:
                raise TypeError(
                    "Policy needs hz= with inputs= or actions= (or the map as one dict)"
                )
            if inputs is None and actions is None:
                raise TypeError("Policy needs inputs=, actions= or both")
            policy = {
                key: value for key, value in keywords.items() if value is not None
            }
        self._policy_json = _json(policy)
        self._publishes_json = _json(publishes) if publishes else None
        self._address = _address(session)
        self._name = name
        self._priority = priority
        self._timeout = timeout
        self._native: V2Controller | None = None
        self._fields: dict[str, tuple[_Field, ...]] = {}

    @property
    def name(self) -> str:
        """The policy's participant id."""
        return self._name

    def bind(
        self,
        robot: str | None = None,
        *,
        under: str | None = None,
        at: Mapping[str, str] | None = None,
        names: Mapping[str, str] | None = None,
        allow: Sequence[Conversion] | None = None,
        timeout: float | None | Literal["policy"] = "policy",
        **captures: str,
    ) -> BindingTable:
        """Attach, bind to ``robot``'s parts and declare the binding.

        With no ``robot`` it binds the session's view of every robot, each
        under its name, so one policy drives several: its actions name
        their robot (``"at": "eye-1:eyes"``).
        ``under`` scopes relative selectors, ``at`` overrides an entry's
        path, ``names={"entry": "positional"}`` binds joints by index,
        ``allow`` adds conversions for every entry, and keyword captures
        fill ``{name}`` segments. Raises :class:`BindingRefused` with the
        binder's explanation, at once; a robot that has not appeared yet is
        waited for, up to ``timeout`` (then ``TimeoutError``, naming it):
        ``"policy"`` (default) is the policy's, a number overrides it for
        this bind alone (:meth:`acquire` and a hand-back keep the policy's),
        ``None`` waits for as long as it takes.
        """
        if self._native is not None:
            raise SessionException(f"policy {self._name} is already bound")
        if isinstance(timeout, str) and timeout != "policy":
            raise ValueError(
                f'timeout={timeout!r}: give seconds, None or "policy" (its own)'
            )
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
            timeout_s=self._timeout if timeout == "policy" else timeout,
            wait_timeout_s=self._timeout,
            publishes_json=self._publishes_json,
        )
        self._fields = {
            cast("str", part["path"]): _fields(
                cast("list[tuple[str, str, list[int]]]", part["fields"])
            )
            for part in self._native.published_parts()
        }
        return BindingTable(self._native)

    def publish(
        self,
        path: str,
        value: object = None,
        /,
        *,
        capture_ns: int | None = None,
        **fields: object,
    ) -> int | None:
        """Publish one sample of one of the policy's own parts (``publishes``), after :meth:`bind`.

        Takes values the way :meth:`Robot.publish` does, and never blocks
        on the network. Returns the sample's seq, or ``None`` when it was
        coalesced.
        """
        native = self._bound()
        if not self._fields:
            raise ValueError(f"policy {self._name} publishes nothing: pass publishes=")
        buffers = _sample(self._name, self._fields, path, value, fields)
        return native.publish(path, buffers, capture_ns)

    def _bound(self) -> V2Controller:
        if self._native is None:
            raise SessionException(
                f"policy {self._name} is not bound: call bind() first"
            )
        return self._native

    @property
    def binding(self) -> BindingTable:
        """The binding table in force (after :meth:`bind`): after a ``rebound`` event, the new one."""
        return BindingTable(self._bound())

    def observe(self) -> Observation:
        """A policy start now; raises :class:`StartSkipped` when one is skipped."""
        return Observation(self._bound().observe())

    def send(
        self, actions: Mapping[str, Any], observation: Observation
    ) -> dict[str, int | None]:
        """Send one timed chunk per action entry, computed from ``observation``.

        ``actions`` maps each action entry to its ``[chunk, ...]`` array
        (or a dict of arrays). Returns each entry's chunk seq, or ``None``
        when this policy does not hold the part's lease.
        """
        native = observation._native  # pyright: ignore[reportPrivateUsage]
        return self._bound().send(_actions(actions), native)

    def acquire(self) -> dict[str, int]:
        """Take the binding's leases now (preempting a lower class), then keep them.

        Waits, up to the policy's ``timeout``, for the host to measure the
        clocks an edge needs, so a chunk sent right after is not rejected
        (``TimeoutError`` naming whose is missing). Returns the lease token
        per part, keyed ``robot:path`` (``{"ur10e:arm": token}``); raises :class:`SessionRefused` (``status`` ``LEASE_BUSY``,
        ``GROUP_BUSY``, ``NOT_READY``, ...) when they cannot be granted.
        """
        return dict(self._bound().acquire())

    def release(self, *, to: str | None = None) -> list[HandBack] | None:
        """Give the leases back; they are not re-requested until acquire() or run().

        ``to="model"`` hands them back to the model this policy took the
        parts from (an operator letting go of a robot it took over by
        :meth:`acquire`): that policy resumes those parts, and only those,
        as if its own :meth:`resume` had been called, and reports a
        ``handed_back`` event. A plain release leaves it parked. The hand-back
        waits, up to the policy's ``timeout``, for the model to hold the parts
        again, and returns one dict per policy handed back to: ``to`` (its
        name), ``parts`` (``robot:path``) and ``taken`` (whether it held them
        again in time).

        Between the release and the model's first chunk the robot does what it
        does after any release: a position part holds its last target, any
        other runs its safe behaviour, for at most the part's ``hold_max``;
        the model's new lease then holds it, untimed, until its first chunk.

        Raises :class:`SessionRefused`, releasing nothing, when a part was taken
        from nobody (no model is parked on it) or from another class, when
        its robot is held (a hand-back never clears a hold), when the model's
        process or the session host is too old to resume on a hand-back
        (``status`` ``INCOMPATIBLE``), or when this policy holds nothing.
        """
        native = self._bound()
        if to is None:
            native.release()
            return None
        return cast("list[HandBack] | None", native.release(to))

    @property
    def holds(self) -> bool:
        """Whether this policy holds every lease its binding commands."""
        return self._bound().holds()

    def run(
        self,
        predict: Predict,
        *,
        duration: float | None = None,
        mode: RunMode = RunMode.COMMAND,
        on_step: Callable[[Step], object] | None = None,
        resume: Resume = Resume.MANUAL,
    ) -> PolicyStats:
        """Run the policy loop on this thread; returns the run's counters (:meth:`stats`).

        ``predict(obs)`` is called at most at the replan rate, one inference
        in flight; a start that falls while it runs is skipped, never
        queued. A ``predict`` with two required positional parameters is
        called as ``predict(obs, policy)``, with this policy (to
        :meth:`publish` from inside it); any other with one. The choice is
        made once, when the run starts. Leases are requested once the inputs are warm, re-requested
        after a lapse, a disarm or a re-arm, and released when the run
        returns.

        A part a higher class takes (an operator preempting the policy) is
        parked: the policy does not take it back when the operator releases
        it or their process dies; the robot's part goes to its safe
        behaviour. A ``parked`` event names the part and who took it, starts
        are skipped as ``parked``, and :meth:`stats` lists it under
        ``parked``, until :meth:`resume` (from any thread) or the operator's
        ``release(to="model")`` hands it back (a ``handed_back`` event).
        ``resume=Resume.AUTO`` takes it back as soon as it is free. If
        ``predict`` raises, the policy stops sending, releases its leases
        (the robot holds, then goes safe) and the exception propagates.
        Raises :class:`SessionLost` (``reason`` ``robot_lost`` or
        ``host_lost``) when the robot's heartbeats lapse or the host is
        gone. Stops after ``duration`` seconds or on :meth:`stop`.

        On a session that is not armed it waits for an operator to arm it,
        commanding nothing: it warns once each time (a ``waiting_to_arm``
        event), and :meth:`stats` says so under ``waiting``.

        When an input's publisher goes away (an ``input_lost`` event) the
        starts that need it are skipped as ``publisher_gone``; when it comes
        back, restarts or declares its parts again, the policy binds again on
        its own (a ``rebound`` event) and keeps the leases it holds. A part
        that comes back as something the policy cannot read raises
        :class:`BindingRefused`, naming it and what changed.

        With ``mode=RunMode.SHADOW`` a start that cannot command (the session
        not armed, preempted, its robot held) still calls ``predict``, so its
        own published parts keep flowing and a failing model surfaces at
        once; its actions are converted and checked, and nothing is sent.
        The run commands from the first start whose inputs were read with a
        lease held, and says so on each change (``shadowing`` and
        ``commanding`` events). :meth:`stats` counts ``shadow_predictions``.

        ``on_step(step)``, if given, is called after each completed start
        with a :class:`Step`: the actions as the edges receive them, marked
        ``shadow`` or with their chunk seqs. If it raises, the run ends as
        when ``predict`` raises.
        """
        native = self._bound()
        if not isinstance(mode, RunMode):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(f"mode must be a RunMode, not {mode!r}")
        if not isinstance(resume, Resume):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(f"resume must be a Resume, not {resume!r}")

        if _arity(predict) == 2:
            two = cast("Callable[[Observation, Policy], Mapping[str, Any]]", predict)

            def call(raw: V2Observation) -> dict[str, Any]:
                return _actions(two(Observation(raw), self))

        else:
            one = cast("Callable[[Observation], Mapping[str, Any]]", predict)

            def call(raw: V2Observation) -> dict[str, Any]:
                return _actions(one(Observation(raw)))

        def step(raw: dict[str, Any]) -> None:
            if on_step is not None:
                on_step(Step(raw))

        native.run(
            call, duration, mode.value, None if on_step is None else step, resume.value
        )
        return self.stats()

    def stop(self) -> None:
        """Ask a :meth:`run` in progress (on another thread) to return."""
        self._bound().stop()

    def health(self, *, wait: bool = False, timeout: float | None = None) -> Health:
        """The session's live link health, as the host last published it (once a second).

        The first call subscribes (nothing is received or decoded before
        it) and waits for the host's first publication; later calls return
        the newest at once, and ``wait=True`` waits for the next one. Raises
        ``TimeoutError`` past ``timeout`` seconds (default: the object's
        ``timeout``).

        Returns ``epoch``, ``host_ns`` (the host's clock when it published)
        and ``links``: one dict per attached participant, keyed by its
        name, as the host measures it. Durations are in ms, as in
        :meth:`Policy.stats` and :meth:`Robot.edge_stats`:

        - ``heartbeats``, ``rtt_ms`` (the latest heartbeat round trip;
          ``None`` before one), ``jitter_ms``, ``heartbeat_gap_max_ms``,
          ``last_heartbeat_age_ms``;
        - ``offset_ms`` and ``offset_error_ms``: its clock mapping, host
          minus participant, and the mapping's error bound (``None`` until
          mapped);
        - ``health``: what it reports of itself (``ok``, ``degraded``,
          ``unhealthy``, ``unknown``);
        - robots: ``last_chunk_accepted`` and ``last_chunk_applied`` (the
          newest chunk seq, 0 for none), ``queue_depth``, ``rejected`` and
          ``rejected_by`` (chunks refused since it attached, by reason, as
          :meth:`Robot.edge_stats` names them), ``clock_refused`` (one dict
          per part refusing for a clock error: ``part``, ``participant``,
          ``error_ms``, ``bound_ms``);
        - ``published``: per part it publishes (``robot:path``), samples
          ``published`` and ``coalesced``; ``last_seq_seen``: the newest seq
          it received per part it reads;
        - ``run``: a bound policy's counters from its latest report
          (``starts``, ``completions``, ``skipped`` by reason,
          ``chunks_sent``, ``pending_ms``, ``latency_p50_ms``,
          ``latency_p99_ms``, ``drifting``), else ``None``.
        """
        limit = self._timeout if timeout is None else timeout
        return cast("Health", self._bound().health(wait, limit))

    def resume(self) -> list[str]:
        """Hand the parts a higher class took back to this policy; returns them (``robot:path``).

        Safe to call from any thread while :meth:`run` is in progress on
        another. The policy asks for them again: at once when they are free,
        or once their holder lets go.
        """
        return self._bound().resume()

    def stats(self) -> PolicyStats:
        """Starts, completions, skips by reason, chunks sent and latency.

        ``aligned_to`` names the input whose publication the starts are
        phased just after (wall clock), ``phase_offset_ms`` how far.
        ``clock_refused`` says, per leased part whose edge refuses this
        policy's chunks for its clock, the measured error and the bound.
        ``waiting`` says what the policy waits for before it can command
        (the session not armed), or is ``None``. ``shadow_predictions``
        counts the starts a shadow run predicted without a lease; they are
        not ``no_lease`` skips. ``parked`` lists the parts a higher class
        took, ``robot:path``, kept from the policy until :meth:`resume` or a
        hand-back.
        """
        return cast("PolicyStats", self._bound().stats())

    @property
    def held(self) -> list[Held]:
        """Every robot held until re-armed by name: ``robot``, ``reason``, ``detail``, ``at_ns``, ``message``."""
        return cast("list[Held]", self._bound().held)

    def events(self) -> list[Event]:
        """Take the events reported so far: ``rate_drift``, ``inference_pending``, ``robot_held``, ``clock_refused``, ``waiting_to_arm``, ``shadowing``, ``commanding``, ``input_lost`` (with ``part``), ``rebound`` (with ``parts``), ``parked`` and ``handed_back`` (with ``part`` and ``by``)."""
        return cast("list[Event]", self._bound().events())

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
        _close_leaving(self.close, exc)


class SessionRecorder:
    """An MCAP recording of the session, attached as an observer.

    Records every robot's streams (``/{robot}/{path}``, with the edge's
    ``/{robot}/{path}/_applied``), every action chunk sent to an edge
    (``/{robot}/{path}/_chunks``), the authority and binding changes
    (``/_session/authority``, ``/_session/bindings``) and the host's health
    (``/_session/health``: every link, edge refusal and observer report),
    each envelope as it travelled. It declares no delivery contract: a stream a policy reads at
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
        _close_leaving(self.close, exc)


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


def drive(
    policy: Mapping[str, Any],
    robot: str | None,
    predict: Predict,
    *,
    session: Host | str,
    name: str = "policy",
    priority: str = "model",
    under: str | None = None,
    at: Mapping[str, str] | None = None,
    names: Mapping[str, str] | None = None,
    allow: Sequence[Conversion] | None = None,
    duration: float | None = None,
    mode: RunMode = RunMode.COMMAND,
    on_step: Callable[[Step], object] | None = None,
    timeout: float | None = 10.0,
    publishes: Mapping[str, Any] | None = None,
    resume: Resume = Resume.MANUAL,
    **captures: str,
) -> PolicyStats:
    """Bind ``policy`` to ``robot`` (``None``: every robot's view) and drive it with ``predict`` until ``duration``.

    Shorthand for :class:`Policy` + :meth:`Policy.bind` + :meth:`Policy.run`;
    ``predict`` may take the policy as its second argument, as for
    :meth:`Policy.run`.
    It keeps no handle: a part a higher class took stays parked for the
    rest of the run unless ``resume=Resume.AUTO``; use :class:`Policy` to
    :meth:`Policy.resume` one.
    """
    with Policy(
        policy,
        session=session,
        name=name,
        priority=priority,
        timeout=timeout,
        publishes=publishes,
    ) as p:
        p.bind(
            robot,
            under=under,
            at=at,
            names=names,
            allow=allow,
            timeout=timeout,
            **captures,
        )
        return p.run(
            predict, duration=duration, mode=mode, on_step=on_step, resume=resume
        )

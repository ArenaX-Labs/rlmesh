"""Typed constructors for robot map parts (Experimental).

Each returns the plain dict the robot map takes, with only the keys given,
so a map can mix them with dicts and the host still checks every rule::

    from rlmesh.realtime import parts

    EYE = {
        "eyes": parts.joints(
            ["pan", "tilt"],
            hz=50,
            fields=["pos"],
            command="position",
            safe=parts.Safe.HOLD,
            limits={"pos": [[-0.6, 0.6], [-0.4, 0.4]]},
        ),
    }
    CAMERA = {"cam": parts.camera(640, 480, hz=30, encoding="jpeg", quality=80)}

Keyword arguments every streaming kind takes: ``send`` (the publication
policy, ``"every"``, ``"latest(hz=30)"``, ``"latest(max_mbps=20)"``),
``priority``, ``doc`` and ``max_age_ms``. A commandable kind also takes the
command keys: ``command`` (the mode), ``control_hz``, ``safe`` (the
behaviour to fall back to, a :class:`Safe`; the host refuses a commanded
part without one), ``hold_max_ms``, ``max_action_age_ms``, ``latch``
(keep applying the last target for as long as the lease is held; only on a
mode that holds its target, and on by default for ``switch`` and ``value``),
``limits`` (``{field: [lo, hi]}``, a pair per element, or
``{"max_norm": x}``), ``merge`` and ``group``. Units are SI unless
``units`` says otherwise.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, cast

__all__ = [
    "Safe",
    "camera",
    "custom",
    "depth",
    "gripper",
    "hand",
    "imu",
    "joints",
    "pose",
    "reward",
    "success",
    "switch",
    "tactile",
    "text",
    "value",
    "velocity",
    "wrench",
]

#: A part of the robot map, as the host takes it.
Part = dict[str, Any]
#: Limits per field: ``[lo, hi]``, one pair per element, or ``{"max_norm": x}``.
Limits = Mapping[str, Any]


_IDENTIFIER = re.compile(r"[a-z][a-z0-9_]{0,63}")


class Safe:
    """What a commanded part does when no one is commanding it.

    ``Safe.STOP`` and ``Safe.HOLD`` are the common behaviours. Any other is
    one the robot's own firmware implements: ``Safe.custom("park")``. RLMesh
    passes the name to the robot and does not interpret it.
    """

    STOP: ClassVar[Safe]
    HOLD: ClassVar[Safe]

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        if not _IDENTIFIER.fullmatch(name):
            raise ValueError(
                f"safe behaviour {name!r}: 1 to 64 characters of [a-z][a-z0-9_]*"
            )
        self.name = name

    @classmethod
    def custom(cls, name: str) -> Safe:
        """A behaviour the robot's firmware implements, by ``name``."""
        if name in ("stop", "hold"):
            raise ValueError(f"{name!r} is built in: use Safe.{name.upper()}")
        return cls(name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Safe) and other.name == self.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __repr__(self) -> str:
        if self.name in ("stop", "hold"):
            return f"Safe.{self.name.upper()}"
        return f"Safe.custom({self.name!r})"


Safe.STOP = Safe("stop")
Safe.HOLD = Safe("hold")


def _part(kind: str, **keys: Any) -> Part:
    part: Part = {"kind": kind}
    for key, value in keys.items():
        if value is None:
            continue
        if key == "safe":
            if not isinstance(value, Safe):
                raise TypeError(
                    f"safe must be a Safe (Safe.STOP, Safe.HOLD or "
                    f"Safe.custom(...)), not {value!r}"
                )
            part[key] = value.name
        elif isinstance(value, Mapping):
            part[key] = dict(cast("Mapping[str, Any]", value))
        elif isinstance(value, Sequence) and not isinstance(value, str):
            part[key] = list(cast("Sequence[Any]", value))
        else:
            part[key] = value
    return part


def camera(
    width: int,
    height: int,
    *,
    hz: float,
    encoding: str | None = None,
    quality: int | None = None,
    format: str | None = None,
    frame: str | None = None,
    fov_deg: float | None = None,
    intrinsics: Sequence[float] | None = None,
    mount: str | None = None,
    send: str | None = None,
    priority: str | None = None,
    doc: str | None = None,
    max_age_ms: float | None = None,
) -> Part:
    """A camera: ``image: uint8 [height, width, 3]`` (``format="mono8"``: ``[..., 1]``).

    ``encoding="jpeg"`` (at ``quality``, default 80) sends JPEG frames: the
    robot encodes what it publishes, or publishes frames already encoded
    (:meth:`rlmesh.realtime.Robot.publish_jpeg`).
    """
    return _part(
        "camera",
        width=width,
        height=height,
        hz=hz,
        encoding=encoding,
        quality=quality,
        format=format,
        frame=frame,
        fov_deg=fov_deg,
        intrinsics=intrinsics,
        mount=mount,
        send=send,
        priority=priority,
        doc=doc,
        max_age_ms=max_age_ms,
    )


def depth(
    width: int,
    height: int,
    *,
    hz: float,
    dtype: str | None = None,
    range: Sequence[float] | None = None,
    frame: str | None = None,
    fov_deg: float | None = None,
    intrinsics: Sequence[float] | None = None,
    send: str | None = None,
    priority: str | None = None,
    doc: str | None = None,
    max_age_ms: float | None = None,
) -> Part:
    """A depth camera: ``depth: float32 [height, width]`` in m (``dtype="uint16"``: mm)."""
    return _part(
        "depth",
        width=width,
        height=height,
        hz=hz,
        dtype=dtype,
        range=range,
        frame=frame,
        fov_deg=fov_deg,
        intrinsics=intrinsics,
        send=send,
        priority=priority,
        doc=doc,
        max_age_ms=max_age_ms,
    )


def _joint_set(
    kind: str,
    names: Sequence[str] | None,
    count: int | None,
    **keys: Any,
) -> Part:
    return _part(kind, names=names, count=count, **keys)


def joints(
    names: Sequence[str] | None = None,
    *,
    hz: float,
    count: int | None = None,
    fields: Sequence[str] | None = None,
    command: str | None = None,
    prismatic: bool | None = None,
    dtype: str | None = None,
    units: Mapping[str, str] | None = None,
    range: Mapping[str, Any] | None = None,
    gains: Mapping[str, float] | None = None,
    control_hz: float | None = None,
    safe: Safe | None = None,
    hold_max_ms: float | None = None,
    max_action_age_ms: float | None = None,
    latch: bool | None = None,
    limits: Limits | None = None,
    merge: str | None = None,
    group: str | None = None,
    send: str | None = None,
    priority: str | None = None,
    doc: str | None = None,
    max_age_ms: float | None = None,
) -> Part:
    """A set of joints, by ``names`` (or ``count``): ``pos``, ``vel``, ``effort``.

    ``fields`` picks which it publishes (default ``pos`` and ``vel``);
    ``command`` makes it commandable: ``position``, ``velocity`` or
    ``torque``.
    """
    return _joint_set(
        "joints",
        names,
        count,
        fields=fields,
        hz=hz,
        command=command,
        prismatic=prismatic,
        dtype=dtype,
        units=units,
        range=range,
        gains=gains,
        control_hz=control_hz,
        safe=safe,
        hold_max_ms=hold_max_ms,
        max_action_age_ms=max_action_age_ms,
        latch=latch,
        limits=limits,
        merge=merge,
        group=group,
        send=send,
        priority=priority,
        doc=doc,
        max_age_ms=max_age_ms,
    )


def hand(
    names: Sequence[str],
    *,
    hz: float,
    fields: Sequence[str] | None = None,
    command: str | None = None,
    prismatic: bool | None = None,
    dtype: str | None = None,
    units: Mapping[str, str] | None = None,
    range: Mapping[str, Any] | None = None,
    gains: Mapping[str, float] | None = None,
    control_hz: float | None = None,
    safe: Safe | None = None,
    hold_max_ms: float | None = None,
    max_action_age_ms: float | None = None,
    latch: bool | None = None,
    limits: Limits | None = None,
    merge: str | None = None,
    group: str | None = None,
    send: str | None = None,
    priority: str | None = None,
    doc: str | None = None,
    max_age_ms: float | None = None,
) -> Part:
    """A hand's joints, by ``names``: :func:`joints` under its own kind, so a hand policy never binds an arm."""
    return _joint_set(
        "hand",
        names,
        None,
        fields=fields,
        hz=hz,
        command=command,
        prismatic=prismatic,
        dtype=dtype,
        units=units,
        range=range,
        gains=gains,
        control_hz=control_hz,
        safe=safe,
        hold_max_ms=hold_max_ms,
        max_action_age_ms=max_action_age_ms,
        latch=latch,
        limits=limits,
        merge=merge,
        group=group,
        send=send,
        priority=priority,
        doc=doc,
        max_age_ms=max_age_ms,
    )


def wrench(
    *,
    frame: str,
    hz: float,
    point: Sequence[float] | None = None,
    command: str | None = None,
    dtype: str | None = None,
    units: Mapping[str, str] | None = None,
    range: Mapping[str, Any] | None = None,
    control_hz: float | None = None,
    safe: Safe | None = None,
    hold_max_ms: float | None = None,
    max_action_age_ms: float | None = None,
    limits: Limits | None = None,
    merge: str | None = None,
    group: str | None = None,
    send: str | None = None,
    priority: str | None = None,
    doc: str | None = None,
    max_age_ms: float | None = None,
) -> Part:
    """A force/torque sensor in ``frame``: ``force [3]`` N, ``torque [3]`` N*m (``command="wrench"``)."""
    return _part(
        "wrench",
        frame=frame,
        hz=hz,
        point=point,
        command=command,
        dtype=dtype,
        units=units,
        range=range,
        control_hz=control_hz,
        safe=safe,
        hold_max_ms=hold_max_ms,
        max_action_age_ms=max_action_age_ms,
        limits=limits,
        merge=merge,
        group=group,
        send=send,
        priority=priority,
        doc=doc,
        max_age_ms=max_age_ms,
    )


def imu(
    *,
    frame: str,
    hz: float,
    fields: Sequence[str],
    rotation: str | None = None,
    dtype: str | None = None,
    units: Mapping[str, str] | None = None,
    range: Mapping[str, Any] | None = None,
    send: str | None = None,
    priority: str | None = None,
    doc: str | None = None,
    max_age_ms: float | None = None,
) -> Part:
    """An IMU in ``frame`` publishing ``fields``: ``orientation``, ``ang_vel``, ``lin_acc``, ``gravity``."""
    return _part(
        "imu",
        frame=frame,
        hz=hz,
        fields=fields,
        rotation=rotation,
        dtype=dtype,
        units=units,
        range=range,
        send=send,
        priority=priority,
        doc=doc,
        max_age_ms=max_age_ms,
    )


def pose(
    *,
    frame: str,
    child: str,
    hz: float,
    rotation: str | None = None,
    reference: str | None = None,
    command: str | None = None,
    dtype: str | None = None,
    units: Mapping[str, str] | None = None,
    range: Mapping[str, Any] | None = None,
    control_hz: float | None = None,
    safe: Safe | None = None,
    hold_max_ms: float | None = None,
    max_action_age_ms: float | None = None,
    latch: bool | None = None,
    limits: Limits | None = None,
    merge: str | None = None,
    group: str | None = None,
    send: str | None = None,
    priority: str | None = None,
    doc: str | None = None,
    max_age_ms: float | None = None,
) -> Part:
    """The pose of ``child`` in ``frame``: ``pos [3]`` m, ``rot`` (``command``: ``absolute`` or ``delta``)."""
    return _part(
        "pose",
        frame=frame,
        child=child,
        hz=hz,
        rotation=rotation,
        reference=reference,
        command=command,
        dtype=dtype,
        units=units,
        range=range,
        control_hz=control_hz,
        safe=safe,
        hold_max_ms=hold_max_ms,
        max_action_age_ms=max_action_age_ms,
        latch=latch,
        limits=limits,
        merge=merge,
        group=group,
        send=send,
        priority=priority,
        doc=doc,
        max_age_ms=max_age_ms,
    )


def gripper(
    *,
    hz: float,
    command: str | None = None,
    width_m: Sequence[float] | None = None,
    binary: bool | None = None,
    dtype: str | None = None,
    units: Mapping[str, str] | None = None,
    range: Mapping[str, Any] | None = None,
    control_hz: float | None = None,
    safe: Safe | None = None,
    hold_max_ms: float | None = None,
    max_action_age_ms: float | None = None,
    latch: bool | None = None,
    limits: Limits | None = None,
    merge: str | None = None,
    group: str | None = None,
    send: str | None = None,
    priority: str | None = None,
    doc: str | None = None,
    max_age_ms: float | None = None,
) -> Part:
    """A gripper opening: ``pos [1]``, 0 closed to 1 open (``command="position"``)."""
    return _part(
        "gripper",
        hz=hz,
        command=command,
        width_m=width_m,
        binary=binary,
        dtype=dtype,
        units=units,
        range=range,
        control_hz=control_hz,
        safe=safe,
        hold_max_ms=hold_max_ms,
        max_action_age_ms=max_action_age_ms,
        latch=latch,
        limits=limits,
        merge=merge,
        group=group,
        send=send,
        priority=priority,
        doc=doc,
        max_age_ms=max_age_ms,
    )


def switch(
    *,
    hz: float,
    command: str | None = None,
    control_hz: float | None = None,
    safe: Safe | None = None,
    hold_max_ms: float | None = None,
    max_action_age_ms: float | None = None,
    latch: bool | None = None,
    merge: str | None = None,
    group: str | None = None,
    send: str | None = None,
    priority: str | None = None,
    doc: str | None = None,
    max_age_ms: float | None = None,
) -> Part:
    """An on/off part: ``on``, a bool (``command="set"`` drives it; without, a button)."""
    return _part(
        "switch",
        hz=hz,
        command=command,
        control_hz=control_hz,
        safe=safe,
        hold_max_ms=hold_max_ms,
        max_action_age_ms=max_action_age_ms,
        latch=latch,
        merge=merge,
        group=group,
        send=send,
        priority=priority,
        doc=doc,
        max_age_ms=max_age_ms,
    )


def value(
    *,
    hz: float,
    unit: str | None = None,
    range: Sequence[float] | None = None,
    command: str | None = None,
    control_hz: float | None = None,
    safe: Safe | None = None,
    hold_max_ms: float | None = None,
    max_action_age_ms: float | None = None,
    latch: bool | None = None,
    limits: Limits | None = None,
    merge: str | None = None,
    group: str | None = None,
    send: str | None = None,
    priority: str | None = None,
    doc: str | None = None,
    max_age_ms: float | None = None,
) -> Part:
    """One number: ``value``, a float32 in ``unit``, within ``range`` ``[lo, hi]`` (``command="set"`` drives it)."""
    return _part(
        "value",
        hz=hz,
        unit=unit,
        range=range,
        command=command,
        control_hz=control_hz,
        safe=safe,
        hold_max_ms=hold_max_ms,
        max_action_age_ms=max_action_age_ms,
        latch=latch,
        limits=limits,
        merge=merge,
        group=group,
        send=send,
        priority=priority,
        doc=doc,
        max_age_ms=max_age_ms,
    )


def velocity(
    *,
    frame: str,
    hz: float,
    dof: str | None = None,
    command: str | None = None,
    dtype: str | None = None,
    units: Mapping[str, str] | None = None,
    range: Mapping[str, Any] | None = None,
    control_hz: float | None = None,
    safe: Safe | None = None,
    hold_max_ms: float | None = None,
    max_action_age_ms: float | None = None,
    limits: Limits | None = None,
    merge: str | None = None,
    group: str | None = None,
    send: str | None = None,
    priority: str | None = None,
    doc: str | None = None,
    max_age_ms: float | None = None,
) -> Part:
    """A base twist in ``frame``: ``linear``, ``angular`` (``dof``: ``planar`` or ``spatial``; ``command="velocity"``)."""
    return _part(
        "velocity",
        frame=frame,
        hz=hz,
        dof=dof,
        command=command,
        dtype=dtype,
        units=units,
        range=range,
        control_hz=control_hz,
        safe=safe,
        hold_max_ms=hold_max_ms,
        max_action_age_ms=max_action_age_ms,
        limits=limits,
        merge=merge,
        group=group,
        send=send,
        priority=priority,
        doc=doc,
        max_age_ms=max_age_ms,
    )


def tactile(
    shape: Sequence[int],
    *,
    hz: float,
    layout: str | None = None,
    dtype: str | None = None,
    units: Mapping[str, str] | None = None,
    range: Mapping[str, Any] | None = None,
    send: str | None = None,
    priority: str | None = None,
    doc: str | None = None,
    max_age_ms: float | None = None,
) -> Part:
    """Taxels: ``values [shape]``, normalized."""
    return _part(
        "tactile",
        shape=shape,
        hz=hz,
        layout=layout,
        dtype=dtype,
        units=units,
        range=range,
        send=send,
        priority=priority,
        doc=doc,
        max_age_ms=max_age_ms,
    )


def text(*, priority: str | None = None, doc: str | None = None) -> Part:
    """Text events (``text: utf8``): a task, a label."""
    return _part("text", priority=priority, doc=doc)


def reward(*, priority: str | None = None, doc: str | None = None) -> Part:
    """Reward events (``reward: float32 []``)."""
    return _part("reward", priority=priority, doc=doc)


def success(*, priority: str | None = None, doc: str | None = None) -> Part:
    """Outcome events (``success: bool []``)."""
    return _part("success", priority=priority, doc=doc)


def custom(
    kind: str,
    *,
    hz: float,
    fields: Mapping[str, Any],
    command: str | None = None,
    command_fields: Mapping[str, Any] | None = None,
    frame: str | None = None,
    units: Mapping[str, str] | None = None,
    range: Mapping[str, Any] | None = None,
    control_hz: float | None = None,
    safe: Safe | None = None,
    hold_max_ms: float | None = None,
    max_action_age_ms: float | None = None,
    limits: Limits | None = None,
    merge: str | None = None,
    group: str | None = None,
    send: str | None = None,
    priority: str | None = None,
    doc: str | None = None,
    max_age_ms: float | None = None,
) -> Part:
    """A custom kind ``ns/name`` with declared ``fields`` (``{name: [dtype, shape]}``).

    A detector's boxes, say: ``custom("yolo/boxes", hz=30, fields={"xyxy":
    ["float32", [8, 4]], "score": ["float32", [8]]})``.
    """
    return _part(
        kind,
        hz=hz,
        fields=fields,
        command=command,
        command_fields=command_fields,
        frame=frame,
        units=units,
        range=range,
        control_hz=control_hz,
        safe=safe,
        hold_max_ms=hold_max_ms,
        max_action_age_ms=max_action_age_ms,
        limits=limits,
        merge=merge,
        group=group,
        send=send,
        priority=priority,
        doc=doc,
        max_age_ms=max_age_ms,
    )

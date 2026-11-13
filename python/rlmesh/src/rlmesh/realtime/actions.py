"""Typed constructors for a policy's actions (Experimental).

Each returns the plain dict the policy map takes, with only the keys given,
so a map can mix them with dicts and the host still checks every rule::

    from rlmesh.realtime import actions

    actions.joints("gimbal-left:servos", command="position", hz=50, chunk=4)
    # {"kind": "joints", "at": "gimbal-left:servos", "command": "position", "hz": 50, "chunk": 4}

The first argument is the part's address, the entry's ``at``, as for
:mod:`rlmesh.realtime.inputs`; leave it out to bind any part of the kind.
Only a kind a robot can be commanded through has an action.

Keyword arguments every action takes: ``command`` (the part's command
mode; the binding refuses a part in another), ``hz`` (the rate of the
actions in a chunk; default the policy's), ``chunk`` (actions per chunk,
default 1), ``t0`` (when a chunk's first action is due: ``"snapshot"``,
the default, at the observation it was computed from, or ``"send"``),
``ttl_ms`` (a cap on a chunk's life), ``optional`` (may stay unbound, its
chunks dropped; it needs ``"drop"`` in ``allow``, the entry's, the policy's
or the bind call's, or the binding is refused) and ``allow``. Numeric kinds take ``units``, ``scale``,
``offset`` and ``range`` as inputs do.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from ._entries import (
    Conversion,
    Entry,
    Rotation,
    choice,
    custom_kind,
    entry,
    field_schema,
    identifier,
    maybe,
)

__all__ = [
    "custom",
    "gripper",
    "hand",
    "joints",
    "pose",
    "switch",
    "value",
    "velocity",
    "wrench",
]

Start = Literal["snapshot", "send"]
JointCommand = Literal["position", "velocity", "torque"]


def _command(value: object, modes: Sequence[str]) -> str | None:
    return maybe(choice, "command", value, modes)


def _joint_set(
    kind: str, at: str | None, command: JointCommand | None, **keys: Any
) -> Entry:
    modes = ("position", "velocity", "torque")
    return entry(kind, at, command=_command(command, modes), **keys)


def joints(
    at: str | None = None,
    *,
    command: JointCommand | None = None,
    hz: float | None = None,
    chunk: int | None = None,
    names: Sequence[str] | Literal["positional"] | None = None,
    count: int | None = None,
    units: Mapping[str, str] | None = None,
    scale: float | Sequence[float] | None = None,
    offset: float | Sequence[float] | None = None,
    range: tuple[float, float] | None = None,
    t0: Start | None = None,
    ttl_ms: float | None = None,
    optional: bool | None = None,
    allow: Sequence[Conversion] | None = None,
) -> Entry:
    """Commands for a set of joints: ``position``, ``velocity`` or ``torque``.

    A chunk is ``[chunk, joints]``. ``names`` lists the model's joints in
    its order (a subset of the robot's needs ``allow=["select"]``);
    ``"positional"`` with ``count`` takes the robot's first ``count`` by
    index.
    """
    return _joint_set(
        "joints",
        at,
        command,
        hz=hz,
        chunk=chunk,
        names=names,
        count=count,
        units=units,
        scale=scale,
        offset=offset,
        range=range,
        t0=t0,
        ttl_ms=ttl_ms,
        optional=optional,
        allow=allow,
    )


def hand(
    at: str | None = None,
    *,
    command: JointCommand | None = None,
    hz: float | None = None,
    chunk: int | None = None,
    names: Sequence[str] | Literal["positional"] | None = None,
    count: int | None = None,
    units: Mapping[str, str] | None = None,
    scale: float | Sequence[float] | None = None,
    offset: float | Sequence[float] | None = None,
    range: tuple[float, float] | None = None,
    t0: Start | None = None,
    ttl_ms: float | None = None,
    optional: bool | None = None,
    allow: Sequence[Conversion] | None = None,
) -> Entry:
    """Commands for a hand's joints: :func:`joints` under its own kind, so it never drives an arm."""
    return _joint_set(
        "hand",
        at,
        command,
        hz=hz,
        chunk=chunk,
        names=names,
        count=count,
        units=units,
        scale=scale,
        offset=offset,
        range=range,
        t0=t0,
        ttl_ms=ttl_ms,
        optional=optional,
        allow=allow,
    )


def wrench(
    at: str | None = None,
    *,
    command: Literal["wrench"] | None = None,
    frame: str | None = None,
    hz: float | None = None,
    chunk: int | None = None,
    units: Mapping[str, str] | None = None,
    scale: float | Sequence[float] | None = None,
    offset: float | Sequence[float] | None = None,
    range: tuple[float, float] | None = None,
    t0: Start | None = None,
    ttl_ms: float | None = None,
    optional: bool | None = None,
    allow: Sequence[Conversion] | None = None,
) -> Entry:
    """A wrench to apply in ``frame``: ``force [3]`` N and ``torque [3]`` N*m."""
    return entry(
        "wrench",
        at,
        command=_command(command, ("wrench",)),
        frame=frame,
        hz=hz,
        chunk=chunk,
        units=units,
        scale=scale,
        offset=offset,
        range=range,
        t0=t0,
        ttl_ms=ttl_ms,
        optional=optional,
        allow=allow,
    )


def pose(
    at: str | None = None,
    *,
    command: Literal["absolute", "delta"] | None = None,
    reference: Literal["current", "target"] | None = None,
    frame: str | None = None,
    child: str | None = None,
    rotation: Rotation | None = None,
    hz: float | None = None,
    chunk: int | None = None,
    units: Mapping[str, str] | None = None,
    scale: float | Sequence[float] | None = None,
    offset: float | Sequence[float] | None = None,
    t0: Start | None = None,
    ttl_ms: float | None = None,
    optional: bool | None = None,
    allow: Sequence[Conversion] | None = None,
) -> Entry:
    """Pose targets for ``child`` in ``frame``: ``absolute``, or ``delta`` from the ``reference``.

    ``rotation`` is the encoding the model writes rotations in; another
    than the part's needs ``allow=["rotation"]`` unless it only reorders a
    quaternion.
    """
    return entry(
        "pose",
        at,
        command=_command(command, ("absolute", "delta")),
        reference=reference,
        frame=frame,
        child=child,
        rotation=rotation,
        hz=hz,
        chunk=chunk,
        units=units,
        scale=scale,
        offset=offset,
        t0=t0,
        ttl_ms=ttl_ms,
        optional=optional,
        allow=allow,
    )


def gripper(
    at: str | None = None,
    *,
    command: Literal["position"] | None = None,
    hz: float | None = None,
    chunk: int | None = None,
    open: float | None = None,
    closed: float | None = None,
    scale: float | Sequence[float] | None = None,
    offset: float | Sequence[float] | None = None,
    range: tuple[float, float] | None = None,
    t0: Start | None = None,
    ttl_ms: float | None = None,
    optional: bool | None = None,
    allow: Sequence[Conversion] | None = None,
) -> Entry:
    """Gripper openings: ``pos [1]``, 0 closed to 1 open.

    ``open`` and ``closed`` are the model's values for each end (default 1
    and 0) when they differ from the robot's.
    """
    return entry(
        "gripper",
        at,
        command=_command(command, ("position",)),
        hz=hz,
        chunk=chunk,
        open=open,
        closed=closed,
        scale=scale,
        offset=offset,
        range=range,
        t0=t0,
        ttl_ms=ttl_ms,
        optional=optional,
        allow=allow,
    )


def switch(
    at: str | None = None,
    *,
    command: Literal["set"] | None = None,
    hz: float | None = None,
    chunk: int | None = None,
    t0: Start | None = None,
    ttl_ms: float | None = None,
    optional: bool | None = None,
    allow: Sequence[Conversion] | None = None,
) -> Entry:
    """On/off targets: ``on``, a bool per action."""
    return entry(
        "switch",
        at,
        command=_command(command, ("set",)),
        hz=hz,
        chunk=chunk,
        t0=t0,
        ttl_ms=ttl_ms,
        optional=optional,
        allow=allow,
    )


def value(
    at: str | None = None,
    *,
    command: Literal["set"] | None = None,
    hz: float | None = None,
    chunk: int | None = None,
    units: Mapping[str, str] | None = None,
    scale: float | Sequence[float] | None = None,
    offset: float | Sequence[float] | None = None,
    range: tuple[float, float] | None = None,
    t0: Start | None = None,
    ttl_ms: float | None = None,
    optional: bool | None = None,
    allow: Sequence[Conversion] | None = None,
) -> Entry:
    """Targets for one number: ``value``, a float32 in the part's unit."""
    return entry(
        "value",
        at,
        command=_command(command, ("set",)),
        hz=hz,
        chunk=chunk,
        units=units,
        scale=scale,
        offset=offset,
        range=range,
        t0=t0,
        ttl_ms=ttl_ms,
        optional=optional,
        allow=allow,
    )


def velocity(
    at: str | None = None,
    *,
    command: Literal["velocity"] | None = None,
    frame: str | None = None,
    hz: float | None = None,
    chunk: int | None = None,
    units: Mapping[str, str] | None = None,
    scale: float | Sequence[float] | None = None,
    offset: float | Sequence[float] | None = None,
    range: tuple[float, float] | None = None,
    t0: Start | None = None,
    ttl_ms: float | None = None,
    optional: bool | None = None,
    allow: Sequence[Conversion] | None = None,
) -> Entry:
    """Base twists in ``frame``: ``linear`` m/s and ``angular`` rad/s."""
    return entry(
        "velocity",
        at,
        command=_command(command, ("velocity",)),
        frame=frame,
        hz=hz,
        chunk=chunk,
        units=units,
        scale=scale,
        offset=offset,
        range=range,
        t0=t0,
        ttl_ms=ttl_ms,
        optional=optional,
        allow=allow,
    )


def custom(
    at: str | None = None,
    *,
    kind: str,
    fields: Mapping[str, Sequence[Any]],
    command: str | None = None,
    frame: str | None = None,
    hz: float | None = None,
    chunk: int | None = None,
    units: Mapping[str, str] | None = None,
    scale: float | Sequence[float] | None = None,
    offset: float | Sequence[float] | None = None,
    range: tuple[float, float] | None = None,
    t0: Start | None = None,
    ttl_ms: float | None = None,
    optional: bool | None = None,
    allow: Sequence[Conversion] | None = None,
) -> Entry:
    """Commands for a custom kind ``ns/name``, with its command ``fields`` (``{name: parts.array(dtype, shape)}``)."""
    return entry(
        custom_kind(kind),
        at,
        fields=field_schema(fields),
        command=maybe(identifier, "command", command),
        frame=frame,
        hz=hz,
        chunk=chunk,
        units=units,
        scale=scale,
        offset=offset,
        range=range,
        t0=t0,
        ttl_ms=ttl_ms,
        optional=optional,
        allow=allow,
    )

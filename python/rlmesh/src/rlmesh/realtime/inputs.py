"""Typed constructors for a policy's inputs (Experimental).

Each returns the plain dict the policy map takes, with only the keys given,
so a map can mix them with dicts and the host still checks every rule::

    from rlmesh.realtime import inputs

    inputs.camera("hallo-cam:cam", size=224, allow=["resize"])
    # {"kind": "camera", "at": "hallo-cam:cam", "size": [224, 224], "allow": ["resize"]}

The first argument is the part's address, the entry's ``at``: ``robot:path``,
a path below the scope (``wrist/cam``), or ``/path`` from the robot's root;
a segment may be ``*`` or a ``{capture}``. Leave it out to bind any part of
the kind.

Keyword arguments every input takes: ``optional`` (may stay unbound, read
as blank; it needs ``"blank"`` in ``allow``, the entry's, the policy's or
the bind call's, or the binding is refused), ``max_age_ms`` (the oldest sample the policy accepts
at a start) and ``allow`` (the conversions this entry may use, beside the
policy's own). ``"hold"`` hands the policy a sample older than
``max_age_ms``, flagged, instead of skipping the start, while its publisher
is still in the session; once the publisher leaves, an optional input
allowed ``"blank"`` reads blank until it is back, ``"hold"`` or not. A streamed kind also takes ``history`` (a window of that many
samples) and ``history_hz`` (the rate it was trained at); a commandable kind
takes ``source="applied"`` to read what its edge dispatched instead of what
the robot published. Numeric kinds take ``dtype``, ``units`` (``{field:
unit}``), ``scale`` and ``offset`` (a number, or one per element) and
``range`` (``(low, high)``: the range ``rescale`` maps, or an image's
normalisation target).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast, get_args

from ._entries import (
    Conversion,
    DType,
    Entry,
    Rotation,
    custom_kind,
    entry,
    field_schema,
    maybe,
)
from ._entries import names as name_list

__all__ = [
    "camera",
    "custom",
    "depth",
    "gripper",
    "hand",
    "imu",
    "joints",
    "of",
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

Source = Literal["published", "applied"]
JointField = Literal["pos", "vel", "effort"]
ImuField = Literal["orientation", "ang_vel", "lin_acc", "gravity"]
WrenchField = Literal["force", "torque"]
PoseField = Literal["pos", "rot"]
VelocityField = Literal["linear", "angular"]


def _fields(value: object, choices: Any) -> list[str] | None:
    return maybe(name_list, "fields", value, get_args(choices))


def camera(
    at: str | None = None,
    *,
    size: int | tuple[int, int] | None = None,
    layout: Literal["hwc", "chw"] | None = None,
    format: Literal["rgb8", "bgr8", "mono8"] | None = None,
    resample: Literal["bilinear", "bilinear_aa", "bicubic_aa", "lanczos3_aa", "area"]
    | None = None,
    crop: float | None = None,
    upside_down: bool | None = None,
    dtype: DType | None = None,
    range: tuple[float, float] | None = None,
    frame: str | None = None,
    optional: bool | None = None,
    max_age_ms: float | None = None,
    history: int | None = None,
    history_hz: float | None = None,
    allow: Sequence[Conversion] | None = None,
) -> Entry:
    """A camera's frames: ``uint8 [height, width, 3]`` (``hwc``, ``rgb8``) by default.

    ``size`` is ``(height, width)``, or one side for a square: resizing
    needs ``allow=["resize"]`` and, for another aspect, ``crop``, ``pad`` or
    ``stretch``. ``crop`` is the side fraction of a center crop the model
    was trained with; ``upside_down`` a model trained on frames rotated
    180°. ``dtype`` with ``range`` (``(0, 1)``) normalises the pixels.
    """
    return entry(
        "camera",
        at,
        size=size,
        layout=layout,
        format=format,
        resample=resample,
        crop=crop,
        upside_down=upside_down,
        dtype=dtype,
        range=range,
        frame=frame,
        optional=optional,
        max_age_ms=max_age_ms,
        history=history,
        history_hz=history_hz,
        allow=allow,
    )


def depth(
    at: str | None = None,
    *,
    size: int | tuple[int, int] | None = None,
    crop: float | None = None,
    upside_down: bool | None = None,
    dtype: DType | None = None,
    units: Mapping[str, str] | None = None,
    range: tuple[float, float] | None = None,
    frame: str | None = None,
    optional: bool | None = None,
    max_age_ms: float | None = None,
    history: int | None = None,
    history_hz: float | None = None,
    allow: Sequence[Conversion] | None = None,
) -> Entry:
    """A depth camera's frames: ``depth: float32 [height, width]`` in m.

    ``size`` resizes by nearest neighbour; ``units={"depth": "mm"}`` asks
    for millimetres.
    """
    return entry(
        "depth",
        at,
        size=size,
        crop=crop,
        upside_down=upside_down,
        dtype=dtype,
        units=units,
        range=range,
        frame=frame,
        optional=optional,
        max_age_ms=max_age_ms,
        history=history,
        history_hz=history_hz,
        allow=allow,
    )


def _joint_set(
    kind: str,
    at: str | None,
    fields: Sequence[JointField] | None,
    **keys: Any,
) -> Entry:
    return entry(kind, at, fields=_fields(fields, JointField), **keys)


def joints(
    at: str | None = None,
    *,
    names: Sequence[str] | Literal["positional"] | None = None,
    count: int | None = None,
    fields: Sequence[JointField] | None = None,
    dtype: DType | None = None,
    units: Mapping[str, str] | None = None,
    scale: float | Sequence[float] | None = None,
    offset: float | Sequence[float] | None = None,
    range: tuple[float, float] | None = None,
    source: Source | None = None,
    optional: bool | None = None,
    max_age_ms: float | None = None,
    history: int | None = None,
    history_hz: float | None = None,
    allow: Sequence[Conversion] | None = None,
) -> Entry:
    """A set of joints: ``fields`` of ``pos``, ``vel`` and ``effort`` (default all published).

    ``names`` lists the model's joints in its order (a subset of the
    robot's needs ``allow=["select"]``); ``"positional"`` with ``count``
    takes the robot's first ``count`` by index.
    """
    return _joint_set(
        "joints",
        at,
        fields,
        names=names,
        count=count,
        dtype=dtype,
        units=units,
        scale=scale,
        offset=offset,
        range=range,
        source=source,
        optional=optional,
        max_age_ms=max_age_ms,
        history=history,
        history_hz=history_hz,
        allow=allow,
    )


def hand(
    at: str | None = None,
    *,
    names: Sequence[str] | Literal["positional"] | None = None,
    count: int | None = None,
    fields: Sequence[JointField] | None = None,
    dtype: DType | None = None,
    units: Mapping[str, str] | None = None,
    scale: float | Sequence[float] | None = None,
    offset: float | Sequence[float] | None = None,
    range: tuple[float, float] | None = None,
    source: Source | None = None,
    optional: bool | None = None,
    max_age_ms: float | None = None,
    history: int | None = None,
    history_hz: float | None = None,
    allow: Sequence[Conversion] | None = None,
) -> Entry:
    """A hand's joints: :func:`joints` under its own kind, so it never binds an arm."""
    return _joint_set(
        "hand",
        at,
        fields,
        names=names,
        count=count,
        dtype=dtype,
        units=units,
        scale=scale,
        offset=offset,
        range=range,
        source=source,
        optional=optional,
        max_age_ms=max_age_ms,
        history=history,
        history_hz=history_hz,
        allow=allow,
    )


def wrench(
    at: str | None = None,
    *,
    frame: str | None = None,
    point: str | None = None,
    fields: Sequence[WrenchField] | None = None,
    dtype: DType | None = None,
    units: Mapping[str, str] | None = None,
    scale: float | Sequence[float] | None = None,
    offset: float | Sequence[float] | None = None,
    range: tuple[float, float] | None = None,
    source: Source | None = None,
    optional: bool | None = None,
    max_age_ms: float | None = None,
    history: int | None = None,
    history_hz: float | None = None,
    allow: Sequence[Conversion] | None = None,
) -> Entry:
    """A force/torque reading: ``force [3]`` N and ``torque [3]`` N*m.

    ``frame`` and ``point`` are the frame and reference point the model
    needs; a part declared in another refuses to bind.
    """
    return entry(
        "wrench",
        at,
        frame=frame,
        point=point,
        fields=_fields(fields, WrenchField),
        dtype=dtype,
        units=units,
        scale=scale,
        offset=offset,
        range=range,
        source=source,
        optional=optional,
        max_age_ms=max_age_ms,
        history=history,
        history_hz=history_hz,
        allow=allow,
    )


def imu(
    at: str | None = None,
    *,
    frame: str | None = None,
    fields: Sequence[ImuField] | None = None,
    rotation: Rotation | None = None,
    dtype: DType | None = None,
    units: Mapping[str, str] | None = None,
    scale: float | Sequence[float] | None = None,
    offset: float | Sequence[float] | None = None,
    optional: bool | None = None,
    max_age_ms: float | None = None,
    history: int | None = None,
    history_hz: float | None = None,
    allow: Sequence[Conversion] | None = None,
) -> Entry:
    """An IMU reading: ``fields`` of ``orientation``, ``ang_vel``, ``lin_acc`` and ``gravity``.

    ``rotation`` is the encoding the model takes the orientation in
    (``"gravity_xyz"`` too); another than the part's needs
    ``allow=["rotation"]`` unless it only reorders a quaternion.
    """
    return entry(
        "imu",
        at,
        frame=frame,
        fields=_fields(fields, ImuField),
        rotation=rotation,
        dtype=dtype,
        units=units,
        scale=scale,
        offset=offset,
        optional=optional,
        max_age_ms=max_age_ms,
        history=history,
        history_hz=history_hz,
        allow=allow,
    )


def pose(
    at: str | None = None,
    *,
    frame: str | None = None,
    child: str | None = None,
    fields: Sequence[PoseField] | None = None,
    rotation: Rotation | None = None,
    dtype: DType | None = None,
    units: Mapping[str, str] | None = None,
    scale: float | Sequence[float] | None = None,
    offset: float | Sequence[float] | None = None,
    source: Source | None = None,
    optional: bool | None = None,
    max_age_ms: float | None = None,
    history: int | None = None,
    history_hz: float | None = None,
    allow: Sequence[Conversion] | None = None,
) -> Entry:
    """The pose of ``child`` in ``frame``: ``pos [3]`` m and ``rot`` in ``rotation``."""
    return entry(
        "pose",
        at,
        frame=frame,
        child=child,
        fields=_fields(fields, PoseField),
        rotation=rotation,
        dtype=dtype,
        units=units,
        scale=scale,
        offset=offset,
        source=source,
        optional=optional,
        max_age_ms=max_age_ms,
        history=history,
        history_hz=history_hz,
        allow=allow,
    )


def gripper(
    at: str | None = None,
    *,
    dtype: DType | None = None,
    scale: float | Sequence[float] | None = None,
    offset: float | Sequence[float] | None = None,
    range: tuple[float, float] | None = None,
    source: Source | None = None,
    optional: bool | None = None,
    max_age_ms: float | None = None,
    history: int | None = None,
    history_hz: float | None = None,
    allow: Sequence[Conversion] | None = None,
) -> Entry:
    """A gripper's opening: ``pos [1]``, 0 closed to 1 open (``range`` with ``allow=["rescale"]`` maps it)."""
    return entry(
        "gripper",
        at,
        dtype=dtype,
        scale=scale,
        offset=offset,
        range=range,
        source=source,
        optional=optional,
        max_age_ms=max_age_ms,
        history=history,
        history_hz=history_hz,
        allow=allow,
    )


def switch(
    at: str | None = None,
    *,
    dtype: DType | None = None,
    source: Source | None = None,
    optional: bool | None = None,
    max_age_ms: float | None = None,
    history: int | None = None,
    history_hz: float | None = None,
    allow: Sequence[Conversion] | None = None,
) -> Entry:
    """An on/off part's state: ``on``, a bool (``dtype`` with ``allow=["dtype"]`` casts it)."""
    return entry(
        "switch",
        at,
        dtype=dtype,
        source=source,
        optional=optional,
        max_age_ms=max_age_ms,
        history=history,
        history_hz=history_hz,
        allow=allow,
    )


def value(
    at: str | None = None,
    *,
    dtype: DType | None = None,
    units: Mapping[str, str] | None = None,
    scale: float | Sequence[float] | None = None,
    offset: float | Sequence[float] | None = None,
    range: tuple[float, float] | None = None,
    source: Source | None = None,
    optional: bool | None = None,
    max_age_ms: float | None = None,
    history: int | None = None,
    history_hz: float | None = None,
    allow: Sequence[Conversion] | None = None,
) -> Entry:
    """One number: ``value``, a float32 in the part's unit (``units={"value": ...}`` asks for another)."""
    return entry(
        "value",
        at,
        dtype=dtype,
        units=units,
        scale=scale,
        offset=offset,
        range=range,
        source=source,
        optional=optional,
        max_age_ms=max_age_ms,
        history=history,
        history_hz=history_hz,
        allow=allow,
    )


def velocity(
    at: str | None = None,
    *,
    frame: str | None = None,
    fields: Sequence[VelocityField] | None = None,
    dtype: DType | None = None,
    units: Mapping[str, str] | None = None,
    scale: float | Sequence[float] | None = None,
    offset: float | Sequence[float] | None = None,
    range: tuple[float, float] | None = None,
    source: Source | None = None,
    optional: bool | None = None,
    max_age_ms: float | None = None,
    history: int | None = None,
    history_hz: float | None = None,
    allow: Sequence[Conversion] | None = None,
) -> Entry:
    """A base twist in ``frame``: ``linear`` m/s and ``angular`` rad/s."""
    return entry(
        "velocity",
        at,
        frame=frame,
        fields=_fields(fields, VelocityField),
        dtype=dtype,
        units=units,
        scale=scale,
        offset=offset,
        range=range,
        source=source,
        optional=optional,
        max_age_ms=max_age_ms,
        history=history,
        history_hz=history_hz,
        allow=allow,
    )


def tactile(
    at: str | None = None,
    *,
    shape: Sequence[int] | None = None,
    dtype: DType | None = None,
    scale: float | Sequence[float] | None = None,
    offset: float | Sequence[float] | None = None,
    range: tuple[float, float] | None = None,
    optional: bool | None = None,
    max_age_ms: float | None = None,
    history: int | None = None,
    history_hz: float | None = None,
    allow: Sequence[Conversion] | None = None,
) -> Entry:
    """Taxels: ``values``, normalized; ``shape`` is the one the model expects."""
    return entry(
        "tactile",
        at,
        shape=shape,
        dtype=dtype,
        scale=scale,
        offset=offset,
        range=range,
        optional=optional,
        max_age_ms=max_age_ms,
        history=history,
        history_hz=history_hz,
        allow=allow,
    )


def text(
    at: str | None = None,
    *,
    optional: bool | None = None,
    max_age_ms: float | None = None,
    allow: Sequence[Conversion] | None = None,
) -> Entry:
    """Text events: the newest ``str`` (a task, a label)."""
    return entry("text", at, optional=optional, max_age_ms=max_age_ms, allow=allow)


def reward(
    at: str | None = None,
    *,
    optional: bool | None = None,
    max_age_ms: float | None = None,
    allow: Sequence[Conversion] | None = None,
) -> Entry:
    """Reward events: the newest ``float32 []``."""
    return entry("reward", at, optional=optional, max_age_ms=max_age_ms, allow=allow)


def success(
    at: str | None = None,
    *,
    optional: bool | None = None,
    max_age_ms: float | None = None,
    allow: Sequence[Conversion] | None = None,
) -> Entry:
    """Outcome events: the newest ``bool []``."""
    return entry("success", at, optional=optional, max_age_ms=max_age_ms, allow=allow)


def custom(
    at: str | None = None,
    *,
    kind: str,
    fields: Mapping[str, Sequence[Any]],
    frame: str | None = None,
    units: Mapping[str, str] | None = None,
    scale: float | Sequence[float] | None = None,
    offset: float | Sequence[float] | None = None,
    range: tuple[float, float] | None = None,
    source: Source | None = None,
    optional: bool | None = None,
    max_age_ms: float | None = None,
    history: int | None = None,
    history_hz: float | None = None,
    allow: Sequence[Conversion] | None = None,
) -> Entry:
    """A custom kind ``ns/name``, read with its declared ``fields`` (``{name: parts.array(dtype, shape)}``).

    The fields must be the part's, as declared: to read a part someone
    declared with :func:`rlmesh.realtime.parts.custom`, :func:`of` copies them.
    """
    return entry(
        custom_kind(kind),
        at,
        fields=field_schema(fields),
        frame=frame,
        units=units,
        scale=scale,
        offset=offset,
        range=range,
        source=source,
        optional=optional,
        max_age_ms=max_age_ms,
        history=history,
        history_hz=history_hz,
        allow=allow,
    )


#: The builder per core kind that :func:`of` hands a declared part to.
_BUILDERS: dict[str, Any] = {
    "camera": camera,
    "depth": depth,
    "joints": joints,
    "hand": hand,
    "wrench": wrench,
    "imu": imu,
    "pose": pose,
    "gripper": gripper,
    "switch": switch,
    "value": value,
    "velocity": velocity,
    "tactile": tactile,
    "text": text,
    "reward": reward,
    "success": success,
}


def of(part: Mapping[str, Any], at: str | None = None, **options: Any) -> Entry:
    """An input that reads ``part``, declared with :mod:`rlmesh.realtime.parts`: its kind and fields copied.

    For a part a policy publishes, declare it once and read it by pointing
    at that declaration::

        POSE = parts.custom(
            "pose/coco17", hz=30, fields={"kp": parts.array("float32", (17, 3))}
        )
        inputs.of(POSE, at="tracker:pose", optional=True)
        # {"kind": "pose/coco17", "at": "tracker:pose",
        #  "fields": {"kp": ["float32", [17, 3]]}, "optional": True}

    ``options`` are the keyword arguments of the kind's own builder
    (:func:`joints`, :func:`custom`, ...), checked the same way.
    """
    if not isinstance(part, Mapping) or not isinstance(part.get("kind"), str):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(f"part must be a part from rlmesh.realtime.parts, not {part!r}")
    kind = cast("str", part["kind"])
    if "kind" in options:
        raise TypeError("of() takes the kind from the part: drop kind=")
    builder = _BUILDERS.get(kind)
    if builder is None:
        fields = options.pop("fields", part.get("fields"))
        return custom(at, kind=kind, fields=fields, **options)
    if kind in ("joints", "hand", "imu") and part.get("fields") is not None:
        options.setdefault("fields", part["fields"])
    return cast("Entry", builder(at, **options))

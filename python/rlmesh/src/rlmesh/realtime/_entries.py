"""What :mod:`inputs` and :mod:`actions` share: the closed sets and the checks.

The checks repeat the policy map's own (``spec/policy.rs`` in the session
core) so a wrong value fails on the line that builds the entry, naming the
argument; the host still checks every rule when the policy binds.
"""

from __future__ import annotations

import math
import numbers as _numbers
import re
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast, get_args

if TYPE_CHECKING:
    from typing import TypeAlias

    import numpy as np

T = TypeVar("T")

#: An input or action entry of a policy map, as the host takes it.
Entry = dict[str, Any]

#: A conversion a policy may allow, for one entry or for all of them.
Conversion = Literal[
    "resize",
    "upscale",
    "crop",
    "pad",
    "stretch",
    "bgr",
    "rotate180",
    "dtype",
    "units",
    "rotation",
    "select",
    "rescale",
    "hold",
    "warmup",
    "blank",
    "drop",
]
#: A rotation encoding.
Rotation = Literal[
    "quat_xyzw",
    "quat_wxyz",
    "axis_angle",
    "rot6d",
    "rot6d_rowmajor",
    "matrix",
    "euler_xyz",
    "gravity_xyz",
]
#: A numeric element type, by name.
DTypeName = Literal[
    "bool",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "int8",
    "int16",
    "int32",
    "int64",
    "float16",
    "float32",
    "float64",
]
if TYPE_CHECKING:
    #: A numeric element type: its name, a NumPy dtype, or a Python type
    #: NumPy understands (``float`` is ``float64``).
    DType: TypeAlias = DTypeName | np.dtype[Any] | type[Any]
else:
    DType = DTypeName | type

CONVERSIONS: tuple[str, ...] = get_args(Conversion)
ROTATIONS: tuple[str, ...] = get_args(Rotation)
DTYPES: tuple[str, ...] = get_args(DTypeName)

#: The highest rate and the longest duration (ms) a policy declares.
MAX_HZ = 10_000.0
MAX_MS = 60_000.0
#: The largest image side, the longest window and the most actions per chunk.
MAX_IMAGE_SIDE = 8192
MAX_HISTORY = 1024
MAX_CHUNK = 1024
MAX_JOINTS = 1024

_IDENTIFIER = re.compile(r"[a-z][a-z0-9_]{0,63}")
_KIND_PART = re.compile(r"[a-z][a-z0-9_]{0,31}")


def one_of(choices: Sequence[str]) -> str:
    """``a``, ``a or b``, ``a, b or c``."""
    quoted = [repr(c) for c in choices]
    if len(quoted) == 1:
        return quoted[0]
    return f"{', '.join(quoted[:-1])} or {quoted[-1]}"


def choice(name: str, value: object, choices: Sequence[str]) -> str:
    if value not in choices:
        raise ValueError(f"{name}={value!r}: {one_of(choices)}")
    return cast("str", value)


def _number(name: str, value: object) -> float:
    """A real number as a plain ``int`` or ``float`` (NumPy scalars too), so the entry is JSON."""
    if isinstance(value, bool) or not isinstance(value, _numbers.Real):
        raise TypeError(f"{name} must be a number, not {value!r}")
    plain = int(value) if isinstance(value, _numbers.Integral) else float(value)
    if not math.isfinite(plain):
        raise ValueError(f"{name}={value!r}: must be finite")
    return plain


def rate(name: str, value: object) -> float:
    hz = _number(name, value)
    if not 0 < hz <= MAX_HZ:
        raise ValueError(f"{name}={value!r}: a rate in (0, 10000] Hz")
    return hz


def millis(name: str, value: object) -> float:
    ms = _number(name, value)
    if not 0 < ms <= MAX_MS:
        raise ValueError(f"{name}={value!r}: a duration in (0, 60000] ms")
    return ms


def integer(name: str, value: object, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, _numbers.Integral):
        raise TypeError(f"{name} must be an int, not {value!r}")
    plain = int(value)
    if not low <= plain <= high:
        raise ValueError(f"{name}={value!r}: {low} to {high}")
    return plain


def flag(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be True or False, not {value!r}")
    return value


def fraction(name: str, value: object) -> float:
    share = _number(name, value)
    if not 0 < share <= 1:
        raise ValueError(f"{name}={value!r}: a fraction in (0, 1]")
    return share


def finite(name: str, value: object) -> float:
    return _number(name, value)


def identifier(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a str, not {value!r}")
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{name}={value!r}: 1 to 64 characters of [a-z][a-z0-9_]*")
    return value


def address(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"at must be a str such as 'robot:path', not {value!r}")
    if not value:
        raise ValueError("at='': name a path, such as 'robot:path'")
    return value


def custom_kind(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"kind must be a str such as 'ns/name', not {value!r}")
    ns, slash, name = value.partition("/")
    if not slash or not _KIND_PART.fullmatch(ns) or not _KIND_PART.fullmatch(name):
        raise ValueError(
            f"kind={value!r}: a custom kind is 'ns/name', both [a-z][a-z0-9_]*"
        )
    if ns == "rlmesh":
        raise ValueError(f"kind={value!r}: the rlmesh/ namespace is reserved")
    return value


def span(name: str, value: object) -> list[float]:
    """``[low, high]``: two finite numbers that differ."""
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise TypeError(f"{name} must be (low, high), not {value!r}")
    items = list(cast("Sequence[object]", value))
    if len(items) != 2:
        raise ValueError(f"{name}={value!r}: (low, high)")
    low, high = (_number(name, item) for item in items)
    if low == high:
        raise ValueError(f"{name}={value!r}: low and high must differ")
    return [low, high]


def numbers(name: str, value: object) -> float | list[float]:
    """One number, or one per element."""
    if isinstance(value, Sequence) and not isinstance(value, str):
        items = [_number(name, item) for item in cast("Sequence[object]", value)]
        if not items:
            raise ValueError(f"{name}=[]: one number, or one per element")
        return items
    return _number(name, value)


def names(name: str, value: object, choices: Sequence[str]) -> list[str]:
    """A list of distinct names, each one of ``choices`` when given."""
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a list of names, not {value!r}")
    items = list(cast("Sequence[object]", value))
    if not items:
        raise ValueError(f"{name}=[]: name at least one")
    for i, item in enumerate(items):
        if not isinstance(item, str) or not item or len(item) > 128:
            raise ValueError(f"{name}: {item!r} is not a name of 1 to 128 characters")
        if choices and item not in choices:
            raise ValueError(f"{name}: {item!r} is not {one_of(choices)}")
        if item in items[:i]:
            raise ValueError(f"{name}: {item!r} is listed twice")
    return cast("list[str]", items)


def joint_names(value: object) -> list[str] | str:
    if value == "positional":
        return "positional"
    return names("names", value, ())


def allowed(value: object) -> list[str]:
    return names("allow", value, CONVERSIONS)


def units(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError(f"units must be a dict of field to unit, not {value!r}")
    out: dict[str, str] = {}
    for field, unit in cast("Mapping[object, object]", value).items():
        if not isinstance(field, str) or not isinstance(unit, str):
            raise TypeError(
                f"units: {field!r}: {unit!r} is not a field name and a unit"
            )
        out[field] = unit
    return out


def dtype_name(name: str, value: object, *, text: bool = False) -> str:
    """A dtype's name, from its name, a NumPy dtype or a Python type."""
    known = (*DTYPES, "utf8") if text else DTYPES
    if isinstance(value, str):
        if value not in known:
            raise ValueError(f"{name}={value!r}: {one_of(known)}")
        return value
    try:
        import numpy as np
    except ImportError:
        plain = {bool: "bool", int: "int64", float: "float64"}
        if isinstance(value, type) and value in plain:
            return plain[value]
        raise TypeError(
            f"{name}={value!r}: name the dtype ('float32') or install NumPy"
        ) from None
    try:
        dtype: Any = np.dtype(cast("Any", value))
    except TypeError:
        raise TypeError(f"{name}={value!r} is not a dtype") from None
    resolved = str(dtype.name)
    if resolved not in DTYPES:
        raise ValueError(f"{name}={value!r} is {resolved}: {one_of(DTYPES)}")
    return resolved


def _one_or_many(value: object, copies: int) -> Sequence[object] | None:
    """One int as ``copies`` of it, a sequence as itself; ``None`` for anything else."""
    if not isinstance(value, bool) and isinstance(value, _numbers.Integral):
        return (value,) * copies
    if isinstance(value, str) or not isinstance(value, Sequence):
        return None
    return cast("Sequence[object]", value)


def shape(name: str, value: object) -> list[int]:
    items = _one_or_many(value, 1)
    if items is None:
        raise TypeError(f"{name} must be a tuple of ints, not {value!r}")
    dims: list[int] = []
    for dim in items:
        if (
            isinstance(dim, bool)
            or not isinstance(dim, _numbers.Integral)
            or int(dim) < 0
        ):
            raise ValueError(f"{name}={value!r}: each dimension is an int >= 0")
        dims.append(int(dim))
    return dims


def field_schema(value: object) -> dict[str, list[Any]]:
    """A custom kind's ``{name: [dtype, shape]}``, as :func:`parts.array` builds it."""
    if not isinstance(value, Mapping) or not value:
        raise TypeError(
            "fields must be a dict of field name to parts.array(dtype, shape), "
            f"not {value!r}"
        )
    out: dict[str, list[Any]] = {}
    for field, decl in cast("Mapping[object, object]", value).items():
        if not isinstance(field, str) or not field:
            raise TypeError(f"fields: {field!r} is not a field name")
        pair = cast("Sequence[object]", decl) if isinstance(decl, list | tuple) else ()
        if len(pair) != 2:
            raise TypeError(
                f"fields[{field!r}] must be parts.array(dtype, shape), not {decl!r}"
            )
        dtype, dims = pair
        out[field] = [
            dtype_name(f"fields[{field!r}] dtype", dtype, text=True),
            shape(f"fields[{field!r}] shape", dims),
        ]
    return out


def image_size(value: object) -> list[int]:
    """``(height, width)``, or one side for a square."""
    items = _one_or_many(value, 2)
    if items is None or len(items) != 2:
        raise TypeError(f"size must be (height, width), not {value!r}")
    sides: list[int] = []
    for side in items:
        if isinstance(side, bool) or not isinstance(side, _numbers.Integral):
            raise TypeError(f"size must be (height, width) in ints, not {value!r}")
        if not 1 <= int(side) <= MAX_IMAGE_SIDE:
            raise ValueError(f"size={value!r}: each side is 1 to {MAX_IMAGE_SIDE}")
        sides.append(int(side))
    return sides


def maybe(check: Callable[..., T], name: str, value: object, *args: Any) -> T | None:
    """``check(name, value, *args)``, or ``None`` for an argument left unset."""
    return None if value is None else check(name, value, *args)


LAYOUTS = ("hwc", "chw")
FORMATS = ("rgb8", "bgr8", "mono8")
RESAMPLES = ("bilinear", "bilinear_aa", "bicubic_aa", "lanczos3_aa", "area")
SOURCES = ("published", "applied")
STARTS = ("snapshot", "send")
REFERENCES = ("current", "target")

#: How each key every kind may carry is checked, by name.
_CHECKS: dict[str, Callable[[str, object], object]] = {
    "optional": flag,
    "allow": lambda _, value: allowed(value),
    "max_age_ms": millis,
    "history_hz": rate,
    "history": lambda name, value: integer(name, value, 1, MAX_HISTORY),
    "source": lambda name, value: choice(name, value, SOURCES),
    "names": lambda _, value: joint_names(value),
    "count": lambda name, value: integer(name, value, 1, MAX_JOINTS),
    "size": lambda _, value: image_size(value),
    "layout": lambda name, value: choice(name, value, LAYOUTS),
    "format": lambda name, value: choice(name, value, FORMATS),
    "resample": lambda name, value: choice(name, value, RESAMPLES),
    "crop": fraction,
    "upside_down": flag,
    "dtype": dtype_name,
    "shape": shape,
    "range": span,
    "scale": numbers,
    "offset": numbers,
    "units": lambda _, value: units(value),
    "rotation": lambda name, value: choice(name, value, ROTATIONS),
    "frame": identifier,
    "child": identifier,
    "point": identifier,
    "hz": rate,
    "chunk": lambda name, value: integer(name, value, 1, MAX_CHUNK),
    "t0": lambda name, value: choice(name, value, STARTS),
    "ttl_ms": millis,
    "reference": lambda name, value: choice(name, value, REFERENCES),
    "open": finite,
    "closed": finite,
}


def entry(kind: str, at: str | None, **keys: Any) -> Entry:
    """The entry: ``kind``, ``at`` when given, then each key that is set, checked.

    A key without a check here (``fields``, ``command``) is one whose values
    depend on the kind: the builder checks it.
    """
    out: Entry = {"kind": kind}
    if at is not None:
        out["at"] = address(at)
    for key, value in keys.items():
        if value is None:
            continue
        check = _CHECKS.get(key)
        out[key] = value if check is None else check(key, value)
    if "history_hz" in out and out.get("history", 1) == 1:
        raise ValueError("history_hz needs history=2 or more")
    joint_list: object = out.get("names")
    count: object = out.get("count")
    if joint_list == "positional" and count is None:
        raise ValueError("names='positional' needs a count")
    if isinstance(joint_list, list) and count is not None:
        listed = len(cast("list[str]", joint_list))
        if listed != count:
            raise ValueError(f"count={count} differs from the {listed} names")
    if out.get("open", 1.0) == out.get("closed", 0.0):
        raise ValueError("open and closed must differ")
    return out

"""``rlmesh.realtime.inputs`` and ``actions`` build the policy map's plain dicts, key for key.

Each builder against the dict it must produce, the refusals naming the
argument, ``parts.array``, ``inputs.of``, the keyword ``Policy`` and how
``run`` reads a ``predict``'s signature. That the host takes every
builder's output is ``integration/test_v2_builders.py``.

The shared fixture ``builders.json`` holds a case for every builder and
argument; the Rust builders are checked against the same cases, which keeps
the two in step.
"""

from __future__ import annotations

import functools
import inspect
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest

np = pytest.importorskip("numpy")

from rlmesh import realtime  # noqa: E402
from rlmesh.realtime import (  # noqa: E402  # pyright: ignore[reportPrivateUsage]
    _arity,
    actions,
    inputs,
    parts,
)

AT = "bench:arm"
SHARED = (
    Path(__file__).resolve().parents[4]
    / "crates"
    / "rlmesh-session-core"
    / "tests"
    / "fixtures"
    / "builders.json"
)
CASES: list[dict[str, Any]] = json.loads(SHARED.read_text())["cases"]
MODULES = {"inputs": inputs, "actions": actions, "parts": parts}


def build_case(case: dict[str, Any]) -> object:
    """``case``'s builder called with its arguments, as a user writes them."""
    module, name = case["builder"].split(".")
    builder = getattr(MODULES[module], name)
    args = dict(case["args"])
    if name == "array":
        return builder(**args)
    if isinstance(args.get("fields"), dict):
        args["fields"] = {k: parts.array(*v) for k, v in args["fields"].items()}
    at = args.pop("at", None)
    if name == "of":
        return builder(case["part"], at, **args)
    return builder(at, **args)


@pytest.mark.parametrize(
    "case", CASES, ids=[f"{c['builder']}-{i}" for i, c in enumerate(CASES)]
)
def test_each_shared_case_gives_its_entry(case: dict[str, Any]) -> None:
    assert build_case(case) == case["entry"]


@pytest.mark.parametrize(
    ("module", "name"),
    [("inputs", name) for name in inputs.__all__]
    + [("actions", name) for name in actions.__all__]
    + [("parts", "array")],
)
def test_the_shared_cases_cover_every_builder_and_argument(
    module: str, name: str
) -> None:
    builder = f"{module}.{name}"
    covered = {
        key
        for case in CASES
        if case["builder"] == builder
        for key in (*case["args"], *(["part"] if "part" in case else []))
    }
    parameters = inspect.signature(getattr(MODULES[module], name)).parameters
    wanted = {
        p
        for p, spec in parameters.items()
        if spec.kind is not inspect.Parameter.VAR_KEYWORD
    }
    assert wanted <= covered, f"{builder}: no case sets {sorted(wanted - covered)}"


@pytest.mark.parametrize(
    ("built", "expected"),
    [
        (inputs.camera("front-cam:cam"), {"kind": "camera", "at": "front-cam:cam"}),
        (
            inputs.camera(
                "cam",
                size=224,
                layout="chw",
                format="bgr8",
                resample="area",
                crop=0.9,
                upside_down=True,
                dtype=np.float32,
                range=(0, 1),
                frame="optical",
                history=2,
                history_hz=15,
                max_age_ms=200,
                optional=True,
                allow=["resize", "bgr"],
            ),
            {
                "kind": "camera",
                "at": "cam",
                "size": [224, 224],
                "layout": "chw",
                "format": "bgr8",
                "resample": "area",
                "crop": 0.9,
                "upside_down": True,
                "dtype": "float32",
                "range": [0, 1],
                "frame": "optical",
                "history": 2,
                "history_hz": 15,
                "max_age_ms": 200,
                "optional": True,
                "allow": ["resize", "bgr"],
            },
        ),
        (inputs.camera(size=(96, 128)), {"kind": "camera", "size": [96, 128]}),
        (
            inputs.depth("d", size=(24, 32), units={"depth": "mm"}, dtype="uint16"),
            {
                "kind": "depth",
                "at": "d",
                "size": [24, 32],
                "units": {"depth": "mm"},
                "dtype": "uint16",
            },
        ),
        (
            inputs.joints(AT, names=("b", "a"), fields=["pos"], units={"pos": "deg"}),
            {
                "kind": "joints",
                "at": AT,
                "names": ["b", "a"],
                "fields": ["pos"],
                "units": {"pos": "deg"},
            },
        ),
        (
            inputs.joints(names="positional", count=6, source="applied"),
            {"kind": "joints", "names": "positional", "count": 6, "source": "applied"},
        ),
        (
            inputs.hand("h", scale=0.5, offset=[1, 2]),
            {"kind": "hand", "at": "h", "scale": 0.5, "offset": [1, 2]},
        ),
        (
            inputs.wrench("ft", frame="tool0", point="tcp", fields=["force"]),
            {
                "kind": "wrench",
                "at": "ft",
                "frame": "tool0",
                "point": "tcp",
                "fields": ["force"],
            },
        ),
        (
            inputs.imu(
                "imu", frame="base", fields=["orientation"], rotation="quat_wxyz"
            ),
            {
                "kind": "imu",
                "at": "imu",
                "frame": "base",
                "fields": ["orientation"],
                "rotation": "quat_wxyz",
            },
        ),
        (
            inputs.pose("tcp", frame="base", child="tool0", rotation="rot6d"),
            {
                "kind": "pose",
                "at": "tcp",
                "frame": "base",
                "child": "tool0",
                "rotation": "rot6d",
            },
        ),
        (
            inputs.gripper("g", range=(0, 255), allow=["rescale"]),
            {"kind": "gripper", "at": "g", "range": [0, 255], "allow": ["rescale"]},
        ),
        (
            inputs.switch("relay", dtype=bool),
            {"kind": "switch", "at": "relay", "dtype": "bool"},
        ),
        (
            inputs.value("temp", units={"value": "degF"}),
            {"kind": "value", "at": "temp", "units": {"value": "degF"}},
        ),
        (
            inputs.velocity("base", frame="base", fields=["linear"]),
            {"kind": "velocity", "at": "base", "frame": "base", "fields": ["linear"]},
        ),
        (
            inputs.tactile("skin", shape=(4, 4)),
            {"kind": "tactile", "at": "skin", "shape": [4, 4]},
        ),
        (inputs.text(), {"kind": "text"}),
        (
            inputs.text("/task", optional=True),
            {"kind": "text", "at": "/task", "optional": True},
        ),
        (
            inputs.reward("r", max_age_ms=50),
            {"kind": "reward", "at": "r", "max_age_ms": 50},
        ),
        (inputs.success(allow=["blank"]), {"kind": "success", "allow": ["blank"]}),
        (
            inputs.custom(
                "tracker:boxes",
                kind="yolo/boxes",
                fields={"xy": parts.array("float32", (8, 2))},
            ),
            {
                "kind": "yolo/boxes",
                "at": "tracker:boxes",
                "fields": {"xy": ["float32", [8, 2]]},
            },
        ),
    ],
)
def test_an_input_is_its_dict(built: dict[str, Any], expected: dict[str, Any]) -> None:
    assert built == expected
    assert json.loads(json.dumps(built)) == expected, "an entry is plain JSON"


@pytest.mark.parametrize(
    ("built", "expected"),
    [
        (
            actions.joints("gimbal-left:servos", command="position", hz=50, chunk=4),
            {
                "kind": "joints",
                "at": "gimbal-left:servos",
                "command": "position",
                "hz": 50,
                "chunk": 4,
            },
        ),
        (actions.joints(), {"kind": "joints"}),
        (
            actions.joints(
                AT, names=["a", "b"], count=2, t0="send", ttl_ms=150, optional=True
            ),
            {
                "kind": "joints",
                "at": AT,
                "names": ["a", "b"],
                "count": 2,
                "t0": "send",
                "ttl_ms": 150,
                "optional": True,
            },
        ),
        (
            actions.hand("h", command="velocity", units={"vel": "deg/s"}),
            {
                "kind": "hand",
                "at": "h",
                "command": "velocity",
                "units": {"vel": "deg/s"},
            },
        ),
        (
            actions.wrench("ft", command="wrench", frame="tool0"),
            {"kind": "wrench", "at": "ft", "command": "wrench", "frame": "tool0"},
        ),
        (
            actions.pose(
                "tcp", command="delta", reference="current", rotation="axis_angle"
            ),
            {
                "kind": "pose",
                "at": "tcp",
                "command": "delta",
                "reference": "current",
                "rotation": "axis_angle",
            },
        ),
        (
            actions.gripper("g", command="position", open=0, closed=1),
            {
                "kind": "gripper",
                "at": "g",
                "command": "position",
                "open": 0,
                "closed": 1,
            },
        ),
        (
            actions.switch("relay", command="set", chunk=2),
            {"kind": "switch", "at": "relay", "command": "set", "chunk": 2},
        ),
        (
            actions.value("dimmer", command="set", range=(0, 1), allow=["rescale"]),
            {
                "kind": "value",
                "at": "dimmer",
                "command": "set",
                "range": [0, 1],
                "allow": ["rescale"],
            },
        ),
        (
            actions.velocity(
                "base", command="velocity", frame="base", scale=[1, 1, 0.5]
            ),
            {
                "kind": "velocity",
                "at": "base",
                "command": "velocity",
                "frame": "base",
                "scale": [1, 1, 0.5],
            },
        ),
        (
            actions.custom(
                "game:move",
                kind="x/move",
                fields={"cell": parts.array(np.int64)},
                command="play",
            ),
            {
                "kind": "x/move",
                "at": "game:move",
                "fields": {"cell": ["int64", []]},
                "command": "play",
            },
        ),
    ],
)
def test_an_action_is_its_dict(built: dict[str, Any], expected: dict[str, Any]) -> None:
    assert built == expected
    assert json.loads(json.dumps(built)) == expected


def test_numpy_scalars_become_plain_numbers() -> None:
    built = actions.joints(
        hz=np.float32(50), chunk=np.int64(4), scale=[np.float64(0.5)]
    )
    assert (
        json.dumps(built)
        == '{"kind": "joints", "hz": 50.0, "chunk": 4, "scale": [0.5]}'
    )
    assert type(built["chunk"]) is int


@pytest.mark.parametrize(
    ("build", "error", "message"),
    [
        (
            lambda: actions.joints(command=cast("Any", "pos")),
            ValueError,
            "command='pos': 'position'",
        ),
        (
            lambda: actions.pose(command=cast("Any", "relative")),
            ValueError,
            "command='relative'",
        ),
        (lambda: actions.joints(chunk=0), ValueError, "chunk=0: 1 to 1024"),
        (lambda: actions.joints(chunk=1025), ValueError, "chunk=1025"),
        (
            lambda: actions.joints(chunk=cast("Any", 2.0)),
            TypeError,
            "chunk must be an int",
        ),
        (
            lambda: actions.joints(chunk=cast("Any", True)),
            TypeError,
            "chunk must be an int",
        ),
        (lambda: actions.joints(hz=0), ValueError, r"hz=0: a rate in \(0, 10000\]"),
        (lambda: actions.joints(hz=float("nan")), ValueError, "hz=nan: must be finite"),
        (
            lambda: actions.joints(hz=cast("Any", "50")),
            TypeError,
            "hz must be a number",
        ),
        (lambda: actions.joints(ttl_ms=60_001), ValueError, "ttl_ms=60001"),
        (
            lambda: actions.joints(t0=cast("Any", "now")),
            ValueError,
            "t0='now': 'snapshot' or 'send'",
        ),
        (lambda: actions.pose(reference=cast("Any", "x")), ValueError, "reference='x'"),
        (
            lambda: actions.gripper(open=1, closed=1),
            ValueError,
            "open and closed must differ",
        ),
        (
            lambda: actions.gripper(open=0, closed=0),
            ValueError,
            "open and closed must differ",
        ),
        (
            lambda: actions.custom(kind="x/y", fields={}),
            TypeError,
            "fields must be a dict",
        ),
        (
            lambda: actions.custom(
                kind="x/y", fields={"a": parts.array("float32")}, command="Set"
            ),
            ValueError,
            "command='Set'",
        ),
        (lambda: inputs.camera(cast("Any", 7)), TypeError, "at must be a str"),
        (lambda: inputs.camera(""), ValueError, "at=''"),
        (
            lambda: inputs.camera(size=(0, 3)),
            ValueError,
            r"size=\(0, 3\): each side is 1 to 8192",
        ),
        (lambda: inputs.camera(size=cast("Any", (1, 2, 3))), TypeError, "size must be"),
        (
            lambda: inputs.camera(layout=cast("Any", "nhwc")),
            ValueError,
            "layout='nhwc'",
        ),
        (lambda: inputs.camera(format=cast("Any", "rgb")), ValueError, "format='rgb'"),
        (
            lambda: inputs.camera(resample=cast("Any", "nearest")),
            ValueError,
            "resample='nearest'",
        ),
        (lambda: inputs.camera(crop=0), ValueError, r"crop=0: a fraction in \(0, 1\]"),
        (
            lambda: inputs.camera(upside_down=cast("Any", 1)),
            TypeError,
            "upside_down must be True",
        ),
        (
            lambda: inputs.camera(dtype=cast("Any", "float")),
            ValueError,
            "dtype='float'",
        ),
        (lambda: inputs.camera(dtype=np.complex64), ValueError, "is complex64"),
        (lambda: inputs.camera(range=(1, 1)), ValueError, "low and high must differ"),
        (lambda: inputs.camera(range=cast("Any", (1,))), ValueError, r"range=\(1,\)"),
        (lambda: inputs.camera(frame="Optical"), ValueError, "frame='Optical'"),
        (lambda: inputs.camera(history=0), ValueError, "history=0: 1 to 1024"),
        (
            lambda: inputs.camera(history_hz=10),
            ValueError,
            "history_hz needs history=2",
        ),
        (
            lambda: inputs.camera(allow=cast("Any", ["resise"])),
            ValueError,
            "'resise' is not",
        ),
        (
            lambda: inputs.camera(allow=cast("Any", "resize")),
            TypeError,
            "allow must be a list",
        ),
        (lambda: inputs.camera(max_age_ms=-1), ValueError, "max_age_ms=-1"),
        (lambda: inputs.joints(names=["a", "a"]), ValueError, "'a' is listed twice"),
        (lambda: inputs.joints(names=[]), ValueError, r"names=\[\]"),
        (
            lambda: inputs.joints(names=cast("Any", "all")),
            TypeError,
            "names must be a list",
        ),
        (lambda: inputs.joints(names="positional"), ValueError, "needs a count"),
        (lambda: inputs.joints(names=["a"], count=2), ValueError, "count=2 differs"),
        (lambda: inputs.joints(count=0), ValueError, "count=0"),
        (
            lambda: inputs.joints(fields=cast("Any", ["position"])),
            ValueError,
            "'position' is not",
        ),
        (lambda: inputs.imu(fields=cast("Any", ["pos"])), ValueError, "'pos' is not"),
        (
            lambda: inputs.imu(rotation=cast("Any", "quat")),
            ValueError,
            "rotation='quat'",
        ),
        (
            lambda: inputs.joints(source=cast("Any", "edge")),
            ValueError,
            "source='edge'",
        ),
        (lambda: inputs.joints(scale=[]), ValueError, r"scale=\[\]"),
        (
            lambda: inputs.joints(units=cast("Any", "deg")),
            TypeError,
            "units must be a dict",
        ),
        (lambda: inputs.tactile(shape=(4, -1)), ValueError, "shape="),
        (
            lambda: inputs.custom(kind="boxes", fields={"a": ["float32", []]}),
            ValueError,
            "'ns/name'",
        ),
        (
            lambda: inputs.custom(kind="rlmesh/x", fields={"a": ["float32", []]}),
            ValueError,
            "reserved",
        ),
        (
            lambda: inputs.custom(kind="x/y", fields={"a": cast("Any", "float32")}),
            TypeError,
            r"fields\['a'\] must be parts.array",
        ),
        (
            lambda: inputs.custom(kind="x/y", fields={"a": ["f32", []]}),
            ValueError,
            r"fields\['a'\] dtype='f32'",
        ),
        (lambda: inputs.camera(colour=1), TypeError, "colour"),  # pyright: ignore[reportCallIssue]
    ],
)
def test_a_wrong_value_fails_on_the_call_naming_the_argument(
    build: Callable[[], object], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        build()


@pytest.mark.parametrize(
    ("dtype", "shape", "expected"),
    [
        (np.float32, (17, 3), ["float32", [17, 3]]),
        (np.dtype("uint8"), [2], ["uint8", [2]]),
        (float, 4, ["float64", [4]]),
        (bool, (), ["bool", []]),
        ("int16", np.int64(3), ["int16", [3]]),
        ("utf8", (), ["utf8", []]),
    ],
)
def test_an_array_is_a_dtype_name_and_a_shape(
    dtype: Any, shape: Any, expected: list[Any]
) -> None:
    assert parts.array(dtype, shape) == expected


@pytest.mark.parametrize(
    ("dtype", "shape", "error", "message"),
    [
        ("float", (), ValueError, "dtype='float'"),
        (np.complex128, (), ValueError, "complex128"),
        (object(), (), TypeError, "is not a dtype"),
        ("float32", (2.5,), ValueError, "each dimension"),
        ("float32", "2", TypeError, "shape must be a tuple"),
    ],
)
def test_an_array_refuses_what_a_field_cannot_be(
    dtype: Any, shape: Any, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        parts.array(dtype, shape)


def test_an_array_takes_dtype_names_and_python_types_without_numpy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    real = builtins.__import__

    def no_numpy(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "numpy" or name.startswith("numpy."):
            raise ImportError(name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_numpy)
    assert parts.array("float32", 2) == ["float32", [2]]
    assert parts.array(float, 2) == ["float64", [2]]
    with pytest.raises(TypeError, match="install NumPy"):
        parts.array(cast("Any", complex))


def test_of_copies_a_declared_custom_part() -> None:
    pose = parts.custom(
        "pose/coco17", hz=30, fields={"kp": parts.array(np.float32, (17, 3))}
    )
    assert inputs.of(pose, at="tracker:pose", optional=True) == {
        "kind": "pose/coco17",
        "at": "tracker:pose",
        "fields": {"kp": ["float32", [17, 3]]},
        "optional": True,
    }


def test_of_copies_a_core_parts_fields_and_takes_its_builders_options() -> None:
    eyes = parts.joints(["pan", "tilt"], hz=50, fields=["pos"], command="position")
    assert inputs.of(eyes, at="eye-1:eyes", history=4) == {
        "kind": "joints",
        "at": "eye-1:eyes",
        "fields": ["pos"],
        "history": 4,
    }
    assert inputs.of(parts.camera(640, 480, hz=30)) == {"kind": "camera"}
    assert inputs.of(parts.text(), at="tracker:note") == {
        "kind": "text",
        "at": "tracker:note",
    }
    with pytest.raises(TypeError, match="size"):
        inputs.of(eyes, size=4)
    with pytest.raises(TypeError, match="drop kind="):
        inputs.of(eyes, kind="hand")
    with pytest.raises(TypeError, match=r"a part from rlmesh\.realtime\.parts"):
        inputs.of(cast("Any", "joints"))


def test_a_policy_takes_its_map_by_keyword() -> None:
    by_keyword = realtime.Policy(
        hz=30,
        inputs={"cam": inputs.camera("front-cam:cam")},
        actions={"left": actions.joints("gimbal-left:servos", command="position")},
        allow=["resize"],
        session=":0",
    )
    one_dict = realtime.Policy(
        {
            "hz": 30,
            "inputs": {"cam": {"kind": "camera", "at": "front-cam:cam"}},
            "actions": {
                "left": {
                    "kind": "joints",
                    "at": "gimbal-left:servos",
                    "command": "position",
                }
            },
            "allow": ["resize"],
        },
        session=":0",
    )
    policy_json = "_policy_json"
    assert json.loads(getattr(by_keyword, policy_json)) == json.loads(
        getattr(one_dict, policy_json)
    )


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (
            lambda: realtime.Policy({"hz": 1, "inputs": {}}, hz=2, session=":0"),
            "not both: drop hz= or the dict",
        ),
        (
            lambda: realtime.Policy(
                {"hz": 1}, inputs={"t": inputs.text()}, allow=["blank"], session=":0"
            ),
            "drop inputs=, allow= or the dict",
        ),
        (
            lambda: realtime.Policy(inputs={"t": inputs.text()}, session=":0"),
            "needs hz=",
        ),
        (lambda: realtime.Policy(hz=10, session=":0"), "inputs=, actions= or both"),
    ],
)
def test_a_policy_refuses_a_dict_and_keywords_together(
    build: Callable[[], object], message: str
) -> None:
    with pytest.raises(TypeError, match=message):
        build()


def test_the_clock_is_wall_or_lockstep_only() -> None:
    assert [c.value for c in realtime.Clock] == ["wall", "lockstep"]
    assert realtime.Clock("lockstep") is realtime.Clock.LOCKSTEP
    with pytest.raises(
        ValueError, match=r"clock='virtual': Clock\.WALL or Clock\.LOCKSTEP"
    ):
        realtime.Robot("r", {}, clock=cast("Any", "virtual"))


def one(obs: object) -> object:
    return obs


def two(obs: object, policy: object) -> object:
    return obs, policy


def with_default(obs: object, temperature: float = 0.5) -> object:
    return obs, temperature


class Model:
    def __call__(self, obs: object, policy: object) -> object:
        return obs, policy

    def predict(self, obs: object) -> object:
        return obs


@pytest.mark.parametrize(
    ("predict", "arity"),
    [
        (one, 1),
        (two, 2),
        (with_default, 1),
        (lambda obs: obs, 1),
        (lambda obs, policy: obs, 2),
        (lambda *args: args, 1),
        (Model(), 2),
        (Model().predict, 1),
        (functools.partial(two, policy=None), 1),
        (print, 1),
    ],
)
def test_predict_takes_the_policy_when_it_has_two_required_arguments(
    predict: Callable[..., Any], arity: int
) -> None:
    assert _arity(predict) == arity


@pytest.mark.parametrize(
    ("predict", "message"),
    [
        (lambda: None, "takes no argument"),
        (lambda a, b, c: None, "takes 3 arguments"),
    ],
)
def test_predict_must_take_the_observation(
    predict: Callable[..., Any], message: str
) -> None:
    with pytest.raises(TypeError, match=message):
        _arity(predict)

"""``rlmesh.realtime.parts`` builds the robot map's plain dicts, key for key."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

pytest.importorskip("numpy")

from rlmesh import realtime
from rlmesh.realtime import parts

FIXTURES = (
    Path(__file__).resolve().parents[4]
    / "crates"
    / "rlmesh-session-core"
    / "tests"
    / "fixtures"
    / "robots"
)
UR_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


def fixture(name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((FIXTURES / f"{name}.json").read_text())
    return data


def test_the_eye_is_its_fixture() -> None:
    assert {
        "gaze": parts.joints(
            ("pan", "tilt"),
            fields=["pos"],
            hz=50,
            command="position",
            safe=parts.Safe.STOP,
            max_action_age_ms=100,
        )
    } == fixture("eye")


def test_the_simulated_ur10e_is_its_fixture() -> None:
    arm = {"safe": parts.Safe.STOP, "hold_max_ms": 100, "max_action_age_ms": 1000}
    jpeg = {"encoding": "jpeg", "quality": 75, "send": "latest(max_mbps=20)"}
    assert {
        "arm": parts.joints(
            UR_JOINTS,
            fields=["pos", "vel", "effort"],
            hz=1000,
            command="position",
            group="arm",
            limits={"pos": [-6.283, 6.283]},
            **arm,
        ),
        "arm/tcp": parts.pose(
            frame="base",
            child="tool0",
            hz=500,
            command="delta",
            reference="current",
            group="arm",
            limits={"pos": {"max_norm": 0.01}, "rot": {"max_norm": 0.05}},
            **arm,
        ),
        "wrist/ft": parts.wrench(frame="tool0", hz=1000),
        "wrist/cam": parts.camera(640, 480, hz=30, frame="wrist_cam_optical", **jpeg),
        "scene/cam": parts.camera(1280, 720, hz=15, frame="scene_cam_optical", **jpeg),
        "gripper": parts.gripper(
            hz=100, command="position", safe=parts.Safe.HOLD, width_m=(0.0, 0.085)
        ),
        "task": parts.text(),
        "success": parts.success(),
    } == fixture("ur10e_sim")


def test_the_go2_is_its_fixture() -> None:
    legs = [
        f"{leg}_{joint}_joint"
        for leg in ("FR", "FL", "RR", "RL")
        for joint in ("hip", "thigh", "calf")
    ]
    locomotion = {
        "safe": parts.Safe.custom("damp"),
        "hold_max_ms": 50,
        "group": "locomotion",
    }
    assert {
        "legs": parts.joints(
            legs,
            fields=["pos", "vel", "effort"],
            hz=500,
            command="position",
            gains={"kp": 20.0, "kd": 0.5},
            **locomotion,
        ),
        "base/imu": parts.imu(
            frame="base",
            hz=500,
            fields=["orientation", "ang_vel", "lin_acc"],
            rotation="quat_wxyz",
        ),
        "base/twist": parts.velocity(
            frame="base",
            hz=50,
            command="velocity",
            limits={"linear": [[-1.5, 1.5], [-0.8, 0.8]], "angular": [-2.0, 2.0]},
            **locomotion,
        ),
        "feet/force": parts.custom(
            "unitree/foot_force",
            hz=500,
            fields={"force": ["int16", [4]]},
            units={"force": "raw"},
        ),
        "goal/cmd_vel": parts.velocity(frame="base", hz=50),
        "front/cam": parts.camera(
            1280,
            720,
            hz=15,
            frame="front_cam_optical",
            encoding="jpeg",
            send="latest(max_mbps=8)",
        ),
    } == fixture("go2")


def test_parts_are_plain_json() -> None:
    part = parts.joints(("pan",), hz=50, limits={"pos": (-1.0, 1.0)})
    assert json.loads(json.dumps(part)) == {
        "kind": "joints",
        "names": ["pan"],
        "hz": 50,
        "limits": {"pos": [-1.0, 1.0]},
    }


@pytest.mark.parametrize(
    ("safe", "name", "shown"),
    [
        (parts.Safe.STOP, "stop", "Safe.STOP"),
        (parts.Safe.HOLD, "hold", "Safe.HOLD"),
        (parts.Safe.custom("park"), "park", "Safe.custom('park')"),
    ],
)
def test_a_safe_lowers_to_its_name(safe: parts.Safe, name: str, shown: str) -> None:
    part = parts.joints(("pan",), hz=50, command="position", safe=safe)
    assert part["safe"] == name
    assert repr(safe) == shown


def test_a_bare_string_is_not_a_safe() -> None:
    with pytest.raises(TypeError, match=r"Safe\.STOP"):
        parts.joints(("pan",), hz=50, command="position", safe=cast("Any", "stop"))


@pytest.mark.parametrize("name", ["", "Park", "a b", "1x", "x" * 65])
def test_a_custom_safe_is_an_identifier(name: str) -> None:
    with pytest.raises(ValueError, match="safe behaviour"):
        parts.Safe.custom(name)


@pytest.mark.parametrize("name", ["stop", "hold"])
def test_a_custom_safe_is_not_a_built_in_one(name: str) -> None:
    with pytest.raises(ValueError, match="built in"):
        parts.Safe.custom(name)


def test_a_safe_is_compared_by_name() -> None:
    assert parts.Safe.custom("park") == parts.Safe.custom("park")
    assert parts.Safe.STOP != parts.Safe.HOLD
    assert len({parts.Safe.custom("park"), parts.Safe.custom("park")}) == 1


def test_a_safe_is_accepted_in_a_plain_map() -> None:
    plain = {"arm": {"kind": "joints", "count": 1, "safe": parts.Safe.HOLD}}
    assert json.loads(realtime._json(plain))["arm"]["safe"] == "hold"


def test_the_lamp_is_its_fixture() -> None:
    assert {
        "relay": parts.switch(hz=10, command="set", safe=parts.Safe.STOP),
        "button": parts.switch(hz=50),
        "dimmer": parts.value(
            hz=20,
            unit="percent",
            range=(0, 100),
            command="set",
            safe=parts.Safe.HOLD,
            limits={"value": [0, 100]},
        ),
        "temp": parts.value(hz=1, unit="degC", range=(-40, 125)),
    } == fixture("lamp")


@pytest.mark.parametrize(
    "build",
    [
        lambda **k: parts.joints(("pan",), hz=50, command="position", **k),
        lambda **k: parts.hand(("f",), hz=50, command="position", **k),
        lambda **k: parts.pose(frame="b", child="t", hz=50, command="absolute", **k),
        lambda **k: parts.gripper(hz=50, command="position", **k),
        lambda **k: parts.switch(hz=50, command="set", **k),
        lambda **k: parts.value(hz=50, command="set", **k),
    ],
)
def test_latch_is_given_only_when_set(build: Any) -> None:
    assert "latch" not in build()
    assert build(latch=True)["latch"] is True
    assert build(latch=False)["latch"] is False


def test_a_commanded_part_takes_a_clock_error_bound() -> None:
    part = parts.joints(
        ("pan",),
        hz=50,
        command="position",
        safe=parts.Safe.STOP,
        max_clock_error_ms=25,
    )
    assert part["max_clock_error_ms"] == 25

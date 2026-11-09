"""``rlmesh.realtime.parts`` builds the robot map's plain dicts, key for key."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("numpy")

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
            max_action_age_ms=100,
        )
    } == fixture("eye")


def test_the_simulated_ur10e_is_its_fixture() -> None:
    arm = {"safe": "stop", "hold_max_ms": 100, "max_action_age_ms": 1000}
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
            hz=100, command="position", safe="hold", width_m=(0.0, 0.085)
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
    locomotion = {"safe": "damp", "hold_max_ms": 50, "group": "locomotion"}
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

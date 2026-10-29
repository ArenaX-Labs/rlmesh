"""MuJoCo UR10e, a Python policy, keyboard teleop pre-emption and an MCAP recording.

The v0.2 milestone demo (protocol v2), three processes over one session:

- ``robot``: the session host, an MCAP recorder and the simulated UR10e
  (``ur10e.xml``, derived from MuJoCo Menagerie). Physics and the edge run
  at 1 kHz: each tick publishes ``arm`` (pos/vel/effort) and ``wrist/ft``,
  asks ``robot.due("arm")`` what to apply and steps the physics. A second
  thread renders the wrist camera offscreen at 20 Hz.
- ``policy``: a scripted reach (a slow sweep of two joints, rate limited
  from the measured pose) that reads the joints, a 10 ms window of the
  1 kHz wrist force/torque and the wrist camera, and sends 12 x 1 ms chunks
  at 100 Hz through ``Policy.run`` (``--policy force``, the default;
  ``--policy reach`` drops the force window).
- ``operator``: a teleop participant at ``operator`` class. ``t`` takes the
  ``arm`` lease (pre-empting the policy: the edge fences the policy's
  chunks and holds its last target until the operator's first chunk), the
  digits ``1``-``6`` pick a joint, ``+``/``-`` jog it, ``t`` again hands
  back (the policy re-requests and resumes under a fresh lease token), ``q``
  quits. Keys come from the terminal (``--keys interactive``) or from a
  script of ``seconds:key`` events (the default). ``--operator haptic``
  makes it the §15.6 shape: 1 kHz, reading the wrench every sample,
  one-action chunks with a 5 ms life.

``all`` (the default) runs the robot here and the policy and the operator
as child processes, then reads the recording back and prints a report:
chunk ages per lease holder, hold ticks, misses, commands applied per
second, the takeover and hand-back, and the MCAP check::

    uv sync --group mujoco
    uv run python examples/python/v2/ur10e_mujoco/demo.py --seconds 10
    uv run python examples/python/v2/ur10e_mujoco/demo.py --keys interactive --seconds 60
    uv run python examples/python/v2/ur10e_mujoco/demo.py --keys none --camera off --seconds 30

Or one role per terminal: ``demo.py robot`` prints the session address;
then ``demo.py policy --address ADDR`` and ``demo.py operator --address
ADDR --keys interactive``.
"""

from __future__ import annotations

import argparse
import bisect
import importlib
import json
import os
import queue
import resource
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

# Offscreen rendering without a display; set MUJOCO_GL=glfw (or osmesa) to override.
os.environ.setdefault("MUJOCO_GL", "egl")
# Without a GPU, EGL falls back to Mesa's llvmpipe, which by default spreads
# each frame over every core and starves the 1 kHz loop. Two threads render
# a 640x480 frame in about 10 ms here.
os.environ.setdefault("LP_NUM_THREADS", "2")

import numpy as np
import rlmesh.v2 as rlmesh

HERE = Path(__file__).resolve().parent
ROBOT = "ur10e"
RING = 1 << 16
CONTROL_HZ = 1000
UR_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
HOME = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
OPERATOR_HZ = 50
OPERATOR_HORIZON = 40  # 40 x 1 ms: twice the operator's replan period
FT_WINDOW = 10  # 10 ms of the 1 kHz wrench
JOG_RAD = 0.1
MAX_SPEED = 1.0  # rad/s, policy and operator alike


# --- maps (§14.1 ur10e_sim, §15.3, §15.6) ---------------------------------------


def robot_map(args: argparse.Namespace) -> dict[str, Any]:
    parts: dict[str, Any] = {
        "arm": {"kind": "joints", "names": UR_JOINTS, "fields": ["pos", "vel", "effort"],
                "hz": CONTROL_HZ, "command": "position", "safe": "stop", "hold_max_ms": 100,
                "max_action_age_ms": 1000, "limits": {"pos": [-6.283, 6.283]}},
        "wrist/ft": {"kind": "wrench", "frame": "tool0", "hz": CONTROL_HZ},
        "task": {"kind": "text"},
    }  # fmt: skip
    for i in range(args.extra_streams):  # synthetic 1 kHz load
        parts[f"extra/s{i}"] = {"kind": "x/load", "hz": CONTROL_HZ,
                                "fields": {"v": ["float32", [6]]}}  # fmt: skip
    if args.camera != "off":
        camera: dict[str, Any] = {"kind": "camera", "width": args.width, "height": args.height,
                                  "hz": args.camera_hz, "frame": "wrist_cam_optical"}  # fmt: skip
        if args.camera == "jpeg":
            camera |= {"encoding": "jpeg", "quality": 75, "send": "latest(max_mbps=20)"}
        parts["wrist/cam"] = camera
    return parts


def policy_map(args: argparse.Namespace) -> dict[str, Any]:
    inputs: dict[str, Any] = {
        "q": {"kind": "joints", "at": "arm", "fields": ["pos"], "names": UR_JOINTS},
    }
    if args.policy == "force":
        # §15.3 at 100 Hz: the newest 10 ms of the wrench, every sample.
        inputs["ft"] = {"kind": "wrench", "at": "wrist/ft", "frame": "tool0",
                        "history": FT_WINDOW}  # fmt: skip
        for i in range(args.extra_streams):
            inputs[f"x{i}"] = {"kind": "x/load", "at": f"extra/s{i}",
                               "fields": {"v": ["float32", [6]]}, "history": FT_WINDOW}  # fmt: skip
    if args.camera != "off":
        inputs["cam"] = {
            "kind": "camera",
            "at": "wrist/cam",
            "size": [96, 96],
            "max_age_ms": 150,
        }
    return {
        "hz": args.replan_hz,
        "inputs": inputs,
        "actions": {"arm": {"kind": "joints", "at": "arm", "command": "position",
                            "names": UR_JOINTS, "hz": CONTROL_HZ, "horizon": args.horizon}},
        "allow": ["resize", "pad"],
    }  # fmt: skip


def operator_map(args: argparse.Namespace) -> dict[str, Any]:
    if args.operator == "haptic":
        # §15.6: 1 kHz, the wrench every sample, one-action chunks, 5 ms life.
        return {
            "hz": CONTROL_HZ,
            "inputs": {"q": {"kind": "joints", "at": "arm", "fields": ["pos"], "names": UR_JOINTS},
                       "ft": {"kind": "wrench", "at": "wrist/ft", "frame": "tool0", "max_age_ms": 3}},
            "actions": {"arm": {"kind": "joints", "at": "arm", "command": "position",
                                "names": UR_JOINTS, "hz": CONTROL_HZ, "horizon": 1, "ttl_ms": 5}},
        }  # fmt: skip
    return {
        "hz": OPERATOR_HZ,
        "inputs": {"q": {"kind": "joints", "at": "arm", "fields": ["pos"], "names": UR_JOINTS}},
        "actions": {"arm": {"kind": "joints", "at": "arm", "command": "position",
                            "names": UR_JOINTS, "hz": CONTROL_HZ, "horizon": OPERATOR_HORIZON}},
    }  # fmt: skip


def ramp(start: np.ndarray, target: np.ndarray, n: int, dt: float) -> np.ndarray:
    """``n`` actions from ``start`` toward ``target`` at most MAX_SPEED rad/s."""
    step = MAX_SPEED * dt * np.arange(1, n + 1)[:, None]
    return start + np.clip(target - start, -step, step)


def cpu_share(before: resource.struct_rusage, wall0: float) -> dict[str, float]:
    """CPU seconds (user + system) of this process since ``before``, and the share of one core."""
    after = resource.getrusage(resource.RUSAGE_SELF)
    cpu = (after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime)
    wall = time.monotonic() - wall0
    return {
        "cpu_s": round(cpu, 3),
        "wall_s": round(wall, 3),
        "cores": round(cpu / wall, 3),
    }


def percentiles(ns: list[int]) -> dict[str, float] | None:
    if not ns:
        return None
    ordered = np.sort(np.asarray(ns, dtype=np.int64))

    def at(q: float) -> float:
        rank = min(max(int(np.ceil(q * len(ordered))), 1), len(ordered))
        return round(float(ordered[rank - 1]) / 1e6, 4)

    return {"n": len(ordered), "p50": at(0.5), "p99": at(0.99), "max": at(1.0)}


# --- robot ---------------------------------------------------------------------


class Sim:
    """The UR10e in MuJoCo, published as ``ROBOT``, its edge ticked every 1 ms."""

    def __init__(self, args: argparse.Namespace, session: rlmesh.Host | str) -> None:
        # MuJoCo is the demo's optional dependency (`uv sync --group mujoco`).
        mujoco: Any = importlib.import_module("mujoco")
        self.mj = mujoco
        self.args = args
        self.model = mujoco.MjModel.from_xml_path(str(args.model))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
        mujoco.mj_forward(self.model, self.data)
        self.qadr = [int(self.model.joint(j).qposadr[0]) for j in UR_JOINTS]
        self.dadr = [int(self.model.joint(j).dofadr[0]) for j in UR_JOINTS]
        self.substeps = max(1, round(1.0 / (self.model.opt.timestep * CONTROL_HZ)))
        self.extras = [f"extra/s{i}" for i in range(args.extra_streams)]
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.robot = rlmesh.Robot(ROBOT, robot_map(args), session=session)
        self.ticks: list[
            tuple[int, int, str, int | None]
        ] = []  # capture, late, mode, token
        self.chunks: dict[tuple[int, int], tuple[int, int, int, int]] = {}
        self.work: list[int] = []
        self.frames = 0
        self.render_ns: list[int] = []
        self.missed = 0
        self.error: BaseException | None = None

    def start(self) -> None:
        # The camera thread's Python code holds the GIL in slices of the
        # switch interval (5 ms by default): keep the 1 kHz loop's wait short.
        sys.setswitchinterval(0.0002)
        self.robot.arm()
        self.robot.publish("task", "sweep the arm over the table")
        self.threads = [
            threading.Thread(target=self._guard(self.control), name="control")
        ]
        if self.args.camera != "off":
            self.threads.append(
                threading.Thread(target=self._guard(self.camera), name="camera")
            )
        for t in self.threads:
            t.start()

    def _guard(self, loop: Any) -> Any:
        def run() -> None:
            try:
                loop()
            except BaseException as exc:
                self.error = exc
                self.stop.set()
                raise

        return run

    def control(self) -> None:
        mj, model, data, robot = self.mj, self.model, self.data, self.robot
        arm_capture = np.zeros(RING, dtype=np.int64)
        viewer = None
        if self.args.view:
            viewer = importlib.import_module("mujoco.viewer").launch_passive(
                model, data
            )
        rate = rlmesh.Rate(CONTROL_HZ)
        late = rate.sleep()
        tick = 0
        while not self.stop.is_set():
            capture = time.monotonic_ns()
            seq = robot.publish(
                "arm",
                pos=data.qpos[self.qadr],
                vel=data.qvel[self.dadr],
                effort=data.qfrc_actuator[self.dadr],
                capture_ns=capture,
            )
            if seq is not None:
                arm_capture[seq % RING] = capture
            robot.publish(
                "wrist/ft",
                force=data.sensordata[0:3],
                torque=data.sensordata[3:6],
                capture_ns=capture,
            )
            for path in self.extras:
                robot.publish(path, data.sensordata, capture_ns=capture)
            due = robot.due("arm")
            mode, token = "idle", None
            if due is not None:
                mode, token = due.mode, due.lease_token
                if isinstance(due.value, np.ndarray):
                    data.ctrl[:] = due.value  # the edge's validated, clamped target
                elif due.safe is not None:
                    data.ctrl[:] = data.qpos[self.qadr]  # "stop": hold where it is
                if (
                    mode == "driving"
                    and due.received_ns is not None
                    and due.slot_ns is not None
                ):
                    key = (due.lease_token or 0, due.chunk_seq or 0)
                    arm = next((r for r in due.based_on if r[0] == "arm"), None)
                    if key not in self.chunks and arm is not None:
                        self.chunks[key] = (
                            int(arm_capture[arm[2] % RING]),
                            due.received_ns,
                            due.slot_ns,
                            due.index or 0,
                        )
            self.work.append(time.monotonic_ns() - capture)
            self.ticks.append((capture, late, mode, token))
            with self.lock:
                for _ in range(self.substeps):
                    mj.mj_step(model, data)
            if viewer is not None and tick % 16 == 0:
                with self.lock:
                    viewer.sync()
            tick += 1
            late = rate.sleep()
        self.missed = rate.missed
        if viewer is not None:
            viewer.close()

    def camera(self) -> None:
        mj, model = self.mj, self.model
        renderer = mj.Renderer(model, self.args.height, self.args.width)
        # A wrist camera needs no shadows, reflections or sky: a fifth of
        # the render time in software GL.
        for flag in ("mjRND_SHADOW", "mjRND_REFLECTION", "mjRND_SKYBOX"):
            renderer.scene.flags[getattr(mj.mjtRndFlag, flag)] = 0
        scratch = mj.MjData(model)
        rate = rlmesh.Rate(self.args.camera_hz)
        try:
            while not self.stop.is_set():
                with self.lock:
                    scratch.qpos[:] = self.data.qpos
                    scratch.time = self.data.time
                    capture = time.monotonic_ns()
                mj.mj_kinematics(model, scratch)
                mj.mj_camlight(model, scratch)
                renderer.update_scene(scratch, camera="wrist_cam")
                frame = renderer.render()
                self.render_ns.append(time.monotonic_ns() - capture)
                self.robot.publish("wrist/cam", frame, capture_ns=capture)
                self.frames += 1
                rate.sleep()
        finally:
            renderer.close()

    def finish(self) -> dict[str, Any]:
        self.stop.set()
        for t in self.threads:
            t.join()
        edge = self.robot.edge_stats("arm")
        self.robot.close()
        if self.error is not None:
            raise RuntimeError("the robot loop failed") from self.error
        return edge


# --- report --------------------------------------------------------------------


def holders(recording: dict[str, Any]) -> tuple[dict[int, str], list[dict[str, Any]]]:
    """Lease token -> holder for the arm, and the arm's lease timeline, from the recording."""
    timeline = [
        lease for lease in recording["leases"] if lease["part"] == f"{ROBOT}/arm"
    ]
    by_token = {
        lease["token"]: lease["holder"]
        for lease in timeline
        if lease["token"] is not None
    }
    return by_token, timeline


def report(
    args: argparse.Namespace,
    sim: Sim,
    edge: dict[str, Any],
    recorded: dict[str, Any] | None,
    recording: dict[str, Any],
    children: dict[str, Any],
) -> dict[str, Any]:
    by_token, timeline = holders(recording)
    operator_hz = CONTROL_HZ if args.operator == "haptic" else OPERATOR_HZ
    period = {"policy": 1e9 / args.replan_hz, "operator": 1e9 / operator_hz}
    dt_ns = 1e9 / CONTROL_HZ

    # Chunk ages per holder (receipt at the edge minus the capture of the arm
    # sample the chunk was computed from), split at the chunk's t0.
    ages: dict[str, dict[str, list[int]]] = {}
    # When chunks were computed from a sample older than half a replan
    # period (seconds into the run): where the phase alignment slipped.
    stale: dict[str, list[float]] = {}
    first_ns = sim.ticks[0][0] if sim.ticks else 0
    for (token, _), (capture, receipt, slot, index) in sim.chunks.items():
        if capture == 0:
            continue
        who = by_token.get(token, "?")
        t0 = slot - int(index * dt_ns)
        if t0 - capture > period.get(who, 0.0) / 2:
            stale.setdefault(who, []).append(round((t0 - first_ns) / 1e9, 3))
        row = ages.setdefault(who, {"chunk": [], "snapshot": [], "transit": []})
        row["chunk"].append(receipt - capture)
        row["snapshot"].append(t0 - capture)
        row["transit"].append(receipt - t0)

    ticks = sim.ticks
    holder_of = [by_token.get(t[3]) if t[3] is not None else None for t in ticks]
    # The policy window: ticks while the policy holds the arm's lease (from
    # the recorded authority timeline; one clock on one host), after one
    # second of warmup, until its last driven tick.
    changes = [(lease["log_ns"], lease["holder"]) for lease in timeline]
    stamps = [c[0] for c in changes]

    def lessee(t: int) -> str | None:
        k = bisect.bisect_right(stamps, t)
        return changes[k - 1][1] if k else None

    policy_driven = [
        i for i, t in enumerate(ticks) if t[2] == "driving" and holder_of[i] == "policy"
    ]
    first = policy_driven[0] if policy_driven else 0
    last = policy_driven[-1] if policy_driven else -1
    warm = ticks[first][0] + 1_000_000_000 if ticks else 0
    policy_ticks = [
        i
        for i in range(first, last + 1)
        if ticks[i][0] >= warm and lessee(ticks[i][0]) == "policy"
    ]
    window = [ticks[i] for i in policy_ticks]
    inside = set(policy_ticks)
    seconds = (
        sum(ticks[i + 1][0] - ticks[i][0] for i in policy_ticks if i + 1 in inside)
        / 1e9
    )
    modes: dict[str, int] = {}
    for t in window:
        modes[t[2]] = modes.get(t[2], 0) + 1
    driving_in_window = modes.get("driving", 0)

    # Takeover and hand-back, as the robot saw them.
    transitions: list[dict[str, Any]] = []
    for i in range(1, len(ticks)):
        before, now = holder_of[i - 1], holder_of[i]
        if now is not None and ticks[i][2] == "driving" and before != now:
            j = i - 1
            while j > 0 and (ticks[j][2] != "driving" or holder_of[j] == now):
                j -= 1
            gap = ticks[j + 1 : i]
            prev = holder_of[j] if ticks[j][2] == "driving" else None
            if prev is None or prev == now:
                continue
            transitions.append(
                {
                    "from": prev,
                    "to": now,
                    "token": ticks[i][3],
                    "gap_ms": round((ticks[i][0] - ticks[j][0]) / 1e6, 3),
                    "gap_modes": {
                        m: sum(1 for t in gap if t[2] == m) for m in {t[2] for t in gap}
                    },
                }
            )

    policy_age = ages.get("policy", {})
    target_ms = 1000.0 / args.replan_hz + 1.0
    chunk = percentiles(policy_age.get("chunk", []))
    return {
        "config": {
            key: getattr(args, key)
            for key in (
                "seconds",
                "camera",
                "camera_hz",
                "width",
                "height",
                "replan_hz",
                "horizon",
                "keys",
            )
        },
        "loadavg": Path("/proc/loadavg").read_text().strip()
        if Path("/proc/loadavg").exists()
        else None,
        "latency": {
            who: {
                "chunks": len(row["chunk"]),
                "chunk_age_ms": percentiles(row["chunk"]),
                "snapshot_age_ms": percentiles(row["snapshot"]),
                "transit_ms": percentiles(row["transit"]),
                "replan_period_ms": round(period.get(who, 0.0) / 1e6, 3),
                "stale_snapshots_at_s": sorted(stale.get(who, []))[:40],
            }
            for who, row in sorted(ages.items())
        },
        "policy_window": {
            "seconds": round(seconds, 3),
            "ticks": len(window),
            "mode_ticks": modes,
            "hold_ticks": modes.get("hold", 0),
            # Safe ticks before the policy's last driven tick (the run's end
            # releases the lease: hold, then the safe behaviour, unlatched).
            "safe_ticks_before_end": sum(
                1 for t in ticks[: last + 1] if t[2] == "safe"
            ),
            "miss_pct": round(
                100.0 * (len(window) - driving_in_window) / max(len(window), 1), 4
            ),
            "applied_per_s": round(driving_in_window / seconds, 1) if seconds else 0.0,
        },
        "done_when": {
            "p99_chunk_age_ms": chunk["p99"] if chunk else None,
            "target_ms": round(target_ms, 3),
            "met": bool(chunk and chunk["p99"] <= target_ms),
        },
        "transitions": transitions,
        "lease_timeline": [
            {k: lease[k] for k in ("part", "holder", "priority", "token")}
            for lease in timeline
        ],
        "robot": {
            "ticks": len(ticks),
            "tick_work_us": {
                k: round(v * 1e3, 1)
                for k, v in (percentiles(sim.work) or {}).items()
                if k != "n"
            },
            "late_ticks_over_1ms": sum(1 for t in ticks if t[1] > 1_000_000),
            "missed_periods": sim.missed,
            "camera_frames": sim.frames,
            "render_ms": percentiles(sim.render_ns),
            "edge": edge,
        },
        "recording": {"stats": recorded, "summary": recording},
        "children": children,
    }


def checks(args: argparse.Namespace, out: dict[str, Any]) -> dict[str, bool]:
    """The demo's done-when, as booleans (``latency`` is measured, not enforced)."""
    summary = out["recording"]["summary"]
    topics = summary["topics"]
    wanted = [
        f"/{ROBOT}/arm",
        f"/{ROBOT}/arm/_applied",
        f"/{ROBOT}/arm/_chunks",
        f"/{ROBOT}/wrist/ft",
        "/_session/authority",
        "/_session/bindings",
    ]
    if args.camera != "off":
        wanted.append(f"/{ROBOT}/wrist/cam")
    result = {
        "mcap_topics": all(t in topics and topics[t]["messages"] > 0 for t in wanted),
        "mcap_decodes": all(t["undecodable"] == 0 for t in topics.values()),
        "mcap_monotonic": all(
            t["log_time_monotonic"] and t["publish_time_monotonic"]
            for t in topics.values()
        ),
        "mcap_count": summary["messages"]
        == (out["recording"]["stats"] or {}).get("messages"),
        "policy_drove": "policy" in out["latency"],
        "never_safe_while_running": out["policy_window"]["safe_ticks_before_end"] == 0,
    }
    independent = out["recording"].get("independent")
    if independent is not None:
        result["mcap_independent_read"] = independent == {
            name: t["messages"] for name, t in topics.items()
        }
    if args.keys not in ("none", "interactive"):
        holders_seen = [
            lease["holder"] for lease in out["lease_timeline"] if lease["holder"]
        ]
        tokens = [lease["token"] for lease in out["lease_timeline"] if lease["holder"]]
        moves = [(t["from"], t["to"]) for t in out["transitions"]]
        result["takeover"] = ("policy", "operator") in moves
        result["hand_back"] = moves[-1:] == [("operator", "policy")]
        result["fresh_token"] = (
            len(holders_seen) >= 3
            and holders_seen[0] == "policy"
            and holders_seen[-1] == "policy"
            and tokens == sorted(tokens)
            and len(set(tokens)) == len(tokens)
        )
        result["held_not_safe"] = all(
            t["gap_modes"].get("safe", 0) == 0 for t in out["transitions"]
        )
    return result


def read_independently(path: Path) -> dict[str, int] | None:
    """Message counts per topic with the ``mcap`` package, if installed."""
    try:
        make_reader: Any = importlib.import_module("mcap.reader").make_reader
    except ImportError:
        return None
    counts: dict[str, int] = {}
    with path.open("rb") as f:
        for _, channel, _ in make_reader(f).iter_messages():
            counts[channel.topic] = counts.get(channel.topic, 0) + 1
    return counts


def child(
    role: str, args: argparse.Namespace, address: str, out: Path
) -> subprocess.Popen[bytes]:
    forwarded = [
        f"--{key.replace('_', '-')}={getattr(args, key)}"
        for key in (
            "seconds",
            "camera",
            "replan_hz",
            "horizon",
            "keys",
            "policy",
            "operator",
            "extra_streams",
        )
    ]
    return subprocess.Popen(
        [
            sys.executable,
            __file__,
            role,
            *forwarded,
            "--address",
            address,
            "--out",
            str(out),
        ],
        env=os.environ.copy(),
    )


def run_robot(args: argparse.Namespace) -> int:
    record = Path(
        args.record or Path(tempfile.mkdtemp(prefix="rlmesh-ur10e-")) / "session.mcap"
    )
    cpu0, wall0 = resource.getrusage(resource.RUSAGE_SELF), time.monotonic()
    with rlmesh.Host() as host:
        recorder = rlmesh.Recorder(record, session=host)
        sim = Sim(args, host)
        sim.start()
        children: dict[str, Any] = {}
        if args.role == "all":
            tmp = Path(tempfile.mkdtemp(prefix="rlmesh-ur10e-"))
            procs = {"policy": child("policy", args, host.address, tmp / "policy.json")}
            if args.keys != "none":
                procs["operator"] = child(
                    "operator", args, host.address, tmp / "operator.json"
                )
            limit = args.seconds + 60
            # Interactive: the run ends when the operator quits.
            order = (
                ["operator", "policy"] if args.keys == "interactive" else list(procs)
            )
            for name in order:
                proc = procs[name]
                try:
                    proc.wait(timeout=limit)
                except subprocess.TimeoutExpired:
                    proc.kill()
                if name == "operator" and args.keys == "interactive":
                    # The operator quit: stop the policy too.
                    procs["policy"].send_signal(signal.SIGINT)
                path = tmp / f"{name}.json"
                children[name] = (
                    json.loads(path.read_text())
                    if proc.wait() == 0 and path.exists()
                    else {"error": proc.returncode}
                )
        else:
            print(f"session: {host.address}", file=sys.stderr, flush=True)
            try:
                end = time.monotonic() + args.seconds
                while time.monotonic() < end and not sim.stop.is_set():
                    time.sleep(0.1)
            except KeyboardInterrupt:
                pass
        edge = sim.finish()
        robot_cpu = cpu_share(cpu0, wall0)
        recorded = recorder.close()
    recording = rlmesh.read_recording(record)
    out = report(args, sim, edge, recorded, recording, children)
    out["cpu"] = {
        "robot_process": robot_cpu,
        "policy_process": (children.get("policy") or {}).get("cpu"),
        "loadavg_after": Path("/proc/loadavg").read_text().split()[:3],
    }
    out["recording"]["path"] = str(record)
    out["recording"]["independent"] = read_independently(record)
    out["checks"] = checks(args, out)
    emit(args, out)
    if not args.json:
        print_summary(out)
    return 0 if all(out["checks"].values()) else 1


def print_summary(out: dict[str, Any]) -> None:
    lines = [
        f"recording: {out['recording']['path']} "
        f"({out['recording']['summary']['messages']} messages)"
    ]
    for topic, t in out["recording"]["summary"]["topics"].items():
        lines.append(
            f"  {topic:<26} {t['messages']:>7}  monotonic={t['log_time_monotonic']}"
            f"  undecodable={t['undecodable']}"
        )
    for who, row in out["latency"].items():
        age = row["chunk_age_ms"] or {}
        lines.append(
            f"{who}: {row['chunks']} chunks, chunk age p50 {age.get('p50')} / "
            f"p99 {age.get('p99')} ms (replan {row['replan_period_ms']} ms)"
        )
    w = out["policy_window"]
    lines.append(
        f"policy window {w['seconds']} s: {w['applied_per_s']} applied/s, "
        f"hold ticks {w['hold_ticks']}, miss {w['miss_pct']} %"
    )
    d = out["done_when"]
    lines.append(
        f"done-when p99 {d['p99_chunk_age_ms']} ms <= {d['target_ms']} ms: {d['met']}"
    )
    for t in out["transitions"]:
        lines.append(
            f"{t['from']} -> {t['to']} (token {t['token']}): {t['gap_ms']} ms, {t['gap_modes']}"
        )
    policy = out["children"].get("policy") or {}
    for entry, age in (policy.get("window_age_ms") or {}).items():
        if age:
            lines.append(
                f"newest {entry} sample at a start: p50 {age['p50']} / p99 {age['p99']} ms old"
            )
    cpu = out["cpu"]
    lines.append(
        f"cpu (cores): robot {cpu['robot_process']['cores']}, policy {(cpu['policy_process'] or {}).get('cores')}"
    )
    lines.append(f"checks: {out['checks']}")
    print("\n".join(lines), file=sys.stderr)


def emit(args: argparse.Namespace, out: dict[str, Any]) -> None:
    text = json.dumps(out, indent=2, default=str)
    if args.out is not None:
        Path(args.out).write_text(text)
    if args.json:
        print(text)


# --- policy --------------------------------------------------------------------


def run_policy(args: argparse.Namespace) -> int:
    horizon, dt = args.horizon, 1.0 / CONTROL_HZ
    brightness: list[float] = []
    origin: list[float] = []
    # Per start, the age of the newest sample in each window input.
    window_ages: dict[str, list[int]] = {}

    def predict(obs: rlmesh.Observation) -> dict[str, Any]:
        for entry, age in obs.meta.ages_ns.items():
            if age is not None and (entry == "ft" or entry.startswith("x")):
                window_ages.setdefault(entry, []).append(age)
        now = obs.meta.time_ns / 1e9
        if not origin:
            origin.append(now)
        # The reach: sweep the pan and the first wrist joint around home.
        t = now - origin[0] + np.arange(horizon) * dt
        target = np.repeat(HOME[None], horizon, axis=0)
        target[:, 0] += 0.5 * np.sin(2 * np.pi * 0.2 * t)
        target[:, 3] += 0.3 * np.sin(2 * np.pi * 0.3 * t)
        if "cam" in obs:
            brightness.append(float(np.asarray(obs["cam"]).mean()))
        q = np.asarray(obs["q"], dtype=np.float64)
        step = MAX_SPEED * dt * np.arange(1, horizon + 1)[:, None]
        return {"arm": (q + np.clip(target - q, -step, step)).astype(np.float32)}

    with rlmesh.Policy(policy_map(args), session=args.address, name="policy") as policy:
        binding = policy.bind(ROBOT)
        print(binding, file=sys.stderr, flush=True)
        cpu0, wall0 = resource.getrusage(resource.RUSAGE_SELF), time.monotonic()
        try:
            stats = policy.run(predict, duration=args.seconds)
        except KeyboardInterrupt:
            stats = policy.stats()
        cpu = cpu_share(cpu0, wall0)
        out = {
            "cpu": cpu,
            "window_age_ms": {
                entry: percentiles(ages) for entry, ages in window_ages.items()
            },
            "binding": binding.json,
            "warnings": binding.warnings,
            "run": stats,
            "events": policy.events(),
            "camera_reads": len(brightness),
            "camera_mean": round(float(np.mean(brightness)), 2) if brightness else None,
        }
    emit(args, out)
    return 0


# --- operator ------------------------------------------------------------------


def default_script(seconds: float) -> str:
    take, give = 0.4 * seconds, 0.65 * seconds
    jogs = [
        (take + 0.2, "4"),
        (take + 0.3, "+"),
        (take + 0.5, "+"),
        (take + 0.7, "+"),
        (take + 0.9, "1"),
        (take + 1.0, "-"),
        (give, "t"),
    ]
    return ",".join([f"{take:.2f}:t", *(f"{t:.2f}:{k}" for t, k in jogs)])


def feed_script(script: str, keys: queue.Queue[str], start: float) -> None:
    events = []
    for item in script.split(","):
        at, key = item.split(":", 1)
        events.append((float(at), key))
    for at, key in sorted(events):
        time.sleep(max(0.0, start + at - time.monotonic()))
        keys.put(key)


def feed_terminal(keys: queue.Queue[str], stop: threading.Event) -> None:
    import select
    import termios
    import tty

    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    try:
        while not stop.is_set():
            ready, _, _ = select.select([fd], [], [], 0.1)
            if ready:
                key = os.read(fd, 1).decode(errors="ignore")
                keys.put(key)
                if key == "q":
                    return
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


def run_operator(args: argparse.Namespace) -> int:
    keys: queue.Queue[str] = queue.Queue()
    stop = threading.Event()
    events: list[tuple[float, str]] = []
    chunks = 0
    haptic = args.operator == "haptic"
    hz = CONTROL_HZ if haptic else OPERATOR_HZ
    horizon = 1 if haptic else OPERATOR_HORIZON
    with rlmesh.Policy(
        operator_map(args), session=args.address, name="operator", priority="operator"
    ) as op:
        op.bind(ROBOT)
        start = time.monotonic()
        if args.keys == "interactive":
            if not sys.stdin.isatty():
                raise SystemExit("--keys interactive needs a terminal")
            print(
                "operator: t take/hand back, 1-6 joint, +/- jog, q quit",
                file=sys.stderr,
            )
            feeder = threading.Thread(
                target=feed_terminal, args=(keys, stop), daemon=True
            )
        else:
            script = (
                args.keys if args.keys != "scripted" else default_script(args.seconds)
            )
            feeder = threading.Thread(
                target=feed_script, args=(script, keys, start), daemon=True
            )
        feeder.start()
        rate = rlmesh.Rate(hz)
        active, joint = False, 0
        # The jog target and the last commanded pose, set from the measured
        # pose on the first start after a takeover (`fresh`).
        target, cmd, fresh = HOME.copy(), HOME.copy(), True
        end = start + args.seconds
        while time.monotonic() < end:
            while not keys.empty():
                key = keys.get()
                stamp = round(time.monotonic() - start, 3)
                if key == "t" and not active:
                    for _ in range(
                        100
                    ):  # NOT_READY until the operator's inputs are installed
                        try:
                            op.acquire()
                            break
                        except rlmesh.SessionRefused:
                            time.sleep(0.02)
                    active, fresh = True, True
                    events.append((stamp, "take"))
                elif key == "t":
                    op.release()
                    active = False
                    events.append((stamp, "hand back"))
                elif key in "123456" and len(key) == 1:
                    joint = int(key) - 1
                elif key in ("+", "=", "k") and active and not fresh:
                    target[joint] += JOG_RAD
                    events.append((stamp, f"jog {UR_JOINTS[joint]} +{JOG_RAD}"))
                elif key in ("-", "j") and active and not fresh:
                    target[joint] -= JOG_RAD
                    events.append((stamp, f"jog {UR_JOINTS[joint]} -{JOG_RAD}"))
                elif key == "q":
                    end = 0.0
            if active:
                try:
                    obs = op.observe()
                except rlmesh.StartSkipped:
                    rate.sleep()
                    continue
                if fresh:
                    cmd = np.asarray(obs["q"], dtype=np.float64).copy()
                    target, fresh = cmd.copy(), False
                chunk = ramp(cmd, target, horizon, 1.0 / CONTROL_HZ)
                sent = op.send({"arm": chunk.astype(np.float32)}, obs)
                if sent.get("arm") is not None:
                    chunks += 1
                cmd = ramp(cmd, target, 1, 1.0 / hz)[0]
            rate.sleep()
        stop.set()
        if active:
            op.release()
    emit(args, {"events": events, "chunks_sent": chunks})
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "role", nargs="?", default="all", choices=["all", "robot", "policy", "operator"]
    )
    parser.add_argument(
        "--seconds", type=float, default=10.0, help="how long the policy runs"
    )
    parser.add_argument("--camera", choices=["jpeg", "raw", "off"], default="jpeg")
    parser.add_argument("--camera-hz", type=float, default=20.0)
    parser.add_argument(
        "--policy",
        choices=["force", "reach"],
        default="force",
        help="force (default): joints, a 10 ms force window and the camera",
    )
    parser.add_argument(
        "--operator",
        choices=["keys", "haptic"],
        default="keys",
        help="haptic: the §15.6 1 kHz operator",
    )
    parser.add_argument(
        "--extra-streams",
        type=int,
        default=0,
        help="synthetic 1 kHz streams the force policy also windows",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--replan-hz", type=float, default=100.0)
    parser.add_argument(
        "--horizon", type=int, default=12, help="1 ms actions per chunk (§9.2)"
    )
    parser.add_argument(
        "--keys",
        default="scripted",
        help="scripted (default), interactive, none, or 'seconds:key,...'",
    )
    parser.add_argument("--model", type=Path, default=HERE / "ur10e.xml")
    parser.add_argument(
        "--record", type=Path, help="the MCAP file (default: a temp dir)"
    )
    parser.add_argument(
        "--view", action="store_true", help="open MuJoCo's passive viewer"
    )
    parser.add_argument("--address", help="the session (policy, operator)")
    parser.add_argument(
        "--json", action="store_true", help="print the full report as JSON"
    )
    parser.add_argument("--out", type=Path, help="also write the JSON report here")
    args = parser.parse_args()
    if args.role in ("policy", "operator") and not args.address:
        parser.error(f"{args.role} needs --address")
    run = {
        "all": run_robot,
        "robot": run_robot,
        "policy": run_policy,
        "operator": run_operator,
    }
    sys.exit(run[args.role](args))


if __name__ == "__main__":
    main()

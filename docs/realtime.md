# Real-time sessions: robots on a LAN

`rlmesh.realtime` (Experimental) connects robots and policies on one
network. A **robot** declares its parts (joints, cameras, ...), publishes
what it senses and plays the commands its edge says are due. A **policy**
declares what it needs by kind and path, binds to the robots that have
it, and sends short timed action chunks. A **host** (`rlmesh host`) runs
the session: who may join, who holds which part, and what is recorded.

This guide builds one setup end to end: two ESP32 "eyes" with a pan and a
tilt servo each, a camera, and a policy on a PC that runs a detector on
the camera and points both eyes. The same pieces fit any robot.

```text
  camera ──frames──▶  host (PC)  ◀──joins── eye-1 (ESP32)
                          ▲                 eye-2 (ESP32)
  policy (PC): camera in, eye targets out
```

## 1. Install

On the PC (and on a Pi, if the camera hangs off one):

```sh
pip install --pre "rlmesh[robot]"
```

0.2 is an alpha, so pip needs `--pre`. Every install has `rlmesh.realtime`
and the `rlmesh` command with `rlmesh host` and, on desktop platforms,
`rlmesh cockpit`; the `robot` extra adds NumPy for samples and frames.

## 2. Host the session

On the PC:

```sh
rlmesh host --presence-grace 2s --lease-grace 500ms --rearm-robots
```

It listens on every interface at port 7447 and keeps its identity (its
key, and the devices it admits in `devices.toml`) in its state directory,
so boards provisioned once stay valid across restarts. The two graces
are raised for Wi-Fi, where a board can go quiet for a moment.
`--rearm-robots` drives a robot again as soon as it rejoins; without it a
returning robot waits to be re-armed by name. See the [CLI
reference](reference/cli.md) for every flag.

Programs on the PC join it by name: `session=""` is this machine's host,
default session. Start the host first: a program that finds no host on
this machine hosts the session itself, on loopback only, where the boards
can't reach it.

A provisioned board remembers the address the host had when it was
provisioned, so give the PC a fixed address on your network (a DHCP
reservation is enough). If the address changes, provision the boards
again.

## 3. Flash and provision the eyes

Every eye runs the same firmware, the RLMesh library's **Robot** example,
whose map is a `joints` part `eyes` (`pan`, `tilt`) commanded by position
at 50 Hz. Follow the [ESP32 board quickstart](../integrations/esp32/README.md#board-quickstart),
then, with `rlmesh host` running and each board on USB in turn:

```sh
rlmesh devices provision --port /dev/ttyUSB0 --name eye-1
rlmesh devices provision --port /dev/ttyUSB0 --name eye-2
```

Provisioning writes the robot's name and the host's key and addresses to
the board, and enrolls the board's own key, so it joins on its own after
every reboot and nothing else can join as `eye-1`. `rlmesh devices ls`
lists what the host admits.

## 4. Publish the camera

A camera is a robot with one part. On the PC:

```python
import cv2  # any capture library
from rlmesh import realtime
from rlmesh.realtime import parts

CAMERA = {"cam": parts.camera(640, 480, hz=15, encoding="jpeg", quality=80)}

with realtime.Robot("webcam", CAMERA, session="") as camera:
    capture = cv2.VideoCapture(0)
    rate = realtime.Rate(15)
    while True:
        ok, bgr = capture.read()
        if ok:
            camera.publish("cam", cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        rate.sleep()
```

`publish` takes an RGB `uint8 [480, 640, 3]` frame and encodes it as
JPEG (with the GIL released). A camera that already gives you JPEG (a Pi
camera, a USB webcam in MJPEG mode) can skip the decode and re-encode
with `camera.publish_jpeg("cam", jpeg_bytes)`. Frames never queue up
behind a slow link: the newest frame replaces one not yet sent.

**On a Pi instead.** Pair the Pi with the PC once, so the host admits it:
run `rlmesh pair pc.local` on the Pi, then `rlmesh pair approve` on the PC
and type the code it shows. Then join with `session="pc.local"` (or the
PC's address).

## 5. Drive both eyes from one policy

The policy reads the camera and sends each eye a pan/tilt target. It
names each part by robot: `webcam:cam`, `eye-1:eyes`, `eye-2:eyes`.

```python
import numpy as np
from rlmesh import realtime
from rlmesh.realtime import parts

TRACKER = {
    "hz": 15,
    "inputs": {"cam": {"kind": "camera", "at": "webcam:cam"}},
    "actions": {
        "left": {"kind": "joints", "at": "eye-1:eyes", "command": "position",
                 "hz": 50, "chunk": 4},
        "right": {"kind": "joints", "at": "eye-2:eyes", "command": "position",
                  "hz": 50, "chunk": 4},
    },
}
# What the detector saw, recorded and shown in the cockpit beside the eyes.
HAND = {"hand": parts.custom("yolo/hand", hz=15, fields={"xy": ["float32", [2]]})}

center = np.zeros(2, np.float32)

def predict(obs):
    xy = detect(obs["cam"])  # your model: the hand's (x, y) in 0..1, or None
    target = center if xy is None else np.array(
        [(xy[0] - 0.5) * 1.2, (0.5 - xy[1]) * 0.8], np.float32
    )
    if xy is not None:
        policy.publish("hand", xy=np.asarray(xy, np.float32))
    chunk = np.tile(target, (4, 1))  # [chunk, (pan, tilt)] in radians
    return {"left": chunk, "right": chunk}

with realtime.Policy(TRACKER, session="", name="tracker", publishes=HAND) as policy:
    print(policy.bind())  # the binding table: what is bound, and how
    policy.run(predict)
```

`bind()` with no robot binds the whole session's view, so one policy
drives every eye it names. Each call of `predict` gets the newest camera
frame as an RGB `uint8 [480, 640, 3]` array and returns a chunk of 4
targets 20 ms apart, more than one replan period at 15 Hz, so an eye
never runs dry between calls. The eye's edge plays the chunk, clamps it to
the limits the board declared, and holds the last target when no new
chunk comes.

The eyes move only while the session is armed: press **Arm** in the
cockpit (next section), or call `camera.arm()` from a robot's program.
Arming an armed session does nothing, so any script may call it.

**When an eye drops off** (Wi-Fi, a brown-out, a reboot), the host stops
that eye alone: its lease ends, the other eye keeps tracking, and the
policy keeps running. When the board is back it rejoins with its key and,
under `--rearm-robots`, is driven again at once. Without the flag it stays
held until you say so by name, from any robot's program:
`camera.rearm("eye-1")` (in a scene, `rlmesh scene arm RUN eye-1`). A held
robot stays held across a disarm or a session-wide SAFE: `arm()` never
clears it.

**When a policy drops off** (a crash, a lost link, or a new process
joining under its name), the host ends only the leases it held. Those
parts go to their declared safe behaviour, as after a release; the session
stays armed, every other policy keeps driving, and nothing needs re-arming.
The host logs who was lost and which `robot:part`s were released.

## 6. Watch it: the cockpit

```sh
rlmesh cockpit default        # on the PC
rlmesh cockpit pc.local       # on a paired laptop
```

The cockpit joins as an observer and shows every part's declared and
measured rate, the lease and mode timeline, and **why each robot is or
isn't moving**: its edge mode with the reason (`queue_dry`: no valid chunk
within its hold time), the chunks it refused and why (`too_old`,
`no_lease`, ...), and each link's heartbeat age and jitter. It
has Arm, Disarm, E-stop and Keep.

## 7. Record and export

The host records every session, always: one-minute MCAP segments under
`rolling/{session id}/` in its recordings directory (on Linux
`~/.local/share/rlmesh/recordings`), the newest 30 minutes. An e-stop,
safe mode, or the cockpit's **Keep** button saves the 2 minutes before
and 30 s after under `incidents/{session id}/{time}-{label}/`, for 14
days. The session id is a UUID; a host with one session has one
directory there.

Export one as a LeRobot dataset (`pip install lerobot`):

```sh
python -m rlmesh.realtime.lerobot \
    ~/.local/share/rlmesh/recordings/incidents/<session-id>/<time>-keep eyes-dataset \
    --repo-id me/halloween-eyes --task "follow the hand" --fps 15
```

Each frame holds the camera image, and as the action what each eye's edge
actually dispatched (`eye-1/eyes/_applied`, `eye-2/eyes/_applied`); an eye
that held keeps its last action. `rlmesh cockpit --replay DIR` plays a
recording back.

## Switches and values

Not every part is a joint. A `switch` is one on/off `on` field: an LED, a
relay or a valve when commanded (`command: "set"`), a button when not. A
`value` is one float, with an optional `unit` and `range`: a thermometer
when observed, a dimmer when commanded (`command: "set"`).

```python
LAMP = {
    "relay": parts.switch(hz=10, command="set", safe=parts.Safe.STOP),
    "button": parts.switch(hz=50),
    "dimmer": parts.value(hz=20, unit="percent", range=(0, 100),
                          command="set", safe=parts.Safe.HOLD),
    "temp": parts.value(hz=1, unit="degC"),
}
```

A policy addresses them by kind like any other part, and a chunk for a
switch is an array of booleans, one per action.

A commanded switch or value is **latched** by default: the edge keeps
applying its last target for as long as the commanding lease is held, so a
policy can set a relay once and go quiet. When the lease is released,
revoked or lost, the part goes to its `safe` behaviour. Set `latch: false`
to give it the usual short life instead. Any part whose command mode holds
its target can opt in with `latch: true`; the host refuses it on a velocity
or torque, which would run away.

## Safety

- The session starts disarmed; nothing moves until it is armed.
- **E-stop** (the cockpit, a robot's `estop()`, or a board's button)
  latches: every edge purges its queue and runs its safe behaviour, here
  `hold`. Each robot keeps its own latch, and an ESP32 keeps it across a
  reboot. To recover, in order:
  1. on every board, hold BOOT for 2 s (with its e-stop button released);
  2. clear the session's latch from the robot that set it, or, when the
     cockpit or a policy set it, from a robot process on the host's
     machine (`camera.clear_estop()` here);
  3. disarm, then arm.
- Every part that accepts commands declares its `safe` behaviour, and the
  host refuses a map that leaves it out. It is `stop`, `hold`, or the name
  of a behaviour the robot's own firmware implements; RLMesh passes the
  name to the robot and does not interpret it. In Python it is
  `parts.Safe.STOP`, `parts.Safe.HOLD` or `parts.Safe.custom("park")`.
- Every part's limits are enforced on the robot's side, by its edge,
  whatever a policy sends.
- Beyond loopback the host encrypts every sample and command; see the
  [threat model](threat-model.md) for what it does and doesn't defend.

## Next

- The [Python API](https://rlmesh.dev/docs/) (`rlmesh.realtime`, and
  `rlmesh.realtime.parts` for every part kind) and its [Node twin](node.md).
- [The CLI reference](reference/cli.md): `rlmesh host`, `devices`,
  `pair`, `cockpit`.

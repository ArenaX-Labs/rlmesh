# Real-time sessions: robots on a LAN

`rlmesh.realtime` (Experimental) connects robots and policies on one
network. A **robot** declares its parts (joints, cameras, ...), publishes
what it senses and plays the commands its edge says are due. A **policy**
declares what it needs by kind and path, binds to the robots that have
it, and sends short timed action chunks. A **host** (`rlmesh host`) runs
the session: who may join, who holds which part, and what is recorded.

This guide builds one setup end to end: two ESP32 "eyes" with a pan and a
tilt servo each, a camera, and a policy on a PC that runs a detector on
the camera and points both eyes. The same pieces fit any robot. It runs on
the wall clock: the robots keep their own time and the policy keeps up. To
step a simulator instead, see [Choosing a clock](clocks.md).

Robots, inputs and actions are declared with builders: `parts.joints(...)`
for what a robot has, `inputs.camera(...)` and `actions.joints(...)` for
what a policy reads and drives. Each returns the plain dict it stands for,
and a dict you write by hand, load from a file or get from another language
is accepted wherever a builder's is:

```python
from rlmesh.realtime import actions, inputs

inputs.camera("webcam:cam")
# {"kind": "camera", "at": "webcam:cam"}
actions.joints("eye-1:eyes", command="position", hz=50, chunk=4)
# {"kind": "joints", "at": "eye-1:eyes", "command": "position", "hz": 50, "chunk": 4}
```

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
from rlmesh.realtime import actions, inputs, parts

# What the detector saw, recorded and shown in the cockpit beside the eyes.
HAND = {"hand": parts.custom("yolo/hand", hz=15, fields={"xy": parts.array(np.float32, 2)})}

tracker = realtime.Policy(
    hz=15,
    inputs={"cam": inputs.camera("webcam:cam")},
    actions={
        "left": actions.joints("eye-1:eyes", command="position", hz=50, chunk=4),
        "right": actions.joints("eye-2:eyes", command="position", hz=50, chunk=4),
    },
    session="",
    name="tracker",
    publishes=HAND,
)
center = np.zeros(2, np.float32)

def predict(obs, policy):
    xy = detect(obs["cam"])  # your model: the hand's (x, y) in 0..1, or None
    target = center if xy is None else np.array(
        [(xy[0] - 0.5) * 1.2, (0.5 - xy[1]) * 0.8], np.float32
    )
    if xy is not None:
        policy.publish("hand", xy=np.asarray(xy, np.float32))
    chunk = np.tile(target, (4, 1))  # [chunk, (pan, tilt)] in radians
    return {"left": chunk, "right": chunk}

with tracker:
    print(tracker.bind())  # the binding table: what is bound, and how
    tracker.run(predict)
```

`bind()` with no robot binds the whole session's view, so one policy
drives every eye it names. Each call of `predict` gets the newest camera
frame as an RGB `uint8 [480, 640, 3]` array and returns a chunk of 4
targets 20 ms apart, more than one replan period at 15 Hz, so an eye
never runs dry between calls. The eye's edge plays the chunk, clamps it to
the limits the board declared, and holds the last target when no new
chunk comes. `run` passes the policy itself as a second
argument when `predict` has two required parameters, so it can publish
what it saw; a `predict(obs)` gets the observation alone. `obs.age_ms("cam")`
says how old the frame was when the policy took it.

Another policy, or a viewer, reads what the tracker publishes by pointing
at the declaration, so its kind and fields are written once:

```python
viewer = realtime.Policy(
    hz=15, inputs={"hand": inputs.of(HAND["hand"], at="tracker:hand")}, session=""
)
```

In Rust, `Policy::builder()` takes the same entries from the
`rlmesh_session::spec::inputs` and `actions` builders, or as JSON, and
builds the `Policy` that `Controller::bind` takes:

```rust
use rlmesh_session::spec::actions::{self, JointCommand};
use rlmesh_session::spec::{Policy, inputs};

let eye = |at: &str| {
    let joints = actions::joints().at(at).command(JointCommand::Position);
    joints.hz(50).chunk(4)
};
let tracker = Policy::builder()
    .hz(15)
    .input("cam", inputs::camera().at("webcam:cam"))
    .action("left", eye("eye-1:eyes"))
    .action("right", eye("eye-2:eyes"))
    .build()?;
```

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
clears it. `camera.held` (or a policy's `held`) lists every robot held, why
and since when.

**When a robot closes** (its program exits cleanly, or restarts), a
policy bound to it keeps running: `policy.events()` says `input_lost`
(`input "q" lost its publisher: ur10e left the session; ...`), its starts
are skipped as `publisher_gone`, and when a robot of that name declares its
parts again the policy binds to them on its own (`rebound`) and drives it
once it is re-armed. It waits for as long as the robot stays away.

**An input the policy can do without** is declared
`optional=True, allow=["blank"]` (an action: `allow=["drop"]`): with its
part absent the policy binds anyway and the input reads blank; `optional`
without that allowance is refused at bind. `allow=["hold"]` is about
staleness, not absence: a sample older than `max_age_ms` reaches the policy,
flagged, instead of the start being skipped, while its publisher is still in
the session. Once the publisher leaves, an optional input allowed `blank`
reads blank until it is back (`input_lost` says so), whether or not it
allows `hold`; any other input's starts are skipped as `publisher_gone`.

**When a policy drops off** (a crash, a lost link, or a new process
joining under its name), the host ends only the leases it held. Those
parts go to their declared safe behaviour, as after a release; the session
stays armed, every other policy keeps driving, and nothing needs re-arming.
The host logs who was lost and which `robot:part`s were released.

**When an operator takes a robot over** from a running model (a policy of
a higher priority class, `priority="operator"`, acquires its parts), the
model's run is parked on those parts: when the operator releases them, or
their process dies, the model does **not** take the robot back by itself,
and the parts go to their declared safe behaviour. A robot never starts
moving because someone let go of a controller. The run keeps going: its
starts are skipped as `parked`, `policy.events()` says `parked` once
(`ur10e:arm was taken by operator at Operator; this policy stays stopped
until policy.resume()`), and `policy.stats()["parked"]` lists the parts.
Hand them back from any thread:

```python
thread = threading.Thread(target=policy.run, args=(predict,))
thread.start()
...                       # an operator jumps in, then lets go
policy.resume()           # ["ur10e:arm"]: the model asks for it again
```

Or the operator hands them back as it lets go, from its own process:

```python
operator.release(to="model")
# [{"to": "model-1", "parts": ["ur10e:arm"], "taken": True}]
```

The model it took those parts from resumes them, and only them, as if its
own `resume()` had been called; its `events()` say `handed_back` (with
`part` and `by`), and `stats()["parked"]` drops them. A plain `release()`
still leaves it parked, and so does an operator whose process dies. The call
waits, up to the policy's `timeout`, for the model to hold the parts again
(`taken`). Between the release and the model's first chunk the robot does
what it does after any release, then after any new holder: a position part
holds its last target and any other runs its safe behaviour, for at most the
part's `hold_max` from when the operator's last action ran out, until the
model's lease lands (the model asks for it as soon as it sees the release:
one notice and one grant); from then the robot holds, untimed, until the
model's first chunk. Past `hold_max` the part goes to its safe behaviour,
unlatched, and the model's lease resumes it.

A hand-back is refused (`SessionRefused`), and nothing is released, when a
part was taken from nobody (no model is parked on it), when the robot is
held (a hand-back never clears a hold), or when the model's process or the
session host predates hand-backs (`INCOMPATIBLE`: they would read it as a
plain release). It goes only to the policy the operator's lease preempted,
never to another participant.

Only a preemption parks: a lapsed heartbeat, a disarm and arm, a robot
held and re-armed or restarting, or another policy of the same class
holding the part first leave the model asking for it again on its own.
`resume=realtime.Resume.AUTO` takes the robot back the moment it is free,
however the operator let go, its process dying too; prefer the default and
a hand-back. In Rust it is `controller.resume()`,
`controller.release_to(PriorityClass::Model)` and
`RunConfig::default().resume(Resume::Auto)`; in Node, `policy.resume()` and
`{ resume: 'auto' }`.

## 6. Watch it: the cockpit

```sh
rlmesh cockpit default        # on the PC
rlmesh cockpit pc.local       # on a paired laptop
```

The cockpit joins as an observer and shows every part's declared and
measured rate, the lease and mode timeline, and **why each robot is or
isn't moving**: its edge mode with the reason (`queue_dry`: no valid chunk
within its hold time), the chunks it refused and why (`too_old`,
`no_lease`, and for `clock_uncertain` the measured error against the
part's bound), and each link's heartbeat age and jitter. It
has Arm, Disarm, E-stop and Keep.

**From a program**, any robot or policy reads the same link health the
cockpit shows, a viewer that only reads too, while the session runs. The
first call subscribes and waits for the host's next publication (once a
second); after that `health()` returns the newest at once, and
`health(wait=True)` waits for the next one:

```python
health = viewer.health()
for name, link in health["links"].items():
    print(name, link["rtt_ms"], link["jitter_ms"], link["rejected_by"])
# ur10e 0.21 0.03 {'stale_token': 2}

while True:
    links = viewer.health(wait=True)["links"]
    show(links["eye-1"]["last_chunk_applied"], links["eye-1"]["rejected_by"])
```

Each link, keyed by participant name, has its round trip, jitter,
heartbeat gaps and clock mapping (durations in ms, as in `stats()`), what a
robot's edges accepted and applied last and refused by reason, the parts it
publishes and reads, and a policy's run counters. In Node it is
`await policy.health()` and `await policy.health({ wait: true })`; in Rust
`participant.health(timeout)` and `participant.next_health(timeout)`.

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

## Shadow runs: watch a model before arming

A policy that commands something calls `predict` only while it holds a
lease, so on a disarmed session it does nothing, and anything reading the
parts it publishes waits. Run it in shadow mode to watch it first:

```python
def predict(obs, policy):
    xy = detect(obs["cam"])
    policy.publish("hand", xy=xy)  # published while disarmed too
    chunk = np.tile(gaze(xy), (4, 1))
    return {"left": chunk, "right": chunk}

with tracker:  # the tracker of section 5
    tracker.bind()
    tracker.run(predict, mode=realtime.RunMode.SHADOW,
                on_step=lambda step: step.shadow and show(step.actions))
```

While it cannot command (the session not armed, an operator holding the
eyes, an eye held, parked) it still calls `predict` on the same rate and inputs,
publishes its own parts, and converts and checks its actions as for
sending, so a model error or an action the binding refuses stops the run
then, not at arming. It sends nothing. Once armed and granted its leases,
the same run starts sending from the next start, with inputs read after
the lease was granted; a viewer bound to `tracker:hand` reads it throughout.
`policy.events()` says `shadowing` (and why) and `commanding` at each
change, `policy.stats()["shadow_predictions"]` counts the starts predicted
in shadow, and `on_step` receives each start's actions as the edges would
take them (`step.actions`), marked `step.shadow`. A recording shows no
chunk and no dispatched action for a shadow start.

The default, `RunMode.COMMAND`, never calls `predict` without a lease,
since a model that keeps state may expect each call to act on the robot.

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

## The datagram path for boards

Over Wi-Fi a board loses packets. On TCP one lost packet holds back
everything behind it until it is sent again, a few hundred milliseconds
later, and then the backlog arrives at once: chunks that are now too old
and are refused, then states the policy has no use for. Everything a board
streams is latest-wins anyway, so by default the host carries it over UDP
instead:

```sh
rlmesh host                           # --board-data udp, the default
rlmesh host --board-data tcp          # or RLMESH_BOARD_DATA=tcp: TCP only
```

The host also listens on UDP at its realtime port (7448 by default:
`RLMESH_PORT + 1`) and offers that in each board's handshake. That UDP port
must be reachable from the boards for the faster path: open it in the
firewall. When it is not, the host moves each board to TCP (below), and
when it can't open the port at all (another process holds it), it warns
once and runs on TCP alone. A host in your own process (Python's `Host`)
does the same on its own port plus one; a session without boards sees no
difference. It reads `RLMESH_BOARD_DATA` as `rlmesh host` does, so where UDP
is forbidden (a sandbox, a locked-down network) one variable keeps every host
an env, a model or a robot starts for itself on TCP; `Host(board_data="tcp")`
(or `realtime.BoardData.TCP`) chooses for one host. A board that takes it gets its realtime parts' action
chunks over UDP, and sends their states and `_applied` entries back the
same way; the host relays them to and from the session, so policies, the
recorder and the cockpit see the same samples. Everything else stays on
TCP: joining, leases, keys, device views, e-stop, heartbeats, and any part
declared `priority: control`. A virtual-clock session keeps TCP, as does a
board built before this path or one told not to take it
(`robot.useDatagram(false)` in a sketch, after `begin()`).

Each datagram carries exactly the sealed message the session would carry,
so it is authenticated (and encrypted, beyond loopback) and replay-checked
as before; one that does not open is dropped and counted. The host sends a
board nothing before it opened a probe from it, sealed under that board's
own key, and follows a board whose address changes with its next probe. A
part whose chunks or samples could be larger than one datagram (1400
bytes) stays on TCP, and the host says so: send fewer actions per chunk to
move it over.

The host decides when the path is broken, from what each end counts: with
the board's heartbeats arriving, one end sent datagrams for at least two
seconds (at least three of them) and the other opened none. Silence can't
tell a broken path from a policy that sent nothing, and loss that lets some
through is loss, so neither moves anything. When the path is broken the
host moves the board's realtime parts back to TCP, and tries the path
again, in the same attach, once the board's probes arrive: 5 s later at
first, doubling up to 2 minutes while it keeps breaking. To measure the path
itself, `rlmesh host --board-data udp-only` (or `RLMESH_BOARD_DATA=udp-only`)
never moves a board back: loss is simply loss, and the host only warns.
Restart the host with `--board-data tcp` to put every board back on TCP:
they switch as they re-attach.

Each attach logs the path the robot takes. The counters to compare the two
paths with are the same on both:

| Where                                                                                                                                                        | What                                                                                                                                                                                                                                                                                                                                                                  |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| A sketch: `robot.printChunkTiming(Serial, part)`, `robot.printDatagram(Serial)`; C: `rlmesh_device_robot_chunk_timing`, `rlmesh_device_robot_datagram_stats` | Per commanded part: chunks received and accepted, refusals by reason, losses (gaps in their seqs), lateness against their slot (log2 buckets in ms, and the maximum) and the longest gap between accepted chunks. The board's end of the datagram path: sent, received, lost, reordered, unopened, its moves back to TCP and onto UDP again, and the seconds on each. |
| Python: `robot.edge_stats(path)`, `Host.health()`                                                                                                            | The same per part for a Python robot: `received`, `lost`, `late_ms`, `late_max_ms`, `gap_max_ms`, beside `accepted` and `rejected_by`; a hosting process's `Host.health()` has every participant's row, as below.                                                                                                                                                     |
| `host/health`, the cockpit and recordings (`/_session/health`); `rlmesh host` logs                                                                           | Each board's chunk timing and accepted count (its heartbeats), the path it rides, both ends' datagram counters, and (`rlmesh host`'s health log, `Host.health()`'s `path_time`) its seconds on each path and the host's moves between them.                                                                                                                           |

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
- A commanded part refuses chunks while a clock's error is above its
  `max_clock_error_ms`: by default a quarter of its `max_action_age_ms`
  (50 ms at the default 200 ms), never below 5 ms, and for a command group
  the smallest of its members. The host refuses a value above half of
  `max_action_age_ms`. A chunk counts as older by the error of both clocks,
  so raising the bound never loosens the age limit. A refusal says what it
  measured against which bound, in the cockpit and in `edge_stats`.
- Beyond loopback the host encrypts every sample and command; see the
  [threat model](threat-model.md) for what it does and doesn't defend.

## Next

- [Choosing a clock](clocks.md): lockstep for a simulator you step, the
  wall clock for hardware.
- The [Python API](https://rlmesh.dev/docs/) (`rlmesh.realtime`;
  `rlmesh.realtime.parts`, `inputs` and `actions` for every kind;
  `rlmesh.realtime.results` for what each call returns) and its
  [Node twin](node.md).
- [The CLI reference](reference/cli.md): `rlmesh host`, `devices`,
  `pair`, `cockpit`.

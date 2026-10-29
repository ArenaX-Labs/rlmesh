# MuJoCo UR10e: a policy, keyboard teleop pre-emption and an MCAP recording

The v0.2 milestone demo (protocol v2). Three participants share one session:

- **robot** (this process): the session host, an MCAP recorder, and a simulated UR10e. Physics and the edge tick at 1 kHz: each tick publishes `arm` (pos, vel, effort) and `wrist/ft`, applies what `robot.due("arm")` returns, and steps MuJoCo. A second thread renders the wrist camera offscreen (640x480 u8, 20 Hz, JPEG on the wire).
- **policy** (a child process): a scripted reach that reads the joints, the newest 10 ms of the 1 kHz wrist force/torque (`history: 10`) and the wrist camera, and sends 12 x 1 ms chunks at 100 Hz through `Policy.run`. `--policy reach` drops the force window.
- **operator** (a child process): teleop at `operator` class (`--operator haptic`: the §15.6 shape, 1 kHz with the wrench every sample and one-action chunks). Taking over pre-empts the policy's lease; the edge fences the policy's chunks and holds its last target until the operator's first chunk. Handing back lets the policy re-request the lease; it resumes under a fresh token.

## Run it

```sh
uv sync --group mujoco      # mujoco and the mcap reader, outside the core dependencies
uv run python examples/python/v2/ur10e_mujoco/demo.py --seconds 10
```

The default run is headless and scripted: the operator takes over at 40% of the run, jogs `wrist_1_joint` and `shoulder_pan_joint`, and hands back at 65%. At the end the demo reads the recording back and prints a report: chunk ages per lease holder, hold ticks, misses, commands applied per second, the takeover and hand-back as the robot saw them, and the MCAP check. `--json` prints the full report; the exit code is non-zero when a check fails.

| Option                     | Default    |                                                                                                                                                                                                                                 |
| -------------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--keys`                   | `scripted` | `interactive` reads the terminal: `t` take over / hand back, `1`-`6` pick a joint, `+`/`-` jog it 0.1 rad, `q` quit. `none` runs the policy alone. Or a script, `"2.0:t,2.3:4,2.4:+,5.0:t"` (seconds after the operator binds). |
| `--camera`                 | `jpeg`     | `raw` (u8 on the wire) or `off`.                                                                                                                                                                                                |
| `--replan-hz`, `--horizon` | 100, 12    | The policy's replan rate and chunk length in 1 ms actions (§9.2: the horizon covers a replan period plus the link).                                                                                                             |
| `--record`                 | a temp dir | The MCAP file.                                                                                                                                                                                                                  |
| `--view`                   | off        | MuJoCo's passive viewer (needs a display).                                                                                                                                                                                      |

One role per terminal works too: `demo.py robot --seconds 120` prints the session address, then `demo.py policy --address ADDR` and `demo.py operator --address ADDR --keys interactive`.

Rendering uses EGL (`MUJOCO_GL=egl`); without a GPU, Mesa's llvmpipe renders a frame in about 10 ms on two threads (`LP_NUM_THREADS=2`, set by the demo). `--camera off` needs no OpenGL at all.

## The recording

`rlmesh.v2.Recorder` attaches as an observer and writes each envelope as it travelled (§13): `/ur10e/{path}` samples, `/ur10e/arm/_applied` (what the edge dispatched each tick), `/ur10e/arm/_chunks` (every chunk sent to the edge), `/_session/authority` and `/_session/bindings`. Schemas are the `rlmesh.session.v2` protobuf descriptors, so Foxglove and the `mcap` tools read it. `rlmesh.v2.read_recording(path)` counts messages per topic, checks time order and decoding, and lists the lease timeline.

## The model

`ur10e.xml` is derived from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)'s `universal_robots_ur10e` (BSD-3-Clause, see `LICENSE-ur10e`): the same kinematics, inertials, joints and position actuators, drawn with its collision shapes instead of meshes, plus a floor, a table, cameras and a wrist force/torque sensor. `--model` loads another MJCF with the same joint names, such as Menagerie's own `ur10e.xml` with its meshes.

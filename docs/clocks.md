# Choosing a clock

RLMesh connects a model to a world. The one thing you have to decide is
**who owns time**, and the thing you are wrapping usually decides it for you.

| What you have                                                  | Clock        | Does the world wait for the model? |
| -------------------------------------------------------------- | ------------ | ---------------------------------- |
| A simulator you step yourself (a Gymnasium env, a MuJoCo loop) | **lockstep** | Yes, for as long as it takes       |
| Hardware, or an engine that runs whether you step it or not    | **wall**     | No                                 |

The world picks the clock, in one argument, where you declare it:
`realtime.Robot(..., clock=realtime.Clock.LOCKSTEP)`, or the wall clock by
default. A robot says which when it joins a session, and the model doesn't
change: the same `realtime.Policy` and the same `predict` run on both
clocks, and the run follows the robot it is bound to.

```python
reach.bind("sim").run(predict, episodes=20)   # lockstep: evaluated, repeatable
reach.bind("ur10e").run(predict)               # wall clock: drives the real robot
```

The session core also has a virtual clock, which keeps the world waiting
but charges the model a set delay; it is not yet available from Python.

## What each clock promises

|                | Lockstep                                                                             | Wall                                                                                           |
| -------------- | ------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------- |
| Time advances  | one step per action                                                                  | always                                                                                         |
| A pause        | changes nothing                                                                      | the world moves on                                                                             |
| A slow model   | costs you waiting, nothing else                                                      | acts late by however slow it was; a chunk older than the part's `max_action_age_ms` is refused |
| Same run twice | the same, given the same seeds and a deterministic `step`                            | never exactly                                                                                  |
| Arming         | `robot.serve` arms the robot; a policy's lease is taken for it                       | you arm the session: `robot.arm()`, or **Arm** in the cockpit                                  |
| Limits         | in the action space, and applied to every command                                    | applied by the robot's edge to every chunk, whatever a policy sends                            |
| Safe behaviour | not used: nothing runs between steps                                                 | runs on the robot when no one commands it                                                      |
| Driven by      | a `realtime.Policy` by episodes, `RemoteEnv`, or anything that steps a Gymnasium env | a `realtime.Policy` for a duration                                                             |
| Use it to      | train and evaluate                                                                   | run the real thing                                                                             |

## Wrapping what you have

You keep your code. RLMesh takes one object at the edge of it.

**A Gymnasium env: lockstep.** Nothing to choose.

```python
import gymnasium as gym
import rlmesh

rlmesh.EnvServer(gym.make("CartPole-v1"), "127.0.0.1:5555/cartpole").serve()
```

**Your own simulator loop: lockstep.** Say what the robot has, and give
RLMesh the function that advances it one step.

```python
import numpy as np
from rlmesh import realtime
from rlmesh.realtime import parts

SERVOS = parts.joints(
    ["pan", "tilt"], hz=50, fields=["pos"], command="position",
    safe=parts.Safe.HOLD, limits={"pos": [[-0.6, 0.6], [-0.4, 0.4]]},
)

robot = realtime.Robot("gimbal", {"servos": SERVOS}, clock=realtime.Clock.LOCKSTEP)
q = np.zeros(2, np.float32)

def step(commands):
    q[:] = simulate(commands["servos"])   # advance your simulation one step
    robot.publish("servos", pos=q)
    return reward(q), done(q), False      # reward, terminated, truncated

def reset(seed, options):
    q[:] = 0
    robot.publish("servos", pos=q)

robot.serve(step, reset)                  # step(commands) is called once per step
```

Whoever steps it owns its time: a `realtime.Policy` (see
[one policy on both clocks](#one-policy-on-both-clocks)), or a Gymnasium
loop, since any lockstep robot is a Gymnasium env at its address, its
observations and actions keyed by part path:

```python
from rlmesh.numpy import RemoteEnv

env = RemoteEnv(robot.address)
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step({"servos": np.array([0.2, 0.1], np.float32)})
```

**Hardware, or an engine that runs free: wall.** The same declaration; you
publish what you measure and apply what is due, at your own rate.

```python
robot = realtime.Robot("gimbal", {"servos": SERVOS})   # the wall clock is the default
servos = robot.part("servos")
rate = realtime.Rate(50)
while True:
    servos.publish(pos=read_position())
    target = servos.command()  # None when nothing is due
    if target is not None:
        move(target)
    rate.sleep()
```

A game engine belongs here when its world keeps running between your
actions (other players, physics you can't pause). When you can fix its time
step and hold it between steps, lockstep is the honest choice, and runs
repeat.

## Driving a wall-clock robot

A policy says what it reads and what it drives, by name, and returns one
array per thing it drives:

```python
from rlmesh.realtime import actions, inputs

tracker = realtime.Policy(
    hz=30,
    inputs={"cam": inputs.camera("front-cam:cam")},
    actions={
        "left": actions.joints("gimbal-left:servos", command="position", hz=50, chunk=4),
        "right": actions.joints("gimbal-right:servos", command="position", hz=50, chunk=4),
    },
    session="",
)

def predict(obs):
    aim = find_hands(obs["cam"])
    return {"left": aim.left, "right": aim.right}

tracker.bind()
tracker.run(predict)
```

`predict` is called at 30 Hz with the newest frame, and what the wall clock
adds is what it can be handed: an input may be older than you would like
(`obs.age_ms("cam")`), and the robot's edge refuses a chunk that arrives
too late. See [real-time sessions](realtime.md) for the whole setup.

## One policy on both clocks

`Policy.bind` learns the clock of the robot it binds to from what the robot
declared when it joined the session, so it knows a lockstep robot before
that robot serves a step; `policy.clock` says which. A robot that declares
nothing, a board included, is on the wall clock. `Policy.run` then follows
the clock, and `predict` is the same function either way:

```python
reach = realtime.Policy(
    hz=25,
    inputs={"q": inputs.joints("arm", fields=["pos"])},
    actions={"arm": actions.joints("arm", command="position", hz=100, chunk=4)},
    session=host,
)

stats = reach.bind("sim").run(predict, episodes=20, seed=7)
for episode in stats["episodes"]:
    print(episode["return"], episode["steps"], episode["terminated"])
```

On a lockstep robot:

- **Episodes, not time.** `run` runs `episodes` episodes (one by default)
  and returns when the last one ends or on `stop()`. Each episode starts
  with the robot's `reset`. `duration=` raises `ValueError` at once.
- **Seeds repeat.** `seed=` is a base seed: each episode's reset seed is
  derived from it, the robot and the episode's number exactly as a model
  run with a base seed derives it (`Model.run_routes(..., base_seed=)`, or
  `base_seed` in Rust's `RunOptions`), so the same seed runs the same
  episodes again, through either API. Without `seed=` the resets are
  unseeded.
- **A chunk is that many steps.** One `predict` returns a chunk of H
  actions per action entry, and the robot takes H steps, one action each,
  stopping early at the end of an episode. Their rewards add up to the
  episode's return.
- **`hz` is session time, not real time.** Each step stands for one action
  period of the policy (`1/hz` of its actions; the policy's `hz` when they
  declare none), so input ages and history windows mean what they mean on
  the wall clock. Nothing waits on a real clock.
- **Nothing is ever stale or missing.** Every start sees each input as the
  robot last published it, and is never skipped. `obs.age_ms(name)` is
  session time: the steps since the robot published that input.
- **Arming and the lease are taken care of.** The robot arms itself when it
  serves, and the policy takes its lease when it binds. `mode=RunMode.SHADOW`
  and `resume=` raise `ValueError`, as do `acquire()`, `release()`, `send()`
  and the other wall-clock verbs (`SessionException`).
- **One lockstep robot per policy.** A policy whose binding names several
  robots, any of them lockstep, is refused at bind with `BindingRefused`:
  stepping several lockstep robots together needs a rule for who advances
  time, and none exists yet.

On the wall clock `run` keeps its meaning: it drives the robot until
`duration` or `stop()`, and `episodes=` and `seed=` raise `ValueError` at
once. Both clocks return one dictionary with a `clock` key: `"wall"` with
the counters of `realtime.results.PolicyStats`, or `"lockstep"` with
`starts`, `chunks_sent` and `episodes`, one `{"return", "steps",
"terminated", "truncated"}` per episode (`realtime.results.LockstepStats`).

A Gym env served with `rlmesh.EnvServer` is a lockstep robot too, its parts
of kinds `gym/box`, `gym/discrete` and so on, so a policy binds it by kind
and runs it the same way:

```python
balance = realtime.Policy(
    hz=50,
    inputs={"x": {"kind": "gym/box", "fields": {"value": ["float32", [4]]}}},
    actions={"push": {"kind": "gym/discrete", "fields": {"value": ["int64", []]}}},
    session="127.0.0.1:5555",
)
lean = lambda obs: {"push": np.array([int(obs["x"][2] > 0)])}  # chunk 1: shape [1]
balance.bind("cartpole").run(lean, episodes=10)
```

Its resets are a model run's: besides the seeds, an env that declared the
`trial_index` reset option (`EnvFactory.reset_options`) gets each episode's
0-based ordinal in it, numbered as `Model.run` numbers them, so it sweeps
the same initial states or goals either way. An env that did not declare it
is never handed the key.

In Rust, `rlmesh::realtime::bind` returns a `BoundPolicy`, `Wall` (the
session's `Controller`) or `Lockstep`, and `BoundPolicy::run` takes the same
`Predictor` on either with `RunSettings` (`duration`, `mode` and `resume`
for the wall clock, `episodes` and `seed` for lockstep), refusing the other
clock's settings the same way.

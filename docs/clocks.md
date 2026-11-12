# Choosing a clock

RLMesh connects a model to a world. The one thing you have to decide is
**who owns time**, and the thing you are wrapping usually decides it for you.

| What you have                                                  | Clock        | Does the world wait for the model? |
| -------------------------------------------------------------- | ------------ | ---------------------------------- |
| A simulator you step yourself (a Gymnasium env, a MuJoCo loop) | **lockstep** | Yes, for as long as it takes       |
| Hardware, or an engine that runs whether you step it or not    | **wall**     | No                                 |

The world picks the clock, in one argument, where you declare it:
`realtime.Robot(..., clock=realtime.Clock.LOCKSTEP)`, or the wall clock by
default. What drives it follows from that: a lockstep robot is stepped as a
Gymnasium env, and a wall-clock robot is driven by a `realtime.Policy`.

The session core also has a virtual clock, which keeps the world waiting
but charges the model a set delay; it is not yet available from Python.

## What each clock promises

|                | Lockstep                                                  | Wall                                                                                           |
| -------------- | --------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| Time advances  | one step per action                                       | always                                                                                         |
| A pause        | changes nothing                                           | the world moves on                                                                             |
| A slow model   | costs you waiting, nothing else                           | acts late by however slow it was; a chunk older than the part's `max_action_age_ms` is refused |
| Same run twice | the same, given the same seeds and a deterministic `step` | never exactly                                                                                  |
| Arming         | `robot.serve` arms the robot                              | you arm the session: `robot.arm()`, or **Arm** in the cockpit                                  |
| Limits         | in the action space, and applied to every command         | applied by the robot's edge to every chunk, whatever a policy sends                            |
| Safe behaviour | not used: nothing runs between steps                      | runs on the robot when no one commands it                                                      |
| Driven by      | `RemoteEnv`, or anything that steps a Gymnasium env       | a `realtime.Policy`                                                                            |
| Use it to      | train and evaluate                                        | run the real thing                                                                             |

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

Whoever steps it owns its time. Any lockstep robot is a Gymnasium env at
its address, its observations and actions keyed by part path:

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
    due = servos.due()
    if due is not None and due.value is not None:
        move(due.value)
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
    inputs={"cam": inputs.camera("hallo-cam:cam")},
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

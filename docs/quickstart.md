# Quickstart

Serve a Gymnasium environment in one process. Connect to it from another process.

## Install

```bash
pip install --pre "rlmesh[gymnasium,numpy]"
```

## Server

Run this in the first process:

```python
import gymnasium as gym
import rlmesh

env = gym.make("CartPole-v1")
rlmesh.EnvServer(env, "127.0.0.1:5555").serve()
```

## Client

Run this in a second process:

```python
from rlmesh.numpy import RemoteEnv

env = RemoteEnv("127.0.0.1:5555")
obs, info = env.reset(seed=0)

terminated = truncated = False
while not (terminated or truncated):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

env.close()
```

The server owns the Gymnasium environment and its dependencies. The client only needs the endpoint
address and the spaces reported by the server.

## Runnable Files

From the repository root:

```bash
uv run python examples/python/quickstart/serve_gymnasium.py
```

In another terminal:

```bash
uv run python examples/python/quickstart/eval.py
```

Swap in another Gymnasium registration with `--env-id`:

```bash
uv run python examples/python/quickstart/serve_gymnasium.py --env-id Acrobot-v1
```

For the smallest custom environment object, use `examples/python/quickstart/serve.py`. It implements
a tiny Gymnasium-style `CounterEnv` without installing Gymnasium.

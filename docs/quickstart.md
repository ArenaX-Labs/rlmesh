# Quickstart

Serve a Gymnasium environment in one process. Connect to it from another process.

## Install

```bash
pip install "rlmesh[gymnasium,numpy]"
```

See {doc}`installation` for the optional extras.

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

The server owns the Gymnasium environment and its dependencies. The client needs only the endpoint address and the spaces the server reports.

## Run a Model Against It

Wrap a prediction function in `rlmesh.Model` and let it pump whole episodes:

```python
import rlmesh


def predict(observation):
    return 0


model = rlmesh.Model(predict)
result = model.run("127.0.0.1:5555", max_episodes=3)
print(result.mean_reward)
```

`run` dials the endpoint, plays three episodes, and returns a `RunResult` with per-episode rewards, `mean_reward`, and `success_rate`.

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

For the smallest custom environment object, use `examples/python/quickstart/serve.py`. It implements a tiny Gymnasium-style `CounterEnv` without installing Gymnasium.

## Next

- {doc}`user-guide/evaluation`: `run`, `session`, seeds, and reading results.
- {doc}`user-guide/models`: wrapping a predict callable or subclassing a backend `Model`.

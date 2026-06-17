# Custom Work

RLMesh works with Gymnasium registrations and Gymnasium-style Python objects. The quickstart serves
one custom environment over gRPC and connects one model worker to it.

## Custom Environment

{source}`examples/python/quickstart/serve.py <examples/python/quickstart/serve.py>` defines a tiny
`CounterEnv` without importing Gymnasium, then serves it with `EnvServer`:

```python
import rlmesh


class CounterEnv:
    observation_space = rlmesh.spaces.Discrete(5)
    action_space = rlmesh.spaces.Discrete(2)

    def __init__(self):
        self.step_count = 0

    def reset(self, seed=None, options=None):
        self.step_count = 0
        return 0, {}

    def step(self, action):
        self.step_count += 1
        observation = self.step_count % 5
        terminated = self.step_count >= 3
        return observation, 1.0, terminated, False, {"action": action}

    def close(self):
        pass


server = rlmesh.EnvServer(CounterEnv(), "127.0.0.1:5555")
print(f"serving CounterEnv on {server.address}")
server.serve()
```

Replace `CounterEnv` with your own environment object if it has the same shape: an
`observation_space`, an `action_space`, `reset(seed=None, options=None)`, `step(action)`, and
`close()`. Run it:

```bash
uv run python examples/python/quickstart/serve.py
```

## Custom Model

{source}`examples/python/quickstart/model.py <examples/python/quickstart/model.py>` wraps a
prediction function as a model worker. `predict` takes an observation and returns an action; `Model`
runs episodes against the served endpoint:

```python
from rlmesh.numpy import Model


def predict(observation):
    return 0


model = Model(predict)
model.run("127.0.0.1:5555", max_episodes=1)
```

Run it against the server:

```bash
uv run python examples/python/quickstart/model.py --episodes 1
```

## Drive It Yourself

If you would rather step the environment by hand instead of handing it to a `Model`,
{source}`examples/python/quickstart/eval.py <examples/python/quickstart/eval.py>` opens a
`RemoteEnv` and runs a sampled-action loop:

```python
from rlmesh.numpy import RemoteEnv

env = RemoteEnv("127.0.0.1:5555")
obs, info = env.reset(seed=0)
for step in range(1, 65):
    action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)
    if term or trunc:
        break
env.close()
```

That is the separation: the environment serves observations, and the model worker (or your own loop)
returns actions.

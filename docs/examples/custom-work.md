# Custom Work

RLMesh works with Gymnasium registrations and Gymnasium-style Python objects. This walks the quickstart end to end: serve one custom environment over gRPC, connect one model worker, then drive the loop by hand. Everything here uses the NumPy backend. For the full authoring surface see {doc}`../user-guide/environments` and {doc}`../user-guide/models`.

## Serve a custom environment

{source}`examples/python/quickstart/serve.py <examples/python/quickstart/serve.py>` defines a tiny `CounterEnv` without importing Gymnasium, then serves it with `EnvServer`.

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

Any object with the same shape serves the same way: an `observation_space`, an `action_space`, `reset(seed=None, options=None)`, `step(action)`, and `close()`. Run it:

```bash
uv run python examples/python/quickstart/serve.py
```

## Serve a custom model

{source}`examples/python/quickstart/model.py <examples/python/quickstart/model.py>` wraps a prediction function as a model worker. `predict` takes an observation and returns an action; `Model` runs episodes against the served endpoint.

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

## Drive the loop yourself

To step the environment by hand instead of handing it to a `Model`, {source}`examples/python/quickstart/eval.py <examples/python/quickstart/eval.py>` opens a `RemoteEnv` and runs a sampled-action loop.

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

That is the separation: the environment serves observations, and the model worker (or your own loop) returns actions.

## Where next

- {doc}`../user-guide/environments`: authoring environments: tags, params, and variants.
- {doc}`../user-guide/models`: wrapping a predict callable or subclassing a backend `Model`.
- {doc}`../user-guide/remote-clients`: connecting a `RemoteEnv` from another process.
- {doc}`multiple-endpoints`: running one evaluator across several endpoints.

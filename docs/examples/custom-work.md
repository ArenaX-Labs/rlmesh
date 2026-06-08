# Custom Work

RLMesh works with Gymnasium registrations and Gymnasium-style Python objects.

## Custom Environment

`examples/python/quickstart/serve.py` defines a tiny `CounterEnv` without importing Gymnasium:

```python
class CounterEnv:
    observation_space = rlmesh.spaces.Discrete(5)
    action_space = rlmesh.spaces.Discrete(2)

    def reset(self, seed=None, options=None):
        return 0, {}

    def step(self, action):
        return obs, reward, terminated, truncated, info
```

Replace `CounterEnv` with your existing environment object if it has the same shape:

- `observation_space`
- `action_space`
- `reset(seed=None, options=None)`
- `step(action)`
- `close()`

Run it:

```bash
uv run python examples/python/quickstart/serve.py
```

## Custom Model

`examples/python/quickstart/model.py` wraps a small prediction function:

```python
def predict(obs):
    return 0


model = Model(predict)
model.run("127.0.0.1:5555", max_episodes=1)
```

Run it against a server:

```bash
uv run python examples/python/quickstart/model.py --episodes 1
```

That is the basic separation: the environment serves observations, and the model worker returns
actions.

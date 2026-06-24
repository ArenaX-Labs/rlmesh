# Connect a Remote Environment

Use a remote client when the environment is served by another process.

```python
from rlmesh.numpy import RemoteEnv

env = RemoteEnv("127.0.0.1:5555")
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(
    env.action_space.sample()
)
env.close()
```

The remote client keeps the usual methods:

- `reset(seed=None, options=None)`
- `step(action)`
- `render()`
- `close()`

It also exposes the environment contract and spaces reported by the server:

```python
print(env.env_contract)
print(env.spec)
print(env.observation_space)
print(env.action_space)
```

`env.spec` is an alias for `env.env_contract`. See {doc}`../api/contracts` for the contract fields.

## Vector Clients

Use `RemoteVectorEnv` when one endpoint serves multiple environment instances:

```python
from rlmesh.numpy import RemoteVectorEnv

envs = RemoteVectorEnv("127.0.0.1:5555")
observations, infos = envs.reset(seed=0)
actions = [envs.single_action_space.sample() for _ in range(envs.num_envs)]
observations, rewards, terminations, truncations, infos = envs.step(actions)
envs.close()
```

`RemoteEnv` accepts endpoints with exactly one environment. If the endpoint reports more than one environment, connect with `RemoteVectorEnv` instead.

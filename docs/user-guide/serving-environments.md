# Serve an Environment

Use `rlmesh.EnvServer` to expose a Gymnasium-style environment over an endpoint.

```python
import gymnasium as gym
import rlmesh

env = gym.make("CartPole-v1")
server = rlmesh.EnvServer(env, "127.0.0.1:5555")
server.serve()
```

`serve()` blocks. Use `start()` when the current process needs to keep doing other work:

```python
server = rlmesh.EnvServer(env, port=5555)
server.start()
print(server.address)
server.wait()
```

## Environment Shape

RLMesh works with standard Gymnasium environments from `gym.make(...)` and with custom objects that
follow the same shape:

- `observation_space`
- `action_space`
- `reset(seed=None, options=None) -> (obs, info)`
- `step(action) -> (obs, reward, terminated, truncated, info)`
- `close()`

Vectorized environments can expose `num_envs`, `single_observation_space`, and
`single_action_space`.

Common Gymnasium wrappers can stay in place. RLMesh reads the spaces and calls the wrapped
environment methods through the normal Gymnasium API.

## Environment Contract

When `EnvServer` wraps the environment, it creates an {py:class}`~rlmesh.specs.EnvContract`. Clients
receive that contract during connection.

```python
server = rlmesh.EnvServer(env, "127.0.0.1:5555")
contract = server.env_contract

print(contract.id)
print(contract.observation_space.kind)
print(contract.action_space.kind)
print(contract.num_envs)
```

`server.spec` is an alias for `server.env_contract`. See {doc}`../api/contracts` for contract
fields.

## Addresses

Use an explicit address string:

```python
rlmesh.EnvServer(env, "tcp://127.0.0.1:5555")
rlmesh.EnvServer(env, "127.0.0.1:5555")
rlmesh.EnvServer(env, "unix:///tmp/rlmesh-env.sock")
```

Or use helpers:

```python
rlmesh.EnvServer(env, host="127.0.0.1", port=5555)
rlmesh.EnvServer(env, path="/tmp/rlmesh-env.sock")
```

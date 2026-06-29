# Connect a Remote Environment

A remote client drives an environment that another process serves. It speaks the same `reset` / `step` / `render` / `close` loop as a local Gymnasium environment, so code that runs against `gym.make(...)` runs against a `RemoteEnv` with no other change. The environment can live in a separate terminal, a container, or another machine; the client only needs its address.

```python
from rlmesh.numpy import RemoteEnv

env = RemoteEnv("127.0.0.1:5555")
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(
    env.action_space.sample()
)
env.close()
```

The import decides how values decode at the boundary. `rlmesh.numpy` returns NumPy arrays; `rlmesh.torch` and `rlmesh.jax` return their tensors; top-level `rlmesh` keeps RLMesh-native values with no array dependency. The client behavior is identical across all four. See {doc}`backends`.

## The loop and the contract

A client connects and handshakes at construction. Building `RemoteEnv(address)` dials the endpoint and fetches its {py:class}`~rlmesh.specs.EnvContract` before returning, so a constructed client is already connected and an unreachable address fails right there rather than on the first `step`.

Once connected, the client carries the usual methods:

- `reset(seed=None, options=None)`
- `step(action)`
- `render(env_index=0)`
- `close()`

It also exposes the contract and spaces the server reported, so you can shape actions and read metadata without a second round trip:

```python
print(env.env_contract)
print(env.spec)               # alias for env_contract
print(env.observation_space)
print(env.action_space)
print(env.address)            # resolved endpoint address
print(env.env_id)             # this connection's container id (UUIDv7)
```

`observation_space` and `action_space` are decoded with the same backend as the client, so `sample()` returns values you can hand straight to `step`. See {doc}`../api/contracts` for the contract fields.

`RemoteEnv` is a context manager, which is the cleanest way to guarantee the connection is released:

```python
with RemoteEnv("127.0.0.1:5555") as env:
    obs, info = env.reset(seed=0)
```

## Single vs vector clients

Pick the client that matches the endpoint's arity. A client checks the contract during the handshake and refuses the wrong one: `RemoteEnv` against an endpoint serving more than one environment raises `ValueError` pointing you at `RemoteVectorEnv`.

| Reach for         | When the endpoint serves | Spaces                                            | Step shape                                 |
| ----------------- | ------------------------ | ------------------------------------------------- | ------------------------------------------ |
| `RemoteEnv`       | exactly one environment  | `observation_space`, `action_space`               | one action in, one transition out          |
| `RemoteVectorEnv` | two or more environments | `single_observation_space`, `single_action_space` | a batch of actions in, batched results out |

```python
from rlmesh.numpy import RemoteVectorEnv

envs = RemoteVectorEnv("127.0.0.1:5555")
observations, infos = envs.reset(seed=0)
actions = [envs.single_action_space.sample() for _ in range(envs.num_envs)]
observations, rewards, terminations, truncations, infos = envs.step(actions)
envs.close()
```

`num_envs` reports the served count. `observation_space` and `action_space` on a vector client are aliases for the `single_*` spaces, and `reset` accepts either one seed or a per-environment list.

## Addresses

A client accepts the same address forms as the server, chosen by scheme:

```python
RemoteEnv("tcp://127.0.0.1:5555")
RemoteEnv("127.0.0.1:5555")
RemoteEnv("unix:///tmp/rlmesh-env.sock")
```

Or build the address from helpers, where `port` is required alongside `host`:

```python
RemoteEnv(host="127.0.0.1", port=5555)
RemoteEnv(path="/tmp/rlmesh-env.sock")
```

`address` and the helpers are mutually exclusive, and unix sockets are unavailable on Windows.

## Connection lifecycle

The connection has three stages: dial-and-handshake at construction, the `reset`/`step` exchange, and teardown.

```{mermaid}
flowchart LR
    A[construct client] --> B[dial endpoint]
    B --> C[handshake: receive contract]
    C --> D[reset / step]
    D --> D
    D --> E{teardown}
    E -->|close| F[detach this client]
    E -->|shutdown| G[ask the endpoint owner to stop]
```

Two teardown paths exist, and they differ in what they affect:

- `close()` detaches this client from the endpoint. The server keeps running for other clients.
- `shutdown(reason="owner shutdown")` requests an owner-level shutdown of the endpoint itself and returns whether the request was accepted. Use it when this client owns the server's lifetime.

## Errors and timeouts

A wrong-arity connection fails at construction with `ValueError`, after the client has already closed its dial, so there is nothing to clean up. An unreachable or refused endpoint surfaces as a transport error from the same construction call.

```{note}
The remote clients do not reconnect on their own, and the public constructors do
not expose connect or per-call timeouts. A dropped endpoint surfaces as a
transport error on the next call; rebuild the client to reconnect. The
{doc}`sandbox sessions <sandbox>`, which own the server process, set a connect
timeout internally because they retry while their container boots.
```

For the broader error model, see {doc}`error-handling`.

## Where next

- {doc}`serving-environments` — start the endpoint a client connects to.
- {doc}`backends` — choose what `step` and `reset` decode into.
- {doc}`sandbox` — get a remote client that also owns the server's container.
- {doc}`evaluation` — drive a model against a remote environment.
- {doc}`adapters` — let a model resolve its IO from the env contract.
- {doc}`../api/remote-envs` — the autodoc for the client classes.

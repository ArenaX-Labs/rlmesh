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

Common Gymnasium wrappers can stay in place. RLMesh reads the spaces and calls the wrapped methods
through the normal Gymnasium API.

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

## Readiness

RLMesh exposes two machine-readable readiness signals. Use them instead of parsing startup prints;
the human-readable output is not a stable interface.

### gRPC Health Service

RLMesh's Rust gRPC serve paths serve the standard
[`grpc.health.v1`](https://github.com/grpc/grpc/blob/master/doc/health-checking.md) health service,
separate from the env/model RPCs. Overall server health, the empty `""` service name, reports
`SERVING` once the listener accepts connections.

Any standard health client can probe it, for example with
[`grpc-health-probe`](https://github.com/grpc-ecosystem/grpc-health-probe):

```sh
grpc-health-probe -addr 127.0.0.1:5555
# status: SERVING
```

Use this for long-lived deployments and container readiness probes.

```{note}
The Rust `EnvServer` / `ModelWorker` serve paths register the health service
today. The Python `rlmesh.EnvServer` wrapper is gaining the same registration;
until that lands, use the ready file descriptor below for the Python CLI path.
```

### Ready File Descriptor (CLI)

The env-serve CLI (`python -m rlmesh._cli.serve_env`) accepts `--ready-fd <int>`. RLMesh writes one
line containing the resolved bind address, for example `tcp://127.0.0.1:54321`, then closes the file
descriptor. This works when the bind port is `0` because the line carries the resolved address.

```sh
# Open fd 3 onto a file, point --ready-fd at it, then read the address back.
# Use a dedicated fd (not stdout/1) so the address line is not mixed with the
# human-readable startup prints.
exec 3>/tmp/rlmesh-ready
python -m rlmesh._cli.serve_env --env CartPole-v1 \
  --address tcp://127.0.0.1:0 --ready-fd 3 &
exec 3>&-                            # close our copy so EOF propagates
addr=$(head -n1 /tmp/rlmesh-ready)   # the resolved bind address
echo "ready at $addr"
```

Python embedders can call `rlmesh._cli.serve_env.write_ready_fd` directly.

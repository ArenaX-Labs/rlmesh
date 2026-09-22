# Serve an Environment

`rlmesh.EnvServer` exposes one Gymnasium-style environment over an endpoint so a model in another process can drive it. This page covers the serving mechanics: the environment shape it accepts, the contract it publishes, addresses, and the two readiness signals. For authoring environments (tags, params, variants, containers) see {doc}`environments`.

```python
import gymnasium as gym
import rlmesh

env = gym.make("CartPole-v1")
server = rlmesh.EnvServer(env, "127.0.0.1:5555")
server.serve()
```

`serve()` blocks the calling thread. Use `start()` when the process needs to keep doing other work, then `wait()` to join later:

```python
server = rlmesh.EnvServer(env, port=5555)
server.start()
print(server.address)
server.wait()
```

`start()` runs the server on a background thread. `wait(timeout=None)` blocks until it stops and returns `True`, or returns `False` if the timeout elapses first; Ctrl-C interrupts it with `KeyboardInterrupt`, so put `wait()` in a `try` / `finally: server.shutdown()` (or use the context manager) to drain and close the environment on the way out. `shutdown()` stops a running server, and `EnvServer` is a context manager that shuts down on exit. A background server is shut down and joined for you if the process exits while it is still running, but that teardown is bounded and best-effort -- shut it down yourself so the drain and `env.close()` run when you expect them to. Unlike the blocking `serve()`, a background server installs no SIGINT/SIGTERM handler of its own; install one if the process needs a graceful `SIGTERM`.

## Environment shape

`EnvServer` works with environments from `gym.make(...)` and with any object that follows the same shape:

- `observation_space`
- `action_space`
- `reset(*, seed=None, options=None) -> (obs, info)`
- `step(action) -> (obs, reward, terminated, truncated, info)`
- `close()`

A vectorized environment exposes `num_envs`, `single_observation_space`, and `single_action_space`. `EnvServer` detects that shape on the raw env and serves a vector endpoint, with no separate server class to choose:

```python
envs = gym.vector.SyncVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(4)])
server = rlmesh.EnvServer(envs, "127.0.0.1:5555")
server.serve()
```

To host several independent copies of a scalar env on one endpoint, pass a list. Each entry becomes a **lane** with its own thread: the endpoint advertises `num_envs = len(list)`, and a runtime keeps one request per lane in flight, so lanes reset and step at their own pace and a slow episode on one never stalls the others. A single env is the one-lane case of the same server, same wire.

```python
server = rlmesh.EnvServer([gym.make("CartPole-v1") for _ in range(4)], "127.0.0.1:5555")
server.serve()
```

`python -m rlmesh.serve` does this for a prebuilt image when `RLMESH_NUM_ENVS` is set: `make()` runs once per lane, and tags publish per lane exactly as on the scalar path. This is the shape to use for a GPU-rendered sim: one process, one GPU, N scenes.

Common Gymnasium wrappers can stay in place. `EnvServer` reads the spaces and calls the wrapped `reset`/`step` through the normal Gymnasium API.

To publish adapter tags so a model resolves its IO with no glue, pass `tags=`. The tags are validated against the env's spaces and merged into the contract metadata. See {doc}`adapters` for the full path.

## Environment contract

When `EnvServer` wraps the environment it builds an {py:class}`~rlmesh.specs.EnvContract`. Clients receive that contract during the connection handshake, which is how a {doc}`remote client <remote-clients>` learns the spaces without importing the environment.

```python
server = rlmesh.EnvServer(env, "127.0.0.1:5555")
contract = server.env_contract

print(contract.id)
print(contract.observation_space.kind)
print(contract.action_space.kind)
print(contract.num_envs)
```

See {doc}`../api/contracts` for the contract fields.

## Action framework and device

The wire is framework-neutral, so the value backend is a client choice (see {doc}`backends`). The one declaration the server side may need is the framework the env's `step` requires its _action_ as. Pass `framework=` (`"torch"`, `"jax"`, or `"numpy"`, the default) only for a framework-strict env, one whose `step` does something like `action.to(...)`. A tolerant env can omit it. Observations need no declaration: a Torch or JAX observation, GPU included, is auto-detected and encoded either way.

```python
rlmesh.EnvServer(env, "127.0.0.1:5555", framework="torch", device="cuda:0")
```

`device=` places the incoming action on a device and is valid only with a Torch or JAX framework; NumPy and the default backend have no device. Because the wire stays neutral, the env's action framework is independent of any consuming model's framework.

## Addresses

Pass an explicit address string, choosing the transport with a scheme:

```python
rlmesh.EnvServer(env, "tcp://127.0.0.1:5555")
rlmesh.EnvServer(env, "127.0.0.1:5555")
rlmesh.EnvServer(env, "unix:///tmp/rlmesh-env.sock")
```

Or build the address from helpers:

```python
rlmesh.EnvServer(env, host="127.0.0.1", port=5555)
rlmesh.EnvServer(env, path="/tmp/rlmesh-env.sock")
```

With no address at all, the server binds `tcp://127.0.0.1:0` and picks a free port; read `server.address` for the resolved one. Unix sockets are not available on Windows.

## Declare a workflow edition

A served env declares the workflow semantics it was authored against, and keeps them across rlmesh upgrades. Set `workflow_edition` on your {class}`~rlmesh.EnvFactory` (or pass `--workflow-edition` / `ServeOptions(workflow_edition=...)` to override it for one endpoint), and the declaration rides every handshake so a connecting runtime runs the session there. Declaring nothing lets the endpoint float to whatever edition its rlmesh build is newest on, and warns once per process. The full precedence table is in {doc}`../editions/index`.

## Exposure and authentication

The RLMesh transport is plaintext gRPC. There is no TLS on any serve or connect path, and an endpoint is unauthenticated unless a bearer token is configured. A token can be set from Rust (`rlmesh::ServeOptions.token`) and from C (`RlmeshServeOptions.token`); the Python `ServeOptions` has no token field today, so a Python-served endpoint accepts every client that can reach it.

Treat reachability as the access control:

- Bind `127.0.0.1` (the default) or a `unix://` socket when the client is on the same host.
- Reach a remote peer over a tunnel, VPN, or service mesh rather than by widening the bind address.
- Use `0.0.0.0` only inside a container whose port is published to a trusted network — for example `docker run -p 127.0.0.1:50051:50051`, which keeps the published port on the host loopback.

## Startup and readiness

A served environment is reachable only once its listener is bound. The order matters when something supervises the process: bind happens after the env is constructed, and only then does a client handshake succeed.

```{mermaid}
flowchart TD
    A[EnvServer constructed] --> B[env built, spaces read into the contract]
    B --> C[listener binds the address]
    C --> D{readiness signal}
    D -->|gRPC health| E["health reports SERVING (empty service name)"]
    D -->|ready file descriptor| F[resolved bind address written, fd closed]
    E --> G[client handshake receives the contract]
    F --> G
    G --> H[reset / step loop]
```

RLMesh exposes two machine-readable readiness signals. Use them instead of parsing startup prints; the human-readable output is not a stable interface.

### gRPC health service

The Rust gRPC serve paths run the standard [`grpc.health.v1`](https://github.com/grpc/grpc/blob/master/doc/health-checking.md) health service alongside the env/model RPCs. Overall server health uses the empty `""` service name and reports `SERVING` once the listener accepts connections.

Any standard health client can probe it, for example [`grpc-health-probe`](https://github.com/grpc-ecosystem/grpc-health-probe):

```sh
grpc-health-probe -addr 127.0.0.1:5555
# status: SERVING
```

Reach for this on long-lived deployments and container readiness probes.

```{note}
The Rust `EnvServer` / `ModelWorker` serve paths register the health service
today. The Python `rlmesh.EnvServer` wrapper is gaining the same registration;
until that lands, use the ready file descriptor below for the Python CLI path.
```

### Ready file descriptor (CLI)

The env-serve CLI (`python -m rlmesh._cli.serve_env`) takes `--ready-fd <int>`. It is the lower-level sibling of the public `python -m rlmesh.serve` entrypoint: use `rlmesh.serve` to run a factory or model in a container, and `rlmesh._cli.serve_env` when a supervisor needs the `--ready-fd` readiness signal, which `rlmesh.serve` does not take. After the listener binds, RLMesh writes one line with the resolved bind address, for example `tcp://127.0.0.1:54321`, then closes the descriptor. Because the line carries the resolved address, this works even when you bind port `0`, and the close gives the reader a clean end-of-file.

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

## Where next

- {doc}`remote-clients`: connect to the endpoint you just served.
- {doc}`backends`: pick the value backend the client decodes into (NumPy, Torch, JAX, or plain Python).
- {doc}`sandbox`: let RLMesh own the environment's process and dependencies in a container.
- {doc}`adapters`: publish `tags=` so a model resolves its IO from the contract alone.
- {doc}`evaluation`: drive a model against the served environment.
- {doc}`../api/serving`: the autodoc for `EnvServer` and its options.

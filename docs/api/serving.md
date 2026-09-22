# Serving and Clients

```{note}
This is the autodoc API reference. For authoring environments (tags, params, variants, containers)
see {doc}`../user-guide/environments`; for the serving mechanics see {doc}`../user-guide/serving-environments`; for the client walkthrough see {doc}`../user-guide/remote-clients`.
```

## Env Server

`EnvServer` owns one Python environment object and exposes it as an RLMesh environment endpoint. It is self-describing: pass a vectorized environment (one exposing `num_envs` and `single_observation_space` / `single_action_space`) and it serves a vector endpoint automatically; otherwise it serves a single-environment endpoint. A model or evaluator connects with `RemoteEnv` for scalar endpoints and `RemoteVectorEnv` for vector endpoints.

### Environment Contract

The server inspects the environment once during construction and caches an {py:class}`~rlmesh.specs.EnvContract`. The contract describes the endpoint id, spaces, render mode, metadata, and number of environments.

```python
server = rlmesh.EnvServer(env, "127.0.0.1:5555")
print(server.env_contract.id)
print(server.env_contract.observation_space.kind)
print(server.env_contract.action_space.kind)
```

See {doc}`contracts` for contract fields.

### Bind Address Environment Variables

When RLMesh serves an environment through its bootstrap entrypoint (for example inside a sandbox container), the bind address follows a single canonical contract so that hosts and downstream images agree on where the environment listens:

| Variable         | Meaning                                                                                        |
| ---------------- | ---------------------------------------------------------------------------------------------- |
| `RLMESH_ADDRESS` | Full bind address (`host:port`, `port`, `tcp://host:port`, `unix:///...`). When set, it wins.  |
| `RLMESH_PORT`    | Port-only fallback, bound on `0.0.0.0`, used when `RLMESH_ADDRESS` is unset (default `50051`). |

`RLMESH_ADDRESS` accepts the same forms as the server `address` argument, so a non-default host or a Unix socket can be selected without code changes. Constructing `EnvServer` directly in your own process ignores these variables. Pass `address`/`host`/`port` explicitly.

The transport is plaintext gRPC and an endpoint is unauthenticated unless a bearer token is configured, so bind `0.0.0.0` only where the port is published to a trusted network. See [Exposure and authentication](../user-guide/serving-environments.md#exposure-and-authentication).

```{eval-rst}
.. autoclass:: rlmesh.EnvServer
   :members:
   :show-inheritance:
```

## Serving Helpers

```{note}
`rlmesh._serving` is **experimental**: it is not covered by the version contract in {doc}`../versioning`,
and its signatures may change in any release. Pin a minor range if you depend on it.
```

`rlmesh._serving` exposes a small surface for constructing an environment to serve through {py:class}`~rlmesh.EnvServer`. It exposes the loaders that `rlmesh._cli.serve_env` uses internally, so scripts and downstream runners can build an environment by Gymnasium id or by `module:callable` entrypoint.

Use it when a script or runner has to build the environment itself before serving. If you already hold an env object, pass it straight to {py:class}`~rlmesh.EnvServer`.

```python
import rlmesh
from rlmesh import _serving

env = _serving.load_env("CartPole-v1")
rlmesh.EnvServer(env).serve()
```

```{eval-rst}
.. autofunction:: rlmesh._serving.load_env
```

```{eval-rst}
.. autofunction:: rlmesh._serving.load_env_entrypoint
```

```{eval-rst}
.. autofunction:: rlmesh._serving.import_packages
```

## Remote Clients

Remote clients connect to an environment endpoint and expose the same core calls as a local Gymnasium environment: `reset`, `step`, `render`, and `close`. Reach for one when the environment runs in another process; the contract from the handshake supplies the spaces, so the client needs no local copy of the env.

Use the concrete backend modules in application code:

- `rlmesh.RemoteEnv` and `rlmesh.RemoteVectorEnv` preserve RLMesh-native values.
- `rlmesh.numpy.RemoteEnv` and `rlmesh.numpy.RemoteVectorEnv` decode tensor leaves as NumPy arrays.
- `rlmesh.torch.RemoteEnv` and `rlmesh.torch.RemoteVectorEnv` decode tensor leaves as Torch tensors.

Every remote client keeps the endpoint handshake in `env_contract`. `observation_space` and `action_space` are client-side wrappers built from the contract's `SpaceSpec` values.

Use `render()` to pull a single decoded frame from the endpoint. The environment must produce frames, typically `render_mode="rgb_array"`; `render()` returns `None` when it has no frame.

### Single Environment Base

```{eval-rst}
.. autoclass:: rlmesh._client.RemoteEnvBase
   :members:
   :show-inheritance:
```

### Vector Environment Base

```{eval-rst}
.. autoclass:: rlmesh._client.RemoteVectorEnvBase
   :members:
   :show-inheritance:
```

### Native Clients

`rlmesh.RemoteEnv` and `rlmesh.RemoteVectorEnv` are the dependency-free clients: the full call surface is the base above, values left RLMesh-native. See {doc}`backends` for the NumPy, Torch, and JAX classes.

```{eval-rst}
.. autoclass:: rlmesh.RemoteEnv
```

```{eval-rst}
.. autoclass:: rlmesh.RemoteVectorEnv
```

## Where next

- {doc}`../user-guide/serving-environments`: the serving flow end to end.
- {doc}`../user-guide/remote-clients`: connecting a model or evaluator.
- {doc}`contracts`: contract fields the handshake exposes.

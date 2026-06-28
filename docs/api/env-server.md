# Env Server

```{note}
This is the autodoc API reference. For authoring environments (tags, params, variants, containers)
see {doc}`../user-guide/environments`; for the serving mechanics see {doc}`../user-guide/serving-environments`.
```

`EnvServer` owns one Python environment object and exposes it as an RLMesh environment endpoint. It is self-describing: pass a vectorized environment (one exposing `num_envs` and `single_observation_space` / `single_action_space`) and it serves a vector endpoint automatically; otherwise it serves a single-environment endpoint. A model or evaluator connects with `RemoteEnv` for scalar endpoints and `RemoteVectorEnv` for vector endpoints.

## Environment Contract

The server inspects the environment once during construction and caches an {py:class}`~rlmesh.specs.EnvContract`. The contract describes the endpoint id, spaces, render mode, metadata, and number of environments. `EnvServer.spec` is an alias for `env_contract` so code that expects a Gymnasium-style `spec` field can still reach the same RLMesh contract.

```python
server = rlmesh.EnvServer(env, "127.0.0.1:5555")
print(server.env_contract.id)
print(server.env_contract.observation_space.kind)
print(server.spec.action_space.kind)
```

See {doc}`contracts` for contract fields.

## Bind Address Environment Variables

When RLMesh serves an environment through its bootstrap entrypoint (for example inside a sandbox container), the bind address follows a single canonical contract so that hosts and downstream images agree on where the environment listens:

| Variable             | Meaning                                                                                        |
| -------------------- | ---------------------------------------------------------------------------------------------- |
| `RLMESH_ADDRESS`     | Full bind address (`host:port`, `port`, `tcp://host:port`, `unix:///...`). When set, it wins.  |
| `RLMESH_PORT`        | Port-only fallback, bound on `0.0.0.0`, used when `RLMESH_ADDRESS` is unset (default `50051`). |
| `RLMESH_ENV_ADDRESS` | Deprecated alias for `RLMESH_ADDRESS`; read only when it is unset.                             |
| `RLMESH_ENV_PORT`    | Deprecated alias for `RLMESH_PORT`; read only when both addresses are unset.                   |

`RLMESH_ADDRESS` is the preferred knob; it accepts the same forms as the server `address` argument, so a non-default host or a Unix socket can be selected without code changes. `RLMESH_ENV_ADDRESS` / `RLMESH_ENV_PORT` are deprecated aliases kept for backward compatibility. Constructing `EnvServer` directly in your own process ignores these variables. Pass `address`/`host`/`port` explicitly.

## API

```{eval-rst}
.. autoclass:: rlmesh.EnvServer
   :members:
   :show-inheritance:
```

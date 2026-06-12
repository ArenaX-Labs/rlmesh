# Env Server

`EnvServer` owns a Python environment object and exposes it as an RLMesh environment endpoint. The
served endpoint supports both local loopback use and process boundaries where a model or evaluator
connects through `RemoteEnv` or `RemoteVectorEnv`.

## Environment Contract

The server inspects the environment once during construction and caches an
{py:class}`~rlmesh.specs.EnvContract`. The contract describes the endpoint id, spaces, render mode,
metadata, and number of environments. `EnvServer.spec` is an alias for `EnvServer.env_contract` so
code that expects a Gymnasium-style `spec` field can still reach the same RLMesh contract.

```python
server = rlmesh.EnvServer(env, "127.0.0.1:5555")
print(server.env_contract.id)
print(server.env_contract.observation_space.kind)
print(server.spec.action_space.kind)
```

See {doc}`contracts` for contract fields.

## Bind Address Environment Variables

When RLMesh serves an environment through its bootstrap entrypoint (for example inside a sandbox
container), the bind address follows a single canonical contract so that hosts and downstream images
agree on where the environment listens:

| Variable             | Meaning                                                                                                 |
| -------------------- | ------------------------------------------------------------------------------------------------------- |
| `RLMESH_ENV_ADDRESS` | Full bind address (`host:port`, `port`, `tcp://host:port`, `unix:///...`). When set, it wins.           |
| `RLMESH_ENV_PORT`    | Port-only fallback, bound on `0.0.0.0`, used only when `RLMESH_ENV_ADDRESS` is unset (default `50051`). |

`RLMESH_ENV_ADDRESS` is the preferred knob; it accepts the same forms as the `EnvServer` `address`
argument, so a non-default host or a Unix socket can be selected without code changes.
`RLMESH_ENV_PORT` is kept for backward compatibility. Constructing `EnvServer` directly in your own
process ignores these variables — pass `address`/`host`/`port` explicitly.

## API

```{eval-rst}
.. autoclass:: rlmesh.server.EnvServer
   :members:
   :show-inheritance:
```

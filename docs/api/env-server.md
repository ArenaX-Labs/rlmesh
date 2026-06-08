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

## API

```{eval-rst}
.. autoclass:: rlmesh.server.EnvServer
   :members:
   :show-inheritance:
```

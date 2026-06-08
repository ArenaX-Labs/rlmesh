# Remote Environments

Remote clients connect to an environment endpoint and expose the same high-level workflow as a local
Gymnasium environment: `reset`, `step`, `render`, and `close`.

Use the concrete backend modules for normal code:

- `rlmesh.RemoteEnv` and `rlmesh.RemoteVectorEnv` preserve RLMesh-native values.
- `rlmesh.numpy.RemoteEnv` and `rlmesh.numpy.RemoteVectorEnv` decode tensor leaves as NumPy arrays.
- `rlmesh.torch.RemoteEnv` and `rlmesh.torch.RemoteVectorEnv` decode tensor leaves as Torch tensors.

Every remote client keeps the endpoint handshake in `env_contract`. The `spec` property is an alias
for that same contract. `observation_space` and `action_space` are client-side wrappers built from
the contract's `SpaceSpec` values.

## Single Environment Base

```{eval-rst}
.. autoclass:: rlmesh.client.remote_env.RemoteEnvBase
   :members:
   :show-inheritance:
```

## Vector Environment Base

```{eval-rst}
.. autoclass:: rlmesh.client.remote_vector_env.RemoteVectorEnvBase
   :members:
   :show-inheritance:
```

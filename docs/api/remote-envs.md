# Remote Environments

Remote clients connect to an environment endpoint and expose the same core calls as a local Gymnasium environment: `reset`, `step`, `render`, and `close`.

Use the concrete backend modules in application code:

- `rlmesh.RemoteEnv` and `rlmesh.RemoteVectorEnv` preserve RLMesh-native values.
- `rlmesh.numpy.RemoteEnv` and `rlmesh.numpy.RemoteVectorEnv` decode tensor leaves as NumPy arrays.
- `rlmesh.torch.RemoteEnv` and `rlmesh.torch.RemoteVectorEnv` decode tensor leaves as Torch tensors.

Every remote client keeps the endpoint handshake in `env_contract`. The `spec` property is an alias for that same contract. `observation_space` and `action_space` are client-side wrappers built from the contract's `SpaceSpec` values.

Use `render()` to pull a single decoded frame from the endpoint. The environment must produce frames, typically `render_mode="rgb_array"`; `render()` returns `None` when it has no frame.

## Single Environment Base

```{eval-rst}
.. autoclass:: rlmesh._client.RemoteEnvBase
   :members:
   :show-inheritance:
```

## Vector Environment Base

```{eval-rst}
.. autoclass:: rlmesh._client.RemoteVectorEnvBase
   :members:
   :show-inheritance:
```
